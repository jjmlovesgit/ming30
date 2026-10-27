#!/usr/bin/env python3
"""First text-to-speech loop: thinker -> flow -> VAE, one latent patch at a time.

Per patch:
  1. thinker forward over [prompt embeddings, projected previous latents] (full recompute)
  2. stop head on the last hidden position (informational in old_shift decode)
  3. flow sample with the per-step noise from a single seeded stream
  4. project the latent through the Aggregator (linear_proj_audio) for the next step
  5. after the loop, decode all latents with the checkpoint VAE

Captures per patch: token count, last hidden checksum, stop logits/probability, noise,
latent, projected feedback vector, VAE chunk.

    python tts_loop.py --model models/ming3-tts-tiny --max-steps 16 --seed 1895 ^
        --out out/reference/tts-loop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinker_reference as tr
import flow_reference as fr

PATCH = 4


def thinker_forward_embeds(embeds: torch.Tensor, store: tr.WeightStore, cfg: dict) -> torch.Tensor:
    """Run all 24 layers over given embeddings; returns the final-normalised hidden."""

    eps = cfg["rms_norm_eps"]
    hidden = embeds
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.model.layers.{layer}."
        residual = hidden
        normed = tr.rms_norm(hidden, store.get(prefix + "input_layernorm.weight"), eps)
        if tr.attention_kind(cfg, layer) == "mla":
            attention = tr.mla_layer(normed, store, cfg, layer)
        else:
            attention = tr.kda_layer(normed, store, cfg, layer)
        hidden = residual + attention

        residual = hidden
        normed = tr.rms_norm(hidden, store.get(prefix + "post_attention_layernorm.weight"), eps)
        if layer < cfg["first_k_dense_replace"]:
            block = tr.mlp_forward(normed, store, prefix + "mlp")
        else:
            block, _, _, _, _ = tr.moe_forward(normed, store, cfg, layer)
        hidden = residual + block
    return tr.rms_norm(hidden, store.get("model.model.norm.weight"), eps)


class Aggregator:
    """linear_proj_audio: CLS + latent through 8 DiT blocks, CLS out to 1536."""

    def __init__(self, store: tr.WeightStore):
        self.store = store
        self.prefix = "linear_proj_audio."
        self.depth = 8
        self.hidden = 1024
        self.heads = 16
        self.head_dim = 64
        self.eps = 1e-6

    def _tensor(self, suffix, shape):
        name = "model.model." + self.prefix + suffix if False else self.prefix + suffix
        return self.store.get(name)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.linear(latent, self._tensor("x_embedder.weight", None), self._tensor("x_embedder.bias", None))
        cls = self._tensor("word_embedder.weight", None).reshape(1, 1, -1)
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
        tokens = x.shape[1]
        inv_freq = self._tensor("rotary_embed.inv_freq", None)
        cos, sin = fr.rotary_tables(tokens, inv_freq)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        for layer in range(self.depth):
            block = f"blocks.{layer}."
            h = fr.rms_norm(x, self._tensor(block + "norm1.weight", None))
            q = F.linear(h, self._tensor(block + "attn.to_q.weight", None), self._tensor(block + "attn.to_q.bias", None))
            k = F.linear(h, self._tensor(block + "attn.to_k.weight", None), self._tensor(block + "attn.to_k.bias", None))
            v = F.linear(h, self._tensor(block + "attn.to_v.weight", None), self._tensor(block + "attn.to_v.bias", None))
            q = q.view(1, tokens, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(1, tokens, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(1, tokens, self.heads, self.head_dim).transpose(1, 2)
            q = fr.apply_rope(q, cos, sin)
            k = fr.apply_rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            a = a.transpose(1, 2).reshape(1, tokens, self.hidden)
            x = x + F.linear(a, self._tensor(block + "attn.to_out.0.weight", None),
                             self._tensor(block + "attn.to_out.0.bias", None))
            h = fr.rms_norm(x, self._tensor(block + "norm2.weight", None))
            h = F.linear(h, self._tensor(block + "mlp.ff.0.0.weight", None), self._tensor(block + "mlp.ff.0.0.bias", None))
            h = F.gelu(h, approximate="tanh")
            h = F.linear(h, self._tensor(block + "mlp.ff.2.weight", None), self._tensor(block + "mlp.ff.2.bias", None))
            x = x + h
        out = fr.rms_norm(x, self._tensor("final_layer.norm_final.weight", None))
        out = F.linear(out, self._tensor("final_layer.linear.weight", None),
                       self._tensor("final_layer.linear.bias", None))
        return out[:, :1, :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", type=Path, default=Path("out/reference/stage-a-segments.json"))
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1895)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    store = tr.WeightStore(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["llm_config"]
    flow_weights = fr.FlowWeights(root)
    aggregator = Aggregator(store)

    prompt_ids = json.loads(args.prompt.read_text(encoding="utf-8"))["flat_ids"]
    table = store.get("model.model.word_embeddings.weight")
    prompt_embeds = table[torch.tensor([prompt_ids])].float()
    print(f"prompt: {len(prompt_ids)} ids, embeds {tuple(prompt_embeds.shape)}")

    torch.manual_seed(args.seed)
    history_embeds = prompt_embeds
    latent_history = torch.zeros(1, 1, 64)
    latents, records = [], []

    for step in range(args.max_steps):
        with torch.no_grad():
            final = thinker_forward_embeds(history_embeds, store, cfg)
            last = final[:, -1:, :]
            stop_logits = F.linear(last, store.get("stop_head.weight"), store.get("stop_head.bias"))
            stop_prob = float(stop_logits.softmax(-1)[0, 0, 1])

            noise = torch.randn(1, 64, 1)
            result = fr.flow_sample(last, noise, latent_history, flow_weights,
                                    steps=args.flow_steps, cfg=args.cfg)
            latent = torch.from_numpy(result["latents"])
            feedback = aggregator(latent)

            latents.append(latent.reshape(1, 1, 64))
            latent_history = latent.reshape(1, 1, 64).clone()
            history_embeds = torch.cat([history_embeds, feedback], dim=1)
            records.append({
                "step": step, "tokens": int(history_embeds.shape[1]),
                "stop_logits": stop_logits.reshape(-1).tolist(), "stop_prob": stop_prob,
                "latent": latent.reshape(-1).tolist(),
                "noise": noise.reshape(-1).tolist(),
                "feedback": feedback.reshape(-1).tolist(),
                "final_last": last.reshape(-1).tolist(),
                "feedback_norm": float(feedback.norm()),
                "hidden_norm": float(last.norm()),
                "noise_norm": float(noise.norm()),
            })
            print(f"  patch {step:>2}: tokens={history_embeds.shape[1]:>3} "
                  f"stop_p={stop_prob:.5f} latent_rms={float(latent.pow(2).mean().sqrt()):.4f} "
                  f"feedback_norm={float(feedback.norm()):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stacked = torch.cat(latents, dim=1)                       # [1, N, 64]
    noises = np.stack([np.asarray(r["noise"], dtype=np.float32) for r in records])
    feedback = np.stack([np.asarray(r["feedback"], dtype=np.float32) for r in records])
    finals = np.stack([np.asarray(r["final_last"], dtype=np.float32) for r in records])
    np.savez_compressed(args.out / "loop.npz",
                        latents=stacked.numpy(), noises=noises, feedback=feedback, finals=finals)
    (args.out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"latents: {tuple(stacked.shape)} ({stacked.shape[1] * 0.08:.2f} s of audio)")
    print(f"written: {args.out / 'loop.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
