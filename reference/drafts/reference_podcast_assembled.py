#!/usr/bin/env python3
"""Assemble the podcast prompt embedding matrix exactly as the reference does.

  embeds = word_embeddings[ids]
  embeds[:, prompt_start:, :]   = linear_proj_audio(grouped reference latents)
  embeds[:, speaker_position, :] = spk_head(speaker_embedding)

Speaker embedding is zeros here (the reference documents this as the missing-reference case),
so the speaker row is exactly spk_head's bias.

    python reference_podcast_assembled.py --model ... --references ... --prompt ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_reference as fr


class Aggregator:
    """linear_proj_audio: CLS + latent through 8 DiT blocks, CLS out to 1536."""

    def __init__(self, state: dict[str, torch.Tensor]):
        self.state = state
        self.depth = 8
        self.hidden = 1024
        self.heads = 16
        self.head_dim = 64

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.linear(latent, self.state["x_embedder.weight"], self.state["x_embedder.bias"])
        cls = self.state["word_embedder.weight"].reshape(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        tokens = x.shape[1]
        cos, sin = fr.rotary_tables(tokens, self.state["rotary_embed.inv_freq"])
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        for layer in range(self.depth):
            block = f"blocks.{layer}."
            h = fr.rms_norm(x, self.state[block + "norm1.weight"])
            q = F.linear(h, self.state[block + "attn.to_q.weight"], self.state[block + "attn.to_q.bias"])
            k = F.linear(h, self.state[block + "attn.to_k.weight"], self.state[block + "attn.to_k.bias"])
            v = F.linear(h, self.state[block + "attn.to_v.weight"], self.state[block + "attn.to_v.bias"])
            q = q.view(x.shape[0], tokens, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(x.shape[0], tokens, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(x.shape[0], tokens, self.heads, self.head_dim).transpose(1, 2)
            q = fr.apply_rope(q, cos, sin)
            k = fr.apply_rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            a = a.transpose(1, 2).reshape(x.shape[0], tokens, self.hidden)
            x = x + F.linear(a, self.state[block + "attn.to_out.0.weight"], self.state[block + "attn.to_out.0.bias"])
            h = fr.rms_norm(x, self.state[block + "norm2.weight"])
            h = F.linear(h, self.state[block + "mlp.ff.0.0.weight"], self.state[block + "mlp.ff.0.0.bias"])
            h = F.gelu(h, approximate="tanh")
            h = F.linear(h, self.state[block + "mlp.ff.2.weight"], self.state[block + "mlp.ff.2.bias"])
            x = x + h
        out = fr.rms_norm(x, self.state["final_layer.norm_final.weight"])
        out = F.linear(out, self.state["final_layer.linear.weight"], self.state["final_layer.linear.bias"])
        return out[:, :1, :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted = {n: s for n, s in index["weight_map"].items() if n.startswith("linear_proj_audio.")}
    handles, state = {}, {}
    for name, shard in wanted.items():
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        state[name[len("linear_proj_audio."):]] = handles[shard].get_tensor(name).float()

    table_name = "model.model.word_embeddings.weight"
    with safe_open(root / index["weight_map"][table_name], framework="pt", device="cpu") as handle:
        table = handle.get_tensor(table_name).float()
    with safe_open(root / index["weight_map"]["spk_head.bias"], framework="pt", device="cpu") as handle:
        spk_bias = handle.get_tensor("spk_head.bias").float()

    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    ids = torch.tensor([prompt["token_ids"]])
    start = prompt["prompt_latent_start_index"]
    speaker_position = prompt["speaker_embedding_position"]
    references = torch.from_numpy(np.load(args.references)["reference_latents"]).float()

    with torch.no_grad():
        embeds = table[ids].clone()
        # tts_config.patch_size is 1, so each latent is its own group and gets its own CLS
        # (the 4-frame grouping happens inside the VAE encoder, not here)
        groups = references.reshape(-1, 1, references.shape[-1])
        projected = Aggregator(state)(groups).reshape(1, -1, 1536)
        embeds[:, start:, :] = projected
        embeds[:, speaker_position, :] = spk_bias.unsqueeze(0)   # spk_head(zeros) == bias

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "podcast-embeddings.npz", embeds=embeds.numpy(),
                        ids=np.array(prompt["token_ids"], dtype=np.int32),
                        speaker_row=spk_bias.numpy())
    print(f"ids             : {len(prompt['token_ids'])}, placeholders {start}..{len(prompt['token_ids'])-1}")
    print(f"reference rows  : {tuple(projected.shape)} rms {float(projected.pow(2).mean().sqrt()):.5f}")
    print(f"speaker row {speaker_position:<3}: spk_head(zeros) = bias, rms {float(spk_bias.pow(2).mean().sqrt()):.5f}")
    print(f"embeds          : {tuple(embeds.shape)}")
    print(f"row 20 == row 20 of table? {torch.equal(embeds[0, speaker_position], spk_bias)}")
    print(f"row {start} (first placeholder) rms {float(embeds[0, start].pow(2).mean().sqrt()):.5f}")
    print(f"row {len(prompt['token_ids'])-1} (last placeholder) rms {float(embeds[0, -1].pow(2).mean().sqrt()):.5f}")
    print(f"written         : {args.out / 'podcast-embeddings.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
