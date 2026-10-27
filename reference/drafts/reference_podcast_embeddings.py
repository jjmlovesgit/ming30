#!/usr/bin/env python3
"""Reference construction of the podcast prompt embeddings.

From modeling_lingv3_tts.py:

  prompt_waveform  -> audio.encode_latent -> prompt_latent [1, steps, 64]
  step 1..: prompt_latent_steps // patch_size  = prompt_latent_tokens
  grouped          = prompt_latent.reshape(-1, patch_size, 64)
  prompt_hidden    = linear_proj_audio(grouped) -> [1, prompt_latent_tokens, 1536]
  inputs_embeds[:, prompt_start:, :] = prompt_hidden

  speaker_hidden = spk_head(speaker_embedding[192]) -> [1, 1536]
  inputs_embeds[:, speaker_position, :] = speaker_hidden

With two references the caller still passes ONE speaker embedding (the model has a single
spk_head and a single <spk> row). Source does not say how to derive it from several clips;
here it is zeros, which the reference itself documents as the "missing reference" case, so the
embedding gate can be tested before CAMPPlus is ported.

    python reference_podcast_embeddings.py --model ... --clips a.npy b.npy --out ...
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
from transformers import Qwen2Config, Qwen2Model

HOP = 882
PATCH = 4
LATENT = 64


def load_state(root: Path, prefix: str) -> dict[str, torch.Tensor]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted = {n: s for n, s in index["weight_map"].items() if n.startswith(prefix)}
    handles, out = {}, {}
    for name, shard in wanted.items():
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        out[name] = handles[shard].get_tensor(name).float()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--clips", required=True, nargs="+", type=Path, help="mono 44.1k .npy per speaker")
    parser.add_argument("--prompt", required=True, type=Path, help="podcast-prompt.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    audio_cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["audio_config"]
    enc_backbone = dict(audio_cfg["enc_kwargs"]["backbone"])
    enc_backbone["_attn_implementation"] = "sdpa"
    agg_backbone = dict(enc_backbone)
    agg_backbone["num_hidden_layers"] = 4

    enc_state = load_state(root, "audio.encoder.")
    encoder = Qwen2Model(Qwen2Config(**enc_backbone))
    encoder.load_state_dict({k[len("audio.encoder.encoder."):]: v for k, v in enc_state.items()
                             if k.startswith("audio.encoder.encoder.")})
    aggregator = Qwen2Model(Qwen2Config(**agg_backbone))
    aggregator.load_state_dict({k[len("audio.encoder.aggregator."):]: v for k, v in enc_state.items()
                                if k.startswith("audio.encoder.aggregator.")})
    encoder.eval()
    aggregator.eval()

    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    ids = prompt["token_ids"]
    speaker_position = prompt["speaker_embedding_position"]
    prompt_tokens = prompt["prompt_latent_tokens"]
    prompt_start = prompt["prompt_latent_start_index"]

    waveform = torch.cat([torch.from_numpy(np.load(clip)).float().reshape(-1) for clip in args.clips]).reshape(1, -1)
    with torch.no_grad():
        frames_total = (waveform.shape[1] + HOP - 1) // HOP
        padded = F.pad(waveform, (0, (frames_total - 1) * HOP + HOP - waveform.shape[1]), value=0.0)
        frames = padded.unfold(-1, HOP, HOP)
        x = F.linear(frames, enc_state["audio.encoder.fc1.weight"], None)
        x = F.linear(x, enc_state["audio.encoder.fc2.weight"], enc_state["audio.encoder.fc2.bias"])
        x = encoder(inputs_embeds=x).last_hidden_state
        remainder = x.shape[1] % PATCH
        if remainder:
            x = F.pad(x, (0, 0, 0, PATCH - remainder), value=0.0)
        patches = x.shape[1] // PATCH
        cls = enc_state["audio.encoder.cls_embed"].expand(patches, -1, -1)
        grouped = x.reshape(patches, PATCH, -1)
        x = aggregator(inputs_embeds=torch.cat([grouped, cls], dim=1).reshape(1, -1, x.shape[-1])).last_hidden_state
        x = x.reshape(patches, PATCH + 1, -1)[:, -1:, :].reshape(1, patches, -1)
        params = F.linear(x, enc_state["audio.encoder.fc3.weight"], enc_state["audio.encoder.fc3.bias"])
        latent = params.chunk(2, dim=-1)[0]                       # posterior mean, deterministic

    if latent.shape[1] < prompt_tokens:
        raise SystemExit(f"references give {latent.shape[1]} patches, prompt expects {prompt_tokens}")
    latent = latent[:, :prompt_tokens, :]

    args.out.mkdir(parents=True, exist_ok=True)
    latent[:, :, :].astype if False else None
    np.savez_compressed(args.out / "podcast-references.npz", reference_latents=latent.numpy(),
                        reference_seconds=np.array([waveform.shape[1] / 44100.0], dtype=np.float32))
    report = {
        "id_count": len(ids),
        "prompt_start": prompt_start,
        "prompt_latent_tokens": prompt_tokens,
        "speaker_position": speaker_position,
        "reference_patches_available": int(latent.shape[1]),
        "reference_latents_rms": float(latent.pow(2).mean().sqrt()),
        "clip_seconds": float(waveform.shape[1] / 44100.0),
        "order": "clips concatenated in argument order, encoded once, then truncated to the placeholder count",
        "speaker_embedding": "zeros (reference documents this as the missing-reference case)",
    }
    (args.out / "podcast-references.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"clips           : {len(args.clips)} -> {waveform.shape[1]} samples ({waveform.shape[1]/44100:.2f} s)")
    print(f"latents         : {tuple(latent.shape)} (need {prompt_tokens}) rms {report['reference_latents_rms']:.4f}")
    print(f"prompt rows     : placeholders {prompt_start}..{len(ids)-1}, speaker row {speaker_position}")
    print(f"written         : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
