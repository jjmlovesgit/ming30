#!/usr/bin/env python3
"""Isolate the small opening artifact: generated latents or decoder start boundary?

Decodes four ways and reports the first 200 ms:
  a) the produced latents (whole sequence)
  b) the same latents with the first patch replaced by zeros
  c) 60 patches of exact zeros (pure decoder start boundary)
  d) the produced latents decoded per patch (streaming), first chunk

    python podcast_open_artifact.py --model ... --latents out/.../cpp-render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mingvae.vae_modules import Decoder
from safetensors import safe_open

FRAME_MS = 50


def envelope(wave: np.ndarray, frame_ms: int = FRAME_MS) -> np.ndarray:
    frame = int(44100 * frame_ms / 1000)
    return np.array([np.sqrt((wave[i:i + frame] ** 2).mean()) for i in range(0, max(1, wave.size - frame), frame)])


def head(values: np.ndarray, count: int = 10) -> list[float]:
    return [round(float(v), 5) for v in values[:count]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["audio_config"]
    bb = dict(cfg["dec_kwargs"]["backbone"])
    bb["_attn_implementation"] = "sdpa"
    bb["use_sliding_window"] = False
    model = Decoder(bb, output_dim=cfg["dec_kwargs"]["output_dim"], latent_dim=cfg["dec_kwargs"]["latent_dim"],
                    patch_size=cfg["patch_size"])
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    handles, state = {}, {}
    for name, shard in index["weight_map"].items():
        if not name.startswith("audio.decoder."):
            continue
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        state[name[len("audio.decoder."):]] = handles[shard].get_tensor(name).float()
    model.load_state_dict(state, strict=False)
    model.eval()

    patches = sorted(args.latents.glob("latent_*.bin"))
    latents = np.stack([np.fromfile(path, dtype="<f4") for path in patches])[None]
    frames = latents.shape[1]

    def decode(tensor):
        with torch.no_grad():
            wave, _, _ = model.low_level_reconstruct(tensor, use_cache=False, stream_state=(None, None, None))
        return wave.reshape(-1).numpy()

    produced = decode(torch.from_numpy(latents).float())

    with_zero_first = latents.copy()
    with_zero_first[0, 0, :] = 0.0
    zero_first = decode(torch.from_numpy(with_zero_first).float())

    silence = np.zeros_like(latents)
    silence_wave = decode(torch.from_numpy(silence).float())

    # per-patch streaming, first chunk only
    state_carry, past = (None, None, None), None
    with torch.no_grad():
        chunk, state_carry, past = model.low_level_reconstruct(
            torch.from_numpy(latents[:, 0:1, :]).float(), past_key_values=past, use_cache=True,
            stream_state=state_carry, last_chunk=False)
    stream_first = chunk.reshape(-1).numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "open-artifact.npz", produced=produced, zero_first=zero_first,
                        silence=silence_wave, stream_first=stream_first)

    print(f"patches: {frames}, latent rms {np.sqrt((latents ** 2).mean()):.5f}")
    for label, wave in (("produced", produced), ("zero first patch", zero_first),
                        ("all-zero latents", silence_wave), ("streaming first chunk", stream_first)):
        env = envelope(wave)
        print(f"{label:<22} samples {wave.size:>6}  first 10 x {FRAME_MS}ms: {head(env)}")
    print()
    first_patch_samples = 3528
    print("first patch window (0-80 ms) energy, produced     :",
          round(float(np.sqrt((produced[:first_patch_samples] ** 2).mean())), 6))
    print("first patch window (0-80 ms) energy, zero-latents  :",
          round(float(np.sqrt((silence_wave[:first_patch_samples] ** 2).mean())), 6))
    print("first patch window (0-80 ms) energy, zero 1st patch:",
          round(float(np.sqrt((zero_first[:first_patch_samples] ** 2).mean())), 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
