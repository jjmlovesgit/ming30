#!/usr/bin/env python3
"""VAE decoder reference for Ming-3.0-tts-tiny, one latent patch.

Path (from vae_modules.Decoder.low_level_reconstruct and istft.py):

  latent [B,1,64] -> fc1 (64->896) -> linear upsample x4 (align_corners=False)
  -> Qwen2 (24 layers, 14 heads, 2 kv heads, rope 1e6, causal) -> last_hidden_state
  -> head.out (896 -> n_fft+2 = 3530) -> split (mag, phase) -> S = exp(mag)*(cos+isin)
  -> iSTFT(n_fft 3528, hop 882, window from the checkpoint, padding "same")
  -> 3528 samples = 80 ms at 44.1 kHz

    python vae_reference.py --model models/ming3-tts-tiny ^
        --latents out/reference/flow/flow-step.npz --out out/reference/vae
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

N_FFT = 3528
HOP = 882
PATCH = 4


def load_state(root: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted: dict[str, str] = {}
    for name, shard in index["weight_map"].items():
        for prefix in prefixes:
            if name.startswith(prefix):
                wanted[name] = shard
                break
    handles, tensors = {}, {}
    for name, shard in wanted.items():
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        tensors[name] = handles[shard].get_tensor(name).float()
    return tensors


def istft(spec: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Same-padding iSTFT exactly as istft.py implements it (non-streaming)."""

    pad = (N_FFT - HOP) // 2
    ifft = torch.fft.irfft(spec, N_FFT, dim=1, norm="backward") * window[None, :, None]
    frames = ifft.shape[-1]
    folded = F.fold(ifft, output_size=(1, (frames - 1) * HOP + N_FFT),
                    kernel_size=(1, N_FFT), stride=(1, HOP))[:, 0, 0, :]
    y = folded[:, pad:-pad]

    window_sq = window.square().expand(1, frames, -1).transpose(1, 2)
    envelope = F.fold(window_sq, output_size=(1, (frames - 1) * HOP + N_FFT),
                      kernel_size=(1, N_FFT), stride=(1, HOP)).squeeze()
    envelope = envelope[pad:-pad]
    return y / envelope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))["audio_config"]
    backbone = dict(config["dec_kwargs"]["backbone"])
    backbone.setdefault("attention_bias", True)
    backbone["_attn_implementation"] = "sdpa"

    state = load_state(root, ("audio.decoder.",))
    qwen = Qwen2Model(Qwen2Config(**backbone))
    qwen.load_state_dict({k[len("audio.decoder.decoder."):]: v for k, v in state.items()
                          if k.startswith("audio.decoder.decoder.")})
    qwen.eval()

    fc1_w = state["audio.decoder.fc1.weight"]
    fc1_b = state["audio.decoder.fc1.bias"]
    head_w = state["audio.decoder.head.out.weight"]
    head_b = state["audio.decoder.head.out.bias"]
    window = state["audio.decoder.head.istft.window"]

    data = np.load(args.latents)
    latent = torch.from_numpy(data["latents"]).float()             # [1,1,64]

    with torch.no_grad():
        x = F.linear(latent, fc1_w, fc1_b)                          # [1,1,896]
        up = F.interpolate(x.transpose(1, 2), scale_factor=PATCH, mode="linear",
                           align_corners=False).transpose(1, 2)     # [1,4,896]
        hidden = qwen(inputs_embeds=up).last_hidden_state            # [1,4,896]
        head = F.linear(hidden, head_w, head_b)                      # [1,4,3530]
        mag, phase = head.transpose(1, 2).chunk(2, dim=1)            # [1,1765,4] each
        magnitude = torch.clamp(torch.exp(mag), max=1e2)
        spec = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        audio = istft(spec, window)                                  # [1,3528]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "vae-step.npz",
                        latent=latent.numpy(), fc1=x.numpy(), upsampled=up.numpy(),
                        decoder_hidden=hidden.numpy(), head_out=head.numpy(),
                        magnitude=magnitude.numpy(), waveform=audio.numpy())
    wave = audio.numpy().reshape(-1)
    print(f"latent          : {tuple(latent.shape)}")
    print(f"upsampled       : {tuple(up.shape)} (patch {PATCH})")
    print(f"decoder hidden  : {tuple(hidden.shape)}")
    print(f"head out        : {tuple(head.shape)}")
    print(f"waveform        : {wave.shape} = {wave.size / 44100 * 1000:.1f} ms at 44.1 kHz")
    print(f"waveform rms    : {float(np.sqrt((wave ** 2).mean())):.6f} max|.|: {float(np.abs(wave).max()):.4f}")
    print(f"written         : {args.out / 'vae-step.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
