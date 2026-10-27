#!/usr/bin/env python3
"""VAE encoder round-trip on real speech: encode -> decode -> compare levels.

Diagnostic for the near-silent decode seen on a flow-generated latent. If real speech
encodes to latents that decode back at a sane level, the decoder is fine and the earlier
latent was simply out of distribution. Uses several patches (one patch is only 80 ms).

    python vae_roundtrip.py --model models/ming3-tts-tiny --wav clip.wav --out out/reference/roundtrip
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


def qwen_from(state: dict[str, torch.Tensor], prefix: str, backbone: dict) -> Qwen2Model:
    config = Qwen2Config(**backbone)
    model = Qwen2Model(config)
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path, help="mono 44.1 kHz float32 .npy")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    audio_cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["audio_config"]
    enc_backbone = dict(audio_cfg["enc_kwargs"]["backbone"])
    enc_backbone["_attn_implementation"] = "sdpa"
    agg_backbone = dict(enc_backbone)
    agg_backbone["num_hidden_layers"] = 4

    state = load_state(root, "audio.encoder.")
    encoder = qwen_from(state, "audio.encoder.encoder.", enc_backbone)
    aggregator = qwen_from(state, "audio.encoder.aggregator.", agg_backbone)

    dec_state = load_state(root, "audio.decoder.")
    dec_backbone = dict(audio_cfg["dec_kwargs"]["backbone"])
    dec_backbone.setdefault("attention_bias", True)
    dec_backbone["_attn_implementation"] = "sdpa"
    # Parity experiment: force full causal attention so it matches the C++ side.
    # The released model runs flash-attention with sliding_window=64 instead.
    dec_backbone["use_sliding_window"] = False
    decoder = qwen_from(dec_state, "audio.decoder.decoder.", dec_backbone)

    waveform = torch.from_numpy(np.load(args.wav)).float().reshape(1, -1)
    length = waveform.shape[1]

    with torch.no_grad():
        # ---- encoder ----
        frames_total = (length + HOP - 1) // HOP
        expected = (frames_total - 1) * HOP + HOP
        padded = F.pad(waveform, (0, expected - length), value=0.0)
        frames = padded.unfold(-1, HOP, HOP)                                  # [1, F, 882]
        x = F.linear(frames, state["audio.encoder.fc1.weight"], None)
        x = F.linear(x, state["audio.encoder.fc2.weight"], state["audio.encoder.fc2.bias"])
        x = encoder(inputs_embeds=x).last_hidden_state                         # [1, F, 896]

        # insert a CLS token after each patch, run the aggregator, keep the CLS rows
        remainder = x.shape[1] % PATCH
        if remainder:
            x = F.pad(x, (0, 0, 0, PATCH - remainder), value=0.0)
        patches = x.shape[1] // PATCH
        cls = state["audio.encoder.cls_embed"].expand(patches, -1, -1)
        grouped = x.reshape(patches, PATCH, -1)
        x = aggregator(inputs_embeds=torch.cat([grouped, cls], dim=1).reshape(1, -1, x.shape[-1])).last_hidden_state
        x = x.reshape(patches, PATCH + 1, -1)[:, -1:, :].reshape(1, patches, -1)

        params = F.linear(x, state["audio.encoder.fc3.weight"], state["audio.encoder.fc3.bias"])
        mean, logvar = params.chunk(2, dim=-1)
        latent = mean                                                          # posterior mean (deterministic)

        # ---- decoder ----
        d = F.linear(latent, dec_state["audio.decoder.fc1.weight"], dec_state["audio.decoder.fc1.bias"])
        up = F.interpolate(d.transpose(1, 2), scale_factor=PATCH, mode="linear",
                           align_corners=False).transpose(1, 2)
        hidden = decoder(inputs_embeds=up).last_hidden_state
        head = F.linear(hidden, dec_state["audio.decoder.head.out.weight"], dec_state["audio.decoder.head.out.bias"])
        mag, phase = head.transpose(1, 2).chunk(2, dim=1)
        magnitude = torch.clamp(torch.exp(mag), max=1e2)
        spec = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))

        # iSTFT (same-padding), as in istft.py
        window = dec_state["audio.decoder.head.istft.window"]
        pad = (mag.shape[1] * 0 + 3528 - HOP) // 2
        ifft = torch.fft.irfft(spec, 3528, dim=1, norm="backward") * window[None, :, None]
        frames_out = ifft.shape[-1]
        folded = F.fold(ifft, output_size=(1, (frames_out - 1) * HOP + 3528),
                        kernel_size=(1, 3528), stride=(1, HOP))[:, 0, 0, :][:, pad:-pad]
        env = F.fold(window.square().expand(1, frames_out, -1).transpose(1, 2),
                     output_size=(1, (frames_out - 1) * HOP + 3528),
                     kernel_size=(1, 3528), stride=(1, HOP)).squeeze()[pad:-pad]
        restored = (folded / env).reshape(-1).numpy()

    original = waveform.reshape(-1).numpy()
    n = min(restored.size, original.size)
    corr = float(np.corrcoef(restored[:n], original[:n])[0, 1]) if n > 1 else float("nan")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "roundtrip.npz", latent=latent.numpy(), waveform=restored,
                        original=original, decoder_hidden=hidden.numpy(),
                        magnitude=magnitude.numpy(), fc1=d.numpy(), upsampled=up.numpy())
    print(f"input            : {original.size} samples ({original.size / 44100:.2f} s), rms {np.sqrt((original ** 2).mean()):.6f}")
    print(f"encoder frames   : {frames_total} -> {patches} patches")
    print(f"latent           : {tuple(latent.shape)} rms {float(latent.pow(2).mean().sqrt()):.5f} "
          f"min {float(latent.min()):.3f} max {float(latent.max()):.3f}")
    print(f"decoder tokens   : {up.shape[1]}")
    print(f"restored         : {restored.size} samples ({restored.size / 44100:.2f} s), rms {np.sqrt((restored ** 2).mean()):.6f}")
    print(f"restored peak    : {np.abs(restored).max():.5f}")
    print(f"correlation with original (first {n}): {corr:.4f}")
    print(f"written          : {args.out / 'roundtrip.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
