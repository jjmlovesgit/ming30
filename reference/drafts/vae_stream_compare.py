#!/usr/bin/env python3
"""VAE per-patch (streaming) vs whole-sequence decode, using the checkpoint classes.

The generation loop decodes one latent patch at a time, so the decoder's overlap-add and
upsampling state must stitch exactly the way a single whole-sequence decode does. This
compares the two paths sample by sample.

    python vae_stream_compare.py --model models/ming3-tts-tiny ^
        --latents out/reference/roundtrip/roundtrip.npz --out out/reference/vae-stream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_decoder(root: Path, device: str = "cpu"):
    from mingvae.vae_modules import Decoder

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))["audio_config"]
    dec_args = dict(config["dec_kwargs"])
    dec_args["backbone"] = dict(dec_args["backbone"])
    dec_args["backbone"]["_attn_implementation"] = "sdpa"
    # parity choice: the reference config enables a 64-token sliding window, which the
    # sdpa backend cannot express, so both sides run full causal attention here.
    dec_args["backbone"]["use_sliding_window"] = False

    # _build_qwen2_config() takes the backbone dict itself, not the wrapper
    model = Decoder(dec_args["backbone"], output_dim=dec_args["output_dim"],
                    latent_dim=dec_args["latent_dim"], patch_size=config["patch_size"])
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted = {n: s for n, s in index["weight_map"].items() if n.startswith("audio.decoder.")}
    handles, state = {}, {}
    for name, shard in wanted.items():
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        state[name[len("audio.decoder."):]] = handles[shard].get_tensor(name).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"state: {len(state)} tensors, missing={list(missing)}, unexpected={list(unexpected)}")
    model.eval().to(device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    model = load_decoder(root)
    latent = torch.from_numpy(np.load(args.latents)["latent"]).float()      # [1, T, 64]
    frames = latent.shape[1]
    print(f"latent: {tuple(latent.shape)} ({frames} patches)")

    with torch.no_grad():
        whole, _, _ = model.low_level_reconstruct(latent, use_cache=False, stream_state=(None, None, None))
        whole = whole.reshape(-1).numpy()

        state, past, chunks = (None, None, None), None, []
        for frame in range(frames):
            chunk, state, past = model.low_level_reconstruct(
                latent[:, frame:frame + 1, :], past_key_values=past, use_cache=True,
                stream_state=state, last_chunk=(frame == frames - 1))
            if chunk.numel():
                chunks.append(chunk.reshape(-1).numpy())
        streamed = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    n = min(whole.size, streamed.size)
    diff = streamed[:n] - whole[:n]
    corr = float(np.corrcoef(whole[:n], streamed[:n])[0, 1]) if n > 1 else float("nan")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "stream-compare.npz", whole=whole, streamed=streamed)
    print(f"whole decode    : {whole.size} samples ({whole.size / 44100:.3f} s) rms {np.sqrt((whole ** 2).mean()):.6f}")
    print(f"per-patch decode: {streamed.size} samples ({streamed.size / 44100:.3f} s) rms {np.sqrt((streamed ** 2).mean()):.6f}")
    print(f"length difference: {whole.size - streamed.size} samples")
    print(f"overlap compared : {n} samples")
    print(f"max_abs          : {np.abs(diff).max():.3e}")
    print(f"rel_rmse         : {np.sqrt((diff ** 2).mean()) / (np.sqrt((whole[:n] ** 2).mean()) or 1e-12):.3e}")
    print(f"correlation      : {corr:.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
