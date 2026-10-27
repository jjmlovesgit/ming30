#!/usr/bin/env python3
"""One seeded podcast patch in Python, to check the C++ prefill at 150 tokens.

Same 150-row embedding matrix and the same first noise vector the C++ render used. Compares
the first generated latent, which exercises the thinker cache at a longer, audio-filled prompt
than the 41-token case.

    python podcast_patch_check.py --model ... --prefill prefill-150.f32 --noise noises.f32 ^
        --cpp out/.../cpp-render/latent_0.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_reference as fr
import thinker_reference as tr
import tts_loop as tl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prefill", required=True, type=Path)
    parser.add_argument("--noise", required=True, type=Path)
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    store = tr.WeightStore(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["llm_config"]
    flow_weights = fr.FlowWeights(root)

    rows = np.fromfile(args.prefill, dtype="<f4").reshape(-1, 1536)
    noises = np.fromfile(args.noise, dtype="<f4").reshape(-1, 64)
    embeds = torch.from_numpy(rows).float().unsqueeze(0)
    print(f"prefill         : {tuple(embeds.shape)} ({rows.shape[0]} rows)")

    with torch.no_grad():
        final = tl.thinker_forward_embeds(embeds, store, cfg)
        last = final[:, -1:, :]
        stop_logits = torch.nn.functional.linear(last, store.get("stop_head.weight"), store.get("stop_head.bias"))
        noise = torch.from_numpy(noises[0]).float().reshape(1, 64, 1)
        result = fr.flow_sample(last, noise, torch.zeros(1, 1, 64), flow_weights, steps=10, cfg=2.0)
        latent = torch.from_numpy(result["latents"]).reshape(-1).numpy()

    cpp = np.fromfile(args.cpp, dtype="<f4").astype(np.float64)
    diff = latent.astype(np.float64) - cpp
    relative = float(np.sqrt((diff ** 2).mean()) / (np.sqrt((cpp ** 2).mean()) or 1e-12))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "podcast-patch0.npz", final_last=last.numpy(), latent=latent,
                        latent_cpp=cpp, stop_logits=stop_logits.numpy())
    print(f"hidden last     : rms {float(last.pow(2).mean().sqrt()):.5f}  norm {float(last.norm()):.2f}")
    print(f"stop logits     : {[round(float(v), 4) for v in stop_logits.reshape(-1)]}")
    print(f"latent python   : rms {np.sqrt((latent ** 2).mean()):.6f}")
    print(f"latent cpp      : rms {np.sqrt((cpp ** 2).mean()):.6f}")
    print(f"max_abs         : {np.abs(diff).max():.3e}")
    print(f"rel_rmse        : {relative:.3e}")
    print(f"verdict         : {'PASS' if np.abs(diff).max() < 1e-3 else 'FAIL'}")
    print(f"written         : {args.out / 'podcast-patch0.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
