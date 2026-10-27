#!/usr/bin/env python3
"""Sweep the whole thinker: per-layer error growth, expert-set flips, final state.

For every layer: attention drift, hidden drift (relative RMSE and max abs), router
score drift, and whether the selected expert set changed. When it did, report the
reference's biased selection margin at the cutoff next to the observed score drift, so
a boundary event can be told apart from a routing error.

    python sweep_thinker.py --reference out/reference/thinker/thinker-41.npz ^
        --cpp out/reference/thinker-cpp --model models/ming3-tts-tiny
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_dump(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["captures"]:
        dtype = "<i4" if entry.get("dtype") == "int32" else "<f4"
        data = np.fromfile(root / entry["file"], dtype=dtype)
        out[entry["name"]] = data.reshape(entry["shape"])
    return out


def rel(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    reference = reference.astype(np.float64)
    actual = actual.astype(np.float64)
    diff = actual - reference
    scale = np.sqrt((reference ** 2).mean()) or 1e-12
    return float(np.sqrt((diff ** 2).mean()) / scale), float(np.abs(diff).max())


def load_bias(model: Path, layer: int) -> np.ndarray:
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    name = f"model.model.layers.{layer}.mlp.gate.expert_bias"
    with safe_open(model / index["weight_map"][name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).float().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    reference = np.load(args.reference)
    cpp = load_dump(args.cpp)
    layers = sorted(int(name.split("_")[-1]) for name in cpp if name.startswith("layer_"))

    print(f"{'layer':>5} {'kind':>4} {'attn_rel':>10} {'hidden_rel':>11} {'hidden_max':>11} "
          f"{'pre_router':>11} {'scores_max':>11} {'flips':>6} {'ref_margin':>11}")
    flips_total = 0
    for layer in layers:
        kind = "MLA" if layer % 4 == 3 else "KDA"
        attn_rel = rel(reference[f"attn_out_layer_{layer}"], cpp[f"attn_out_layer_{layer}"])[0]
        hidden_rel, hidden_max = rel(reference[f"layer_{layer}"], cpp[f"layer_{layer}"])
        pre = "-"
        scores_max = "-"
        flips = "-"
        margin = "-"
        if f"routing_layer_{layer}_ids" in cpp:
            pre = f"{rel(reference[f'pre_router_layer_{layer}'], cpp[f'pre_router_layer_{layer}'])[0]:.1e}"
            scores_max = f"{np.abs(cpp[f'router_scores_layer_{layer}'] - reference[f'router_scores_layer_{layer}']).max():.1e}"
            ref_ids = reference[f"routing_layer_{layer}_ids"]
            cpp_ids = cpp[f"routing_layer_{layer}_ids"]
            bad = [token for token in range(ref_ids.shape[0])
                   if set(ref_ids[token].tolist()) != set(cpp_ids[token].tolist())]
            flips = str(len(bad))
            flips_total += len(bad)
            if bad:
                bias = load_bias(args.model.resolve(), layer)
                worst = None
                for token in bad:
                    biased = reference[f"router_scores_layer_{layer}"][token] + bias
                    order = np.argsort(-biased)
                    cut = float(biased[order[7]] - biased[order[8]])
                    if worst is None or cut < worst[0]:
                        worst = (cut, token)
                margin = f"{worst[0]:+.1e}"
        print(f"{layer:>5} {kind:>4} {attn_rel:>10.1e} {hidden_rel:>11.1e} {hidden_max:>11.1e} "
              f"{pre:>11} {scores_max:>11} {flips:>6} {margin:>11}")

    final_rel, final_max = rel(reference["final_norm"], cpp["final_norm"])
    stop_ref = reference["stop_logits"].astype(np.float64).reshape(-1)
    stop_cpp = cpp["stop_logits"].astype(np.float64).reshape(-1)
    print()
    print(f"final_norm      rel_rmse={final_rel:.3e} max_abs={final_max:.3e}")
    print(f"stop_logits     reference={np.round(stop_ref, 5).tolist()} cpp={np.round(stop_cpp, 5).tolist()} "
          f"max_abs={np.abs(stop_cpp - stop_ref).max():.3e}")
    print(f"stop_decision   reference={int(np.argmax(stop_ref))} cpp={int(np.argmax(stop_cpp))} "
          f"{'SAME' if int(np.argmax(stop_ref)) == int(np.argmax(stop_cpp)) else 'DIFFERENT'}")
    print(f"\nexpert-set flips total: {flips_total} token-layers of {len(layers) * 41} compared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
