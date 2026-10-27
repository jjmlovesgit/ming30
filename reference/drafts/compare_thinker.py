#!/usr/bin/env python3
"""Compare a C++ thinker dump against the plain-torch reference.

Protocol (agreed 2026-10-26):
  1. embedding first — it involves no model maths, so a mismatch there is tokenizer
     or weight-loading, not attention;
  2. then per-layer full-value comparison: max abs error and relative RMSE, not L2
     (different vectors can share an L2);
  3. at MoE layers also compare expert id sets and routing weights;
  4. report the *first* substantial mismatch — that is the component to inspect.

L2 is printed as a warning signal only, never as a pass condition.

    python compare_thinker.py --reference out/reference/thinker/thinker-41.npz ^
        --cpp out/reference/thinker-cpp
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# relative RMSE thresholds per stage. Embedding is exact (same bf16 weights, same
# conversion), so it must match bit-for-bit; deeper stages allow bf16/f32 ordering.
THRESHOLDS = {
    "embedding": 1e-7,
    "default": 2e-2,
}


def load_cpp_dump(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = {}
    for entry in manifest.get("captures", []):
        path = root / entry["file"]
        dtype = "<i4" if entry.get("dtype") == "int32" else "<f4"
        data = np.fromfile(path, dtype=dtype)
        captures[entry["name"]] = data.reshape(entry["shape"])

    def order(name: str):
        # embedding first, then per layer: routing ids, routing weights, hidden
        if name == "embedding":
            return (0, 0, 0)
        match = re.match(r"routing_layer_(\d+)_(ids|weights)$", name)
        if match:
            return (1, int(match[1]), 0 if match[2] == "ids" else 1)
        match = re.match(r"layer_(\d+)$", name)
        if match:
            return (1, int(match[1]), 2)
        return (2, 0, 0)

    captures = dict(sorted(captures.items(), key=lambda item: order(item[0])))
    return {"manifest": manifest, "captures": captures}


def compare(name: str, reference: np.ndarray, actual: np.ndarray, threshold: float) -> dict:
    if reference.shape != actual.shape:
        return {"name": name, "status": "SHAPE", "detail": f"{reference.shape} vs {actual.shape}"}
    ref = reference.astype(np.float64)
    act = actual.astype(np.float64)
    diff = act - ref
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    rmse = float(np.sqrt((diff ** 2).mean())) if diff.size else 0.0
    signal = float(np.sqrt((ref ** 2).mean())) or 1e-12
    rel = rmse / signal
    status = "PASS" if rel <= threshold and max_abs <= max(threshold * max(1.0, float(np.abs(ref).max())), 1e-6) else "FAIL"
    return {
        "name": name,
        "status": status,
        "max_abs": max_abs,
        "rel_rmse": rel,
        "ref_l2": float(np.sqrt((ref ** 2).sum())),
        "act_l2": float(np.sqrt((act ** 2).sum())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path, help="reference .npz")
    parser.add_argument("--cpp", required=True, type=Path, help="C++ dump directory")
    parser.add_argument("--stage", default="", help="compare only this stage (e.g. embedding)")
    args = parser.parse_args()

    ref_npz = np.load(args.reference)
    cpp = load_cpp_dump(args.cpp)
    cpp_captures = cpp["captures"]
    print(f"reference: {args.reference}")
    print(f"cpp      : {args.cpp} ({len(cpp_captures)} captures)\n")

    names = [name for name in cpp_captures if (not args.stage or name == args.stage)]
    if "embedding" in names:
        names.remove("embedding")
        names.insert(0, "embedding")

    rows = []
    first_failure = None
    for name in names:
        if name not in ref_npz:
            print(f"{name:<14} reference has no such capture")
            continue
        if cpp_captures[name].dtype.kind == "i":
            reference_ids = ref_npz[name].astype(np.int64)
            actual_ids = cpp_captures[name].astype(np.int64)
            mismatches = int((reference_ids != actual_ids).any(axis=-1).sum())
            total = int(reference_ids.shape[0])
            status = "PASS" if mismatches == 0 else "FAIL"
            print(f"{status:<5} {name:<24} token rows matching={total - mismatches}/{total}")
            if mismatches and status == "FAIL" and first_failure is None:
                first_failure = name
                bad = int(np.argmax((reference_ids != actual_ids).any(axis=-1)))
                print(f"      first differing token {bad}: reference {reference_ids[bad].tolist()} "
                      f"vs cpp {actual_ids[bad].tolist()}")
            continue
        threshold = THRESHOLDS.get(name, THRESHOLDS["default"])
        row = compare(name, ref_npz[name], cpp_captures[name], threshold)
        rows.append(row)
        if row["status"] == "SHAPE":
            print(f"{row['status']:<5} {name:<14} {row['detail']}")
            first_failure = first_failure or name
            continue
        print(f"{row['status']:<5} {name:<14} max_abs={row['max_abs']:.3e} "
              f"rel_rmse={row['rel_rmse']:.3e} l2_ref={row['ref_l2']:.3f} l2_cpp={row['act_l2']:.3f}")
        if row["status"] == "FAIL" and first_failure is None:
            first_failure = name

    print()
    if first_failure is None:
        print("all compared captures agree")
        return 0
    print(f"first substantial mismatch: {first_failure} — inspect that component")
    return 1


if __name__ == "__main__":
    sys.exit(main())
