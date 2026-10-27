#!/usr/bin/env python3
"""Compare C++ flow captures against the Python reference, step by step.

    python compare_flow.py --reference out/reference/flow/flow-step.npz ^
        --cpp out/reference/flow-cpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["captures"]:
        data = np.fromfile(directory / entry["file"], dtype="<f4")
        out[entry["name"]] = data.reshape(entry["shape"])
    return out


def compare(name: str, reference: np.ndarray, actual: np.ndarray) -> dict:
    reference = reference.astype(np.float64).reshape(-1)
    actual = actual.astype(np.float64).reshape(-1)
    if reference.shape != actual.shape:
        return {"name": name, "status": "SHAPE", "detail": f"{reference.shape} vs {actual.shape}"}
    diff = actual - reference
    scale = np.sqrt((reference ** 2).mean()) or 1e-12
    return {"name": name, "status": "PASS" if np.abs(diff).max() < 2e-2 else "FAIL",
            "max_abs": float(np.abs(diff).max()), "rel_rmse": float(np.sqrt((diff ** 2).mean()) / scale)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--cpp", required=True, type=Path)
    args = parser.parse_args()

    reference = np.load(args.reference)
    cpp = load(args.cpp)

    pairs = [
        ("cond_post_c_embedder", "cond_post_c_embedder"),
        ("flow_velocity_cond", "velocities_pre_cfg"),
        ("flow_velocity_null", "velocities_null"),
        ("flow_velocity_guided", "velocities"),
        ("flow_latents", "latents"),
    ]
    failures = []
    print(f"{'stage':<22} {'max_abs':>10} {'rel_rmse':>10}")
    for cpp_name, ref_name in pairs:
        if cpp_name not in cpp or ref_name not in reference:
            print(f"{cpp_name:<22} missing (cpp has {cpp_name in cpp}, reference has {ref_name in reference})")
            continue
        row = compare(cpp_name, reference[ref_name], cpp[cpp_name])
        if row["status"] == "SHAPE":
            print(f"{row['name']:<22} SHAPE {row['detail']}")
            failures.append(row["name"])
            continue
        print(f"{row['status']:<5} {cpp_name:<16} {row['max_abs']:>10.2e} {row['rel_rmse']:>10.2e}")
        if row["status"] != "PASS":
            failures.append(cpp_name)

    # trajectory: reference has (steps+1, ...) including y0; cpp has flow_trajectory + per-step x
    trajectory = reference["trajectory"]
    print()
    for step in range(trajectory.shape[0]):
        name = "flow_trajectory" if step == 0 else f"flow_x_step_{step - 1}"
        if name not in cpp:
            continue
        row = compare(name, trajectory[step], cpp[name])
        print(f"{row['status']:<5} x[{step:<2}] {row['max_abs']:>10.2e} {row['rel_rmse']:>10.2e}")
        if row["status"] != "PASS":
            failures.append(name)

    print()
    if failures:
        print(f"first substantial mismatch: {failures[0]}")
        return 1
    print("all flow captures agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
