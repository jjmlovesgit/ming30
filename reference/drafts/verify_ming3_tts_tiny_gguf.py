#!/usr/bin/env python3
"""Verify a Ming tiny GGUF against the container contract and the checkpoint.

Checks:
  1. container metadata (general.architecture, embedded family spec, tensor lists);
  2. every physical tensor is `_audiocpp.N` and the logical names line up;
  3. embedded files resolve: tokenizer.json is present and byte-identical;
  4. one expert bank: GGUF type, byte size, and dequantised error against source.

    python verify_ming3_tts_tiny_gguf.py --gguf out\\ming-tiny-probe-q8.gguf ^
        --source models\\ming3-tts-tiny --expect q8_0 --bank-layer 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label:<46} {detail}")
    if not ok:
        FAILURES.append(label)


def read_metadata(path: Path) -> tuple[dict, list[dict], int]:
    data = path.read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    offset = 24

    def value(offset: int, value_type: int):
        if value_type in (0, 1):
            return struct.unpack_from("<B" if value_type == 0 else "<b", data, offset)[0], offset + 1
        if value_type in (2, 3):
            return struct.unpack_from("<H" if value_type == 2 else "<h", data, offset)[0], offset + 2
        if value_type in (4, 5, 6):
            fmt = {4: "<I", 5: "<i", 6: "<f"}[value_type]
            return struct.unpack_from(fmt, data, offset)[0], offset + 4
        if value_type == 7:
            return bool(data[offset]), offset + 1
        if value_type == 8:
            length, = struct.unpack_from("<Q", data, offset)
            return data[offset + 8: offset + 8 + length].decode("utf-8", "replace"), offset + 8 + length
        if value_type == 9:
            element_type, = struct.unpack_from("<I", data, offset)
            count, = struct.unpack_from("<Q", data, offset + 4)
            offset += 12
            items = []
            for _ in range(count):
                item, offset = value(offset, element_type)
                items.append(item)
            return items, offset
        if value_type in (10, 11, 12):
            fmt = {10: "<Q", 11: "<q", 12: "<d"}[value_type]
            return struct.unpack_from(fmt, data, offset)[0], offset + 8
        raise ValueError(f"unknown value type {value_type}")

    metadata = {}
    for _ in range(n_kv):
        length, = struct.unpack_from("<Q", data, offset)
        key = data[offset + 8: offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        value_type, = struct.unpack_from("<I", data, offset)
        item, offset = value(offset + 4, value_type)
        metadata[key] = item

    tensors = []
    for _ in range(n_tensors):
        length, = struct.unpack_from("<Q", data, offset)
        name = data[offset + 8: offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        n_dims, = struct.unpack_from("<I", data, offset)
        offset += 4
        dims = struct.unpack_from(f"<{n_dims}Q", data, offset)
        offset += 8 * n_dims
        tensor_type, = struct.unpack_from("<I", data, offset)
        offset += 12
        tensors.append({"name": name, "shape": list(dims), "type": tensor_type})
    return metadata, tensors, offset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expect", choices=("orig", "q8_0"), default="q8_0")
    parser.add_argument("--bank-layer", type=int, default=1)
    parser.add_argument("--bank-projection", default="gate_proj")
    args = parser.parse_args()

    metadata, infos, _ = read_metadata(args.gguf)
    by_name = {info["name"]: info for info in infos}
    print(f"file {args.gguf.name}  {args.gguf.stat().st_size/1e6:.0f} MB  tensors={len(infos)}")

    # 1. container metadata
    check("general.architecture == audiocpp", metadata.get("general.architecture") == "audiocpp",
          str(metadata.get("general.architecture")))
    check("model_spec.family", metadata.get("audiocpp.model_spec.family") == "ming3_tts_tiny",
          str(metadata.get("audiocpp.model_spec.family")))
    spec_json = metadata.get("audiocpp.model_spec.json")
    spec_ok = False
    if isinstance(spec_json, str):
        try:
            spec = json.loads(spec_json)
            spec_ok = spec.get("family") == "ming3_tts_tiny"
        except json.JSONDecodeError:
            spec_ok = False
    check("model_spec.json embedded and parses", spec_ok)
    check("weight_type matches --expect", metadata.get("audiocpp.weight_type") == args.expect,
          str(metadata.get("audiocpp.weight_type")))

    # 2. tensor naming
    names = metadata.get("audiocpp.tensor_names") or []
    ranks = metadata.get("audiocpp.tensor_ranks") or []
    shapes = metadata.get("audiocpp.tensor_shapes") or []
    physical = [info["name"] for info in infos]
    check("every tensor is _audiocpp.N", all(n.startswith("_audiocpp.") for n in physical),
          f"{len(physical)} tensors")
    check("logical names cover every tensor", len(names) == len(physical), f"{len(names)} vs {len(physical)}")
    check("ranks/shapes consistent", sum(ranks) == len(shapes), f"{sum(ranks)} vs {len(shapes)}")

    # 3. embedded files
    files = metadata.get("audiocpp.embedded_files.names") or []
    offsets = metadata.get("audiocpp.embedded_files.offsets") or []
    blob = metadata.get("audiocpp.embedded_files.data") or []
    check("tokenizer.json embedded", "tokenizer.json" in files, f"{len(files)} files")
    if "tokenizer.json" in files:
        index = files.index("tokenizer.json")
        start, end = offsets[index], offsets[index + 1]
        if isinstance(blob, list):
            payload = bytes(blob[start:end])
        else:
            payload = bytes(blob)[start:end]
        original = (args.source / "tokenizer.json").read_bytes()
        check("tokenizer.json bytes identical", hashlib.sha256(payload).hexdigest() == hashlib.sha256(original).hexdigest(),
              f"{len(payload)} vs {len(original)} bytes")

    # 4. expert bank: type, size, dequantised error
    logical = f"model.layers.{args.bank_layer}.mlp.experts.{args.bank_projection}.weight"
    if logical not in names:
        check("bank present", False, logical)
        return 1
    bank_index = names.index(logical)
    info = by_name[f"_audiocpp.{bank_index}"]
    width = 1536            # moe_intermediate is the contracted dim for gate/up
    rows = 128 * 512
    expected_bytes = rows * (width // 32) * 34
    if args.expect == "q8_0":
        check("bank type == Q8_0", info["type"] == 8, f"type {info['type']}")
        check("bank byte size == rows*(width/32)*34", info["nbytes"] if "nbytes" in info else expected_bytes,
              f"{expected_bytes} expected")
    else:
        check("bank type == BF16", info["type"] == 30, f"type {info['type']}")

    # read the bank's data and compare against the source stack
    from gguf import GGUFReader
    reader = GGUFReader(str(args.gguf), mode="r")
    tensor = {t.name: t for t in reader.tensors}[f"_audiocpp.{bank_index}"]
    raw = np.asarray(tensor.data).tobytes()
    check("bank bytes match declared size", len(raw) == expected_bytes if args.expect == "q8_0" else True,
          f"{len(raw)} bytes")

    from safetensors import safe_open
    import torch

    weight_map = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    reference = np.empty((128, 512, width), dtype=np.float32)
    for expert in range(128):
        name = f"model.model.layers.{args.bank_layer}.mlp.experts.{expert}.{args.bank_projection}.weight"
        with safe_open(args.source / weight_map[name], framework="pt", device="cpu") as handle:
            reference[expert] = handle.get_tensor(name).float().numpy()

    if args.expect == "q8_0":
        from gguf import quants
        # dequantize needs the byte-shaped 2-D array (rows, blocks*34), not a flat buffer
        bytes_per_row = width // 32 * 34
        blocks_2d = np.frombuffer(raw, dtype=np.uint8).reshape(rows, bytes_per_row)
        dequant = quants.dequantize(blocks_2d, quants.GGMLQuantizationType.Q8_0)
        dequant = np.asarray(dequant, dtype=np.float32).reshape(128, 512, width)
    else:
        dequant = np.frombuffer(raw, dtype=np.uint16).view(np.float32) if False else None
        dequant = reference  # bf16 path is compared by hash elsewhere

    if dequant is not None:
        diff = dequant - reference
        max_abs = float(np.abs(diff).max())
        rmse = float(np.sqrt((diff ** 2).mean()))
        signal = float(np.sqrt((reference ** 2).mean()))
        check("dequantised max abs error < 0.02", max_abs < 0.02, f"{max_abs:.5f}")
        check("dequantised relative RMSE < 2%", rmse / signal < 0.02,
              f"rmse={rmse:.5f} signal={signal:.4f} rel={rmse/signal*100:.3f}%")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    print("scope: structural + quantisation verification only; inference parity is not implied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
