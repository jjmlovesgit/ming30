#!/usr/bin/env python3
"""Validate a Ming-3.0-tts-tiny GGUF written by write_ming3_tts_tiny_gguf.py.

Checks, in order:
  1. the file parses and reports the expected tensor count;
  2. `audiocpp.*` metadata is present and internally consistent (names/ranks/shapes);
  3. expert banks carry the expected shape and dtype;
  4. one bank's bytes match the source checkpoint (proves stacking order + dtype);
  5. a sidecar round-trips byte-for-byte.

    python validate_ming3_tts_tiny_gguf.py --gguf out\\ming-tiny-probe.gguf ^
        --source C:\\Projects\\ming30\\models\\ming3-tts-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import struct

import numpy as np

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label:<44} {detail}")
    if not ok:
        FAILURES.append(label)


def parse_gguf_metadata(path: Path) -> tuple[dict, list[dict]]:
    """Minimal GGUF v3 reader: key/value metadata plus tensor infos."""

    data = path.read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version, = struct.unpack_from("<I", data, 4)
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    offset = 24

    def read_value(offset: int, value_type: int):
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
            raw = data[offset + 8: offset + 8 + length]
            return raw.decode("utf-8", "replace"), offset + 8 + length
        if value_type == 9:
            element_type, = struct.unpack_from("<I", data, offset)
            count, = struct.unpack_from("<Q", data, offset + 4)
            offset += 12
            items = []
            for _ in range(count):
                item, offset = read_value(offset, element_type)
                items.append(item)
            return items, offset
        if value_type in (10, 11, 12):
            fmt = {10: "<Q", 11: "<q", 12: "<d"}[value_type]
            return struct.unpack_from(fmt, data, offset)[0], offset + 8
        raise ValueError(f"unknown GGUF value type {value_type}")

    metadata = {}
    for _ in range(n_kv):
        length, = struct.unpack_from("<Q", data, offset)
        key = data[offset + 8: offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        value_type, = struct.unpack_from("<I", data, offset)
        value, offset = read_value(offset + 4, value_type)
        metadata[key] = value

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
        offset += 4 + 8
        tensors.append({"name": name, "shape": list(dims), "type": tensor_type})

    return metadata, tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bank-layer", type=int, default=1)
    args = parser.parse_args()

    metadata, tensor_infos = parse_gguf_metadata(args.gguf)
    tensors = {info["name"]: info for info in tensor_infos}
    print(f"file {args.gguf}  {args.gguf.stat().st_size/1e6:.0f} MB  tensors={len(tensors)}")

    check("parses as GGUF", True, f"{len(tensors)} tensor infos")
    check("arch metadata", metadata.get("general.architecture") == "ming3_tts_tiny",
          str(metadata.get("general.architecture")))
    check("audiocpp.family", metadata.get("audiocpp.family") == "ming3_tts_tiny",
          str(metadata.get("audiocpp.family")))

    names = metadata.get("audiocpp.tensor_names") or []
    ranks = metadata.get("audiocpp.tensor_ranks") or []
    shapes = metadata.get("audiocpp.tensor_shapes") or []
    banks = [name for name in tensors if name.startswith("_audiocpp.")]
    check("metadata names == _audiocpp tensors", len(names) == len(banks),
          f"names={len(names)} tensors={len(banks)}")
    check("rank/shape metadata consistent", sum(ranks) == len(shapes),
          f"sum(ranks)={sum(ranks)} shapes={len(shapes)}")

    expected = [f"_audiocpp.{index}" for index, name in enumerate(names) if ".mlp.experts." in name and name.endswith(".weight")]
    check("expert banks present", bool(expected), f"{len(expected)} banks")

    if expected:
        bank_index = next(
            index for index, name in enumerate(names)
            if name == f"model.layers.{args.bank_layer}.mlp.experts.gate_proj.weight"
        ) if f"model.layers.{args.bank_layer}.mlp.experts.gate_proj.weight" in names else None
        if bank_index is None:
            check("layer bank found", False, f"layer {args.bank_layer}")
        else:
            info = tensors[f"_audiocpp.{bank_index}"]
            # GGUF serialises dims in ggml order (fastest-varying first), so the
            # numpy [128, 512, 1536] bank appears as [1536, 512, 128].
            check("bank shape (ggml order)", info["shape"] == [1536, 512, 128],
                  f"{info['shape']} == reversed([128, 512, 1536])")
            check("bank dtype", info["type"] == 30, f"type {info['type']} (30=BF16)")

            # compare hashes against the source stack
            from safetensors import safe_open

            index_map = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
            digest = hashlib.sha256()
            import torch

            for expert in range(128):
                name = f"model.model.layers.{args.bank_layer}.mlp.experts.{expert}.gate_proj.weight"
                with safe_open(args.source / index_map[name], framework="pt", device="cpu") as handle:
                    digest.update(handle.get_tensor(name).view(torch.uint16).numpy().tobytes())
            from gguf import GGUFReader
            reader = GGUFReader(str(args.gguf), mode="r")
            gguf_bytes = {t.name: t for t in reader.tensors}[f"_audiocpp.{bank_index}"].data.tobytes()
            check("bank bytes equal source stack",
                  hashlib.sha256(gguf_bytes).hexdigest() == digest.hexdigest(),
                  f"{len(gguf_bytes)/1e6:.0f} MB compared")

    # sidecar round-trip
    embedded = metadata.get("audiocpp.files") or []
    check("sidecars listed", "config.json" in embedded, f"{len(embedded)} embedded")
    if "config.json" in embedded:
        payload = bytes(metadata.get("audiocpp.file.config.json") or b"")
        original = (args.source / "config.json").read_bytes()
        check("sidecar bytes match file", payload == original,
              f"{len(payload)} vs {len(original)} bytes")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
