#!/usr/bin/env python3
"""Build and verify an explicit HF -> package tensor name map from GGUF metadata.

Not a blanket prefix rule: `lm_head`, the talker/vae components and the CAMPPlus
speaker tensors each have their own path, and routed expert banks are many-to-one
(128 HF tensors -> one package tensor). The map is derived from the package itself
(`audiocpp.tensor_names`) and checked against the checkpoint index, so a wrong rule
fails here rather than 20 layers into a port.

    python build_name_map.py --package out/ming-tiny-bf16-full.gguf ^
        --model models/ming3-tts-tiny --out reference/name-map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")


def gguf_tensor_names(path: Path) -> list[dict]:
    """Read audiocpp.tensor_names/ranks/shapes plus the physical tensor list."""

    data = path.read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError("not GGUF")
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
        raise ValueError(f"unknown GGUF value type {value_type}")

    metadata = {}
    for _ in range(n_kv):
        length, = struct.unpack_from("<Q", data, offset)
        key = data[offset + 8: offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        value_type, = struct.unpack_from("<I", data, offset)
        metadata[key], offset = value(offset + 4, value_type)

    names = metadata.get("audiocpp.tensor_names") or []
    ranks = metadata.get("audiocpp.tensor_ranks") or []
    shapes = metadata.get("audiocpp.tensor_shapes") or []
    entries, position = [], 0
    for name, rank in zip(names, ranks):
        entries.append({"name": name, "shape": shapes[position: position + rank]})
        position += rank
    return entries


def classify(name: str) -> str:
    if EXPERT_RE.match(name):
        return "expert_bank"
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.word_embeddings.weight":
        return "embedding"
    if name == "model.norm.weight":
        return "final_norm"
    if name.startswith("model.layers."):
        return "backbone_layer"
    if name.startswith(("audio.encoder.", "audio.decoder.")):
        return "vae"
    if name.startswith(("flowloss.", "linear_proj_audio.", "spk_head", "stop_head")):
        return "talker"
    if name.startswith("speaker_encoder."):
        return "speaker_campplus"
    return "unmapped"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    entries = gguf_tensor_names(args.package)
    index = json.loads((args.model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    hf = set(index["weight_map"])

    mapped: dict[str, set[str]] = {}
    by_family: dict[str, int] = {}
    unmapped = []
    for entry in entries:
        name = entry["name"]
        family = classify(name)
        by_family[family] = by_family.get(family, 0) + 1
        if family == "expert_bank":
            match = EXPERT_RE.match(name)
            layer, projection = int(match[1]), match[2]
            sources = [f"model.model.layers.{layer}.mlp.experts.{expert}.{projection}.weight" for expert in range(128)]
            missing = [source for source in sources if source not in hf]
            if missing:
                print(f"FAIL bank source missing: {missing[:2]}")
                return 2
            mapped[name] = set(sources)
        elif family == "embedding":
            mapped[name] = {"model.model.word_embeddings.weight"}
        elif family == "final_norm":
            mapped[name] = {"model.model.norm.weight"}
        elif family == "lm_head":
            mapped[name] = {"model.lm_head.weight"}
        elif family == "backbone_layer":
            mapped[name] = {"model.model." + name[len("model."):]}
        elif family == "speaker_campplus":
            mapped[name] = set()  # derived from campplus.onnx, not a safetensors tensor
        elif family in ("vae", "talker"):
            mapped[name] = {name}
        else:
            unmapped.append(name)

    consumed = set().union(*mapped.values()) if mapped else set()
    missing_sources = sorted({source for sources in mapped.values() for source in sources if source not in hf})
    unconsumed = sorted(hf - consumed)

    print(f"package tensors : {len(entries)}")
    print(f"HF tensors      : {len(hf)}")
    print(f"families        : {json.dumps(by_family, sort_keys=True)}")
    print(f"unmapped names  : {len(unmapped)}")
    print(f"missing sources : {len(missing_sources)}")
    print(f"unconsumed HF   : {len(unconsumed)}")

    report = {
        "package": str(args.package),
        "model": str(args.model),
        "family_counts": by_family,
        "mapping": {name: sorted(sources) for name, sources in sorted(mapped.items())},
        "missing_sources": missing_sources,
        "unconsumed_hf": unconsumed,
        "notes": {
            "expert_bank": "128 HF tensors collapse into one package tensor, stacked on the expert axis",
            "speaker_campplus": "derived from campplus.onnx, so it has no safetensors source",
            "backbone_layer": "package drops exactly one leading 'model.'",
            "lm_head": "package uses 'lm_head.weight'; HF keeps it under 'model.'",
            "vae/talker": "names are unchanged between HF and package",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nsample per family:")
    seen = set()
    for entry in entries:
        family = classify(entry["name"])
        if family in seen:
            continue
        seen.add(family)
        sources = sorted(mapped[entry["name"]])
        source = sources[0] if sources else "(campplus.onnx)"
        extra = f"  (+{len(sources) - 1} more)" if len(sources) > 1 else ""
        print(f"  {family:<16} {entry['name']:<56} <- {source}{extra}")
    print(f"\nwritten: {args.out}")

    ok = not unmapped and not missing_sources
    if unconsumed:
        print(f"note: {len(unconsumed)} HF tensors are not consumed (expected for excluded training tensors)")
    print("name map:", "VERIFIED" if ok else "FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
