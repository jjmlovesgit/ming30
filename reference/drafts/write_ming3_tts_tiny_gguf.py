#!/usr/bin/env python3
"""Write a Ming-3.0-tts-tiny GGUF using the observed audio.cpp container contract.

Layout (from docs/gguf-container-contract.md, read off a real Ming package):

  general.architecture                 "audiocpp"
  audiocpp.model_spec.family           "ming3_tts_tiny"
  audiocpp.model_spec.json             family spec, embedded (self-describing)
  audiocpp.tensor_names/ranks/shapes   logical name of EVERY tensor
  physical tensors                     _audiocpp.{i}
  audiocpp.embedded_files.{names,offsets,data}   config, tokenizer, chat template, ...

`--type q8_0` quantises only the routed expert banks; every other tensor keeps
source precision, including `lm_head` — a quantised output head ends turns early.

Modes: --dry-run (classify only), --probe (layer-0 + one layer's banks + all
components), --write (full package).

Scope: a probe package is a STRUCTURAL test. It proves layout, quantisation and
embedded-file delivery. It says nothing about inference parity.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXPERT_RE = re.compile(r"^model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

COMPONENT_PREFIXES = (
    ("model.model.", "thinker"),
    ("model.lm_head", "thinker"),
    ("audio.encoder.", "vae"),
    ("audio.decoder.", "vae"),
    ("flowloss.cfm.", "talker"),
    ("linear_proj_audio.", "talker"),
    ("spk_head", "talker"),
    ("stop_head", "talker"),
)

SIDECARS = (
    "config.json", "generation_config.json", "conversion_manifest.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja",
)

TYPE = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30}
DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def is_bank(logical: str) -> bool:
    """Routed expert bank: model.layers.N.mlp.experts.{gate,up,down}_proj.weight.

    The predicate shape matters: bank names end in `experts.gate_proj.weight`, so a
    check for `experts.weight` matches nothing and silently skips quantisation.
    """

    return ".mlp.experts." in logical and logical.endswith(".weight")


def classify(name: str) -> str:
    for prefix, component in COMPONENT_PREFIXES:
        if name.startswith(prefix):
            return component
    return "unclassified"


def load_plan(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {"config": config, "weight_map": index["weight_map"]}


def analyse(plan: dict) -> dict:
    llm = plan["config"].get("llm_config") or {}
    layers = int(llm.get("num_hidden_layers", 0))
    experts = int(llm.get("num_experts", 0))
    components: Counter[str] = Counter()
    unclassified: list[str] = []
    expert_layers: dict[int, Counter[str]] = defaultdict(Counter)
    expert_tensors = 0

    for name in plan["weight_map"]:
        component = classify(name)
        components[component] += 1
        if component == "unclassified":
            unclassified.append(name)
            continue
        match = EXPERT_RE.match(name)
        if match:
            expert_layers[int(match.group(1))][match.group(3)] += 1
            expert_tensors += 1

    banks = {
        layer: {"experts": max(counter.values()), "projections": sorted(counter),
                "complete": all(counter[p] == experts for p in counter)}
        for layer, counter in sorted(expert_layers.items())
    }
    return {
        "layers": layers, "experts": experts, "top_k": int(llm.get("num_experts_per_tok") or 0),
        "components": dict(components), "unclassified": unclassified,
        "expert_tensors": expert_tensors, "banks": banks,
    }


def open_headers(source: Path, weight_map: dict) -> dict:
    headers = {}
    for shard in dict.fromkeys(weight_map.values()):
        with (source / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        header.pop("__metadata__", None)
        headers.update(header)
    return headers


class Tensors:
    def __init__(self, source: Path, weight_map: dict):
        from safetensors import safe_open

        self.source, self.weight_map = source, weight_map
        self.handles: dict[str, object] = {}
        self._safe_open = safe_open

    def handle(self, name):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = self._safe_open(self.source / shard, framework="pt", device="cpu")
        return self.handles[shard]

    def get(self, name, dtype):
        import torch

        tensor = self.handle(name).get_tensor(name)
        if dtype == "BF16":
            return np.ascontiguousarray(tensor.view(torch.uint16).numpy())
        return np.ascontiguousarray(tensor.numpy())


def build_plan(plan: dict, headers: dict, campplus_path: Path) -> list[dict]:
    weight_map = plan["weight_map"]
    llm = plan["config"]["llm_config"]
    layers = int(llm["num_hidden_layers"])
    experts = int(llm["num_experts"])
    first_moe = int(llm.get("first_k_dense_replace", 1))

    items: list[dict] = []
    for name in weight_map:
        if classify(name) != "thinker" or EXPERT_RE.match(name):
            continue
        logical = name[len("model."):] if name.startswith("model.model.") else "lm_head.weight"
        items.append({"logical": logical, "shape": list(headers[name]["shape"]),
                      "dtype": headers[name]["dtype"], "kind": "single", "source": name})

    for layer in range(first_moe, layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            sources = [f"model.model.layers.{layer}.mlp.experts.{i}.{projection}.weight"
                       for i in range(experts)]
            first = headers[sources[0]]
            items.append({"logical": f"model.layers.{layer}.mlp.experts.{projection}.weight",
                          "shape": [experts, *first["shape"]], "dtype": first["dtype"],
                          "kind": "bank", "source": sources})

    for name in weight_map:
        kind = classify(name)
        if kind in ("talker", "vae"):
            items.append({"logical": name, "shape": list(headers[name]["shape"]),
                          "dtype": headers[name]["dtype"], "kind": "single", "source": name})

    if campplus_path.exists():
        from convert_ming3_tts_gguf import campplus_weights

        for name, values in campplus_weights(campplus_path).items():
            items.append({"logical": name, "shape": list(values.shape), "dtype": "F32",
                          "kind": "speaker", "source": name, "values": values})
    return items


def gather(item: dict, tensors: Tensors) -> np.ndarray:
    if item["kind"] == "speaker":
        return np.ascontiguousarray(item["values"])
    if item["kind"] == "bank":
        first = tensors.get(item["source"][0], item["dtype"])
        out = np.empty((len(item["source"]), *first.shape), dtype=first.dtype)
        for index, name in enumerate(item["source"]):
            out[index] = tensors.get(name, item["dtype"])
        return out
    return tensors.get(item["source"], item["dtype"])


def q8_blocks(values: np.ndarray) -> np.ndarray:
    from gguf import quants

    width = int(values.shape[-1])
    if width % 32:
        raise ValueError(f"last dimension {width} is not a multiple of 32 for Q8_0")
    rows = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
    flat = values.reshape(rows, width)
    if flat.dtype == np.uint16:
        # gather() hands back bf16 bit patterns as uint16; quantising those bits as
        # integers produces garbage, so widen bf16 -> f32 first (low half zeroed).
        flat = (flat.astype(np.uint32) << 16).view(np.float32)
    else:
        flat = flat.astype(np.float32)
    return np.ascontiguousarray(quants.quantize(flat, quants.GGMLQuantizationType.Q8_0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec", type=Path, help="family spec JSON to embed")
    parser.add_argument("--type", choices=("orig", "q8_0"), default="orig")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-layer", type=int, default=1)
    parser.add_argument("--max-sidecar-bytes", type=int, default=64_000_000)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    plan = load_plan(source)
    report = analyse(plan)

    print(f"checkpoint      : {source}")
    print(f"model_type      : {plan['config'].get('model_type')}  arch={plan['config'].get('architectures')}")
    print(f"layers / experts: {report['layers']} / {report['experts']} (top-{report['top_k']})")
    for component, count in sorted(report["components"].items()):
        print(f"  {component:<13} {count:>5}")
    if report["unclassified"]:
        print("UNCLASSIFIED:", report["unclassified"][:10])
        return 2
    incomplete = [layer for layer, info in report["banks"].items() if not info["complete"]]
    if incomplete:
        print(f"INCOMPLETE banks on layers {incomplete}")
        return 3
    print(f"expert tensors  : {report['expert_tensors']} in {len(report['banks'])} MoE layers, all complete")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    if not args.output:
        parser.error("--output is required unless --dry-run")

    import gguf

    headers = open_headers(source, plan["weight_map"])
    tensors = Tensors(source, plan["weight_map"])
    items = build_plan(plan, headers, source / "campplus.onnx")

    if args.probe:
        kept = []
        for item in items:
            logical = item["logical"]
            match = re.match(r"^model\.layers\.(\d+)\.", logical)
            if is_bank(logical):
                if match and int(match[1]) == args.probe_layer:
                    kept.append(item)
            elif match is not None:
                if int(match[1]) == 0:
                    kept.append(item)
            else:
                kept.append(item)
        items = kept

    quant_banks = args.type == "q8_0"
    writer = gguf.GGUFWriter(str(args.output), "audiocpp")
    writer.add_string("general.architecture", "audiocpp")
    writer.add_string("general.name", f"ming3-tts-tiny-{'probe' if args.probe else 'full'}")
    writer.add_string("audiocpp.tensor_name_format", "native")
    writer.add_string("audiocpp.source_format", ".gguf")
    writer.add_string("audiocpp.weight_type", "q8_0" if quant_banks else "orig")
    writer.add_uint32("audiocpp.model_spec.version", 1)
    writer.add_string("audiocpp.model_spec.family", "ming3_tts_tiny")
    if args.spec and args.spec.exists():
        writer.add_string("audiocpp.model_spec.json", args.spec.read_text(encoding="utf-8"))

    writer.add_array("audiocpp.tensor_names", [item["logical"] for item in items])
    writer.add_key_value("audiocpp.tensor_ranks", [len(item["shape"]) for item in items],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_key_value("audiocpp.tensor_shapes", [d for item in items for d in item["shape"]],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)
    writer.add_array("audiocpp.tensor_sources.names", [""])
    writer.add_array("audiocpp.tensor_sources.paths", [str(args.output.resolve())])

    file_names, file_blobs, total = [], [], 0
    for name in SIDECARS:
        path = source / name
        if not path.exists():
            print(f"  sidecar absent, not embedded: {name}")
            continue
        blob = path.read_bytes()
        if len(blob) > args.max_sidecar_bytes:
            raise SystemExit(f"sidecar {name} is {len(blob)} B, above --max-sidecar-bytes")
        file_names.append(name)
        file_blobs.append(blob)
        total += len(blob)
    offsets, running = [], 0
    for blob in file_blobs:
        offsets.append(running)
        running += len(blob)
    offsets.append(running)
    writer.add_array("audiocpp.embedded_files.names", file_names)
    writer.add_key_value("audiocpp.embedded_files.offsets", offsets,
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)
    writer.add_array("audiocpp.embedded_files.data", b"".join(file_blobs))
    print(f"  embedded files  : {len(file_names)} totaling {total/1e6:.1f} MB "
          f"(tokenizer.json {'included' if 'tokenizer.json' in file_names else 'MISSING'})")

    payloads = []
    for index, item in enumerate(items):
        dtype = item["dtype"]
        values = gather(item, tensors)
        if quant_banks and is_bank(item["logical"]):
            blocks = q8_blocks(values)
            # gguf-py wants the *byte* shape for a quantised tensor: the last dim is
            # bytes-per-row (values/32*34), which it converts back to logical dims.
            byte_shape = list(item["shape"])
            byte_shape[-1] = byte_shape[-1] // 32 * 34
            writer.add_tensor_info(f"_audiocpp.{index}", byte_shape, np.dtype(np.uint8),
                                   blocks.nbytes, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            payloads.append(blocks)
        else:
            numpy_type = DTYPES[dtype]
            raw = np.ascontiguousarray(values.astype(numpy_type, copy=False))
            writer.add_tensor_info(f"_audiocpp.{index}", item["shape"], np.dtype(numpy_type),
                                   raw.nbytes, raw_dtype=gguf.GGMLQuantizationType(TYPE[dtype]))
            payloads.append(raw)
        if args.log and index % 500 == 0:
            print(f"  planned {index + 1}/{len(items)}", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    banks_written = 0
    for index, payload in enumerate(payloads):
        writer.write_tensor_data(payload)
        banks_written += int(is_bank(items[index]["logical"]))
        if args.log and index % 500 == 0:
            print(f"  wrote {index + 1}/{len(payloads)}", flush=True)
    writer.close()

    print(f"\nwrote {args.output}  {args.output.stat().st_size/1e6:.1f} MB")
    print(f"  tensors          : {len(items)} ({banks_written} expert banks, "
          f"{'q8_0' if quant_banks else 'source precision'})")
    print("  scope            : STRUCTURAL test only; no inference parity is implied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
