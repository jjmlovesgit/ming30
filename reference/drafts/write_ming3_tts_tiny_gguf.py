#!/usr/bin/env python3
"""Write a Ming-3.0-tts-tiny GGUF from the published checkpoint (route B).

Route B because `audiocpp_gguf` validates `--family` against the registry, and the
`ming3_tts_tiny` family does not exist in the runtime yet. This writer therefore
*defines* the layout the C++ family will implement, deliberately mirroring
`ming3_audio`:

  * MoE backbone as `_audiocpp.{i}` banks; logical names, ranks and shapes travel in
    `audiocpp.tensor_names` / `tensor_ranks` / `tensor_shapes` metadata, exactly the
    convention `convert_ming3_audio_gguf.py` uses;
  * talker / vae / speaker tensors under `talker/`, `vae/`, `speaker/` prefixes;
  * sidecar files embedded as `audiocpp.file.<relative>` byte arrays plus an index.

Modes:
    --dry-run   classify only, write nothing
    --probe     metadata + talker/vae/speaker + layer-0 dense + layer-1 banks only
    --write     full conversion

Quantisation: `--type orig` writes bf16/f32 as stored. `--type q8_0` quantises the
expert banks per row and keeps every other tensor at source precision, including
`lm_head` (the MOSS-TTSD lesson: a quantised output head ends turns early).
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
LAYER_RE = re.compile(r"^model\.model\.layers\.(\d+)\.")

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

DTYPES = {
    "BF16": (np.uint16, 30),
    "F16": (np.float16, 1),
    "F32": (np.float32, 0),
}


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
    config = plan["config"]
    llm = config.get("llm_config") or {}
    layers = int(llm.get("num_hidden_layers", 0))
    experts = int(llm.get("num_experts", 0))
    top_k = int(llm.get("num_experts_per_tok") or 0)

    components: Counter[str] = Counter()
    unclassified: list[str] = []
    expert_layers: dict[int, Counter[str]] = defaultdict(Counter)
    dense_tensors = 0
    expert_tensor_count = 0

    for name in plan["weight_map"]:
        component = classify(name)
        components[component] += 1
        if component == "unclassified":
            unclassified.append(name)
            continue
        match = EXPERT_RE.match(name)
        if match:
            expert_layers[int(match.group(1))][match.group(3)] += 1
            expert_tensor_count += 1
        elif LAYER_RE.match(name):
            dense_tensors += 1

    banks = {}
    for layer, counter in sorted(expert_layers.items()):
        banks[layer] = {"experts": max(counter.values()), "projections": sorted(counter),
                        "complete": all(counter[p] == experts for p in counter)}

    return {
        "layers": layers, "experts": experts, "top_k": top_k,
        "components": dict(components), "unclassified": unclassified,
        "expert_layers": sorted(expert_layers),
        "expert_tensors": expert_tensor_count, "dense_layer_tensors": dense_tensors,
        "banks": banks,
    }


# ---------------------------------------------------------------- source access

def open_headers(source: Path, weight_map: dict):
    """Read per-tensor shape/dtype from safetensors headers without loading data."""

    headers = {}
    for shard in dict.fromkeys(weight_map.values()):
        with (source / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        header.pop("__metadata__", None)
        for name, info in header.items():
            headers[name] = info
    return headers


class Tensors:
    """Lazy per-shard safetensors access."""

    def __init__(self, source: Path, weight_map: dict):
        from safetensors import safe_open

        self.source = source
        self.weight_map = weight_map
        self.handles = {}
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


# ---------------------------------------------------------------- thinker plan

def build_thinker_plan(plan: dict, headers: dict) -> list[tuple[str, list[int], str, object]]:
    """(logical_name, shape, dtype, single_source_name | [sources]) in write order."""

    weight_map = plan["weight_map"]
    llm = plan["config"]["llm_config"]
    layers = int(llm["num_hidden_layers"])
    experts = int(llm["num_experts"])
    first_moe = int(llm.get("first_k_dense_replace", 1))

    plan_items = []
    for name in weight_map:
        if classify(name) != "thinker":
            continue
        match = EXPERT_RE.match(name)
        if match:
            continue  # banks are appended below
        logical = name[len("model."):] if name.startswith("model.model.") else "lm_head.weight"
        plan_items.append((logical, headers[name]["shape"], headers[name]["dtype"], name))

    for layer in range(first_moe, layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            sources = [f"model.model.layers.{layer}.mlp.experts.{i}.{projection}.weight"
                       for i in range(experts)]
            first = headers[sources[0]]
            for name in sources:
                if headers[name]["shape"] != first["shape"] or headers[name]["dtype"] != first["dtype"]:
                    raise ValueError(f"Inconsistent expert bank at {name}")
            plan_items.append((
                f"model.layers.{layer}.mlp.experts.{projection}.weight",
                [experts, *first["shape"]], first["dtype"], sources,
            ))
    return plan_items


def gather_thinker(item, tensors):
    _, _, dtype, source_names = item
    if isinstance(source_names, list):
        first = tensors.get(source_names[0], dtype)
        values = np.empty((len(source_names), *first.shape), dtype=first.dtype)
        for index, name in enumerate(source_names):
            values[index] = tensors.get(name, dtype)
        return values
    return tensors.get(source_names, dtype)


# ---------------------------------------------------------------- gguf writing

def quantize_rows(values: np.ndarray, quant_type) -> bytes:
    """Q8_0 quantisation of the last axis, flattened to 2-D rows."""

    from gguf import quants

    rows = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
    width = values.shape[-1]
    if width % 32:
        raise ValueError(f"last dimension {width} is not a multiple of 32; cannot use Q8_0")
    flat = values.reshape(rows, width).astype(np.float32)
    blocks = quants.quantize(flat, quant_type)
    return np.ascontiguousarray(blocks).tobytes()


def add_tensor(writer, gguf, name, values, logical_shape, dtype, quant_type=None):
    numpy_type, raw_type_id = DTYPES[dtype]
    if quant_type is not None:
        data = quantize_rows(values, quant_type)
        raw_dtype = quant_type
        nbytes = len(data)
    else:
        data = np.ascontiguousarray(values.astype(numpy_type, copy=False)).tobytes()
        raw_dtype = gguf.GGMLQuantizationType(raw_type_id)
        nbytes = len(data)
    writer.add_tensor_info(name, logical_shape, np.dtype(numpy_type), nbytes, raw_dtype=raw_dtype)
    # GGUFWriter.write_tensor_data needs a numpy array (it asserts on .nbytes)
    return np.frombuffer(data, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--type", choices=("orig", "q8_0"), default="orig")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-layer", type=int, default=1)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--sidecar-limit", type=int, default=1_000_000,
                        help="embed sidecars up to this many bytes; larger ones are listed as skipped")
    args = parser.parse_args()

    source = args.source.resolve()
    plan = load_plan(source)
    report = analyse(plan)

    print(f"checkpoint      : {source}")
    print(f"model_type      : {plan['config'].get('model_type')}  arch={plan['config'].get('architectures')}")
    print(f"layers / experts: {report['layers']} / {report['experts']}  (top-{report['top_k']})")
    print(f"tensors         : {sum(report['components'].values())}")
    for component, count in sorted(report["components"].items()):
        print(f"  {component:<13} {count:>5}")
    if report["unclassified"]:
        print("\nUNCLASSIFIED:", report["unclassified"][:10])
        return 2
    incomplete = [layer for layer, info in report["banks"].items() if not info["complete"]]
    if incomplete:
        print(f"INCOMPLETE expert banks on layers {incomplete}")
        return 3
    print(f"expert tensors  : {report['expert_tensors']} in {len(report['banks'])} MoE layers, all banks complete")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    if not args.output:
        parser.error("--output is required unless --dry-run")

    import gguf
    from gguf import quants
    from convert_ming3_tts_gguf import campplus_weights  # shared CAMPPlus repacking

    headers = open_headers(source, plan["weight_map"])
    tensors = Tensors(source, plan["weight_map"])
    thinker = build_thinker_plan(plan, headers)
    quant_type = quants.GGMLQuantizationType.Q8_0 if args.type == "q8_0" else None

    # probe narrows the write set but keeps the full metadata
    keep_banks = {args.probe_layer} if args.probe else None
    selected = []
    for item in thinker:
        logical = item[0]
        if keep_banks is not None:
            match = re.match(r"^model\.layers\.(\d+)\.", logical)
            is_bank = ".mlp.experts." in logical and logical.endswith(".weight")
            if is_bank:
                if match is None or int(match[1]) not in keep_banks:
                    continue
            elif match is not None and int(match[1]) != 0:
                continue
        selected.append(item)

    writer = gguf.GGUFWriter(str(args.output), "ming3_tts_tiny")
    writer.add_string("audiocpp.family", "ming3_tts_tiny")
    writer.add_string("audiocpp.probe", "layer-%d" % args.probe_layer if args.probe else "full")
    writer.add_string("audiocpp.expert_dtype", "q8_0" if quant_type else "source")

    thinker_out = selected if args.probe else thinker
    writer.add_array("audiocpp.tensor_names", [name for name, *_ in thinker_out])
    writer.add_key_value("audiocpp.tensor_ranks", [len(shape) for _, shape, _, _ in thinker_out],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_key_value("audiocpp.tensor_shapes", [d for _, shape, _, _ in thinker_out for d in shape],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)

    # sidecars
    embedded, skipped = [], []
    for name in SIDECARS:
        path = source / name
        if not path.exists():
            skipped.append(f"{name}:absent")
            continue
        data = path.read_bytes()
        if len(data) > args.sidecar_limit:
            skipped.append(f"{name}:>{args.sidecar_limit}B")
            continue
        writer.add_array(f"audiocpp.file.{name}", data)
        embedded.append(name)
    writer.add_array("audiocpp.files", embedded)
    writer.add_array("audiocpp.files_skipped", skipped)
    print(f"  sidecars embedded: {embedded}")
    if skipped:
        print(f"  sidecars skipped : {skipped}")

    # component tensors keep real names with a prefix
    components = []
    for name in plan["weight_map"]:
        kind = classify(name)
        if kind == "thinker":
            continue
        prefix = "talker" if kind == "talker" else "vae"
        components.append((f"{prefix}/{name}", name, headers[name]["dtype"]))

    speaker = {}
    campplus = source / "campplus.onnx"
    if campplus.exists():
        speaker = campplus_weights(campplus)
        for name in speaker:
            components.append((f"speaker/{name}", None, "F32"))

    # tensor infos: thinker as _audiocpp.{i}, components by real name
    payload = []
    thinker_quant = {}
    for index, item in enumerate(thinker_out):
        logical, shape, dtype, _ = item
        use_quant = quant_type if (quant_type is not None and logical.endswith("experts.weight")) else None
        thinker_quant[index] = use_quant
        values = gather_thinker(item, tensors)
        data = add_tensor(writer, gguf, f"_audiocpp.{index}", values, shape, dtype, use_quant)
        payload.append((f"_audiocpp.{index}", data, shape, dtype, use_quant))

    for physical, original, dtype in components:
        if original is not None:
            shape = list(headers[original]["shape"])
        else:
            shape = list(speaker[physical.split("speaker/", 1)[1]].shape)
        numpy_type, raw = DTYPES[dtype]
        nbytes = int(np.prod(shape)) * np.dtype(numpy_type).itemsize
        writer.add_tensor_info(physical, shape, np.dtype(numpy_type), nbytes,
                               raw_dtype=gguf.GGMLQuantizationType(raw))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    written = 0
    for name, data, shape, dtype, _ in payload:
        if data is None:
            numpy_type, _ = DTYPES[dtype]
            zeros = np.zeros(int(np.prod(shape)), dtype=numpy_type)
            writer.write_tensor_data(zeros)
        else:
            writer.write_tensor_data(data)
        written += 1
        if args.log and written % 200 == 0:
            print(f"  thinker {written}/{len(payload)}", flush=True)

    comp_data = []
    for physical, original, dtype in components:
        if original is None:
            values = speaker[physical.split("speaker/", 1)[1]]
            comp_data.append(values.astype(np.float32))
        else:
            comp_data.append(tensors.get(original, dtype))
    for values in comp_data:
        writer.write_tensor_data(np.ascontiguousarray(values))
    writer.close()

    size = args.output.stat().st_size
    print(f"\nwrote {args.output}  {size/1e6:.1f} MB")
    print(f"  thinker tensors : {len(payload)} (data for {len(selected)})")
    print(f"  component tensor: {len(components)}")
    print(f"  sidecars        : {embedded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
