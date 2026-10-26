#!/usr/bin/env python3
"""Package Ming-3.0-tts-tiny for audio.cpp.

DRAFT (decision D8: developed in our fork, shaped to move under community_models/).
Modelled on tools/models/convert_ming3_audio_gguf.py, which is the precedent for a
multi-component Ming conversion:

  * the MoE backbone is written here, stacking routed experts into `_audiocpp.{i}`
    banks whose logical names and shapes travel in GGUF metadata;
  * the acoustic components are handed to `audiocpp_gguf` as named safetensors
    inputs (`thinker=`, `talker=`, `vae=`, `speaker=`) plus the family spec and
    per-prefix `--keep-type` rules;
  * CAMPPlus is repacked from the published ONNX by the shared helper.

`--dry-run` classifies the checkpoint from config.json + the weight index only and
writes nothing, so the tensor plan can be reviewed before a 16 GB conversion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

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

    bank_report = {}
    for layer, counter in sorted(expert_layers.items()):
        bank_report[layer] = {
            "experts": max(counter.values()),
            "projections": sorted(counter),
            "complete": all(counter[p] == experts for p in counter),
        }

    return {
        "layers": layers,
        "experts": experts,
        "top_k": top_k,
        "components": dict(components),
        "unclassified": unclassified,
        "expert_layers": sorted(expert_layers),
        "expert_tensors": expert_tensor_count,
        "dense_layer_tensors": dense_tensors,
        "banks": bank_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--output", type=Path, help="target GGUF (required unless --dry-run)")
    parser.add_argument("--type", choices=("orig", "q8_0", "q4_k"), default="q8_0")
    parser.add_argument("--converter", type=Path, help="path to audiocpp_gguf")
    parser.add_argument("--temporary-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", action="store_true")
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
        print("\nUNCLASSIFIED tensors (mapping decision needed):")
        for name in report["unclassified"][:20]:
            print("  ", name)
        return 2

    print(f"\nexpert tensors  : {report['expert_tensors']} across layers {report['expert_layers']}")
    print(f"dense layer tens : {report['dense_layer_tensors']}")
    incomplete = [layer for layer, info in report["banks"].items() if not info["complete"]]
    if incomplete:
        print(f"INCOMPLETE expert banks on layers {incomplete}")
        return 3
    for layer, info in report["banks"].items():
        print(f"  layer {layer}: {info['experts']} experts x {len(info['projections'])} projections")

    # naive size estimate: 2 bytes per parameter at bf16, experts at the target type
    total_tensors = sum(report["components"].values())
    print(f"\nplan: {total_tensors} tensors -> thinker banks + talker/vae/speaker components")

    if args.dry_run:
        print("\ndry run: nothing written. Next step is the real conversion once weights are local.")
        return 0

    if not args.output or not args.converter or not args.temporary_dir:
        parser.error("--output, --converter and --temporary-dir are required without --dry-run")

    print("\nconversion path not implemented yet: the expert-bank writer and the audiocpp_gguf "
          "invocation are decided (see the audio converter), but the tensor-name mapping for "
          "tiny must be confirmed against a real shard before writing 16 GB.", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
