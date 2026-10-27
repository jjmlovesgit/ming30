#!/usr/bin/env python3
"""Stage A reference: prompt token IDs for the tiny checkpoint, no reference audio.

Uses the checkpoint's own `prompt_contract.py` (pure Python, no torch/fla) and its
tokenizer, so this is the authoritative expected input to the thinker for a given
target text. The C++ prompt assembly must reproduce this list exactly.

    python reference_prompt_stage_a.py --model <checkpoint dir> --text "..." --out stage-a.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--text", default="The garden will remain open until sunset.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    sys.path.insert(0, str(root))

    import prompt_contract as pc
    from transformers import PreTrainedTokenizerFast

    manifest = json.loads((root / "conversion_manifest.json").read_text(encoding="utf-8"))
    tts_config = manifest["tts_config"]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"),
        tokenizer_config_file=str(root / "tokenizer_config.json"),
    )
    adjusted = pc.ensure_legacy_antmmf_tokenizer(tokenizer)
    print(f"tokenizer prepared; adjusted special tokens: {adjusted}")

    ids, speaker_position = pc.build_tts_prompt_ids(tokenizer, args.text, tts_config)
    decoded = tokenizer.decode(ids)
    digest = hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()

    report = {
        "stage": "A_prompt_token_ids",
        "model": str(root),
        "text": args.text,
        "reference_audio": None,
        "prompt_latent_tokens": 0,
        "token_count": len(ids),
        "token_ids": ids,
        "speaker_embedding_position": speaker_position,
        "sha256_of_ids": digest,
        "decoded_text": decoded,
        "tts_config_echo": {
            key: tts_config.get(key)
            for key in ("protocol_version", "prompt_format", "tokenizer_contract",
                        "sampling_contract", "conditioning_mode", "decode_alignment",
                        "task_decode_alignment")
            if key in tts_config
        },
        "prompt_eos_token_id": manifest["tokenizer"]["prompt_eos_token_id"],
        "tts_token_ids": manifest["tokenizer"]["tts_token_ids"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"text            : {args.text!r}")
    print(f"tokens          : {len(ids)}")
    print(f"first 12 ids    : {ids[:12]}")
    print(f"last 12 ids     : {ids[-12:]}")
    print(f"speaker position: {speaker_position}")
    print(f"sha256(ids)     : {digest}")
    print(f"decoded preview : {decoded[:120]!r}")
    print(f"written         : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
