#!/usr/bin/env python3
"""Stage A reference, segment by segment.

`_build_legacy_antmmf_prompt_ids` does not tokenise one concatenated string: it
encodes each constant, then the target text, then the assistant prefix, and inserts
special token ids in between. Tokenising the joined string can differ at segment
boundaries, so the C++ side must reproduce the same segmentation — and this file
records the expected ids for every segment, not just the total.

    python reference_prompt_segments_stage_a.py --model <ckpt> --text "..." --out segments.json
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
    pc.ensure_legacy_antmmf_tokenizer(tokenizer)

    rml = pc.with_rml(tts_config)
    segments = []

    def add(name: str, text: str | None = None, ids: list[int] | None = None) -> None:
        entry = {"name": name}
        if text is not None:
            entry["text"] = text
        encoded = ids if ids is not None else pc.encode(tokenizer, text)
        entry["ids"] = encoded
        segments.append(entry)

    add("user_prefix", pc.LEGACY_USER_PREFIX if rml else pc.LEGACY_USER_PREFIX_NO_RML)
    if rml:
        add("speaker_block", ids=[
            int(tts_config["speaker_start_token_id"]),
            int(tts_config["audio_patch_token_id"]),
            int(tts_config["speaker_end_token_id"]),
        ])
    add("text_prefix", pc.LEGACY_TEXT_PREFIX if rml else pc.LEGACY_TEXT_PREFIX_NO_RML)
    add("target_text", args.text)
    add("assistant_prefix", pc.LEGACY_ASSISTANT_PREFIX)
    add("audio_start", ids=[int(tts_config["audio_start_token_id"])])

    flat = [token for segment in segments for token in segment["ids"]]
    ids, speaker_position = pc.build_tts_prompt_ids(tokenizer, args.text, tts_config)
    consistent = flat == ids

    report = {
        "stage": "A_prompt_segments",
        "text": args.text,
        "with_rml": rml,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_file": "tokenizer.json",
        },
        "special_token_ids": {
            "speaker_start": int(tts_config["speaker_start_token_id"]),
            "audio_patch": int(tts_config["audio_patch_token_id"]),
            "speaker_end": int(tts_config["speaker_end_token_id"]),
            "audio_start": int(tts_config["audio_start_token_id"]),
            "prompt_eos": manifest["tokenizer"]["prompt_eos_token_id"],
        },
        "segments": segments,
        "flat_ids": flat,
        "reference_ids": ids,
        "segments_match_reference": consistent,
        "speaker_embedding_position": speaker_position,
        "sha256_of_ids": hashlib.sha256(",".join(str(i) for i in flat).encode()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"with_rml        : {rml}")
    print(f"segments        : {len(segments)}")
    for segment in segments:
        text = segment.get("text")
        preview = (text[:40] + "...") if text and len(text) > 40 else text
        print(f"  {segment['name']:<18} ids={len(segment['ids']):>3}  {preview!r}")
    print(f"total ids       : {len(flat)}  (segments match builder: {consistent})")
    print(f"sha256(ids)     : {report['sha256_of_ids']}")
    print(f"written         : {args.out}")
    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
