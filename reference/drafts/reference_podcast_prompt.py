#!/usr/bin/env python3
"""Podcast prompt reference: token ids for a two-speaker exchange.

Construction, from modeling_lingv3_tts.py:

  input_ids = build_tts_prompt_ids(tokenizer, target_text,
                                   prompt_text=<tagged reference transcript>,
                                   prompt_latent_tokens=<reference patches>)

  - `target_text` is the multi-speaker script: lines of the form " speaker_N:text\\n"
  - `prompt_text` is the reference transcript in the same tagged form
  - `prompt_latent_tokens` = ceil(reference latent frames / patch_size); the final
    `prompt_latent_tokens` ids are `<audioPatch>` placeholders that get replaced by the
    projected reference latents (see the embeddings substitution in the reference)
  - the `<spk>` slot takes one CAMPPlus embedding; podcast multi-speaker comes from the
    script text, not from multiple speaker slots

    python reference_podcast_prompt.py --model models/ming3-tts-tiny --out out/reference/podcast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT = " speaker_1:Good morning, and welcome to the show.\n speaker_2:Thanks for having me.\n"
REFERENCE = " speaker_1:Good morning.\n speaker_2:Glad to be here.\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt-latent-tokens", type=int, default=100)
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

    ids, speaker_position = pc.build_tts_prompt_ids(
        tokenizer, SCRIPT, tts_config,
        prompt_text=REFERENCE, prompt_latent_tokens=args.prompt_latent_tokens,
    )
    audio_start = int(tts_config["audio_start_token_id"])
    audio_patch = int(tts_config["audio_patch_token_id"])
    prompt_start = len(ids) - args.prompt_latent_tokens
    placeholders = ids[prompt_start:]

    report = {
        "stage": "podcast_prompt_ids",
        "target_text": SCRIPT,
        "reference_text": REFERENCE,
        "prompt_latent_tokens": args.prompt_latent_tokens,
        "token_count": len(ids),
        "token_ids": ids,
        "sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
        "speaker_embedding_position": speaker_position,
        "audio_start_token_id": audio_start,
        "audio_patch_token_id": audio_patch,
        "prompt_latent_start_index": prompt_start,
        "all_trailing_tokens_are_patch_placeholders": all(t == audio_patch for t in placeholders),
        "decoded_tail": tokenizer.decode(ids[-12:]),
        "seed": 1235,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "podcast-prompt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"script        : {SCRIPT!r}")
    print(f"reference     : {REFERENCE!r}")
    print(f"latent tokens : {args.prompt_latent_tokens}")
    print(f"total ids     : {len(ids)}  sha256 {report['sha256'][:16]}...")
    print(f"speaker pos   : {speaker_position}")
    print(f"prompt start  : index {prompt_start}, trailing ids all <audioPatch>: "
          f"{report['all_trailing_tokens_are_patch_placeholders']}")
    print(f"decoded tail  : {report['decoded_tail']!r}")
    print(f"written       : {args.out / 'podcast-prompt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
