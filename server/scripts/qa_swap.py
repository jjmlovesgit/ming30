"""Round-trip speech-intelligibility QA for a voice-swapped mix.

Transcribes each swapped speaker track with the Ming ASR entry and scores it
against the exact text that was synthesized, using word edit distance.

    python scripts/qa_swap.py --summary out\\sam-swap\\voice_swap_ming_full.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import httpx


def words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", normalized, flags=re.UNICODE)


def word_errors(reference: list[str], recognized: list[str]) -> int:
    previous = list(range(len(recognized) + 1))
    for row, reference_word in enumerate(reference, start=1):
        current = [row]
        for column, recognized_word in enumerate(recognized, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != recognized_word),
                )
            )
        previous = current
    return previous[-1]


def score(reference_text: str, recognized_text: str) -> dict:
    reference = words(reference_text)
    recognized = words(recognized_text)
    errors = word_errors(reference, recognized)
    wer = errors / len(reference) if reference else 0.0
    return {
        "reference_words": len(reference),
        "recognized_words": len(recognized),
        "word_errors": errors,
        "word_error_rate": round(wer, 4),
        "match_percent": round(max(0.0, 1.0 - wer) * 100.0, 1),
    }


def transcribe(client: httpx.Client, path: Path, language: str | None) -> str:
    data = {"model": "whisper-1", "response_format": "json"}
    if language:
        data["language"] = language
    with path.open("rb") as handle:
        response = client.post(
            "/v1/audio/transcriptions",
            data=data,
            files={"file": (path.name, handle, "audio/wav")},
        )
    response.raise_for_status()
    return str(response.json().get("text", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="voice_swap_ming_*.json")
    parser.add_argument("--base", default="http://127.0.0.1:8070")
    parser.add_argument("--language", default="en")
    parser.add_argument("--mix", action="store_true", help="also score the mixed track as one stream")
    parser.add_argument("--compare-wav", default="", help="extra mix to score with the same reference and judge")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = summary_path.parent
    timing = summary.get("segment_timing") or {}

    client = httpx.Client(base_url=args.base, timeout=600.0)
    totals = {"reference_words": 0, "recognized_words": 0, "word_errors": 0}
    print(f"engine: {summary.get('engine')}  voices: {summary.get('voices')}")

    for stem, key in (("target", "speaker_one"), ("residual", "speaker_two")):
        path = out_dir / summary["outputs"][key]
        reference_text = " ".join(
            str(record.get("text", "")).strip()
            for record in timing.get(stem, [])
            if str(record.get("text", "")).strip()
        )
        recognized = transcribe(client, path, args.language)
        result = score(reference_text, recognized)
        for name in totals:
            totals[name] += result[name]
        print(
            f"{stem:<9} ref={result['reference_words']:>4}w rec={result['recognized_words']:>4}w "
            f"errors={result['word_errors']:>2} WER={result['word_error_rate'] * 100:5.2f}% "
            f"match={result['match_percent']}%"
        )

    errors = totals["word_errors"]
    reference_words = totals["reference_words"]
    wer = errors / reference_words if reference_words else 0.0
    print(
        f"{'overall':<9} ref={reference_words:>4}w errors={errors:>2} "
        f"WER={wer * 100:5.2f}% match={(max(0.0, 1.0 - wer) * 100):.1f}%"
    )

    if args.mix:
        # Reference for the mix: turns in rebuilt (output) order, both speakers.
        ordered = sorted(
            (
                (float(record.get("rebuilt_start_seconds", record.get("start_seconds", 0.0))), stem, record)
                for stem, records in timing.items()
                for record in records
            ),
            key=lambda item: item[0],
        )
        mix_reference = " ".join(str(record.get("text", "")).strip() for _, _, record in ordered)
        mix_path = out_dir / summary["outputs"]["mix_wav"]
        mix_result = score(mix_reference, transcribe(client, mix_path, args.language))
        print(
            f"{'mix':<9} ref={mix_result['reference_words']:>4}w rec={mix_result['recognized_words']:>4}w "
            f"errors={mix_result['word_errors']:>2} WER={mix_result['word_error_rate'] * 100:5.2f}% "
            f"match={mix_result['match_percent']}%"
        )
        if args.compare_wav:
            compare_path = Path(args.compare_wav).resolve()
            compare_result = score(mix_reference, transcribe(client, compare_path, args.language))
            print(
                f"{'compare':<9} ({compare_path.name}) "
                f"ref={compare_result['reference_words']:>4}w rec={compare_result['recognized_words']:>4}w "
                f"errors={compare_result['word_errors']:>2} "
                f"WER={compare_result['word_error_rate'] * 100:5.2f}% match={compare_result['match_percent']}%"
            )

    print("\nbaseline (PocketTTS, same run, faster-whisper judge): WER 1.87%, match 98.1%")
    print("note: per-stem and mix numbers above use the Ming ASR judge for both engines")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
