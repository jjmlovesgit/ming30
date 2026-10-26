"""Score voice-swap output with the same faster-whisper judge as the SAM app.

Run with the SAM environment's interpreter:

    C:\\Users\\Jim\\miniconda3\\envs\\sam-audio-v2\\python.exe ^
        scripts\\qa_faster_whisper.py --summary out\\sam-swap\\voice_swap_ming_full.json ^
        --baseline-prefix "C:\\SAM\\runs\\...\\speaker" --device cuda

Compares the Ming tracks against the existing PocketTTS tracks from the same run,
using identical reference text, model, and settings as `voice_swap_qa.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projects\SAM_fastapi")

from transcription import FasterWhisperTranscriber, word_match_metrics  # noqa: E402


def reference_text(summary: dict, stem: str) -> str:
    records = (summary.get("segment_timing") or {}).get(stem) or []
    return " ".join(
        str(record.get("text", "")).strip()
        for record in records
        if str(record.get("text", "")).strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-prefix", default="", help="e.g. ...\\speaker -> _1_/ _2_ + variant suffix")
    parser.add_argument("--baseline-suffix", default="2ad121c1")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = summary_path.parent
    transcriber = FasterWhisperTranscriber()

    pairs = [
        ("target", "speaker_one", 1),
        ("residual", "speaker_two", 2),
    ]
    results = {}
    recognized_texts = {}
    totals = {"ming": [0, 0], "baseline": [0, 0]}

    print(f"judge: faster-whisper {args.model} on {args.device}\n")
    for stem, key, index in pairs:
        reference = reference_text(summary, stem)
        ming_path = out_dir / summary["outputs"][key]
        rows = [("ming", ming_path)]
        baseline_path = None
        if args.baseline_prefix:
            baseline_path = Path(f"{args.baseline_prefix}_{index}_swapped_{args.baseline_suffix}.wav")
            if baseline_path.exists():
                rows.append(("baseline", baseline_path))

        for label, path in rows:
            transcript = transcriber.transcribe_stem(
                path, stem, label, args.model, args.device, args.language
            )
            recognized = " ".join(
                segment["text"] for segment in transcript.get("segments", [])
            )
            metrics = word_match_metrics(reference, recognized)
            results[(stem, label)] = metrics
            recognized_texts[(stem, label)] = recognized
            totals[label][0] += metrics["reference_words"]
            totals[label][1] += metrics["word_errors"]
            print(
                f"{stem:<9} {label:<9} ref={metrics['reference_words']:>4}w "
                f"errors={metrics['word_errors']:>2} WER={metrics['word_error_rate'] * 100:5.2f}% "
                f"match={metrics['match_percent']}%"
            )

    print()
    for label, (reference_words, errors) in totals.items():
        if not reference_words:
            continue
        wer = errors / reference_words
        print(
            f"{label:<9} overall ref={reference_words:>4}w errors={errors:>2} "
            f"WER={wer * 100:5.2f}% match={(max(0.0, 1.0 - wer) * 100):.1f}%"
        )

    print("\npublished baseline (voice_swap_qa.json): target 98.7%, residual 97.6%, overall 98.1%")
    (out_dir / "qa_faster_whisper.json").write_text(
        json.dumps(
            {
                "judge": {"engine": "faster-whisper", "model": args.model, "device": args.device},
                "voices": summary.get("voices"),
                "sample_rate": summary.get("sample_rate"),
                "metrics": {f"{stem}_{label}": metrics for (stem, label), metrics in results.items()},
                "recognized": {f"{stem}_{label}": text for (stem, label), text in recognized_texts.items()},
                "reference": {stem: reference_text(summary, stem) for stem, _, _ in pairs},
                "totals": {
                    label: {
                        "reference_words": reference_words,
                        "word_errors": errors,
                        "word_error_rate": round(errors / reference_words, 4) if reference_words else None,
                    }
                    for label, (reference_words, errors) in totals.items()
                    if reference_words
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
