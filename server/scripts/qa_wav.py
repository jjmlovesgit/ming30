"""Score one or more WAVs against a reference transcript with faster-whisper.

Run with the SAM environment's interpreter (it has faster-whisper):

    C:\\Users\\Jim\\miniconda3\\envs\\sam-audio-v2\\python.exe ^
        scripts\\qa_wav.py --reference out\\ab\\excerpt_reference.txt ^
        --wav out\\ab\\excerpt_ming.wav --wav out\\ab\\excerpt_moss.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projects\SAM_fastapi")

from transcription import FasterWhisperTranscriber, word_match_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--wav", action="append", required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    reference = Path(args.reference).read_text(encoding="utf-8").strip()
    transcriber = FasterWhisperTranscriber()
    print(f"judge: faster-whisper {args.model} on {args.device}\n")
    for wav in args.wav:
        transcript = transcriber.transcribe_stem(
            Path(wav), "excerpt", "excerpt", args.model, args.device, args.language
        )
        recognized = " ".join(segment["text"] for segment in transcript.get("segments", []))
        metrics = word_match_metrics(reference, recognized)
        print(
            f"{Path(wav).name:<26} ref={metrics['reference_words']:>4}w "
            f"rec={metrics['recognized_words']:>4}w errors={metrics['word_errors']:>2} "
            f"WER={metrics['word_error_rate'] * 100:5.2f}% match={metrics['match_percent']:>5}%"
        )
        print(f"   heard: {recognized[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
