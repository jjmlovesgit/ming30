"""Probe Ming chunking strategies on the turns that lost words.

Run with the SAM environment (needs faster-whisper + requests):

    C:\\Users\\Jim\\miniconda3\\envs\\sam-audio-v2\\python.exe ^
        scripts\\tune_chunking.py --summary out\\sam-swap\\voice_swap_ming_full.json --turns 4
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, r"C:\Projects\SAM_fastapi")
from transcription import FasterWhisperTranscriber, word_match_metrics  # noqa: E402

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def synth(base: str, text: str, voice: str, options: dict) -> tuple[int, np.ndarray]:
    payload = {"model": "tts-1", "input": text, "voice": voice, "response_format": "wav", "language": "en"}
    if options:
        payload["options"] = options
    response = requests.post(f"{base.rstrip('/')}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    return int(rate), audio.mean(axis=1)


def synth_sentences(base: str, text: str, voice: str, rate: int) -> tuple[int, np.ndarray]:
    pieces = [piece.strip() for piece in SPLIT_RE.split(text) if piece.strip()]
    chunks = []
    for piece in pieces:
        piece_rate, audio = synth(base, piece, voice, {})
        if piece_rate != rate:
            raise RuntimeError(f"unexpected rate {piece_rate}")
        chunks.append(audio)
        chunks.append(np.zeros(int(0.12 * rate), dtype=np.float32))
    return rate, np.concatenate(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8070")
    parser.add_argument("--turns", type=int, default=4, help="how many of the longest turns to test")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    voices = summary["voices"]
    records = [
        (stem, record)
        for stem, items in (summary.get("segment_timing") or {}).items()
        for record in items
    ]
    records.sort(key=lambda item: float(item[1]["transcript_window_seconds"]), reverse=True)
    selected = records[: args.turns]

    transcriber = FasterWhisperTranscriber()
    variants = [
        ("default", {"kind": "options", "options": {}}),
        ("chunk120", {"kind": "options", "options": {"text_chunk_size": 120}}),
        ("chunk50", {"kind": "options", "options": {"text_chunk_size": 50}}),
        ("raw", {"kind": "options", "options": {"raw_text": True}}),
        ("sentences", {"kind": "sentences"}),
    ]

    print(f"testing {len(selected)} turns x {len(variants)} strategies\n")
    totals = {name: [0, 0] for name, _ in variants}
    for stem, record in selected:
        text = str(record["text"]).strip()
        voice = voices["speaker_1_target" if stem == "target" else "speaker_2_residual"]
        print(f"== {record['segment_id']} ({stem}, {float(record['transcript_window_seconds']):.1f}s source, {len(text.split())} words)")
        print(f"   {text[:110]}{'...' if len(text) > 110 else ''}")
        for name, spec in variants:
            if spec["kind"] == "sentences":
                rate, audio = synth_sentences(args.base, text, voice, 44100)
            else:
                rate, audio = synth(args.base, text, voice, spec["options"])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(summary_path.parent)) as handle:
                path = Path(handle.name)
            sf.write(path, np.clip(audio, -1.0, 1.0), rate)
            transcript = transcriber.transcribe_stem(path, stem, "probe", args.model, args.device, "en")
            path.unlink(missing_ok=True)
            recognized = " ".join(segment["text"] for segment in transcript.get("segments", []))
            metrics = word_match_metrics(text, recognized)
            totals[name][0] += metrics["reference_words"]
            totals[name][1] += metrics["word_errors"]
            print(
                f"   {name:<10} gen={audio.size / rate:5.2f}s "
                f"WER={metrics['word_error_rate'] * 100:5.2f}% match={metrics['match_percent']:5.1f}%"
            )
        print()

    print("strategy totals")
    for name, (reference_words, errors) in totals.items():
        wer = errors / reference_words * 100 if reference_words else 0.0
        print(f"  {name:<10} ref={reference_words:>4}w errors={errors:>2} WER={wer:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
