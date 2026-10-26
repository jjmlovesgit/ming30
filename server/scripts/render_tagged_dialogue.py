"""Render a speaker-tagged dialogue through any dialogue-capable TTS family.

    python scripts/render_tagged_dialogue.py `
        --engine http://127.0.0.1:8090 --model moss-ttsd `
        --text out\\ab\\excerpt_tagged.txt `
        --ref-one server\\voices\\james_ref.wav --ref-two server\\voices\\emma_ref.wav `
        --reference-text "The garden will remain open until sunset, ..." `
        --out out\\ab\\excerpt_moss.wav

Some dialogue families (for example MOSS-TTSD) clone by *continuation*: the references are handed to the model as
audio it already produced, so ``--reference-text`` must transcribe them, tagged
per speaker in the same [S1]/[S2] order.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import httpx
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:8090")
    parser.add_argument("--model", default="moss-ttsd"  # any configured family id)
    parser.add_argument("--text", required=True, help="file with the [S1]/[S2] tagged script")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref-one", default="", help="reference wav for [S1]")
    parser.add_argument("--ref-two", default="", help="reference wav for [S2]")
    parser.add_argument("--reference-text", default="", help="transcript of the references")
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    script = Path(args.text).read_text(encoding="utf-8").strip()
    payload: dict = {
        "model": args.model,
        "input": script,
        "response_format": "wav",
        "options": {"language": args.language},
    }
    if args.ref_one or args.ref_two:
        payload["options"]["voice_samples"] = f"{args.ref_one},{args.ref_two}"
    if args.reference_text:
        payload["options"]["reference_text"] = args.reference_text

    print(f"script: {len(script.split())} words, tags S1={script.count('[S1]')} S2={script.count('[S2]')}")
    with httpx.Client(timeout=1800.0) as client:
        response = client.post(f"{args.engine.rstrip('/')}/v1/audio/speech", json=payload)
    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {response.text[:500]}")
        return 1

    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mono, rate)
    print(f"saved {out}  {mono.size / rate:.2f}s @{rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

