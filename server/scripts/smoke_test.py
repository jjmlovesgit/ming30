"""End-to-end smoke test for the ming30 adapter.

    python scripts/smoke_test.py [--base http://127.0.0.1:8070] [--reference ref.wav]

Requires the adapter and its audio.cpp backend to be running. Each check prints
what it verified, so a partial failure still shows the last good state.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import sys
import time

import httpx

TEXT = "The garden will remain open until sunset, provided the weather stays clear."
HITS: list[tuple[str, bool, str]] = []


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()


def check(name: str, ok: bool, detail: str = "") -> None:
    HITS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8070")
    parser.add_argument("--reference", default="", help="WAV used for the clone-upload check")
    parser.add_argument("--voice-name", default="smoke_voice")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=300.0)

    # 1. health
    health = client.get("/health")
    upstream = health.json().get("backend", {}) if health.status_code == 200 else {}
    check("health", health.status_code == 200 and upstream.get("status") == "ok", f"audio.cpp {upstream.get('version', '?')}")

    # 2. models
    models = client.get("/v1/models").json().get("data", [])
    check("models listed", len(models) > 0, f"{len(models)} ids")

    # 3. plain wav
    started = time.time()
    wav_response = client.post("/v1/audio/speech", json={"input": TEXT, "response_format": "wav"})
    wall = time.time() - started
    seconds = float(wav_response.headers.get("x-ming30-audio-seconds", 0) or 0)
    ok = wav_response.status_code == 200 and wav_response.content[:4] == b"RIFF" and seconds > 0
    rtf = wall / seconds if seconds else 0
    check("speech wav", ok, f"{len(wav_response.content)} B, {seconds:.2f}s audio, {wall:.2f}s wall, rtf {rtf:.3f}")

    # 4. mp3 transcode
    response = client.post("/v1/audio/speech", json={"input": TEXT, "response_format": "mp3"})
    check("speech mp3", response.status_code == 200 and len(response.content) > 1000, f"{len(response.content)} B")

    # 4b. transcription round-trip on the WAV generated above
    transcript = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("speech.wav", wav_response.content, "audio/wav")},
    )
    if transcript.status_code == 200:
        payload = transcript.json()
        heard = normalize(payload.get("text", ""))
        expected = normalize(TEXT)
        similarity = difflib.SequenceMatcher(None, heard, expected).ratio()
        check(
            "transcribe wav",
            similarity > 0.8,
            f"{payload.get('language')} {payload.get('duration')}s, similarity {similarity:.2f}",
        )
    else:
        check("transcribe wav", False, f"HTTP {transcript.status_code}: {transcript.text[:120]}")

    srt = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "srt"},
        files={"file": ("speech.wav", wav_response.content, "audio/wav")},
    )
    check("transcribe srt", srt.status_code == 200 and b"-->" in srt.content, srt.text.splitlines()[1] if srt.status_code == 200 else "")

    # 5. built-in voice
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Good evening.", "voice": "US_James", "response_format": "wav"},
    )
    check("built-in voice", response.status_code == 200 and response.content[:4] == b"RIFF", "US_James")

    # 6. stream pcm
    chunks = 0
    payload = b""
    started = time.time()
    first = None
    with client.stream(
        "POST", "/v1/audio/speech", json={"input": TEXT, "stream": True, "response_format": "pcm"}
    ) as stream_response:
        for chunk in stream_response.iter_bytes():
            if not chunk:
                continue
            chunks += 1
            payload += chunk
            if first is None:
                first = time.time() - started
    check(
        "stream pcm",
        chunks > 1 and len(payload) > 10000,
        f"{chunks} chunks, {len(payload)} B, ttft {('%.2fs' % first) if first else 'n/a'}",
    )

    # 7. stream sse
    events = 0
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={"input": "Good evening.", "stream": True, "stream_format": "sse", "response_format": "pcm"},
    ) as stream_response:
        for line in stream_response.iter_lines():
            if line.startswith("data:"):
                events += 1
    check("stream sse", events > 1, f"{events} events")

    # 8. clone upload, then use it
    if args.reference:
        reference = open(args.reference, "rb").read()
        created = client.post(
            "/v1/audio/voices",
            data={"name": args.voice_name, "transcript": "Please leave the lamp beside the window."},
            files={"file": ("reference.wav", reference, "audio/wav")},
        )
        check("voice upload", created.status_code == 201, created.text[:120])

        voices = client.get("/v1/audio/voices").json().get("data", [])
        check("voice listed", any(v["id"] == args.voice_name for v in voices), f"{len(voices)} voices")

        cloned = client.post(
            "/v1/audio/speech",
            json={"input": "Please return the keys before six.", "voice": args.voice_name, "response_format": "wav"},
        )
        check("clone speech", cloned.status_code == 200 and cloned.content[:4] == b"RIFF", f"{len(cloned.content)} B")

        deleted = client.delete(f"/v1/audio/voices/{args.voice_name}")
        check("voice delete", deleted.status_code == 204, "")

    # 9. error shape
    bad = client.post("/v1/audio/speech", json={"input": "", "response_format": "wav"})
    error = bad.json().get("error", {})
    check("openai error shape", bad.status_code == 400 and "message" in error, error.get("message", "")[:60])

    client.close()
    failed = [hit for hit in HITS if not hit[1]]
    print(f"\n{len(HITS) - len(failed)}/{len(HITS)} checks passed")
    if failed:
        print(json.dumps([{"check": n, "detail": d} for n, _, d in failed], indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
