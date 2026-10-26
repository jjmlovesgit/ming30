"""Minimal client examples for the ming30 adapter.

    python examples/client.py --base http://127.0.0.1:8070

Uses httpx so it runs without the OpenAI SDK. The SDK variant is shown in the
docstring below; it works because the adapter speaks the same shape.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

# With the OpenAI SDK:
#
#   from openai import OpenAI
#   client = OpenAI(base_url="http://127.0.0.1:8070/v1", api_key="not-needed")
#   audio = client.audio.speech.create(
#       model="tts-1", voice="US_James", input="Good evening.",
#       response_format="mp3",
#   )
#   audio.write_to_file("speech.mp3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8070")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.base, timeout=300.0)

    print("offline mp3")
    started = time.time()
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "The garden will remain open until sunset, provided the weather stays clear.",
            "voice": "US_James",
            "response_format": "mp3",
        },
    )
    response.raise_for_status()
    path = destination / "speech.mp3"
    path.write_bytes(response.content)
    print(f"  {len(response.content)} bytes, {time.time() - started:.2f}s -> {path}")

    print("streaming pcm")
    started = time.time()
    first = None
    total = 0
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "Streaming from a local Ming 3.0 model.",
            "stream": True,
            "response_format": "pcm",
        },
    ) as stream:
        with open(destination / "stream.pcm", "wb") as handle:
            for chunk in stream.iter_bytes():
                if not chunk:
                    continue
                if first is None:
                    first = time.time() - started
                total += len(chunk)
                handle.write(chunk)
    print(f"  first chunk {first:.2f}s, {total} bytes total -> {destination / 'stream.pcm'}")

    print("voice library")
    for voice in client.get("/v1/audio/voices").json()["data"]:
        print(f"  {voice['id']:<16} {voice['type']}")

    client.close()


if __name__ == "__main__":
    main()
