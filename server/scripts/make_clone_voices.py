"""Register short clone voices for built-in speakers.

Built-in Ming voices (`US_Emma`, `US_James`, ...) accept only structured controls,
so raw per-turn delivery hints are rejected. Cloning the built-in voice once gives
an equivalent identity that *does* accept `instruct`.

The reference is kept short (~7 s) on purpose: reference conditioning shares the
model's 512-position context (~80 ms per patch), so a long reference leaves little
room for output on long turns.

    python scripts/make_clone_voices.py --base http://127.0.0.1:8070
"""

from __future__ import annotations

import argparse

import httpx

REFERENCE_TEXT = (
    "The garden will remain open until sunset, provided the weather stays clear. "
    "Visitors should leave the lamp beside the window."
)

VOICES = {
    "james_ref": "US_James",
    "emma_ref": "US_Emma",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8070")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=300.0)
    existing = {item["id"] for item in client.get("/v1/audio/voices").json()["data"]}

    for name, builtin in VOICES.items():
        if name in existing and not args.force:
            print(f"{name}: already registered")
            continue
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": REFERENCE_TEXT,
                "voice": builtin,
                "response_format": "wav",
                "language": "en",
            },
        )
        response.raise_for_status()
        created = client.post(
            "/v1/audio/voices",
            data={"name": name, "transcript": REFERENCE_TEXT},
            files={"file": (f"{name}.wav", response.content, "audio/wav")},
        )
        if created.status_code == 409 or created.status_code == 400:
            print(f"{name}: {created.status_code} {created.text[:120]}")
            continue
        created.raise_for_status()
        print(f"{name}: registered from {builtin} ({len(response.content)} B)")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
