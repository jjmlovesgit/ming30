"""Transcript formatting for the OpenAI transcription surface.

The Ming backend returns ``"<Language>\\t<text>"``; OpenAI's transcription
object splits that into ``language`` plus ``text``.
"""

from __future__ import annotations


def split_language_prefix(raw: str) -> tuple[str | None, str]:
    """Split ``"English\\tthe garden..."`` into ``("English", "the garden...")``."""

    if "\t" in raw:
        head, _, rest = raw.partition("\t")
        head = head.strip()
        # Language labels are short alphabetic names; anything else is content.
        if head and len(head) <= 24 and head.replace("_", "").replace("-", "").isalpha():
            return head, rest.strip()
    return None, raw.strip()


def _timestamp(seconds: float, *, comma: bool) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_srt(text: str, duration: float) -> str:
    start = _timestamp(0.0, comma=True)
    end = _timestamp(max(duration, 0.001), comma=True)
    return f"1\n{start} --> {end}\n{text}\n"


def to_vtt(text: str, duration: float) -> str:
    start = _timestamp(0.0, comma=False)
    end = _timestamp(max(duration, 0.001), comma=False)
    return f"WEBVTT\n\n{start} --> {end}\n{text}\n"


def language_code(label: str | None) -> str | None:
    """Map the model's language label to an ISO-ish code when it is a known name."""

    if not label:
        return None
    known = {
        "chinese": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "cantonese": "yue",
    }
    lowered = label.strip().lower()
    return known.get(lowered, lowered)
