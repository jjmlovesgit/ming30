"""Audio helpers: WAV introspection and transcoding through ffmpeg."""

from __future__ import annotations

import asyncio
import shutil
import struct
import uuid
from pathlib import Path

SAMPLE_RATE = 44100
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le

# OpenAI-style response formats this server understands.
SUPPORTED_FORMATS = ("wav", "pcm", "mp3", "opus", "flac", "aac")

_FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "aac": ["-c:a", "aac", "-b:a", "192k", "-f", "adts"],
    "pcm": ["-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1"],
}

CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


class AudioError(RuntimeError):
    """Transcoding failed or the payload is not usable audio."""


def ffmpeg_available(ffmpeg: str) -> bool:
    return shutil.which(ffmpeg) is not None


def wav_info(data: bytes) -> dict:
    """Read the fmt/data chunks of a PCM or float WAV file."""

    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("payload is not a RIFF/WAVE file")

    pos = 12
    fmt = None
    data_size = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        body = pos + 8
        if chunk_id == b"fmt " and chunk_size >= 16:
            audio_format, channels, rate, _byte_rate, _block_align, bits = struct.unpack_from("<HHIIHH", data, pos + 8)
            fmt = {"audio_format": audio_format, "channels": channels, "rate": rate, "bits": bits}
        elif chunk_id == b"data":
            data_size = min(chunk_size, len(data) - body)
            break
        pos = body + chunk_size + (chunk_size & 1)

    if fmt is None or data_size is None:
        raise AudioError("WAV header is missing fmt or data chunks")

    frame_bytes = max(1, fmt["channels"] * (fmt["bits"] // 8))
    frames = data_size // frame_bytes
    return {**fmt, "frames": frames, "duration": frames / fmt["rate"] if fmt["rate"] else 0.0}


async def transcode(data: bytes, target_format: str, ffmpeg: str, workdir: Path) -> bytes:
    """Convert a WAV payload to the requested container.

    Uses temporary files rather than pipes: on Windows, asyncio subprocess pipes
    are anonymous named pipes, which restricted environments refuse to create.
    Files also make ffmpeg's errors easier to read. Unique filenames are used
    instead of ``tempfile`` because its 0o700 directories cannot be written to
    by a restricted token.
    """

    if target_format == "wav":
        return data
    if target_format not in _FFMPEG_ARGS:
        raise AudioError(f"unsupported response_format '{target_format}'")
    if not ffmpeg_available(ffmpeg):
        raise AudioError(f"ffmpeg not found ('{ffmpeg}'); only wav and pcm work without it")

    workdir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    suffix = "ogg" if target_format == "opus" else target_format
    source = workdir / f"in-{token}.wav"
    target = workdir / f"out-{token}.{suffix}"
    try:
        source.write_bytes(data)
        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            *_FFMPEG_ARGS[target_format],
            str(target),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        if process.returncode != 0 or not target.exists():
            raise AudioError(f"ffmpeg failed to encode {target_format} (exit {process.returncode})")
        return target.read_bytes()
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def wav_stream_header() -> bytes:
    """44-byte WAV header with unknown sizes, for streaming delivery."""

    byte_rate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, CHANNELS, SAMPLE_RATE, byte_rate, block_align, BYTES_PER_SAMPLE * 8)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )
