"""Runtime configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """Configuration for the adapter process."""

    # audio.cpp server holding the model warm
    backend_url: str = field(default_factory=lambda: _env("MING30_BACKEND_URL", "http://127.0.0.1:8090"))
    # backend model id for offline generation
    model: str = field(default_factory=lambda: _env("MING30_MODEL", "ming-tts"))
    # backend model id configured with mode=streaming; empty disables true streaming
    stream_model: str = field(default_factory=lambda: _env("MING30_STREAM_MODEL", "ming-tts-stream"))
    # backend model id for transcription (Ming 3.0 Audio Tiny); empty disables the route
    asr_model: str = field(default_factory=lambda: _env("MING30_ASR_MODEL", "ming-asr"))
    # where uploaded reference recordings live (resolved to an absolute path;
    # the audio.cpp backend opens these files itself, from its own cwd)
    voice_dir: Path = field(default_factory=lambda: Path(_env("MING30_VOICE_DIR", "voices")))
    # scratch space for ffmpeg; must be writable by this process
    tmp_dir: Path = field(default_factory=lambda: Path(_env("MING30_TMP_DIR", ".ming30-tmp")))
    # model aliases OpenAI clients may send
    model_aliases: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            a.strip()
            for a in _env("MING30_MODEL_ALIASES", "ming30,ming-3.0-tts,tts-1,tts-1-hd,ming-3.0-tts-0.5b").split(",")
            if a.strip()
        )
    )
    ffmpeg: str = field(default_factory=lambda: _env("MING30_FFMPEG", "ffmpeg"))
    # seconds; a cold model load can take ~40 s, so keep this generous
    backend_timeout: float = field(default_factory=lambda: float(_env("MING30_BACKEND_TIMEOUT", "300")))
    # cap on upload size for a reference recording
    max_upload_bytes: int = field(default_factory=lambda: int(_env("MING30_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024))))

    # voices baked into the 0.5B checkpoint, selectable through the ip template
    builtin_voices: tuple[str, ...] = ("US_Emma", "US_James", "UK_George", "UK_Charlotte")

    @property
    def streaming_enabled(self) -> bool:
        return bool(self.stream_model)

    @property
    def transcription_enabled(self) -> bool:
        return bool(self.asr_model)

    def resolved_voice_dir(self) -> Path:
        return self.voice_dir.expanduser().resolve()

    def resolved_tmp_dir(self) -> Path:
        return self.tmp_dir.expanduser().resolve()

    def ensure_dirs(self) -> None:
        self.resolved_voice_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_tmp_dir().mkdir(parents=True, exist_ok=True)
