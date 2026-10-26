"""Voice library: built-in voices plus cloned reference recordings on disk.

Layout under ``voice_dir``::

    voices/
      narrator.wav          # mono/stereo WAV reference recording
      voices.json           # {"narrator": {"transcript": "...", "created_at": "..."}}
      prompt_text           # ``name|transcript`` lines, the audio.cpp library format

``prompt_text`` is kept in sync so the native audio.cpp server can also resolve
these voices by name; the adapter itself always passes an explicit ``voice_ref``
path, which works even when the backend has a stale library cache.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class VoiceError(ValueError):
    """Invalid voice name or payload."""


@dataclass(frozen=True)
class Voice:
    name: str
    kind: str  # "builtin" | "clone"
    transcript: str | None = None
    created_at: str | None = None

    def to_json(self) -> dict:
        payload = {"id": self.name, "name": self.name, "type": self.kind}
        if self.transcript:
            payload["transcript"] = self.transcript
        if self.created_at:
            payload["created_at"] = self.created_at
        return payload


class VoiceStore:
    """File-backed clone library plus the checkpoint's built-in voices."""

    def __init__(self, root: Path, builtins: tuple[str, ...]) -> None:
        self.root = root
        self.builtins = builtins
        self.index_path = root / "voices.json"
        self.prompt_text_path = root / "prompt_text"
        self.root.mkdir(parents=True, exist_ok=True)

    # -- index ---------------------------------------------------------
    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index(self, index: dict[str, dict]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [f"{name}|{meta.get('transcript', '')}" for name, meta in sorted(index.items())]
        self.prompt_text_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # -- queries -------------------------------------------------------
    def list(self) -> list[Voice]:
        voices = [Voice(name=n, kind="builtin") for n in self.builtins]
        for name, meta in sorted(self._read_index().items()):
            if (self.root / f"{name}.wav").exists():
                voices.append(
                    Voice(name=name, kind="clone", transcript=meta.get("transcript"), created_at=meta.get("created_at"))
                )
        return voices

    def get(self, name: str) -> Voice | None:
        for voice in self.list():
            if voice.name == name:
                return voice
        return None

    def wav_path(self, name: str) -> Path | None:
        if not _NAME_RE.match(name):
            return None
        path = self.root / f"{name}.wav"
        return path if path.exists() else None

    def is_builtin(self, name: str) -> bool:
        return name in self.builtins

    # -- mutations -----------------------------------------------------
    def add(self, name: str, wav_bytes: bytes, transcript: str | None) -> Voice:
        if not _NAME_RE.match(name):
            raise VoiceError("voice name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
        if self.is_builtin(name):
            raise VoiceError(f"'{name}' is a built-in voice and cannot be replaced")
        if len(wav_bytes) < 1024:
            raise VoiceError("reference recording is too small to be a WAV file")
        if wav_bytes[:4] != b"RIFF":
            raise VoiceError("reference recording must be a WAV file (RIFF)")

        path = self.root / f"{name}.wav"
        path.write_bytes(wav_bytes)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        index = self._read_index()
        index[name] = {"transcript": (transcript or "").strip(), "created_at": created}
        self._write_index(index)
        return Voice(name=name, kind="clone", transcript=index[name]["transcript"], created_at=created)

    def delete(self, name: str) -> bool:
        if self.is_builtin(name):
            raise VoiceError(f"'{name}' is a built-in voice and cannot be deleted")
        index = self._read_index()
        existed = name in index or (self.root / f"{name}.wav").exists()
        if not existed:
            return False
        index.pop(name, None)
        (self.root / f"{name}.wav").unlink(missing_ok=True)
        self._write_index(index)
        return True

    def reset(self) -> None:
        """Remove every cloned voice (used by tests)."""
        for wav in self.root.glob("*.wav"):
            wav.unlink(missing_ok=True)
        self.index_path.unlink(missing_ok=True)
        self.prompt_text_path.unlink(missing_ok=True)


def copy_into(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
