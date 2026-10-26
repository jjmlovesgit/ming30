"""Client for the audio.cpp server that holds the model warm."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class BackendError(RuntimeError):
    """The audio.cpp backend rejected or failed a request."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Backend:
    """Thin async wrapper over the audio.cpp HTTP API."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- metadata ------------------------------------------------------
    async def health(self) -> dict:
        response = await self._client.get("/health")
        return _json_or_error(response)

    async def models(self) -> dict:
        response = await self._client.get("/v1/models")
        return _json_or_error(response)

    async def voices(self, model: str) -> dict:
        response = await self._client.get("/v1/audio/voices", params={"model": model})
        return _json_or_error(response)

    # -- speech --------------------------------------------------------
    async def speech(self, payload: dict) -> tuple[bytes, str]:
        """Return (audio bytes, content type) for one offline request."""

        response = await self._client.post("/v1/audio/speech", json=payload)
        if response.status_code >= 400:
            raise BackendError(response.status_code, _error_message(response))
        return response.content, response.headers.get("content-type", "audio/wav")

    async def speech_events(self, payload: dict) -> AsyncIterator[dict]:
        """Yield decoded SSE events from a streaming request.

        The backend emits ``speech.audio.delta`` events carrying base64 PCM and
        terminates with ``speech.audio.done`` followed by ``[DONE]``.
        """

        async with self._client.stream("POST", "/v1/audio/speech", json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise BackendError(response.status_code, body[:400] or response.reason_phrase)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        return
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    # -- transcription -------------------------------------------------
    async def transcribe(
        self,
        *,
        model: str,
        audio: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> dict:
        """POST a WAV payload to the backend's transcription route."""

        data = {"model": model}
        if language:
            data["language"] = language
        response = await self._client.post(
            "/v1/audio/transcriptions",
            data=data,
            files={"file": (filename, audio, "audio/wav")},
        )
        if response.status_code >= 400:
            raise BackendError(response.status_code, _error_message(response))
        return _json_or_error(response)


def _json_or_error(response: httpx.Response) -> dict:
    if response.status_code >= 400:
        raise BackendError(response.status_code, _error_message(response))
    try:
        return response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise BackendError(502, f"backend returned non-JSON: {exc}") from exc


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text[:400] or response.reason_phrase
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if payload.get("message"):
            return str(payload["message"])
    return json.dumps(payload)[:400]
