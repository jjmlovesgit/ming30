"""OpenAI-compatible FastAPI surface over the audio.cpp Ming 3.0 backend."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .audio import CONTENT_TYPES, SUPPORTED_FORMATS, AudioError, ffmpeg_available, transcode, wav_info, wav_stream_header
from .backend import Backend, BackendError
from .settings import Settings
from .transcripts import language_code, split_language_prefix, to_srt, to_vtt
from .voices import VoiceError, VoiceStore


class SpeechRequest(BaseModel):
    """OpenAI speech request plus the Ming-specific extras."""

    model: str | None = None
    input: str | None = None
    text: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    stream: bool = False
    stream_format: str | None = None
    instructions: str | None = None
    instruct: str | None = None
    language: str | None = None
    seed: int | None = None
    reference_text: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


def _error(message: str, *, status: int = 400, code: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": code, "code": status}}, status_code=status)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        app.state.settings = settings
        app.state.voices = VoiceStore(settings.resolved_voice_dir(), settings.builtin_voices)
        app.state.backend = Backend(settings.backend_url, settings.backend_timeout)
        try:
            yield
        finally:
            await app.state.backend.aclose()

    app = FastAPI(title="ming30", version=__version__, lifespan=lifespan)
    if os.environ.get("MING30_CORS"):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- error mapping -------------------------------------------------
    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError):
        return _error(exc.message, status=502 if exc.status >= 500 else exc.status, code="backend_error")

    @app.exception_handler(VoiceError)
    async def _voice_error(_: Request, exc: VoiceError):
        return _error(str(exc))

    @app.exception_handler(AudioError)
    async def _audio_error(_: Request, exc: AudioError):
        return _error(str(exc), status=500, code="audio_error")

    # -- metadata ------------------------------------------------------
    @app.get("/")
    async def root():
        return {
            "object": "ming30",
            "version": __version__,
            "backend": settings.backend_url,
            "model": settings.model,
            "stream_model": settings.stream_model or None,
            "endpoints": [
                "GET /health",
                "GET /v1/models",
                "GET /v1/audio/voices",
                "POST /v1/audio/voices",
                "DELETE /v1/audio/voices/{name}",
                "POST /v1/audio/speech",
            ],
        }

    @app.get("/health")
    async def health():
        state = {"status": "ok", "version": __version__, "model": settings.model, "ffmpeg": ffmpeg_available(settings.ffmpeg)}
        try:
            state["backend"] = await app.state.backend.health()
        except Exception as exc:  # noqa: BLE001 - report any transport failure
            return JSONResponse(
                {**state, "status": "degraded", "backend_error": str(exc)},
                status_code=503,
            )
        state["voices"] = len(app.state.voices.list())
        state["transcription"] = settings.asr_model or None
        return state

    @app.get("/v1/models")
    async def models():
        data = []
        try:
            listed = (await app.state.backend.models()).get("data", [])
        except BackendError:
            listed = []
        loaded = {entry.get("id"): entry.get("loaded") for entry in listed}
        for alias in (settings.model, *settings.model_aliases):
            unique = alias not in {entry["id"] for entry in data}
            if unique:
                data.append(
                    {
                        "id": alias,
                        "object": "model",
                        "owned_by": "ming30",
                        "backend_model": settings.model,
                        "loaded": loaded.get(settings.model),
                    }
                )
        if settings.transcription_enabled:
            data.append(
                {
                    "id": "whisper-1",
                    "object": "model",
                    "owned_by": "ming30",
                    "backend_model": settings.asr_model,
                    "loaded": loaded.get(settings.asr_model),
                }
            )
        return {"object": "list", "data": data}

    # -- transcription -------------------------------------------------
    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
    ):
        if not settings.transcription_enabled:
            return _error(
                "transcription is disabled; set MING30_ASR_MODEL to an audio.cpp model id "
                "(family ming3_audio, task asr)",
                status=503,
                code="transcription_unavailable",
            )

        payload = await file.read()
        if len(payload) > settings.max_upload_bytes:
            return _error(f"audio exceeds {settings.max_upload_bytes} bytes", status=413)
        if payload[:4] != b"RIFF":
            return _error("audio must be a WAV file (RIFF)")

        result = await app.state.backend.transcribe(
            model=settings.asr_model,
            audio=payload,
            filename=file.filename or "audio.wav",
            language=language,
        )
        detected, text = split_language_prefix(str(result.get("text", "")))
        timing = result.get("timing") or {}
        duration = float(timing.get("audio_duration_ms") or 0) / 1000.0
        code = language_code(detected)
        fmt = (response_format or "json").lower()

        if fmt == "text":
            return Response(text, media_type="text/plain; charset=utf-8")
        if fmt == "srt":
            return Response(to_srt(text, duration), media_type="application/x-subrip; charset=utf-8")
        if fmt == "vtt":
            return Response(to_vtt(text, duration), media_type="text/vtt; charset=utf-8")
        if fmt == "verbose_json":
            return {
                "task": "transcribe",
                "language": code,
                "duration": round(duration, 3),
                "text": text,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": round(max(duration, 0.001), 3),
                        "text": text,
                    }
                ],
            }
        return {"text": text}

    # -- voices --------------------------------------------------------
    @app.get("/v1/audio/voices")
    async def list_voices():
        items = []
        for voice in app.state.voices.list():
            item = voice.to_json()
            # Older audio.cpp-style clients read `voice_id` from a `voices` array;
            # newer OpenAI-style clients read `id` from `data`. Serve both.
            item["voice_id"] = item["id"]
            items.append(item)
        return {"object": "list", "data": items, "voices": items}

    @app.post("/v1/audio/voices", status_code=201)
    async def create_voice(
        name: str = Form(...),
        file: UploadFile = File(...),
        transcript: str = Form(""),
    ):
        payload = await file.read()
        if len(payload) > settings.max_upload_bytes:
            return _error(f"reference recording exceeds {settings.max_upload_bytes} bytes", status=413)
        voice = app.state.voices.add(name, payload, transcript)
        return voice.to_json()

    @app.get("/v1/audio/voices/{name}/audio")
    async def voice_audio(name: str):
        path = app.state.voices.wav_path(name)
        if path is None:
            return _error(f"unknown cloned voice '{name}'", status=404, code="not_found")
        return Response(path.read_bytes(), media_type="audio/wav")

    @app.delete("/v1/audio/voices/{name}", status_code=204)
    async def delete_voice(name: str):
        if not app.state.voices.delete(name):
            return _error(f"unknown cloned voice '{name}'", status=404, code="not_found")
        return Response(status_code=204)

    # -- speech --------------------------------------------------------
    @app.post("/v1/audio/speech")
    async def speech(body: SpeechRequest):
        text = (body.input or body.text or "").strip()
        if not text:
            return _error("'input' is required")

        audio_format = (body.response_format or "wav").lower()
        if audio_format not in SUPPORTED_FORMATS:
            return _error(f"unsupported response_format '{audio_format}'; use one of {', '.join(SUPPORTED_FORMATS)}")

        payload: dict[str, Any] = {"input": text}
        options = {key: value for key, value in body.options.items() if value is not None}
        instruction = body.instructions or body.instruct
        if instruction:
            options["instruct"] = instruction
        if body.language:
            options["language"] = body.language
        if body.seed is not None:
            options["seed"] = body.seed
        if options:
            payload["options"] = options

        voice = (body.voice or "").strip()
        if voice:
            if app.state.voices.is_builtin(voice):
                if instruction:
                    return _error(
                        f"'{voice}' is a built-in voice and uses structured controls only; "
                        "raw instructions are rejected by the engine. Drop 'instructions', "
                        "or use a cloned voice (post a reference to /v1/audio/voices) which "
                        "accepts instructions.",
                    )
                options["template_name"] = "ip"
                options["voice_id"] = voice
                payload["options"] = options
            elif (path := app.state.voices.wav_path(voice)) is not None:
                stored = app.state.voices.get(voice)
                payload["voice_ref"] = path.as_posix()
                reference_text = body.reference_text or (stored.transcript if stored else "")
                if reference_text:
                    payload["reference_text"] = reference_text
            else:
                payload["voice"] = voice

        wants_stream = body.stream or body.stream_format is not None
        if not wants_stream:
            return await _offline(app, settings, payload, audio_format)

        if audio_format not in ("pcm", "wav"):
            return _error(
                f"streaming supports response_format 'pcm' or 'wav', not '{audio_format}'; "
                "request without stream=true for mp3/opus/flac",
            )
        if not settings.streaming_enabled:
            # No streaming session configured: chunk the finished file instead of
            # pretending to be incremental.
            response = await _offline(app, settings, payload, audio_format)
            response.headers["x-ming30-streaming"] = "buffered"
            return response
        return _streaming(app, settings, payload, audio_format, body.stream_format)

    return app


async def _offline(app: FastAPI, settings: Settings, payload: dict, audio_format: str):
    request = {"model": settings.model, **payload, "response_format": "wav"}
    try:
        audio, _content_type = await app.state.backend.speech(request)
        info = wav_info(audio)
    except BackendError as exc:
        raise exc
    except AudioError:
        info = {}

    body = await transcode(audio, audio_format, settings.ffmpeg, settings.resolved_tmp_dir())
    headers = {}
    if info:
        headers["x-ming30-audio-seconds"] = f"{info['duration']:.3f}"
        headers["x-ming30-sample-rate"] = str(info["rate"])
    return Response(body, media_type=CONTENT_TYPES[audio_format], headers=headers)


def _streaming(app: FastAPI, settings: Settings, payload: dict, audio_format: str, stream_format: str | None):
    request = {
        "model": settings.stream_model,
        **payload,
        "stream": True,
        "response_format": "pcm",
        "stream_format": "sse",
    }

    if stream_format == "sse":

        async def sse():
            yield f"data: {json.dumps({'type': 'speech.start', 'model': settings.stream_model})}\n\n"
            try:
                async for event in app.state.backend.speech_events(request):
                    if event.get("type") == "speech.audio.delta" and event.get("audio"):
                        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                    elif event.get("type") == "speech.audio.done":
                        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            except BackendError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': exc.message})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    async def audio_stream():
        if audio_format == "wav":
            yield wav_stream_header()
        try:
            async for event in app.state.backend.speech_events(request):
                if event.get("type") == "speech.audio.delta" and event.get("audio"):
                    yield base64.b64decode(event["audio"])
        except BackendError:
            # Headers are already sent; ending the stream is all that is left.
            return

    return StreamingResponse(
        audio_stream(),
        media_type=CONTENT_TYPES[audio_format],
        headers={"cache-control": "no-cache", "x-ming30-streaming": "pcm"},
    )
