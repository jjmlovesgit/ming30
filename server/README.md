# ming30 server

OpenAI-compatible HTTP surface for **Ming 3.0 TTS** on top of the
[audio.cpp](https://github.com/0xShug0/audio.cpp) native runtime.

This is an *adapter*, not an inference engine. `audiocpp_server` holds the GGUF
model warm and does the generation; this process adds the things an OpenAI-shaped
client expects and that the native server does not do:

| Gap in the native server | What this adapter does |
|---|---|
| `response_format: mp3` is silently returned as WAV | transcodes with ffmpeg (mp3, opus, flac, aac, pcm) |
| reference audio must be a path on the *server's* disk | `POST /v1/audio/voices` accepts an upload and passes an absolute path |
| no voice library API | list / add / fetch / delete cloned voices, plus the checkpoint's built-ins |
| clients send `model: "tts-1"` | model aliases mapped to the configured backend id |
| voice + generation options are model-specific | OpenAI fields (`input`, `voice`, `instructions`, `speed`) translated to Ming options |
| errors are engine-shaped | OpenAI `{"error": {...}}` envelope |

```
OpenAI client ──▶ ming30 adapter (:8070) ──▶ audiocpp_server (:8090) ──▶ GGUF
                        │                         (model stays warm)
                        ├─ voices/ (uploaded reference WAVs + transcripts)
                        └─ .ming30-tmp/ (ffmpeg scratch)
```

## Quick start

```powershell
# 1. runtime + model (see ../RUNNING-ming30.md)
#    audio.cpp-0.9.3-cuda12.4\  and  models\ming3-tts\ming-3.0-tts-0.5b-q8_0.gguf

# 2. both processes, Ctrl+C stops them
cd server
.\run.ps1

# 3. verify
python scripts\smoke_test.py --reference ..\out\en\en_2.wav
```

`run.ps1` starts `audiocpp_server` with `config/dev.server.json` (an offline entry
`ming-tts` and a streaming entry `ming-tts-stream`), waits for `/health`, then
starts the adapter. The CUDA toolkit `bin` is prepended to `PATH` because
`ggml-cuda.dll` needs `cublas64_12` / `cufft64_11` from the 12.9 toolkit.

Adapter only, backend already running: `.\run.ps1 -AdapterOnly`.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | adapter + backend status, ffmpeg availability, voice count |
| `GET` | `/v1/models` | configured id plus aliases |
| `GET` | `/v1/audio/voices` | built-in voices and cloned voices |
| `POST` | `/v1/audio/voices` | multipart `name`, `file` (WAV), `transcript` |
| `GET` | `/v1/audio/voices/{name}/audio` | download the stored reference |
| `DELETE` | `/v1/audio/voices/{name}` | remove a cloned voice |
| `POST` | `/v1/audio/speech` | the main route; JSON in, audio out |

`POST /v1/audio/speech` accepts:

```json
{
  "model": "tts-1",
  "input": "The garden will remain open until sunset.",
  "voice": "US_James",
  "response_format": "mp3",
  "instructions": "Speak slowly and warmly.",
  "language": "en",
  "seed": 1895,
  "options": { "text_chunk_size": 300 },
  "stream": false
}
```

`voice` resolves in this order: built-in (`US_Emma`, `US_James`, `UK_George`,
`UK_Charlotte` → `ip` template), cloned voice from the library (passes
`voice_ref` + `reference_text`), otherwise passed through as a backend-native id.

Streaming: `stream: true` returns raw PCM (`response_format: pcm`) or WAV
(`response_format: wav`) as chunked bytes, which is what OpenAI SDK
`with_streaming_response` consumers read. `stream_format: "sse"` returns the
engine's event stream (`speech.audio.delta` / `speech.audio.done`). Streaming
requires the backend entry configured with `mode: "streaming"`; set
`MING30_STREAM_MODEL=""` to disable and fall back to buffered responses, which
are marked `x-ming30-streaming: buffered`. `mp3`/`opus` cannot be streamed —
request them without `stream`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MING30_BACKEND_URL` | `http://127.0.0.1:8090` | audio.cpp server |
| `MING30_MODEL` | `ming-tts` | backend model id for offline generation |
| `MING30_STREAM_MODEL` | `ming-tts-stream` | backend model id configured `mode: streaming`; empty disables |
| `MING30_MODEL_ALIASES` | `ming30,ming-3.0-tts,tts-1,tts-1-hd,ming-3.0-tts-0.5b` | names clients may send |
| `MING30_VOICE_DIR` | `voices` | cloned reference WAVs (resolved absolute) |
| `MING30_TMP_DIR` | `.ming30-tmp` | ffmpeg scratch space |
| `MING30_FFMPEG` | `ffmpeg` | binary or absolute path |
| `MING30_BACKEND_TIMEOUT` | `300` | seconds; a cold load can take ~40 s |
| `MING30_PORT` / `MING30_HOST` | `8070` / `127.0.0.1` | adapter bind |
| `MING30_CORS` | unset | set to anything to enable permissive CORS (local demos only) |

## Measured on an RTX 5090

Backend and adapter on the same machine, Q8_0 checkpoint, warm model:

| | value |
|---|---|
| Offline WAV, 4.50 s of speech | 1.08–1.17 s wall through the adapter, **RTF ≈ 0.24** |
| Engine-side warm RTF (6.40 s clip) | 0.206 |
| Streaming PCM, time to first chunk | **0.13–0.32 s** |
| Streaming, 4.50 s of audio | 15 chunks |
| MP3 transcode overhead | included above; ffmpeg writes ~110 KB |
| Model load (once per backend process) | ~35–40 s |

The adapter adds roughly 0.05–0.1 s on top of the engine's own RTF; the
dominant term is inference, not HTTP.

## Limitations

- **No auth, no rate limiting.** Bind to `127.0.0.1`, or put it behind your own
  proxy. The backend accepts server-local paths, so do not expose it to an
  untrusted network.
- **One inference at a time per backend model.** The native server serialises
  requests and returns `503 server_busy` past `busy_timeout_ms`; the adapter maps
  that to a 502 with the backend's message.
- **No `/v1/audio/transcriptions`.** Ming 3.0 *Audio* Tiny can transcribe, but it
  is a different checkpoint and would need its own backend entry.
- **No voice-design route yet.** `voice_design` is reachable today only by sending
  `options: {"instruct": "...", "template_name": "voice_design"}`.
- Streaming is PCM/WAV only; mp3 streaming would need a persistent encoder.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `0xC0000135` in backend logs | CUDA toolkit `bin` missing from `PATH` |
| `502 ... could not open WAV input` | reference path not absolute, or the backend cannot read the file |
| `500 ffmpeg failed to encode mp3` | ffmpeg missing `libmp3lame`; check `ffmpeg -encoders` |
| first request takes ~40 s | backend model load; it is lazy by default |
| `503` from `/health` | adapter is up, `audiocpp_server` is not |
