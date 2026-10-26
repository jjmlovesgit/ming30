# ming30

Local [Ming 3.0](https://github.com/inclusionAI/Ming-3.0-tts) speech on Windows through
[audio.cpp](https://github.com/0xShug0/audio.cpp) — native GGUF runtime, no PyTorch,
no cu128 wheel juggling, no pynini dead end.

**Status:** working and measured on this machine (RTX 5090 32 GB, Windows, CUDA 12.9
toolkit, audio.cpp v0.9.3 `windows-x64-cuda12.4`).

## What runs

| Checkpoint | File | Size | Tasks |
|---|---|---|---|
| Ming-3.0-tts-0.5b (Q8_0) | `ming-3.0-tts-0.5b-q8_0.gguf` | 1.66 GB | tts, clone, voice design, **streaming** |
| Ming-3.0-audio-tiny (Q4_K) | `ming-3.0-audio-tiny-q4_k.gguf` | 9.28 GB | tts, clone, asr, speech-to-speech |

Both are 44.1 kHz mono. Sources:
[`audio-cpp/Ming-3.0-TTS-0.5B-GGUF`](https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF),
[`audio-cpp/Ming-3.0-Audio-Tiny-GGUF`](https://huggingface.co/audio-cpp/Ming-3.0-Audio-Tiny-GGUF).
MIT, following the upstream checkpoints.

## Quick start

The CUDA 12.9 toolkit's `bin` must be on `PATH`, otherwise `audiocpp_cli` exits with
`0xC0000135` (missing `cufft64_11.dll` / `cublas64_12.dll`).

```powershell
# runtime: https://github.com/0xShug0/audio.cpp/releases  -> audio-v0.9.3-bin-windows-x64-cuda12.4.zip
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:PATH"
$cli = ".\audio.cpp-0.9.3-cuda12.4\audiocpp_cli.exe"
$tts = ".\models\ming3-tts\ming-3.0-tts-0.5b-q8_0.gguf"

# plain TTS
& $cli --family ming3_tts --task tts --backend cuda --model $tts `
  --text "Good evening. The harbor is quiet tonight." `
  --request-option language=en --out out\speech.wav --metrics

# built-in voice (ip template only: US_Emma, US_James, UK_George, UK_Charlotte)
& $cli --family ming3_tts --task tts --backend cuda --model $tts --text "..." `
  --request-option language=en --request-option template_name=ip `
  --request-option voice_id=US_James --out out\ip.wav

# voice design
& $cli --family ming3_tts --task vdes --backend cuda --model $tts --text "..." `
  --request-option language=en `
  --request-option instruct="A calm adult man with a clear, warm voice, speaking slowly." `
  --out out\vdes.wav

# clone
& $cli --family ming3_tts --task clon --backend cuda --model $tts `
  --voice-ref ref.wav --request-option reference_text="Transcript of ref.wav." `
  --text "..." --request-option language=en --out out\clone.wav
```

Multiple requests in one loaded session (`--metrics` gives per-request `rtf`):

```powershell
& $cli --family ming3_tts --task tts --backend cuda --model $tts `
  --request-sequence reqs_tts05.json --out-dir out\tts05 --metrics
```

Per-request options (`voice_id`, `language`, `template_name`) go **inside** each JSON item;
`--request-option` alongside `--request-sequence` is rejected.

## Measured

| | tts-0.5b Q8_0 | audio-tiny Q4_K |
|---|---|---|
| Cold start, one-shot | 35.7–41.8 s | 28.6–31 s |
| Warm RTF | 0.212–0.229 (**~4.4× realtime**) | 0.212–0.235 (~4.3–4.7×) |
| Peak VRAM | **3.5 GB** | 4.2 GB |
| Host RSS / commit | 2.0 / 3.0 GB | 2.9 / 3.9–8.8 GB |

Cold start is dominated by fixed per-process setup, not model size — the 1.66 GB model
takes longer to become ready than the 9.28 GB one. Run `audiocpp_server` for real
traffic so that cost leaves the request path.

Quality spot-check: every generated WAV (English plain, English `US_James`, Chinese ×2,
clone, voice design, streaming) transcribed back **exactly** through audio-tiny's ASR.
Transcripts do not separate the two checkpoints at this length — judge naturalness by ear.

## Server

`server/` is an OpenAI-compatible adapter over the audio.cpp runtime, covering
both directions of speech: `/v1/audio/speech` (PCM/WAV streaming, mp3/opus/flac
transcoding, voice library with uploads) and `/v1/audio/transcriptions`
(the Audio Tiny checkpoint, behind `whisper-1` in clients). Details and the
env-var reference: [server/README.md](server/README.md).

```powershell
cd server
.\run.cmd                                                # engine + adapter, no Docker
python scripts\smoke_test.py --reference ..\out\en\en_2.wav
```

`run.cmd` exists because the default Windows execution policy blocks unsigned
`.ps1` files; `scripts\preflight.py` checks the engine binary, CUDA DLLs, GPU,
GGUFs, python deps and ports before you start. The identical stack is also
described for containers in `compose.yaml`, but that path is untested here —
the native one is what was verified.

Measured through the adapter on the RTX 5090, warm Q8_0 model: RTF ≈ 0.24
offline, streaming time-to-first-chunk 0.13–0.32 s, transcription ~0.1 s for
4.5 s of audio, 14/14 smoke checks. The compose stack itself is untested here
(Docker is not installed on this machine); the Windows engine path is what was
verified.

## Gotchas

- **LM Studio cannot load these GGUFs.** `ming3_audio` / `ming3_tts` are registered only
  in audio.cpp; a GGUF container is not a portable architecture.
- No FST text normalizer in the native frontend: spell out numbers and abbreviations.
- `--metrics` is rejected in `--mode streaming`.
- Long clones can drop words inside a fragment; `text_chunk_size` defaults are tuned for it.
- The `cuda13.3` release needs `cublas64_13.dll` + `cufft64_12.dll` (CUDA 13 sonames);
  the `cuda12.4` package matches a 12.9 toolkit and does run on sm_120.

## Files

- `RUNNING-ming30.md` — full runbook: both runtime routes, every command, all caveats.
- `server/` — OpenAI-compatible adapter (FastAPI), smoke test, Dockerfile, client example.
- `reqs_tts05.json`, `reqs_en2.json`, `reqs_zh.json` — session request examples.
- `out/` — verification WAVs (gitignored).

Upstream: [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) ·
[runtime](https://github.com/0xShug0/audio.cpp). Model weights: MIT.
