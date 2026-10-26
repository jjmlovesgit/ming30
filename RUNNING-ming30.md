# Running Ming 3.0 TTS / Audio on this machine

Machine: Windows, RTX 5090 32 GB (Blackwell, sm_120), driver 610.88 (CUDA UMD 13.3),
CUDA toolkit **12.9** installed, 162 GB free, Python 3.10–3.13 available.

There are two independent ways to run Ming 3.0 speech here. They are different models,
not two builds of one model.

| | Route A — audio.cpp GGUF | Route B — official PyTorch repo |
|---|---|---|
| Models | `Ming-3.0-audio-tiny` (omni: ASR + TTS + clone + s2s), `Ming-3.0-tts-0.5b` | `Ming-3.0-tts-0.5b` only |
| Runtime | native C++ + CUDA, no Python | torch 2.7.1+cu128, transformers 4.46.3 |
| Works here now | **yes, verified below** | needs a venv + cu128 wheels, no text normalization |
| Text normalization | none (FST normalizer not ported) | pynini missing on Windows → also off |
| Streaming / server | yes, OpenAI-compatible `/v1/audio/speech` | no server in repo |

## Route A — verified working setup

Runtime: `audio.cpp` v0.9.3, **windows-x64-cuda12.4** package, unpacked at
`C:\Projects\ming30\audio.cpp-0.9.3-cuda12.4`. The `cuda13.3` package does **not** start
as-is: it needs `cublas64_13.dll` + `cufft64_12.dll` (CUDA 13 sonames), and only a CUDA
12.9 toolkit is installed. The cuda12.4 package imports `cublas64_12` / `cufft64_11` /
`cudart64_12`, which the 12.9 toolkit already provides — and its kernels do run on
sm_120 (verified: compute capability 12.0, CUDA graphs warm up, output is correct).

CUDA 12.9's `bin` must be on `PATH` for the process:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:PATH"
$cli  = "C:\Projects\ming30\audio.cpp-0.9.3-cuda12.4\audiocpp_cli.exe"
$gguf = "C:\Users\Jim\.lmstudio\models\audio-cpp\Ming-3.0-Audio-Tiny-GGUF\ming-3.0-audio-tiny-q4_k.gguf"
```

Named-voice TTS:

```powershell
& $cli --family ming3_audio --task tts --backend cuda --model $gguf `
  --text "The sun warms the earth, and a soft breeze moves through the garden." `
  --request-option voice_id=US_James --request-option language=en `
  --out out\tts_en.wav --metrics
```

Clone (reference + transcript):

```powershell
& $cli --family ming3_audio --task clon --backend cuda --model $gguf `
  --voice-ref out\en\en_2.wav `
  --request-option reference_text="Please leave the lamp beside the window." `
  --text "The reading room closes at six, so please return the keys before then." `
  --request-option language=en --out out\clone_en.wav --metrics
```

ASR / round-trip check:

```powershell
& $cli --family ming3_audio --task asr --backend cuda --model $gguf --audio out\tts_en.wav --metrics
```

Multiple requests in one loaded session (`--request-option` is **not** accepted with
`--request-sequence`; put request options inside each item):

```json
{"requests":[
 {"id":"en_1","text":"The sun warms the earth, and a soft breeze moves through the garden.","voice_id":"US_James","language":"en"},
 {"id":"en_2","text":"Please leave the lamp beside the window.","voice_id":"US_James","language":"en"}
]}
```

```powershell
& $cli --family ming3_audio --task tts --backend cuda --model $gguf `
  --request-sequence reqs_en2.json --out-dir out\en --metrics
```

### Measured on this box (Q4_K, audio-tiny)

| | value |
|---|---|
| Cold start (process load + first request) | 28.6–31 s per process |
| Warm request, RTF | 0.222–0.235 (**~4.3× realtime**) |
| e.g. en_3 | 0.91 s for 4.10 s audio |
| Peak VRAM | **4.2 GB** (32 GB card, huge headroom) |
| Host RSS / commit | 2.9 GB / 3.9–8.8 GB |
| Output | mono 44.1 kHz |
| Transcript check | EN and ZH both round-trip exactly through the model's own ASR |

Q4_K only quantizes the Thinker expert banks, so TTS-side memory does not shrink much
from BF16; the win here is disk/host size (9.3 GB vs 18 GB). Quality on these samples
matched, but the model card warns Q4 answers can differ from BF16.

## Route A2 — the dedicated TTS model as GGUF (verified, recommended)

`audio-cpp` also ships Ming-3.0-tts-0.5b as a GGUF family (`ming3_tts`, status
`supported`, tasks `tts`/`clone`/`design`, offline **and streaming**), from
`audio-cpp/Ming-3.0-TTS-0.5B-GGUF`. Downloaded Q8_0 (1.66 GB) to
`models\ming3-tts\ming-3.0-tts-0.5b-q8_0.gguf`. Same runtime, same `PATH` requirement,
no PyTorch and no pynini.

```powershell
& $cli --family ming3_tts --task tts --backend cuda --model models\ming3-tts\ming-3.0-tts-0.5b-q8_0.gguf `
  --text "Good evening. The harbor is quiet tonight." `
  --request-option language=en --out out\speech.wav --metrics
```

Built-in voices come only from the `ip` template, and only four English ones exist:

```powershell
& $cli --family ming3_tts --task tts --backend cuda --model <gguf> --text "..." `
  --request-option language=en --request-option template_name=ip --request-option voice_id=US_James `
  --out out\ip.wav --metrics
```

Voice design (`vdes`), which the audio-tiny route cannot do:

```powershell
& $cli --family ming3_tts --task vdes --backend cuda --model <gguf> --text "..." `
  --request-option language=en `
  --request-option instruct="A calm adult man with a clear, warm voice, speaking slowly." `
  --out out\vdes.wav --metrics
```

Streaming (`--metrics` is rejected in streaming mode; time the process yourself):

```powershell
& $cli --family ming3_tts --task tts --mode streaming --backend cuda --model <gguf> --text "..." `
  --request-option language=en --out out\stream.wav
```

Request options for this family: `template_name` (tts / multilingual / basic / emotion /
dialect / ip / voice_design / freeform), `language` (zh en ja ko de fr es), `instruct`,
`speed` / `pitch` / `volume` (basic), `emotion`, `dialect`, `voice_id` (ip only:
US_Emma, US_James, UK_George, UK_Charlotte), `reference_text`, `seed` (default 1895),
`guidance_scale` (2), `sigma` (0.25), `max_tokens` (1000, bounded by a 512-position
context), `text_chunk_size`. ~80 ms of audio per acoustic patch.

### Head-to-head measured on this box

Identical sentences, same GPU, same runtime, both driven through one long-lived session
to separate load from inference.

| | audio-tiny Q4_K (9.28 GB) | **tts-0.5b Q8_0 (1.66 GB)** |
|---|---|---|
| Cold start, one-shot | 28.6–31 s | **35.7–41.8 s** (tts 39.2, clone 39.3, vdes 35.7, streaming 41.8) |
| Warm RTF | 0.212–0.235 (~4.3–4.7× RT) | **0.212–0.229 (~4.4–4.7× RT)** |
| Warm example | 0.88 s → 3.90 s audio | 0.76 s → 3.60 s audio |
| Peak VRAM | 4.2 GB | **3.5 GB** |
| Host RSS / commit | 2.9 GB / 3.9–8.8 GB | **2.0 GB / 3.0 GB** |
| Output | 44.1 kHz mono | 44.1 kHz mono |
| Tasks | asr, s2s, tts, clon | tts, clon, **vdes**, **streaming** |
| Built-in voices | ~200, incl. Chinese, default 灵小柔 | 4 English (`ip` template only) |
| Languages | zh, en | zh en ja ko de fr es |

Both are near word-perfect on these samples: the seven 0.5B WAVs (English named-voice,
plain TTS, Chinese ×2, clone, voice-design, streaming) and the audio-tiny WAVs were all
transcribed back exactly by audio-tiny's ASR, modulo punctuation. Transcript checks
therefore do **not** separate them — naturalness and speaker similarity need ears.

Cold start is *not* explained by model size: the 1.66 GB model takes longer to become
ready than the 9.28 GB one, which points at fixed per-process setup (CUDA graph capture,
VAE init) rather than disk. One long-lived `audiocpp_server` instance removes it from the
request path entirely.

**Recommendation:** use `ming3_tts` (0.5b GGUF) for speech synthesis — smaller, less VRAM,
streaming, voice design, and it is the model the benchmarks describe. Keep the audio-tiny
GGUF you already have for ASR / speech-to-speech / its ~200 built-in voices.

### Server / WebUI

`audiocpp_server.exe --config server.json` exposes `/v1/audio/speech`,
`/v1/audio/transcriptions`, `/v1/tasks/run`, `/v1/tasks/batch`. Keep one long-lived
process: the ~29 s cold load is per process, not per request.

### Known caveats

- **LM Studio cannot run these GGUFs.** The `ming3_audio` family is registered only in
  audio.cpp; a GGUF container is not a portable architecture.
- No FST text normalizer: spell out numbers/abbreviations where pronunciation matters.
  Chinese numerals inside characters were read correctly in testing.
- Long clones can drop words inside a fragment; `text_chunk_size` defaults to 100, keep it.
- Only offline mode for `ming3_audio`; `ming3_tts` adds streaming.
- `--session-option ming3_audio.disk_experts=true` exists for lower VRAM; unnecessary at 4.2 GB.

## Route B — official PyTorch 0.5b

`requirements.txt` pins torch 2.6.0 (cu124), which has no sm_120 kernels. Replace with
torch/torchaudio 2.7.1 from the cu128 index, use Python 3.12, pass
`--no_text_normalization`, and expect the SDPA fallback (no flash-attn). See
[REPORT links] `inclusionAI/Ming-3.0-tts` issues #3 and #8. That route still offers the
best raw TTS metrics (lower WER than tiny/audio variants) but has no server and no
streaming.

## Files produced by the verification runs

audio-tiny (`out\`):

- `tts_en.wav` — named voice US_James, English
- `en\en_1.wav` … `en_3.wav` — 3-request English session
- `zh\zh_1.wav`, `zh_2.wav` — Chinese, default voice 灵小柔
- `clone_en.wav` — same-language clone from `en_2.wav`

tts-0.5b (`out\tts05\`, plus two at `out\`):

- `en_3.wav`, `en_four.wav` — plain TTS, English
- `en_ip_james.wav` — `ip` template, US_James
- `zh_1.wav`, `zh_2.wav` — Chinese, plain TTS
- `load_probe.wav` — cold-start timing run
- `out\clone_05.wav` — clone from `out\en\en_2.wav`
- `out\vdes_05.wav` — voice design from a text description
- `out\stream_05.wav` — streaming mode output

