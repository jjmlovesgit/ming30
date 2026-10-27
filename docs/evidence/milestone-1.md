# Milestone 1 — family recognises and describes the package

Command (from `C:\Projects\audio.cpp`, branch `ming3-tts-tiny`):

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe --family ming3_tts_tiny --task tts --backend cpu `
  --model C:\Projects\ming30\out\ming-tiny-probe-q8.gguf `
  --text "Structural probe." --out C:\Projects\ming30\out\probe-run.wav
```

Output:

```
[ming3_tts_tiny] structural package report
  family          : ming3_tts_tiny
  spec            : embedded, contract resolved for family ming3_tts_tiny
  tokenizer_json  : 12210216 bytes (resolved from package)
  chat_template   : 5984 bytes
  generation_conf : resolved
  config          : layers=24 experts=128 top=8 hidden=1536 vocab=157184 flow_depth=16 sample_rate=44100
  tensors         : 1645
  lm_head.weight  : present
  expert bank     : model.layers.1.mlp.experts.down_proj.weight dtype=q8_0 shape=[128,1536,512]
  inference       : not implemented (milestone 1: structural verification only)
[ming3_tts_tiny] contract: schema v1 loaded for family ming3_tts_tiny, task=5 mode=0
audiocpp_cli failed: Ming 3.0 TTS Tiny is loaded as a structural probe: inference is not implemented in this build
```

Exit code 1, deliberately: the family refuses inference instead of attempting speech.

## What this proves, and what it does not

| requirement | evidence |
|---|---|
| family recognised | `--list-loaders` lists `ming3_tts_tiny`; the report prints it |
| package opened | `--model ...ming-tiny-probe-q8.gguf` loads |
| embedded spec resolved | `audiocpp.model_spec.json` parsed; contract schema v1 loaded |
| tokenizer resolved from package | `tokenizer_json: 12210216 bytes` — matches the source file byte count exactly |
| tensors enumerated | `tensors: 1645`, `lm_head.weight: present` |
| q8 expert bank seen | `model.layers.1.mlp.experts.down_proj.weight dtype=q8_0 shape=[128,1536,512]` |
| probe reported incomplete | explicit report line + refusal message |

Not proved: inference, audio quality, parity. The package is a **structural** probe
(layer-0 tensors plus one layer's banks); a complete lossless model is the first
speech test.

## Metadata requirements discovered while getting here

Each of these failed loudly on the way, so they are recorded for whoever writes the
next package or loader:

1. `audiocpp.embedded_files.offsets` must be an **ARRAY of UINT64** — INT64 fails
   validation with "embedded binary sidecar metadata is invalid".
2. `voice_id` is a **shared option with a registered type**: it must be `enum`, not
   `string`, or spec parsing fails.
3. `ui.tags` is a **closed vocabulary** — `Dialogue` is rejected; `TTS` and `GGUF` are
   accepted.
4. Capability strings are validated too: `multi_speaker` was rejected, so the
   multi-speaker claim lives in the description and docs rather than capabilities.
