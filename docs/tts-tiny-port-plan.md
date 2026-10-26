# Porting Ming-3.0-tts-tiny into audio.cpp — plan

Goal: get the `podcast` task (speaker-tagged dialogue with per-speaker references,
one generation pass) running on this machine, in the runtime we already use.

The blocker was never VRAM. Upstream shipped tiny for a Linux PyTorch stack, and no
GGUF/Windows port exists — verified: HF search `ming gguf` returns only TTS-0.5B and
Audio-Tiny, `audio-cpp/audio.cpp-gguf` has no tts-tiny files, and the only working
conversions are MLX/CoreAI for Apple Silicon.

## Status: checkout refreshed, all three binaries built (2026-10-26)

| step | result |
|---|---|
| `git fetch origin` | `origin/main` = `41c1f8ff` (v0.9.3, 2026-10-26); 1,052 commits ahead of `release-0.1` |
| branch | `ming3-tts-tiny` from `origin/main`; `release-0.1` untouched, 4 untracked files preserved |
| build | `scripts\build_windows.cmd -Preset windows-cuda-release -ModelSet custom -Models ming3_tts,ming3_audio`, targets `audiocpp_cli` / `audiocpp_gguf` / `audiocpp_server` |
| binaries | `audiocpp_cli.exe` 36 MB · `audiocpp_gguf.exe` 35 MB · `audiocpp_server.exe` 43 MB (all rebuilt from source) |
| version | `audio.cpp dev`, `git 41c1f8ff`, Release msvc 19.44, backends `cpu,cuda` |
| inference | Ming 0.5B GGUF through the from-source binary: 1.60 s audio, RTF 0.34 |
| registry | `--list-loaders` reports `ming3_tts` and `ming3_audio` |
| specs in tree | `model_specs/ming3_tts.json` (9.7 KB), `model_specs/ming3_audio.json` (29.5 KB) |

Gotcha worth keeping: a `dev` build has **no compiled model specs**, so
`--list-loaders` fails from an arbitrary working directory with *"model contract
spec not found for family 'ming3_tts'"*. Run from the repo root, or pass
`--model-spec-override`. Inference alone still works because the released GGUF
embeds its own spec.

## Why it is tractable

Evidence from `inclusionAI/Ming-3.0-tts-tiny/config.json` and the weight index
(10,285 tensors, 30 shards), compared against the two Ming families audio.cpp
already ships (`ming3_tts` = 0.5B, `ming3_audio` = Audio Tiny):

| component | tiny | already in audio.cpp | verdict |
|---|---|---|---|
| language backbone | `bailing_moe_v3`, hidden 1536, 24 layers, **128 experts / 8 active**, vocab 157,184 | Audio Tiny is `bailing_moe_v3_audio` (same family, MoE + expert offload) | reuse code, new dims |
| audio VAE | `audio_config` enc/dec, 44.1 kHz | both Ming families ship a 44.1 kHz audio VAE | reuse |
| CFM flow head | `flow_config`: hidden 1024, **depth 16**, 16 heads | 0.5B `ditar_config`: hidden 1024, depth 8, 16 heads | reuse blocks, depth 16 |
| aggregator | hidden 1024, depth 8, 16 heads | 0.5B `aggregator_config`: **identical** | reuse as-is |
| patch head (DiT) | `linear_proj_audio.*` (x/word embedder, rotary, 112 block tensors) | 0.5B ditar + Audio Tiny DiT | reuse |
| speaker encoder | CAMPPlus (`campplus.onnx`) | shared in both converters | reuse |
| `spk_head`, `stop_head` | 4 tensors, small projections | analogous heads exist in both families | wire up |
| **prompt/task contract** | 9 tasks incl. `podcast` | 0.5B has 8 tasks, **no podcast** | **new** |

Tensor prefixes: `model.model.*` 9,282 · `audio.encoder` 348 · `audio.decoder` 295 ·
`flowloss.cfm` 236 · `linear_proj_audio.*` 119 · `spk_head`/`stop_head` 4.

Conversion is favourable too: `convert_ming3_tts_gguf.py` is a thin wrapper around
`audiocpp_gguf`, driven by `model_specs/<family>.json`, which declares component
roles (`sources.tensors.{thinker,talker,vae,speaker}` with prefixes). Much of a
port is spec plus the C++ name mapping.

## Podcast contract (extracted from upstream)

From `reference/ming_tiny/task_protocol.py` and `prompt_contract.py`. These are the
details a C++ port must reproduce **byte-exactly**:

- **Script syntax is `speaker_N:` line prefixes**, not `[S1]`/`[S2]` (that is
  MOSS-TTSD's convention). Every line must match `^\s*speaker_(\d+)\s*:`; turns
  must be non-empty; the speaker ids present must be exactly `1..speaker_count`.
- **Canonical serialization** is `" speaker_{n}:{text}\n"` per turn — **leading
  space** on every line, trailing newline. That is the training-time string.
- **References are per speaker and many**: `TaskSpec("podcast", "multi_speaker",
  True, "none", "many", "many")` = many reference wavs, many reference texts, no
  structured controls. Reference transcripts are serialized the same way by
  `build_podcast_prompt_transcript`: `" speaker_1:{text}\n speaker_2:{text}\n"`.
- **Default seed is 1235** (`FORMAL_PODCAST_SEED`), not 1895.
- **Decode alignment is per task**: `task_decode_alignment.podcast` must be
  `"old_shift"`, and the runtime must fail closed otherwise. Frame→decode bounds go
  through `podcast_frame_bounds_to_decode_steps` / `podcast_generation_frame_bounds`
  — the label shift differs from the default.
- **Speaker conditioning** uses CAMPPlus (192-dim), with
  `speaker_start_token_id = 157175`, `speaker_end_token_id = 157176`; podcast
  refuses to run without a speaker extractor.

Three things are genuinely new relative to the 0.5B family: the tagged serializer
(trivial), multi-reference conditioning in one prompt (assembly), and the per-task
decode shift (touches the decode loop — where a silent quality bug would live).

## Work items

| # | item | where | rough size |
|---|---|---|---|
| 1 | checkout refresh + build | done | — |
| 2 | fetch checkpoint (~16 GB, 30 shards) | user | downloading |
| 3 | `model_specs/ming3_tts_tiny.json` (spec v1: 9 tasks incl. `podcast`, options, packages, sources) | audio.cpp tree | 1–2 days |
| 4 | conversion mapping in `audiocpp_gguf`: 9,282 backbone tensors, expert banks (128 stacked, routers F32), heads | C++ | 3–5 days |
| 5 | family runtime: reuse Audio Tiny's MoE/KDA/MLA graph, 0.5B's CFM/DiT/VAE, wire `spk_head`/`stop_head` | C++ | 1–2 weeks |
| 6 | prompt contracts: 8 existing tasks + `podcast` (tags, multi-reference, old-shift decode) | C++/spec | 3–5 days, plus iteration |
| 7 | validation: parity vs the official Python runtime, RTF/VRAM, docs page, license row, path tests | tests/docs | 2–3 days |

Total: **4–6 weeks** for one person outside the codebase, item 6 the uncertainty —
their own 0.5B port needed follow-up PRs #1020/#1021/#1034 before it was right.

## Risks worth pricing in

- **Podcast prompt + decode shift.** Reference: `task_protocol.py` (26.8 KB) and
  `prompt_contract.py` (16.6 KB), both now local under `reference/ming_tiny/`.
  Getting the leading spaces, the reference ordering and `old_shift` byte-exact is
  the difference between "talks" and "talks like the demo".
- **Expert routing.** 128 experts / 8 active at hidden 1536 differs from Audio
  Tiny's Thinker; the MoE path must be parameterised, not copied.
- **Quantisation of the small heads.** MOSS-TTSD's lesson (`lm_heads` at q4_k ends
  turns early): keep `stop_head`/`spk_head` at bf16 or f32 in the package.
- **Tokenizer.** vocab 157,184 plus the Bailing chat template — `<role>` tags, not
  Qwen's `<|im_start|>`.
- **Context.** The 0.5B has a 512-position output budget; tiny's dialogue takes
  need their own limit or the take truncates mid-turn.

## Sequence once the weights are local

```powershell
# 1. weights
#    C:\Projects\ming30\models\ming3-tts-tiny\
#    (config.json, model-*-of-00030.safetensors, campplus.onnx, tokenizer files)

# 2. convert (converter to be written, modelled on convert_ming3_tts_gguf.py)
python tools\models\convert_ming3_tts_tiny_gguf.py `
  --source C:\Projects\ming30\models\ming3-tts-tiny `
  --output C:\Projects\ming30\models\ming3-tts-tiny\ming-3.0-tts-tiny-q8_0.gguf --type q8_0

# 3. smoke: one tagged dialogue with two references
#    podcast_script.txt
#      speaker_1: So what did you make of it?
#      speaker_2: Honestly, I was not expecting that ending.
audiocpp_cli --family ming3_tts_tiny --task tts --backend cuda `
  --model ...q8_0.gguf --language English `
  --text "$(Get-Content -Raw podcast_script.txt)" `
  --request-option voice_samples=ref1.wav,ref2.wav `
  --request-option "reference_text= speaker_1: ...`nspeaker_2: ..." `
  --out dialogue.wav --metrics
```

## Spec skeleton (draft)

```json
{
  "schema_version": 1,
  "family": "ming3_tts_tiny",
  "display_name": "Ming 3.0 TTS Tiny",
  "category": "tts",
  "status": "experimental",
  "tasks": ["tts", "clone", "design", "podcast"],
  "modes": ["offline"],
  "languages": ["zh", "en", "ja", "ko", "de", "fr", "es"],
  "capabilities": {
    "tts": ["speaker_reference", "built_in_voices", "style_control", "emotion_control"],
    "podcast": ["multi_speaker", "speaker_reference"]
  },
  "runtime": { "tags": ["gguf"] },
  "options": { "request": [ /* per-task controls from the official task_protocol.py */ ] },
  "packages": [{ "id": "ming3_tts_tiny_q8_0", "precision": "q8_0", "files": ["ming-3.0-tts-tiny-q8_0.gguf"] }]
}
```

## References on disk

`reference/ming_tiny/` holds upstream copies (MIT) for byte-exact comparison:
`prompt_contract.py`, `task_protocol.py`, `inference.py`, `api.py`, `config.json`,
`model.safetensors.index.json`, plus `PROVENANCE.md`. Weights are not included.
