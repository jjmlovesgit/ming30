# Porting Ming-3.0-tts-tiny into audio.cpp â€” plan

Goal: get the `podcast` task (speaker-tagged dialogue with per-speaker references,
one generation pass) running on this machine, in the runtime we already use.

The blocker was never VRAM. It is that upstream shipped tiny for a Linux PyTorch
stack, and no GGUF/Windows port exists (verified: HF search `ming gguf` returns
only TTS-0.5B and Audio-Tiny; `audio-cpp/audio.cpp-gguf` has no tts-tiny files;
the only working conversions are MLX/CoreAI for Apple Silicon).

## Status: checkout refreshed and build verified (2026-10-26)

| step | result |
|---|---|
| `git fetch origin` | `origin/main` = `41c1f8ff` (v0.9.3, 2026-10-26); 1,052 commits ahead of `release-0.1` |
| branch | `ming3-tts-tiny` from `origin/main`; `release-0.1` untouched, 4 untracked files preserved |
| build | `scripts\build_windows.cmd -Preset windows-cuda-release -Target audiocpp_cli -ModelSet custom -Models ming3_tts,ming3_audio` â†’ 392/392, exit 0 |
| binary | `build\windows-cuda-release\bin\audiocpp_cli.exe` â€” `audio.cpp dev`, `git 41c1f8ff`, backends `cpu,cuda` |
| inference | Ming 0.5B GGUF through the from-source binary: 1.60 s audio, RTF 0.34 |
| registry | `--list-loaders` reports `ming3_tts` and `ming3_audio` (custom model set) |
| specs in tree | `model_specs/ming3_tts.json` (9.7 KB), `model_specs/ming3_audio.json` (29.5 KB) |

Two things learned that matter for port work:

- **A `dev` build has no compiled model specs.** `--list-loaders` fails from an
  arbitrary working directory with *"model contract spec not found for family
  'ming3_tts'"*; run from the repo root so `model_specs/` is found, or pass
  `--model-spec-override`. Inference alone still works because the released GGUF
  embeds its spec.
- **`audiocpp_server.exe` was not rebuilt** (target was `audiocpp_cli` only); it is
  still the July binary. Rebuild with `-Target audiocpp_server` when needed, and
  build `audiocpp_gguf` before conversion work.



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

Tensor prefixes are unambiguous: `model.model.*` 9,282 Â· `audio.encoder` 348 Â·
`audio.decoder` 295 Â· `flowloss.cfm` 236 Â· `linear_proj_audio.*` 119 Â·
`spk_head`/`stop_head` 4.

The conversion path is also favourable: `convert_ming3_tts_gguf.py` is a thin
wrapper around `audiocpp_gguf`, driven by `model_specs/<family>.json`. The spec
declares component roles (`sources.tensors.{thinker,talker,vae,speaker}` with
prefixes), so much of a port is spec plus the C++ name mapping.

## Work items

| # | item | where | rough size |
|---|---|---|---|
| 1 | refresh checkout: `release-0.1` @ `acde132` (2026-07-02) â†’ current `main` (v0.9.3), rebuild | their repo | hours; toolchain already proven (`windows-cuda-release`, arch `120a-real`) |
| 2 | fetch checkpoint | `inclusionAI/Ming-3.0-tts-tiny` | 30 shards, ~16 GB bf16 |
| 3 | `model_specs/ming3_tts_tiny.json` (spec v1: tasks incl. `podcast`, options, packages, sources) | audio.cpp tree | 1â€“2 days |
| 4 | conversion mapping in `audiocpp_gguf`: 9,282 backbone tensors, expert banks (128 experts stacked, routers F32), heads | C++ | 3â€“5 days |
| 5 | family runtime: reuse Audio Tiny's MoE/KDA/MLA graph, 0.5B's CFM/DiT/VAE, wire `spk_head`/`stop_head` | C++ | 1â€“2 weeks |
| 6 | prompt contracts: 8 existing tasks + `podcast` (speaker tags, per-speaker references) | C++/spec | 3â€“5 days, plus iteration |
| 7 | validation: parity vs the official Python runtime, RTF/VRAM, docs page, license row, path tests | tests/docs | 2â€“3 days |

Total: **4â€“6 weeks** for one person working outside the codebase, with item 6 as
the uncertainty (their own 0.5B port needed follow-up PRs #1020/#1021/#1034 for
language, chunking and text splitting before it was right).

## Risks worth pricing in

- **Podcast prompt format.** The authoritative reference is the official repo:
  `ming_tiny/task_protocol.py` and `ming_tiny/prompt_contract.py` (27 KB each on
  HF). Podcast has per-speaker references and speaker-tagged assembly; getting
  the tags and reference ordering byte-exact is the difference between "talks"
  and "talks like the demo".
- **Expert routing.** 128 experts / 8 active with hidden 1536 differs from
  Audio Tiny's Thinker; the MoE path must be parameterised, not copied.
- **Quantisation of the small heads.** MOSS-TTSD's lesson (`lm_heads` at q4_k ends
  turns early): keep `stop_head`/`spk_head` at bf16 or f32 in the package.
- **Tokenizer.** vocab 157,184 plus the Bailing chat template â€” the prompt
  assembly uses `<role>` tags, not Qwen's `<|im_start|>`.
- **Context.** The 0.5B has a 512-position output budget; tiny's dialogue takes
  need their own limit or the podcast take will truncate mid-turn.

## Sequence once the weights are local

```powershell
# 1. current source (leave release-0.1 untouched)
git -C C:\Projects\audio.cpp fetch origin
git -C C:\Projects\audio.cpp switch -c ming3-tts-tiny origin/main
cmake --build C:\Projects\audio.cpp\build\windows-cuda-release --parallel 8 --target audiocpp_cli audiocpp_server audiocpp_gguf

# 2. weights (user-supplied)
#    C:\Projects\ming30\models\ming3-tts-tiny\  (config.json, model-*-of-00030.safetensors, campplus.onnx, tokenizer files)

# 3. convert (converter to be written, modelled on convert_ming3_tts_gguf.py)
python tools\models\convert_ming3_tts_tiny_gguf.py `
  --source C:\Projects\ming30\models\ming3-tts-tiny `
  --output C:\Projects\ming30\models\ming3-tts-tiny\ming-3.0-tts-tiny-q8_0.gguf --type q8_0

# 4. smoke: one tagged dialogue, one reference pair
audiocpp_cli --family ming3_tts_tiny --task tts --backend cuda `
  --model ...q8_0.gguf --language English `
  --text "[S1] So what did you make of it? [S2] Honestly, I was not expecting that ending." `
  --request-option voice_samples=ref1.wav,ref2.wav `
  --request-option "reference_text=[S1] ... [S2] ..." --out dialogue.wav --metrics
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

## What can be done before the weights land

1. **Refresh and build** the checkout (validates the toolchain against Ming-era
   code â€” the current tree predates `ming3_audio`, so this alone is informative).
2. **Write the spec + converter**, tested as far as a dry run on the weight index
   allows (tensor names, dtypes, expert stacking order).
3. **Extract the prompt contracts** from `ming_tiny/task_protocol.py` and
   `prompt_contract.py` into a test fixture set, so the C++ side has something to
   match on the first run.
4. **File the upstream issue** with this table â€” if the maintainers take it, the
   port lands faster and maintained.

