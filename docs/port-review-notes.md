# Notes for the audio.cpp maintainers — Ming 3.0 TTS Tiny port

Status: **pre-port, asking for direction.** No PR is open, nothing is pushed to
`0xShug0/audio.cpp`. This file records what we found, what we propose, and the
questions we need answered before writing C++ that you would have to review.

Prepared 2026-10-26 against `main` @ `41c1f8ff` (v0.9.3).

## What we are proposing

Add a third Ming family, `ming3_tts_tiny`, converting
[`inclusionAI/Ming-3.0-tts-tiny`](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny)
(7.9B total / 1.3B active, Bailing-MoE-V3 backbone, 44.1 kHz). The reason is one
capability the existing two families do not have: the upstream **`podcast`** task —
speaker-tagged dialogue with per-speaker references in a single generation pass.

We are **not** proposing framework, ggml or backend changes. If something needs
one, we will split it out as its own PR first, per CONTRIBUTING.

## Why we think it is tractable

The two families you already ship cover most of the parts (verified against the
checkpoint config and weight index, not inferred from the README):

| component | tiny | existing coverage |
|---|---|---|
| backbone | `bailing_moe_v3`, hidden 1536, 24 layers, **128 experts / 8 active**, vocab 157,184 | `ming3_audio` is `bailing_moe_v3_audio` (MoE + expert offload, PR #1023) |
| audio VAE | `audio_config` enc/dec, 44.1 kHz | both Ming families |
| CFM head | `flow_config`: hidden 1024, **depth 16**, 16 heads | `ming3_tts` ditar: hidden 1024, depth 8, 16 heads |
| aggregator | hidden 1024, depth 8, 16 heads | `ming3_tts`: identical |
| patch head | `linear_proj_audio.*` | `ming3_tts` ditar / Audio Tiny DiT |
| speaker encoder | CAMPPlus | shared by both converters |
| `spk_head`, `stop_head` | 4 small tensors | analogous heads exist |
| **tasks incl. podcast** | 10 tasks | 0.5B has 8, no podcast |

Tensor counts: `model.model.*` 9,282 · `audio.encoder` 348 · `audio.decoder` 295 ·
`flowloss.cfm` 236 · `linear_proj_audio.*` 119 · `spk_head`/`stop_head` 4.

## Upstream behaviour we must reproduce exactly

From `ming_tiny/task_protocol.py` and `prompt_contract.py` (copies in this repo
under `reference/ming_tiny/`):

1. **Dialogue syntax is `speaker_N:` line prefixes**, matching
   `^\s*speaker_(\d+)\s*:`. Not `[S1]`/`[S2]` — that is MOSS-TTSD's convention.
2. **Canonical serialisation** is `" speaker_{n}:{text}\n"` — leading space on
   every line, trailing newline. Reference transcripts are serialised the same way
   and prepended, so podcast conditioning is (wav + tagged transcript) per speaker.
3. **Reference counts are strict**: speaker ids used in the script must be exactly
   `1..speaker_count`, and each needs a reference.
4. **Seed defaults differ per task**: 1895 general, **1235 for podcast**
   (`FORMAL_PODCAST_SEED`).
5. **Decode alignment is per task**: `task_decode_alignment.podcast` must be
   `old_shift`, and upstream fails closed otherwise. Frame→decode bounds go through
   `podcast_frame_bounds_to_decode_steps`. A wrong shift would not crash — it would
   produce mistimed speech.
6. **Numerics pins** in the reference environment:
   `BAILING_KDA_FUSED_RECURRENT=0`, `BAILING_MLA_ATTN=sdpa`,
   `MING_TINY_AUDIO_VAE_ATTN=sdpa`, `FLA_USE_CUDA_KDA_FWD/BWD=0`. The published
   numbers come from SDPA attention with a non-fused KDA recurrence.

## Questions for you (blocking design decisions)

1. **Task vs template.** The 10 upstream modes fit the existing `template_name`
   pattern used by `ming3_tts` (extended with `seed`, `paralinguistic`,
   `podcast`), rather than adding a `podcast`/`dialogue` task to the audio.cpp task
   enum. We would prefer that — smaller surface, no server/spec churn. Do you want
   `podcast` exposed as a task instead?
2. **Multi-reference option names.** We drafted
   `speaker_references` + `speaker_reference_texts` (positional lists). The
   MOSS-TTSD family uses `voice_samples` + `reference_text`. Which convention do
   you want new multi-speaker families to follow?
3. **`reference_cache` keying.** For podcast it must key on the whole reference
   set; the 0.5B session option assumes one waveform. Is a per-family cache shape
   acceptable, or would you rather keep the option and give podcast its own?
4. **`flow_steps`.** Upstream exposes 10 Euler steps as a formal default; the
   `ming3_tts` spec does not surface step count. Should tiny expose `flow_steps`,
   or hard-code the formal value?
5. **`max_tokens` default.** Tiny's context is not the 0.5B's 512 positions. We
   will measure before setting a default; do you want the spec to leave it unset
   until then?
6. **Parity mode.** Given the numerics pins, would you accept a documented
   "reference parity" configuration (SDPA, non-fused KDA) alongside the optimised
   path, so quality comparisons stay meaningful on future PRs?
7. **Packaging.** There is no public GGUF host yet. Is `audio-cpp/Ming-3.0-TTS-Tiny-GGUF`
   the intended home, and is a bf16 default acceptable given ~16 GB, or should the
   default package be q8_0?
8. **Placement.** `community_models/` first (per CONTRIBUTING), or core, given it
   is the third Ming family and shares runtimes with `ming3_audio`?
9. **Hosting the PR.** We have no write access; the PR would come from a fork. Is
   that acceptable for a port of this size, or would you rather take the design
   record and implement it in-tree?

## Evidence we will provide with the PR

Mapped to CONTRIBUTING's New Model PRs checklist:

- exact configure/build commands and the target set used;
- exact CLI and server invocations, model spec and package id;
- generated audio artifacts plus the text that produced them;
- **parity table against the official Python runtime** (same seed, same prompt,
  same references) covering `seed`, `podcast` (two speakers), and `emotion`;
- path-test results, including `podcast` with mismatched speaker counts (must
  reject), empty turns (must reject) and a single-speaker script (must reject);
- backend tested: CUDA on RTX 5090 (Blackwell, sm_120);
- timing: RTF, RSS, peak VRAM, including a long-lived multi-request session and a
  long-form request through the framework text chunker (the 0.5B took three
  follow-up PRs — #1020, #1021, #1034 — to get chunking and language handling
  right, so we expect the same);
- known limitations, honestly listed.

## Things that surprised us (worth knowing before review)

- **`--list-loaders` fails on a dev build** from an arbitrary working directory:
  *"model contract spec not found for family 'ming3_tts'"*. Running from the repo
  root works; the released GGUF embeds its spec, so inference alone is unaffected.
  We hit this while validating the toolchain and mention it in case the error text
  could be friendlier.
- **Windows build needs the repo's own script** (`scripts/build_windows.cmd`); a
  bare `cmake -S . -B build/windows-cuda-release` in a normal shell fails the
  compiler test with `LNK1104: cannot open file 'kernel32.lib'` because the MSVC
  environment is not initialised.
- **The only existing ports are Apple**: `mlx-community/Ming-3.0-tts-tiny-{bf16,8bit,4bit}`,
  `vanch007/...-mlx-*`, and `coder543/ming-3.0-tts-tiny-glade` (CoreAI/ANE). Nothing
  runs on Windows/CUDA today, which is why we are asking rather than using one.

## Where we are

- Checkout refreshed to `41c1f8ff` on a local branch `ming3-tts-tiny`;
  `audiocpp_cli`, `audiocpp_gguf` and `audiocpp_server` all build (custom model set:
  `ming3_tts,ming3_audio`), and Ming 0.5B inference runs through the from-source
  binary.
- Upstream reference sources vendored under `reference/ming_tiny/` with provenance.
- Checkpoint downloading (~16 GB, 30 shards).
- Draft spec: `reference/drafts/ming3_tts_tiny.json` (with TODOs where we refuse to
  guess).
- No C++ written yet. We would rather agree on the four design questions above than
  produce a large PR that has to be reshaped.
