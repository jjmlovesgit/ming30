# Next session

Ordered plan, with the state each step starts from.

## 1. Podcast path (the original goal, CPU first)

1. **Match the podcast prompt.** The checkpoint's `podcast` task is the multi-speaker mode:
   script lines prefixed `speaker_<N>:`, a `reference_text` carrying the same tagged
   transcript, `voice_samples` with one reference wav per speaker, seed **1235**, and
   `old_shift` decode per fragment. Our current prompt builder emits the single-speaker
   `seed` prompt only (`port/ming3_tts_tiny/prompt.cpp`).
2. **Add per-speaker conditioning.** One CAMPPlus embedding per reference wav, mapped onto the
   `<spk>` slot for each speaker's line. The thinker already has the speaker slot
   (`speaker_embedding_position`), but nothing fills it from references.
3. **Render a short two-speaker exchange on CPU** and listen. This is the first audible test
   of the actual goal; the 2.40 s single-speaker render does not count for it.

Reference material: `docs/tts-tiny-port-plan.md`, `docs/ming-tiny-task-matrix.md`, and the
checkpoint's `prompt_contract.py` (`build_tts_prompt_ids` with `prompt_text` and
`prompt_latent_tokens`).

## 2. Long-line splitting

The fragment ceiling is 250 patches × 80 ms = **20 s**, and `old_shift` supplies no
end-of-sentence decision. A podcast line longer than that must be split into repeated
`speaker_N:` lines. A splitter does not exist yet; chunking mid-line is not acceptable because
the model expects the speaker tag to open each fragment.

## 3. GPU path (resume with the BF16 resident-bank test)

First step, unchanged: **one BF16 expert read directly from a resident bank at a device
offset**, GEMM in BF16, compared against the CPU expert (the f32 path already achieves
~5e-08, so that is the target). Do **not** revive the host-slicing path — 9.0 ms per layer of
re-extracted float32 is exactly the work device offsets avoid.

Then all 23 MoE layers through that path, values compared, timed warm step with peak VRAM
against the **3.02 s** CPU baseline. Only that run turns the 0.4–0.6 s estimate into a
measurement, and only then is the ggml graph port judged worth it (with the standing check
that KDA and routing actually execute on CUDA rather than falling back).

## State to start from

- Single-speaker CPU path verified: `docs/evidence/milestone-12-cached-decode.md` and earlier.
- Scope correction (what is and is not done): `docs/evidence/milestone-21-status-correction.md`.
- GPU components: projections `milestone-15`, experts `milestone-18`, residency `milestone-19`,
  hybrid cost arithmetic `milestone-20` (explicitly not a measured step).
- Binary: `build/windows-cuda-release/bin/audiocpp_cli.exe`; probes are env-gated
  (`MING3_TINY_*`), so a normal run is unaffected.
