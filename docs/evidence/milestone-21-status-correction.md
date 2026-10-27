# Project status — corrected scope

Two things need stating plainly, because earlier notes drifted towards "done".

## Complete: single-speaker CPU port

| stage | evidence |
|---|---|
| A prompt ids | exact match, sha256 identical |
| B thinker (24 layers, KDA+MLA, MoE, final norm, stop) | ~1e-06 per layer, 0 expert flips over 984 token-layers |
| C flow (condition projection, both guidance branches, 10 Euler steps, latents) | ~2e-05 velocity, ~8e-06 latents |
| D VAE (single and 50-patch, streaming == whole sequence) | waveform correlation 1.00000000 |
| E loop (prefill + cached steps, 30 patches) | 2.40 s of generated speech, bitwise vs full recompute |
| duration cap | `max_tokens` = 250 patches = 20 s, the only length control under `old_shift` |
| CPU speed | 2.31 s/step cached, 266 s for 30 patches |

That is **single-speaker text to speech, locally, numerically verified against the reference**.

## NOT complete: the original podcast goal

The reason this port started was voice-swapping a two-speaker NotebookLM podcast. That is
**not** covered by anything above:

- **Podcast prompt behaviour is unimplemented.** The podcast task needs the multi-speaker
  script form (`speaker_N:` line prefixes), a `reference_text` carrying the same tagged
  transcript, `voice_samples` with one reference wav per speaker, seed **1235**, and
  `old_shift` decode per fragment. Our loop builds the single-speaker `seed` prompt and takes
  no references at all.
- **Multi-speaker references are unimplemented.** No per-speaker CAMPPlus conditioning, no
  reference-latent path, no speaker-position handling beyond the fixed `<spk>` slot.
- **No audible test of either.** The 2.40 s render is one speaker from one prompt; nothing has
  been listened to as a two-speaker exchange.
- The 20 s fragment ceiling (250 patches at 80 ms) means a podcast line longer than ~20 s must
  be split into repeated `speaker_N:` lines, and that splitter does not exist yet.

## Also outstanding: GPU speed (not needed for correctness)

GPU status is component-verified only: seven projections to 1e-06, eight experts to ~5e-08,
BF16 bank residency at 12.94 GiB with a 14.52 GiB peak. **No hybrid step has run**, so the
0.4–0.6 s step is a hypothesis built from measured components.

Correction to the plan: **drop the host-slicing path** (9.0 ms per layer). The banks are
already resident on the device, so the right implementation indexes into the BF16 banks at a
device offset and runs the expert GEMM in BF16, avoiding the float32 extraction entirely.

Next steps, in order:

1. **One BF16 expert directly from a resident bank**, output compared against the CPU expert
   (target ~5e-08 as the f32 path already achieves).
2. Then all 23 MoE layers through that path, values compared, and a timed warm step with peak
   VRAM — that is the only thing that turns the 0.4–0.6 s estimate into a measurement.
3. Separately, and more importantly for the original goal: the **podcast path** — multi-speaker
   prompt, per-speaker references, and an audible two-speaker test.

GPU speed and podcast support are independent. A fast single-speaker model does not satisfy
the podcast goal, and a correct podcast render would still be too slow for practical use
without the GPU work.
