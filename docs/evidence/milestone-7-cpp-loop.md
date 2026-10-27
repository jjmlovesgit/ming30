# C++ generation loop — parity patch by patch, then amplification

## What was built

`loop.cpp` runs the same loop as the Python reference, with the Python noise stream
replayed from a file so both sides see identical inputs:

- `thinker_forward_embeds` (new entry point) runs the 24 layers over given embeddings,
  so a projected latent can be appended;
- `flow_sample_from_noise` (new entry point) runs one full Euler solve from an explicit
  noise vector;
- the Aggregator (`linear_proj_audio`: CLS + latent, 8 blocks, CLS out to 1536) produces
  the feedback embedding;
- per patch it records the latent and feedback vector.

4-patch parity check (Python noise replayed):

| patch | latent max_abs | latent rel | feedback max_abs | feedback rel |
|---|---|---|---|---|
| 0 | 4.47e-06 | 2.07e-06 | 3.07e-07 | 3.90e-06 |
| 1 | 1.08e-05 | 4.89e-06 | 3.99e-07 | 6.15e-06 |
| 2 | 7.56e-06 | 4.39e-06 | 5.55e-07 | 5.15e-06 |
| 3 | 1.37e-05 | 6.38e-06 | 4.47e-07 | 6.59e-06 |

## Full 30-patch run

Error grows smoothly, then the trajectories split:

| patches | latent rel_rmse |
|---|---|
| 0–4 | 2e-06 – 9e-06 |
| 5–13 | 4e-05 – 4e-04 |
| 14–16 | 1.6e-03 – 6.2e-03 |
| **17** | **6.2e-02** |
| 18–29 | 7e-02 – 6e-01 |

Latent RMS per patch stayed matched to 4 decimals throughout, and the stop head stayed
quiet (4e-05 – 1.3e-04) on both sides.

Decoding the C++ latents with the same VAE gives a waveform whose correlation with the
Python render is **0.927**.

## Reading it

This is the expected behaviour of an autoregressive feedback loop, not a code error: a
~1e-5 per-step difference compounds through the fed-back latent until it crosses one of
the narrow routing/flow margins documented earlier, at which point the two trajectories
separate and stay separated. The smooth growth up to patch 16 and the abrupt jump at 17
are the signature.

So the honest status is: the C++ loop is correct to the same tolerance as its parts for
the first ~16 patches, and beyond that the comparison is no longer meaningful without
per-step routing capture to identify which margin was crossed.

## Next steps (in the agreed order)

1. **`old_shift` length rule.** The loop currently runs a fixed patch count because
   `old_shift` decode ignores the stop head. Reproduce the intended length mechanism (or
   record explicitly that fixed length is the behaviour), so a sentence ends where the
   model intends.
2. **GPU speed.** 30 patches take ~25 min on CPU in C++ (~50 s/patch with full recompute);
   the same loop on the RTX 5090 needs a KV-cache path and CUDA placement for the thinker
   and flow matmuls.
3. Optional: per-patch routing capture in the loop, to name the margin crossed at patch 17
   rather than inferring it.
