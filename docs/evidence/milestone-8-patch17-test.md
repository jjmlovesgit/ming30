# Patch-17 targeted test — the split is accumulated drift, not a bug

The full loop's latent error grew smoothly and then jumped at patch 17, which is
*consistent with* feedback amplification but did not exclude a bug at that patch. This
test removes the ambiguity: C++ enters patch 17 with Python's exact embedding history
(prompt + feedback vectors for patches 0–16, 58 tokens), Python's saved noise vector for
patch 17, and Python's latent history, then runs one patch.

```
seeded patch: tokens=58, latent_rms=0.866468

latent max_abs = 4.768e-06
latent rel_rmse = 2.254e-06
python latent rms 0.866468   cpp 0.866468
```

**Verdict: the C++ patch-17 computation is correct to the same ~5e-06 tolerance as its
parts.** The split in the full run therefore came from accumulated earlier differences
crossing a narrow margin, not from a defect in the patch where it became visible.

That also means no routing/flow trace of patch 17 is needed: the directed trace the
coordinator asked for would only have been required had this test still diverged.

## What this establishes about the loop

- Each patch is computed correctly given correct inputs (this test, plus the 4-patch
  parity at 1e-5).
- The full-loop divergence is the expected behaviour of an autoregressive feedback loop
  whose per-step residual is ~1e-5: error grows smoothly, then the trajectory separates at
  the first sufficiently narrow routing/flow margin (~patch 17 here).
- Consequently, comparing full loops patch-by-patch beyond that point measures chaotic
  sensitivity, not implementation quality. The per-patch seeded test is the meaningful
  comparison, and it is now a reusable tool (`MING3_TINY_SEED_EMBEDS` /
  `MING3_TINY_SEED_NOISE` / `MING3_TINY_SEED_HISTORY`, ~25 s per patch instead of 25 min
  for a full run).

## Next steps (unchanged order)

1. **`old_shift` length rule** — the loop runs a fixed patch count because `old_shift`
   decode ignores the stop head; either reproduce the intended length mechanism or record
   fixed length as the behaviour.
2. **KV-cache / CUDA path** — 30 patches at ~50 s each on CPU; the seeded single-patch
   path also makes benchmarking one patch practical.
3. Optional: a per-patch seeded sweep (17, 20, 25) if we later want to confirm the
   tolerance holds deep into a sequence.
