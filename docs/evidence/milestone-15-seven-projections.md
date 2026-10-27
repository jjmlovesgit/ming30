# All seven projections verified; transfer is the thing to engineer away

## Per-projection verification (layer 0 KDA, 20 iterations each)

| projection | shape | max_abs | rel_rmse | mismatched | GEMM ms | xfer ms |
|---|---|---|---|---|---|---|
| q_proj | 2048×1536 | 1.49e-06 | 6.71e-07 | 0/2048 | 1.830 | 0.073 |
| k_proj | 2048×1536 | 1.19e-06 | 7.04e-07 | 0/2048 | 0.034 | 0.073 |
| v_proj | 2048×1536 | 1.49e-06 | 6.89e-07 | 0/2048 | 0.035 | 0.066 |
| f_proj | 2048×1536 | 2.26e-06 | 6.52e-07 | 0/2048 | 0.037 | 0.074 |
| g_proj | 2048×1536 | 8.11e-06 | 7.39e-07 | 0/2048 | 0.036 | 0.073 |
| b_proj | 16×1536 | 6.56e-07 | 6.11e-07 | 0/16 | 0.051 | 0.070 |
| o_proj | 1536×2048 | 1.61e-06 | 8.64e-07 | 0/1536 | 0.040 | 0.079 |

`o_proj` consumes `q_proj`'s output, so chaining is exercised, not just independent products.
All seven pass with every value compared.

## Two numbers that matter

1. **The first GEMM costs 1.83 ms; the warm ones cost ~0.035 ms.** That first figure is
   cuBLAS initialisation and first-call setup, not arithmetic. Steady state for a layer's
   seven projections is therefore ≈ **0.25 ms** of GPU compute.

2. **Transfer totals 0.508 ms for the same seven calls (~0.073 ms each).** That is ~2x the
   warm compute. If activations round-trip per projection, transfer eats the gain — exactly
   the failure mode to avoid. Keeping activations on the device between layers is what
   makes the GPU path worthwhile, which is what the seeded-patch gate will test.

For reference, the CPU path took 12.443 ms for the same seven products in this harness
(including weight loads), against 2.572 ms GPU end to end (compute + transfer) and ~0.25 ms
warm compute.

## What this still does not show

- **Whole-model speed.** One layer's projections, at `n = 1`, on 12 MiB of weights.
- **Whole-model VRAM.** The thinker's weights are ~33 GB at float32 and ~16.8 GB at bf16;
  neither fits alongside activations at float32 on a 32 GB card. Residency strategy is an
  open question, and the 72 MiB / 12 MiB figures here are meaningless for it.
- **The seeded patch through the actual model on GPU** — still the real gate.

## Next

1. Seeded patch on GPU: weights and activations resident, prefill + one cached step,
   compared value-by-value against the CPU result (4.8e-06 baseline), with wall time and
   peak VRAM measured.
2. Decide bf16 residency (and whether to stay on cuBLAS or move to ggml graphs) from those
   measurements, not from this microbenchmark.
