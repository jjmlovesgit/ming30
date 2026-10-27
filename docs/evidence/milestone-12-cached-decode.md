# Cached decode: bitwise identical, 5.6x faster end to end

## Parity, three ways

| comparison | max_abs |
|---|---|
| isolated: full recompute (58 tokens) vs prefill 41 + 17 cached steps | **0.000e+00** |
| end to end: cached 30-patch loop vs previous full-recompute 30-patch loop | **0.000e+00** |
| cached loop vs Python loop | same as before (1.4e+00 at the documented patch-19 split; 4.5e-06 … 1.4e-05 for patches 0–3) |

The cached path is bitwise identical to recomputation both in isolation and through the
whole generation loop. The difference against Python is unchanged from the earlier
analysis, i.e. not introduced by caching.

## Speed

| configuration | 30-patch loop | per step | notes |
|---|---|---|---|
| full recompute per patch | ~1500 s | ~50 s | original loop |
| cached steps, no weight residency | — | 4.48 s | 17-step check |
| cached steps + resident floats | — | 4.07 s | dense weights only |
| **cached steps + resident expert banks** | **266.5 s** | **2.31 s** | current |

5.6x end-to-end, and 21x better per incremental result than recomputing the sequence.

Residency is what unlocked it: the three expert banks per MoE layer are ~200 MB each, so a
single-token step was reading ~14 GB from the package. With bank bytes resident under a
16 GB budget, steps drop from 4.07 s to 2.31 s.

Remaining per-step cost is now mostly arithmetic in the 24 layers at one token, which is
exactly what CUDA can address.

## State of the decode path

- KV cache: KDA recurrent + convolution rows, MLA keys/values — verified bitwise.
- Weight residency: dense weights and expert banks held in a bounded FIFO cache.
- Loop: prefill once, one projected latent per patch.
- Duration cap: `max_tokens` (default 250 = 20 s), the only length control under `old_shift`.

## Next: CUDA

Two options were considered:

1. **cuBLAS-backed linear layers with device-resident weights and a device KV cache** — a
   bounded first increment that directly attacks the 2.31 s/step arithmetic, keeping the
   existing layer code.
2. **Porting the thinker onto ggml graphs** (as the other families do) — architecturally
   right long-term, substantially larger.

Starting with (1), gated on the seeded patch: one patch must reproduce the CPU result
within a *measured* delta (4.8e-06 is the CPU baseline, not a pass/fail line), reported
alongside wall time and peak VRAM. After any shared-engine change, re-run the `ming3_tts`
0.5B command and confirm it still says the intended sentence.
