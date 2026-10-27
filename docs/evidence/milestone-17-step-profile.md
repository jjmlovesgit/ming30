# Where the 2.31 s/step actually goes — MoE is 87%

Measured with per-stage timers on one warm decode step (seed patch, prefill 57 + 1 step):

```
profile : kda 329.3 ms | mla 40.9 ms | moe 2633.2 ms | dense 18.8 ms | total 3022.1 ms
```

| stage | layers | total ms | ms/layer | share |
|---|---|---|---|---|
| MoE (routed experts + shared + router) | 23 | 2633.2 | 114.4 | **87.1%** |
| KDA attention (conv, norm, gate, recurrence) | 18 | 329.3 | 18.3 | 10.9% |
| MLA attention | 6 | 40.9 | 6.8 | 1.4% |
| dense MLP (layer 0) | 1 | 18.8 | 18.8 | 0.6% |

(The two larger profile lines are the prefill and full passes; the step is the third.)

## What this changes about the plan

The operation map said the *transfers* are dominated by KDA state. The profile says the
*time* is dominated by MoE. Those are different questions, and the profile is the one that
decides where to spend effort:

- **MoE is 87% of a step**, and its work is dominated by expert GEMMs — exactly what cuBLAS
  does well. A MoE layer has only two natural host boundaries (router input before, expert
  outputs after), so moving it to GPU costs ~2 syncs per layer: 23 × 2 × 0.073 ms ≈
  **3.4 ms of transfer per step**, against 2633 ms of compute.
- **KDA is 11%**, and its recurrence state is the expensive *transfer*. Leaving KDA on the
  CPU costs the KDA compute (329 ms) but avoids writing custom recurrence kernels.
- **MLA is 1.4%** — not worth touching first.

So the hybrid has a much better shape than the map alone suggested: put the **MoE expert
GEMMs on the GPU**, keep KDA/MLA/norms on the CPU, and accept ~3 ms of transfers plus the
syncs. Even if the GPU only halves the MoE time, the step falls from ~3.0 s to ~1.7 s, and if
it takes MoE to a third, to ~1.2 s.

## What this does not establish

- The hybrid has not been run end to end; these are stage timings of the current CPU path
  plus a transfer budget computed from the measured 0.073 ms per round trip.
- Peak VRAM is unmeasured, and MoE residency is the hard part: all 23 layers' expert banks
  as bf16 are ~13.9 GB, so a device-resident MoE needs the bf16 budget confirmed.
- CUDA-side synchronisation has not been included in any hybrid timing yet, because no
  hybrid exists.

## Next

1. GPU-resident MoE experts (bf16), router on CPU, expert outputs copied back per layer;
   compare the seeded step against the CPU result and report warm-step time, transfer bytes,
   peak VRAM, and total wall time against the 3.02 s warm-step baseline.
2. Only if that lands close to CPU time is the ggml graph route the better answer — and if it
   goes that way, verifying that KDA and routing actually execute on CUDA (rather than
   silently falling back) becomes the first check, as noted.
