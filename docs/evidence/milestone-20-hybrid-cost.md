# Hybrid MoE: the measured costs, and what they imply

The payoff check needs bf16 GEMMs against resident banks. Before building that, I measured
the cost that would decide whether the cheaper f32 route is even viable: extracting the
selected experts from the host banks.

```
8 experts     : compute 2.334 ms  transfer 0.471 ms  sum 2.805 ms
host slicing  : 9.019 ms to extract 8 experts x 3 matrices (f32)
hybrid note   : slicing + f32 upload per layer would cost 9.491 ms per MoE layer
verdict       : all selected experts PASS
```

(First-expert cuBLAS setup dominates the 2.334 ms; warm compute is ~0.045 ms per expert, so
eight experts are ~0.36 ms.)

## What the components imply for a step

Assembled arithmetic — **not a measured step** — from measured pieces:

| component | per step | source |
|---|---|---|
| KDA attention (18 layers, CPU) | 329 ms | profile |
| MLA attention (6 layers, CPU) | 41 ms | profile |
| dense MLP (CPU) | 19 ms | profile |
| MoE, hybrid f32 path | ~23 × 9.5 ms ≈ **219 ms** | slice 9.0 + transfer 0.47 + compute 0.36 per layer |
| **total** | **≈ 0.61 s** | vs 3.02 s measured CPU step |

So even the *cheap* hybrid (host slicing, f32 per expert, no resident-bank GEMMs) implies
roughly a **5x step improvement**. The host slicing dominates that MoE number — 9.0 of the
9.5 ms — which is why the next step is bf16 GEMMs against the resident banks: it removes the
slicing entirely and the MoE term collapses towards the ~10 ms of transfers plus ~8 ms of
compute, putting the step nearer 0.4 s.

## Honest status

- Every component above is measured; the **0.61 s step is arithmetic on those measurements**,
  not an executed hybrid step. Peak VRAM during inference is also still unmeasured (the
  14.52 GiB figure was a bulk upload).
- The decisive experiment remains: 23 MoE layers through the GPU path with the router on CPU,
  outputs compared value-by-value against the CPU step, warm-step time and peak VRAM against
  3.02 s.

## Recommendation

The numbers say the hybrid is worth one implementation push: a ~5x step improvement is
available with the code already written, and ~7x if the resident-bank bf16 path replaces host
slicing. That is enough to justify finishing the hybrid as the interim path before committing
to the ggml graph port — and the same measurement will show whether the remaining gap to
graph-level performance is worth that larger move.

If the assembled hybrid comes in near 0.6 s, the interim path is real. If it comes in near
3 s, the transfers and synchronisations are confirmed as the wall, and the graph port (with
its KDA/routing-on-CUDA check) is the answer.
