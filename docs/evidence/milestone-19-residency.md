# BF16 expert residency: 12.94 GiB of banks, 14.52 GiB peak, 17.3 GiB left

Streamed every MoE layer's three expert banks onto the device as the raw BF16 bytes they are
stored as — **no float32 copy is materialised on the device**, which was the explicit
requirement.

```
gpu residency : 69 BF16 bank tensors streamed, total 13248 MiB (12.94 GiB)
  peak VRAM used : 14865 MiB (14.52 GiB)   free after: 17742 MiB
```

## What this settles

- **Full MoE residency fits.** All 23 layers × 3 banks (128 experts each) occupy 12.94 GiB as
  BF16. Peak during the upload, including staging and allocator overhead, is 14.52 GiB.
- **Overhead above the raw banks is ~1.6 GiB**, consistent with driver context, staging and
  allocator round-up — worth knowing, because at float32 the same banks would be ~25.9 GiB
  and would not have left room for the rest of the model.
- **17.3 GiB remains** for everything else: the dense (non-expert) thinker weights, KV cache,
  activations, the Aggregator, the flow model and the VAE decoder. That is the budget any
  hybrid design has to live within, and it is the first time it has been measured rather than
  estimated.

## What this does not settle

- Residency is proven, but **no hybrid step has been run**. The comparison that matters is
  still outstanding: all 23 MoE layers with the router on CPU, expert outputs compared
  value-by-value against the CPU step, warm-step wall time against the **3.02 s** baseline,
  plus transfer bytes and peak VRAM during inference (not during a bulk upload).
- Transfer is expected to exceed compute in this design (0.5 ms vs 0.36 ms per layer in the
  expert test), so the hybrid's step time will be dominated by how the expert outputs are
  combined and returned — the next thing to measure, and a candidate for keeping the
  combination on the device.

## Next

1. Hybrid step across all 23 MoE layers, values compared against the CPU step.
2. Report warm-step time, transfer bytes and peak VRAM against 3.02 s.
3. If the hybrid lands close to CPU time, the ggml graph route becomes the answer — and its
   first check must be that KDA and routing genuinely execute on CUDA rather than silently
   falling back to CPU.
