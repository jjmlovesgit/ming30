# cuBLAS harness corrected — one projection verified value by value

Reduced to a single projection exactly as directed, then verified against `linear()`:

```
gpu gemm        : one projection 2048x1536, 20 iterations
  weight upload :  1.032 ms   (12 MiB VRAM for this matrix)
  xfer per call :  0.074 ms
  gemm per call :  1.838 ms
  verification  : max_abs=1.490e-06 rel_rmse=6.710e-07 mismatched=0/2048  PASS
  values        : cpu[0..3]=0.372101 0.793413 -0.108548 -0.180659
                  gpu[0..3]=0.372101 0.793413 -0.108548 -0.180659
```

Every value compared, not a checksum. A transposed, mis-strided or mis-shaped GEMM could
not pass this: the input is deterministic, finite and asymmetric
(`0.5·sin(0.01·i) + 0.1·(i mod 7) − 0.3`).

## What the previous failure actually was

The old harness chained seven products through one `buffer`, feeding the *last projection's
output* into the next operator regardless of which one it was meant to be, and reported a
checksum of accumulated garbage (1e17). The earlier "1.5–3.9 ms GPU vs 6.6 ms CPU" numbers
were measuring nonsense and are withdrawn. The cuBLAS call itself was fine; my buffer
plumbing was not.

Orientation is now explicitly documented in the kernel: row-major `W[out, in]` is passed to
`cublasSgemm` as `CUBLAS_OP_T` with `A` interpreted column-major `[in, out]`, `m = out`,
`n = 1`, `k = in`, `lda = in`, `ldc = out`.

## What is still unmeasured

- **Full-model VRAM.** The 12 MiB above is one 2048×1536 matrix. At float32 the thinker's
  weights alone are ~33 GB, over the 32 GB card, so the real port needs bf16 storage
  (~16.8 GB) plus activations and KV state. Nothing here says the full model fits.
- **The seven projections, chained correctly** — next step, adding one at a time with the
  same per-value check after each.
- Transfer is cheap at this shape (0.074 ms), which matters: it supports keeping weights
  resident and only moving activations.
- The GEMM time (1.838 ms) is dominated by launch/synchronisation at `n = 1`, not by
  arithmetic (~6.3 M MACs). Batching or fusing layers is what would move this number; a
  single GEMM at this shape cannot show a 5090's throughput.

## Next, in order

1. Add projections back one at a time (q, k, v, f, g, b, o), each verified before the
   next, with the correct buffer feeding each operator.
2. Then the seeded patch on GPU: prefill and one cached step, compared against the CPU
   result and reported with wall time and peak VRAM.
3. Full-model residency (bf16, ~16.8 GB) is a separate question that must be measured
   before claiming the GPU path is viable.
