# GPU MoE expert test — correct to 1e-7, and fast where it counts

Family-local CUDA path for one MoE layer's routed experts (layer 1), expert ids fixed from
the reference run (`104, 80, 109, 126, 13, 3, 102, 9`), each expert computed entirely on the
device: three GEMMs plus the `silu(gate)·up` elementwise done by a device kernel, so there is
no host round trip between the GEMMs.

```
expert 104 : max_abs=6.75e-08 rel=1.18e-06 0/1536 PASS  compute 2.172 ms  xfer 0.058 ms
expert  80 : max_abs=5.40e-08 rel=1.11e-06 0/1536 PASS  compute 0.064 ms  xfer 0.064 ms
expert 109 : max_abs=4.75e-08 rel=1.02e-06 0/1536 PASS  compute 0.043 ms  xfer 0.058 ms
expert 126 : max_abs=5.31e-08 rel=1.20e-06 0/1536 PASS  compute 0.048 ms  xfer 0.073 ms
expert  13 : max_abs=5.87e-08 rel=1.16e-06 0/1536 PASS  compute 0.050 ms  xfer 0.068 ms
expert   3 : max_abs=7.08e-08 rel=1.14e-06 0/1536 PASS  compute 0.042 ms  xfer 0.058 ms
expert 102 : max_abs=5.77e-08 rel=1.26e-06 0/1536 PASS  compute 0.044 ms  xfer 0.062 ms
expert   9 : max_abs=5.17e-08 rel=1.21e-06 0/1536 PASS  compute 0.046 ms  xfer 0.064 ms

combined routed sum : max_abs=2.05e-08 rel=1.16e-06
8 experts           : compute 2.510 ms  transfer 0.505 ms  sum 3.015 ms
verdict             : all selected experts PASS
```

## Reading the timings

- **Warm expert compute is ~0.045 ms** (the 2.17 ms first entry is cuBLAS/kernel setup), so
  eight experts cost ≈ **0.36 ms of compute per MoE layer**.
- **Transfer is 0.058–0.073 ms per expert**, ~0.5 ms for eight. Per layer that is larger than
  the compute, and across 23 layers ≈ 11.5 ms per step — small against the 3.02 s step, but it
  means the transfer, not the maths, is the first thing to optimise (keep the activation on
  the device between experts and across layers).
- Against the measured CPU MoE cost of **114.4 ms per layer**, the GPU path is roughly
  **two orders of magnitude faster per layer** in warm steady state.
- VRAM here is 12 MiB per expert (float32 weights for one expert plus buffers) — this says
  nothing about residency of all banks, which is the ~13.9 GB bf16 question still open.

## What this establishes and what it does not

Established: the expert maths on GPU is correct (1e-7), the silu-gate elementwise belongs on
the device (no intermediate transfers), and warm per-expert cost is small.

Not established: an end-to-end hybrid step. That needs (a) all 23 layers' banks resident as
bf16 (≈13.9 GB) with peak VRAM measured, (b) the router kept on the CPU as planned with only
its 8 indices crossing, and (c) the expert outputs combined and added to the residual. No
hybrid step time exists yet; the 3.02 s step remains the baseline to beat.

## Next

1. Residency probe: upload all 23 layers' expert banks as bf16 and report peak VRAM
   (expect ≈13.9 GB) — this decides whether full MoE residency is viable at all.
2. Then a hybrid step: router on CPU, experts on GPU, residual and the rest unchanged,
   compared value-by-value against the CPU step, with warm-step time, transfer bytes and
   peak VRAM against the 3.02 s baseline.
