# cuBLAS step 1 — infrastructure in place, benchmark harness not yet trustworthy

## What is now built and verified

- CUDA and cuBLAS are available in this tree (`CUDA_cublas_LIBRARY`,
  `CUDA_cublasLt_LIBRARY` resolved; `.cu` runtimes are an established pattern at
  `CMakeLists.txt` L2936–2951).
- A family-local CUDA source (`src/models/ming3_tts_tiny/gpu_bench.cu`) is compiled into
  `engine_model_ming3_tts_tiny` under `ENGINE_ENABLE_CUDA`, linked against
  `CUDA::cudart CUDA::cublas`, with `ENGINE_HAS_MING_GPU_BENCH` defined. No shared engine
  files were modified — the block is additive and model-local.
- The benchmark runs: cuBLAS initialises, 72 MiB of layer weights upload in 5.6 ms,
  activations transfer in 0.057 ms per call, and the seven projection GEMMs execute at the
  one-token decode shape.

## Why the numbers are withheld

The harness's own consistency check fails:

```
|q_w| rms = 0.020817          (weights load correctly)
|cpu_out| rms = 1.51e+17      (should be O(1))
|gpu_out| rms = 8.53e+16
```

The CPU reference chain — the same seven products with our own `linear()` — produces
values around 1e17, so both the CPU and GPU figures are nonsense. Until that is fixed, the
measured times (1.5–3.9 ms GPU vs 6.6 ms CPU per layer) are **not** evidence about
cuBLAS performance and must not be used to decide anything.

Most likely a harness mistake in the CPU block (the buffers I feed forward are not the
outputs I think they are), not a model or cuBLAS problem: the weights are sane and the
upload path is exercised.

## What the benchmark will report once fixed

Per the agreed measurement split, for one KDA layer at the decode shape:

- weight upload (one-time) and peak VRAM delta,
- per-call activation transfer,
- per-call GEMM compute,
- per-call total, against the CPU time for the same products,
- and a numeric cross-check (checksum or max-abs) so the comparison is meaningful.

Then the same harness extends to a full seeded patch: prefill on GPU, one cached step,
compare against the CPU result (4.8e-06 baseline, measured not asserted), and report time
and peak VRAM.

## Honest status of the GPU step

| item | state |
|---|---|
| CUDA/cuBLAS available and linked | done |
| family-local `.cu`, no shared-engine edits | done |
| weights upload + transfer measured | done (5.6 ms / 0.057 ms / 72 MiB) |
| GEMM timing meaningful | **no — harness check fails** |
| seeded-patch GPU comparison | not started |

Next action is to fix the harness's CPU reference and re-run; only then does a cuBLAS
conclusion (or a choice for or against the ggml graph port) rest on data.
