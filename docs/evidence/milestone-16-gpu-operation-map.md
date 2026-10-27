# Operation map: what a cuBLAS-only GPU path would leave on the CPU

The question is not whether cuBLAS can do the GEMMs (it does, verified to 1e-06), but how
much data and how many synchronisation points remain between them if only the GEMMs move to
the GPU. Sizes are per decode step (one patch) unless stated.

## Per-layer, KDA layers (18 of 24)

| operation | kind | today | data crossing if elementwise stays on CPU | size |
|---|---|---|---|---|
| `input_layernorm` | reduce + scale | CPU | hidden out and normalised back | 6 KB ×2 |
| q/k/v projections | GEMM | CPU | GPU-capable | — |
| short conv (k=4) | depthwise, needs 3 prior rows | CPU | q/k/v plus history | 24 KB + 73 KB |
| q/k L2 norm | per-head reduce | CPU | q, k | 16 KB |
| gate (`exp(A_log)·(f+dt)`) | elementwise + per-head params | CPU | f in, gate out | 16 KB |
| beta | elementwise | CPU | 64 B | — |
| **KDA recurrence** | sequential, state 16×128×128 | CPU | **state down and up** | **1 MB ×2** |
| output gate + gated RMS norm | reduce + elementwise | CPU | heads in/out | 16 KB |
| `o_proj` | GEMM | CPU | GPU-capable | — |

**The recurrence state is the expensive boundary: 1 MB per KDA layer, 18 MB both ways per
step for the KDA layers alone.** Leaving the recurrence on the CPU forces that transfer even
though the GEMMs are fast.

## MLA layers (6 of 24)

| operation | data crossing if attention stays on CPU | size |
|---|---|---|
| q_a/kv_a projections | GPU-capable | — |
| q/kv layernorms (256, 512) | 3 KB | small |
| RoPE | 20 KB | small |
| **attention over cached KV** | keys+values for every prior position | ~20 KB/token/layer → ~1.4 MB per layer at 70 tokens |
| head-wise gate, `dense` | 6 KB | small |

## Between layers and around the loops

| stage | if left on CPU | size per step |
|---|---|---|
| MoE router (sigmoid, bias, group top-k sort) | 128 logits down, 8 indices up | 512 B + 32 B, but a **sync point per layer per step** (23 of them) |
| MoE expert GEMMs + shared expert | GPU-capable if experts are resident | — |
| residual adds | 6 KB per layer | small |
| Aggregator (feedback) | 8 blocks, mostly GEMM + attention | small |
| Flow DiT (10 Euler steps × 16 blocks, 3 tokens) | GEMMs GPU-capable; noise and timesteps CPU-side | small, but 10× per patch |
| VAE decoder (24-layer Qwen2, 4 tokens/patch) | GEMMs GPU-capable | small |
| **iSTFT** | FFT today on CPU | spectrum 1765×4 complex ≈ 56 KB |

## What the map says

1. **Transfer volume is dominated by two things**: KDA recurrent state (~18 MB/step round
   trip) and MLA KV if attention stays on CPU (~8 MB/step). Everything else is kilobytes.
2. **Synchronisation count is dominated by the MoE router and the per-layer norms**: roughly
   24 layers × 4–6 boundaries ≈ 100–150 host round trips per step, each with launch/sync
   overhead in the tens of microseconds — a few milliseconds per step, not seconds.
3. **So a hybrid is not obviously disqualified by data volume**, but it is disqualified as a
   *bounded* change: to avoid the recurrence and attention transfers you must move KDA and
   MLA onto the GPU (custom kernels), and to avoid ~100 syncs you must keep the whole layer
   resident. That is most of a GPU port, not a cuBLAS shim.

## Decision framing

| option | what it requires | verdict |
|---|---|---|
| cuBLAS for GEMMs only, everything else CPU | accept 18 MB + 8 MB transfers and ~100–150 syncs per step | bounded, but the transfers and syncs are exactly what we set out to avoid |
| cuBLAS + custom KDA/MLA/norm/router kernels, fully resident | CUDA kernels for recurrence, conv, norms, top-k, iSTFT | this *is* a GPU port; roughly what ggml already provides |
| **audio.cpp ggml graph backend** | express the thinker as a ggml graph; reuse existing CUDA kernels for norms, attention, MoE, and the VAE | most likely the right long-term route, and the families already do this |

The measurement that decides it is still the seeded patch: if a **hybrid** seeded patch
(GEMMs on GPU, elementwise on CPU, accepting the transfers) comes in well under the current
2.31 s/step CPU figure, the cheap route has value as an interim; if it lands close to CPU
time, the transfers and syncs are confirmed as the wall and the ggml graph port is the
answer.

## Corrections carried forward

- **BF16 resident weights are plausible**: 16.77 GB of weights leaves room on a 32 GB card
  for activations, KV state and workspace. Peak VRAM must be measured, not assumed.
- The 72 MiB / 12 MiB figures from the microbenchmark say nothing about whole-model residency.
