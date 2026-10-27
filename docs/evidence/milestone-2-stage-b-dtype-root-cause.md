# Root cause found: F32 parameters were being loaded as BF16

The staged KDA trace localised the divergence to the gate, and the gate pointed at a
dtype problem. It was not the chunk-vs-recurrence ordering at all.

## The trace, before and after

Single KDA layer (layer 2), C++ seeded with the exact Python layer-1 hidden state:

| stage | before | after |
|---|---|---|
| q_conv (conv + SiLU) | 5.7e-07 | 5.7e-07 |
| q_norm (per-head L2) | 5.6e-07 | 5.6e-07 |
| **gate** | **2.9e-03 rel / 3.35e-02 max** | **7.8e-07 rel / 1.74e-05 max** |
| beta | 1.3e-07 | 1.3e-07 |
| **delta-rule output** | **1.15e-03 rel / 4.78e-05 max** | **5.6e-07 rel / 2.98e-08 max** |
| attn (after o_proj) | 9.5e-04 rel / 2.43e-04 max | 1.0e-06 rel / 1.27e-07 max |

## The bug

The package keeps **678 tensors in F32**: `attention.A_log`, `attention.dt_bias` and
`mlp.gate.expert_bias` on every layer. The C++ loader requested
`TensorStorageType::BF16` for every tensor uniformly, so the tensor source converted
those F32 parameters down to BF16 before use.

Consequences:

- `A_log` and `dt_bias` feed the lower-bounded gate
  (`-5·sigmoid(exp(A_log)·(f+dt_bias))`), so bf16 rounding perturbed the decay of every
  KDA layer — the source of the ~2e-04 per-layer drift.
- `expert_bias` feeds routing selection (scores + bias), so its rounding shifted expert
  cutoffs by roughly the same magnitude as the near-tie margins we had been recording.
  That is why 28 token-layers flipped.

Fix: load each tensor in its **stored** dtype (`require_metadata(name).dtype` → F32 or
BF16), rather than assuming BF16.

## Full 24-layer sweep after the fix

```
layer kind   attn_rel  hidden_rel  hidden_max  pre_router  scores_max  flips
    0  KDA    8.4e-07     1.1e-06     6.4e-07           -           -      -
    1  KDA    1.1e-06     1.1e-06     6.0e-07     1.1e-06     6.1e-07      0
   ...
   23  MLA    1.1e-06     9.6e-07     8.8e-05     1.2e-06     1.4e-06      0

final_norm      rel_rmse=1.369e-06 max_abs=8.464e-06
stop_logits     reference=[12.66214, 6.66677] cpp=[12.66215, 6.66677] max_abs=6.676e-06
stop_decision   SAME
expert-set flips total: 0 of 984
```

| measure | before | after |
|---|---|---|
| per-layer hidden rel error | 2.4e-03 – 6.3e-03 | 6.2e-07 – 1.1e-06 |
| `final_norm` rel RMSE | 4.53e-02 | **1.37e-06** |
| stop logits max abs | 5.43e-03 | 6.68e-06 |
| expert-set flips | 28 | **0** |

## What this means

1. **`chunk_kda` does not need porting.** The float32 recurrence matches FLA's chunk path
   to ~1e-06 per layer on real weights. The chunk-vs-recurrence hypothesis was wrong; the
   drift had a mundane cause.
2. The earlier "boundary event" flips were not numerical fate at 1e-4 margins — they were
   the bias being rounded. With the bias loaded correctly, no cutoff is crossed.
3. Stage B is now a genuine pass: embedding, attention (KDA and MLA), MoE routing,
   dense MLP, final norm and stop head all agree to ~1e-06 across 24 layers.

## Success measure met

The change was made against a pre-declared measure — fewer flips and smaller
`final_norm` error — and it delivered: 28 → 0 flips, 4.5% → 1.4e-06.

Flow work can proceed with the thinker as a solid base. The seeded-input and stage-trace
machinery stays in place; it is what turned an ambiguous 4.5% into a one-line loader fix.
