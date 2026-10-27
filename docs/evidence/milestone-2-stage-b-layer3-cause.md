# Layer 3 cause: inherited KDA drift at a 7.6e-05 router margin — MLA is exact

Three experiments, as directed: attention-only capture, seeded input, and the
pre-router/scores comparison at token 35.

## E2 — seeded from the exact Python layer-2 hidden state

C++ starts at layer 3 with the Python layer-2 tensor as its input, removing everything
inherited from layers 0–2.

```
PASS  attn_out_layer_3      max_abs=3.427e-07 rel_rmse=1.236e-06
PASS  pre_router_layer_3    max_abs=8.047e-07 rel_rmse=6.450e-07
PASS  router_scores_layer_3 max_abs=2.980e-07 rel_rmse=1.865e-06
PASS  routing_layer_3_ids   expert sets matching=41/41
PASS  routing_layer_3_weights max_abs by expert id=2.384e-06
PASS  layer_3               max_abs=2.980e-07 rel_rmse=6.425e-07
```

**The MLA implementation is exact to float32 rounding.** With the same input, it
reproduces attention output, pre-router state, router scores, routing and hidden state
to ~3e-07. The token-35 was not an MLA bug.

## E1 — normal run: attention drift per layer

```
attn_out_layer_0   max_abs=1.468e-04   (KDA)
attn_out_layer_1   max_abs=2.045e-04   (KDA)
attn_out_layer_2   max_abs=2.744e-04   (KDA)
attn_out_layer_3   max_abs=5.300e-05   (MLA)
pre_router_layer_3 max_abs=1.156e-03   (RMS norm amplifies the drift ~4x)
router_scores_layer_3 max_abs=3.035e-04
```

The KDA layers drift 1.5–2.7e-04 per layer (recurrence vs FLA's chunk ordering); MLA
drifts 5e-05. Everything downstream is fed by that KDA drift, not by MLA.

## The token-35 cutoff, in biased-score terms

Selection uses `sigmoid(logits) + expert_bias`; the weights reported are the unbiased
scores, which is why an unbiased ranking does not show the cutoff.

| | biased score(67) | biased score(86) | 8th-slot margin |
|---|---|---|---|
| reference | 0.055251 | 0.055175 | **+7.59e-05** (67 wins) |
| C++ normal | 0.055177 | 0.055187 | **+1.03e-05** (86 wins) |

Score drift between implementations at those two experts: **7.4e-05 and 1.2e-05** —
the same order as the 7.6e-05 margin that decides the slot. The flip is fully explained:
the router cutoff sits inside the numerical drift of the KDA layers.

## What this means for the port

1. MLA: verified, no action.
2. KDA: the residual difference is the chunk-vs-recurrence ordering, ~2e-4 per layer.
   It is not an algorithmic error — the two forms are the same recurrence.
3. Routing: any slot whose biased margin is below ~1e-4 can flip. This is inherent to
   float32 disagreement between two correct implementations, and would also flip
   between the reference's own SDPA variants on different hardware.
4. Consequence at this token: hidden state differs by 1.9e-03 there, 4e-05 median, and
   the layer's relative RMSE is 3.9e-03.

**Recommendation:** treat a routing-set difference as expected when the biased margin is
below the observed drift (~1e-4), and keep the hidden-state comparison as the real
signal. If tighter agreement is later needed, the fix is to implement the chunk
algorithm in C++ — *not* to raise precision, because the difference is operation order,
not rounding.

Coverage: embedding, layers 0–3 (KDA ×3 + MLA), routing at layers 1–3, attention-only
and pre-router captures at every layer, plus a reusable seeded-input mode.
