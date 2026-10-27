# Stage B, captures 4–5 — layer 2 clean, layer 3 has a knife-edge expert swap

## Layer 2 (KDA + MoE, no new code path) — verified

```
PASS  routing_layer_2_ids      expert sets matching=41/41  order differs on 4/41 token(s), same experts
PASS  routing_layer_2_weights  max_abs by expert id=7.517e-04 (token 35)
PASS  layer_2                  max_abs=2.224e-04 rel_rmse=4.239e-04  (l2 5.067 vs 5.068)
```

The 4 order differences are **near-ties**: the same eight experts, with the last two
slots swapped (`…102,105` vs `…105,102`). The routed sum is a sum over slots, so the
model is invariant to that order — confirmed by layer 2's hidden state agreeing to
2.2e-04. The comparer was changed to treat the expert **set** as the failure condition
and report ordering separately, and to match weights **by expert id** rather than by
slot; otherwise a harmless swap reads as a weight error.

## Layer 3 (first MLA layer) — hidden agrees, one expert flips

```
FAIL  routing_layer_3_ids      expert sets matching=40/41
      token 35: reference [8,9,11,67,113,117,118,126] vs cpp [8,9,11,86,113,117,118,126]
PASS  routing_layer_3_weights  max_abs by expert id=1.327e-03 (token 35)
PASS  layer_3                  max_abs=1.866e-03 rel_rmse=3.946e-03  (l2 5.613 both)
```

Token 35, slot 8 — the last selected expert:

| | expert | weight |
|---|---|---|
| reference | 67 | 0.15073 |
| C++ | 86 | 0.14503 |

The other seven experts agree, and their weights differ by ~1e-3. The hidden state
differs by `1.87e-03` at that token and `4.1e-05` (median) elsewhere; excluding token
35 the layer-3 relative RMSE is `3.8e-04` — in line with the KDA layers.

### Reading it

The 8th slot is a near-tie between two experts whose normalised mass differs by
~0.4 percentage points of the router total. A ~1e-3 shift in the layer-3 attention
output — which is the router's input — is enough to swap which one is selected. The
layer-3 hidden state then differs at that token because a different expert contributes.

This is consistent with the MLA path carrying larger numerical differences than KDA
(≈1e-3 vs ≈2e-4), not with a routing logic error: KDA layers 1 and 2 route identically,
and layer 3's other seven experts and their weights agree.

## What to do next, in order

1. **Isolate the MLA path from the router**: capture the attention-only output
   (`attn_layer_3`) on both sides and compare it directly. If that splits ~1e-3, the
   router flip is a downstream symptom and the fix belongs in MLA.
2. Candidate MLA causes, in likelihood order: softmax accumulation order (reference
   uses `F.scaled_dot_product_attention`, this side a naive loop), RoPE precision on the
   64-dim rotary slice, gate placement relative to the reshape.
3. Only after MLA agrees to KDA-level error should the layer-3 routing comparison be
   expected to pass on every token.

Coverage so far: embedding, layer 0 (KDA + dense MLP), layer 1 (KDA + MoE), layer 2
(KDA + MoE), layer 3 (MLA + MoE, hidden agrees, one expert swap at a near-tie).
