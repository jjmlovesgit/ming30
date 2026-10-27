# Stage B, capture 3 — layer 1 MoE agrees, routing included

The first MoE layer, compared in the order the protocol requires: routing ids, then
routing weights, then the hidden state.

```
PASS  embedding                max_abs=0.000e+00 rel_rmse=0.000e+00
PASS  layer_0                  max_abs=8.003e-05 rel_rmse=3.839e-04
PASS  routing_layer_1_ids      token rows matching=41/41
PASS  routing_layer_1_weights  max_abs=6.644e-04 rel_rmse=2.620e-04  (l2 6.659 both)
PASS  layer_1                  max_abs=1.913e-04 rel_rmse=4.731e-04  (l2 3.265 both)
```

## Reading the result

| capture | what it isolates | outcome |
|---|---|---|
| `routing_layer_1_ids` | sigmoid gate, `expert_bias`, group scores, top-4 groups, top-8 ordering, tie-breaking — compared at **every token position, in order** | **41/41 rows identical** |
| `routing_layer_1_weights` | gather from unbiased scores, normalisation, `routed_scaling_factor` 2.5 | 6.6e-04 abs |
| `layer_1` | the selected experts' gate/up/down, the shared expert, the residual | 1.9e-04 abs |

Because the ids agree, the diagnosis path for "ids differ but hidden agrees" never had
to be entered, and the expert matrix maths is confirmed rather than merely consistent
with a wrong router. Had the ids diverged, the next step would have been to diff router
scores and group selection *before* touching expert maths — which is exactly why the
routing capture is separate from the hidden capture.

Reference router for comparison, token 0: `[104, 80, 109, 126, 13, 3, 102, 9]`.

## What is now validated end to end

- embedding, layer-0 KDA + dense MLP + residuals (previous capture);
- layer-1 KDA + MoE: routing ids, routing weights, expert maths, shared expert, residual.

Two of 24 layers agree on the 41-token prompt, including the component most likely to
diverge first (routing).

## Next

Layer 2 is KDA + MoE again, so it adds no new code path — the useful next gate is
**layer 3, the first MLA layer**, which introduces LoRA q/kv projections, interleaved
RoPE on the 64-dim rotary slice, and the head-wise output gate.
