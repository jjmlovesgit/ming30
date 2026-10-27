# Stage B, capture 2 — layer 0 agrees

Layer 0 is the isolation point: KDA attention, a dense MLP, and the residual order,
with no MoE routing involved.

```
PASS  embedding      max_abs=0.000e+00 rel_rmse=0.000e+00 l2_ref=2.501 l2_cpp=2.501
PASS  layer_0        max_abs=8.003e-05 rel_rmse=3.839e-04 l2_ref=2.618 l2_cpp=2.618
```

`max_abs = 8.0e-05`, relative RMSE `3.8e-04`, and matching L2. This is float32
ordering noise, not a divergence: the C++ side runs the **recurrent** KDA form while
the reference runs FLA's **chunk** path, and they land within 1e-4 relative — which is
the chunk/recurrent equivalence measured earlier (`5.9e-09` on random inputs), now
confirmed on real weights.

## What this validates

| component | status |
|---|---|
| embedding gather + bf16→f32 | exact |
| input RMS norm | agrees |
| q/k/v projection, causal depthwise conv (k=4) + SiLU | agrees |
| q/k per-head L2 normalisation | agrees |
| lower-bounded gate `-5·sigmoid(exp(A_log)·(f+dt_bias))` | agrees |
| KDA recurrence (decay, delta update, q·S, 1/√K scale) | agrees |
| gated RMS norm from `g_proj` | agrees |
| dense MLP `down(silu(gate)·up)` | agrees |
| residual order (attn → MLP) | agrees |

A mismatch here would have been attributable to one of those, with no routing to
confound it. None of them diverged.

## One alignment made before comparing

The first C++ draft used `x / (sqrt(mean) + eps)`; the reference uses
`x · rsqrt(mean + eps)`. The difference is ~1e-5 relative, small but avoidable noise,
so the C++ side was matched to the reference formula *before* this run rather than
being left to absorb into the tolerance.

## Next

Layer 1 — the first MoE layer. Compare the full hidden values **and** the selected
expert ids and routing weights: a router divergence that does not show up in the
hidden state yet would still poison every later layer.
