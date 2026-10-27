# Stage B, capture 1 — embedding agrees exactly

The first cross-implementation comparison point. No model maths involved: prompt
token ids → embedding rows → bf16→f32 conversion. A mismatch here would have been
tokenizer or weight loading.

```
reference: out/reference/thinker/thinker-41.npz
cpp      : out/reference/thinker-cpp (1 capture)

PASS  embedding      max_abs=0.000e+00 rel_rmse=0.000e+00 l2_ref=2.501 l2_cpp=2.501
```

Bit-identical: `max_abs = 0.0`. That validates, end to end:

- the fallback `BailingMoe` tokenizer against `transformers` on this prompt;
- the 41 prompt ids the C++ family builds (already verified as stage A);
- GGUF weight loading of `model.word_embeddings.weight` from the BF16 package;
- the bf16 → float32 widening used on both sides.

## Naming rule discovered here

The package stores logical names with **one leading `model.` stripped**, so the HF
name and the tensor name to request differ:

| HF / safetensors | package logical name |
|---|---|
| `model.model.word_embeddings.weight` | `model.word_embeddings.weight` |
| `model.model.norm.weight` | `model.norm.weight` |
| `model.model.layers.N.attention.q_proj.weight` | `model.layers.N.attention.q_proj.weight` |
| `model.lm_head.weight` | `lm_head.weight` |

The first attempt requested the HF name and failed with `missing tensor:
model.model.word_embeddings.weight` — worth knowing before writing the rest of the
loader.

## Protocol for the remaining captures

Agreed with the coordinator:

1. L2 is a warning signal, **not** a pass condition — different vectors can share one.
2. Per-layer comparison is `max_abs` and relative RMSE on full values.
3. At MoE layers, compare expert id sets and routing weights as well.
4. The first substantial mismatch names the component to inspect.
5. The plain-torch run stays a *candidate* reference until its assembly receives an
   independent check.

`reference/drafts/compare_thinker.py` implements this: it loads any number of
captures from a C++ dump directory (`manifest.json` + raw little-endian float32),
compares each against the reference `.npz`, prints `max_abs`/`rel_rmse`/L2 and names
the first failure.

The C++ dump is opt-in via `MING3_TINY_DUMP_DIR`, so normal runs are unaffected.
