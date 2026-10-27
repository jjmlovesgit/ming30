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

## Name mapping, verified against GGUF metadata

Not a blanket prefix rule. The map is derived from the package's own
`audiocpp.tensor_names` and checked in both directions by
`reference/drafts/build_name_map.py`, which writes `reference/name-map.json`.
Verification: **2,141 package tensors, 10,285 HF tensors, 0 unmapped names,
0 missing sources, 0 unconsumed HF tensors** — exact coverage both ways.

| family | package tensor | source | count |
|---|---|---|---|
| embedding | `model.word_embeddings.weight` | `model.model.word_embeddings.weight` | 1 |
| final_norm | `model.norm.weight` | `model.model.norm.weight` | 1 |
| **lm_head** | `lm_head.weight` | `model.lm_head.weight` | 1 |
| backbone_layer | `model.layers.*` | `model.model.layers.*` (drop exactly one `model.`) | 448 |
| expert_bank | `model.layers.N.mlp.experts.{gate,up,down}_proj.weight` | **128** × `model.model.layers.N.mlp.experts.E.{...}.weight` stacked on the expert axis | 69 banks ← 8,832 |
| vae | `audio.encoder.*`, `audio.decoder.*` | **unchanged** | 643 |
| talker | `flowloss.*`, `linear_proj_audio.*`, `spk_head*`, `stop_head*` | **unchanged** | 359 |
| speaker_campplus | `speaker_encoder.*` | **derived from `campplus.onnx`**, not a safetensors tensor | 619 |

Three different paths exist, which is why one rule cannot cover them: the backbone
drops a prefix, `lm_head` is renamed while the components keep HF names verbatim,
the speaker tensors have no safetensors source at all, and routed experts are
many-to-one.

The first attempt requested the HF name `model.model.word_embeddings.weight` and
failed with `missing tensor` — the map now makes that class of error impossible
before a layer is written.

## Rerun after the mapping work

```
PASS  embedding      max_abs=0.000e+00 rel_rmse=0.000e+00 l2_ref=2.501 l2_cpp=2.501
```

Unchanged: bit-identical over the 41 rows, against a freshly regenerated dump.

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
