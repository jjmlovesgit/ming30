# Milestone 2, stage B — thinker reference forward (41 tokens)

Text: `The garden will remain open until sunset.`  ·  prompt ids: 41  ·  layers: 24

Capture points, as recorded by the reference run:

| capture | shape | dtype | mean | std | l2 |
|---|---|---|---|---|---|
| embedding | [1, 41, 1536] | torch.float32 | -0.00005 | 0.00996 | 2.501 |
| final_norm | [1, 41, 1536] | torch.float32 | -0.01692 | 0.45820 | 115.062 |
| stop_logits | [1, 2] | torch.float32 | 9.66446 | 4.23937 | 14.310 |

Per-layer hidden state (all `(1, 41, 1536)` float32):

| layer | mean | std | l2 |
|---|---|---|---|
| 0 | -0.00018 | 0.01043 | 2.618 |
| 1 | -0.00039 | 0.01300 | 3.265 |
| 2 | -0.00100 | 0.02017 | 5.067 |
| 3 | -0.00143 | 0.02232 | 5.613 |
| 4 | -0.00191 | 0.02591 | 6.519 |
| 5 | -0.00254 | 0.03252 | 8.187 |
| 6 | -0.00340 | 0.05590 | 14.055 |
| 7 | -0.00453 | 0.06395 | 16.089 |
| 8 | -0.00507 | 0.06751 | 16.989 |
| 9 | -0.00492 | 0.08012 | 20.145 |
| 10 | -0.00509 | 0.10965 | 27.547 |
| 11 | -0.00651 | 0.11594 | 29.140 |
| 12 | -0.00755 | 0.12909 | 32.451 |
| 13 | -0.00917 | 0.17889 | 44.950 |
| 14 | -0.01182 | 0.23573 | 59.229 |
| 15 | -0.01976 | 0.26507 | 66.704 |
| 16 | -0.02371 | 0.30133 | 75.852 |
| 17 | -0.02768 | 0.33241 | 83.707 |
| 18 | -0.03713 | 0.45606 | 114.826 |
| 19 | -0.04403 | 0.56122 | 141.269 |
| 20 | -0.04325 | 0.61018 | 153.507 |
| 21 | -0.03818 | 0.76145 | 191.324 |
| 22 | -0.02725 | 1.79038 | 449.344 |
| 23 | -0.02483 | 1.78996 | 449.230 |

Stop head:

- logits `[12.662140846252441, 6.666770935058594]` (shape `(1, 2)`)
- decision 0 (class 0 of 2) at the last prompt position

Router (layer 1, first token): `[104, 80, 109, 126, 13, 3, 102, 9]`

## What this is, and is not

- It is a fixed target for the C++ thinker: 24 hidden states, the final norm, and the stop decision,
  all captured from real weights with shapes and dtypes recorded.
- It is **not** yet validated against the released model. The KDA path is FLA's `naive_chunk_kda`
  (the chunk path the model runs) and the MLA path is transcribed from `modeling_bailing_moe_v3.py`,
  but the assembly — layer order, residual placement, MoE combine — has not been compared to a
  known-good output. Agreement on the 41-token case is the gate before longer inputs.
- One observation to check when the C++ side is compared: the residual-stream L2 grows from
  ~2.6 after layer 0 to ~449 after layer 23 (per-element RMS from ~0.010 to ~1.8). That could be
  genuine for this architecture, but a divergence in the C++ trace at a specific layer will need
  this curve to compare against.
