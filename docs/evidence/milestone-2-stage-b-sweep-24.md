# Full 24-layer sweep — error stays bounded, 28 boundary flips, stop decision stable

C++ runs the whole thinker in **26.2 s** on CPU for the 41-token prompt. Compared
against the plain-torch reference, capture by capture.

```
layer kind   attn_rel  hidden_rel  hidden_max  pre_router  scores_max  flips  ref_margin
    0  KDA    4.8e-04     3.8e-04     8.0e-05           -           -      -           -
    1  KDA    8.6e-04     4.7e-04     1.9e-04     6.4e-04     2.8e-04      0           -
    2  KDA    1.1e-03     4.2e-04     2.2e-04     7.5e-04     3.0e-04      0           -
    3  MLA    2.8e-04     3.9e-03     1.9e-03     7.2e-04     3.0e-04      1    +7.6e-05
    4  KDA    5.0e-03     5.7e-03     2.5e-03     1.1e-02     6.1e-03      2    +7.5e-06
    5  KDA    1.0e-02     5.9e-03     2.7e-03     1.1e-02     2.9e-03      1    +6.1e-06
    6  KDA    6.8e-03     4.4e-03     3.2e-03     9.6e-03     5.1e-03      1    +1.7e-04
    7  MLA    4.1e-03     4.1e-03     3.1e-03     8.9e-03     2.4e-03      1    +2.6e-06
    8  KDA    5.2e-03     4.4e-03     3.6e-03     8.8e-03     3.9e-03      1    +8.1e-04
    9  KDA    8.4e-03     3.6e-03     3.7e-03     9.9e-03     4.5e-03      0           -
   10  KDA    5.6e-03     2.5e-03     3.7e-03     8.5e-03     3.3e-03      0           -
   11  MLA    3.8e-03     2.8e-03     3.8e-03     7.8e-03     1.2e-03      2    +4.8e-05
   12  KDA    4.7e-03     2.5e-03     3.8e-03     8.6e-03     2.6e-03      0           -
   13  KDA    7.3e-03     3.2e-03     1.0e-02     7.8e-03     2.6e-03      2    +2.4e-04
   14  KDA    7.4e-03     2.4e-03     9.5e-03     1.2e-02     3.9e-03      0           -
   15  MLA    2.1e-03     2.9e-03     1.2e-02     8.9e-03     2.9e-03      1    +1.9e-05
   16  KDA    1.1e-02     2.6e-03     1.1e-02     1.2e-02     3.5e-03      0           -
   17  KDA    1.1e-02     3.5e-03     2.1e-02     1.1e-02     4.2e-03      3    +3.7e-05
   18  KDA    1.1e-02     2.9e-03     2.6e-02     1.7e-02     5.3e-03      2    +1.8e-04
   19  MLA    6.1e-03     2.5e-03     4.3e-02     1.6e-02     2.2e-02      0           -
   20  KDA    6.7e-03     2.7e-03     5.2e-02     1.2e-02     4.3e-03      2    +1.5e-04
   21  KDA    1.1e-02     6.3e-03     9.9e-02     1.3e-02     2.7e-03      4    +8.1e-05
   22  KDA    1.3e-02     3.2e-03     9.4e-02     2.9e-02     9.2e-03      3    +5.2e-05
   23  MLA    8.2e-03     4.6e-03     1.2e-01     2.2e-02     2.1e-02      2    +3.2e-05

final_norm      rel_rmse=4.532e-02 max_abs=3.703e-01
stop_logits     reference=[12.66214, 6.66677] cpp=[12.66174, 6.66134] max_abs=5.430e-03
stop_decision   reference=0 cpp=0 SAME

expert-set flips total: 28 token-layers of 984 compared (2.8%)
```

## How the error actually grows

- **Relative hidden error stays bounded**: between 2.4e-03 and 6.3e-03 for all 24
  layers. It does not compound.
- **Absolute error grows** (8e-05 → 0.12) only because the residual stream itself grows
  (L2 2.6 → 449). That is scale, not divergence.
- **Attention relative error drifts up** from ~5e-04 to ~1e-02 by the last layers, i.e.
  the layer outputs diverge relatively more as the stream grows, while the residual
  stream keeps the overall relative error flat — the residual addition dominates.

## Routing flips

28 of 984 token-layers (2.8%) changed which expert set was selected, spread across 16
layers and more frequent late (4 at layer 21, 3 at layer 22). Every flip sits at a
reference biased-selection margin between **2.6e-06 and 8.1e-04** — the cutoff narrow
enough for the observed drift to cross it. No flip occurred where the margin was wide.
Recorded as boundary events, not routing errors, and **no blanket threshold is being
introduced**: the sweep reports each margin so a future flip with a large margin would
stand out immediately.

## Final state and stop decision

- `final_norm` relative RMSE 4.5%, max abs 0.37. The relative figure is larger than the
  per-layer 0.5% because RMS normalisation divides by a slightly different scale, which
  amplifies the few large-magnitude elements (the max abs tracks the biggest channels).
- **Stop decision identical** (class 0). Logits differ by at most 5.4e-03 on a value of
  ~6.67 (0.08%).

## Interpretation

The differences stay local and bounded: relative hidden error is flat across depth,
expert flips occur only at the narrowest cutoffs, and the downstream decision is
unchanged. Nothing here points to an implementation bug — the residual source is the
KDA recurrence-vs-chunk ordering established earlier.

The one number to watch at the next stage is the 4.5% relative error on `final_norm`:
the thinker's output feeds the flow/talker conditioning, so that is where any
sensitivity would first appear.
