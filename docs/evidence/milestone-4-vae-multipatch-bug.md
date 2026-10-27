# VAE multi-patch: interpolation fixed, divergence narrowed to attention

## 1. Upsampling was wrong for multi-patch — fixed

`F.interpolate(scale_factor=4, mode="linear", align_corners=False)` interpolates
*between* frames. My first C++ implementation replicated each frame, which is only
correct when there is a single frame (and is exactly why the one-patch result passed).

Fixed to `pos = (i + 0.5)/4 - 0.5`, clamped to `[0, frames-1]`, linear between
neighbours. After the fix:

```
fc1        max_abs=5.96e-07  rel_rmse=1.15e-07
upsampled  max_abs=5.96e-07  rel_rmse=1.15e-07
```

So the checkpoint the coordinator asked for now agrees, and the input to the decoder is
identical on both sides.

## 2. The decoder still diverges, and it starts in layer 0

Per-layer comparison over 24 layers (200 tokens, full causal on both sides):

```
layer  max_abs   rel_rmse
    0  6.58e-01   1.58e-01   <-- diverges
    1  7.02e-01   2.08e-01
   ...
   23  1.58e+01   1.32e+00
```

Within layer 0, per-token max abs:

```
token  0: 0.000    token  1: 0.000    token  2: 0.017
token  3: 0.023    token  4: 0.112    token  5: 0.095
```

**Tokens 0 and 1 are bit-identical; divergence starts at token 2.** With full causal
attention, token 2 is the first position that attends over three keys. That rules out:

- projections (tokens 0–1 exercise the same weights),
- RoPE positions 0 and 1 (exact),
- MLP and residual updates (tokens 0–1 exact, and those are per-token),
- GQA head mapping and the attention scale (a wrong mapping or scale would already
  change tokens 0–1),
- upsampling and fc1 (matched above).

It leaves the attention aggregation itself for sequences of length ≥ 3 — most likely a
mask/indexing detail rather than a numeric one, since the error appears abruptly at the
third key rather than growing from the first.

## 3. What is *not* in question

The one-patch VAE parity (waveform 7.6e-10), the encoder round-trip on real speech
(rms 0.0624 vs 0.0634, correlation 0.913), and the stage A/B/C results are untouched.
This is a multi-patch-specific implementation bug in the C++ decoder.

## 4. Next step

Dump the C++ layer-0 `q`, `k`, `v` after RoPE per token and compare against the Python
tensors from the same inputs. If they match for token 2, the fault is in the softmax or
mask indexing; if they differ, it is in the projection/RoPE path at position 2. The
trace infrastructure (`MING3_TINY_VAE_TRACE`) is in place for exactly this.

## 5. Sliding window — still a separate compatibility item

The reference config sets `use_sliding_window=True, sliding_window=64`; the Python
`sdpa` run used here is forced to full causal (`use_sliding_window=False`) to be
comparable with C++. Once the multi-patch bug is fixed, the 64-token window is a
separate check against the intended inference backend (flash-attention), not part of
this parity result.
