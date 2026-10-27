# Podcast fill gate passes: the 150-row prompt matrix matches

C++ assembles the podcast prompt embeddings and matches the Python reference row by row.

```
ids equal      : True (150 vs 150)
matrix shape   : python (150, 1536) vs cpp (150, 1536)
prompt_start   : 75    speaker row 20
whole matrix   : max_abs=3.260e-07  rel=1.557e-06
worst row      : 143 (max_abs 3.260e-07)
speaker row 20 : max_abs 0.000e+00
placeholder rows (75..149): max_abs 3.260e-07, 75 distinct rows
verdict        : PASS
```

- The **speaker row is exact** (0.0): `spk_head(zeros)` is its bias, and both sides write it
  to row 20.
- The **75 reference rows are distinct and correct** — each latent passes through the
  Aggregator separately and lands in its own placeholder row, which is the structure the
  earlier four-latents-per-row misreading would have filled wrongly.
- The worst row overall is 143 at 3.3e-07, i.e. float32 ordering noise from the Aggregator,
  consistent with the ~1e-06 tolerance the rest of the port holds.

## What this establishes, and what it deliberately does not

Established: given reference latents, C++ builds the same prompt embedding matrix the
reference does — ids, placeholder fill order, and the single speaker row.

**Not established, and recorded as such:** the reference latents came from a **file** produced
by the Python VAE encoder. That is a test bridge, not a feature. A usable podcast path that
accepts WAV references must either

- encode those WAVs in C++ (VAE encoder port: frame unfolding, Qwen2 encoder, 4-frame
  aggregator, mean/logvar projection), or
- have a clearly defined preprocessing contract: an external encoder produces the 75×64
  latents and their transcript, and the loader consumes them.

The choice affects packaging more than maths: the encoder is ~2 GB of the package and adds a
second Qwen2 model to the family.

Also still open: which CAMPPlus vector a multi-speaker caller supplies for the single
`<spk>` row. Zeros is a valid explicit test case; it is not a decision.

## Next

1. Short two-speaker CPU render using this matrix (latents still from the file), and listen —
   the first audible podcast output.
2. Then long-line splitting, and the WAV-reference preprocessing decision above.
