# WAV path verified, and a real discovery: the 64-token window is active

## WAV boundary (first gate)

```
ref1.wav -> 132300 samples (3.00 s)
ref2.wav -> 132300 samples (3.00 s)
wav encode: 264600 samples -> 300 frames -> 75 latents
```

- sample counts match the `.npy` arrays used by the Python test exactly (132300 each);
- 44100 Hz mono requested and enforced — the loader **throws** for any other rate, and for
  formats other than 16-bit PCM or 32-bit float, rather than silently reinterpreting audio;
- 16-bit quantization was shown to be immaterial: the float concatenation produced the same
  conclusion as the WAV path.

## The discovery

Initially the WAV path's latents had rms 0.752 vs the saved reference's 0.863. Stage comparison
showed C++ matched Python **full-causal** exactly (both 0.752464), so the encoder maths was right
and the difference was attention scope. Testing both settings against the saved reference:

| Python setting | latents rms | vs saved reference |
|---|---|---|
| sliding window enabled | 0.863003 | **max_abs 0.000e+00** (exact) |
| full causal | 0.752464 | max_abs 2.678, rel 0.648 |

So the checkpoint's Qwen2 stacks **do** apply the 64-token window
(`use_sliding_window=true`, `max_window_layers=0` → every layer, window 64). The
`transformers` warning that sliding window "is not implemented for sdpa" is misleading — it is
applied, as the exact reproduction proves.

After adding the window to the C++ encoder (and its 4-layer aggregator):

```
saved ref rms 0.863003   cpp rms 0.863003
max_abs 1.287e-05  rel 1.747e-06  corr 1.00000000
```

## Consequence for the decoder — and for earlier results

The VAE decoder shares the same `use_sliding_window=true` setting. Our decoder parity tests were
run against a Python model with `use_sliding_window=False` (chosen for comparability), so:

- C++ decoder **matches our Python reference**, which was the point of those tests;
- but for sequences longer than 64 tokens it does **not** match the released model's behaviour:
  the 50-patch round-trip (200 tokens) and the 60-patch podcast decode are both in that range.

This is the "sliding window compatibility item" flagged earlier, now with proof that it matters.
Plan: add the same window to the C++ VAE decoder, then re-run the 50-patch round-trip against a
window-enabled Python reference (expect a *different* waveform from the earlier full-causal one —
that difference is the point), and confirm the output still reconstructs real speech.

Note the 80 ms single-patch case is unaffected (4 tokens < 64).

## Standing items

- Encoder maths: verified against the saved reference (1.3e-05).
- Normal CLI path (options instead of env vars): still to wire.
- `spk_head(zeros)`: unchanged, as directed.
