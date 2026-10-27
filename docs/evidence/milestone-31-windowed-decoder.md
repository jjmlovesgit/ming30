# Windowed decoder: corrected, verified, and the podcast clip re-decoded

## The correction

The VAE decoder shares the checkpoint's `use_sliding_window=true, max_window_layers=0`, so every
layer attends to at most the last 64 tokens. Our decoder had been full causal, which matched our
Python reference (deliberately configured the same way) but not the released behaviour for
sequences longer than 64 tokens.

## Verification on the 50-patch round-trip (no new generation needed)

Decoding the saved 50 latents (200 tokens) three ways:

| decoder | samples | rms |
|---|---|---|
| Python, windowed | 176400 | 0.062416 |
| Python, full causal | 176400 | 0.059439 |
| C++, windowed (new) | 176400 | 0.062416 |

```
cpp windowed vs python windowed    : max_abs 2.059e-05  rel 7.469e-06  corr 1.000000
cpp windowed vs python full causal : max_abs 5.074e-01  rel 3.494e-01  corr 0.943077
```

So the windowed C++ decoder reproduces windowed Python almost exactly, and differs materially
from the earlier full-causal result — the difference is real, not noise.

## Podcast clip re-decoded from the same 60 latents

```
60 latents (rms 0.76013) -> 240 tokens -> 211680 samples (4.80 s), rms 0.051400
first 12 x 50 ms: 0.0104, 0.0011, 0.0001 x5, 0.0155, then speech-level energy
```

The first ~16 patches decode identically to the previous render, as expected: they sit inside the
first 64 tokens, which the window does not restrict. Divergence begins only where windowing
starts to matter, and the overall level changes slightly (0.0505 → 0.0514).

The revised clip carries the same 60 ms trim + 10 ms fade-in as before (its onset is unchanged,
0-50 ms burst, speech from ~150 ms), and is written as
`out/reference/podcast/cpp-render-windowed/podcast-exchange-windowed.wav`.

## Standing

- Listening: the earlier approval was for the **full-causal** clip. This one needs its own check.
- Encoder and decoder now both use the checkpoint's window; the single-patch 4-token case is
  unaffected either way.
- Still to do: the normal CLI options path (no test env vars).
