# First two-speaker podcast render (60 patches, 4.80 s)

## Parity at the longer prompt

Python check of **one seeded podcast patch** using the identical 150-row matrix and first noise
vector, which also exercises the thinker cache at a longer, audio-filled prompt than the
41-token case:

```
prefill       : (1, 150, 1536)
hidden last   : rms 0.42916  norm 16.82
stop logits   : [14.3207, 5.1458]
latent python : rms 0.834123
latent cpp    : rms 0.834128
max_abs       : 1.907e-05   rel_rmse 7.861e-06   PASS
```

So the C++ prefill over the podcast matrix, the cached step after it, and the first flow sample
all agree with Python at this longer prompt.

## The render

C++ cached loop, 60 patches, seed 1235, cfg 2.0: 503 s wall (≈8.4 s/patch at 150–209 tokens).
Decoded with the checkpoint VAE:

```
latents (1, 60, 64) -> 211680 samples = 4.80 s, rms 0.050473
envelope (50 ms): min 0.0 max 0.1282 mean 0.0306
opening          : ~350 ms of near-silence, then speech-level energy (~0.10) sustained
ending           : the final ~0.5 s is exactly 0.0
```

## What this says before listening

- Audio exists at speech level and is **not** cut mid-word by a decode failure.
- The trailing half-second of exact zeros is the thing to watch: with a **fixed** 60-patch cap
  and no end-of-sentence decision under `old_shift`, a second line can simply not fit. The
  4.80 s cap may be truncating the exchange rather than ending it.
- Nothing here proves two distinct voices; that needs listening, and the wav is saved as
  `out/reference/podcast/cpp-render/podcast-exchange.wav`.

## Next

1. Listen. If the second line is truncated, re-render with a longer cap (120 patches = 9.6 s)
   before drawing conclusions about voice stability.
2. Then long-line splitting, and the encoder/preprocessing decision for WAV references.
