# Opening artifact isolated: it is the decoder's start boundary, not the first patch

The small click at the start of the podcast render, diagnosed by decoding four ways and
measuring the first 50 ms windows.

| decode | first 10 × 50 ms energy |
|---|---|
| produced latents | **0.0104**, 0.0011, 0.0001, 0.0464, 0.0760, 0.1079, … |
| first latent zeroed | **0.0469**, 0.0078, 0.0013, 0.0440, 0.0801, 0.0939, … |
| all-zero latents | 0.0429, 0.0428, 0.0427, 0.0428, 0.0428, 0.0427, … (constant) |

Two findings:

1. **Zeroing the first generated latent does not remove the burst** — it gets larger
   (0.0104 → 0.0469), converging towards the all-zero level.
2. **All-zero latents decode to a constant ~0.043 rms**, so the decoder emits a steady
   non-zero floor from silence, and the opening transient is a property of the decoder's start
   boundary (first iSTFT frames / window envelope), not of the generated content.

So a trim is justified on evidence rather than by taste, and it is safe for the first word:

```
speech onset (first 50 ms frame above 0.03) : 150 ms
opening burst location                      : 0–50 ms
chosen treatment                            : trim 60 ms + 10 ms fade-in
result                                      : 4.74 s, starts after the burst, before speech
```

`podcast-exchange-clean.wav` keeps the whole utterance (onset 150 ms, trim ends at 60 ms) while
removing the click.

## What this does not settle

- **Both voices distinct and both lines complete** — still the actual podcast success criteria,
  and still only answerable by listening. The waveform numbers cannot establish either.
- The `old_shift` fixed-length cap means line completeness is a duration question, not a stop
  decision; a 4.80 s cap may truncate a second line even when the audio sounds finished.

## Note for the port

The all-zero-latent floor (~0.043 rms) is worth remembering: it means a "silent" latent is not
silent through this VAE, and any future silence check must use the produced latents' own level
rather than assuming zero output. It also explains why the silent tail measured exactly 0.0
earlier only in regions where the latents actively cancelled that floor.
