# First text-to-speech loop — 2.40 s generated

## The loop

Per patch, with full recompute each step:

1. thinker forward over `[prompt embeddings, projected previous latents]`;
2. stop head on the last hidden position;
3. flow sample (cfg 2.0, 10 solver steps) with one seeded noise stream;
4. latent projected through the Aggregator (`linear_proj_audio`) and appended for the
   next step;
5. after the loop, all latents decoded with the checkpoint VAE (whole-sequence, verified
   equivalent to per-patch streaming earlier).

Run: prompt 41 ids, 30 patches, seed 1895, cfg 2.0. Python, ~30 s for 4 patches and
~4–5 min for 30 (cost grows with the recomputed sequence).

## Per-patch captures

`out/reference/tts-loop-30/records.json` holds, for every patch: token count, stop
logits, stop probability, latent vector, projected-feedback norm, hidden norm, noise norm.

Observations:

| quantity | value |
|---|---|
| stop probability across 30 patches | 2.5e-03 down to 2e-05 — never fires |
| latent rms per patch | 0.79 – 0.89 |
| feedback norm | 0.60 – 0.70 |
| tokens | 42 → 71 |

The stop head staying quiet is consistent with `old_shift` decode, which ignores it and
runs a fixed number of patches; the loop was capped at 30 (2.40 s).

## The audio

`generated.wav`, 2.40 s at 44.1 kHz:

```
rms 0.088493   peak 0.52964
100 ms energy envelope varies 0.001 -> 0.168 -> 0.015  (temporal structure, not a drone)
spectral energy:  0-300 Hz 30.1%,  300-1000 Hz 47.3%,  1000-3000 Hz 22.4%,
                  3000-8000 Hz 0.17%,  8000-20000 Hz 0.02%
```

## What this changes, and what it does not

- **Level is no longer the problem.** The single patch decoded from a prompt-only
  condition was near-silent (rms 9.8e-05); with 30 patches of accumulated audio history
  the same decoder produces rms 0.088, about 900x louder. That is strong support for the
  "missing audio history" explanation — but, as noted, it does not prove it: a different
  flow-conditioning behaviour across patches remains possible.
- **Content is speech-like in structure** (varying energy envelope, dominant
  300–1000 Hz) but **clearly muffled**: only 0.17% of energy above 3 kHz. That is the
  first quality flag, and it points at one of: the VAE decoder's high band (real-speech
  round-trip preserved HF, so less likely), the generated latents lacking HF detail, or the
  flow sampling.
- The stop mechanism is unreproduced for this task (old_shift ignores the stop head), so
  utterance length is currently a cap, not a decision.

## Next

1. Compare this generation with the C++ path once the loop is ported (the C++ thinker
   needs an embeddings input path; the rest of the stages already agree to ~1e-5).
2. Investigate the muffled band: decode latents from the real-speech round-trip through
   the same path and compare spectra (it preserved HF), then check whether the generated
   latents' own spectrum differs in the high band.
3. Reproduce the stop/length mechanism for this task, or record explicitly that
   `old_shift` uses a fixed length.
