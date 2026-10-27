# Stage D — VAE decoder reference, one latent patch

## Path, from source

```
latent [1,1,64]
 -> fc1 Linear(64 -> 896)
 -> linear upsample x4 (align_corners=False)            audio_config.patch_size = 4
 -> Qwen2    24 layers, hidden 896, 14 heads, 2 kv heads, rope 1e6, causal
 -> head.out Linear(896 -> n_fft+2 = 3530)
 -> split (mag, phase) of 1765 bins -> S = exp(mag)*(cos + i sin), mag clipped at 1e2
 -> iSTFT n_fft 3528, hop 882, window from the checkpoint, padding "same" (pad 1323)
 -> 3528 samples = 80.0 ms at 44.1 kHz
```

That matches `output_dim 882 × patch 4 = 3528` and the 12.5 latent/s rate, so the single
patch is exactly the 80 ms fragment the coordinator described.

Run (latent taken from the stage-C flow output, seed 1234, cfg 1):

```
latent          : (1, 1, 64)
upsampled       : (1, 4, 896) (patch 4)
decoder hidden  : (1, 4, 896)
head out        : (1, 4, 3530)
waveform        : (3528,) = 80.0 ms at 44.1 kHz
waveform rms    : 0.000098   max|.|: 0.0003
```

Captures saved for the C++ comparison: `fc1`, `upsampled`, `decoder_hidden`, `head_out`,
`magnitude`, `waveform`.

## Flag: the amplitude is implausibly low, and the cause is not yet known

The decoded fragment is near-silent (`rms 9.8e-05`), and mean spectral magnitude is
`0.0015` — meaning the head emits `mag ≈ exp(-6.5)`. Checks done so far:

| check | result |
|---|---|
| state dict load | no missing / unexpected keys |
| `fc1` output | rms 0.463, max 1.73 — sane |
| decoder hidden | rms 0.688, max 2.07 — sane |
| head bias | mean −0.026, rms 0.060 — small, so the bias is not forcing silence |
| head weight · hidden | rms 5.07 — large, so the head is *not* starved |
| repeated latent ×9 (720 ms) | magnitude identical per frame (0.0015), rms 6.7e-05 |
| random latent | even quieter (3e-06), magnitude 0.0007 |

So the decoder's internals look structurally healthy, yet the produced magnitudes are
tiny and essentially constant across frames. Two candidate explanations remain:

1. **The latent is out of distribution.** This latent came from a flow pass conditioned on
   a *prompt-only* thinker state — the model has never seen audio context for this
   condition. The VAE may decode such latents to silence. This is the explanation the
   coordinator's point predicts: a single fragment proves decoder math, not voice.
2. **A still-missing magnitude/phase detail** in the head path.

Distinguishing them needs one of:
- an **encoder round-trip** on a real waveform (in-distribution latent → decode → compare),
  which validates the decoder independently of the flow;
- the **multi-patch generation loop**, which produces latents with real audio context.

Neither has been done yet. I am not claiming VAE parity on the strength of this result.

## Next

1. Port the VAE decoder to C++ and compare `fc1`, `upsampled`, `decoder_hidden`,
   `head_out`, `magnitude`, `waveform` against these captures — a mismatch will be
   located inside the decoder, and an agreement will confirm the maths is shared even if
   the amplitude question stays open.
2. Resolve the amplitude question with an encoder round-trip on a real wav (preferred, it
   is independent of the generation loop), then run the multi-patch loop for audio.


## C++ parity — all five checkpoints

The C++ decoder (fc1, upsample, Qwen2 24 layers with GQA and the Qwen2 halves-convention
RoPE, head, naive iDFT + windowed overlap-add) was fed the identical latent bytes.

| checkpoint | max_abs | rel_rmse |
|---|---|---|
| after `fc1` | 3.58e-07 | 1.59e-07 |
| after upsampling | 3.58e-07 | 1.60e-07 |
| after decoder (final norm) | 2.29e-06 | 7.12e-07 |
| `head.out` (iSTFT input path) | 1.34e-05 | 4.61e-07 |
| spectral magnitude | 2.68e-07 | 1.43e-06 |
| **final waveform** | **7.57e-10** | 2.52e-06 |

(One comparison artefact worth recording: the reference stores magnitude as
`[1,1765,4]` and the C++ dump as `[1,4,1765]`, so the arrays must be aligned before
comparison — that transposition, not the decoder, produced the only initially red row.)

Because the checkpoints bracket the transformer and the waveform reconstruction
separately, the agreement locates any future regression precisely: a mismatch before
`decoder_hidden` is Qwen2, one after it but before the waveform is the head or iSTFT.

## Consequence for the amplitude question

The near-silence is **shared by both implementations** (waveform rms 9.8e-05 on each
side, agreeing to 1e-9), so it is not a porting error. The remaining explanations stay
open and are now modelling questions rather than parity questions:

1. the latent is out of distribution for the VAE (it came from a flow pass conditioned on
   a prompt-only thinker state, with no audio context);
2. a shared detail of magnitude/phase interpretation that both implementations mirror.

An encoder round-trip on a real waveform will settle it without depending on the
generation loop.
