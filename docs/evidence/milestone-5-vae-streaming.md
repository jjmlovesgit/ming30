# VAE streaming vs whole-sequence decode — per-patch is consistent

Before the TTS loop can decode patch-by-patch, the decoder's overlap-add, upsampling state
and KV cache have to stitch exactly like one whole-sequence decode. Compared using the
checkpoint's own `Decoder` class (not a reimplementation), on the 50-patch real-speech
latents.

```
latent: (1, 50, 64) -> 200 decoder tokens

whole decode     : 176400 samples (4.000 s) rms 0.059439
per-patch decode : 176400 samples (4.000 s) rms 0.059439
length difference: 0 samples
max_abs          : 8.769e-06
rel_rmse         : 4.934e-06
correlation      : 1.00000000
```

The two paths agree to 8.8e-06 across all 176,400 samples with identical length and
level. So the streaming state carried between patches (`upsample_state`,
`audio_buffer`, `window_buffer`, `past_key_values`) is consistent with whole-sequence
decoding, and the loop may decode per patch.

Notes:

- `low_level_reconstruct` requires an explicit `(None, None, None)` stream state even on
  the non-streaming path, because it unpacks unconditionally.
- `transformers` warns that the tuple-form `past_key_values` is deprecated; behaviour is
  correct on the installed version, and keeping tuple form preserves the checkpoint
  path's expected format.
- Both paths run with `use_sliding_window=False` for the reasons recorded earlier; the
  64-token window remains a separate compatibility check.

## Next: the text-to-speech loop

Per patch, iterating until the stop decision:

1. thinker forward over `[prompt embeddings, projected previous latents]`;
2. stop head on the last hidden position (except in `old_shift` decode, which runs a
   fixed length and takes no stop decision);
3. flow sample conditioned on that hidden + latent history, using the per-step noise;
4. append the latent, project it through `linear_proj_audio` for the next step;
5. decode the accumulated latents (per patch, now verified equivalent to whole-sequence).

Captures at every patch: prompt/step token count, the last hidden (or its checksum), the
stop logit and probability, the flow's initial noise, the latent, the projected feedback
vector, and the VAE chunk — so a bad sentence is traceable to the patch and the component.

The remaining engineering work is on the thinker side: it currently consumes token ids and
recomputes the whole sequence per call. The loop needs (a) an embeddings input path so a
projected latent can be appended, and (b) either a KV cache or the acceptance of
O(steps^2) recompute for the first sentence.
