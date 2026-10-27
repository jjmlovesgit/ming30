# VAE encoder round-trip on real speech — the decoder is fine

Diagnostic for the near-silent decode seen on a flow-generated latent: encode real
speech, decode it back with the checkpoint's Python path, and check level and content.

Clip: 4.00 s of speech from the NotebookLM podcast, mono 44.1 kHz, rms 0.0634.

```
encoder frames   : 200 -> 50 patches
latent           : (1, 50, 64) rms 0.86991  min -4.245  max 2.927
decoder tokens   : 200
restored         : 176400 samples (4.00 s), rms 0.062416, peak 0.390
correlation with original : 0.9131
```

## What this settles

1. **The decoder is correct.** Real speech round-trips at the right level
   (rms 0.0624 vs 0.0634 in) with correlation 0.913 over the full 4 s. The earlier
   near-silence was not a decoder or head problem.
2. **The encoder and decoder agree on latent scale.** Real speech encodes to latents
   with rms 0.870, and the flow's generated latent had rms 0.813 — the same order. So
   the flow output is *not* wrong by scale.
3. Therefore the near-silent decode came from **latent content, not latent magnitude**:
   the flow latent was produced from a prompt-only thinker condition with no audio
   context, i.e. it is not a latent the VAE recognises as speech. That is exactly what
   the multi-patch generation loop is for.

This also confirms the earlier hypothesis ranking: the "shared interpretation mistake"
branch is now ruled out for magnitude/phase, since a real-latent decode is loud and
correlated.

## Multi-patch behaviour

The round-trip used **50 patches / 200 decoder tokens**, so the decoder path is exercised
well beyond a single 80 ms fragment, and the reconstruction stays coherent across the
whole clip (one global level, corr 0.91).

## Open question to carry forward

`Qwen2` in the reference config sets `use_sliding_window=True, sliding_window=64`, but the
`transformers` backend we use (`sdpa`) warns that sliding-window attention is not
implemented and falls back to full causal attention. With 200 tokens that difference is
material for the *released* model (flash-attention path), so:

- the Python reference and the C++ port should both use full causal attention to stay
  comparable;
- if C++ and Python agree but the audio differs from released behaviour, the sliding
  window is the first thing to add.

## Next

Run the C++ decoder over these same 50 captured latents (multi-patch, full causal) and
compare `fc1`, upsampled tokens, decoder hidden, head output and waveform — the C++
decoder probe currently handles one patch and needs the frame count parameterised.
