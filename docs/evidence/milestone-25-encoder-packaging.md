# Packaging check: the encoder is already in the package

Question raised: would WAV reference support need another ~2 GB download (the VAE encoder)?

Answer: **no.** The full package already carries both halves of the VAE.

```
package tensors  : 2141
audio.encoder.*  : 348
audio.decoder.*  : 295
```

Present and confirmed by name:

| tensor | status |
|---|---|
| `audio.encoder.fc1.weight` | present |
| `audio.encoder.fc3.weight` | present |
| `audio.encoder.encoder.norm.weight` | present |
| `audio.encoder.aggregator.norm.weight` | present |
| `audio.encoder.cls_embed` | present |
| `audio.decoder.fc1.weight` | present |
| `audio.decoder.head.out.weight` | present |

So the podcast path's WAV support is an **implementation** task, not a packaging one:

- **C++ must port the encoder**: unfold the waveform into 882-sample frames, run the 24-layer
  Qwen2 encoder, apply the 4-frame aggregator with its CLS insertion, then `fc3` and the
  mean/logvar split (taking the mean, as our Python round-trip does).
- **No additional weights are required** — the same GGUF serves encode and decode.
- The 348 encoder tensors are already counted in the package's 16.77 GB; nothing to download.

This supersedes the earlier note that the encoder "adds ~2 GB of package weight". It does not
add anything; it only adds code.

## Status of the two paths

| path | state |
|---|---|
| saved-latent bridge (file → prompt matrix) | works, verified to 3.3e-07, explicitly a **test** |
| C++ encoder (WAV → latents) | **not implemented**, but weights are present |
| external preprocessing contract | alternative: an external encoder emits 75×64 latents + transcript; documented, not built |

The bridge stays labelled as a test until one of the latter two exists.

## Next

1. **Two-speaker CPU render** using the verified prompt matrix: assemble the 150-row embeddings,
   prefill the thinker with them, generate ~40–60 patches (3.2–4.8 s) under the 250-patch cap,
   decode, and listen. This is the first audible podcast output and the gate on distinct,
   stable voices.
2. Then long-line splitting, and the encoder-implementation-or-preprocessing decision.
