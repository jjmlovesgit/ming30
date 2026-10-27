# Podcast fill: reference latents and the speaker row

Continuing the podcast gate. The prompt ids matched; this is what goes into the slots.

## What the reference actually does

```
prompt_waveform                    (one waveform, references concatenated)
  -> audio.encode_latent           -> prompt_latent [1, steps, 64]
prompt_latent_steps   = steps
prompt_latent_tokens  = steps // tts_config.patch_size
grouped               = prompt_latent.reshape(-1, patch_size, 64)
prompt_hidden         = linear_proj_audio(grouped) -> [1, tokens, 1536]
inputs_embeds[:, prompt_start:, :] = prompt_hidden

speaker_hidden = spk_head(speaker_embedding[192]) -> [1, 1536]
inputs_embeds[:, speaker_position, :] = speaker_hidden
```

Two details I had wrong and corrected from source:

1. **`tts_config.patch_size` is 1**, not 4. The 4 is the VAE encoder's *internal* aggregator
   (waveform frames → latent frames). So each latent is its own Aggregator group (CLS plus one
   latent), and `prompt_latent_tokens` equals the latent frame count.
2. **The 12.5 Hz rate holds**: 6.00 s of reference audio → 300 waveform frames → **75 latent
   frames → 75 placeholder tokens**, matching `12.5 × 6 s`.

## And for the speaker slot

`spk_head = Linear(speaker_embedding_dim=192 → 1536, bias=True)` and the model writes exactly
one row: `inputs_embeds[:, speaker_position, :] = spk_head(speaker_embedding)`.

So with two references there is still **one** speaker row. Source does not say how to derive
that single 192-dim vector from several clips — it is a caller convention, not model logic.
For this gate it is zeros, which the reference itself documents as the missing-reference case,
so the speaker row is exactly `spk_head`'s bias. That keeps the gate focused on the 75
reference rows while leaving the CAMPPlus-from-two-clips question explicitly open.

## Reference artifacts produced

Two 3 s clips (different offsets of the NotebookLM podcast) concatenated to 6.00 s, encoded
once, truncated/padded to the placeholder count:

```
prompt        : 150 ids (75 prefix, 75 placeholders), speaker row 20   sha 88be16a1...
references    : (1, 75, 64) rms 0.8630
projected     : (1, 75, 1536) rms 0.01781        (per-latent Aggregator, patch_size 1)
speaker row   : spk_head(zeros) = bias, rms 0.04098
embeds        : (1, 150, 1536)
row 75 rms    : 0.01902     row 149 rms 0.01918    (distinct, so not a duplicated row)
```

Saved as `out/reference/podcast/podcast-embeddings.npz` for the position-by-position
comparison.

## Next

C++ must assemble the same matrix — ids from the podcast builder, reference latents projected
through its Aggregator, `spk_head(zeros)` into row 20 — and the two matrices must be compared
row by row, not just at the endpoints. The C++ VAE encoder is not ported yet, so reference
latents will be supplied from this file first; that is a supply bridge, not a claim of parity
on the encoder.

After that comparison: the short two-speaker CPU render, then long-line splitting.
