# Podcast step 1: prompt ids match the reference exactly

The podcast path is the original goal, so it starts where the single-speaker path started —
with the prompt, checked token by token before any audio.

## Construction, read off `modeling_lingv3_tts.py`

```
input_ids = build_tts_prompt_ids(tokenizer, target_text=script,
                                 prompt_text=tagged reference transcript,
                                 prompt_latent_tokens=reference patches)
```

- the script is the multi-speaker text: lines of ` speaker_N:text\n`
- the reference transcript is inserted **before** the script, in the same tagged form
- the final `prompt_latent_tokens` ids are `<audioPatch>` placeholders; the reference later
  replaces those embeddings with the projected reference latents
  (`inputs_embeds[:, prompt_start:, :] = prompt_hidden`)
- one `<spk>` slot takes a single CAMPPlus embedding — podcast multi-speaker comes from the
  **script text**, not from several speaker slots

## Verified result

Test exchange: ` speaker_1:Good morning, and welcome to the show.\n speaker_2:Thanks for having me.\n`
with reference ` speaker_1:Good morning.\n speaker_2:Glad to be here.\n` and 100 latent placeholders.

```
reference ids : 175  sha 4ae3b51a31ca70ba...
cpp ids       : 175  sha 4ae3b51a31ca70ba...
PASS: podcast prompt ids match the reference exactly
```

Segment structure on both sides:

| segment | ids |
|---|---|
| user_prefix | 19 |
| speaker_block | 3 |
| text_prefix | 5 |
| reference_text | 18 |
| target_text | 24 |
| assistant_prefix | 5 |
| audio_start | 1 |
| prompt_latents (`<audioPatch>` × N) | 100 |
| **total** | **175** |

The speaker position is 20, identical to the single-speaker prompt, and every trailing id is
an `<audioPatch>` placeholder, which is what the embedding substitution requires.

## What this does and does not establish

Establishes: the C++ builder reproduces the podcast prompt exactly for this exchange, and the
segmentation/order matches the reference's construction.

Does not establish: reference conditioning (CAMPPlus embedding in the `<spk>` slot, projected
reference latents in the placeholder positions), nor any audio. Those are the next two steps,
and the render after them is the first audible podcast test.

## Next

1. Encode two reference wavs through the VAE and the Aggregator to fill the 100 placeholder
   positions; compute the speaker embedding for the `<spk>` slot.
2. Render a short two-speaker exchange on CPU and listen.
3. Then long-line splitting for the 20 s / 250-patch fragment ceiling.
