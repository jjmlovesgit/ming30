# `old_shift` output length — resolved from source, plus shared-engine regression

## The rule

`decode_alignment` is **the training-time label-shift convention**, not a duration model
(`prompt_contract.py`: "Return the training label shift used by autoregressive TTS
decode."). This checkpoint sets:

| key | value |
|---|---|
| `tts_config.decode_alignment` | `old_shift` |
| `tts_config.task_decode_alignment` | `{"podcast": "old_shift"}` |
| `tts_config.maximum_decode_steps` | **250** |
| `tts_config.minimum_decode_steps` | 13 |
| `tts_config.stop_threshold` | 0.5 |

Consequences in `modeling_lingv3_tts.py`:

- L335: `old_shift_decode = decode_alignment(...) == OLD_SHIFT_DECODE_ALIGNMENT` → **true**
  for every task of this checkpoint, including `tts`.
- L537: `decode_iterations = max_decode_steps if old_shift_decode else max_decode_steps + 1`.
- L575 / L584–587: the stop head is only evaluated **and only allowed to break** when
  `not old_shift_decode`. In this checkpoint it is never consulted.
- L598: the loop breaks when `generated_steps >= max_decode_steps`.

**So `old_shift` fixes the output length: exactly `maximum_decode_steps` patches = 250
× 80 ms = 20.0 s at 12.5 patches/s.** There is no content-driven stop for this
checkpoint, and no duration estimator anywhere in the reference code (searched: no
`duration`/`steps_from`/`length_pred`). A caller wanting a shorter utterance must pass
`max_decode_steps` explicitly; if the key were missing entirely the hard-coded fallback
is 800 patches = 64 s.

## What this means for our loop

- The 30-patch (2.40 s) render was **our cap**, not the model's rule. The model default is
  250 patches / 20 s.
- Our 30-patch run measured stop probabilities of 2.5e-03 falling to 2e-05 — three to
  four orders of magnitude below the 0.5 threshold. So even in an `unshifted` variant the
  stop head would not have fired; length here is entirely policy-driven.
- Practical rule for the port: expose the patch count as a caller parameter mirroring
  `max_decode_steps`, defaulting to 250, and document that shorter renders are explicit
  caps. Any future "natural length" behaviour would be a caller-side duration estimate,
  since the checkpoint provides none.

## Shared-engine regression

The family is added through the supported `audiocpp_add_model(...)` extension point, and
all its code lives under `src/models/ming3_tts_tiny/` + one CMake block. To confirm the
shared registry and build changes do not disturb other models, the CLI was rebuilt with an
additional independent family:

```
model composite: custom selected [ming3_tts;ming3_audio;ming3_tts_tiny;plapre]
build: success
--list-loaders: ming3_tts_tiny ✓  ming3_tts ✓  ming3_audio ✓  plapre ✓
```

All four register, including `plapre`, which was untouched by this work. That is the
baseline regression to re-run after any change to shared engine files.

## Next: GPU gate

With the length rule documented, the seeded single-patch test becomes the GPU gate: after
adding a KV cache and CUDA placement, one seeded patch must still match the reference
(4.8e-06 tolerance established at patch 17) before its time and VRAM are measured.
