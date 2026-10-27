# C++ family plan — ming3_tts_tiny

Milestones are ordered; each must pass before the next starts. The 3.85 GB probe
package is a **structural** artifact: it proves layout, quantisation and embedded
file delivery. It establishes **no** inference parity and cannot be run (no family
exists yet).

## Component sources to reuse

| piece | take from | verification before reuse |
|---|---|---|
| MoE thinker (Bailing V3, 128 experts / 8 active) | `src/models/ming3_audio/backbone.cpp` | its expected logical expert names match the writer's bank names (`model.layers.N.mlp.experts.{gate,up,down}_proj.weight`); dims differ — hidden 1536, moe_intermediate 512, 24 layers, layer 0 dense |
| flow head (CFM, depth 16) | `src/models/ming3_tts/generator.cpp` + `framework/modules/speech_decoders/ming_flow_runtime.cpp` | 0.5B uses depth 8; tiny needs 16 — parameterise, do not hard-code |
| audio VAE (enc/dec 24 layers, 44.1 kHz) | `framework/codecs/ming_audio_vae_runtime.cpp` | check tensor names and hop/latent dims against the tiny checkpoint |
| speaker conditioning (CAMPPlus, 192) | `module_speakers/campplus_encoder.h` | same encoder as both Ming families |
| session/loader pattern | `src/models/ming3_tts/session.cpp` | `SpecBackedVoiceModelConfig` + `make_spec_backed_voice_loader` |
| assets loader | `src/models/ming3_tts/assets.cpp` | `load_resource_bundle_for_family` + `open_tensor_source` |

**Rule:** check every tensor name and shape against the tiny checkpoint before
copying a component. The logical-name list is in the GGUF metadata
(`audiocpp.tensor_names`), so this is a lookup, not a guess.

## Milestone order

1. **Family registers** — `audiocpp_add_model(ming3_tts_tiny ...)` plus a loader that
   opens the probe package, reads `audiocpp.model_spec.json`, and enumerates tensors.
   No inference. Pass: `--list-loaders` shows the family and a load attempt reports
   the tensor contract without crashing.
2. **Lossless single-fragment generation** — `seed` task, one text fragment, no
   reference audio, bf16 package. Pass: audio is produced and the waveform matches
   the Python reference for the same seed within a stated tolerance. Everything else
   (podcast, multi-reference, chunking, streaming) waits behind this.
3. **Reference conditioning** — one reference wav + transcript (`clone` route).
4. **Podcast** — `speaker_N:` script, `voice_samples` + tagged `reference_text`,
   seed 1235, `old_shift` decode. Multi-fragment, 13–250 decode steps per fragment.
5. **Pipeline integration** — drive it from `ming_voice_swap.py` and compare against
   the 0.5B render with the same QA.

## Deferred until step 2 passes

- podcast prompt assembly and per-speaker references;
- long-form chunking and the 20 s fragment ceiling;
- streaming;
- any quantisation of `lm_head` or `stop_head` (keep source precision).

## Known contract facts the loader must implement

- tensors are physically `_audiocpp.{i}`; logical names in `audiocpp.tensor_names`;
- embedded files via `audiocpp.embedded_files.{names,offsets,data}` — `tokenizer.json`
  (12.2 MB) is inside the package, so the loader must resolve it from the container
  rather than from disk;
- `task_decode_alignment.podcast == "old_shift"` and a 13–250 step decode policy;
- speaker tokens `<spk>` 157175 / `</spk>` 157176, `<audioPatch>` 157164.
