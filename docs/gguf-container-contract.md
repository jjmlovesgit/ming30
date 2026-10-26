# GGUF container contract (observed from a real Ming TTS GGUF)

Read directly from `ming-3.0-tts-0.5b-q8_0.gguf` metadata on 2026-10-26. This is
what a family loader actually reads, and it differs from the layout I first
invented for the tiny writer — so it is recorded here as the contract to match.

## Top-level metadata

| key | type | example | meaning |
|---|---|---|---|
| `general.architecture` | string | `audiocpp` | constant — **not** the family name |
| `general.name` | string | hash-like id | package identity |
| `audiocpp.tensor_name_format` | string | `native` | names are family-native, not HF |
| `audiocpp.source_format` | string | `.gguf` | provenance |
| `audiocpp.weight_type` | string | `orig` | quantisation of the package |
| `audiocpp.model_spec.version` | uint32 | — | spec schema version |
| `audiocpp.model_spec.family` | string | `ming3_tts` | family id |
| `audiocpp.model_spec.json` | string | full spec | **the model spec is embedded**, so the file is self-describing |

## Tensors

| key | type | notes |
|---|---|---|
| `audiocpp.tensor_names` | string array | logical name of **every** tensor, e.g. `audio.decoder.decoder.layers.0...` |
| `audiocpp.tensor_ranks` | int32 array | one per tensor |
| `audiocpp.tensor_shapes` | int64 array | flattened dims, ggml order (fastest-varying first) |

Physical tensor names in the GGUF are `_audiocpp.{i}`, and `audiocpp.tensor_names[i]`
carries the logical name. The 0.5B package has 1,799 tensors, all named this way —
including the VAE and codec tensors, not just the backbone.

## Embedded files (config, tokenizer, ...)

| key | type | notes |
|---|---|---|
| `audiocpp.embedded_files.names` | string array | e.g. `config.json`, `tokenizer.json`, `tokenizer_config.json` |
| `audiocpp.embedded_files.offsets` | uint64 array | offsets into `data` (length = names + 1) |
| `audiocpp.embedded_files.data` | uint8 array | concatenated file bytes |

## External weights

| key | type | notes |
|---|---|---|
| `audiocpp.tensor_sources.names` | string array | component ids (`thinker`, `talker`, ...), empty for the primary |
| `audiocpp.tensor_sources.paths` | string array | paths recorded at conversion time |

## Consequences for our writer

The tiny writer currently emits `audiocpp.family`, `audiocpp.tensor_names/ranks/shapes`
(fine), and embeds files as one array per file under `audiocpp.file.<name>`. To be
loadable by a real family loader it must instead:

1. set `general.architecture = audiocpp` and the `audiocpp.model_spec.*` keys,
   embedding the family spec JSON;
2. give **every** tensor a logical name in `tensor_names` (thinker banks, talker, vae
   and speaker alike) and write all data as `_audiocpp.{i}`;
3. pack embedded files as `names` + `offsets` + `data`;
4. record tensor sources.

Item 2 is the structural change: the current probe writes talker/vae/speaker under
their real names, which a loader keyed on `_audiocpp.N` would not find.

## How this was found

The family loader path was traced before writing code: `CMakeLists.txt` registers
each family through `audiocpp_add_model(<family> ... <loader factory>)`, and
`make_ming3_tts_loader()` builds a `SpecBackedVoiceModelConfig` whose
`load_assets` calls `model_spec::load_resource_bundle_for_family(path, family)` and
`open_tensor_source(...)`. Those helpers read the metadata above — which is why the
container layout matters more than the spec file for loadability.
