# Ming-3.0-tts-tiny — task matrix for the port

Extracted from upstream `ming_tiny/task_protocol.py` (MIT), retrieved 2026-10-26.
This is the authoritative task surface a C++ port must reproduce; it feeds
`model_specs/ming3_tts_tiny.json` directly.

## TaskSpec fields

`TaskSpec(name, conditioning, include_target_language, instruction_kind,
reference_count, prompt_text_mode, allowed_languages)`

| task | conditioning | target language | instruction | reference wavs | reference texts | languages |
|---|---|---|---|---|---|---|
| `seed` | prompt_latent | yes | none | one | one | Chinese, English |
| `multilingual` | prompt_latent | yes | none | one | one | German, Spanish, French, Korean, Japanese |
| **`podcast`** | **multi_speaker** | yes | none | **many** | **many** | any |
| `basic` | speaker_only | no | basic | one | none | any |
| `emotion` | speaker_only | no | emotion | one | none | any |
| `dialect` | speaker_only | no | dialect | one | none | any |
| `voice_design` | masked_speaker | no | style | none | none | any |
| `ip` | masked_speaker | no | ip | none | none | any |
| `freeform` | masked_speaker | no | freeform | optional one | none | any |
| `paralinguistic` | prompt_latent | yes | none | one | one | Chinese, English |

Ten tasks. `podcast` is the only `multi_speaker` conditioning, and the only one
that takes both many reference wavs and many reference texts.

## Control vocabularies

| control | values |
|---|---|
| `emotion` | 中性 · 悲伤 · 愤怒 · 高兴 · 惊讶 · 恐惧 · 厌恶 |
| `dialect` | 广粤话 · 川渝话 |
| `speed` | 慢速 · 中速 · 快速 |
| `pitch` | 低 · 中 · 高 |
| `volume` | 低 · 中 · 高 |

Aliases exist in upstream (`DIALECT_ALIASES`, `EMOTION_ALIASES`) that map English
and alternate spellings onto those Chinese labels — the port should accept the
same aliases or the CLI will reject inputs upstream accepts.

## Sampler defaults (formal runtime)

| setting | value |
|---|---|
| seed | 1895 for general tasks, **1235 for `podcast`** (`FORMAL_PODCAST_SEED`) |
| cfg (guidance) | 2.0 |
| temperature | 0.0 |
| flow steps | 10 |
| sigma | 0.25 |

## Numerics pins (parity-critical)

Upstream pins its reference environment, which means a port that differs here
will produce *plausible but drifting* output rather than an error:

```
BAILING_KDA_FUSED_RECURRENT = 0   # fused KDA recurrent path disabled
BAILING_MLA_ATTN            = sdpa
MING_TINY_AUDIO_VAE_ATTN    = sdpa
FLA_USE_CUDA_KDA_FWD        = 0   # flash-linear-attention kernels off
FLA_USE_CUDA_KDA_BWD        = 0
```

Practical reading: the published reference numbers come from **SDPA attention and
a non-fused KDA recurrence**. audio.cpp's existing Ming families were tuned
against their own Python references, so the C++ tiny implementation should be
compared under these settings first, and only then optimised — otherwise a
"faster" kernel becomes an unexplained quality difference.

## Where the byte-exact prompt details live

See `tts-tiny-port-plan.md` § *Podcast contract*: `speaker_N:` line syntax,
`" speaker_N:text\n"` canonical form with a leading space, reference transcripts
serialised the same way, and the `task_decode_alignment.podcast = old_shift`
requirement.
