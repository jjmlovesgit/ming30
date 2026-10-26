# Voice-swap POC: Ming 3.0 TTS over an existing SAM_fastapi run

Goal: replace both hosts of a two-speaker NotebookLM podcast with Ming 3.0 TTS
voices, reusing an already-processed `SAM_fastapi` run (separated stems,
word-timed transcripts, editorial script).

No new models, no diarization, no ASR beyond the judge: the run already contains
`target.wav` / `residual.wav` plus `target_transcript.json` / `residual_transcript.json`.

## Pipeline

```
run_manifest.json + transcripts
   └─ tts_pipeline.apply_editorial_script      (128 reviewed edits, reused as-is)
   └─ tts_pipeline.resolve_cross_stem_leakage  (0 conflicts on this run)
   └─ tts_pipeline.build_speaker_turns         (49 turns: 24 target, 25 residual)
   └─ per turn: POST /v1/audio/speech → Ming adapter (US_James / US_Emma)
        └─ optional: transcribe the take, retake with another seed if WER is high
   └─ tts_pipeline.build_tight_dialogue_schedule  (source-gap preserving)
   └─ add_to_timeline / mix_speaker_tracks / encode_mp3
```

Scripts:

| file | role |
|---|---|
| `server/scripts/ming_voice_swap.py` | driver; reuses SAM_fastapi's turn/schedule/assembly code, swaps only the TTS client |
| `server/scripts/qa_faster_whisper.py` | intelligibility QA with the SAM app's own judge (faster-whisper large-v3-turbo) |
| `server/scripts/qa_swap.py` | same idea using the Ming ASR entry (short audio only) |
| `server/scripts/tune_chunking.py` | compares chunking strategies on the worst turns |

## Results

Source 412.0 s, 1072 reference words (520 target / 552 residual), mono 48 kHz.

| take | WER | match | output | synthesis |
|---|---:|---:|---:|---:|
| Ming, default (no QA) | 4.94% | 95.1% | 370.7 s | 78.9 s |
| **Ming, seeds + per-turn QA retake** | **1.77%** | **98.2%** | 376.6 s | 150.6 s |
| PocketTTS baseline, same judge | 0.75% | 99.3% | — | — |
| PocketTTS baseline, as published | 1.87% | 98.1% | — | — |

All numbers from faster-whisper `large-v3-turbo` on CUDA, same reference text.
Ming voices: `US_James` (speaker 1) and `US_Emma` (speaker 2), 44.1 kHz.

Retakes: 10 of 49 turns exceeded the 4% WER threshold and were retaken once with
a different seed; 39 passed first time; none needed a third take.

## What the failures were

The first take lost 53 words. Every loss was a **whole span inside a long turn**,
not phonetic noise:

```
[the ai gave the developer the]                    -> dropped
[abstraction was so context poor they had]         -> dropped
[generating documentation automatically from raw]  -> "shamika nyakna taylor kunra"
[detailed architecture]                            -> "detail targets your"
```

Re-synthesizing the same turn in isolation scored 100%, so this is a
**stochastic draw, not a deterministic chunking bug**. Verified:

- identical request (same text, voice, seed) twice → byte-identical output;
- the same text in a different session position → different output, because the
  sampler's noise stream is shared with the session.

Hence the fix is QA + retake, not a chunking parameter.

Chunk-strategy probe on the three longest turns (143 words):

| strategy | WER |
|---|---:|
| default | 0.00% |
| `text_chunk_size=120` | 2.80% |
| `text_chunk_size=50` | 2.80% |
| `raw_text=true` (one fragment) | 5.59% |
| sentence-split, concatenated | 0.70% |

Defaults are the right choice; forcing one fragment hurts.

## Reproduce

```powershell
# Ming stack (engine + adapter), no Docker
cd server
.\run.cmd

# swap, with per-turn QA retakes (run in the SAM app's environment for faster-whisper)
C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe scripts\ming_voice_swap.py `
  --run "C:\SAM\runs\Podcast Test_20260913_214654_03222214" `
  --out-dir out\sam-swap --tag retake `
  --voice-one US_James --voice-two US_Emma --retakes 2

# score it against the PocketTTS tracks from the same run
C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe scripts\qa_faster_whisper.py `
  --summary out\sam-swap\voice_swap_ming_retake.json `
  --baseline-prefix "C:\SAM\runs\Podcast Test_20260913_214654_03222214\speaker"
```

## Notes and limitations

- **Writes to the run directory are optional**: `--out-dir` keeps outputs
  elsewhere, so a read-only run directory is fine.
- **Output is ~9% shorter** than the source (376.6 s vs 412.0 s): Ming speaks
  faster than the NotebookLM hosts and the schedule preserves gaps rather than
  stretching turns. For a length-matched cut, apply `atempo` per turn — not done here.
- **Prosody is per turn.** Conversational rhythm across turns is scheduled, not
  performed; that is inherent to per-turn synthesis with a single-voice model.
- **Transcript artifacts carry through.** `"bug. bug."` and
  `"The event underscore message mirror?"` come from the source ASR/editorial
  text and are spoken as written; fix them in the editorial script before synthesis.
- **The Ming ASR entry cannot judge long audio.** `ming3_audio` transcribed only
  ~198 words of a 170 s stem regardless of `max_tokens` (no long-audio chunking),
  so QA runs through faster-whisper. Short clips (a few seconds) are exact.
- **faster-whisper lives in the SAM environment.** The plain Python 3.12 has
  torch/transformers/sklearn but not faster-whisper; the QA steps run under
  `C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe`.
