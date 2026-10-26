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
Every take scored by faster-whisper `large-v3-turbo` on CUDA against the same text.

| take | configuration | WER | match | output | pacing median \|err\| | within ±10% |
|---|---|---:|---:|---:|---:|---:|
| 1 | default, no QA | 4.94% | 95.1% | 370.7 s | 0.129 | 19/49 |
| 2 | seeds + QA retake | 1.77% | 98.2% | 376.6 s | 0.129 | 19/49 |
| 3 | + 3-candidate pacing selection | 1.96% | 98.0% | 383.8 s | 0.098 | 26/49 |
| 4 | cloned references + hints | 0.84% | 99.2% | 351.3 s | 0.154 | 13/49 |
| **5** | **cloned references + selection + tempo fit** | **0.84%** | **99.2%** | **394.7 s** | **0.005** | **37/49** |
| — | PocketTTS baseline, same judge | 0.75% | 99.3% | — | — | — |
| — | PocketTTS baseline, as published | 1.87% | 98.1% | — | — | — |

Take 5 is the recommended configuration and the mix shipped in `out/sam-swap/`.

## What each cheap fix actually bought

**Candidate selection on pacing.** Generate N passing takes per turn, pick the one
whose duration is closest to the source window. Median pacing error 0.129 → 0.098,
turns within ±10% 19 → 26. Intelligibility unchanged (98.2% → 98.0%, within noise).
Cost: 2.5× synthesis time.

**Cloned references — the biggest single win.** Built-in voices (`US_James`,
`US_Emma`) are conditioned only on a voice id, and on this checkpoint that
occasionally degenerates: whole spans dropped or replaced with invented speech.
Cloning the same built-in voice once (`scripts/make_clone_voices.py`, ~7 s
reference) gives the model an acoustic prefix to continue, and the degenerate
draws mostly disappear: **21 errors → 9**, i.e. WER 1.96% → 0.84%, matching the
PocketTTS baseline.

**Tempo fit.** The cloned delivery runs ~8% fast, so the mix shrank to 351 s.
Resampling each turn toward its source window (clamped to 0.85–1.15×, so no
audible warble) restored it: **394.7 s vs 412 s source**, median pacing error
0.005, 37/49 turns within ±10%. Clamping still leaves 12 short turns outside the
window — a backchannel cannot be stretched 1.55× without sounding wrong.

**Backchannel hints — mostly a dead end here.**

- Raw `instruct` is **rejected for built-in voices**: `Ming IP uses structured
  controls, not raw instruct`. The adapter now returns a clear 400 instead of a 502.
- It *is* accepted on a cloned voice, so hints work — but only 2 of 49 turns
  matched the backchannel heuristic (short or an exact interjection), because most
  reactive lines in this transcript are glued to the start of long turns
  (`"Right? They simply aren't stopping…"`). Extending the heuristic would mean
  splitting turns mid-sentence, which reintroduces the seams the hint was meant
  to hide.

**Energy matching — disabled by default.** Selection originally also matched the
take's loudness to the source window. That signal is wrong on this run: per-stem
turn timings do not reliably align to the stem WAVs after editorial reassignment,
so some windows land on digital silence (`peak 0.000`), producing +55 dB
"mismatches". `--energy-weight` stays available but defaults to `0`.

## Reproduce

```powershell
# Ming stack (engine + adapter), no Docker
cd server
.\run.cmd

# register 7 s clone references for the two built-in voices (once)
python scripts\make_clone_voices.py

# recommended render: clones + selection + tempo fit + hints
C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe scripts\ming_voice_swap.py `
  --run "C:\SAM\runs\Podcast Test_20260913_214654_03222214" `
  --out-dir out\sam-swap --tag final `
  --voice-one james_ref --voice-two emma_ref `
  --candidates 2 --retakes 2 --fit-pacing `
  --backchannel-instruct "Quick, low-energy conversational backchannel."

# score it against the PocketTTS tracks from the same run
C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe scripts\qa_faster_whisper.py `
  --summary out\sam-swap\voice_swap_ming_final.json `
  --baseline-prefix "C:\SAM\runs\Podcast Test_20260913_214654_03222214\speaker"
```

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

## Notes and limitations

- **Writes to the run directory are optional**: `--out-dir` keeps outputs
  elsewhere, so a read-only run directory is fine.
- **Pacing**: with tempo fit the final mix is 394.7 s vs 412.0 s source (95.8%);
  without it the cloned delivery runs ~8% fast and lands at 351–377 s.
  Twelve short turns sit outside the 0.85–1.15 clamp and stay off-window.
- **Prosody is per turn.** Conversational rhythm across turns is scheduled, not
  performed; that is inherent to per-turn synthesis with a single-voice model.
  Delivery hints help only where a turn is short enough to be a pure interjection.
- **Built-in voices accept no raw instructions** (`Ming IP uses structured
  controls`). Clone the built-in voice once to get a hint-capable, more stable
  voice — that is why the recommended render uses `james_ref` / `emma_ref`.
- **Transcript artifacts carry through.** `"bug. bug."` and
  `"The event underscore message mirror?"` come from the source ASR/editorial
  text and are spoken as written; fix them in the editorial script before synthesis.
- **The Ming ASR entry cannot judge long audio.** `ming3_audio` transcribed only
  ~198 words of a 170 s stem regardless of `max_tokens` (no long-audio chunking),
  so QA runs through faster-whisper. Short clips (a few seconds) are exact.
- **faster-whisper lives in the SAM environment.** The plain Python 3.12 has
  torch/transformers/sklearn but not faster-whisper; the QA steps run under
  `C:\Users\Jim\miniconda3\envs\sam-audio-v2\python.exe`.
