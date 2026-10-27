# Podcast listening gate: PASSED (user-confirmed)

**Result: the first audible podcast milestone is met.** The user listened to
`podcast-exchange-clean.wav` and confirmed **two distinct voices and both lines complete**, with
the opening artifact handled in the cleaned clip without cutting speech.

Agent side, for clarity: the criteria are auditory and this agent cannot hear audio, so the pass
is recorded **from the user's listening**, not from any measurement here. Everything below is
supporting analysis, not the basis of the pass.

---


## The honest part

The podcast milestone's success criteria are **did two distinct voices speak, and were both
lines complete** — both auditory judgements. This agent cannot hear audio; it can only measure
waveforms. So those criteria remain **unverified from the implementation side**. The "sounds
good overall" observation in this session came from the user, not from any measurement here.

Recording that explicitly so the milestone is not marked complete by implication.

## A measured proxy, and its limits

Crude pitch/timbre analysis of `podcast-exchange-clean.wav` (4.74 s, 85 voiced frames,
40 ms windows, autocorrelation F0 with a 70–400 Hz search):

```
F0 percentiles (Hz) p10/p25/median/p75/p90 : 105.3 / 143.2 / 226.2 / 256.4 / 337.2
F0 histogram, 70-300 Hz (8 bins)           : 7, 10, 9, 6, 8, 13, 15, 4
spectral centroid median                   : 1323 Hz
median F0 by third of the clip             : 249.2 / 142.3 / 249.2 Hz
```

What this suggests: the pitch distribution is broad and not obviously unimodal, and the clip
alternates between a higher band (~249 Hz) and a lower one (~142 Hz) across thirds. That is
*consistent with* two voices taking turns.

What it cannot show:

- whether the voices are the two reference speakers (no speaker verification was run);
- whether the words are intelligible or the lines semantically complete;
- whether the alternation is real turn-taking or just intonation within one voice — the thirds
  are a blunt instrument with no transcript alignment.

## Why no transcript check either

There is no ASR in this loop to compare words against the script. Running one would test
intelligibility and line completeness far better than F0 statistics, and is the right next
measurement if the listening result is ambiguous — `faster-whisper` is available in the
`sam-audio-v2` environment and was used earlier in this project for the voice-swap QA.

## Status

| criterion | how it can be settled |
|---|---|
| two distinct voices | listening, or speaker verification against the two reference clips |
| both lines complete | listening, or ASR against the script |
| artifact at the start | **settled** (decoder boundary; trim justified for this clip) |

Also standing: the 60 ms trim is specific to this render (onset at 150 ms); it must not be
applied blindly to future renders whose onset differs.
