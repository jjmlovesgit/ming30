# Port decisions — Ming 3.0 TTS Tiny

Policy: these are **our** calls, made for our pipeline (voice-swap over SAM runs,
OpenAI-shaped adapter on Windows/CUDA). The audio.cpp maintainers are not consulted
until the port demonstrably works. Every decision below names the condition under
which we would revisit it.

Status of `docs/port-review-notes.md`: it is now an **internal** record. It becomes
an external hand-over document once the success gate passes.

## Decisions

### D1 — Expose upstream modes as `template_name`, not new tasks

Ten upstream modes (`seed`, `multilingual`, `podcast`, `paralinguistic`, `basic`,
`emotion`, `dialect`, `ip`, `voice_design`, `freeform`) become values of the
existing `template_name` option. No new audio.cpp task enum entry.

*Why:* our adapter already sends `options.template_name`, and the server's task
surface stays untouched — zero changes needed on our side to reach `podcast`.
*Revisit if:* the server needs a first-class dialogue route for streaming or the
UI needs to distinguish dialogue from TTS.

### D2 — Multi-reference options: `voice_samples` + `reference_text`

Podcast takes `voice_samples` (positional list of reference wavs) and
`reference_text` (a `speaker_N:`-tagged transcript string), mirroring the
`moss_ttsd` family rather than inventing `speaker_references` /
`speaker_reference_texts`.

*Why:* the runtime already implements this shape for another dialogue family, and
our scripts can send it unchanged — one convention across every dialogue model we
touch. Clarity of the new names is not worth a second convention.
*Revisit if:* a family lands that needs per-speaker structured objects (ids,
languages) rather than positional lists.

### D3 — `reference_cache` keys the whole reference set

For podcast, the cache key is a hash of the ordered reference paths, their texts
and the target language. Never the single-file assumption the 0.5B uses.

*Why:* our swap synthesises ~49 turns against the same two references; caching is
a large win, and a wrong key would silently swap a speaker's identity.
*Revisit if:* cache invalidation proves unreliable across sessions.

### D4 — Expose `flow_steps` (default 10)

*Why:* we want to A/B steps against quality in our own pipeline, and 10 matches
upstream's formal value so the default is parity-correct.

### D5 — Leave `max_tokens` unset until measured

*Why:* tiny's context is not the 0.5B's 512 positions. Inventing a default would
bake a guess into the spec; our pipeline can pass explicit values per request.

### D6 — Add a parity mode (`parity=reference`)

Session option that forces SDPA attention and the non-fused KDA recurrence, i.e.
the upstream reference numerics. The fast path remains the default once parity is
established.

*Why:* we cannot judge quality without a reference-matching configuration, and it
gives a reproducible comparison later without extra work.

### D7 — Default package `q8_0`, hosted by us initially

q8_0 (~9.5 GB) is the default over bf16 (~18 GB). The spec's download entry points
at a host we control for now; local flows pass `--model <path>` so hosting is not
blocking.

*Why:* 9.5 GB sits comfortably beside the other models on a 32 GB card.
*Revisit if:* the port is accepted upstream — the package should move to the
`audio-cpp` org and the spec's `package_defaults` with it.

### D8 — Develop in our fork; shape the code for `community_models/`

Work stays on branch `ming3-tts-tiny` in `C:\Projects\audio.cpp`. New family code
is written so it can be relocated under `community_models/` without restructuring,
matching CONTRIBUTING's guidance for first-time ports.

### D9 — No maintainer contact until the success gate passes

Then we hand over the decision record plus evidence, as one package.

## Success gate (what "successful" means for us)

Ordered; we do not skip ahead.

1. **Converts.** Converter produces a GGUF that loads and reports the expected
   tensor set.
2. **Speaks.** One `seed` task utterance renders at 44.1 kHz with sane duration.
3. **Podcast runs.** Two references plus a `speaker_1:` / `speaker_2:` script
   produce one take, with no truncation and both speakers present.
4. **Intelligible.** The podcast take transcribes back through faster-whisper at a
   WER comparable to the 0.5B pipeline (we accepted ≤2% on our baseline podcast).
5. **Two identities.** Speaker separation survives: the two speakers differ in
   median F0 and in WAVLM/CAMPPlus embedding distance by at least as much as the
   reference stems do. (Upstream itself warns podcast identity drifts across turns;
   we measure rather than assume.)
6. **Pipeline-compatible.** Our `ming_voice_swap.py` can drive it for a full
   episode, and the QA stage reproduces against the same reference text.

Parity with the Python reference is **not** in the gate: the reference stack is
Linux-only, so proving parity requires either WSL with a Blackwell-capable torch or
a collaborator's outputs. We treat it as the hand-over step, not a prerequisite.

## Explicitly out of scope

- No framework, ggml or backend changes. If one becomes necessary, it is a separate
  branch and a separate conversation.
- No changes to other families, `model_licenses.md`, or the catalog until the gate
  passes.
- No speculative optimisations before parity — a faster kernel that shifts quality
  is worse than a slow correct one.

## Risks this policy accepts

- **We may invest in a shape the maintainers would have rejected.** D1 and D2 are
  the likely disagreements; both are cheap to change late because the spec is data,
  not code.
- **Quality may disappoint for reasons outside our control** (the model's own
  identity drift, 24 → 44.1 kHz interactions). The gate's step 5 exists to catch
  that before the pipeline investment.
