# From audible demo to usable feature

Status: the two-speaker CPU render is **audible and user-confirmed** — distinct voices, both
lines complete. What follows is what a usable feature still needs, in dependency order.

## 1. Accept reference WAVs in C++ (replaces the saved-latent bridge)

Today the reference latents arrive as a file produced by the Python encoder; that is a test
bridge. The package already contains the encoder (348 `audio.encoder.*` tensors), so this is
code, not weights.

Work:

1. Frame unfold the waveform: `ceil(samples / 882)` frames of 882 samples, zero-padded to
   `(frames-1)*882 + 882`.
2. `fc1` (882 → 896, no bias), `fc2` (896 → 896, bias).
3. 24-layer Qwen2 encoder (same architecture as the decoder's, different weights), final
   `decoder.norm` equivalent = `audio.encoder.encoder.norm`.
4. 4-frame aggregator: pad frames to a multiple of 4, group per 4, prepend `cls_embed`, run the
   4-layer aggregator, keep the CLS row per group.
5. `fc3` (896 → 128), split mean/logvar, take the **mean** (deterministic, as the Python
   round-trip does).
6. Feed those latents into the existing prompt assembly (already verified to 3.3e-07).

Acceptance: encode a 3 s WAV in C++ and match the Python encoder's latents to ~1e-05; the
downstream prompt matrix then follows from the already-verified assembly.

**Open decision (needs a human call):** with two reference clips there is one `<spk>` row, and
the source does not specify which 192-dim CAMPPlus vector to supply. Options: encode from the
first clip, from the concatenation, or require the caller to pass one embedding. CAMPPlus is not
ported either, so the first feature cut may keep `spk_head(zeros)` (the documented
missing-reference case) and be honest that speaker identity comes from the reference *latents*.

## 2. Longer dialogue within the fixed 20 s fragment cap

`old_shift` gives no end-of-sentence decision, so length is a cap: 250 patches × 80 ms = **20 s**
per fragment. A dialogue longer than that must be split into multiple fragments, each generated
as its own run, then concatenated.

Work:

1. A splitter that takes the tagged script and groups lines into fragments under the cap
   (leaving headroom, e.g. 18 s), never splitting mid-line.
2. Each fragment generated with the same references and seed policy; cross-fade or butt-join the
   fragments, watching for discontinuities at the joins.
3. Decide whether later fragments keep the reference conditioning identical (so voices stay
   stable across fragments). This is the main quality risk of splitting.

Acceptance: a dialogue longer than 20 s renders as joined fragments with the same two voices and
no audible seam at the joins.

## 3. GPU acceleration

Resume at the step the earlier work reached, with its remaining gate:

1. **One BF16 expert read directly from a resident bank at a device offset** (no host slicing),
   GEMM in BF16, matched against the CPU expert (~5e-08 target).
2. All 23 MoE layers through that path, values compared, timed warm step and peak VRAM against
   the **3.02 s** CPU baseline.
3. Only then judge cuBLAS-hybrid versus the ggml graph port — and if the graph route, verify KDA
   and routing actually execute on CUDA rather than silently falling back.

Note: the podcast path adds encoder work and reference conditioning to every fragment, so the GPU
work should be judged on the *render* time (encode + prefill + patches + decode), not just the
thinker step.

## Ordering rationale

Reference WAVs first: without them the feature cannot be used by anyone but us, and the encoder
is pure porting with a clear acceptance test. Splitting second, because it changes how generation
is *driven* and would otherwise be re-tested after every speed change. GPU last, because it
touches everything and would multiply the cost of re-testing the other two.
