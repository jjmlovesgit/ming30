# KV cache: correct (bitwise) but slower without weight caching

## Run attribution

Build and comparison were run as separate steps, with the binary identity recorded:

```
build            exit 0, binary modified 22:51:24, sha256 3EEEB5DD666C5871...
before run       binary modified 22:51:24, sha256 3EEEB5DD666C5871... (unchanged)
run              exit code 1 (the family deliberately refuses inference after the probes)
output           full stdout/stderr captured to out/reference/seed-patch17/cache-check.log
```

The cache check itself completed; the non-zero exit comes from the probe's final
`InvalidRequestError`, which is expected behaviour for this family.

## Result — 58 tokens, prefill 41 + 17 steps

```
cache timings : full 27.9 s | prefill 19.4 s + 17 steps 76.1 s (4.48 s/step)
cache check   : tokens=58 (prefill 41 + 17 steps) last-token max_abs=0.000e+00 rel_rmse=0.000e+00
```

**Bitwise identical.** That is the expected outcome, not a suspicious one: the cached path
repeats exactly the same operations in the same order — same projection summations, same
convolution taps (the step feeds the three cached rows plus the new one), same KDA
recurrence order, and MLA attention over the same keys in the same order. Any difference
would have indicated a state bug; there is none. KDA recurrent state, KDA convolution
history and MLA key/value state are all being carried correctly.

## The surprise: caching alone is a pessimisation

| path | time | per token |
|---|---|---|
| full recompute, 58 tokens | 27.9 s | 0.48 s |
| prefill, 41 tokens | 19.4 s | 0.47 s |
| cached steps, 17 × 1 token | 76.1 s | **4.48 s** |

A single-token step costs ~9x more than a token inside a fresh pass. The reason is that
every step reloads and converts all layer weights from the package; in a full pass that
cost is amortised over the whole sequence. So the KV cache removes redundant *arithmetic*
but not redundant *I/O*, and I/O dominates at one token per call.

Implication for the speed work: **weight caching (and/or keeping weights resident on the
GPU) is the real lever**, not the KV cache by itself. The KV cache is nevertheless
required for practical decode, because it makes each step O(1) in sequence length; it just
has to be paired with resident weights to pay off.

## Status against the plan

1. KV cache for both attention types — **done and verified bitwise**.
2. Compare cached vs full on the seeded patch before CUDA — **done** (this note).
3. Next: weight residency (CPU first, then CUDA) with the same seeded comparison, then
   time/VRAM on GPU.
4. After any shared-engine change: re-run the `ming3_tts` 0.5B command and check the
   sentence is still the intended one, not only level and sample count.
