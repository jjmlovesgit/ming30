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

## Correction: the cache is faster for the same task

The first version of this note compared 27.9 s for a 58-token full pass against 76.1 s for
17 cached steps and called caching a pessimisation. That comparison was wrong: the two runs
do not produce the same amount of work.

- the full pass produces **one** result, the hidden state at token 58;
- the cached run produces **seventeen** successive results (tokens 42 through 58).

For the token-58 result alone, one warm cached step costs **4.48 s** versus **27.9 s** to
recompute 58 tokens — about **6x faster**, and the gap widens with sequence length because
the cached step is O(1) in the number of prior tokens.

What the 4.48 s/step vs 0.48 s/token figure actually measures is **lost batching
efficiency**: in a full pass, one weight load feeds 58 tokens; in a one-token call it feeds
one. That is a real cost, but it is not an argument against caching — it is an argument for
keeping weights resident.

| measurement | value | what it means |
|---|---|---|
| full recompute, 58 tokens | 27.9 s | cost of producing the token-58 state by recomputation |
| one warm cached step | 4.48 s | cost of producing the same state incrementally |
| 17 cached steps | 76.1 s | 17 states, i.e. 4.48 s each |
| 0.48 s/token in a batch | — | batching efficiency, not the decode cost |

## Status against the plan

1. KV cache for both attention types — **done and verified bitwise**.
2. Compare cached vs full on the seeded patch before CUDA — **done** (this note).
3. Next: weight residency (CPU first, then CUDA) with the same seeded comparison, then
   time/VRAM on GPU.
4. After any shared-engine change: re-run the `ming3_tts` 0.5B command and check the
   sentence is still the intended one, not only level and sample count.
