# final_norm investigated: the norm is exact; the 4.5% is concentrated on flipped tokens

## 1. Seeded test — the norm implementation is sound

C++ final norm fed the **exact Python layer-23 hidden state** (`MING3_TINY_SEED_LAYER=24`
runs zero layers):

```
seeded from Python layer_23   final_norm rel_rmse=4.466e-08  max_abs=3.576e-07
seeded stop_logits            ref=[12.66214, 6.66677] cpp=[12.66215, 6.66677] max_abs=6.7e-06
```

Formula, epsilon, weight mapping and normalisation axis are all correct. The 4.5% is
**entirely inherited**, not a norm bug.

## 2. Where the inherited error actually lives

Whole-tensor figures hide the structure. Per token, on `layer_23` (the norm's input):

| token group | hidden rms | hidden abs err | hidden rel err |
|---|---|---|---|
| 0, 4, 8, 12, 24, 36 | 1.18 – 7.37 | 1.5e-05 – 1.0e-02 | 4e-07 – 2e-03 |
| 16, 20, 28, 32, 40 | 0.12 – 0.30 | 1.6e-03 – 2.0e-02 | 1.3e-03 – 2.1e-02 |
| **17, 19, 31** | 0.15 – 0.25 | **1.0e-01 – 1.2e-01** | **8.9e-02 – 2.4e-01** |

Elementwise across the whole tensor: **median relative error 8.8e-04**, p90 4.7e-02.
The error is not concentrated in outlier channels (top-10 channels hold 0.5% of the
error energy); it is concentrated in *particular tokens*.

## 3. Those tokens are exactly the ones whose experts flipped

Expert-set flips by layer:

```
3:[35] 4:[15,39] 5:[5] 6:[7] 7:[19] 8:[19] 11:[17,19] 13:[31,32] 15:[17]
17:[17,19,38] 18:[19,38] 20:[20,31] 21:[5,17,19,31] 22:[17,19,32] 23:[17,19]
```

- Worst layer-23 tokens by relative error: `[19, 31, 17, 32, 35, 20, 39, 33]`
- Tokens touched by at least one flip: `[5, 7, 15, 17, 19, 20, 31, 32, 35, 38, 39]`
- Tokens with relative error > 1%: `[17, 19, 20, 31, 32, 33, 35, 38, 39]` — all but 33
  are flipped tokens
- **Untouched tokens (30 of 41): median relative error 3.3e-04, max 1.0e-02**

Tokens 17 and 19 flip at *seven* and *eight* different layers respectively; each flip
substitutes a different expert, so their hidden states diverge cumulatively to 9–26%.

## 4. What this means before wiring flow

The honest summary is not "4.5% is harmless". It is:

- the final norm is exact;
- 30 of 41 tokens agree to ~3e-04 relative — genuinely tight;
- 11 tokens diverge, up to 26%, because a narrow routing cutoff flipped and a different
  expert's output entered their residual stream, repeatedly for 17 and 19;
- the stop decision is unchanged and the last token (its input) agrees to 8.3e-03.

Whether that is audible depends on how the talker consumes the thinker's hidden states.
If it consumes all token positions, 11 tokens carry a materially different residual — not
a rounding difference. If it consumes only the final position, the relevant error is
8.3e-03.

## 5. Recommendation

**Fix the root cause before wiring flow: implement `chunk_kda` in C++.**

The seeded MLA experiment showed the C++ path agrees to 3e-07 when its input is exact.
The KDA recurrence-vs-chunk ordering is what injects ~2e-04 per layer, and that drift is
what crosses the 2.6e-06–8.1e-04 selection margins. Removing it should collapse the 28
flips the way seeding collapsed the layer-3 one, and it is bounded work: `naive_chunk_kda`
in `reference/fla/naive.py` is an exact, readable specification.

Accepting the current difference and letting audio quality arbitrate is the alternative,
but it trades a known, fixable cause for an ambiguous listening test.
