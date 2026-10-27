# VAE encoder ported: reference latents from a WAV, matching Python

## Pipeline (as implemented in `encoder.cpp`)

```
waveform
 -> unfold into 882-sample frames (ceil(samples/882), zero-padded tail)
 -> fc1 (882 -> 896, no bias), fc2 (896 -> 896, bias)
 -> 24-layer Qwen2 encoder (halves RoPE, GQA 14/2, SiLU MLP)
 -> pad frames to a multiple of 4, pack each group as [frame0..frame3, CLS], CLS row LAST
 -> 4-layer aggregator over the whole packed sequence (so a CLS also sees earlier groups)
 -> keep every 5th (CLS) row
 -> fc3 (896 -> 128), take the posterior mean (first 64)
```

## Two bugs the stage comparison found

1. **RoPE convention.** The encoder's Qwen2 uses the **halves** rotation, like the decoder —
   not the interleaved pairs used by the flow DiT. My first version applied interleaved
   everywhere, which is the same class of error as the decoder's earlier in-place bug: correct
   weights, wrong positions.
2. **Aggregator packing order.** The reference packs each group as `[4 frames, CLS]` and takes
   the **last** row. I had put the CLS first and then extracted row 4, so the "CLS" output was a
   frame.

## Parity, stage by stage (3 s clip, 150 frames, 38 latents)

| stage | max_abs | rel_rmse | corr |
|---|---|---|---|
| fc2 | 2.98e-07 | 6.10e-07 | 1.000000 |
| encoder (24 layers) | 1.37e-05 | 2.46e-06 | 1.000000 |
| aggregated (CLS rows) | 3.58e-06 | 1.30e-06 | 1.000000 |
| **latents** | **7.03e-06** | **1.51e-06** | 1.000000 |

The 24-layer encoder was already right before the aggregator fix, which is what made the
localisation quick: fc2 and encoder matched, so the fault had to be in packing or the
aggregator itself.

## Decisions applied

- `spk_head(zeros)` stays the convention for the first WAV-backed version, as directed — it is
  the same case the verified two-speaker render used, so the target is known.
- CAMPPlus remains unported; voice identity for now comes from the reference latents, and this
  is stated rather than implied.

## Still required for the acceptance test

The encoder currently runs behind a temporary probe (`MING3_TINY_ENCODE_WAV`). The acceptance
test the coordinator asked for is the **normal user path**: two WAVs, the tagged reference
transcript and the tagged script passed through the CLI with no saved-latent files and no test
environment variables, then two complete voices heard. That means:

1. family request options that accept reference WAV paths plus their transcript,
2. the loader/encoder wired into the session path rather than a probe,
3. the generation loop driven from those options, and
4. `spk_head(zeros)` applied at the speaker row.

Encoder maths is now verified; the plumbing for the normal path is the next step.
