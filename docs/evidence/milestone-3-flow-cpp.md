# Stage C — C++ flow agrees with the reference, guidance branches included

C++ DiT (16 layers, 3-token sequence, interleaved RoPE) + swayed epss timesteps + Euler
solver, fed the exact reference condition and noise (`cond.f32`, `noise.f32` from the
Python npz). Captures: condition after `c_embedder`, conditioned velocity, unconditioned
velocity, guided velocity, per-step `x`, final latents.

## cfg = 1 (single pass)

```
PASS  cond_post_c_embedder   5.01e-06   7.05e-07
PASS  flow_velocity_guided   1.99e-05   1.81e-06
PASS  flow_latents           4.41e-06   1.99e-06
PASS  x[0..10]               0 .. 4.41e-06  (x[0] exact, growing smoothly)
latent rms 0.813142 on both sides
```

## cfg = 2 (guidance, both branches captured)

```
PASS  cond_post_c_embedder   5.01e-06   7.05e-07
PASS  flow_velocity_cond     2.24e-05   2.59e-06      conditioned branch
PASS  flow_velocity_null     2.19e-05   2.87e-06      unconditioned branch
PASS  flow_velocity_guided   3.25e-05   3.95e-06      combined
PASS  flow_latents           7.82e-06   3.39e-06
x[0..10]                     0 .. 7.82e-06
latent rms 0.820754 on both sides
```

Because the conditioned and unconditioned velocities are captured separately, a
disagreement in either branch is distinguishable from a disagreement in the combination;
here both branches and the guided value agree to ~2–3e-05, and the latents to 8e-06.

## Notes

- `cond_post_c_embedder` agrees to 5e-06 at both cfg values, which pins the conditioning
  projection independently of the solver.
- The latent RMS matching to six digits on both sides (0.813142, 0.820754) is a useful
  coarse check that the whole ten-step trajectory lands in the same place.
- The CLI still exits 1 by design: the family refuses inference and only runs probes.

## Status

Flow is verified at the same level as the thinker: intermediate values, not just the
endpoint. The VAE is the remaining stage before audio exists.

## Next: VAE

Feed both sides the identical latents captured here (`flow_latents`, a `[1,1,64]` patch)
with the same latent history, compare the decoded waveform — and capture the VAE's
intermediate stages (input projection, decoder block outputs) so a mismatch is located
inside the decoder rather than in the final samples.
