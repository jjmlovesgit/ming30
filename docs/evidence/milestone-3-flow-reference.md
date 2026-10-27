# Stage C — flow reference runs on captured thinker output with fixed noise

## The flow contract, from source

| piece | value |
|---|---|
| latent channels | 64 (`z_channels`, `latent_dim`) |
| patch size / history | 1 / 1 |
| DiT | hidden 1024, depth 16, heads 16, head_dim 64, mlp_ratio 4, pre-norm RMSNorm eps 1e-6 |
| conditioning | `c_embedder` Linear(1536 → 1024) on the thinker hidden at the last position |
| DiT input | `[t + c, x_history, x_now]` — **sequence length 3** per solver step |
| timesteps | `get_epss_timesteps(10) = [0,2,4,6,8,12,16,20,24,28,32]/32` |
| sway | `t + coef·(cos(π/2·t) − 1 + t)`, coef = −1 |
| solver | Euler with SDE shift `σ·√temperature·√|dt|·noise` |
| determinism | **temperature = 0 → shift = 0**, so the step is fully deterministic given the initial noise |
| CFG | at `cfg = 1` a single DiT pass; above 1 the model is run with real and zeroed conditioning and combined |
| blocks | attention (q/k/v with bias, RoPE, SDPA non-causal, out proj) + MLP (GELU tanh) with residuals |

236 flow tensors, all present in the package (`x_embedder`, `t_embedder`, `c_embedder`,
`rotary_embed.inv_freq`, 16 × 14 block tensors, `final_layer`).

## First run

```
cond            : (1, 1, 1536) (last thinker position)
noise           : (1, 64, 1) seed=1234
timesteps       : 10 -> [0.0, 0.0048, 0.0192, 0.0431]...
latents         : (1, 1, 64) rms=0.81920
first velocity  : [0.0709, -0.5788, -0.7500, 0.4210, 1.1906, -1.9558]
latent[0,:8]    : [-0.1286, -1.6031, -0.5125, -0.8437, 1.1533, -0.5984, -0.7370, -0.1452]
```

Saved to `out/reference/flow/flow-step.npz`: cond, noise, timesteps, per-step velocities,
trajectory, final latents. That is the exact target the C++ flow must reproduce from the
same cond and the same noise.

## Status and caveats

This is a **candidate** reference, in the same sense the thinker was before its staged
comparison: transcribed from `CFM.py`, `dit.py` and `modules.py`, running on real
weights, but not yet validated against released audio. Two conventions are implemented
from reading the code and will be mirrored exactly in C++ so that the comparison is
meaningful:

1. RoPE uses the rotate-half convention (freqs duplicated across halves), applied to all
   heads since `pe_attn_head` is null.
2. `sway_sampling_coef = -1` is applied to the timestep grid before integration.

If C++ and Python agree per step and the resulting audio is wrong, these two are the first
places to look — but they are cheap to flip and retest once the comparison harness exists.

## Next

1. Port the DiT forward and Euler solver to C++, mirroring these conventions.
2. Compare per step: velocity, `x`, and the final latents, given the same cond and noise.
3. Then the VAE: feed both sides identical latents and compare the waveform.
