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

## Conventions pinned to source (not assumed)

Two behaviours decide whether agreement between C++ and Python means anything, so both
are pinned to the checkpoint's own dependencies rather than to my reading of them.

**RoPE — interleaved, from `x_transformers`:**

| reference | what it says |
|---|---|
| `x_transformers/x_transformers.py` L835 `RotaryEmbedding.__init__` | `inv_freq = 1/(base ** (arange(0, dim, 2)/dim))`, `base = 10000` |
| same, `forward()` | `freqs = einsum(t, inv_freq)`; `freqs = stack((freqs, freqs), -1)` then flattened — each frequency appears **twice consecutively**, i.e. interleaved |
| same, L893 `rotate_half` | `rearrange('... (d r) -> ... d r', r=2)` then `stack((-x2, x1))` — rotate **adjacent pairs** `(2j, 2j+1)` |
| same, L900 `apply_rotary_pos_emb` | `t*cos(freqs) + rotate_half(t)*sin(freqs)` |
| `modules.py` L~100 | `RotaryEmbedding(hidden_size // num_heads)` with defaults; `pe_attn_head` is null so RoPE applies to **all** heads |
| `dit.py` L114 | same construction, `dim = 1024 // 16 = 64` |
| checkpoint tensor | `flowloss.cfm.model.rotary_embed.inv_freq`, shape `[32]` — matches `arange(0,64,2)` |
| `use_xpos = False` (default) | `forward` returns `(freqs, 1.)`, so `apply_rotary_pos_emb` runs with `scale = 1` |

**Correction:** my first implementation used the rotate-half (halves) convention. That was
wrong against this source. Switching to interleaved changed the result — latents rms
`0.81920 → 0.81314` — which is exactly the kind of silent convention error the pinning was
meant to catch.

**Sway and epss schedule — defaults, not configured:**

| reference | what it says |
|---|---|
| `CFM.py` L~133 `CFM.sample` signature | `sway_sampling_coef=-1.0`, `use_epss=True`, `t_start=0` |
| `CFM.py` `get_epss_timesteps` | `10 -> [0,2,4,6,8,12,16,20,24,28,32] / 32` |
| `CFM.py` sway application | `t = t + coef*(cos(pi/2*t) - 1 + t)`, applied before integration |
| `flowloss.py` `FlowLoss.sample` | forwards `cfg_scale, patch_size, sigma, temperature, steps, generator` only — **no sway or epss override** |
| `modeling_lingv3_tts.py` L601 | calls `flowloss.sample(...)` with no sway argument, so the CFM defaults apply |

## Captures for the C++ comparison

`out/reference/flow/flow-step.npz` (cfg = 1) and `flow-cfg2/flow-step.npz` (cfg = 2):

| capture | shape | note |
|---|---|---|
| `cond` | (1,1,1536) | thinker hidden, last position |
| `cond_post_c_embedder` | (1,1,1024) | step-independent; rms 1.1447; **identical across cfg runs** |
| `noise` | (1,64,1) | seed 1234, the literal tensor both sides must use |
| `times` | (10,) | swayed grid |
| `velocities` | (10,1,64) | post-guidance velocity per step |
| `velocities_pre_cfg` | (10,1,64) | **only at cfg > 1**; max abs pre-vs-post at step 0 = 2.514 |
| `trajectory` | (11,1,64) | `x` after each Euler step, including the initial `y0` |
| `latents` | (1,1,64) | final latent; rms 0.81314 (cfg 1) / 0.82075 (cfg 2) |

## Next

Port the DiT and Euler solver to C++ using the interleaved RoPE and the CFM default
sway/epss schedule above, then compare `cond_post_c_embedder`, every step's velocity,
the trajectory and the final latents — then the VAE with identical latents.
