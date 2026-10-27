#!/usr/bin/env python3
"""Flow (CFM/DiT) reference for Ming-3.0-tts-tiny, one latent step.

Everything is conditioned on captured thinker output and an explicitly seeded noise
tensor, so C++ can be fed exactly the same inputs and compared per solver step.

Contract (from config.json and CFM.py/dit.py):

  z_channels 64, hidden 1024, depth 16, heads 16, head_dim 64, mlp_ratio 4
  latent_history 1 x 64, patch_size 1
  timesteps   get_epss_timesteps(10) = [0,2,4,6,8,12,16,20,24,28,32] / 32
  sway        t + coef*(cos(pi/2 t) - 1 + t), coef = -1
  solver      Euler; SDE shift = sigma*sqrt(temperature)*sqrt|dt|*noise -> 0 when
              temperature = 0, which makes the whole step deterministic
  DiT input   [t + c, x_history, x_now]  (sequence length 3)

    python flow_reference.py --model models/ming3-tts-tiny ^
        --hidden out/reference/thinker/thinker-41.npz --seed 1234 ^
        --out out/reference/flow
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

PREFIX = "flowloss.cfm.model."
EPS = 1e-6


class FlowWeights:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.map = {name: shard for name, shard in index["weight_map"].items() if name.startswith(PREFIX)}
        self._handles = {}

    def get(self, suffix: str) -> torch.Tensor:
        name = PREFIX + suffix
        shard = self.map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.root / shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(name).float()

    def linear(self, x: torch.Tensor, suffix: str) -> torch.Tensor:
        return F.linear(x, self.get(suffix + ".weight"), self.get(suffix + ".bias"))


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + EPS)
    return (value * weight).to(dtype)


def rotary_tables(tokens: int, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [T, head_dim], interleaved, exactly as x_transformers builds them.

    x_transformers.RotaryEmbedding.forward stacks (freqs, freqs) on the last axis and
    flattens, so each frequency appears twice consecutively: [f0,f0,f1,f1,...].
    """

    positions = torch.arange(tokens, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.float())                       # [T, Dh/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)                         # [T, Dh]
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,H,T,Dh]; x_transformers.rotate_half pairs adjacent (2j, 2j+1) elements."""

    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([-x2, x1], dim=-1).reshape(x.shape)
    return x * cos + rotated * sin


def time_embed(timestep: torch.Tensor, weights: FlowWeights) -> torch.Tensor:
    half = 256 // 2
    emb = math.log(10000) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    angles = 1000.0 * timestep.view(-1, 1) * freqs.view(1, -1)
    table = torch.cat([angles.sin(), angles.cos()], dim=-1)   # [B, 256]
    hidden = F.silu(weights.linear(table, "t_embedder.time_mlp.0"))
    return weights.linear(hidden, "t_embedder.time_mlp.2")


def dit_forward(x: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor, history: torch.Tensor,
                weights: FlowWeights, capture: dict | None = None) -> torch.Tensor:
    """x [B,1,64] noise latent, cond [B,1,1536] thinker hidden, history [B,1,64]."""

    t_emb = time_embed(timestep, weights).unsqueeze(1)          # [B,1,1024]
    x_now = weights.linear(x, "x_embedder")                     # [B,1,1024]
    x_hist = weights.linear(history, "x_embedder")
    c = weights.linear(cond, "c_embedder.cond_embedder")        # [B,1,1024]
    if capture is not None:
        capture["cond_post_c_embedder"] = c.detach().numpy()
    tokens = torch.cat([t_emb + c, x_hist, x_now], dim=1)       # [B,3,1024]

    cos, sin = rotary_tables(tokens.shape[1], weights.get("rotary_embed.inv_freq"))
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,Dh]

    for layer in range(16):
        block = f"blocks.{layer}."
        h = rms_norm(tokens, weights.get(block + "norm1.weight"))
        q = weights.linear(h, block + "attn.to_q").view(x.shape[0], -1, 16, 64).transpose(1, 2)
        k = weights.linear(h, block + "attn.to_k").view(x.shape[0], -1, 16, 64).transpose(1, 2)
        v = weights.linear(h, block + "attn.to_v").view(x.shape[0], -1, 16, 64).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(x.shape[0], -1, 1024)
        tokens = tokens + weights.linear(attn, block + "attn.to_out.0")

        h = rms_norm(tokens, weights.get(block + "norm2.weight"))
        h = weights.linear(h, block + "mlp.ff.0.0")
        h = F.gelu(h, approximate="tanh")
        h = weights.linear(h, block + "mlp.ff.2")
        tokens = tokens + h

    out = rms_norm(tokens, weights.get("final_layer.norm_final.weight"))
    out = F.linear(out, weights.get("final_layer.linear.weight"), weights.get("final_layer.linear.bias"))
    if capture is not None:
        capture["tokens_out"] = tokens.detach().numpy()
    return out


def epss_timesteps(steps: int) -> torch.Tensor:
    table = {5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
             10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]}
    if steps not in table:
        return torch.linspace(0, 1, steps + 1)
    return torch.tensor(table[steps], dtype=torch.float32) / 32.0


def flow_sample(cond: torch.Tensor, noise: torch.Tensor, history: torch.Tensor, weights: FlowWeights,
                steps: int = 10, cfg: float = 1.0, sigma: float = 0.25, temperature: float = 0.0,
                sway: float = -1.0) -> dict:
    # condition after c_embedder is step-independent: compute it once
    cond_post = weights.linear(cond, "c_embedder.cond_embedder").detach().numpy()

    y = noise.transpose(1, 2)                                   # [B,1,64]
    t = epss_timesteps(steps)
    t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)

    trajectory, velocities, velocities_pre, times = [y.clone()], [], [], []
    capture_before = {}
    for t0, t1 in zip(t[:-1], t[1:]):
        if cfg == 1.0:
            pred = dit_forward(y, t0.expand(1), cond, history, weights, capture_before)[:, -1:, :]
        else:
            both = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            y2 = torch.cat([y, y], dim=0)
            h2 = torch.cat([history, history], dim=0)
            out = dit_forward(y2, t0.expand(2), both, h2, weights, capture_before)[:, -1:, :]
            pred, null_pred = torch.chunk(out, 2, dim=0)
            velocities_pre.append(pred.detach().numpy())
            pred = pred + (pred - null_pred) * cfg
        dt = t1 - t0
        y = y + dt * pred
        if temperature > 0:
            shift = sigma * (temperature ** 0.5) * (abs(dt) ** 0.5) * torch.randn(y.shape)
            y = y + shift
        times.append(float(t0))
        velocities.append(pred.detach().numpy())
        trajectory.append(y.clone())
    return {"times": times, "velocities": velocities, "velocities_pre_cfg": velocities_pre,
            "trajectory": trajectory, "latents": trajectory[-1].detach().numpy(),
            "cond_post": cond_post}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--hidden", required=True, type=Path, help="thinker npz with final_norm")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.model.resolve()
    weights = FlowWeights(root)
    data = np.load(args.hidden)
    cond = torch.from_numpy(data["final_norm"][:, -1:, :]).float()   # last position, [1,1,1536]
    condition = torch.zeros(1, 1, 64)
    torch.manual_seed(args.seed)
    noise = torch.randn(1, 64, 1)

    result = flow_sample(cond, noise, condition, weights, steps=args.steps, cfg=args.cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "cond": cond.numpy(), "noise": noise.numpy(),
        "times": np.array(result["times"], dtype=np.float32),
        "latents": result["latents"],
        "trajectory": np.concatenate([x.numpy() for x in result["trajectory"]], axis=0),
        "velocities": np.concatenate(result["velocities"], axis=0),
    }
    if result["cond_post"] is not None:
        arrays["cond_post_c_embedder"] = result["cond_post"]
    if result["velocities_pre_cfg"]:
        arrays["velocities_pre_cfg"] = np.concatenate(result["velocities_pre_cfg"], axis=0)
    np.savez_compressed(args.out / "flow-step.npz", **arrays)
    print(f"cond            : {tuple(cond.shape)} (last thinker position)")
    print(f"noise           : {tuple(noise.shape)} seed={args.seed}")
    print(f"timesteps       : {len(result['times'])} -> {[round(t, 4) for t in result['times'][:4]]}...")
    print(f"latents         : {result['latents'].shape} rms={float(np.sqrt((result['latents']**2).mean())):.5f}")
    print(f"first velocity  : {np.round(result['velocities'][0].reshape(-1)[:6], 5).tolist()}")
    print(f"latent[0,:8]    : {np.round(result['latents'].reshape(-1)[:8], 5).tolist()}")
    print(f"written         : {args.out / 'flow-step.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
