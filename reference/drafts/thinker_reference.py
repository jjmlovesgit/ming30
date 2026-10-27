#!/usr/bin/env python3
"""Plain-torch thinker reference for Ming-3.0-tts-tiny (start).

Bypasses `atorch` and `fla` so it can run on Windows/CUDA with the torch that is
already installed. Scope of this first cut:

  * weight loading straight from the published shards
  * RMSNorm / GroupRMSNorm
  * causal depthwise short convolution (kernel 4) used by KDA
  * the group-limited top-k router, decoded from modeling_bailing_moe_v3.py

Deliberately NOT included yet: the KDA token-mixing itself. Upstream's own comment
says the fused recurrent kernel is "not numerically compatible with the checkpoint's
chunk path: short prompts collapse to punctuation", so a naive per-token recurrence
would produce a reference that disagrees with the released model. The faithful
options are to port `fla`'s chunk algorithm or to run the official stack under WSL.

    python thinker_reference.py --model <ckpt> --layer 1
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class WeightStore:
    """Lazy per-shard safetensors access, returning float32 tensors."""

    def __init__(self, root: Path, dtype: torch.dtype = torch.float32):
        self.root = root
        self.dtype = dtype
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        from safetensors import safe_open

        if shard not in self._handles:
            self._handles[shard] = safe_open(self.root / shard, framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(name)
        tensor = self._handle(self.weight_map[name]).get_tensor(name)
        return tensor.to(self.dtype)

    def has(self, name: str) -> bool:
        return name in self.weight_map


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normed.to(dtype))


def group_rms_norm(x: torch.Tensor, weight: torch.Tensor, group_norm_size: int, eps: float = 1e-6) -> torch.Tensor:
    """BailingMoeV3GroupRMSNorm: reshape last dim to (groups, inner), normalise inner."""

    dtype = x.dtype
    shape = x.shape
    if shape[-1] % group_norm_size:
        raise ValueError("hidden size must be divisible by group_norm_size")
    grouped = x.view(*shape[:-1], group_norm_size, shape[-1] // group_norm_size).float()
    variance = grouped.pow(2).mean(-1, keepdim=True)
    normed = grouped * torch.rsqrt(variance + eps)
    return weight * normed.to(dtype).view(shape)


def short_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Causal depthwise convolution, weight [channels, 1, k]."""

    context = weight.shape[-1]
    padded = F.pad(x.transpose(1, 2), (context - 1, 0))
    out = F.conv1d(padded, weight, groups=weight.shape[0])  # depthwise
    return out.transpose(1, 2)


def group_limited_topk(scores: torch.Tensor, n_group: int, topk_group: int, top_k: int):
    tokens, experts = scores.shape
    grouped = scores.view(tokens, n_group, experts // n_group)
    group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(tokens, n_group, experts // n_group)
        .reshape(tokens, experts)
    )
    masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_idx = torch.topk(masked, k=top_k, dim=-1)
    return probs, top_idx


def router(hidden: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor,
           n_group: int, topk_group: int, top_k: int, scaling: float):
    """BailingMoeV3Gate.forward, float32 throughout, faithful to the source."""

    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat.float(), weight.float())
    scores = torch.sigmoid(logits.float())
    _, top_idx = group_limited_topk(scores + expert_bias, n_group, topk_group, top_k)
    gathered = torch.gather(scores, dim=1, index=top_idx)
    if top_k > 1:
        weights = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        weights = gathered
    return top_idx, weights * scaling, logits


def smoke(model: Path, layer: int) -> int:
    store = WeightStore(model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    llm = config["llm_config"]
    eps = llm["rms_norm_eps"]
    hidden = llm["hidden_size"]

    print(f"weights: {len(store.weight_map)} tensors")
    ok = True

    # norms
    norm_weight = store.get(f"model.model.layers.{layer}.input_layernorm.weight")
    x = torch.randn(4, hidden)
    y = rms_norm(x, norm_weight, eps)
    mean_sq = y.float().pow(2).mean(-1)
    print(f"rms_norm           shape={tuple(y.shape)} mean_sq={mean_sq.mean():.4f}")
    ok &= tuple(y.shape) == (4, hidden) and torch.isfinite(y).all().item()

    # short convolution
    conv_weight = store.get(f"model.model.layers.{layer}.attention.q_conv1d.weight")
    xc = torch.randn(1, 7, conv_weight.shape[0])
    yc = short_conv1d(xc, conv_weight)
    print(f"short_conv1d       shape={tuple(yc.shape)} finite={torch.isfinite(yc).all().item()}")
    ok &= tuple(yc.shape) == tuple(xc.shape)

    # router on the real weights
    gate_weight = store.get(f"model.model.layers.{layer}.mlp.gate.weight")
    expert_bias = store.get(f"model.model.layers.{layer}.mlp.gate.expert_bias")
    h = torch.randn(3, hidden)
    top_idx, top_weights, logits = router(h, gate_weight, expert_bias,
        llm["n_group"], llm["topk_group"], llm["num_experts_per_tok"], llm["routed_scaling_factor"])
    print(f"router             logits={tuple(logits.shape)} topk={tuple(top_idx.shape)}")
    print(f"  experts          {top_idx[0].tolist()}")
    print(f"  weights          {[round(v, 4) for v in top_weights[0].tolist()]}")
    print(f"  weight sum       {top_weights[0].sum().item():.4f} (expect {llm['routed_scaling_factor']})")
    print(f"  distinct experts {len(set(top_idx[0].tolist()))}")
    ok &= top_idx.shape == (3, llm["num_experts_per_tok"])
    ok &= abs(top_weights[0].sum().item() - llm["routed_scaling_factor"]) < 1e-3
    ok &= len(set(top_idx[0].tolist())) == llm["num_experts_per_tok"]

    for name in (f"model.model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
                 f"model.model.layers.{layer}.mlp.experts.0.up_proj.weight"):
        t = store.get(name)
        print(f"  {name.split('.mlp.')[1]:<32} {tuple(t.shape)}")
        ok &= t.ndim == 2

    print("\nKDA token mixing: NOT implemented (chunk path required; see module docstring)")
    print("smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()
    return smoke(args.model.resolve(), args.layer)


if __name__ == "__main__":
    sys.exit(main())
