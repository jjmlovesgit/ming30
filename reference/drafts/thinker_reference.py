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


FLA_REF = Path(__file__).resolve().parents[2] / "reference" / "fla"


def kda_gate(raw_g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """FLA's lower-bounded KDA gate (kda_safe_gate=True, kda_lower_bound=-5.0).

    g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias)), computed in float32.
    """

    heads, head_dim = raw_g.shape[-2:]
    g = raw_g.float() + dt_bias.view(heads, head_dim).float()
    return lower_bound * torch.sigmoid(A_log.view(heads, 1).float().exp() * g)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.float().pow(2).sum(-1, keepdim=True).sqrt() + eps)


def gated_rms_norm(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FusedRMSNormGated(activation='sigmoid'): y = rms_norm(x) * weight * sigmoid(gate)."""

    normed = rms_norm(x, weight, eps)
    return normed * torch.sigmoid(gate.float())


def load_naive_chunk_kda():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fla_naive_kda", FLA_REF / "naive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_chunk_kda, module.naive_recurrent_kda


def kda_layer(hidden: torch.Tensor, store: "WeightStore", cfg: dict, layer: int,
              chunk_size: int = 64) -> torch.Tensor:
    """One KDA attention block, following modeling_bailing_moe_v3.

    q/k/v -> silu short conv -> l2norm(q,k) -> gate -> chunked KDA -> gated RMS norm -> o_proj
    """

    prefix = f"model.model.layers.{layer}.attention."
    eps = cfg["rms_norm_eps"]
    heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    tokens = hidden.shape[1]

    # pad to a whole number of chunks (causal, so trailing padding cannot affect
    # the outputs at real positions)
    pad = (-tokens) % chunk_size
    if pad:
        hidden = torch.cat([hidden, hidden.new_zeros(hidden.shape[0], pad, hidden.shape[2])], dim=1)
    total = hidden.shape[1]

    q = short_conv1d(F.linear(hidden, store.get(prefix + "q_proj.weight")), store.get(prefix + "q_conv1d.weight"))
    k = short_conv1d(F.linear(hidden, store.get(prefix + "k_proj.weight")), store.get(prefix + "k_conv1d.weight"))
    v = short_conv1d(F.linear(hidden, store.get(prefix + "v_proj.weight")), store.get(prefix + "v_conv1d.weight"))
    q, k, v = torch.nn.functional.silu(q), torch.nn.functional.silu(k), torch.nn.functional.silu(v)

    q = l2norm(q.view(1, total, heads, head_dim))
    k = l2norm(k.view(1, total, heads, head_dim))
    v = v.view(1, total, heads, head_dim)

    raw_g = F.linear(hidden, store.get(prefix + "f_proj.weight")).view(1, total, heads, head_dim)
    g = kda_gate(raw_g, store.get(prefix + "A_log"), store.get(prefix + "dt_bias"), cfg["kda_lower_bound"])
    beta = torch.sigmoid(F.linear(hidden, store.get(prefix + "b_proj.weight"))).view(1, total, heads)

    naive_chunk_kda, _ = load_naive_chunk_kda()
    o, _state = naive_chunk_kda(q, k, v, g, beta, chunk_size=chunk_size)

    gate_out = F.linear(hidden, store.get(prefix + "g_proj.weight")).view(1, total, heads, head_dim)
    o = gated_rms_norm(o, store.get(prefix + "o_norm.weight"), gate_out, eps)
    o = F.linear(o.reshape(1, total, heads * head_dim), store.get(prefix + "o_proj.weight"))
    return o[:, :tokens]


def smoke_kda(model: Path, layer: int = 1) -> int:
    store = WeightStore(model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    llm = config["llm_config"]
    torch.manual_seed(0)
    hidden = torch.randn(1, 41, llm["hidden_size"]) * 0.1

    out = kda_layer(hidden, store, llm, layer)
    print(f"kda_layer          shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"rms={out.pow(2).mean().sqrt().item():.5f}")

    # FLA's own chunk path is the reference; compare it to FLA's recurrent path to
    # document the numerical gap rather than to assert equality.
    naive_chunk_kda, naive_recurrent_kda = load_naive_chunk_kda()
    heads, head_dim, tokens = llm["num_attention_heads"], llm["head_dim"], 64
    torch.manual_seed(1)
    q = torch.randn(1, tokens, heads, head_dim) * 0.1
    k = torch.randn(1, tokens, heads, head_dim) * 0.1
    v = torch.randn(1, tokens, heads, head_dim) * 0.1
    g = -torch.rand(1, tokens, heads, head_dim)
    beta = torch.rand(1, tokens, heads)
    chunk_out, chunk_state = naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=64)
    rec_out, _ = naive_recurrent_kda(q, k, v, g, beta)
    gap = (chunk_out - rec_out).abs().max().item()
    scale = chunk_out.abs().max().item()
    print(f"chunk vs recurrent max|diff|={gap:.3e} (signal max {scale:.3f}, relative {gap/max(scale,1e-9):.3e})")
    ok = tuple(out.shape) == (1, 41, llm["hidden_size"]) and torch.isfinite(out).all().item()
    print("kda smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


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
    parser.add_argument("--kda", action="store_true", help="run the KDA layer smoke test")
    args = parser.parse_args()
    if args.kda:
        return smoke_kda(args.model.resolve(), args.layer)
    return smoke(args.model.resolve(), args.layer)


if __name__ == "__main__":
    sys.exit(main())
