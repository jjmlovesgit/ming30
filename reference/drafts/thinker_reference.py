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


def rope_tables(tokens: int, rotary_dim: int, theta: float, device):
    """Interleaved RoPE tables (rope_interleave=True): cos/sin of shape [T, rotary_dim/2]."""

    positions = torch.arange(tokens, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent element pairs, which is what rope_interleave=True means."""

    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = pairs[..., 0], pairs[..., 1]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).reshape(*x.shape)


def mla_layer(hidden: torch.Tensor, store: "WeightStore", cfg: dict, layer: int) -> torch.Tensor:
    """BailingMoeV3MultiLatentAttention: LoRA q/kv, rope on 64 dims, SDPA, head-wise gate."""

    prefix = f"model.model.layers.{layer}.attention."
    heads = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    kv_lora = cfg["kv_lora_rank"]
    eps = cfg["rms_norm_eps"]
    batch, tokens, _ = hidden.shape

    q = rms_norm(F.linear(hidden, store.get(prefix + "q_a_proj.weight")),
                 store.get(prefix + "q_a_layernorm.weight"), eps)
    q = F.linear(q, store.get(prefix + "q_b_proj.weight")).view(batch, tokens, heads, qk_nope + qk_rope)
    q = q.transpose(1, 2)                                    # [B, H, T, nope+rope]
    q_pass, q_rot = q[..., :qk_nope], q[..., qk_nope:]

    kv = F.linear(hidden, store.get(prefix + "kv_a_proj_with_mqa.weight"))
    compressed, k_rot = kv[..., :kv_lora], kv[..., kv_lora:]
    compressed = rms_norm(compressed, store.get(prefix + "kv_a_layernorm.weight"), eps)
    k_pass = F.linear(compressed, store.get(prefix + "kv_b_proj.weight"))
    k_pass = k_pass.view(batch, tokens, heads, qk_nope + v_head).transpose(1, 2)
    k_nope, values = k_pass[..., :qk_nope], k_pass[..., qk_nope:]
    k_rot = k_rot.view(batch, 1, tokens, qk_rope)

    cos, sin = rope_tables(tokens, qk_rope, cfg["rope_theta"], hidden.device)
    q_rot = apply_rope_interleaved(q_rot, cos, sin)
    k_rot = apply_rope_interleaved(k_rot, cos, sin).expand(batch, heads, tokens, qk_rope)

    query = torch.cat([q_pass, q_rot], dim=-1)
    key = torch.cat([k_nope, k_rot], dim=-1)
    attn = F.scaled_dot_product_attention(query, key, values, is_causal=True,
                                          scale=(qk_nope + qk_rope) ** -0.5)
    attn = attn.transpose(1, 2)                              # [B, T, H, v_head]

    gate = torch.sigmoid(F.linear(hidden, store.get(prefix + "g_proj.weight")))  # head_wise
    attn = attn * gate[..., None]
    return F.linear(attn.reshape(batch, tokens, heads * v_head), store.get(prefix + "dense.weight"))


def attention_kind(cfg: dict, layer: int) -> str:
    """Hybrid: every fourth layer (3, 7, 11, ...) is MLA, the rest are KDA."""

    return "mla" if (layer % 4 == 3) else "kda"


def mlp_forward(x: torch.Tensor, store: "WeightStore", prefix: str) -> torch.Tensor:
    gate = F.linear(x, store.get(prefix + ".gate_proj.weight"))
    up = F.linear(x, store.get(prefix + ".up_proj.weight"))
    return F.linear(F.silu(gate) * up, store.get(prefix + ".down_proj.weight"))


def moe_forward(hidden: torch.Tensor, store: "WeightStore", cfg: dict, layer: int):
    """BailingMoeV3SparseMoeBlock: routed top-k + shared expert on the same input."""

    prefix = f"model.model.layers.{layer}.mlp"
    x = hidden.reshape(-1, hidden.shape[-1])
    top_idx, weights, logits = router(
        x, store.get(prefix + ".gate.weight"), store.get(prefix + ".gate.expert_bias"),
        cfg["n_group"], cfg["topk_group"], cfg["num_experts_per_tok"], cfg["routed_scaling_factor"])

    y = torch.zeros_like(x)
    for slot in range(cfg["num_experts_per_tok"]):
        idx = top_idx[:, slot]
        weight = weights[:, slot]
        for expert in idx.unique().tolist():
            mask = idx == expert
            expert_prefix = f"{prefix}.experts.{expert}"
            contribution = mlp_forward(x[mask], store, expert_prefix)
            y[mask] += contribution * weight[mask].unsqueeze(-1)

    y = y + mlp_forward(x, store, prefix + ".shared_experts")
    return y.view_as(hidden), logits, top_idx


def run_thinker(model: Path, text: str, prompt_ids: list[int], out_dir: Path) -> int:
    """Full 24-layer forward with a capture point per layer."""

    import numpy as np

    store = WeightStore(model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    cfg = config["llm_config"]
    eps = cfg["rms_norm_eps"]

    ids = torch.tensor([prompt_ids], dtype=torch.long)
    hidden = store.get("model.model.word_embeddings.weight")[ids]        # [1, T, 1536]
    captures = {"embedding": hidden.clone()}
    router_captures = {}

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.model.layers.{layer}."
        residual = hidden
        normed = rms_norm(hidden, store.get(prefix + "input_layernorm.weight"), eps)
        if attention_kind(cfg, layer) == "mla":
            hidden = residual + mla_layer(normed, store, cfg, layer)
        else:
            hidden = residual + kda_layer(normed, store, cfg, layer)

        residual = hidden
        normed = rms_norm(hidden, store.get(prefix + "post_attention_layernorm.weight"), eps)
        if layer < cfg["first_k_dense_replace"]:
            block = mlp_forward(normed, store, prefix + "mlp")
        else:
            block, logits, top_idx = moe_forward(normed, store, cfg, layer)
            router_captures[layer] = {
                "logits": logits.detach().numpy(),
                "top_idx": top_idx.detach().numpy(),
            }
        hidden = residual + block
        captures[f"layer_{layer}"] = hidden.clone()

    final = rms_norm(hidden, store.get("model.model.norm.weight"), eps)
    stop_logits = F.linear(final[:, -1], store.get("stop_head.weight"), store.get("stop_head.bias"))

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().float().numpy() for name, tensor in captures.items()}
    arrays["final_norm"] = final.detach().float().numpy()
    arrays["stop_logits"] = stop_logits.detach().float().numpy()
    np.savez_compressed(out_dir / "thinker-41.npz", **arrays)

    manifest = {
        "stage": "B_thinker",
        "text": text,
        "prompt_ids": prompt_ids,
        "token_count": len(prompt_ids),
        "layers": cfg["num_hidden_layers"],
        "capture_points": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "mean": float(tensor.mean()), "std": float(tensor.std()),
                   "l2": float(tensor.norm())}
            for name, tensor in [("embedding", captures["embedding"]),
                                 ("final_norm", final),
                                 ("stop_logits", stop_logits)]
        },
        "layer_checksums": {
            name: {"mean": float(t.mean()), "std": float(t.std()), "l2": float(t.norm())}
            for name, t in captures.items() if name.startswith("layer_")
        },
        "router_layers": {str(k): {"top_idx_first_token": v["top_idx"][0].tolist()} for k, v in router_captures.items()},
        "stop_logits": stop_logits.detach().numpy().tolist(),
        "stop_decision": int(torch.argmax(stop_logits, dim=-1).item()),
    }
    (out_dir / "thinker-41.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"prompt tokens      : {len(prompt_ids)}")
    print(f"embedding          : {tuple(captures['embedding'].shape)} {captures['embedding'].dtype}")
    for layer in (0, 1, 11, 23):
        t = captures[f"layer_{layer}"]
        print(f"layer {layer:>2} hidden    : {tuple(t.shape)} {t.dtype} l2={t.norm().item():.4f}")
    print(f"final_norm         : {tuple(final.shape)} {final.dtype} l2={final.norm().item():.4f}")
    print(f"stop_logits        : {stop_logits.tolist()} decision={manifest['stop_decision']}")
    print(f"expert choices L1  : {router_captures[1]['top_idx'][0].tolist()}")
    print(f"written            : {out_dir/'thinker-41.npz'}")
    return 0


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
    parser.add_argument("--forward", action="store_true", help="run the full thinker forward")
    parser.add_argument("--out", type=Path, default=Path("out/reference"))
    args = parser.parse_args()
    root = args.model.resolve()
    if args.forward:
        ref = json.loads(Path(r"C:\Projects\ming30\out\reference\stage-a-segments.json").read_text(encoding="utf-8"))
        return run_thinker(root, "The garden will remain open until sunset.", ref["flat_ids"], args.out)
    if args.kda:
        return smoke_kda(root, args.layer)
    return smoke(root, args.layer)


if __name__ == "__main__":
    sys.exit(main())
