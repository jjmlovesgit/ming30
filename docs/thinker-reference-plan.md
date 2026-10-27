# Thinker reference plan — plain torch, no atorch/no fla

Both sides (C++ family and a Python reference) must implement the same math. This
records what that math is, what already runs, and the one component that is blocked.

## The blocker, in upstream's own words

From `modeling_bailing_moe_v3.py` (BailingMoeV3KimiDeltaAttention.forward):

```
# The fused recurrent KDA kernel available in this training environment is not
# numerically compatible with the checkpoint's chunk path: short prompts collapse
# to punctuation while the same model is coherent once q_len > 64 selects chunk.
# Keep recurrent execution opt-in until it is aligned against source golden logits.
```

Consequence: **a naive per-token recurrence is not a valid reference.** The released
model runs `chunk_kda`; the recurrent form is mathematically the same recurrence but
numerically divergent in their kernels, badly enough to change output qualitatively.
Any plain-torch reference must port the **chunked** algorithm (from `fla`'s source),
not the recurrence. This also tells the C++ side what to implement — chunk semantics,
not a simple scan.

## Verified inventory (from the checkpoint, not inferred)

| component | layers | tensors (per layer) |
|---|---|---|
| **dense MLP** | 0 only (`first_k_dense_replace=1`) | `mlp.{gate,up,down}_proj` [4608,1536] / [1536,4608] |
| **MoE** | 1–23 (`moe_layer_freq` = 0 then 1…) | `mlp.gate.weight` [128,1536], `mlp.gate.expert_bias` [128] F32, `mlp.shared_experts.{gate,up,down}` [512,1536], `mlp.experts.{0..127}.{gate,up,down}` |
| **attention (KDA)** | 0–23 | `q/k/v_proj` [2048,1536], `q/k/v_conv1d` [2048,1,**4**], `f_proj` [2048,1536] (no LoRA: `no_kda_lora=True`), `b_proj` [16,1536], `A_log` [16] F32, `dt_bias` [2048] F32, `o_norm` [128], `o_proj` [1536,2048] |
| norms | every layer | `input_layernorm` [1536], `post_attention_layernorm` [1536] |
| embeddings / head | — | `word_embeddings` [157184,1536], `lm_head` [157184,1536] (**untied**), final `norm` [1536] |

Config: 24 layers, hidden 1536, **16 heads**, head_dim 128, v_head_dim 128,
128 experts, top-8, `n_group=8`, `topk_group=4`, `routed_scaling_factor=2.5`,
`rms_norm_eps=1e-6`, rope_theta 10000.

Note the attention is KDA on **every** layer — there are no MLA tensors in the
checkpoint (`q_a_proj`/`kv_a_proj` absent), so a C++ implementation can ignore
`BailingMoeV3MultiLatentAttention` for this model.

## What already runs (this is not a plan, it is executed)

`reference/drafts/thinker_reference.py` loads the real shards and runs:

```
weights: 10285 tensors
rms_norm           shape=(4, 1536) mean_sq=0.0584
short_conv1d       shape=(1, 7, 2048) finite=True          # causal depthwise, k=4
router             logits=(3, 128) topk=(3, 8)
  experts          [40, 101, 99, 42, 27, 23, 96, 98]
  weights          [0.3129, 0.3131, 0.3130, 0.3127, 0.3119, 0.3114, 0.3123, 0.3125]
  weight sum       2.5000 (expect 2.5)
  distinct experts 8
smoke: PASS
```

The router is transcribed from source (sigmoid → `+ expert_bias` → group-limited
top-k over `n_group=8`/`topk_group=4` → gather from unbiased scores → normalise →
× `routed_scaling_factor`). The weight sum hitting exactly 2.5 is a real check on
that transcription, not a tautology.

## Next implementation steps

1. **Port `chunk_kda` to plain torch.** Fetch `fla`'s KDA chunk implementation from
   its repository, transcribe the algorithm (not the Triton kernels) into torch, and
   validate it against the recurrence on random inputs *in float32* — they must agree
   to ~1e-4 in exact arithmetic, which is what proves the transcription before it is
   used as a reference. This is the gate for everything else.
2. **Full 24-layer forward** for the prompt: per-layer hidden state, router decisions,
   final norm, stop-head logits.
3. **Capture points for comparison**, one artifact per stage:
   prompt ids (done) → layer-0 hidden → per-layer hidden (or a checksum per layer) →
   final hidden → stop logits → flow latents → VAE waveform.
4. **C++ thinker** reuses `ming3_audio/backbone.cpp` for MoE and KDA structure, but
   must implement chunk semantics; verify every tensor name and shape against the
   table above before copying a component.

## C++ mapping status

`ming3_audio/backbone.cpp` (61 KB) implements a `bailing_moe_v3_audio` thinker — the
same family. Its expert banks are addressed by the same logical names the writer
emits (`model.layers.N.mlp.experts.{gate,up,down}_proj.weight`). The parts to verify
before reuse: head count and head_dim (audio-tiny may differ), `n_group`/`topk_group`
routing, and whether its attention is KDA or MLA — this checkpoint is KDA-only.
