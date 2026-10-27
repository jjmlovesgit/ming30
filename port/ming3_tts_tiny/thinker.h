#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// Stage B probe: runs as much of the thinker as is implemented and writes capture
// points (raw little-endian float32 + a manifest) for comparison against the
// plain-torch reference. Layer 0 only for now: KDA attention, dense MLP, residuals.
// cuBLAS microbenchmark of one KDA layer's projections at the one-token decode shape.
void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations);
// GPU test of routed experts for one MoE layer, with fixed expert ids from the
// reference run: per-expert outputs and the combined layer output.
void run_gpu_moe_benchmark(const MingTinyAssets & assets, int iterations);

// Per-layer decode state.
//   KDA keeps the recurrent matrix S plus the last three pre-convolution rows for
//   q/k/v; MLA keeps the keys and values of every position seen so far.
struct KdaCache {
    std::vector<float> state;    // heads * head_dim * head_dim
    std::vector<float> q_hist;   // 3 rows of projection_size
    std::vector<float> k_hist;
    std::vector<float> v_hist;
    bool initialized = false;
};

struct MlaCache {
    std::vector<float> keys;     // tokens * heads * (nope + rope)
    std::vector<float> values;   // tokens * heads * v_head
    size_t tokens = 0;
};

struct ThinkerCache {
    std::vector<KdaCache> kda;   // indexed by layer
    std::vector<MlaCache> mla;   // indexed by layer
};

// Cached decode: prefill the prompt once, then advance one token at a time.
void thinker_prefill_cached(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    ThinkerCache & cache, std::vector<float> & final_norm);
void thinker_step_cached(const MingTinyAssets & assets, const std::vector<float> & embed,
    ThinkerCache & cache, std::vector<float> & final_norm);

// Forward pass over given embeddings (prompt plus projected feedback latents).
void thinker_forward_embeds(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    std::vector<float> & final_norm);

void dump_thinker_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
