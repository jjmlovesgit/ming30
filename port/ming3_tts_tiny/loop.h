#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

void thinker_forward_embeds(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    std::vector<float> & final_norm);

// One flow step from an explicit noise vector, so a Python run's noise stream can be
// replayed exactly.
std::vector<float> flow_sample_from_noise(const MingTinyAssets & assets, const std::vector<float> & cond,
    const std::vector<float> & noise, const std::vector<float> & history, float cfg, int steps);

// Compare full recompute against prefill+step decoding on the same embeddings.
void run_cache_check(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    size_t prefill_tokens);

// One patch from a supplied embedding history (prompt + projected feedbacks) and noise.
void run_loop_patch_seeded(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    const std::vector<float> & noise, const std::vector<float> & history, float cfg, int flow_steps,
    std::vector<float> & latent_out, std::vector<float> & feedback_out);

void dump_loop_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::vector<float> & noises, size_t patches, float cfg, int flow_steps,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
