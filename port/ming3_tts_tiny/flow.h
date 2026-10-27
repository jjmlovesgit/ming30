#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// Flow (CFM/DiT) probe: one latent step from a supplied condition and initial noise.
// Conventions are pinned to the checkpoint's dependencies:
//   * RoPE is the interleaved variant (x_transformers RotaryEmbedding/rotate_half)
//   * timesteps are get_epss_timesteps(steps) with sway_sampling_coef = -1
//   * the Euler solver's SDE term vanishes at temperature = 0
struct FlowRequest {
    std::vector<float> cond;    // [1536] thinker hidden at the last position
    std::vector<float> noise;   // [64] initial latent noise
    float cfg = 1.0f;
    int steps = 10;
};

void dump_flow_probe(const MingTinyAssets & assets, const FlowRequest & request,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
