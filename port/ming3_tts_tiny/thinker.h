#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// Stage B probe: runs as much of the thinker as is implemented and writes capture
// points (raw little-endian float32 + a manifest) for comparison against the
// plain-torch reference. Layer 0 only for now: KDA attention, dense MLP, residuals.
// Forward pass over given embeddings (prompt plus projected feedback latents).
void thinker_forward_embeds(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    std::vector<float> & final_norm);

void dump_thinker_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
