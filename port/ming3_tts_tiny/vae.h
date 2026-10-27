#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// VAE decoder probe: `frames` latent patches -> frames * 4 * 882 samples.
// Attention is full causal across all decoder tokens, matching the Python SDPA run;
// the checkpoint's 64-token sliding window is a separate compatibility question.
void dump_vae_probe(const MingTinyAssets & assets, const std::vector<float> & latent, size_t frames,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
