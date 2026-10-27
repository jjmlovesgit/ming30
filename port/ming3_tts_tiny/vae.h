#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// VAE decoder probe: one latent patch -> 3528 samples (80 ms at 44.1 kHz).
// Captures after fc1, after upsampling, after the Qwen2 decoder, the iSTFT spectrum,
// and the final waveform, so a parity mismatch is localised to the transformer or to
// waveform reconstruction.
void dump_vae_probe(const MingTinyAssets & assets, const std::vector<float> & latent,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
