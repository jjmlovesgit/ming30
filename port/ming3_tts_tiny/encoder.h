#pragma once

#include "engine/models/ming3_tts_tiny/assets.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// VAE encoder: waveform -> reference latents.
//   unfold into 882-sample frames, fc1/fc2, 24-layer Qwen2 encoder, 4-frame aggregator with a
//   CLS row per group, fc3 (896 -> 128), take the posterior mean (first 64).
// Latents are what the podcast prompt places in its <audioPatch> rows.
void encode_reference(const MingTinyAssets & assets, const std::vector<float> & waveform,
    std::vector<float> & latents, size_t & frames,
    std::vector<float> * fc2_out = nullptr, std::vector<float> * encoder_out = nullptr,
    std::vector<float> * aggregated_out = nullptr);

// Load a RIFF/WAVE file as mono float samples. Supports 16-bit PCM and 32-bit float,
// averaging channels. Requires 44100 Hz (no resampling here).
std::vector<float> load_wav_mono(const std::filesystem::path & path);

// Probe: encode a raw f32 waveform file and write latents plus selected intermediates.
void dump_encoder_probe(const MingTinyAssets & assets, const std::vector<float> & waveform,
    const std::filesystem::path & directory);

}  // namespace engine::models::ming3_tts_tiny
