#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::ming3_tts_tiny {

// Structure only for milestone 1: enough of the checkpoint's shape to describe what
// the loader found. Generation parameters are added when inference is wired up.
struct MingTinyConfig {
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t experts_per_token = 0;
    int64_t hidden_size = 0;
    int64_t vocabulary_size = 0;
    int64_t first_k_dense_replace = 1;
    int64_t flow_depth = 0;
    int64_t flow_hidden_size = 0;
    int64_t sample_rate = 0;
};

struct MingTinyAssets {
    assets::ResourceBundle resources;
    MingTinyConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
};

// Opens a Ming-3.0-tts-tiny package (GGUF), resolves the embedded spec and sidecars,
// and reports the tensor contract. Structural only: no inference is performed.
std::shared_ptr<const MingTinyAssets> load_ming3_tts_tiny_assets(const std::filesystem::path & path);

}  // namespace engine::models::ming3_tts_tiny
