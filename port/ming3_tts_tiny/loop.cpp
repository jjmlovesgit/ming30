#include "engine/models/ming3_tts_tiny/loop.h"

#include "engine/models/ming3_tts_tiny/flow.h"
#include "engine/models/ming3_tts_tiny/prompt.h"
#include "engine/models/ming3_tts_tiny/thinker.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <map>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

constexpr size_t kHiddenLlm = 1536;
constexpr size_t kLatent = 64;
constexpr size_t kHidden = 1024;   // aggregator width
constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 64;
constexpr size_t kDepth = 8;
constexpr size_t kMlp = 4096;
constexpr float kEps = 1e-6f;

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(float));
    return value;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

void linear(const float * x, const float * weight, const float * bias, size_t in, size_t out, float * y) {
    for (size_t row = 0; row < out; ++row) {
        const float * w = weight + row * in;
        float sum = bias != nullptr ? bias[row] : 0.0f;
        for (size_t column = 0; column < in; ++column) sum += x[column] * w[column];
        y[row] = sum;
    }
}

void rms_norm(const float * x, const float * weight, size_t dim, float * y) {
    double sum = 0.0;
    for (size_t index = 0; index < dim; ++index) sum += static_cast<double>(x[index]) * x[index];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(dim)) + kEps);
    for (size_t index = 0; index < dim; ++index) y[index] = x[index] * inv * weight[index];
}

// linear_proj_audio: CLS + latent through 8 interleaved-RoPE blocks, CLS out to 1536.
class Aggregator {
public:
    Aggregator(const assets::TensorSource & source, const std::string & prefix) : source_(source), prefix_(prefix) {}

    void operator()(const float * latent, float * embedding) {
        const size_t tokens = 2;
        std::vector<float> x(tokens * kHidden);
        // CLS token from word_embedder, then the projected latent
        const auto & cls = tensor("word_embedder.weight", {1, static_cast<int64_t>(kHidden)});
        std::memcpy(x.data(), cls.data(), kHidden * sizeof(float));
        linear(latent, tensor("x_embedder.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kLatent)}).data(),
            tensor("x_embedder.bias", {static_cast<int64_t>(kHidden)}).data(), kLatent, kHidden,
            x.data() + kHidden);

        float cos_table[tokens * kHeadDim], sin_table[tokens * kHeadDim];
        const auto & inv_freq = tensor("rotary_embed.inv_freq", {static_cast<int64_t>(kHeadDim / 2)});
        for (size_t position = 0; position < tokens; ++position) {
            for (size_t pair = 0; pair < kHeadDim / 2; ++pair) {
                const float angle = static_cast<float>(position) * inv_freq[pair];
                cos_table[position * kHeadDim + 2 * pair] = std::cos(angle);
                cos_table[position * kHeadDim + 2 * pair + 1] = std::cos(angle);
                sin_table[position * kHeadDim + 2 * pair] = std::sin(angle);
                sin_table[position * kHeadDim + 2 * pair + 1] = std::sin(angle);
            }
        }

        std::vector<float> normed(kHidden), q(tokens * kHidden), k(tokens * kHidden), v(tokens * kHidden);
        std::vector<float> attn(tokens * kHidden), projected(kHidden), mlp_in(kMlp), mlp_out(kHidden);
        for (size_t layer = 0; layer < kDepth; ++layer) {
            const std::string block = "blocks." + std::to_string(layer) + ".";
            const auto & norm1 = tensor(block + "norm1.weight", {static_cast<int64_t>(kHidden)});
            const auto & norm2 = tensor(block + "norm2.weight", {static_cast<int64_t>(kHidden)});
            for (size_t token = 0; token < tokens; ++token) {
                rms_norm(x.data() + token * kHidden, norm1.data(), kHidden, normed.data());
                linear(normed.data(), tensor(block + "attn.to_q.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kHidden)}).data(),
                    tensor(block + "attn.to_q.bias", {static_cast<int64_t>(kHidden)}).data(), kHidden, kHidden, q.data() + token * kHidden);
                linear(normed.data(), tensor(block + "attn.to_k.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kHidden)}).data(),
                    tensor(block + "attn.to_k.bias", {static_cast<int64_t>(kHidden)}).data(), kHidden, kHidden, k.data() + token * kHidden);
                linear(normed.data(), tensor(block + "attn.to_v.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kHidden)}).data(),
                    tensor(block + "attn.to_v.bias", {static_cast<int64_t>(kHidden)}).data(), kHidden, kHidden, v.data() + token * kHidden);
            }
            for (size_t token = 0; token < tokens; ++token) {
                for (size_t head = 0; head < kHeads; ++head) {
                    for (int which = 0; which < 2; ++which) {
                        std::vector<float> & vec = which == 0 ? q : k;
                        float * base = vec.data() + token * kHidden + head * kHeadDim;
                        float original[kHeadDim];
                        std::memcpy(original, base, sizeof(original));
                        for (size_t j = 0; j < kHeadDim / 2; ++j) {
                            const float c = cos_table[token * kHeadDim + 2 * j];
                            const float s = sin_table[token * kHeadDim + 2 * j];
                            const float x1 = original[2 * j], x2 = original[2 * j + 1];
                            base[2 * j] = x1 * c - x2 * s;
                            base[2 * j + 1] = x1 * s + x2 * c;
                        }
                    }
                }
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
            for (size_t head = 0; head < kHeads; ++head) {
                for (size_t token = 0; token < tokens; ++token) {
                    std::vector<float> scores(tokens);
                    float best = -1e30f;
                    for (size_t key = 0; key < tokens; ++key) {   // bidirectional (mask is None)
                        float sum = 0.0f;
                        for (size_t dim = 0; dim < kHeadDim; ++dim) {
                            sum += q[token * kHidden + head * kHeadDim + dim] *
                                   k[key * kHidden + head * kHeadDim + dim];
                        }
                        scores[key] = sum * scale;
                        best = std::max(best, scores[key]);
                    }
                    float total = 0.0f;
                    for (size_t key = 0; key < tokens; ++key) {
                        scores[key] = std::exp(scores[key] - best);
                        total += scores[key];
                    }
                    for (size_t dim = 0; dim < kHeadDim; ++dim) {
                        float sum = 0.0f;
                        for (size_t key = 0; key < tokens; ++key) {
                            sum += scores[key] / total * v[key * kHidden + head * kHeadDim + dim];
                        }
                        attn[token * kHidden + head * kHeadDim + dim] = sum;
                    }
                }
            }
            for (size_t token = 0; token < tokens; ++token) {
                linear(attn.data() + token * kHidden, tensor(block + "attn.to_out.0.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kHidden)}).data(),
                    tensor(block + "attn.to_out.0.bias", {static_cast<int64_t>(kHidden)}).data(), kHidden, kHidden, projected.data());
                for (size_t index = 0; index < kHidden; ++index) x[token * kHidden + index] += projected[index];

                rms_norm(x.data() + token * kHidden, norm2.data(), kHidden, normed.data());
                linear(normed.data(), tensor(block + "mlp.ff.0.0.weight", {static_cast<int64_t>(kMlp), static_cast<int64_t>(kHidden)}).data(),
                    tensor(block + "mlp.ff.0.0.bias", {static_cast<int64_t>(kMlp)}).data(), kHidden, kMlp, mlp_in.data());
                for (auto & value : mlp_in) value = gelu_tanh(value);
                linear(mlp_in.data(), tensor(block + "mlp.ff.2.weight", {static_cast<int64_t>(kHidden), static_cast<int64_t>(kMlp)}).data(),
                    tensor(block + "mlp.ff.2.bias", {static_cast<int64_t>(kHidden)}).data(), kMlp, kHidden, mlp_out.data());
                for (size_t index = 0; index < kHidden; ++index) x[token * kHidden + index] += mlp_out[index];
            }
        }
        std::vector<float> normed_final(kHidden);
        rms_norm(x.data(), tensor("final_layer.norm_final.weight", {static_cast<int64_t>(kHidden)}).data(), kHidden,
            normed_final.data());
        linear(normed_final.data(), tensor("final_layer.linear.weight", {static_cast<int64_t>(kHiddenLlm), static_cast<int64_t>(kHidden)}).data(),
            tensor("final_layer.linear.bias", {static_cast<int64_t>(kHiddenLlm)}).data(), kHidden, kHiddenLlm, embedding);
    }

private:
    const std::vector<float> & tensor(const std::string & name, const Shape & shape) {
        const std::string full = prefix_ + name;
        const auto found = cache_.find(full);
        if (found != cache_.end()) return found->second;
        const auto meta = source_.require_metadata(full);
        std::vector<float> values;
        if (meta.dtype == "F32" || meta.dtype == "f32") {
            const auto value = source_.require_tensor(full, assets::TensorStorageType::F32, shape);
            const auto * raw = reinterpret_cast<const float *>(value.bytes.data());
            values.assign(raw, raw + value.bytes.size() / sizeof(float));
        } else {
            const auto value = source_.require_tensor(full, assets::TensorStorageType::BF16, shape);
            const auto * raw = reinterpret_cast<const uint16_t *>(value.bytes.data());
            values.resize(value.bytes.size() / sizeof(uint16_t));
            for (size_t index = 0; index < values.size(); ++index) values[index] = bf16_to_float(raw[index]);
        }
        return cache_.emplace(full, std::move(values)).first->second;
    }

    const assets::TensorSource & source_;
    std::string prefix_;
    std::map<std::string, std::vector<float>> cache_;
};

}  // namespace

void run_cache_check(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    size_t prefill_tokens) {
    if (prefill_tokens >= tokens) throw std::runtime_error("cache check: prefill must be shorter than the sequence");
    const auto clock = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [](auto from, auto to) {
        return std::chrono::duration<double>(to - from).count();
    };

    std::vector<float> full;
    auto start_full = clock();
    thinker_forward_embeds(assets, embeds, tokens, full);
    const double full_seconds = elapsed(start_full, clock());

    ThinkerCache cache;
    std::vector<float> prefill_final;
    std::vector<float> head(embeds.begin(), embeds.begin() + static_cast<long>(prefill_tokens) * kHiddenLlm);
    auto start_cached = clock();
    thinker_prefill_cached(assets, head, prefill_tokens, cache, prefill_final);
    const double prefill_seconds = elapsed(start_cached, clock());

    std::vector<float> stepped;
    auto start_steps = clock();
    for (size_t index = prefill_tokens; index < tokens; ++index) {
        std::vector<float> row(embeds.begin() + static_cast<long>(index) * kHiddenLlm,
                               embeds.begin() + static_cast<long>(index + 1) * kHiddenLlm);
        thinker_step_cached(assets, row, cache, stepped);
    }
    const double step_seconds = elapsed(start_steps, clock());
    std::printf("  cache timings   : full %.1f s | prefill %.1f s + %zu steps %.1f s (%.2f s/step)\n",
        full_seconds, prefill_seconds, tokens - prefill_tokens, step_seconds,
        step_seconds / static_cast<double>(tokens - prefill_tokens));
    double max_abs = 0.0;
    double sum = 0.0, reference = 0.0;
    for (size_t index = 0; index < kHiddenLlm; ++index) {
        const double difference = static_cast<double>(stepped[index]) - static_cast<double>(full[full.size() - kHiddenLlm + index]);
        max_abs = std::max(max_abs, std::abs(difference));
        sum += difference * difference;
        const double value = full[full.size() - kHiddenLlm + index];
        reference += value * value;
    }
    std::printf("  cache check     : tokens=%zu (prefill %zu + %zu steps) last-token max_abs=%.3e rel_rmse=%.3e\n",
        tokens, prefill_tokens, tokens - prefill_tokens, max_abs,
        std::sqrt(sum / (reference > 0.0 ? reference : 1.0)));
}

void run_loop_patch_seeded(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    const std::vector<float> & noise, const std::vector<float> & history, float cfg, int flow_steps,
    std::vector<float> & latent_out, std::vector<float> & feedback_out) {
    Aggregator aggregator(*assets.weights, "linear_proj_audio.");
    std::vector<float> final_norm;
    thinker_forward_embeds(assets, embeds, tokens, final_norm);
    std::vector<float> cond(final_norm.end() - kHiddenLlm, final_norm.end());
    latent_out = flow_sample_from_noise(assets, cond, noise, history, cfg, flow_steps);
    feedback_out.assign(kHiddenLlm, 0.0f);
    aggregator(latent_out.data(), feedback_out.data());
}

void dump_podcast_prompt_probe(const MingTinyAssets & assets, const std::string & script,
    const std::string & reference_text, const std::vector<float> & reference_latents,
    size_t latent_tokens, const std::filesystem::path & directory) {
    if (reference_latents.size() != latent_tokens * kLatent) {
        throw std::runtime_error("podcast probe: reference latents do not match the placeholder count");
    }

    MingTinyPromptBuilder prompts{tokenizers::LlamaBpeTokenizerSpec{{}, {},
        assets.resources.require_file("tokenizer_config"), assets.resources.require_file("tokenizer_json"),
        tokenizers::LlamaBpePreTokenizer::BailingMoe}, MingTinySpecialTokens{}};
    const auto segments = prompts.build_podcast_segments(script, reference_text, latent_tokens);
    std::vector<int32_t> ids;
    size_t prompt_start = 0;
    size_t speaker_position = 0;
    for (const auto & [name, segment] : segments) {
        if (name == "prompt_latents") prompt_start = ids.size();
        if (name == "speaker_block") speaker_position = ids.size() + 1;
        ids.insert(ids.end(), segment.begin(), segment.end());
    }

    const auto table_tensor = assets.weights->require_tensor("model.word_embeddings.weight",
        assets::TensorStorageType::BF16,
        Shape{assets.config.vocabulary_size, assets.config.hidden_size});
    std::vector<float> table(table_tensor.bytes.size() / sizeof(uint16_t));
    {
        const auto * raw = reinterpret_cast<const uint16_t *>(table_tensor.bytes.data());
        for (size_t index = 0; index < table.size(); ++index) table[index] = bf16_to_float(raw[index]);
    }

    const size_t tokens = ids.size();
    std::vector<float> embeds(tokens * kHiddenLlm);
    for (size_t row = 0; row < tokens; ++row) {
        std::memcpy(embeds.data() + row * kHiddenLlm,
            table.data() + static_cast<size_t>(ids[row]) * kHiddenLlm, kHiddenLlm * sizeof(float));
    }

    Aggregator aggregator(*assets.weights, "linear_proj_audio.");
    for (size_t patch = 0; patch < latent_tokens; ++patch) {
        aggregator(reference_latents.data() + patch * kLatent, embeds.data() + (prompt_start + patch) * kHiddenLlm);
    }

    // spk_head(zeros) == its bias, which is the documented missing-reference case
    const auto bias_tensor = assets.weights->require_tensor("spk_head.bias",
        assets::TensorStorageType::BF16, Shape{static_cast<int64_t>(kHiddenLlm)});
    const auto * bias_raw = reinterpret_cast<const uint16_t *>(bias_tensor.bytes.data());
    for (size_t index = 0; index < kHiddenLlm; ++index) {
        embeds[speaker_position * kHiddenLlm + index] = bf16_to_float(bias_raw[index]);
    }

    std::filesystem::create_directories(directory);
    std::ofstream(directory / "embeds.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(embeds.data()),
            static_cast<std::streamsize>(embeds.size() * sizeof(float)));
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"tokens\": " << tokens << ",\n  \"prompt_start\": " << prompt_start
             << ",\n  \"speaker_position\": " << speaker_position << ",\n  \"latent_tokens\": "
             << latent_tokens << ",\n  \"ids\": [";
    for (size_t index = 0; index < ids.size(); ++index) manifest << (index ? "," : "") << ids[index];
    manifest << "]\n}\n";
    std::printf("  podcast assemble: %zu ids (placeholders %zu..%zu), speaker row %zu\n", tokens, prompt_start,
        tokens - 1, speaker_position);
}

void dump_loop_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::vector<float> & noises, size_t patches, float cfg, int flow_steps,
    const std::filesystem::path & directory) {
    if (noises.size() != patches * kLatent) throw std::runtime_error("loop probe: noise size mismatch");

    Aggregator aggregator(*assets.weights, "linear_proj_audio.");
    const auto table_tensor = assets.weights->require_tensor("model.word_embeddings.weight",
        assets::TensorStorageType::BF16,
        Shape{assets.config.vocabulary_size, assets.config.hidden_size});
    std::vector<float> table_values(table_tensor.bytes.size() / sizeof(uint16_t));
    {
        const auto * raw = reinterpret_cast<const uint16_t *>(table_tensor.bytes.data());
        for (size_t index = 0; index < table_values.size(); ++index) table_values[index] = bf16_to_float(raw[index]);
    }
    const std::vector<float> & table = table_values;

    std::vector<float> embeds(prompt_ids.size() * kHiddenLlm);
    for (size_t row = 0; row < prompt_ids.size(); ++row) {
        std::memcpy(embeds.data() + row * kHiddenLlm,
            table.data() + static_cast<size_t>(prompt_ids[row]) * kHiddenLlm, kHiddenLlm * sizeof(float));
    }

    std::vector<float> history(kLatent, 0.0f);
    std::vector<float> final_norm, latent(kLatent), feedback(kHiddenLlm);
    std::vector<float> latents_all;
    std::filesystem::create_directories(directory);

    // Prefill the prompt once, then advance one projected latent per patch. This is the
    // decode path: each patch costs a single-token step instead of a full recompute.
    ThinkerCache cache;
    thinker_prefill_cached(assets, embeds, prompt_ids.size(), cache, final_norm);

    for (size_t patch = 0; patch < patches; ++patch) {
        const size_t tokens = prompt_ids.size() + patch;
        if (patch > 0) {
            thinker_step_cached(assets, feedback, cache, final_norm);
        }
        std::vector<float> cond(final_norm.end() - kHiddenLlm, final_norm.end());
        std::vector<float> noise(noises.begin() + patch * kLatent, noises.begin() + (patch + 1) * kLatent);
        latent = flow_sample_from_noise(assets, cond, noise, history, cfg, flow_steps);
        aggregator(latent.data(), feedback.data());

        latents_all.insert(latents_all.end(), latent.begin(), latent.end());
        history = latent;

        std::ofstream(directory / ("latent_" + std::to_string(patch) + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char *>(latent.data()), kLatent * sizeof(float));
        std::ofstream(directory / ("feedback_" + std::to_string(patch) + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char *>(feedback.data()), kHiddenLlm * sizeof(float));
        std::printf("  patch %2zu: tokens=%3zu latent_rms=%.4f feedback_norm=%.3f\n", patch, tokens,
            std::sqrt(std::inner_product(latent.begin(), latent.end(), latent.begin(), 0.0) / kLatent),
            std::sqrt(std::inner_product(feedback.begin(), feedback.end(), feedback.begin(), 0.0)));
    }

    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"stage\": \"E_loop\",\n  \"patches\": " << patches << ",\n  \"captures\": [";
    manifest << "\n    {\"name\": \"loop_latents\", \"file\": \"latents.bin\", \"shape\": [1, "
             << patches << ", " << kLatent << "], \"dtype\": \"float32\"}\n  ]\n}\n";
    std::ofstream(directory / "latents.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(latents_all.data()),
            static_cast<std::streamsize>(latents_all.size() * sizeof(float)));
    std::printf("  loop probe      : %zu patches, latents written to %s\n", patches, directory.string().c_str());
}

}  // namespace engine::models::ming3_tts_tiny
