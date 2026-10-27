#include "engine/models/ming3_tts_tiny/vae.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

constexpr size_t kLatent = 64;
constexpr size_t kHidden = 896;
constexpr size_t kLayers = 24;
constexpr size_t kHeads = 14;
constexpr size_t kKvHeads = 2;
constexpr size_t kHeadDim = 64;
constexpr size_t kMlp = 4864;
constexpr size_t kPatch = 4;                // latent patch -> decoder tokens
constexpr size_t kNBins = 1765;             // n_fft/2 + 1
constexpr size_t kFft = 3528;
constexpr size_t kHop = 882;
constexpr size_t kPad = (kFft - kHop) / 2;
constexpr float kEps = 1e-6f;
constexpr float kTheta = 1e6f;

std::string prefix() { return "audio.decoder."; }

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(float));
    return value;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

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

class VaeWeights {
public:
    explicit VaeWeights(const assets::TensorSource & source) : source_(source) {}

    [[nodiscard]] bool has(const std::string & name) const { return source_.has_tensor(name); }

    const std::vector<float> & tensor(const std::string & name, const Shape & shape) {
        const auto found = cache_.find(name);
        if (found != cache_.end()) return found->second;
        const auto meta = source_.require_metadata(name);
        std::vector<float> values;
        if (meta.dtype == "F32" || meta.dtype == "f32") {
            const auto tensor = source_.require_tensor(name, assets::TensorStorageType::F32, shape);
            const auto * raw = reinterpret_cast<const float *>(tensor.bytes.data());
            values.assign(raw, raw + tensor.bytes.size() / sizeof(float));
        } else {
            const auto tensor = source_.require_tensor(name, assets::TensorStorageType::BF16, shape);
            const auto * raw = reinterpret_cast<const uint16_t *>(tensor.bytes.data());
            values.resize(tensor.bytes.size() / sizeof(uint16_t));
            for (size_t index = 0; index < values.size(); ++index) values[index] = bf16_to_float(raw[index]);
        }
        return cache_.emplace(name, std::move(values)).first->second;
    }

    void linear_op(const float * x, const std::string & suffix, size_t in, size_t out, float * y) {
        const float * bias = has(suffix + ".bias")
            ? tensor(suffix + ".bias", {static_cast<int64_t>(out)}).data()
            : nullptr;
        linear(x, tensor(suffix + ".weight", {static_cast<int64_t>(out), static_cast<int64_t>(in)}).data(),
            bias, in, out, y);
    }

private:
    const assets::TensorSource & source_;
    std::map<std::string, std::vector<float>> cache_;
};

// Qwen2 rotary: halves convention, [cos, cos] layout.
void rope_tables(size_t tokens, std::vector<float> & cos_table, std::vector<float> & sin_table) {
    const size_t half = kHeadDim / 2;
    cos_table.assign(tokens * kHeadDim, 0.0f);
    sin_table.assign(tokens * kHeadDim, 0.0f);
    for (size_t position = 0; position < tokens; ++position) {
        for (size_t index = 0; index < half; ++index) {
            const float freq = 1.0f / std::pow(kTheta, static_cast<float>(2 * index) / static_cast<float>(kHeadDim));
            const float angle = static_cast<float>(position) * freq;
            cos_table[position * kHeadDim + index] = std::cos(angle);
            cos_table[position * kHeadDim + half + index] = std::cos(angle);
            sin_table[position * kHeadDim + index] = std::sin(angle);
            sin_table[position * kHeadDim + half + index] = std::sin(angle);
        }
    }
}

void apply_rope(float * head_vector, const float * cos_table, const float * sin_table, size_t position) {
    const size_t half = kHeadDim / 2;
    // rotate_half must read the *original* values: writing in place would corrupt the
    // second half, whose rotation reads the first half.
    float original[kHeadDim];
    std::memcpy(original, head_vector, sizeof(original));
    for (size_t index = 0; index < kHeadDim; ++index) {
        const float rotated = index < half ? -original[index + half] : original[index - half];
        head_vector[index] = original[index] * cos_table[position * kHeadDim + index] +
                             rotated * sin_table[position * kHeadDim + index];
    }
}

struct Capture {
    std::string name;
    Shape shape;
    std::vector<float> values;
};

void write_captures(const std::filesystem::path & directory, const std::vector<Capture> & captures) {
    std::filesystem::create_directories(directory);
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"stage\": \"D_vae\",\n  \"captures\": [";
    for (size_t index = 0; index < captures.size(); ++index) {
        const auto & capture = captures[index];
        const std::string file = capture.name + ".bin";
        std::ofstream(directory / file, std::ios::binary)
            .write(reinterpret_cast<const char *>(capture.values.data()),
                static_cast<std::streamsize>(capture.values.size() * sizeof(float)));
        manifest << (index ? "," : "") << "\n    {\"name\": \"" << capture.name << "\", \"file\": \""
                 << file << "\", \"shape\": [";
        for (size_t axis = 0; axis < capture.shape.size(); ++axis) {
            manifest << (axis ? "," : "") << capture.shape[axis];
        }
        manifest << "], \"dtype\": \"float32\"}";
    }
    manifest << "\n  ]\n}\n";
}

}  // namespace

void dump_vae_probe(const MingTinyAssets & assets, const std::vector<float> & latent, size_t frames,
    const std::filesystem::path & directory) {
    if (frames == 0 || latent.size() != frames * kLatent) {
        throw std::runtime_error("VAE probe: latent size does not match frame count");
    }
    const size_t tokens = frames * kPatch;
    VaeWeights weights(*assets.weights);
    std::vector<Capture> captures;

    // fc1 per latent frame: 64 -> 896
    std::vector<float> x(frames * kHidden);
    for (size_t frame = 0; frame < frames; ++frame) {
        weights.linear_op(latent.data() + frame * kLatent, prefix() + "fc1", kLatent, kHidden,
            x.data() + frame * kHidden);
    }
    captures.push_back({"vae_fc1", {1, static_cast<int64_t>(frames), static_cast<int64_t>(kHidden)}, x});

    // linear upsample x4 with align_corners=False: output i samples the input at
    // pos = (i + 0.5)/4 - 0.5, clamped, interpolating between neighbouring frames.
    // (A single frame collapses to replication, which is why the one-patch case passed.)
    std::vector<float> up(tokens * kHidden);
    for (size_t index = 0; index < tokens; ++index) {
        const float position = (static_cast<float>(index) + 0.5f) / static_cast<float>(kPatch) - 0.5f;
        const float clamped = std::min(std::max(position, 0.0f), static_cast<float>(frames) - 1.0f);
        const size_t left = static_cast<size_t>(std::floor(clamped));
        const size_t right = std::min(left + 1, frames - 1);
        const float weight = clamped - static_cast<float>(left);
        for (size_t column = 0; column < kHidden; ++column) {
            up[index * kHidden + column] = (1.0f - weight) * x[left * kHidden + column] +
                                           weight * x[right * kHidden + column];
        }
    }
    captures.push_back({"vae_upsampled", {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kHidden)}, up});

    std::vector<float> cos_table, sin_table;
    rope_tables(tokens, cos_table, sin_table);
    std::vector<float> normed(kHidden);
    std::vector<float> q(tokens * kHeads * kHeadDim), k(tokens * kKvHeads * kHeadDim);
    std::vector<float> v(tokens * kKvHeads * kHeadDim), attn(tokens * kHeads * kHeadDim);
    std::vector<float> projected(kHidden), gate(kMlp), upv(kMlp), mlp_out(kHidden);
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const std::string block = prefix() + "decoder.layers." + std::to_string(layer) + ".";
        const auto & layernorm = weights.tensor(block + "input_layernorm.weight", {static_cast<int64_t>(kHidden)});
        const auto & post_norm = weights.tensor(block + "post_attention_layernorm.weight", {static_cast<int64_t>(kHidden)});
        for (size_t token = 0; token < tokens; ++token) {
            rms_norm(up.data() + token * kHidden, layernorm.data(), kHidden, normed.data());
            weights.linear_op(normed.data(), block + "self_attn.q_proj", kHidden, kHeads * kHeadDim,
                q.data() + token * kHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.k_proj", kHidden, kKvHeads * kHeadDim,
                k.data() + token * kKvHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.v_proj", kHidden, kKvHeads * kHeadDim,
                v.data() + token * kKvHeads * kHeadDim);
        }
        const bool token_trace = (std::getenv("MING3_TINY_VAE_TOKEN") != nullptr) && layer == 0;
        const size_t traced_token = token_trace ? static_cast<size_t>(std::atoi(std::getenv("MING3_TINY_VAE_TOKEN"))) : 0;
        if (token_trace && traced_token < tokens) {
            captures.push_back({"trace_q_pre_" + std::to_string(layer),
                {1, static_cast<int64_t>(kHeads), static_cast<int64_t>(kHeadDim)},
                std::vector<float>(q.begin() + traced_token * kHeads * kHeadDim,
                                   q.begin() + (traced_token + 1) * kHeads * kHeadDim)});
            captures.push_back({"trace_k_pre_" + std::to_string(layer),
                {1, static_cast<int64_t>(kKvHeads), static_cast<int64_t>(kHeadDim)},
                std::vector<float>(k.begin() + traced_token * kKvHeads * kHeadDim,
                                   k.begin() + (traced_token + 1) * kKvHeads * kHeadDim)});
            captures.push_back({"trace_v_" + std::to_string(layer),
                {1, static_cast<int64_t>(kKvHeads), static_cast<int64_t>(kHeadDim)},
                std::vector<float>(v.begin() + traced_token * kKvHeads * kHeadDim,
                                   v.begin() + (traced_token + 1) * kKvHeads * kHeadDim)});
        }
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t head = 0; head < kHeads; ++head) {
                apply_rope(q.data() + token * kHeads * kHeadDim + head * kHeadDim,
                    cos_table.data(), sin_table.data(), token);
            }
            for (size_t head = 0; head < kKvHeads; ++head) {
                apply_rope(k.data() + token * kKvHeads * kHeadDim + head * kHeadDim,
                    cos_table.data(), sin_table.data(), token);
            }
        }
        std::vector<float> trace_scores, trace_weights, trace_weighted;
        if (token_trace && traced_token < tokens) {
            captures.push_back({"trace_q_post_" + std::to_string(layer),
                {1, static_cast<int64_t>(kHeads), static_cast<int64_t>(kHeadDim)},
                std::vector<float>(q.begin() + traced_token * kHeads * kHeadDim,
                                   q.begin() + (traced_token + 1) * kHeads * kHeadDim)});
            captures.push_back({"trace_k_post_" + std::to_string(layer),
                {1, static_cast<int64_t>(kKvHeads), static_cast<int64_t>(kHeadDim)},
                std::vector<float>(k.begin() + traced_token * kKvHeads * kHeadDim,
                                   k.begin() + (traced_token + 1) * kKvHeads * kHeadDim)});
            trace_scores.assign(kHeads * (traced_token + 1), 0.0f);
            trace_weights.assign(kHeads * (traced_token + 1), 0.0f);
            trace_weighted.assign(kHeads * kHeadDim, 0.0f);
        }
        // full causal attention across every token (matches the Python SDPA run)
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        std::vector<float> scores(tokens);
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t kv_head = head / (kHeads / kKvHeads);
            for (size_t token = 0; token < tokens; ++token) {
                float best = -1e30f;
                for (size_t key = 0; key <= token; ++key) {
                    float sum = 0.0f;
                    for (size_t dim = 0; dim < kHeadDim; ++dim) {
                        sum += q[token * kHeads * kHeadDim + head * kHeadDim + dim] *
                               k[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    scores[key] = sum * scale;
                    best = std::max(best, scores[key]);
                    if (token_trace && token == traced_token) trace_scores[head * (traced_token + 1) + key] = scores[key];
                }
                float total = 0.0f;
                for (size_t key = 0; key <= token; ++key) {
                    scores[key] = std::exp(scores[key] - best);
                    total += scores[key];
                }
                if (token_trace && token == traced_token) {
                    for (size_t key = 0; key <= token; ++key) {
                        trace_weights[head * (traced_token + 1) + key] = scores[key] / total;
                    }
                }
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    float sum = 0.0f;
                    for (size_t key = 0; key <= token; ++key) {
                        sum += scores[key] / total * v[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    attn[token * kHeads * kHeadDim + head * kHeadDim + dim] = sum;
                    if (token_trace && token == traced_token) trace_weighted[head * kHeadDim + dim] = sum;
                }
            }
        }
        if (token_trace && traced_token < tokens) {
            captures.push_back({"trace_scores_" + std::to_string(layer),
                {1, static_cast<int64_t>(kHeads), static_cast<int64_t>(traced_token + 1)}, trace_scores});
            captures.push_back({"trace_weights_" + std::to_string(layer),
                {1, static_cast<int64_t>(kHeads), static_cast<int64_t>(traced_token + 1)}, trace_weights});
            captures.push_back({"trace_weighted_" + std::to_string(layer),
                {1, static_cast<int64_t>(kHeads), static_cast<int64_t>(kHeadDim)}, trace_weighted});
        }
        for (size_t token = 0; token < tokens; ++token) {
            weights.linear_op(attn.data() + token * kHeads * kHeadDim, block + "self_attn.o_proj",
                kHeads * kHeadDim, kHidden, projected.data());
            for (size_t index = 0; index < kHidden; ++index) up[token * kHidden + index] += projected[index];

            rms_norm(up.data() + token * kHidden, post_norm.data(), kHidden, normed.data());
            weights.linear_op(normed.data(), block + "mlp.gate_proj", kHidden, kMlp, gate.data());
            weights.linear_op(normed.data(), block + "mlp.up_proj", kHidden, kMlp, upv.data());
            for (size_t index = 0; index < kMlp; ++index) gate[index] = silu(gate[index]) * upv[index];
            weights.linear_op(gate.data(), block + "mlp.down_proj", kMlp, kHidden, mlp_out.data());
            for (size_t index = 0; index < kHidden; ++index) up[token * kHidden + index] += mlp_out[index];
        }
        if (const char * trace_env = std::getenv("MING3_TINY_VAE_TRACE");
            trace_env != nullptr && *trace_env != '\0') {
            captures.push_back({"vae_layer_" + std::to_string(layer),
                {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kHidden)}, up});
        }
    }

    std::vector<float> hidden(tokens * kHidden);
    const auto & final_norm = weights.tensor(prefix() + "decoder.norm.weight", {static_cast<int64_t>(kHidden)});
    for (size_t token = 0; token < tokens; ++token) {
        rms_norm(up.data() + token * kHidden, final_norm.data(), kHidden, hidden.data() + token * kHidden);
    }
    captures.push_back({"vae_decoder_hidden", {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kHidden)}, hidden});

    std::vector<float> head(tokens * (kFft + 2));
    for (size_t token = 0; token < tokens; ++token) {
        weights.linear_op(hidden.data() + token * kHidden, prefix() + "head.out", kHidden, kFft + 2,
            head.data() + token * (kFft + 2));
    }
    captures.push_back({"vae_head_out", {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kFft + 2)}, head});

    std::vector<std::complex<double>> spectrum(tokens * kNBins);
    std::vector<float> magnitude(tokens * kNBins), phase_capture(tokens * kNBins);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t bin = 0; bin < kNBins; ++bin) {
            const float mag = std::min(std::exp(head[token * (kFft + 2) + bin]), 100.0f);
            const float phase = head[token * (kFft + 2) + kNBins + bin];
            magnitude[token * kNBins + bin] = mag;
            phase_capture[token * kNBins + bin] = phase;
            spectrum[token * kNBins + bin] = std::polar(static_cast<double>(mag), static_cast<double>(phase));
        }
    }
    captures.push_back({"vae_magnitude", {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kNBins)}, magnitude});
    captures.push_back({"vae_phase", {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kNBins)}, phase_capture});

    const auto & window = weights.tensor(prefix() + "head.istft.window", {static_cast<int64_t>(kFft)});
    const size_t output_size = (tokens - 1) * kHop + kFft;
    std::vector<double> folded(output_size, 0.0), envelope(output_size, 0.0);
    std::vector<double> frame(kFft);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t sample = 0; sample < kFft; ++sample) {
            double sum = spectrum[token * kNBins].real();
            for (size_t bin = 1; bin < kNBins - 1; ++bin) {
                const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(bin) *
                                      static_cast<double>(sample) / static_cast<double>(kFft);
                const auto & value = spectrum[token * kNBins + bin];
                sum += 2.0 * (value.real() * std::cos(angle) - value.imag() * std::sin(angle));
            }
            sum += spectrum[token * kNBins + kNBins - 1].real() * ((sample % 2 == 0) ? 1.0 : -1.0);
            frame[sample] = sum / static_cast<double>(kFft) * window[sample];
            folded[token * kHop + sample] += frame[sample];
            envelope[token * kHop + sample] += static_cast<double>(window[sample]) * window[sample];
        }
    }
    std::vector<float> waveform(tokens * kHop);
    for (size_t index = 0; index < waveform.size(); ++index) {
        const size_t position = index + kPad;
        waveform[index] = static_cast<float>(folded[position] / envelope[position]);
    }
    captures.push_back({"vae_waveform", {1, static_cast<int64_t>(waveform.size())}, waveform});

    write_captures(directory, captures);
    double sum = 0.0;
    for (float value : waveform) sum += static_cast<double>(value) * value;
    std::printf("  vae probe       : %zu frames, %zu tokens, %zu captures, waveform %zu samples, rms=%.6f\n",
        frames, tokens, captures.size(), waveform.size(),
        std::sqrt(sum / static_cast<double>(waveform.size())));
}

}  // namespace engine::models::ming3_tts_tiny
