#include "engine/models/ming3_tts_tiny/vae.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
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
constexpr size_t kPatch = 4;
constexpr size_t kTokens = kPatch;          // one latent frame -> 4 decoder tokens
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

    [[nodiscard]] bool has(const std::string & name) const { return source_.has_tensor(name); }

    // Qwen2 keeps bias on q/k/v but not on o_proj or the MLP projections.
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

// Qwen2-style rotary: halves convention, [cos, cos] layout.
void rope_tables(float * cos_table, float * sin_table) {
    const size_t half = kHeadDim / 2;
    for (size_t position = 0; position < kTokens; ++position) {
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
    for (size_t index = 0; index < kHeadDim; ++index) {
        const float value = head_vector[index];
        const float rotated = index < half ? -head_vector[index + half] : head_vector[index - half];
        head_vector[index] = value * cos_table[position * kHeadDim + index] +
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

void dump_vae_probe(const MingTinyAssets & assets, const std::vector<float> & latent,
    const std::filesystem::path & directory) {
    VaeWeights weights(*assets.weights);
    std::vector<Capture> captures;

    // fc1: 64 -> 896
    std::vector<float> x(kHidden);
    weights.linear_op(latent.data(), prefix() + "fc1", kLatent, kHidden, x.data());
    captures.push_back({"vae_fc1", {1, 1, static_cast<int64_t>(kHidden)}, x});

    // linear upsample x4 (align_corners=False): 1 frame -> 4 frames
    // A single input frame interpolated to 4 frames is that frame repeated: with one
    // sample there is no second point to interpolate towards.
    std::vector<float> up(kTokens * kHidden);
    for (size_t token = 0; token < kTokens; ++token) {
        std::memcpy(up.data() + token * kHidden, x.data(), kHidden * sizeof(float));
    }
    captures.push_back({"vae_upsampled", {1, static_cast<int64_t>(kTokens), static_cast<int64_t>(kHidden)}, up});

    // decoder layers
    float cos_table[kTokens * kHeadDim], sin_table[kTokens * kHeadDim];
    rope_tables(cos_table, sin_table);
    std::vector<float> normed(kHidden), q(kTokens * kHeads * kHeadDim), k(kTokens * kKvHeads * kHeadDim);
    std::vector<float> v(kTokens * kKvHeads * kHeadDim), attn(kTokens * kHeads * kHeadDim);
    std::vector<float> projected(kHidden), mlp_in(kMlp), mlp_out(kHidden), gate(kMlp), upv(kMlp);
    std::vector<float> residual(kHidden);
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const std::string block = prefix() + "decoder.layers." + std::to_string(layer) + ".";
        for (size_t token = 0; token < kTokens; ++token) {
            const float * input = up.data() + token * kHidden;
            rms_norm(input, weights.tensor(block + "input_layernorm.weight", {static_cast<int64_t>(kHidden)}).data(),
                kHidden, normed.data());
            weights.linear_op(normed.data(), block + "self_attn.q_proj", kHidden, kHidden, q.data() + token * kHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.k_proj", kHidden, kKvHeads * kHeadDim,
                k.data() + token * kKvHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.v_proj", kHidden, kKvHeads * kHeadDim,
                v.data() + token * kKvHeads * kHeadDim);
        }
        for (size_t token = 0; token < kTokens; ++token) {
            for (size_t head = 0; head < kHeads; ++head) {
                apply_rope(q.data() + token * kHeads * kHeadDim + head * kHeadDim, cos_table, sin_table, token);
            }
            for (size_t head = 0; head < kKvHeads; ++head) {
                apply_rope(k.data() + token * kKvHeads * kHeadDim + head * kHeadDim, cos_table, sin_table, token);
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t kv_head = head / (kHeads / kKvHeads);
            for (size_t token = 0; token < kTokens; ++token) {
                std::vector<float> scores(token + 1);
                float best = -1e30f;
                for (size_t key = 0; key <= token; ++key) {  // causal
                    float sum = 0.0f;
                    for (size_t dim = 0; dim < kHeadDim; ++dim) {
                        sum += q[token * kHeads * kHeadDim + head * kHeadDim + dim] *
                               k[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    scores[key] = sum * scale;
                    best = std::max(best, scores[key]);
                }
                float total = 0.0f;
                for (size_t key = 0; key <= token; ++key) {
                    scores[key] = std::exp(scores[key] - best);
                    total += scores[key];
                }
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    float sum = 0.0f;
                    for (size_t key = 0; key <= token; ++key) {
                        sum += scores[key] / total * v[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    attn[token * kHeads * kHeadDim + head * kHeadDim + dim] = sum;
                }
            }
        }
        for (size_t token = 0; token < kTokens; ++token) {
            weights.linear_op(attn.data() + token * kHeads * kHeadDim, block + "self_attn.o_proj",
                kHeads * kHeadDim, kHidden, projected.data());
            for (size_t index = 0; index < kHidden; ++index) up[token * kHidden + index] += projected[index];

            rms_norm(up.data() + token * kHidden,
                weights.tensor(block + "post_attention_layernorm.weight", {static_cast<int64_t>(kHidden)}).data(),
                kHidden, normed.data());
            weights.linear_op(normed.data(), block + "mlp.gate_proj", kHidden, kMlp, gate.data());
            weights.linear_op(normed.data(), block + "mlp.up_proj", kHidden, kMlp, upv.data());
            for (size_t index = 0; index < kMlp; ++index) gate[index] = silu(gate[index]) * upv[index];
            weights.linear_op(gate.data(), block + "mlp.down_proj", kMlp, kHidden, mlp_out.data());
            for (size_t index = 0; index < kHidden; ++index) up[token * kHidden + index] += mlp_out[index];
        }
    }

    // final norm then head
    std::vector<float> hidden(kTokens * kHidden);
    for (size_t token = 0; token < kTokens; ++token) {
        rms_norm(up.data() + token * kHidden,
            weights.tensor(prefix() + "decoder.norm.weight", {static_cast<int64_t>(kHidden)}).data(),
            kHidden, hidden.data() + token * kHidden);
    }
    captures.push_back({"vae_decoder_hidden", {1, static_cast<int64_t>(kTokens), static_cast<int64_t>(kHidden)}, hidden});

    std::vector<float> head(kTokens * (kFft + 2));
    for (size_t token = 0; token < kTokens; ++token) {
        weights.linear_op(hidden.data() + token * kHidden, prefix() + "head.out", kHidden, kFft + 2,
            head.data() + token * (kFft + 2));
    }
    captures.push_back({"vae_head_out", {1, static_cast<int64_t>(kTokens), static_cast<int64_t>(kFft + 2)}, head});

    // spectrum: mag = exp(clip), phase -> complex bins
    std::vector<std::complex<double>> spectrum(kTokens * kNBins);
    std::vector<float> magnitude(kTokens * kNBins), phase_capture(kTokens * kNBins);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t bin = 0; bin < kNBins; ++bin) {
            const float mag_raw = head[token * (kFft + 2) + bin];
            const float phase = head[token * (kFft + 2) + kNBins + bin];
            const float mag = std::min(std::exp(mag_raw), 100.0f);
            magnitude[token * kNBins + bin] = mag;
            phase_capture[token * kNBins + bin] = phase;
            spectrum[token * kNBins + bin] = std::polar(static_cast<double>(mag), static_cast<double>(phase));
        }
    }
    captures.push_back({"vae_magnitude", {1, static_cast<int64_t>(kTokens), static_cast<int64_t>(kNBins)}, magnitude});
    captures.push_back({"vae_phase", {1, static_cast<int64_t>(kTokens), static_cast<int64_t>(kNBins)}, phase_capture});

    // iSTFT: naive irfft (n_fft 3528), window, overlap-add, trim, envelope
    const auto & window = weights.tensor(prefix() + "head.istft.window", {static_cast<int64_t>(kFft)});
    std::vector<double> folded((kTokens - 1) * kHop + kFft, 0.0);
    std::vector<double> envelope((kTokens - 1) * kHop + kFft, 0.0);
    for (size_t token = 0; token < kTokens; ++token) {
        std::vector<double> frame(kFft);
        for (size_t sample = 0; sample < kFft; ++sample) {
            double sum = spectrum[token * kNBins].real();
            for (size_t bin = 1; bin < kNBins - 1; ++bin) {
                const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(bin) * static_cast<double>(sample) / static_cast<double>(kFft);
                const auto & value = spectrum[token * kNBins + bin];
                sum += 2.0 * (value.real() * std::cos(angle) - value.imag() * std::sin(angle));
            }
            sum += spectrum[token * kNBins + kNBins - 1].real() * ((sample % 2 == 0) ? 1.0 : -1.0);
            frame[sample] = sum / static_cast<double>(kFft) * window[sample];
            folded[token * kHop + sample] += frame[sample];
            envelope[token * kHop + sample] += static_cast<double>(window[sample]) * window[sample];
        }
    }
    std::vector<float> waveform(kTokens * kHop);
    for (size_t index = 0; index < waveform.size(); ++index) {
        const size_t position = index + kPad;
        waveform[index] = static_cast<float>(folded[position] / envelope[position]);
    }
    captures.push_back({"vae_waveform", {1, static_cast<int64_t>(kTokens * kHop)}, waveform});

    write_captures(directory, captures);
    double sum = 0.0;
    for (float value : waveform) sum += static_cast<double>(value) * value;
    std::printf("  vae probe       : %zu captures, waveform %zu samples, rms=%.6f\n",
        captures.size(), waveform.size(), std::sqrt(sum / static_cast<double>(waveform.size())));
}

}  // namespace engine::models::ming3_tts_tiny
