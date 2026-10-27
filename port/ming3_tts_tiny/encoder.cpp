#include "engine/models/ming3_tts_tiny/encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

constexpr size_t kFrame = 882;        // unfold size and hop
constexpr size_t kHidden = 896;
constexpr size_t kHeads = 14;
constexpr size_t kKvHeads = 2;
constexpr size_t kHeadDim = 64;
constexpr size_t kMlp = 4864;
constexpr size_t kEncoderLayers = 24;
constexpr size_t kAggregatorLayers = 4;
constexpr size_t kPatch = 4;
constexpr size_t kLatent = 64;
constexpr float kEps = 1e-6f;
constexpr float kTheta = 1e6f;
constexpr size_t kWindow = 64;   // Qwen2 sliding window: use_sliding_window=true, max_window_layers=0

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

class Weights {
public:
    explicit Weights(const assets::TensorSource & source) : source_(source) {}

    const std::vector<float> & tensor(const std::string & name, const Shape & shape) {
        const auto found = cache_.find(name);
        if (found != cache_.end()) return found->second;
        const auto meta = source_.require_metadata(name);
        std::vector<float> values;
        if (meta.dtype == "F32" || meta.dtype == "f32") {
            const auto value = source_.require_tensor(name, assets::TensorStorageType::F32, shape);
            const auto * raw = reinterpret_cast<const float *>(value.bytes.data());
            values.assign(raw, raw + value.bytes.size() / sizeof(float));
        } else {
            const auto value = source_.require_tensor(name, assets::TensorStorageType::BF16, shape);
            const auto * raw = reinterpret_cast<const uint16_t *>(value.bytes.data());
            values.resize(value.bytes.size() / sizeof(uint16_t));
            for (size_t index = 0; index < values.size(); ++index) values[index] = bf16_to_float(raw[index]);
        }
        return cache_.emplace(name, std::move(values)).first->second;
    }

    void linear_op(const float * x, const std::string & name, size_t in, size_t out, float * y) {
        const float * bias = source_.has_tensor(name + ".bias")
            ? tensor(name + ".bias", {static_cast<int64_t>(out)}).data()
            : nullptr;
        linear(x, tensor(name + ".weight", {static_cast<int64_t>(out), static_cast<int64_t>(in)}).data(), bias,
            in, out, y);
    }

    [[nodiscard]] bool has(const std::string & name) const { return source_.has_tensor(name); }

private:
    const assets::TensorSource & source_;
    std::map<std::string, std::vector<float>> cache_;
};

// One Qwen2-style causal stack (encoder or aggregator), rope positions from 0 per call.
void qwen_forward(Weights & weights, const std::string & prefix, size_t layers, size_t tokens,
    std::vector<float> & hidden, size_t window = 0) {
    // Qwen2 rotates halves: cos/sin are [half, half] and pairs are (i, i + half).
    std::vector<float> cos_table(tokens * kHeadDim), sin_table(tokens * kHeadDim);
    const size_t half = kHeadDim / 2;
    for (size_t position = 0; position < tokens; ++position) {
        for (size_t index = 0; index < half; ++index) {
            const float freq = 1.0f / std::pow(kTheta, 2.0f * static_cast<float>(index) / static_cast<float>(kHeadDim));
            const float angle = static_cast<float>(position) * freq;
            cos_table[position * kHeadDim + index] = std::cos(angle);
            cos_table[position * kHeadDim + half + index] = std::cos(angle);
            sin_table[position * kHeadDim + index] = std::sin(angle);
            sin_table[position * kHeadDim + half + index] = std::sin(angle);
        }
    }

    std::vector<float> normed(kHidden), q(tokens * kHeads * kHeadDim), k(tokens * kKvHeads * kHeadDim);
    std::vector<float> v(tokens * kKvHeads * kHeadDim), attn(tokens * kHeads * kHeadDim);
    std::vector<float> projected(kHidden), gate(kMlp), up(kMlp), down(kHidden);
    for (size_t layer = 0; layer < layers; ++layer) {
        const std::string block = prefix + "layers." + std::to_string(layer) + ".";
        const auto & norm1 = weights.tensor(block + "input_layernorm.weight", {static_cast<int64_t>(kHidden)});
        const auto & norm2 = weights.tensor(block + "post_attention_layernorm.weight", {static_cast<int64_t>(kHidden)});
        for (size_t token = 0; token < tokens; ++token) {
            rms_norm(hidden.data() + token * kHidden, norm1.data(), kHidden, normed.data());
            weights.linear_op(normed.data(), block + "self_attn.q_proj", kHidden, kHeads * kHeadDim,
                q.data() + token * kHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.k_proj", kHidden, kKvHeads * kHeadDim,
                k.data() + token * kKvHeads * kHeadDim);
            weights.linear_op(normed.data(), block + "self_attn.v_proj", kHidden, kKvHeads * kHeadDim,
                v.data() + token * kKvHeads * kHeadDim);
        }
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t head = 0; head < kHeads; ++head) {
                float * base = q.data() + token * kHeads * kHeadDim + head * kHeadDim;
                float original[kHeadDim];
                std::memcpy(original, base, sizeof(original));
                for (size_t index = 0; index < kHeadDim; ++index) {
                    const float c = cos_table[token * kHeadDim + index];
                    const float s = sin_table[token * kHeadDim + index];
                    const float rotated = index < half ? -original[index + half] : original[index - half];
                    base[index] = original[index] * c + rotated * s;
                }
            }
            for (size_t head = 0; head < kKvHeads; ++head) {
                float * base = k.data() + token * kKvHeads * kHeadDim + head * kHeadDim;
                float original[kHeadDim];
                std::memcpy(original, base, sizeof(original));
                for (size_t index = 0; index < kHeadDim; ++index) {
                    const float c = cos_table[token * kHeadDim + index];
                    const float s = sin_table[token * kHeadDim + index];
                    const float rotated = index < half ? -original[index + half] : original[index - half];
                    base[index] = original[index] * c + rotated * s;
                }
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        std::vector<float> scores(tokens);
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t kv_head = head / (kHeads / kKvHeads);
            for (size_t token = 0; token < tokens; ++token) {
                const size_t first_key = (window != 0 && token + 1 > window) ? token + 1 - window : 0;
                float best = -1e30f;
                for (size_t key = first_key; key <= token; ++key) {
                    float sum = 0.0f;
                    for (size_t dim = 0; dim < kHeadDim; ++dim) {
                        sum += q[token * kHeads * kHeadDim + head * kHeadDim + dim] *
                               k[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    scores[key] = sum * scale;
                    best = std::max(best, scores[key]);
                }
                float total = 0.0f;
                for (size_t key = first_key; key <= token; ++key) {
                    scores[key] = std::exp(scores[key] - best);
                    total += scores[key];
                }
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    float sum = 0.0f;
                    for (size_t key = first_key; key <= token; ++key) {
                        sum += scores[key] / total * v[key * kKvHeads * kHeadDim + kv_head * kHeadDim + dim];
                    }
                    attn[token * kHeads * kHeadDim + head * kHeadDim + dim] = sum;
                }
            }
        }
        for (size_t token = 0; token < tokens; ++token) {
            weights.linear_op(attn.data() + token * kHeads * kHeadDim, block + "self_attn.o_proj",
                kHeads * kHeadDim, kHidden, projected.data());
            for (size_t index = 0; index < kHidden; ++index) hidden[token * kHidden + index] += projected[index];

            rms_norm(hidden.data() + token * kHidden, norm2.data(), kHidden, normed.data());
            weights.linear_op(normed.data(), block + "mlp.gate_proj", kHidden, kMlp, gate.data());
            weights.linear_op(normed.data(), block + "mlp.up_proj", kHidden, kMlp, up.data());
            for (size_t index = 0; index < kMlp; ++index) gate[index] = silu(gate[index]) * up[index];
            weights.linear_op(gate.data(), block + "mlp.down_proj", kMlp, kHidden, down.data());
            for (size_t index = 0; index < kHidden; ++index) hidden[token * kHidden + index] += down[index];
        }
    }
    const auto & final_norm = weights.tensor(prefix + "norm.weight", {static_cast<int64_t>(kHidden)});
    for (size_t token = 0; token < tokens; ++token) {
        rms_norm(hidden.data() + token * kHidden, final_norm.data(), kHidden, normed.data());
        std::memcpy(hidden.data() + token * kHidden, normed.data(), kHidden * sizeof(float));
    }
}

}  // namespace

std::vector<float> load_wav_mono(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path.string());
    const auto read_u32 = [&](const char * expected) {
        char tag[4] = {0, 0, 0, 0};
        in.read(tag, 4);
        if (std::memcmp(tag, expected, 4) != 0) {
            throw std::runtime_error("WAV: expected '" + std::string(expected, 4) + "' in " + path.string());
        }
    };
    const auto read_le32 = [&]() {
        unsigned char bytes[4];
        in.read(reinterpret_cast<char *>(bytes), 4);
        return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
               (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    };
    const auto read_le16 = [&]() {
        unsigned char bytes[2];
        in.read(reinterpret_cast<char *>(bytes), 2);
        return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
    };

    read_u32("RIFF");
    (void) read_le32();                     // riff size
    read_u32("WAVE");

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<char> data;
    while (in) {
        char tag[4];
        in.read(tag, 4);
        if (!in) break;
        const uint32_t size = read_le32();
        if (std::memcmp(tag, "fmt ", 4) == 0) {
            format = read_le16();
            channels = read_le16();
            rate = read_le32();
            (void) read_le32();                 // byte rate
            (void) read_le16();                 // block align
            bits = read_le16();
            for (uint32_t skip = 16; skip < size; ++skip) in.get();
        } else if (std::memcmp(tag, "data", 4) == 0) {
            data.resize(size);
            in.read(data.data(), static_cast<std::streamsize>(size));
        } else {
            for (uint32_t skip = 0; skip < size; ++skip) in.get();
        }
    }

    if (rate != 44100) {
        throw std::runtime_error("WAV must be 44100 Hz; got " + std::to_string(rate) + " (" + path.string() + ")");
    }
    if (channels == 0) throw std::runtime_error("WAV has no channels: " + path.string());

    std::vector<float> samples;
    if (format == 1 && bits == 16) {
        const auto * raw = reinterpret_cast<const int16_t *>(data.data());
        const size_t frames = data.size() / (sizeof(int16_t) * channels);
        samples.resize(frames);
        for (size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0f;
            for (size_t channel = 0; channel < channels; ++channel) {
                sum += static_cast<float>(raw[frame * channels + channel]) / 32768.0f;
            }
            samples[frame] = sum / static_cast<float>(channels);
        }
    } else if (format == 3 && bits == 32) {
        const auto * raw = reinterpret_cast<const float *>(data.data());
        const size_t frames = data.size() / (sizeof(float) * channels);
        samples.resize(frames);
        for (size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0f;
            for (size_t channel = 0; channel < channels; ++channel) sum += raw[frame * channels + channel];
            samples[frame] = sum / static_cast<float>(channels);
        }
    } else {
        throw std::runtime_error("WAV: unsupported format " + std::to_string(format) + " / " +
                                 std::to_string(bits) + " bits (" + path.string() + ")");
    }
    return samples;
}

void encode_reference(const MingTinyAssets & assets, const std::vector<float> & waveform,
    std::vector<float> & latents, size_t & frames,
    std::vector<float> * fc2_out, std::vector<float> * encoder_out, std::vector<float> * aggregated_out) {
    Weights weights(*assets.weights);
    const size_t samples = waveform.size();
    frames = (samples + kFrame - 1) / kFrame;
    const size_t padded = (frames - 1) * kFrame + kFrame;

    std::vector<float> x(frames * kHidden);
    std::vector<float> frame(kFrame, 0.0f), fc1(kHidden), fc2(kHidden);
    const auto & fc1_w = weights.tensor("audio.encoder.fc1.weight",
        {static_cast<int64_t>(kHidden), static_cast<int64_t>(kFrame)});
    for (size_t index = 0; index < frames; ++index) {
        std::fill(frame.begin(), frame.end(), 0.0f);
        const size_t start = index * kFrame;
        const size_t count = std::min(kFrame, samples > start ? samples - start : 0);
        std::memcpy(frame.data(), waveform.data() + start, count * sizeof(float));
        (void) padded;
        linear(frame.data(), fc1_w.data(), nullptr, kFrame, kHidden, fc1.data());
        weights.linear_op(fc1.data(), "audio.encoder.fc2", kHidden, kHidden, fc2.data());
        std::memcpy(x.data() + index * kHidden, fc2.data(), kHidden * sizeof(float));
    }

    if (fc2_out != nullptr) *fc2_out = x;
    qwen_forward(weights, "audio.encoder.encoder.", kEncoderLayers, frames, x, kWindow);
    if (encoder_out != nullptr) *encoder_out = x;

    // 4-frame aggregator: pad to a multiple of 4, prepend the CLS row per group, keep the CLS.
    const size_t groups = (frames + kPatch - 1) / kPatch;
    const size_t padded_frames = groups * kPatch;
    std::vector<float> packed(groups * (kPatch + 1) * kHidden, 0.0f);
    const auto & cls = weights.tensor("audio.encoder.cls_embed",
        {1, 1, static_cast<int64_t>(kHidden)});
    // The reference packs each group as [frame0..frame3, CLS], i.e. the CLS row comes LAST.
    for (size_t group = 0; group < groups; ++group) {
        float * destination = packed.data() + group * (kPatch + 1) * kHidden;
        for (size_t slot = 0; slot < kPatch; ++slot) {
            const size_t source = group * kPatch + slot;
            if (source < frames) {
                std::memcpy(destination + slot * kHidden, x.data() + source * kHidden, kHidden * sizeof(float));
            }
        }
        std::memcpy(destination + kPatch * kHidden, cls.data(), kHidden * sizeof(float));
    }
    (void) padded_frames;

    // The reference runs the aggregator ONCE over the packed sequence, so a group's CLS also
    // attends to earlier groups; only the CLS rows are kept.
    qwen_forward(weights, "audio.encoder.aggregator.", kAggregatorLayers, groups * (kPatch + 1), packed,
        kWindow);
    std::vector<float> aggregated(groups * kHidden);
    if (aggregated_out != nullptr) *aggregated_out = aggregated;   // sized, filled below
    for (size_t group = 0; group < groups; ++group) {
        std::memcpy(aggregated.data() + group * kHidden,
            packed.data() + (group * (kPatch + 1) + kPatch) * kHidden, kHidden * sizeof(float));
    }

    if (aggregated_out != nullptr) *aggregated_out = aggregated;
    std::vector<float> params(2 * kLatent);
    latents.assign(groups * kLatent, 0.0f);
    for (size_t group = 0; group < groups; ++group) {
        weights.linear_op(aggregated.data() + group * kHidden, "audio.encoder.fc3", kHidden, 2 * kLatent,
            params.data());
        for (size_t index = 0; index < kLatent; ++index) latents[group * kLatent + index] = params[index];
    }
}

void dump_encoder_probe(const MingTinyAssets & assets, const std::vector<float> & waveform,
    const std::filesystem::path & directory) {
    std::vector<float> latents, fc2_stage, encoder_stage, aggregated_stage;
    size_t frames = 0;
    encode_reference(assets, waveform, latents, frames, &fc2_stage, &encoder_stage, &aggregated_stage);
    const size_t groups = latents.size() / kLatent;
    std::filesystem::create_directories(directory);
    const auto write = [&](const std::string & name, const std::vector<float> & values) {
        std::ofstream(directory / (name + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char *>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
    };
    write("fc2", fc2_stage);
    write("encoder", encoder_stage);
    write("aggregated", aggregated_stage);
    std::ofstream(directory / "latents.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(latents.data()),
            static_cast<std::streamsize>(latents.size() * sizeof(float)));
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"frames\": " << frames << ",\n  \"latents\": " << groups << ",\n  \"shape\": [1, "
             << groups << ", " << kLatent << "]\n}\n";
    double sum = 0.0;
    for (float value : latents) sum += static_cast<double>(value) * value;
    std::printf("  encoder probe   : %zu samples -> %zu frames -> %zu latents, rms %.6f\n", waveform.size(),
        frames, groups, std::sqrt(sum / static_cast<double>(latents.size())));
}

}  // namespace engine::models::ming3_tts_tiny
