#include "engine/models/ming3_tts_tiny/thinker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(float));
    return value;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float silu(float x) { return x * sigmoid(x); }

std::vector<float> load_bf16(const assets::TensorSource & source, const std::string & name, const Shape & shape) {
    const auto tensor = source.require_tensor(name, assets::TensorStorageType::BF16, shape);
    const auto * raw = reinterpret_cast<const uint16_t *>(tensor.bytes.data());
    const size_t count = tensor.bytes.size() / sizeof(uint16_t);
    std::vector<float> values(count);
    for (size_t index = 0; index < count; ++index) {
        values[index] = bf16_to_float(raw[index]);
    }
    return values;
}

// y[o] = sum_i x[i] * weight[o * in + i], i.e. nn.Linear without bias.
void linear(const float * x, const float * weight, size_t in, size_t out, float * y) {
    for (size_t row = 0; row < out; ++row) {
        const float * w = weight + row * in;
        float sum = 0.0f;
        for (size_t column = 0; column < in; ++column) {
            sum += x[column] * w[column];
        }
        y[row] = sum;
    }
}

// Causal depthwise convolution over time with kernel k, then SiLU (fla ShortConvolution).
void short_conv_silu(const std::vector<float> & input, const std::vector<float> & weight, size_t tokens,
    size_t channels, size_t kernel, std::vector<float> & output) {
    output.assign(tokens * channels, 0.0f);
    for (size_t time = 0; time < tokens; ++time) {
        for (size_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < kernel; ++tap) {
                const int64_t source = static_cast<int64_t>(time) - static_cast<int64_t>(kernel - 1 - tap);
                if (source < 0) continue;
                sum += input[static_cast<size_t>(source) * channels + channel] * weight[channel * kernel + tap];
            }
            output[time * channels + channel] = silu(sum);
        }
    }
}

void rms_norm(const std::vector<float> & input, const std::vector<float> & weight, size_t tokens, size_t hidden,
    float eps, std::vector<float> & output) {
    output.assign(input.size(), 0.0f);
    for (size_t time = 0; time < tokens; ++time) {
        double sum = 0.0;
        for (size_t column = 0; column < hidden; ++column) {
            const double value = input[time * hidden + column];
            sum += value * value;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(hidden)) + eps);
        for (size_t column = 0; column < hidden; ++column) {
            output[time * hidden + column] = input[time * hidden + column] * inv * weight[column];
        }
    }
}

struct Capture {
    std::string name;
    std::string file;
    Shape shape;
};

class DumpWriter {
public:
    explicit DumpWriter(std::filesystem::path directory) : directory_(std::move(directory)) {
        std::filesystem::create_directories(directory_);
    }

    void add(const std::string & name, const std::vector<float> & values, const Shape & shape) {
        const std::string file = name + ".bin";
        std::ofstream(directory_ / file, std::ios::binary)
            .write(reinterpret_cast<const char *>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
        captures_.push_back({name, file, shape});
    }

    void manifest(const std::vector<int32_t> & prompt_ids) const {
        std::ofstream out(directory_ / "manifest.json");
        out << "{\n  \"stage\": \"B_thinker\",\n  \"prompt_ids\": [";
        for (size_t index = 0; index < prompt_ids.size(); ++index) {
            out << (index ? "," : "") << prompt_ids[index];
        }
        out << "],\n  \"captures\": [";
        for (size_t index = 0; index < captures_.size(); ++index) {
            const auto & capture = captures_[index];
            out << (index ? "," : "") << "\n    {\"name\": \"" << capture.name << "\", \"file\": \""
                << capture.file << "\", \"shape\": [";
            for (size_t axis = 0; axis < capture.shape.size(); ++axis) {
                out << (axis ? "," : "") << capture.shape[axis];
            }
            out << "], \"dtype\": \"float32\"}";
        }
        out << "\n  ]\n}\n";
    }

private:
    std::filesystem::path directory_;
    std::vector<Capture> captures_;
};

// Layer 0: KDA attention (float32 recurrence), dense MLP, residual additions.
void run_layer_zero(const MingTinyAssets & assets, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output) {
    const auto & source = *assets.weights;
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const size_t heads = 16;             // num_attention_heads
    const size_t head_dim = 128;         // head_dim
    const size_t projection = heads * head_dim;
    const float eps = 1e-6f;
    const float lower_bound = -5.0f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::string prefix = "model.layers.0.";

    const auto norm_weight = load_bf16(source, prefix + "input_layernorm.weight", {static_cast<int64_t>(hidden)});
    const auto post_norm = load_bf16(source, prefix + "post_attention_layernorm.weight", {static_cast<int64_t>(hidden)});
    const auto q_w = load_bf16(source, prefix + "attention.q_proj.weight", {static_cast<int64_t>(projection), static_cast<int64_t>(hidden)});
    const auto k_w = load_bf16(source, prefix + "attention.k_proj.weight", {static_cast<int64_t>(projection), static_cast<int64_t>(hidden)});
    const auto v_w = load_bf16(source, prefix + "attention.v_proj.weight", {static_cast<int64_t>(projection), static_cast<int64_t>(hidden)});
    const auto q_c = load_bf16(source, prefix + "attention.q_conv1d.weight", {static_cast<int64_t>(projection), 1, 4});
    const auto k_c = load_bf16(source, prefix + "attention.k_conv1d.weight", {static_cast<int64_t>(projection), 1, 4});
    const auto v_c = load_bf16(source, prefix + "attention.v_conv1d.weight", {static_cast<int64_t>(projection), 1, 4});
    const auto f_w = load_bf16(source, prefix + "attention.f_proj.weight", {static_cast<int64_t>(projection), static_cast<int64_t>(hidden)});
    const auto g_w = load_bf16(source, prefix + "attention.g_proj.weight", {static_cast<int64_t>(projection), static_cast<int64_t>(hidden)});
    const auto b_w = load_bf16(source, prefix + "attention.b_proj.weight", {static_cast<int64_t>(heads), static_cast<int64_t>(hidden)});
    const auto a_log = load_bf16(source, prefix + "attention.A_log", {static_cast<int64_t>(heads)});
    const auto dt_bias = load_bf16(source, prefix + "attention.dt_bias", {static_cast<int64_t>(projection)});
    const auto o_norm_w = load_bf16(source, prefix + "attention.o_norm.weight", {static_cast<int64_t>(head_dim)});
    const auto o_w = load_bf16(source, prefix + "attention.o_proj.weight", {static_cast<int64_t>(hidden), static_cast<int64_t>(projection)});
    const auto gate_w = load_bf16(source, prefix + "mlp.gate_proj.weight", {4608, static_cast<int64_t>(hidden)});
    const auto up_w = load_bf16(source, prefix + "mlp.up_proj.weight", {4608, static_cast<int64_t>(hidden)});
    const auto down_w = load_bf16(source, prefix + "mlp.down_proj.weight", {static_cast<int64_t>(hidden), 4608});

    std::vector<float> normed, q(projection), k(projection), v(projection), f(projection), g(projection), b(heads);
    std::vector<float> q_conv, k_conv, v_conv;
    std::vector<float> attn_out(hidden), head_out(projection), gated(projection);
    std::vector<float> mlp_hidden(4608), mlp_out(hidden);
    std::vector<float> state(heads * head_dim * head_dim, 0.0f);
    std::vector<float> residual = input;

    rms_norm(input, norm_weight, tokens, hidden, eps, normed);

    std::vector<float> q_all(tokens * projection), k_all(tokens * projection), v_all(tokens * projection);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, q_w.data(), hidden, projection, q.data());
        linear(row, k_w.data(), hidden, projection, k.data());
        linear(row, v_w.data(), hidden, projection, v.data());
        std::memcpy(q_all.data() + time * projection, q.data(), projection * sizeof(float));
        std::memcpy(k_all.data() + time * projection, k.data(), projection * sizeof(float));
        std::memcpy(v_all.data() + time * projection, v.data(), projection * sizeof(float));
    }
    short_conv_silu(q_all, q_c, tokens, projection, 4, q_conv);
    short_conv_silu(k_all, k_c, tokens, projection, 4, k_conv);
    short_conv_silu(v_all, v_c, tokens, projection, 4, v_conv);

    // q/k l2-normalised per head, as use_qk_norm=True does
    for (size_t time = 0; time < tokens; ++time) {
        for (size_t head = 0; head < heads; ++head) {
            for (int which = 0; which < 2; ++which) {
                std::vector<float> & vec = which == 0 ? q_conv : k_conv;
                float * base = vec.data() + time * projection + head * head_dim;
                double sum = 0.0;
                for (size_t dim = 0; dim < head_dim; ++dim) sum += static_cast<double>(base[dim]) * base[dim];
                const float denom = static_cast<float>(std::sqrt(sum)) + 1e-6f;
                for (size_t dim = 0; dim < head_dim; ++dim) base[dim] /= denom;
            }
        }
    }

    std::vector<float> output_all(tokens * projection, 0.0f);
    for (size_t time = 0; time < tokens; ++time) {
        const float * q_t = q_conv.data() + time * projection;
        const float * k_t = k_conv.data() + time * projection;
        const float * v_t = v_conv.data() + time * projection;

        // gate: lower_bound * sigmoid(exp(A_log) * (f_proj(h) + dt_bias))
        linear(normed.data() + time * hidden, f_w.data(), hidden, projection, f.data());
        linear(normed.data() + time * hidden, b_w.data(), hidden, heads, b.data());
        for (size_t head = 0; head < heads; ++head) {
            const float a = std::exp(a_log[head]);
            const float beta = sigmoid(b[head]);
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const size_t index = head * head_dim + dim;
                g[index] = lower_bound * sigmoid(a * (f[index] + dt_bias[index]));
            }
            float * s = state.data() + head * head_dim * head_dim;
            // decay
            for (size_t row = 0; row < head_dim; ++row) {
                const float decay = std::exp(g[head * head_dim + row]);
                float * srow = s + row * head_dim;
                for (size_t column = 0; column < head_dim; ++column) srow[column] *= decay;
            }
            // prediction k^T S per value dim
            std::vector<float> predicted(head_dim, 0.0f);
            for (size_t column = 0; column < head_dim; ++column) {
                float sum = 0.0f;
                for (size_t row = 0; row < head_dim; ++row) {
                    sum += k_t[head * head_dim + row] * s[row * head_dim + column];
                }
                predicted[column] = sum;
            }
            // state update: S += beta * k (v - predicted)^T
            for (size_t row = 0; row < head_dim; ++row) {
                const float outer = beta * k_t[head * head_dim + row];
                float * srow = s + row * head_dim;
                for (size_t column = 0; column < head_dim; ++column) {
                    srow[column] += outer * (v_t[head * head_dim + column] - predicted[column]);
                }
            }
            // output: q^T S, with the 1/sqrt(K) scale applied to q
            for (size_t column = 0; column < head_dim; ++column) {
                float sum = 0.0f;
                for (size_t row = 0; row < head_dim; ++row) {
                    sum += scale * q_t[head * head_dim + row] * s[row * head_dim + column];
                }
                output_all[time * projection + head * head_dim + column] = sum;
            }
        }
    }

    // output gate: rms_norm per head over v-dim, times sigmoid(g_proj(h))
    for (size_t time = 0; time < tokens; ++time) {
        linear(normed.data() + time * hidden, g_w.data(), hidden, projection, g.data());
        for (size_t head = 0; head < heads; ++head) {
            const float * base = output_all.data() + time * projection + head * head_dim;
            double sum = 0.0;
            for (size_t dim = 0; dim < head_dim; ++dim) sum += static_cast<double>(base[dim]) * base[dim];
            const float inv = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(head_dim)) + eps);
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const size_t index = head * head_dim + dim;
                gated[index] = base[dim] * inv * o_norm_w[dim] * sigmoid(g[index]);
            }
        }
        linear(gated.data(), o_w.data(), projection, hidden, attn_out.data());
        for (size_t column = 0; column < hidden; ++column) {
            residual[time * hidden + column] += attn_out[column];
        }
    }

    // dense MLP with residual
    rms_norm(residual, post_norm, tokens, hidden, eps, normed);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        std::vector<float> up(4608);
        linear(row, gate_w.data(), hidden, 4608, mlp_hidden.data());
        linear(row, up_w.data(), hidden, 4608, up.data());
        for (size_t index = 0; index < 4608; ++index) mlp_hidden[index] = silu(mlp_hidden[index]) * up[index];
        linear(mlp_hidden.data(), down_w.data(), 4608, hidden, mlp_out.data());
        for (size_t column = 0; column < hidden; ++column) {
            residual[time * hidden + column] += mlp_out[column];
        }
    }

    output = std::move(residual);
}

}  // namespace

void dump_thinker_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::filesystem::path & directory) {
    const size_t tokens = prompt_ids.size();
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    DumpWriter writer(directory);

    const auto table = load_bf16(*assets.weights, "model.word_embeddings.weight",
        {assets.config.vocabulary_size, assets.config.hidden_size});
    std::vector<float> embedding(tokens * hidden);
    for (size_t row = 0; row < tokens; ++row) {
        std::memcpy(embedding.data() + row * hidden, table.data() + static_cast<size_t>(prompt_ids[row]) * hidden,
            hidden * sizeof(float));
    }
    writer.add("embedding", embedding, {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});

    std::vector<float> layer_zero;
    run_layer_zero(assets, embedding, tokens, layer_zero);
    writer.add("layer_0", layer_zero, {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
    writer.manifest(prompt_ids);
}

}  // namespace engine::models::ming3_tts_tiny
