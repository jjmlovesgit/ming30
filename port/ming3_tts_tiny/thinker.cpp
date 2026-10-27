#include "engine/models/ming3_tts_tiny/thinker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 128;
constexpr size_t kProjection = kHeads * kHeadDim;
constexpr float kEps = 1e-6f;
constexpr float kLowerBound = -5.0f;
constexpr size_t kTopK = 8;
constexpr size_t kGroups = 8;
constexpr size_t kTopGroups = 4;
constexpr float kRoutedScaling = 2.5f;

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(float));
    return value;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

struct RawTensor {
    std::vector<std::byte> bytes;
    Shape shape;
};

std::vector<float> load_float(const assets::TensorSource & source, const std::string & name, const Shape & shape) {
    const auto tensor = source.require_tensor(name, assets::TensorStorageType::BF16, shape);
    const auto * raw = reinterpret_cast<const uint16_t *>(tensor.bytes.data());
    std::vector<float> values(tensor.bytes.size() / sizeof(uint16_t));
    for (size_t index = 0; index < values.size(); ++index) values[index] = bf16_to_float(raw[index]);
    return values;
}

RawTensor load_raw(const assets::TensorSource & source, const std::string & name, const Shape & shape) {
    auto tensor = source.require_tensor(name, assets::TensorStorageType::BF16, shape);
    return {std::move(tensor.bytes), std::move(shape)};
}

// y[o] = sum_i x[i] * weight[o * in + i], i.e. nn.Linear without bias.
void linear(const float * x, const float * weight, size_t in, size_t out, float * y) {
    for (size_t row = 0; row < out; ++row) {
        const float * w = weight + row * in;
        float sum = 0.0f;
        for (size_t column = 0; column < in; ++column) sum += x[column] * w[column];
        y[row] = sum;
    }
}

// One expert out of a stacked bank; bank layout is [experts, out, in], bf16 on disk.
void bank_linear(const RawTensor & bank, size_t expert, size_t out, size_t in, const float * x, float * y) {
    const auto * raw = reinterpret_cast<const uint16_t *>(bank.bytes.data()) + expert * out * in;
    for (size_t row = 0; row < out; ++row) {
        float sum = 0.0f;
        for (size_t column = 0; column < in; ++column) sum += x[column] * bf16_to_float(raw[row * in + column]);
        y[row] = sum;
    }
}

void rms_norm(const float * input, const float * weight, size_t tokens, size_t hidden, std::vector<float> & output) {
    output.assign(tokens * hidden, 0.0f);
    for (size_t time = 0; time < tokens; ++time) {
        double sum = 0.0;
        for (size_t column = 0; column < hidden; ++column) {
            const double value = input[time * hidden + column];
            sum += value * value;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(hidden)) + kEps);
        for (size_t column = 0; column < hidden; ++column) {
            output[time * hidden + column] = input[time * hidden + column] * inv * weight[column];
        }
    }
}

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

std::string layer_prefix(size_t layer) { return "model.layers." + std::to_string(layer) + "."; }

struct Routing {
    std::vector<int32_t> ids;    // [tokens * kTopK], ordered by weight
    std::vector<float> weights;  // [tokens * kTopK]
};

// KDA attention for one layer: float32 recurrence. The reference uses FLA's chunk
// path; the two agree to float32 precision on these inputs.
void run_kda(const MingTinyAssets & assets, size_t layer, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output) {
    const auto & source = *assets.weights;
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const std::string prefix = layer_prefix(layer);

    const auto norm_weight = load_float(source, prefix + "input_layernorm.weight", {static_cast<int64_t>(hidden)});
    std::vector<float> normed;
    rms_norm(input.data(), norm_weight.data(), tokens, hidden, normed);

    const auto q_w = load_float(source, prefix + "attention.q_proj.weight", {static_cast<int64_t>(kProjection), static_cast<int64_t>(hidden)});
    const auto k_w = load_float(source, prefix + "attention.k_proj.weight", {static_cast<int64_t>(kProjection), static_cast<int64_t>(hidden)});
    const auto v_w = load_float(source, prefix + "attention.v_proj.weight", {static_cast<int64_t>(kProjection), static_cast<int64_t>(hidden)});
    const auto q_c = load_float(source, prefix + "attention.q_conv1d.weight", {static_cast<int64_t>(kProjection), 1, 4});
    const auto k_c = load_float(source, prefix + "attention.k_conv1d.weight", {static_cast<int64_t>(kProjection), 1, 4});
    const auto v_c = load_float(source, prefix + "attention.v_conv1d.weight", {static_cast<int64_t>(kProjection), 1, 4});
    const auto f_w = load_float(source, prefix + "attention.f_proj.weight", {static_cast<int64_t>(kProjection), static_cast<int64_t>(hidden)});
    const auto g_w = load_float(source, prefix + "attention.g_proj.weight", {static_cast<int64_t>(kProjection), static_cast<int64_t>(hidden)});
    const auto b_w = load_float(source, prefix + "attention.b_proj.weight", {static_cast<int64_t>(kHeads), static_cast<int64_t>(hidden)});
    const auto a_log = load_float(source, prefix + "attention.A_log", {static_cast<int64_t>(kHeads)});
    const auto dt_bias = load_float(source, prefix + "attention.dt_bias", {static_cast<int64_t>(kProjection)});
    const auto o_norm_w = load_float(source, prefix + "attention.o_norm.weight", {static_cast<int64_t>(kHeadDim)});
    const auto o_w = load_float(source, prefix + "attention.o_proj.weight", {static_cast<int64_t>(hidden), static_cast<int64_t>(kProjection)});

    std::vector<float> q_all(tokens * kProjection), k_all(tokens * kProjection), v_all(tokens * kProjection);
    std::vector<float> q_row(kProjection), k_row(kProjection), v_row(kProjection);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, q_w.data(), hidden, kProjection, q_row.data());
        linear(row, k_w.data(), hidden, kProjection, k_row.data());
        linear(row, v_w.data(), hidden, kProjection, v_row.data());
        std::memcpy(q_all.data() + time * kProjection, q_row.data(), kProjection * sizeof(float));
        std::memcpy(k_all.data() + time * kProjection, k_row.data(), kProjection * sizeof(float));
        std::memcpy(v_all.data() + time * kProjection, v_row.data(), kProjection * sizeof(float));
    }
    std::vector<float> q_conv, k_conv, v_conv;
    short_conv_silu(q_all, q_c, tokens, kProjection, 4, q_conv);
    short_conv_silu(k_all, k_c, tokens, kProjection, 4, k_conv);
    short_conv_silu(v_all, v_c, tokens, kProjection, 4, v_conv);

    for (size_t time = 0; time < tokens; ++time) {
        for (size_t head = 0; head < kHeads; ++head) {
            for (int which = 0; which < 2; ++which) {
                std::vector<float> & vec = which == 0 ? q_conv : k_conv;
                float * base = vec.data() + time * kProjection + head * kHeadDim;
                double sum = 0.0;
                for (size_t dim = 0; dim < kHeadDim; ++dim) sum += static_cast<double>(base[dim]) * base[dim];
                const float denom = static_cast<float>(std::sqrt(sum)) + 1e-6f;
                for (size_t dim = 0; dim < kHeadDim; ++dim) base[dim] /= denom;
            }
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> state(kHeads * kHeadDim * kHeadDim, 0.0f);
    std::vector<float> f(kProjection), g(kProjection), b(kHeads), predicted(kHeadDim);
    std::vector<float> output_all(tokens * kProjection, 0.0f), gated(kProjection), attn(hidden), fused(tokens * hidden);
    for (size_t time = 0; time < tokens; ++time) {
        const float * q_t = q_conv.data() + time * kProjection;
        const float * k_t = k_conv.data() + time * kProjection;
        const float * v_t = v_conv.data() + time * kProjection;
        linear(normed.data() + time * hidden, f_w.data(), hidden, kProjection, f.data());
        linear(normed.data() + time * hidden, b_w.data(), hidden, kHeads, b.data());
        for (size_t head = 0; head < kHeads; ++head) {
            const float a = std::exp(a_log[head]);
            const float beta = sigmoid(b[head]);
            for (size_t dim = 0; dim < kHeadDim; ++dim) {
                const size_t index = head * kHeadDim + dim;
                g[index] = kLowerBound * sigmoid(a * (f[index] + dt_bias[index]));
            }
            float * s = state.data() + head * kHeadDim * kHeadDim;
            for (size_t row = 0; row < kHeadDim; ++row) {
                const float decay = std::exp(g[head * kHeadDim + row]);
                float * srow = s + row * kHeadDim;
                for (size_t column = 0; column < kHeadDim; ++column) srow[column] *= decay;
            }
            for (size_t column = 0; column < kHeadDim; ++column) {
                float sum = 0.0f;
                for (size_t row = 0; row < kHeadDim; ++row) sum += k_t[head * kHeadDim + row] * s[row * kHeadDim + column];
                predicted[column] = sum;
            }
            for (size_t row = 0; row < kHeadDim; ++row) {
                const float outer = beta * k_t[head * kHeadDim + row];
                float * srow = s + row * kHeadDim;
                for (size_t column = 0; column < kHeadDim; ++column) {
                    srow[column] += outer * (v_t[head * kHeadDim + column] - predicted[column]);
                }
            }
            for (size_t column = 0; column < kHeadDim; ++column) {
                float sum = 0.0f;
                for (size_t row = 0; row < kHeadDim; ++row) sum += scale * q_t[head * kHeadDim + row] * s[row * kHeadDim + column];
                output_all[time * kProjection + head * kHeadDim + column] = sum;
            }
        }
    }

    for (size_t time = 0; time < tokens; ++time) {
        linear(normed.data() + time * hidden, g_w.data(), hidden, kProjection, g.data());
        for (size_t head = 0; head < kHeads; ++head) {
            const float * base = output_all.data() + time * kProjection + head * kHeadDim;
            double sum = 0.0;
            for (size_t dim = 0; dim < kHeadDim; ++dim) sum += static_cast<double>(base[dim]) * base[dim];
            const float inv = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(kHeadDim)) + kEps);
            for (size_t dim = 0; dim < kHeadDim; ++dim) {
                const size_t index = head * kHeadDim + dim;
                gated[index] = base[dim] * inv * o_norm_w[dim] * sigmoid(g[index]);
            }
        }
        linear(gated.data(), o_w.data(), kProjection, hidden, attn.data());
        for (size_t column = 0; column < hidden; ++column) {
            fused[time * hidden + column] = input[time * hidden + column] + attn[column];
        }
    }
    output = std::move(fused);
}

void run_dense_mlp(const MingTinyAssets & assets, size_t layer, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output) {
    const auto & source = *assets.weights;
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const size_t intermediate = 4608;
    const std::string prefix = layer_prefix(layer);
    const auto post = load_float(source, prefix + "post_attention_layernorm.weight", {static_cast<int64_t>(hidden)});
    const auto gate_w = load_float(source, prefix + "mlp.gate_proj.weight", {static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto up_w = load_float(source, prefix + "mlp.up_proj.weight", {static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto down_w = load_float(source, prefix + "mlp.down_proj.weight", {static_cast<int64_t>(hidden), static_cast<int64_t>(intermediate)});

    std::vector<float> normed;
    rms_norm(input.data(), post.data(), tokens, hidden, normed);
    std::vector<float> gate(intermediate), up(intermediate), down(hidden);
    output.assign(input.size(), 0.0f);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, gate_w.data(), hidden, intermediate, gate.data());
        linear(row, up_w.data(), hidden, intermediate, up.data());
        for (size_t index = 0; index < intermediate; ++index) gate[index] = silu(gate[index]) * up[index];
        linear(gate.data(), down_w.data(), intermediate, hidden, down.data());
        for (size_t column = 0; column < hidden; ++column) {
            output[time * hidden + column] = input[time * hidden + column] + down[column];
        }
    }
}

// MoE block: sigmoid gate + expert bias, group-limited top-k, routed experts from the
// stacked banks, and the shared expert applied to the same normed input.
void run_moe(const MingTinyAssets & assets, size_t layer, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output, Routing & routing) {
    const auto & source = *assets.weights;
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const size_t intermediate = 512;
    const size_t experts = static_cast<size_t>(assets.config.experts);
    const size_t group_size = experts / kGroups;
    const std::string prefix = layer_prefix(layer);

    const auto post = load_float(source, prefix + "post_attention_layernorm.weight", {static_cast<int64_t>(hidden)});
    const auto gate_w = load_float(source, prefix + "mlp.gate.weight", {static_cast<int64_t>(experts), static_cast<int64_t>(hidden)});
    const auto bias = load_float(source, prefix + "mlp.gate.expert_bias", {static_cast<int64_t>(experts)});
    const auto gate_bank = load_raw(source, prefix + "mlp.experts.gate_proj.weight",
        {static_cast<int64_t>(experts), static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto up_bank = load_raw(source, prefix + "mlp.experts.up_proj.weight",
        {static_cast<int64_t>(experts), static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto down_bank = load_raw(source, prefix + "mlp.experts.down_proj.weight",
        {static_cast<int64_t>(experts), static_cast<int64_t>(hidden), static_cast<int64_t>(intermediate)});
    const auto shared_gate = load_float(source, prefix + "mlp.shared_experts.gate_proj.weight",
        {static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto shared_up = load_float(source, prefix + "mlp.shared_experts.up_proj.weight",
        {static_cast<int64_t>(intermediate), static_cast<int64_t>(hidden)});
    const auto shared_down = load_float(source, prefix + "mlp.shared_experts.down_proj.weight",
        {static_cast<int64_t>(hidden), static_cast<int64_t>(intermediate)});

    std::vector<float> normed;
    rms_norm(input.data(), post.data(), tokens, hidden, normed);
    routing.ids.assign(tokens * kTopK, 0);
    routing.weights.assign(tokens * kTopK, 0.0f);
    output.assign(input.size(), 0.0f);

    std::vector<float> logits(experts), scores(experts), routed(hidden);
    std::vector<float> expert_gate(intermediate), expert_up(intermediate), expert_down(hidden);
    std::vector<float> shared_a(intermediate), shared_b(intermediate), shared_out(hidden);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, gate_w.data(), hidden, experts, logits.data());
        for (size_t expert = 0; expert < experts; ++expert) scores[expert] = sigmoid(logits[expert]);

        std::vector<std::pair<float, size_t>> groups(kGroups);
        for (size_t group = 0; group < kGroups; ++group) {
            float best = -1e30f, second = -1e30f;
            for (size_t index = 0; index < group_size; ++index) {
                const size_t expert = group * group_size + index;
                const float value = scores[expert] + bias[expert];
                if (value > best) {
                    second = best;
                    best = value;
                } else if (value > second) {
                    second = value;
                }
            }
            groups[group] = {best + second, group};
        }
        std::vector<std::pair<float, size_t>> ordered = groups;
        std::sort(ordered.begin(), ordered.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });
        std::vector<bool> selected(kGroups, false);
        for (size_t index = 0; index < kTopGroups; ++index) selected[ordered[index].second] = true;

        std::vector<std::pair<float, size_t>> masked;
        masked.reserve(experts);
        for (size_t expert = 0; expert < experts; ++expert) {
            if (!selected[expert / group_size]) continue;
            masked.push_back({scores[expert] + bias[expert], expert});
        }
        std::sort(masked.begin(), masked.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        float denominator = 0.0f;
        for (size_t slot = 0; slot < kTopK; ++slot) denominator += scores[masked[slot].second];
        std::fill(routed.begin(), routed.end(), 0.0f);
        for (size_t slot = 0; slot < kTopK; ++slot) {
            const size_t expert = masked[slot].second;
            const float weight = scores[expert] / (denominator + 1e-20f) * kRoutedScaling;
            routing.ids[time * kTopK + slot] = static_cast<int32_t>(expert);
            routing.weights[time * kTopK + slot] = weight;
            bank_linear(gate_bank, expert, intermediate, hidden, row, expert_gate.data());
            bank_linear(up_bank, expert, intermediate, hidden, row, expert_up.data());
            for (size_t index = 0; index < intermediate; ++index) {
                expert_gate[index] = silu(expert_gate[index]) * expert_up[index];
            }
            bank_linear(down_bank, expert, hidden, intermediate, expert_gate.data(), expert_down.data());
            for (size_t column = 0; column < hidden; ++column) routed[column] += weight * expert_down[column];
        }

        linear(row, shared_gate.data(), hidden, intermediate, shared_a.data());
        linear(row, shared_up.data(), hidden, intermediate, shared_b.data());
        for (size_t index = 0; index < intermediate; ++index) shared_a[index] = silu(shared_a[index]) * shared_b[index];
        linear(shared_a.data(), shared_down.data(), intermediate, hidden, shared_out.data());
        for (size_t column = 0; column < hidden; ++column) {
            output[time * hidden + column] = input[time * hidden + column] + routed[column] + shared_out[column];
        }
    }
}

struct Capture {
    std::string name;
    std::string file;
    Shape shape;
    std::string dtype;
};

class DumpWriter {
public:
    explicit DumpWriter(std::filesystem::path directory) : directory_(std::move(directory)) {
        std::filesystem::create_directories(directory_);
    }

    void add_bytes(const std::string & name, const void * data, size_t size, const Shape & shape,
        const std::string & dtype) {
        const std::string file = name + ".bin";
        std::ofstream(directory_ / file, std::ios::binary)
            .write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
        captures_.push_back({name, file, shape, dtype});
    }

    void add(const std::string & name, const std::vector<float> & values, const Shape & shape) {
        add_bytes(name, values.data(), values.size() * sizeof(float), shape, "float32");
    }

    void manifest(const std::vector<int32_t> & prompt_ids) const {
        std::ofstream out(directory_ / "manifest.json");
        out << "{\n  \"stage\": \"B_thinker\",\n  \"prompt_ids\": [";
        for (size_t index = 0; index < prompt_ids.size(); ++index) out << (index ? "," : "") << prompt_ids[index];
        out << "],\n  \"captures\": [";
        for (size_t index = 0; index < captures_.size(); ++index) {
            const auto & capture = captures_[index];
            out << (index ? "," : "") << "\n    {\"name\": \"" << capture.name << "\", \"file\": \""
                << capture.file << "\", \"shape\": [";
            for (size_t axis = 0; axis < capture.shape.size(); ++axis) {
                out << (axis ? "," : "") << capture.shape[axis];
            }
            out << "], \"dtype\": \"" << capture.dtype << "\"}";
        }
        out << "\n  ]\n}\n";
    }

private:
    std::filesystem::path directory_;
    std::vector<Capture> captures_;
};

}  // namespace

void dump_thinker_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::filesystem::path & directory) {
    const size_t tokens = prompt_ids.size();
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const size_t layers = static_cast<size_t>(assets.config.layers);
    DumpWriter writer(directory);

    const auto table = load_float(*assets.weights, "model.word_embeddings.weight",
        {assets.config.vocabulary_size, assets.config.hidden_size});
    std::vector<float> state(tokens * hidden);
    for (size_t row = 0; row < tokens; ++row) {
        std::memcpy(state.data() + row * hidden, table.data() + static_cast<size_t>(prompt_ids[row]) * hidden,
            hidden * sizeof(float));
    }
    writer.add("embedding", state, {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});

    const size_t limit = std::min<size_t>(layers, 2);  // layers 0 and 1 for now
    for (size_t layer = 0; layer < limit; ++layer) {
        std::vector<float> after_attention;
        run_kda(assets, layer, state, tokens, after_attention);
        std::vector<float> after_block;
        if (layer < static_cast<size_t>(assets.config.first_k_dense_replace)) {
            run_dense_mlp(assets, layer, after_attention, tokens, after_block);
        } else {
            Routing routing;
            run_moe(assets, layer, after_attention, tokens, after_block, routing);
            const std::string tag = "routing_layer_" + std::to_string(layer);
            writer.add_bytes(tag + "_ids", routing.ids.data(), routing.ids.size() * sizeof(int32_t),
                {static_cast<int64_t>(tokens), static_cast<int64_t>(kTopK)}, "int32");
            writer.add(tag + "_weights", routing.weights,
                {static_cast<int64_t>(tokens), static_cast<int64_t>(kTopK)});
        }
        state = std::move(after_block);
        writer.add("layer_" + std::to_string(layer), state,
            {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
    }
    writer.manifest(prompt_ids);
}

}  // namespace engine::models::ming3_tts_tiny
