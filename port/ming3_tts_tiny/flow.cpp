#include "engine/models/ming3_tts_tiny/flow.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

using Shape = std::vector<int64_t>;

constexpr size_t kLatent = 64;
constexpr size_t kHidden = 1024;
constexpr size_t kLayers = 16;
constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 64;
constexpr size_t kMlp = 4096;
constexpr size_t kCond = 1536;
constexpr float kEps = 1e-6f;

std::string prefix() { return "flowloss.cfm.model."; }

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(float));
    return value;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

// y[o] = sum_i x[i] * weight[o * in + i] + bias[o]
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

class FlowModel {
public:
    explicit FlowModel(const assets::TensorSource & source) : source_(source) {}

    const std::vector<float> & tensor(const std::string & suffix, const Shape & shape) {
        const std::string name = prefix() + suffix;
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
        linear(x, tensor(suffix + ".weight", {static_cast<int64_t>(out), static_cast<int64_t>(in)}).data(),
            tensor(suffix + ".bias", {static_cast<int64_t>(out)}).data(), in, out, y);
    }

    void norm(const float * x, const std::string & suffix, float * y) {
        rms_norm(x, tensor(suffix + ".weight", {static_cast<int64_t>(kHidden)}).data(), kHidden, y);
    }

private:
    const assets::TensorSource & source_;
    std::map<std::string, std::vector<float>> cache_;
};

// DiT forward for the three-token sequence [t + c, x_history, x_now].
// cond_out receives the condition after c_embedder for the comparison harness.
void dit_forward(FlowModel & model, const std::vector<float> & cond, float timestep,
    const std::vector<float> & history, const std::vector<float> & x, std::vector<float> & velocity,
    std::vector<float> * cond_out) {
    // timestep embedding: sinusoidal 256 (scale 1000), then MLP with SiLU
    std::vector<float> time_in(256);
    {
        const size_t half = 128;
        const float log_scale = std::log(10000.0f) / static_cast<float>(half - 1);
        for (size_t index = 0; index < half; ++index) {
            const float freq = std::exp(-static_cast<float>(index) * log_scale);
            const float angle = 1000.0f * timestep * freq;
            time_in[index] = std::sin(angle);
            time_in[half + index] = std::cos(angle);
        }
    }
    std::vector<float> time_hidden(kHidden), t_emb(kHidden);
    model.linear_op(time_in.data(), "t_embedder.time_mlp.0", 256, kHidden, time_hidden.data());
    for (auto & value : time_hidden) value = silu(value);
    model.linear_op(time_hidden.data(), "t_embedder.time_mlp.2", kHidden, kHidden, t_emb.data());

    std::vector<float> x_now(kHidden), x_hist(kHidden), c(kHidden);
    model.linear_op(x.data(), "x_embedder", kLatent, kHidden, x_now.data());
    model.linear_op(history.data(), "x_embedder", kLatent, kHidden, x_hist.data());
    model.linear_op(cond.data(), "c_embedder.cond_embedder", kCond, kHidden, c.data());
    if (cond_out != nullptr) *cond_out = c;

    std::vector<float> tokens(3 * kHidden);
    for (size_t index = 0; index < kHidden; ++index) {
        tokens[index] = t_emb[index] + c[index];
        tokens[kHidden + index] = x_hist[index];
        tokens[2 * kHidden + index] = x_now[index];
    }

    // interleaved RoPE tables for positions 0..2
    std::vector<float> cos_table(3 * kHeadDim), sin_table(3 * kHeadDim);
    {
        const auto & inv_freq = model.tensor("rotary_embed.inv_freq", {static_cast<int64_t>(kHeadDim / 2)});
        for (size_t position = 0; position < 3; ++position) {
            for (size_t pair = 0; pair < kHeadDim / 2; ++pair) {
                const float angle = static_cast<float>(position) * inv_freq[pair];
                cos_table[position * kHeadDim + 2 * pair] = std::cos(angle);
                cos_table[position * kHeadDim + 2 * pair + 1] = std::cos(angle);
                sin_table[position * kHeadDim + 2 * pair] = std::sin(angle);
                sin_table[position * kHeadDim + 2 * pair + 1] = std::sin(angle);
            }
        }
    }

    std::vector<float> normed(kHidden), q(3 * kHidden), k(3 * kHidden), v(3 * kHidden);
    std::vector<float> attn(3 * kHidden), projected(3 * kHidden), mlp_in(kMlp), mlp_out(kHidden);
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const std::string block = "blocks." + std::to_string(layer) + ".";
        for (size_t token = 0; token < 3; ++token) {
            model.norm(tokens.data() + token * kHidden, block + "norm1", normed.data());
            model.linear_op(normed.data(), block + "attn.to_q", kHidden, kHidden, q.data() + token * kHidden);
            model.linear_op(normed.data(), block + "attn.to_k", kHidden, kHidden, k.data() + token * kHidden);
            model.linear_op(normed.data(), block + "attn.to_v", kHidden, kHidden, v.data() + token * kHidden);
        }
        // rotate q/k in place, per token, per head, interleaved pairs
        for (size_t token = 0; token < 3; ++token) {
            for (size_t head = 0; head < kHeads; ++head) {
                for (int which = 0; which < 2; ++which) {
                    std::vector<float> & vec = which == 0 ? q : k;
                    float * base = vec.data() + token * kHidden + head * kHeadDim;
                    for (size_t pair = 0; pair < kHeadDim / 2; ++pair) {
                        const float c0 = cos_table[token * kHeadDim + 2 * pair];
                        const float s0 = sin_table[token * kHeadDim + 2 * pair];
                        const float x1 = base[2 * pair], x2 = base[2 * pair + 1];
                        base[2 * pair] = x1 * c0 - x2 * s0;
                        base[2 * pair + 1] = x1 * s0 + x2 * c0;
                    }
                }
            }
        }
        // full attention over the three tokens (no causal mask)
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        for (size_t head = 0; head < kHeads; ++head) {
            std::vector<float> scores(3);
            for (size_t token = 0; token < 3; ++token) {
                float best = -1e30f;
                for (size_t key = 0; key < 3; ++key) {
                    float sum = 0.0f;
                    for (size_t dim = 0; dim < kHeadDim; ++dim) {
                        sum += q[token * kHidden + head * kHeadDim + dim] * k[key * kHidden + head * kHeadDim + dim];
                    }
                    scores[key] = sum * scale;
                    best = std::max(best, scores[key]);
                }
                float total = 0.0f;
                for (size_t key = 0; key < 3; ++key) {
                    scores[key] = std::exp(scores[key] - best);
                    total += scores[key];
                }
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    float sum = 0.0f;
                    for (size_t key = 0; key < 3; ++key) {
                        sum += scores[key] / total * v[key * kHidden + head * kHeadDim + dim];
                    }
                    attn[token * kHidden + head * kHeadDim + dim] = sum;
                }
            }
        }
        for (size_t token = 0; token < 3; ++token) {
            model.linear_op(attn.data() + token * kHidden, block + "attn.to_out.0", kHidden, kHidden,
                projected.data() + token * kHidden);
            for (size_t index = 0; index < kHidden; ++index) tokens[token * kHidden + index] += projected[token * kHidden + index];

            model.norm(tokens.data() + token * kHidden, block + "norm2", normed.data());
            model.linear_op(normed.data(), block + "mlp.ff.0.0", kHidden, kMlp, mlp_in.data());
            for (auto & value : mlp_in) value = gelu_tanh(value);
            model.linear_op(mlp_in.data(), block + "mlp.ff.2", kMlp, kHidden, mlp_out.data());
            for (size_t index = 0; index < kHidden; ++index) tokens[token * kHidden + index] += mlp_out[index];
        }
    }

    // final layer on the last token only (the sampled patch)
    model.norm(tokens.data() + 2 * kHidden, "final_layer.norm_final", normed.data());
    linear(normed.data(), model.tensor("final_layer.linear.weight", {static_cast<int64_t>(kLatent), static_cast<int64_t>(kHidden)}).data(),
        model.tensor("final_layer.linear.bias", {static_cast<int64_t>(kLatent)}).data(), kHidden, kLatent,
        velocity.data());
}

std::vector<float> epss_timesteps(int steps) {
    std::vector<float> grid;
    if (steps == 10) {
        grid = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
    } else {
        for (int index = 0; index <= steps; ++index) grid.push_back(32.0f * static_cast<float>(index) / static_cast<float>(steps));
    }
    for (auto & value : grid) value /= 32.0f;
    return grid;
}

struct Capture {
    std::string name;
    Shape shape;
    std::vector<float> values;
};

}  // namespace

void dump_flow_probe(const MingTinyAssets & assets, const FlowRequest & request,
    const std::filesystem::path & directory) {
    FlowModel model(*assets.weights);
    std::vector<float> history(kLatent, 0.0f);
    const std::vector<float> zero_cond(kCond, 0.0f);

    std::vector<float> grid = epss_timesteps(request.steps);
    for (auto & value : grid) value = value - (std::cos(3.14159265358979323846f / 2.0f * value) - 1.0f + value);  // sway -1

    std::vector<float> y(kLatent);
    std::memcpy(y.data(), request.noise.data(), kLatent * sizeof(float));

    std::vector<Capture> captures;
    captures.push_back({"flow_noise", {1, 1, static_cast<int64_t>(kLatent)}, y});
    {
        std::vector<float> velocity(kLatent), cond_post;
        dit_forward(model, request.cond, grid.front(), history, y, velocity, &cond_post);
        captures.push_back({"cond_post_c_embedder", {1, 1, static_cast<int64_t>(kHidden)}, cond_post});
    }
    captures.push_back({"flow_trajectory", {1, 1, static_cast<int64_t>(kLatent)}, y});

    std::vector<float> cond_velocities, null_velocities, guided_velocities;
    for (size_t step = 0; step + 1 < grid.size(); ++step) {
        std::vector<float> conditioned(kLatent), nulled(kLatent);
        dit_forward(model, request.cond, grid[step], history, y, conditioned, nullptr);
        if (request.cfg != 1.0f) {
            dit_forward(model, zero_cond, grid[step], history, y, nulled, nullptr);
        } else {
            nulled = conditioned;
        }
        std::vector<float> guided(kLatent);
        for (size_t index = 0; index < kLatent; ++index) {
            guided[index] = conditioned[index] + (conditioned[index] - nulled[index]) * request.cfg;
        }
        const float dt = grid[step + 1] - grid[step];
        for (size_t index = 0; index < kLatent; ++index) y[index] += dt * guided[index];

        cond_velocities.insert(cond_velocities.end(), conditioned.begin(), conditioned.end());
        null_velocities.insert(null_velocities.end(), nulled.begin(), nulled.end());
        guided_velocities.insert(guided_velocities.end(), guided.begin(), guided.end());
        captures.push_back({"flow_x_step_" + std::to_string(step), {1, 1, static_cast<int64_t>(kLatent)}, y});
    }

    const int64_t steps = static_cast<int64_t>(cond_velocities.size() / kLatent);
    captures.push_back({"flow_velocity_cond", {steps, 1, static_cast<int64_t>(kLatent)}, cond_velocities});
    captures.push_back({"flow_velocity_null", {steps, 1, static_cast<int64_t>(kLatent)}, null_velocities});
    captures.push_back({"flow_velocity_guided", {steps, 1, static_cast<int64_t>(kLatent)}, guided_velocities});
    captures.push_back({"flow_latents", {1, 1, static_cast<int64_t>(kLatent)}, y});

    std::filesystem::create_directories(directory);
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"stage\": \"C_flow\",\n  \"captures\": [";
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
    std::printf("  flow probe      : %zu captures, cfg=%.2f steps=%d, latent rms=%.6f\n",
        captures.size(), request.cfg, request.steps,
        std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0) / static_cast<double>(kLatent)));
}

}  // namespace engine::models::ming3_tts_tiny
