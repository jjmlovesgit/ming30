#include "engine/models/ming3_tts_tiny/thinker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <chrono>
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

// Load a tensor in its *stored* dtype. The package keeps A_log, dt_bias and the MoE
// expert_bias in F32; requesting BF16 for them silently rounds the gate and the router
// bias, which is enough to move routing cutoffs.
// Resident weight cache. Decode steps need the same weights every step, and re-reading
// plus converting them from the package dominates single-token latency. Bounded by a byte
// budget with FIFO eviction so the process cannot grow without limit.
struct WeightCache {
    std::map<std::string, std::vector<float>> floats;
    std::map<std::string, RawTensor> raws;
    std::deque<std::string> order;
    size_t bytes = 0;
    size_t budget = static_cast<size_t>(16) << 30;

    bool enabled() const {
        const char * flag = std::getenv("MING3_TINY_WEIGHT_CACHE");
        return flag == nullptr || *flag == '\0' || std::atoi(flag) != 0;
    }

    void evict() {
        while (bytes > budget && !order.empty()) {
            const std::string victim = order.front();
            order.pop_front();
            const auto raw = raws.find(victim);
            if (raw != raws.end()) {
                bytes -= raw->second.bytes.size();
                raws.erase(raw);
                continue;
            }
            const auto found = floats.find(victim);
            if (found == floats.end()) continue;
            bytes -= found->second.size() * sizeof(float);
            floats.erase(found);
        }
    }
};

WeightCache & weight_cache() {
    static WeightCache cache;
    return cache;
}

std::vector<float> load_float(const assets::TensorSource & source, const std::string & name, const Shape & shape) {
    auto & cache = weight_cache();
    if (cache.enabled()) {
        const auto found = cache.floats.find(name);
        if (found != cache.floats.end()) return found->second;   // memcpy out, no conversion
    }
    const auto meta = source.require_metadata(name);
    std::vector<float> values;
    if (meta.dtype == "F32" || meta.dtype == "f32") {
        const auto tensor = source.require_tensor(name, assets::TensorStorageType::F32, shape);
        const auto * raw = reinterpret_cast<const float *>(tensor.bytes.data());
        values.assign(raw, raw + tensor.bytes.size() / sizeof(float));
    } else {
        const auto tensor = source.require_tensor(name, assets::TensorStorageType::BF16, shape);
        const auto * raw = reinterpret_cast<const uint16_t *>(tensor.bytes.data());
        values.resize(tensor.bytes.size() / sizeof(uint16_t));
        for (size_t index = 0; index < values.size(); ++index) values[index] = bf16_to_float(raw[index]);
    }
    if (cache.enabled()) {
        cache.bytes += values.size() * sizeof(float);
        cache.order.push_back(name);
        cache.floats.emplace(name, values);
        cache.evict();
    }
    return values;
}

// Expert banks are the dominant read: three banks per MoE layer, ~200 MB each, needed
// again on every decode step. Keep their bytes resident under the same budget.
RawTensor load_raw(const assets::TensorSource & source, const std::string & name, const Shape & shape) {
    auto & cache = weight_cache();
    if (cache.enabled()) {
        const auto found = cache.raws.find(name);
        if (found != cache.raws.end()) return found->second;
    }
    auto tensor = source.require_tensor(name, assets::TensorStorageType::BF16, shape);
    RawTensor value{std::move(tensor.bytes), shape};
    if (cache.enabled()) {
        cache.bytes += value.bytes.size();
        cache.order.push_back(name);
        cache.raws.emplace(name, value);
        cache.evict();
    }
    return value;
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

struct KdaTrace {
    bool enabled = false;
    std::vector<float> q_conv, k_conv, v_conv;   // after short conv + SiLU
    std::vector<float> q_norm, k_norm;           // after per-head L2 normalisation
    std::vector<float> gate;                     // lower-bounded gate, log space
    std::vector<float> beta;                     // per head
    std::vector<float> heads;                    // q^T S before the gated norm
    std::vector<float> attn;                     // after gated norm + o_proj
};

struct Routing {
    std::vector<int32_t> ids;    // [tokens * kTopK], ordered by weight
    std::vector<float> weights;  // [tokens * kTopK]
};

// KDA attention for one layer: float32 recurrence. The reference uses FLA's chunk
// path; the two agree to float32 precision on these inputs.
void run_kda(const MingTinyAssets & assets, size_t layer, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output, std::vector<float> & attn_only, KdaTrace * trace = nullptr,
    KdaCache * cache = nullptr) {
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

    attn_only.assign(tokens * hidden, 0.0f);
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
    const bool resume = cache != nullptr && cache->initialized;
    if (resume) {
        const auto conv_with_history = [&](const std::vector<float> & row, const std::vector<float> & history) {
            std::vector<float> buffer(4 * kProjection);
            std::memcpy(buffer.data(), history.data(), 3 * kProjection * sizeof(float));
            std::memcpy(buffer.data() + 3 * kProjection, row.data(), kProjection * sizeof(float));
            return buffer;
        };
        const auto q_buf = conv_with_history(q_all, cache->q_hist);
        const auto k_buf = conv_with_history(k_all, cache->k_hist);
        const auto v_buf = conv_with_history(v_all, cache->v_hist);
        std::vector<float> q_tmp, k_tmp, v_tmp;
        short_conv_silu(q_buf, q_c, 4, kProjection, 4, q_tmp);
        short_conv_silu(k_buf, k_c, 4, kProjection, 4, k_tmp);
        short_conv_silu(v_buf, v_c, 4, kProjection, 4, v_tmp);
        q_conv.assign(q_tmp.end() - kProjection, q_tmp.end());
        k_conv.assign(k_tmp.end() - kProjection, k_tmp.end());
        v_conv.assign(v_tmp.end() - kProjection, v_tmp.end());
        cache->q_hist.erase(cache->q_hist.begin(), cache->q_hist.begin() + kProjection);
        cache->q_hist.insert(cache->q_hist.end(), q_all.begin(), q_all.end());
        cache->k_hist.erase(cache->k_hist.begin(), cache->k_hist.begin() + kProjection);
        cache->k_hist.insert(cache->k_hist.end(), k_all.begin(), k_all.end());
        cache->v_hist.erase(cache->v_hist.begin(), cache->v_hist.begin() + kProjection);
        cache->v_hist.insert(cache->v_hist.end(), v_all.begin(), v_all.end());
    } else {
        short_conv_silu(q_all, q_c, tokens, kProjection, 4, q_conv);
        short_conv_silu(k_all, k_c, tokens, kProjection, 4, k_conv);
        short_conv_silu(v_all, v_c, tokens, kProjection, 4, v_conv);
    }
    if (trace != nullptr && trace->enabled) {
        trace->q_conv = q_conv;
        trace->k_conv = k_conv;
        trace->v_conv = v_conv;
    }

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

    if (trace != nullptr && trace->enabled) {
        trace->q_norm = q_conv;
        trace->k_norm = k_conv;
        trace->beta.assign(tokens * kHeads, 0.0f);
        trace->gate.assign(tokens * kProjection, 0.0f);
        trace->heads.assign(tokens * kProjection, 0.0f);
        trace->attn.assign(tokens * hidden, 0.0f);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> state(kHeads * kHeadDim * kHeadDim, 0.0f);
    if (resume && cache->state.size() == state.size()) state = cache->state;
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
            if (trace != nullptr && trace->enabled) {
                trace->beta[time * kHeads + head] = beta;
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    trace->gate[time * kProjection + head * kHeadDim + dim] = g[head * kHeadDim + dim];
                }
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
                if (trace != nullptr && trace->enabled) {
                    trace->heads[time * kProjection + head * kHeadDim + column] = sum;
                }
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
        if (trace != nullptr && trace->enabled) {
            for (size_t column = 0; column < hidden; ++column) trace->attn[time * hidden + column] = attn[column];
        }
        for (size_t column = 0; column < hidden; ++column) {
            attn_only[time * hidden + column] = attn[column];
            fused[time * hidden + column] = input[time * hidden + column] + attn[column];
        }
    }
    if (cache != nullptr) {
        cache->state = state;
        if (!resume) {
            cache->q_hist.assign(kProjection * 3, 0.0f);
            cache->k_hist.assign(kProjection * 3, 0.0f);
            cache->v_hist.assign(kProjection * 3, 0.0f);
            const size_t take = tokens < 3 ? tokens : 3;
            for (size_t row = 0; row < take; ++row) {
                const size_t source = (tokens - take + row) * kProjection;
                const size_t destination = (3 - take + row) * kProjection;
                std::memcpy(cache->q_hist.data() + destination, q_all.data() + source, kProjection * sizeof(float));
                std::memcpy(cache->k_hist.data() + destination, k_all.data() + source, kProjection * sizeof(float));
                std::memcpy(cache->v_hist.data() + destination, v_all.data() + source, kProjection * sizeof(float));
            }
        }
        cache->initialized = true;
    }
    output = std::move(fused);
}

// MLA attention: LoRA q/kv projections, interleaved RoPE on the 64-dim slice, causal
// SDPA, head-wise sigmoid output gate, then the dense projection.
void run_mla(const MingTinyAssets & assets, size_t layer, const std::vector<float> & input, size_t tokens,
    std::vector<float> & output, std::vector<float> & attn_only, KdaTrace * trace = nullptr,
    MlaCache * cache = nullptr) {
    const auto & source = *assets.weights;
    const size_t hidden = static_cast<size_t>(assets.config.hidden_size);
    const size_t heads = kHeads;
    const size_t qk_nope = 128, qk_rope = 64, v_head = 128;
    const size_t q_lora = 256, kv_lora = 512;
    const size_t qk_head = qk_nope + qk_rope;
    const std::string prefix = layer_prefix(layer);

    const auto norm_w = load_float(source, prefix + "input_layernorm.weight", {static_cast<int64_t>(hidden)});
    const auto q_a = load_float(source, prefix + "attention.q_a_proj.weight", {static_cast<int64_t>(q_lora), static_cast<int64_t>(hidden)});
    const auto q_a_norm = load_float(source, prefix + "attention.q_a_layernorm.weight", {static_cast<int64_t>(q_lora)});
    const auto q_b = load_float(source, prefix + "attention.q_b_proj.weight", {static_cast<int64_t>(heads * qk_head), static_cast<int64_t>(q_lora)});
    const auto kv_a = load_float(source, prefix + "attention.kv_a_proj_with_mqa.weight", {static_cast<int64_t>(kv_lora + qk_rope), static_cast<int64_t>(hidden)});
    const auto kv_a_norm = load_float(source, prefix + "attention.kv_a_layernorm.weight", {static_cast<int64_t>(kv_lora)});
    const auto kv_b = load_float(source, prefix + "attention.kv_b_proj.weight", {static_cast<int64_t>(heads * (qk_nope + v_head)), static_cast<int64_t>(kv_lora)});
    const auto g_w = load_float(source, prefix + "attention.g_proj.weight", {static_cast<int64_t>(heads), static_cast<int64_t>(hidden)});
    const auto dense = load_float(source, prefix + "attention.dense.weight", {static_cast<int64_t>(hidden), static_cast<int64_t>(heads * v_head)});

    std::vector<float> normed;
    rms_norm(input.data(), norm_w.data(), tokens, hidden, normed);
    attn_only.assign(tokens * hidden, 0.0f);
    const size_t offset = cache != nullptr ? cache->tokens : 0;

    std::vector<float> q_all(tokens * heads * qk_head, 0.0f);
    std::vector<float> k_all(tokens * heads * qk_head, 0.0f);
    std::vector<float> v_all(tokens * heads * v_head, 0.0f);
    std::vector<float> qa(q_lora), qn(q_lora), q_full(heads * qk_head);
    std::vector<float> kvrow(kv_lora + qk_rope), kvn(kv_lora), kv_full(heads * (qk_nope + v_head));
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, q_a.data(), hidden, q_lora, qa.data());
        double qsum = 0.0;
        for (size_t index = 0; index < q_lora; ++index) qsum += static_cast<double>(qa[index]) * qa[index];
        const float qinv = 1.0f / std::sqrt(static_cast<float>(qsum / static_cast<double>(q_lora)) + kEps);
        for (size_t index = 0; index < q_lora; ++index) qn[index] = qa[index] * qinv * q_a_norm[index];
        linear(qn.data(), q_b.data(), q_lora, heads * qk_head, q_full.data());

        linear(row, kv_a.data(), hidden, kv_lora + qk_rope, kvrow.data());
        double kvsum = 0.0;
        for (size_t index = 0; index < kv_lora; ++index) kvsum += static_cast<double>(kvrow[index]) * kvrow[index];
        const float kv_inv = 1.0f / std::sqrt(static_cast<float>(kvsum / static_cast<double>(kv_lora)) + kEps);
        for (size_t index = 0; index < kv_lora; ++index) kvn[index] = kvrow[index] * kv_inv * kv_a_norm[index];
        linear(kvn.data(), kv_b.data(), kv_lora, heads * (qk_nope + v_head), kv_full.data());

        for (size_t head = 0; head < heads; ++head) {
            for (size_t dim = 0; dim < qk_nope; ++dim) {
                q_all[(time * heads + head) * qk_head + dim] = q_full[head * qk_head + dim];
                k_all[(time * heads + head) * qk_head + dim] = kv_full[head * (qk_nope + v_head) + dim];
            }
            for (size_t dim = 0; dim < qk_rope; ++dim) {
                q_all[(time * heads + head) * qk_head + qk_nope + dim] = q_full[head * qk_head + qk_nope + dim];
                k_all[(time * heads + head) * qk_head + qk_nope + dim] = kvrow[kv_lora + dim];
            }
            for (size_t dim = 0; dim < v_head; ++dim) {
                v_all[(time * heads + head) * v_head + dim] = kv_full[head * (qk_nope + v_head) + qk_nope + dim];
            }
        }
    }

    // interleaved RoPE on the rotary slice (rope_interleave=True)
    for (size_t time = 0; time < tokens; ++time) {
        for (size_t pair = 0; pair < qk_rope / 2; ++pair) {
            const float inv_freq = std::pow(10000.0f, -2.0f * static_cast<float>(pair) / static_cast<float>(qk_rope));
            const float angle = static_cast<float>(offset + time) * inv_freq;
            const float c = std::cos(angle), s = std::sin(angle);
            for (size_t head = 0; head < heads; ++head) {
                for (int which = 0; which < 2; ++which) {
                    std::vector<float> & vec = which == 0 ? q_all : k_all;
                    float * base = vec.data() + (time * heads + head) * qk_head + qk_nope;
                    const float x1 = base[2 * pair], x2 = base[2 * pair + 1];
                    base[2 * pair] = x1 * c - x2 * s;
                    base[2 * pair + 1] = x1 * s + x2 * c;
                }
            }
        }
    }

    // With a cache, keys and values of every earlier position are kept; a one-token
    // step appends its own and attends over the whole history.
    const std::vector<float> * keys_ptr = &k_all;
    const std::vector<float> * values_ptr = &v_all;
    size_t history = tokens;
    if (cache != nullptr) {
        const size_t previous = cache->tokens;
        cache->keys.resize((previous + tokens) * heads * qk_head);
        cache->values.resize((previous + tokens) * heads * v_head);
        std::memcpy(cache->keys.data() + previous * heads * qk_head, k_all.data(),
            tokens * heads * qk_head * sizeof(float));
        std::memcpy(cache->values.data() + previous * heads * v_head, v_all.data(),
            tokens * heads * v_head * sizeof(float));
        cache->tokens = previous + tokens;
        keys_ptr = &cache->keys;
        values_ptr = &cache->values;
        history = cache->tokens;
    }
    const auto & keys = *keys_ptr;
    const auto & values = *values_ptr;

    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head));
    std::vector<float> attention_out(tokens * heads * v_head, 0.0f);
    std::vector<float> scores(history);
    for (size_t time = 0; time < tokens; ++time) {
        const size_t query_index = offset + time;
        for (size_t head = 0; head < heads; ++head) {
            const float * q = q_all.data() + (time * heads + head) * qk_head;
            float best = -1e30f;
            for (size_t key = 0; key <= query_index; ++key) {
                const float * k = keys.data() + (key * heads + head) * qk_head;
                float sum = 0.0f;
                for (size_t dim = 0; dim < qk_head; ++dim) sum += q[dim] * k[dim];
                scores[key] = sum * scale;
                best = std::max(best, scores[key]);
            }
            float total = 0.0f;
            for (size_t key = 0; key <= query_index; ++key) {
                scores[key] = std::exp(scores[key] - best);
                total += scores[key];
            }
            float * out = attention_out.data() + (time * heads + head) * v_head;
            for (size_t key = 0; key <= query_index; ++key) {
                const float weight = scores[key] / total;
                const float * v = values.data() + (key * heads + head) * v_head;
                for (size_t dim = 0; dim < v_head; ++dim) out[dim] += weight * v[dim];
            }
        }
    }

    std::vector<float> gated(heads * v_head), attn(hidden), fused(tokens * hidden);
    std::vector<float> gate_scores(heads), row_out(hidden);
    for (size_t time = 0; time < tokens; ++time) {
        linear(normed.data() + time * hidden, g_w.data(), hidden, heads, gate_scores.data());
        for (size_t index = 0; index < gated.size(); ++index) {
            const size_t head = index / v_head;
            gated[index] = attention_out[time * heads * v_head + index] * sigmoid(gate_scores[head]);
        }
        linear(gated.data(), dense.data(), heads * v_head, hidden, attn.data());
        for (size_t column = 0; column < hidden; ++column) {
            attn_only[time * hidden + column] = attn[column];
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
    std::vector<float> & output, Routing & routing, std::vector<float> & pre_router,
    std::vector<float> & router_scores) {
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
    pre_router = normed;
    router_scores.assign(tokens * experts, 0.0f);
    routing.ids.assign(tokens * kTopK, 0);
    routing.weights.assign(tokens * kTopK, 0.0f);
    output.assign(input.size(), 0.0f);

    std::vector<float> logits(experts), scores(experts), routed(hidden);
    std::vector<float> expert_gate(intermediate), expert_up(intermediate), expert_down(hidden);
    std::vector<float> shared_a(intermediate), shared_b(intermediate), shared_out(hidden);
    for (size_t time = 0; time < tokens; ++time) {
        const float * row = normed.data() + time * hidden;
        linear(row, gate_w.data(), hidden, experts, logits.data());
        for (size_t expert = 0; expert < experts; ++expert) {
            scores[expert] = sigmoid(logits[expert]);
            router_scores[time * experts + expert] = scores[expert];
        }

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

namespace {

void run_all_layers(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    ThinkerCache * cache, std::vector<float> & final_norm) {
    std::vector<float> state = embeds;
    for (size_t layer = 0; layer < static_cast<size_t>(assets.config.layers); ++layer) {
        std::vector<float> after_attention, attention_only;
        if (layer % 4 == 3) {
            MlaCache * mla = cache != nullptr ? &cache->mla[layer] : nullptr;
            run_mla(assets, layer, state, tokens, after_attention, attention_only, nullptr, mla);
        } else {
            KdaCache * kda = cache != nullptr ? &cache->kda[layer] : nullptr;
            run_kda(assets, layer, state, tokens, after_attention, attention_only, nullptr, kda);
        }
        std::vector<float> after_block;
        if (layer < static_cast<size_t>(assets.config.first_k_dense_replace)) {
            run_dense_mlp(assets, layer, after_attention, tokens, after_block);
        } else {
            Routing routing;
            std::vector<float> pre_router, router_scores;
            run_moe(assets, layer, after_attention, tokens, after_block, routing, pre_router, router_scores);
        }
        state = std::move(after_block);
    }
    const auto norm_weight = load_float(*assets.weights, "model.norm.weight",
        {static_cast<int64_t>(assets.config.hidden_size)});
    rms_norm(state.data(), norm_weight.data(), tokens, static_cast<size_t>(assets.config.hidden_size), final_norm);
}

}  // namespace

void thinker_forward_embeds(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    std::vector<float> & final_norm) {
    run_all_layers(assets, embeds, tokens, nullptr, final_norm);
}

void thinker_prefill_cached(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    ThinkerCache & cache, std::vector<float> & final_norm) {
    cache.kda.assign(static_cast<size_t>(assets.config.layers), KdaCache{});
    cache.mla.assign(static_cast<size_t>(assets.config.layers), MlaCache{});
    run_all_layers(assets, embeds, tokens, &cache, final_norm);
}

void thinker_step_cached(const MingTinyAssets & assets, const std::vector<float> & embed,
    ThinkerCache & cache, std::vector<float> & final_norm) {
    run_all_layers(assets, embed, 1, &cache, final_norm);
}

#if defined(ENGINE_HAS_MING_GPU_BENCH)
extern "C" int ming_gpu_layer_benchmark(const float * q_w, const float * k_w, const float * v_w,
    const float * f_w, const float * g_w, const float * b_w, const float * o_w, const float * x,
    int iterations, double * out);
#endif

void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {
    constexpr size_t kHiddenLlm = 1536;
#if !defined(ENGINE_HAS_MING_GPU_BENCH)
    std::printf("  gpu benchmark   : not built with CUDA\n");
    (void) assets;
    (void) iterations;
#else
    const auto & source = *assets.weights;
    const std::string prefix = "model.layers.0.attention.";
    const Shape projection{static_cast<int64_t>(kProjection), static_cast<int64_t>(kHiddenLlm)};
    const auto q_w = load_float(source, prefix + "q_proj.weight", projection);
    const auto k_w = load_float(source, prefix + "k_proj.weight", projection);
    const auto v_w = load_float(source, prefix + "v_proj.weight", projection);
    const auto f_w = load_float(source, prefix + "f_proj.weight", projection);
    const auto g_w = load_float(source, prefix + "g_proj.weight", projection);
    const auto b_w = load_float(source, prefix + "b_proj.weight",
        {static_cast<int64_t>(kHeads), static_cast<int64_t>(kHiddenLlm)});
    const auto o_w = load_float(source, prefix + "o_proj.weight",
        {static_cast<int64_t>(kHiddenLlm), static_cast<int64_t>(kProjection)});

    std::vector<float> x(kHiddenLlm);
    for (size_t index = 0; index < x.size(); ++index) x[index] = 0.01f * static_cast<float>((index % 17) - 8);

    // CPU reference for the same seven products
    std::vector<float> cpu_out(kHiddenLlm, 0.0f), q(projection.size() / 2, 0.0f), buffer(kProjection);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        linear(x.data(), q_w.data(), kHiddenLlm, kProjection, buffer.data());
        linear(x.data(), k_w.data(), kHiddenLlm, kProjection, buffer.data());
        linear(x.data(), v_w.data(), kHiddenLlm, kProjection, buffer.data());
        linear(x.data(), f_w.data(), kHiddenLlm, kProjection, buffer.data());
        linear(x.data(), g_w.data(), kHiddenLlm, kProjection, buffer.data());
        std::vector<float> b_out(kHeads);
        linear(x.data(), b_w.data(), kHiddenLlm, kHeads, b_out.data());
        std::vector<float> o_out(kHiddenLlm);
        linear(buffer.data(), o_w.data(), kProjection, kHiddenLlm, o_out.data());
        for (size_t index = 0; index < kHiddenLlm; ++index) cpu_out[index] = o_out[index];
    }
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count() /
        std::max(iterations, 1);

    double timings[6] = {0, 0, 0, 0, 0, 0};
    const int status = ming_gpu_layer_benchmark(q_w.data(), k_w.data(), v_w.data(), f_w.data(), g_w.data(),
        b_w.data(), o_w.data(), x.data(), iterations, timings);
    if (status != 0) {
        std::printf("  gpu benchmark   : cublas init failed\n");
        return;
    }
    std::printf("  gpu benchmark   : %d iterations, one KDA layer at the decode shape\n", iterations);
    std::printf("    weight upload : %8.2f ms (one-time, %.0f MiB VRAM)\n", timings[0], timings[4]);
    std::printf("    per call xfer : %8.3f ms\n", timings[1]);
    std::printf("    per call gemm : %8.3f ms\n", timings[2]);
    std::printf("    per call total: %8.3f ms   (cpu same products: %.3f ms)\n", timings[3], cpu_ms);
    {
        double weight_sum = 0.0, weight_sq = 0.0, out_sq = 0.0;
        for (float value : q_w) { weight_sum += value; weight_sq += static_cast<double>(value) * value; }
        for (float value : cpu_out) { out_sq += static_cast<double>(value) * value; }
        std::printf("    diagnostics   : |q_w| rms=%.6f mean=%.6f |cpu_out| rms=%.6f |gpu_out| rms=%.6f\n",
            std::sqrt(weight_sq / q_w.size()), weight_sum / q_w.size(),
            std::sqrt(out_sq / cpu_out.size()),
            std::sqrt(static_cast<double>(timings[5]) * timings[5] / cpu_out.size()));
    }
    std::printf("    gpu checksum  : %12.4f   cpu checksum: %12.4f\n", timings[5],
        std::accumulate(cpu_out.begin(), cpu_out.end(), 0.0));
#endif
}

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

    // Seeded mode: start from a captured hidden state instead of the embedding, so a
    // later layer can be judged independently of error inherited from earlier layers.
    size_t first_layer = 0;
    size_t limit = layers;  // all 24: 18 KDA + 6 MLA
    if (const char * seed_path = std::getenv("MING3_TINY_SEED_STATE");
        seed_path != nullptr && *seed_path != '\0') {
        std::ifstream in(seed_path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("cannot open seed state: ") + seed_path);
        in.read(reinterpret_cast<char *>(state.data()), static_cast<std::streamsize>(state.size() * sizeof(float)));
        const char * layer_env = std::getenv("MING3_TINY_SEED_LAYER");
        first_layer = layer_env != nullptr ? static_cast<size_t>(std::atoi(layer_env)) : 3;
        if (first_layer < limit) limit = std::min(limit, first_layer + 1);  // one layer from a seed
        std::printf("  stage B seed    : %s -> starting at layer %zu\n", seed_path, first_layer);
    }

    for (size_t layer = first_layer; layer < limit; ++layer) {
        std::vector<float> after_attention, attention_only;
        KdaTrace trace;
        if (const char * trace_env = std::getenv("MING3_TINY_TRACE_LAYER");
            trace_env != nullptr && *trace_env != '\0' && static_cast<size_t>(std::atoi(trace_env)) == layer) {
            trace.enabled = true;
        }
        if (layer % 4 == 3) {
            run_mla(assets, layer, state, tokens, after_attention, attention_only);
        } else {
            run_kda(assets, layer, state, tokens, after_attention, attention_only, &trace);
            if (trace.enabled) {
                const std::string tag = "trace_" + std::to_string(layer);
                const Shape flat{1, static_cast<int64_t>(tokens), static_cast<int64_t>(kProjection)};
                writer.add(tag + "_q_conv", trace.q_conv, flat);
                writer.add(tag + "_k_conv", trace.k_conv, flat);
                writer.add(tag + "_v_conv", trace.v_conv, flat);
                writer.add(tag + "_q_norm", trace.q_norm, flat);
                writer.add(tag + "_k_norm", trace.k_norm, flat);
                writer.add(tag + "_gate", trace.gate, flat);
                writer.add(tag + "_beta", trace.beta, {1, static_cast<int64_t>(tokens), static_cast<int64_t>(kHeads)});
                writer.add(tag + "_heads", trace.heads, flat);
                writer.add(tag + "_attn", trace.attn,
                    {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
            }
        }
        writer.add("attn_out_layer_" + std::to_string(layer), attention_only,
            {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
        std::vector<float> after_block;
        if (layer < static_cast<size_t>(assets.config.first_k_dense_replace)) {
            run_dense_mlp(assets, layer, after_attention, tokens, after_block);
        } else {
            Routing routing;
            std::vector<float> pre_router, router_scores;
            run_moe(assets, layer, after_attention, tokens, after_block, routing, pre_router, router_scores);
            const std::string tag = "routing_layer_" + std::to_string(layer);
            writer.add_bytes(tag + "_ids", routing.ids.data(), routing.ids.size() * sizeof(int32_t),
                {static_cast<int64_t>(tokens), static_cast<int64_t>(kTopK)}, "int32");
            writer.add(tag + "_weights", routing.weights,
                {static_cast<int64_t>(tokens), static_cast<int64_t>(kTopK)});
            writer.add("pre_router_layer_" + std::to_string(layer), pre_router,
                {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
            writer.add("router_scores_layer_" + std::to_string(layer), router_scores,
                {static_cast<int64_t>(tokens), static_cast<int64_t>(assets.config.experts)});
        }
        state = std::move(after_block);
        writer.add("layer_" + std::to_string(layer), state,
            {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});
    }
    // final norm and stop head, so the whole thinker can be judged end to end
    const auto final_norm_w = load_float(*assets.weights, "model.norm.weight", {static_cast<int64_t>(hidden)});
    std::vector<float> final_norm;
    rms_norm(state.data(), final_norm_w.data(), tokens, hidden, final_norm);
    writer.add("final_norm", final_norm, {1, static_cast<int64_t>(tokens), static_cast<int64_t>(hidden)});

    const auto stop_w = load_float(*assets.weights, "stop_head.weight", {2, static_cast<int64_t>(hidden)});
    const auto stop_b = load_float(*assets.weights, "stop_head.bias", {2});
    std::vector<float> stop_logits(2, 0.0f);
    linear(final_norm.data() + (tokens - 1) * hidden, stop_w.data(), hidden, 2, stop_logits.data());
    for (size_t index = 0; index < stop_logits.size(); ++index) stop_logits[index] += stop_b[index];
    writer.add("stop_logits", stop_logits, {1, 2});
    writer.manifest(prompt_ids);
}

}  // namespace engine::models::ming3_tts_tiny
