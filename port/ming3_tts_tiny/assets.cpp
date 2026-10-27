#include "engine/models/ming3_tts_tiny/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/ming3_tts_tiny/prompt.h"
#include "engine/models/ming3_tts_tiny/thinker.h"
#include "engine/models/ming3_tts_tiny/flow.h"
#include "engine/models/ming3_tts_tiny/vae.h"
#include "engine/models/ming3_tts_tiny/loop.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace engine::models::ming3_tts_tiny {
namespace {

MingTinyConfig parse_config(const io::json::Value & value) {
    MingTinyConfig config;
    const auto & llm = value.require("llm_config");
    config.layers = llm.require("num_hidden_layers").as_i64();
    config.experts = llm.require("num_experts").as_i64();
    config.experts_per_token = llm.require("num_experts_per_tok").as_i64();
    config.hidden_size = llm.require("hidden_size").as_i64();
    config.vocabulary_size = llm.require("vocab_size").as_i64();
    config.first_k_dense_replace = llm.require("first_k_dense_replace").as_i64();
    const auto & flow = value.require("flow_config");
    config.flow_depth = flow.require("depth").as_i64();
    config.flow_hidden_size = flow.require("hidden_size").as_i64();
    config.sample_rate = value.require("tts_config").require("sample_rate").as_i64();
    return config;
}

const assets::TensorMetadata * first_expert_bank(const std::vector<assets::TensorMetadata> & tensors) {
    for (const auto & tensor : tensors) {
        if (tensor.name.find(".mlp.experts.") != std::string::npos && tensor.name.size() > 7 &&
            tensor.name.compare(tensor.name.size() - 7, 7, ".weight") == 0) {
            return &tensor;
        }
    }
    return nullptr;
}

std::string shape_text(const std::vector<int64_t> & shape) {
    std::string text = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index) text += ",";
        text += std::to_string(shape[index]);
    }
    return text + "]";
}

}  // namespace

std::shared_ptr<const MingTinyAssets> load_ming3_tts_tiny_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<MingTinyAssets>();
    assets->resources = model_spec::load_resource_bundle_for_family(path, "ming3_tts_tiny");
    assets->config = parse_config(assets->resources.parse_json("config"));

    // Resolving these proves the embedded sidecars were materialized from the package.
    const size_t tokenizer_bytes = assets->resources.read_text("tokenizer_json").size();
    const size_t chat_template_bytes = assets->resources.read_text("chat_template").size();
    const bool has_generation_config = assets->resources.has_file("generation_config");

    assets->weights = assets->resources.open_tensor_source("weights");
    const auto tensors = assets->weights->tensors();
    const auto * bank = first_expert_bank(tensors);

    std::printf("\n[ming3_tts_tiny] structural package report\n");
    std::printf("  family          : ming3_tts_tiny\n");
    std::printf("  spec            : embedded, contract resolved for family ming3_tts_tiny\n");
    std::printf("  tokenizer_json  : %zu bytes (resolved from package)\n", tokenizer_bytes);
    std::printf("  chat_template   : %zu bytes\n", chat_template_bytes);
    std::printf("  generation_conf : %s\n", has_generation_config ? "resolved" : "missing");
    std::printf("  config          : layers=%lld experts=%lld top=%lld hidden=%lld vocab=%lld "
                "flow_depth=%lld sample_rate=%lld\n",
        static_cast<long long>(assets->config.layers), static_cast<long long>(assets->config.experts),
        static_cast<long long>(assets->config.experts_per_token), static_cast<long long>(assets->config.hidden_size),
        static_cast<long long>(assets->config.vocabulary_size), static_cast<long long>(assets->config.flow_depth),
        static_cast<long long>(assets->config.sample_rate));
    std::printf("  tensors         : %zu\n", tensors.size());
    std::printf("  lm_head.weight  : %s\n", assets->weights->has_tensor("lm_head.weight") ? "present" : "MISSING");
    if (bank != nullptr) {
        std::printf("  expert bank     : %s dtype=%s shape=%s\n", bank->name.c_str(), bank->dtype.c_str(),
            shape_text(bank->shape).c_str());
    } else {
        std::printf("  expert bank     : MISSING\n");
    }
    // Stage A: prompt ids. Stage B dump (opt-in): the embedding for the same prompt.
    // The embedding is the first capture point because it needs no model maths -- a
    // mismatch there is tokenizer or weight loading, not attention.
    try {
        tokenizers::LlamaBpeTokenizerSpec spec{{}, {}, assets->resources.require_file("tokenizer_config"),
            assets->resources.require_file("tokenizer_json"),
            tokenizers::LlamaBpePreTokenizer::BailingMoe};
        MingTinyPromptBuilder prompts(std::move(spec), MingTinySpecialTokens{});
        const std::string probe_text = "The garden will remain open until sunset.";
        const auto segments = prompts.build_segments(probe_text);
        std::printf("  prompt stage A  : text=%s\n", probe_text.c_str());
        size_t total = 0;
        for (const auto & [name, ids] : segments) {
            total += ids.size();
            std::printf("    %-16s ids=%zu\n", name.c_str(), ids.size());
        }
        const auto prompt = prompts.build(probe_text);
        std::printf("    total              =%zu (segments %zu)\n", prompt.ids.size(), total);
        std::printf("    speaker_position   =%zu\n", prompt.speaker_embedding_position);
        std::printf("    ids=");
        for (size_t index = 0; index < prompt.ids.size(); ++index) {
            std::printf("%s%d", index ? "," : "", prompt.ids[index]);
        }
        std::printf("\n");

        // Podcast prompt (reference transcript + trailing reference-latent placeholders).
        if (const char * podcast = std::getenv("MING3_TINY_PODCAST_SCRIPT"); podcast != nullptr && *podcast != '\0') {
            const char * reference_env = std::getenv("MING3_TINY_PODCAST_REFERENCE");
            const std::string reference = reference_env != nullptr ? reference_env : "";
            size_t latent_tokens = 100;
            if (const char * latent_env = std::getenv("MING3_TINY_PODCAST_LATENTS"); latent_env != nullptr) {
                latent_tokens = static_cast<size_t>(std::atoi(latent_env));
            }
            const auto segments = prompts.build_podcast_segments(podcast, reference, latent_tokens);
            std::printf("  podcast prompt  : script=%zu chars reference=%zu chars latents=%zu\n",
                std::strlen(podcast), reference.size(), latent_tokens);
            std::vector<int32_t> ids;
            for (const auto & [name, segment] : segments) {
                std::printf("    %-16s ids=%zu\n", name.c_str(), segment.size());
                ids.insert(ids.end(), segment.begin(), segment.end());
            }
            std::printf("    total=%zu\n    ids=", ids.size());
            for (size_t index = 0; index < ids.size() && index < 200; ++index) {
                std::printf("%s%d", index ? "," : "", ids[index]);
            }
            std::printf("\n");
        }

        if (const char * dump_dir = std::getenv("MING3_TINY_DUMP_DIR"); dump_dir != nullptr && *dump_dir != '\0') {
            dump_thinker_probe(*assets, prompt.ids, std::filesystem::path{dump_dir});
            std::printf("  stage B dump    : %s (embedding + layer_0)\n", dump_dir);
        }

        // Stage C: one flow step from a supplied condition and initial noise, so the
        // same inputs can be fed to the Python reference.
        if (const char * cond_path = std::getenv("MING3_TINY_FLOW_COND"); cond_path != nullptr && *cond_path != '\0') {
            const auto read_floats = [](const char * path, size_t count) {
                std::vector<float> values(count);
                std::ifstream in(path, std::ios::binary);
                if (!in) throw std::runtime_error(std::string("cannot open ") + path);
                in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
                return values;
            };
            FlowRequest request;
            request.cond = read_floats(cond_path, 1536);
            const char * noise_path = std::getenv("MING3_TINY_FLOW_NOISE");
            if (noise_path == nullptr) throw std::runtime_error("MING3_TINY_FLOW_NOISE is required with MING3_TINY_FLOW_COND");
            request.noise = read_floats(noise_path, 64);
            if (const char * cfg_env = std::getenv("MING3_TINY_FLOW_CFG"); cfg_env != nullptr) {
                request.cfg = static_cast<float>(std::atof(cfg_env));
            }
            if (const char * steps_env = std::getenv("MING3_TINY_FLOW_STEPS"); steps_env != nullptr) {
                request.steps = std::atoi(steps_env);
            }
            const char * flow_dir = std::getenv("MING3_TINY_FLOW_DUMP_DIR");
            dump_flow_probe(*assets, request, std::filesystem::path{flow_dir != nullptr ? flow_dir : "flow-dump"});
        }

        // Stage D: decode latent patches through the VAE decoder. The frame count is
        // inferred from the file size (64 floats per patch).
        if (const char * latent_path = std::getenv("MING3_TINY_VAE_LATENT"); latent_path != nullptr && *latent_path != '\0') {
            std::ifstream in(latent_path, std::ios::binary | std::ios::ate);
            if (!in) throw std::runtime_error(std::string("cannot open ") + latent_path);
            const auto bytes = static_cast<size_t>(in.tellg());
            in.seekg(0);
            if (bytes % (64 * sizeof(float)) != 0) throw std::runtime_error("latent file is not a whole number of 64-float patches");
            std::vector<float> latent(bytes / sizeof(float));
            in.read(reinterpret_cast<char *>(latent.data()), static_cast<std::streamsize>(bytes));
            const size_t frames = latent.size() / 64;
            const char * vae_dir = std::getenv("MING3_TINY_VAE_DUMP_DIR");
            dump_vae_probe(*assets, latent, frames, std::filesystem::path{vae_dir != nullptr ? vae_dir : "vae-dump"});
        }

        // Stage E: the generation loop, replaying a Python noise stream.
        if (const char * noise_path = std::getenv("MING3_TINY_LOOP_NOISE"); noise_path != nullptr && *noise_path != '\0') {
            std::ifstream in(noise_path, std::ios::binary | std::ios::ate);
            if (!in) throw std::runtime_error(std::string("cannot open ") + noise_path);
            const auto bytes = static_cast<size_t>(in.tellg());
            in.seekg(0);
            if (bytes % (64 * sizeof(float)) != 0) throw std::runtime_error("loop noise file is not a whole number of 64-float patches");
            std::vector<float> noises(bytes / sizeof(float));
            in.read(reinterpret_cast<char *>(noises.data()), static_cast<std::streamsize>(bytes));
            float cfg = 2.0f;
            if (const char * cfg_env = std::getenv("MING3_TINY_LOOP_CFG"); cfg_env != nullptr) cfg = static_cast<float>(std::atof(cfg_env));
            int flow_steps = 10;
            if (const char * steps_env = std::getenv("MING3_TINY_LOOP_FLOW_STEPS"); steps_env != nullptr) flow_steps = std::atoi(steps_env);
            const char * loop_dir = std::getenv("MING3_TINY_LOOP_DUMP_DIR");
            dump_loop_probe(*assets, prompt.ids, noises, noises.size() / 64, cfg, flow_steps,
                std::filesystem::path{loop_dir != nullptr ? loop_dir : "loop-dump"});
        }

        // Seeded single patch: enter with an exact embedding history and noise, so a
        // split point can be judged independently of earlier drift.
        if (const char * seeds = std::getenv("MING3_TINY_SEED_EMBEDS"); seeds != nullptr && *seeds != '\0') {
            const auto read_all = [](const char * path, size_t stride, std::vector<float> & values) {
                std::ifstream in(path, std::ios::binary | std::ios::ate);
                if (!in) throw std::runtime_error(std::string("cannot open ") + path);
                const auto bytes = static_cast<size_t>(in.tellg());
                in.seekg(0);
                if (bytes % (stride * sizeof(float)) != 0) throw std::runtime_error("seed file has the wrong stride");
                values.resize(bytes / sizeof(float));
                in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes));
            };
            std::vector<float> embeds;
            read_all(seeds, 1536, embeds);
            std::vector<float> noise;
            const char * noise_path = std::getenv("MING3_TINY_SEED_NOISE");
            if (noise_path == nullptr) throw std::runtime_error("MING3_TINY_SEED_NOISE is required with MING3_TINY_SEED_EMBEDS");
            read_all(noise_path, 64, noise);
            std::vector<float> history(64, 0.0f);
            if (const char * history_path = std::getenv("MING3_TINY_SEED_HISTORY"); history_path != nullptr && *history_path != '\0') {
                read_all(history_path, 64, history);
            }
            float cfg = 2.0f;
            if (const char * cfg_env = std::getenv("MING3_TINY_LOOP_CFG"); cfg_env != nullptr) cfg = static_cast<float>(std::atof(cfg_env));
            int flow_steps = 10;
            if (const char * steps_env = std::getenv("MING3_TINY_LOOP_FLOW_STEPS"); steps_env != nullptr) flow_steps = std::atoi(steps_env);
            std::vector<float> latent, feedback;
            run_loop_patch_seeded(*assets, embeds, embeds.size() / 1536, noise, history, cfg, flow_steps, latent, feedback);
            const char * out_dir = std::getenv("MING3_TINY_SEED_DUMP_DIR");
            const std::filesystem::path out{out_dir != nullptr ? out_dir : "seed-dump"};
            std::filesystem::create_directories(out);
            std::ofstream(out / "latent.bin", std::ios::binary)
                .write(reinterpret_cast<const char *>(latent.data()), static_cast<std::streamsize>(latent.size() * sizeof(float)));
            std::ofstream(out / "feedback.bin", std::ios::binary)
                .write(reinterpret_cast<const char *>(feedback.data()), static_cast<std::streamsize>(feedback.size() * sizeof(float)));
            std::printf("  seeded patch    : tokens=%zu latent_rms=%.6f\n", embeds.size() / 1536,
                std::sqrt(std::inner_product(latent.begin(), latent.end(), latent.begin(), 0.0) / 64.0));
        }

        // Cache verification: full recompute vs prefill + one-token steps.
        if (const char * cache_file = std::getenv("MING3_TINY_CACHE_CHECK"); cache_file != nullptr && *cache_file != '\0') {
            std::ifstream in(cache_file, std::ios::binary | std::ios::ate);
            if (!in) throw std::runtime_error(std::string("cannot open ") + cache_file);
            const auto bytes = static_cast<size_t>(in.tellg());
            in.seekg(0);
            std::vector<float> embeds(bytes / sizeof(float));
            in.read(reinterpret_cast<char *>(embeds.data()), static_cast<std::streamsize>(bytes));
            size_t prefill = 41;
            if (const char * p = std::getenv("MING3_TINY_CACHE_PREFILL"); p != nullptr) prefill = static_cast<size_t>(std::atoi(p));
            run_cache_check(*assets, embeds, embeds.size() / 1536, prefill);
        }

        if (const char * gpu_env = std::getenv("MING3_TINY_GPU_BENCH"); gpu_env != nullptr && *gpu_env != '\0') {
            run_gpu_layer_benchmark(*assets, std::atoi(gpu_env));
        }

        if (const char * moe_env = std::getenv("MING3_TINY_GPU_MOE"); moe_env != nullptr && *moe_env != '\0') {
            run_gpu_moe_benchmark(*assets, std::atoi(moe_env));
        }

        if (const char * res_env = std::getenv("MING3_TINY_GPU_RESIDENCY"); res_env != nullptr && *res_env != '\0') {
            run_gpu_residency_probe(*assets);
        }

        // Podcast prompt assembly: reference latents supplied from file (test bridge).
        if (const char * latent_file = std::getenv("MING3_TINY_PODCAST_LATENTS_FILE");
            latent_file != nullptr && *latent_file != '\0') {
            std::ifstream in(latent_file, std::ios::binary | std::ios::ate);
            if (!in) throw std::runtime_error(std::string("cannot open ") + latent_file);
            const auto bytes = static_cast<size_t>(in.tellg());
            in.seekg(0);
            std::vector<float> latents(bytes / sizeof(float));
            in.read(reinterpret_cast<char *>(latents.data()), static_cast<std::streamsize>(bytes));
            const char * script_env = std::getenv("MING3_TINY_PODCAST_SCRIPT");
            const char * reference_env = std::getenv("MING3_TINY_PODCAST_REFERENCE");
            const char * out_env = std::getenv("MING3_TINY_PODCAST_ASSEMBLE_DIR");
            dump_podcast_prompt_probe(*assets, script_env != nullptr ? script_env : "",
                reference_env != nullptr ? reference_env : "", latents, latents.size() / 64,
                std::filesystem::path{out_env != nullptr ? out_env : "podcast-embeds"});
        }
    } catch (const std::exception & error) {
        std::printf("  prompt stage A  : FAILED: %s\n", error.what());
    }
    std::printf("  inference       : not implemented (milestone 1: structural verification only)\n");
    std::fflush(stdout);
    return assets;
}

}  // namespace engine::models::ming3_tts_tiny
