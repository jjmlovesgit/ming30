#include "engine/models/ming3_tts_tiny/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/ming3_tts_tiny/prompt.h"
#include "engine/models/ming3_tts_tiny/thinker.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
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

        if (const char * dump_dir = std::getenv("MING3_TINY_DUMP_DIR"); dump_dir != nullptr && *dump_dir != '\0') {
            dump_thinker_probe(*assets, prompt.ids, std::filesystem::path{dump_dir});
            std::printf("  stage B dump    : %s (embedding + layer_0)\n", dump_dir);
        }
    } catch (const std::exception & error) {
        std::printf("  prompt stage A  : FAILED: %s\n", error.what());
    }
    std::printf("  inference       : not implemented (milestone 1: structural verification only)\n");
    std::fflush(stdout);
    return assets;
}

}  // namespace engine::models::ming3_tts_tiny
