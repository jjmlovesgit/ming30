from pathlib import Path

h = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\loop.h")
text = h.read_text(encoding="utf-8")
if "dump_podcast_prompt_probe" not in text:
    text = text.replace(
        "void dump_loop_probe(",
        "// Assemble the podcast prompt embedding matrix: word embeddings for the podcast ids,\n"
        "// per-latent Aggregator output in the placeholder rows, spk_head(zeros) in the speaker row.\n"
        "// Reference latents are supplied from a file (test bridge until the encoder is ported).\n"
        "void dump_podcast_prompt_probe(const MingTinyAssets & assets, const std::string & script,\n"
        "    const std::string & reference_text, const std::vector<float> & reference_latents,\n"
        "    size_t latent_tokens, const std::filesystem::path & directory);\n\n"
        "void dump_loop_probe(",
    )
    h.write_text(text, encoding="utf-8")
    print("loop.h updated")

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\loop.cpp")
text = c.read_text(encoding="utf-8")
if '#include "engine/models/ming3_tts_tiny/prompt.h"' not in text:
    text = text.replace('#include "engine/models/ming3_tts_tiny/flow.h"',
                        '#include "engine/models/ming3_tts_tiny/flow.h"\n'
                        '#include "engine/models/ming3_tts_tiny/prompt.h"')

anchor = "void dump_loop_probe(const MingTinyAssets & assets"
impl = '''void dump_podcast_prompt_probe(const MingTinyAssets & assets, const std::string & script,
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
    manifest << "{\\n  \\"tokens\\": " << tokens << ",\\n  \\"prompt_start\\": " << prompt_start
             << ",\\n  \\"speaker_position\\": " << speaker_position << ",\\n  \\"latent_tokens\\": "
             << latent_tokens << ",\\n  \\"ids\\": [";
    for (size_t index = 0; index < ids.size(); ++index) manifest << (index ? "," : "") << ids[index];
    manifest << "]\\n}\\n";
    std::printf("  podcast assemble: %zu ids (placeholders %zu..%zu), speaker row %zu\\n", tokens, prompt_start,
        tokens - 1, speaker_position);
}

void dump_loop_probe(const MingTinyAssets & assets'''
assert anchor in text
text = text.replace(anchor, impl, 1)
c.write_text(text, encoding="utf-8")
print("podcast assembly added to loop.cpp")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
anchor2 = '''        if (const char * res_env = std::getenv("MING3_TINY_GPU_RESIDENCY"); res_env != nullptr && *res_env != '\\0') {
            run_gpu_residency_probe(*assets);
        }'''
gate = anchor2 + '''

        // Podcast prompt assembly: reference latents supplied from file (test bridge).
        if (const char * latent_file = std::getenv("MING3_TINY_PODCAST_LATENTS_FILE");
            latent_file != nullptr && *latent_file != '\\0') {
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
        }'''
assert anchor2 in src
a.write_text(src.replace(anchor2, gate), encoding="utf-8")
print("assets.cpp wires podcast assembly")
