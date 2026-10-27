from pathlib import Path

h = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\prompt.h")
text = h.read_text(encoding="utf-8")
if "build_podcast" not in text:
    text = text.replace(
        "    [[nodiscard]] MingTinyPromptIds build(const std::string & text) const;",
        "    [[nodiscard]] MingTinyPromptIds build(const std::string & text) const;\n\n"
        "    // Podcast form: the reference transcript is inserted before the script and the\n"
        "    // reference latents occupy trailing <audioPatch> placeholders.\n"
        "    [[nodiscard]] std::vector<std::pair<std::string, std::vector<int32_t>>> build_podcast_segments(\n"
        "        const std::string & script, const std::string & reference_text, size_t latent_tokens) const;",
    )
    h.write_text(text, encoding="utf-8")
    print("prompt.h updated")

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\prompt.cpp")
text = c.read_text(encoding="utf-8")
anchor = "MingTinyPromptIds MingTinyPromptBuilder::build(const std::string & text) const {"
impl = '''std::vector<std::pair<std::string, std::vector<int32_t>>> MingTinyPromptBuilder::build_podcast_segments(
    const std::string & script, const std::string & reference_text, size_t latent_tokens) const {
    std::vector<std::pair<std::string, std::vector<int32_t>>> segments;
    segments.emplace_back("user_prefix", tokenizer_.encode(kUserPrefix, true));
    segments.emplace_back("speaker_block",
        std::vector<int32_t>{tokens_.speaker_start, tokens_.audio_patch, tokens_.speaker_end});
    segments.emplace_back("text_prefix", tokenizer_.encode(kTextPrefix, true));
    segments.emplace_back("reference_text", tokenizer_.encode(reference_text, true));
    segments.emplace_back("target_text", tokenizer_.encode(script, true));
    segments.emplace_back("assistant_prefix", tokenizer_.encode(kAssistantPrefix, true));
    segments.emplace_back("audio_start", std::vector<int32_t>{tokens_.audio_start});
    segments.emplace_back("prompt_latents", std::vector<int32_t>(latent_tokens, tokens_.audio_patch));
    return segments;
}

MingTinyPromptIds MingTinyPromptBuilder::build(const std::string & text) const {'''
assert anchor in text
text = text.replace(anchor, impl, 1)
c.write_text(text, encoding="utf-8")
print("podcast builder added")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
anchor2 = '''        if (const char * dump_dir = std::getenv("MING3_TINY_DUMP_DIR"); dump_dir != nullptr && *dump_dir != '\\0') {'''
podcast = '''        // Podcast prompt (reference transcript + trailing reference-latent placeholders).
        if (const char * podcast = std::getenv("MING3_TINY_PODCAST_SCRIPT"); podcast != nullptr && *podcast != '\\0') {
            const char * reference_env = std::getenv("MING3_TINY_PODCAST_REFERENCE");
            const std::string reference = reference_env != nullptr ? reference_env : "";
            size_t latent_tokens = 100;
            if (const char * latent_env = std::getenv("MING3_TINY_PODCAST_LATENTS"); latent_env != nullptr) {
                latent_tokens = static_cast<size_t>(std::atoi(latent_env));
            }
            const auto segments = prompts.build_podcast_segments(podcast, reference, latent_tokens);
            std::printf("  podcast prompt  : script=%zu chars reference=%zu chars latents=%zu\\n",
                std::strlen(podcast), reference.size(), latent_tokens);
            std::vector<int32_t> ids;
            for (const auto & [name, segment] : segments) {
                std::printf("    %-16s ids=%zu\\n", name.c_str(), segment.size());
                ids.insert(ids.end(), segment.begin(), segment.end());
            }
            std::printf("    total=%zu\\n    ids=", ids.size());
            for (size_t index = 0; index < ids.size() && index < 200; ++index) {
                std::printf("%s%d", index ? "," : "", ids[index]);
            }
            std::printf("\\n");
        }

''' + anchor2
assert anchor2 in src
a.write_text(src.replace(anchor2, podcast, 1), encoding="utf-8")
print("assets.cpp publishes podcast prompt ids")
