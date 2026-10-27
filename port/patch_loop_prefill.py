from pathlib import Path

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\loop.cpp")
text = c.read_text(encoding="utf-8")

old = """void dump_loop_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::vector<float> & noises, size_t patches, float cfg, int flow_steps,
    const std::filesystem::path & directory) {
    if (noises.size() != patches * kLatent) throw std::runtime_error("loop probe: noise size mismatch");"""
new = """void dump_loop_probe(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::vector<float> & noises, size_t patches, float cfg, int flow_steps,
    const std::filesystem::path & directory) {
    dump_loop_probe_with_embeds(assets, prompt_ids, {}, noises, patches, cfg, flow_steps, directory);
}

void dump_loop_probe_with_embeds(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,
    const std::vector<float> & prefill_embeds, const std::vector<float> & noises, size_t patches, float cfg,
    int flow_steps, const std::filesystem::path & directory) {
    if (noises.size() != patches * kLatent) throw std::runtime_error("loop probe: noise size mismatch");"""
assert old in text
text = text.replace(old, new)

old = """    std::vector<float> embeds(prompt_ids.size() * kHiddenLlm);
    for (size_t row = 0; row < prompt_ids.size(); ++row) {
        std::memcpy(embeds.data() + row * kHiddenLlm,
            table.data() + static_cast<size_t>(prompt_ids[row]) * kHiddenLlm, kHiddenLlm * sizeof(float));
    }"""
new = """    std::vector<float> embeds;
    if (!prefill_embeds.empty()) {
        embeds = prefill_embeds;   // podcast prompt assembled by the caller
    } else {
        embeds.resize(prompt_ids.size() * kHiddenLlm);
        for (size_t row = 0; row < prompt_ids.size(); ++row) {
            std::memcpy(embeds.data() + row * kHiddenLlm,
                table.data() + static_cast<size_t>(prompt_ids[row]) * kHiddenLlm, kHiddenLlm * sizeof(float));
        }
    }"""
assert old in text
text = text.replace(old, new)

old = """    ThinkerCache cache;
    thinker_prefill_cached(assets, embeds, prompt_ids.size(), cache, final_norm);"""
new = """    ThinkerCache cache;
    const size_t prefill_rows = embeds.size() / kHiddenLlm;
    thinker_prefill_cached(assets, embeds, prefill_rows, cache, final_norm);"""
assert old in text
text = text.replace(old, new)

old = """    for (size_t patch = 0; patch < patches; ++patch) {
        const size_t tokens = prompt_ids.size() + patch;"""
new = """    for (size_t patch = 0; patch < patches; ++patch) {
        const size_t tokens = prefill_rows + patch;"""
assert old in text
text = text.replace(old, new)
c.write_text(text, encoding="utf-8")
print("loop accepts prefill embeddings")

h = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\loop.h")
text = h.read_text(encoding="utf-8")
if "dump_loop_probe_with_embeds" not in text:
    text = text.replace(
        "void dump_loop_probe(",
        "// Same loop, but prefilled with a caller-assembled prompt embedding matrix (podcast).\n"
        "void dump_loop_probe_with_embeds(const MingTinyAssets & assets, const std::vector<int32_t> & prompt_ids,\n"
        "    const std::vector<float> & prefill_embeds, const std::vector<float> & noises, size_t patches, float cfg,\n"
        "    int flow_steps, const std::filesystem::path & directory);\n\n"
        "void dump_loop_probe(",
    )
    h.write_text(text, encoding="utf-8")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
old = """            const char * loop_dir = std::getenv("MING3_TINY_LOOP_DUMP_DIR");
            dump_loop_probe(*assets, prompt.ids, noises, noises.size() / 64, cfg, flow_steps,
                std::filesystem::path{loop_dir != nullptr ? loop_dir : "loop-dump"});"""
new = """            const char * loop_dir = std::getenv("MING3_TINY_LOOP_DUMP_DIR");
            const std::filesystem::path destination{loop_dir != nullptr ? loop_dir : "loop-dump"};
            std::vector<float> prefill;
            if (const char * embeds_path = std::getenv("MING3_TINY_LOOP_PREEMBEDS");
                embeds_path != nullptr && *embeds_path != '\\0') {
                std::ifstream embeds_in(embeds_path, std::ios::binary | std::ios::ate);
                if (!embeds_in) throw std::runtime_error(std::string("cannot open ") + embeds_path);
                const auto embeds_bytes = static_cast<size_t>(embeds_in.tellg());
                embeds_in.seekg(0);
                if (embeds_bytes % (1536 * sizeof(float)) != 0) {
                    throw std::runtime_error("prefill embeddings are not a whole number of 1536-wide rows");
                }
                prefill.resize(embeds_bytes / sizeof(float));
                embeds_in.read(reinterpret_cast<char *>(prefill.data()), static_cast<std::streamsize>(embeds_bytes));
                std::printf("  loop prefill    : %zu rows from %s\\n", prefill.size() / 1536, embeds_path);
            }
            if (prefill.empty()) {
                dump_loop_probe(*assets, prompt.ids, noises, noises.size() / 64, cfg, flow_steps, destination);
            } else {
                dump_loop_probe_with_embeds(*assets, prompt.ids, prefill, noises, noises.size() / 64, cfg,
                    flow_steps, destination);
            }"""
assert old in src
a.write_text(src.replace(old, new), encoding="utf-8")
print("assets.cpp passes prefill embeddings")
