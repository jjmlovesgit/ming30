from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

old = """void run_all_layers(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
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
    }"""

new = """void run_all_layers(const MingTinyAssets & assets, const std::vector<float> & embeds, size_t tokens,
    ThinkerCache * cache, std::vector<float> & final_norm) {
    const bool profile = std::getenv("MING3_TINY_STEP_PROFILE") != nullptr;
    double kda_ms = 0.0, mla_ms = 0.0, moe_ms = 0.0, dense_ms = 0.0;
    const auto clock = [] { return std::chrono::steady_clock::now(); };
    const auto ms = [](auto from) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
    };

    std::vector<float> state = embeds;
    for (size_t layer = 0; layer < static_cast<size_t>(assets.config.layers); ++layer) {
        std::vector<float> after_attention, attention_only;
        const auto attention_start = clock();
        if (layer % 4 == 3) {
            MlaCache * mla = cache != nullptr ? &cache->mla[layer] : nullptr;
            run_mla(assets, layer, state, tokens, after_attention, attention_only, nullptr, mla);
        } else {
            KdaCache * kda = cache != nullptr ? &cache->kda[layer] : nullptr;
            run_kda(assets, layer, state, tokens, after_attention, attention_only, nullptr, kda);
        }
        if (profile) {
            const double spent = ms(attention_start);
            if (layer % 4 == 3) mla_ms += spent; else kda_ms += spent;
        }
        std::vector<float> after_block;
        const auto block_start = clock();
        if (layer < static_cast<size_t>(assets.config.first_k_dense_replace)) {
            run_dense_mlp(assets, layer, after_attention, tokens, after_block);
            if (profile) dense_ms += ms(block_start);
        } else {
            Routing routing;
            std::vector<float> pre_router, router_scores;
            run_moe(assets, layer, after_attention, tokens, after_block, routing, pre_router, router_scores);
            if (profile) moe_ms += ms(block_start);
        }
        state = std::move(after_block);
    }
    if (profile) {
        std::printf("    profile       : kda %.1f ms | mla %.1f ms | moe %.1f ms | dense %.1f ms | total %.1f ms\\n",
            kda_ms, mla_ms, moe_ms, dense_ms, kda_ms + mla_ms + moe_ms + dense_ms);
    }"""

assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("stage profiling added to the layer loop")
