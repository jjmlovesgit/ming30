from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

start = text.index("void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {")
end = text.index("void dump_thinker_probe(const MingTinyAssets & assets")
replacement = '''void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {
    constexpr size_t kHiddenLlm = 1536;
    constexpr size_t kProjectionLlm = 2048;
    constexpr size_t kHeadsLlm = 16;
#if !defined(ENGINE_HAS_MING_GPU_BENCH)
    std::printf("  gpu projections : not built with CUDA\\n");
    (void) assets;
    (void) iterations;
#else
    const auto & source = *assets.weights;
    const std::string prefix = "model.layers.0.attention.";
    auto weight = [&](const char * name, size_t out, size_t in) {
        return load_float(source, prefix + name,
            {static_cast<int64_t>(out), static_cast<int64_t>(in)});
    };

    std::vector<float> x(kHiddenLlm);
    for (size_t index = 0; index < x.size(); ++index) {
        x[index] = 0.5f * std::sin(0.01f * static_cast<float>(index)) + 0.1f * static_cast<float>(index % 7) - 0.3f;
    }

    struct Case {
        const char * name;
        const char * tensor;
        size_t out;
        size_t in;
        const std::vector<float> * chain_input;  // null = use x
    };
    std::vector<float> chain(0);
    const Case cases[7] = {
        {"q_proj", "q_proj.weight", kProjectionLlm, kHiddenLlm, nullptr},
        {"k_proj", "k_proj.weight", kProjectionLlm, kHiddenLlm, nullptr},
        {"v_proj", "v_proj.weight", kProjectionLlm, kHiddenLlm, nullptr},
        {"f_proj", "f_proj.weight", kProjectionLlm, kHiddenLlm, nullptr},
        {"g_proj", "g_proj.weight", kProjectionLlm, kHiddenLlm, nullptr},
        {"b_proj", "b_proj.weight", kHeadsLlm, kHiddenLlm, nullptr},
        {"o_proj", "o_proj.weight", kHiddenLlm, kProjectionLlm, nullptr},
    };

    double gpu_gemm_total = 0.0, gpu_xfer_total = 0.0;
    std::vector<float> cpu_chain(kProjectionLlm, 0.0f);
    const auto cpu_start = std::chrono::steady_clock::now();
    bool all_pass = true;

    for (size_t index = 0; index < 7; ++index) {
        const Case & item = cases[index];
        const auto matrix = weight(item.tensor, item.out, item.in);
        const std::vector<float> & input = (index == 6) ? cpu_chain : x;   // o_proj consumes q

        std::vector<float> reference(item.out, 0.0f);
        linear(input.data(), matrix.data(), item.in, item.out, reference.data());

        std::vector<float> gpu(item.out, 0.0f);
        double timings[4] = {0, 0, 0, 0};
        if (ming_gpu_gemm(matrix.data(), input.data(), static_cast<int>(item.out), static_cast<int>(item.in),
                iterations, gpu.data(), timings) != 0) {
            std::printf("    %-8s : cuBLAS failure\\n", item.name);
            all_pass = false;
            continue;
        }
        gpu_gemm_total += timings[2];
        gpu_xfer_total += timings[1];

        double max_abs = 0.0, sum_sq = 0.0, reference_sq = 0.0;
        size_t mismatched = 0;
        for (size_t element = 0; element < reference.size(); ++element) {
            const double difference = static_cast<double>(gpu[element]) - reference[element];
            max_abs = std::max(max_abs, std::abs(difference));
            sum_sq += difference * difference;
            reference_sq += static_cast<double>(reference[element]) * reference[element];
            const double tolerance = 1e-4 * std::max(1.0, static_cast<double>(std::abs(reference[element])));
            if (std::abs(difference) > tolerance) ++mismatched;
        }
        const double relative = std::sqrt(sum_sq / (reference_sq > 0.0 ? reference_sq : 1.0));
        const bool pass = mismatched == 0 && relative < 1e-5;
        all_pass = all_pass && pass;
        std::printf("    %-8s %4zux%-4zu max_abs=%.2e rel=%.2e %zu/%zu %s   gemm %.3f ms  xfer %.3f ms\\n",
            item.name, item.out, item.in, max_abs, relative, mismatched, reference.size(),
            pass ? "PASS" : "FAIL", timings[2], timings[1]);
        if (index == 0) cpu_chain = reference;   // o_proj consumes q_proj's CPU result
    }
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count() /
        static_cast<double>(iterations > 0 ? iterations : 1);

    std::printf("  gpu projections : %d iterations, all seven per iteration\\n", iterations);
    std::printf("    gpu gemm total: %.3f ms   xfer total: %.3f ms   sum: %.3f ms\\n", gpu_gemm_total,
        gpu_xfer_total, gpu_gemm_total + gpu_xfer_total);
    std::printf("    cpu same work : %.3f ms (includes weight loads; see note)\\n", cpu_ms);
    std::printf("    verdict       : %s\\n", all_pass ? "all seven projections PASS" : "FAILURES present");
#endif
}

'''
text = text[:start] + replacement + text[end:]
p.write_text(text, encoding="utf-8")
print("seven-projection chained verification added")
