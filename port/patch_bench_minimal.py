from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

start = text.index("void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {")
end = text.index("void dump_thinker_probe(const MingTinyAssets & assets")
replacement = '''void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {
    constexpr size_t kHiddenLlm = 1536;
    constexpr size_t kProjectionLlm = 2048;
#if !defined(ENGINE_HAS_MING_GPU_BENCH)
    std::printf("  gpu gemm        : not built with CUDA\\n");
    (void) assets;
    (void) iterations;
#else
    const auto & source = *assets.weights;
    const auto q_w = load_float(source, "model.layers.0.attention.q_proj.weight",
        {static_cast<int64_t>(kProjectionLlm), static_cast<int64_t>(kHiddenLlm)});

    // known, finite, non-symmetric input so a transposed or mis-strided GEMM cannot pass
    std::vector<float> x(kHiddenLlm);
    for (size_t index = 0; index < x.size(); ++index) {
        x[index] = 0.5f * std::sin(0.01f * static_cast<float>(index)) + 0.1f * static_cast<float>(index % 7) - 0.3f;
    }

    std::vector<float> reference(kProjectionLlm);
    linear(x.data(), q_w.data(), kHiddenLlm, kProjectionLlm, reference.data());

    std::vector<float> gpu(kProjectionLlm, 0.0f);
    double timings[4] = {0, 0, 0, 0};
    const int status = ming_gpu_gemm(q_w.data(), x.data(), static_cast<int>(kProjectionLlm),
        static_cast<int>(kHiddenLlm), iterations, gpu.data(), timings);
    if (status != 0) {
        std::printf("  gpu gemm        : cuBLAS initialisation failed\\n");
        return;
    }

    double max_abs = 0.0, sum_sq = 0.0, reference_sq = 0.0;
    size_t mismatched = 0;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double difference = static_cast<double>(gpu[index]) - reference[index];
        max_abs = std::max(max_abs, std::abs(difference));
        sum_sq += difference * difference;
        reference_sq += static_cast<double>(reference[index]) * reference[index];
        if (std::abs(difference) > 1e-4 * std::max(1.0, std::abs(reference[index]))) ++mismatched;
    }
    const double relative = std::sqrt(sum_sq / (reference_sq > 0.0 ? reference_sq : 1.0));

    std::printf("  gpu gemm        : one projection %zux%zu, %d iterations\\n", kProjectionLlm, kHiddenLlm,
        iterations);
    std::printf("    weight upload : %8.3f ms (%.0f MiB VRAM for this matrix)\\n", timings[0], timings[3]);
    std::printf("    xfer per call : %8.3f ms\\n", timings[1]);
    std::printf("    gemm per call : %8.3f ms\\n", timings[2]);
    std::printf("    verification  : max_abs=%.3e rel_rmse=%.3e mismatched=%zu/%zu  %s\\n", max_abs, relative,
        mismatched, reference.size(), (mismatched == 0 && relative < 1e-5) ? "PASS" : "FAIL");
    std::printf("    values        : cpu[0..3]=%.6f %.6f %.6f %.6f | gpu[0..3]=%.6f %.6f %.6f %.6f\\n",
        reference[0], reference[1], reference[2], reference[3], gpu[0], gpu[1], gpu[2], gpu[3]);
#endif
}

'''
text = text[:start] + replacement + text[end:]
p.write_text(text, encoding="utf-8")
print("host harness reduced to one projection with per-value verification")
