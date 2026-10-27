from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.h")
text = p.read_text(encoding="utf-8")
if "run_gpu_moe_benchmark" not in text:
    text = text.replace(
        "void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations);",
        "void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations);\n"
        "// GPU test of routed experts for one MoE layer, with fixed expert ids from the\n"
        "// reference run: per-expert outputs and the combined layer output.\n"
        "void run_gpu_moe_benchmark(const MingTinyAssets & assets, int iterations);",
    )
    p.write_text(text, encoding="utf-8")
    print("header updated")

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = c.read_text(encoding="utf-8")
text = text.replace('#include "engine/models/ming3_tts_tiny/gpu_bench.h"',
                    '#include "engine/models/ming3_tts_tiny/gpu_bench.h"\n'
                    '#include "engine/models/ming3_tts_tiny/gpu_moe.h"')

anchor = "void dump_thinker_probe(const MingTinyAssets & assets"
impl = '''void run_gpu_moe_benchmark(const MingTinyAssets & assets, int iterations) {
    constexpr size_t kHiddenLlm = 1536;
    constexpr size_t kInter = 512;
#if !defined(ENGINE_HAS_MING_GPU_BENCH)
    std::printf("  gpu moe         : not built with CUDA\\n");
    (void) assets;
    (void) iterations;
#else
    const auto & source = *assets.weights;
    constexpr size_t layer = 1;
    // expert ids fixed from the reference run (layer 1, first prompt token)
    const int experts[8] = {104, 80, 109, 126, 13, 3, 102, 9};
    const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.";
    const auto gate_bank = load_raw(source, prefix + "experts.gate_proj.weight",
        {128, static_cast<int64_t>(kInter), static_cast<int64_t>(kHiddenLlm)});
    const auto up_bank = load_raw(source, prefix + "experts.up_proj.weight",
        {128, static_cast<int64_t>(kInter), static_cast<int64_t>(kHiddenLlm)});
    const auto down_bank = load_raw(source, prefix + "experts.down_proj.weight",
        {128, static_cast<int64_t>(kHiddenLlm), static_cast<int64_t>(kInter)});

    std::vector<float> x(kHiddenLlm);
    for (size_t index = 0; index < x.size(); ++index) {
        x[index] = 0.4f * std::sin(0.013f * static_cast<float>(index)) + 0.05f * static_cast<float>(index % 5) - 0.1f;
    }

    const auto slice = [](const RawTensor & bank, size_t expert, size_t count) {
        std::vector<float> values(count);
        const auto * raw = reinterpret_cast<const uint16_t *>(bank.bytes.data()) + expert * count;
        for (size_t index = 0; index < count; ++index) values[index] = bf16_to_float(raw[index]);
        return values;
    };

    std::vector<float> combined_cpu(kHiddenLlm, 0.0f), combined_gpu(kHiddenLlm, 0.0f);
    double total_gemm = 0.0, total_xfer = 0.0;
    bool all_pass = true;

    for (size_t slot = 0; slot < 8; ++slot) {
        const size_t expert = static_cast<size_t>(experts[slot]);
        const auto gate_w = slice(gate_bank, expert, kInter * kHiddenLlm);
        const auto up_w = slice(up_bank, expert, kInter * kHiddenLlm);
        const auto down_w = slice(down_bank, expert, kHiddenLlm * kInter);

        std::vector<float> g(kInter), u(kInter), act(kInter), y_cpu(kHiddenLlm);
        linear(x.data(), gate_w.data(), kHiddenLlm, kInter, g.data());
        linear(x.data(), up_w.data(), kHiddenLlm, kInter, u.data());
        for (size_t index = 0; index < kInter; ++index) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-g[index]));
            act[index] = g[index] * sigmoid * u[index];
        }
        linear(act.data(), down_w.data(), kInter, kHiddenLlm, y_cpu.data());

        std::vector<float> y_gpu(kHiddenLlm, 0.0f);
        double timings[4] = {0, 0, 0, 0};
        if (ming_gpu_expert(gate_w.data(), up_w.data(), down_w.data(), x.data(), static_cast<int>(kHiddenLlm),
                static_cast<int>(kInter), iterations, y_gpu.data(), timings) != 0) {
            std::printf("    expert %3zu : cuBLAS failure\\n", expert);
            all_pass = false;
            continue;
        }
        total_gemm += timings[2];
        total_xfer += timings[1];

        double max_abs = 0.0, sum_sq = 0.0, reference_sq = 0.0;
        size_t mismatched = 0;
        for (size_t index = 0; index < kHiddenLlm; ++index) {
            const double difference = static_cast<double>(y_gpu[index]) - y_cpu[index];
            max_abs = std::max(max_abs, std::abs(difference));
            sum_sq += difference * difference;
            reference_sq += static_cast<double>(y_cpu[index]) * y_cpu[index];
            const double tolerance = 1e-4 * std::max(1.0, static_cast<double>(std::abs(y_cpu[index])));
            if (std::abs(difference) > tolerance) ++mismatched;
            combined_cpu[index] += 0.125f * y_cpu[index];
            combined_gpu[index] += 0.125f * y_gpu[index];
        }
        const double relative = std::sqrt(sum_sq / (reference_sq > 0.0 ? reference_sq : 1.0));
        const bool pass = mismatched == 0 && relative < 1e-5;
        all_pass = all_pass && pass;
        std::printf("    expert %3zu : max_abs=%.2e rel=%.2e %zu/%zu %s  compute %.3f ms  xfer %.3f ms  vram %.0f MiB\\n",
            expert, max_abs, relative, mismatched, kHiddenLlm, pass ? "PASS" : "FAIL", timings[2], timings[1],
            timings[3]);
    }

    double combined_max = 0.0, combined_sq = 0.0, combined_ref = 0.0;
    for (size_t index = 0; index < kHiddenLlm; ++index) {
        const double difference = static_cast<double>(combined_gpu[index]) - combined_cpu[index];
        combined_max = std::max(combined_max, std::abs(difference));
        combined_sq += difference * difference;
        combined_ref += static_cast<double>(combined_cpu[index]) * combined_cpu[index];
    }
    std::printf("  gpu moe         : %d experts of layer %zu, %d iterations each\\n", 8, layer, iterations);
    std::printf("    combined      : max_abs=%.2e rel=%.2e (routed sum at equal weights)\\n", combined_max,
        std::sqrt(combined_sq / (combined_ref > 0.0 ? combined_ref : 1.0)));
    std::printf("    8 experts     : compute %.3f ms  transfer %.3f ms  sum %.3f ms\\n", total_gemm,
        total_xfer, total_gemm + total_xfer);
    std::printf("    verdict       : %s\\n", all_pass ? "all selected experts PASS" : "FAILURES present");
#endif
}

void dump_thinker_probe(const MingTinyAssets & assets'''
assert anchor in text
text = text.replace(anchor, impl, 1)
c.write_text(text, encoding="utf-8")
print("gpu moe benchmark added")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
anchor2 = '''        if (const char * gpu_env = std::getenv("MING3_TINY_GPU_BENCH"); gpu_env != nullptr && *gpu_env != '\\0') {
            run_gpu_layer_benchmark(*assets, std::atoi(gpu_env));
        }'''
gate = anchor2 + '''

        if (const char * moe_env = std::getenv("MING3_TINY_GPU_MOE"); moe_env != nullptr && *moe_env != '\\0') {
            run_gpu_moe_benchmark(*assets, std::atoi(moe_env));
        }'''
assert anchor2 in src
a.write_text(src.replace(anchor2, gate), encoding="utf-8")
print("assets.cpp gates the gpu moe benchmark")
