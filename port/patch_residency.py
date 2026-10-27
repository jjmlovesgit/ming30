from pathlib import Path

h = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.h")
text = h.read_text(encoding="utf-8")
if "run_gpu_residency_probe" not in text:
    text = text.replace(
        "void run_gpu_moe_benchmark(const MingTinyAssets & assets, int iterations);",
        "void run_gpu_moe_benchmark(const MingTinyAssets & assets, int iterations);\n"
        "// Stream every MoE layer's BF16 expert banks onto the device and report residency.\n"
        "void run_gpu_residency_probe(const MingTinyAssets & assets);",
    )
    h.write_text(text, encoding="utf-8")

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = c.read_text(encoding="utf-8")
text = text.replace('#include "engine/models/ming3_tts_tiny/gpu_moe.h"',
                    '#include "engine/models/ming3_tts_tiny/gpu_moe.h"\n'
                    '#include "engine/models/ming3_tts_tiny/gpu_residency.h"')

anchor = "void dump_thinker_probe(const MingTinyAssets & assets"
impl = '''void run_gpu_residency_probe(const MingTinyAssets & assets) {
#if !defined(ENGINE_HAS_MING_GPU_BENCH)
    std::printf("  gpu residency   : not built with CUDA\\n");
    (void) assets;
#else
    const auto & source = *assets.weights;
    constexpr size_t kHiddenLlm = 1536;
    constexpr size_t kInter = 512;
    ming_gpu_residency_begin();
    size_t banks = 0;
    for (size_t layer = 1; layer < static_cast<size_t>(assets.config.layers); ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.";
        const std::vector<std::pair<std::string, std::array<int64_t, 3>>> tensors{
            {prefix + "experts.gate_proj.weight", {128, static_cast<int64_t>(kInter), static_cast<int64_t>(kHiddenLlm)}},
            {prefix + "experts.up_proj.weight", {128, static_cast<int64_t>(kInter), static_cast<int64_t>(kHiddenLlm)}},
            {prefix + "experts.down_proj.weight", {128, static_cast<int64_t>(kHiddenLlm), static_cast<int64_t>(kInter)}},
        };
        for (const auto & [name, shape] : tensors) {
            const auto bank = load_raw(source, name, Shape{shape[0], shape[1], shape[2]});
            if (ming_gpu_residency_add(bank.bytes.data(), bank.bytes.size()) != 0) {
                std::printf("  gpu residency   : allocation failed at layer %zu\\n", layer);
                ming_gpu_residency_release();
                return;
            }
            ++banks;
        }
    }
    double out[3] = {0, 0, 0};
    ming_gpu_residency_end(out);
    std::printf("  gpu residency   : %zu BF16 bank tensors streamed, total %.0f MiB (%.2f GiB)\\n", banks,
        out[0], out[0] / 1024.0);
    std::printf("    peak VRAM used: %.0f MiB (%.2f GiB)  free after: %.0f MiB\\n", out[1], out[1] / 1024.0,
        out[2]);
    ming_gpu_residency_release();
#endif
}

void dump_thinker_probe(const MingTinyAssets & assets'''
assert anchor in text
text = text.replace(anchor, impl, 1)
if "#include <array>" not in text:
    text = text.replace("#include <algorithm>", "#include <algorithm>\n#include <array>")
c.write_text(text, encoding="utf-8")
print("residency probe added")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
anchor2 = '''        if (const char * moe_env = std::getenv("MING3_TINY_GPU_MOE"); moe_env != nullptr && *moe_env != '\\0') {
            run_gpu_moe_benchmark(*assets, std::atoi(moe_env));
        }'''
gate = anchor2 + '''

        if (const char * res_env = std::getenv("MING3_TINY_GPU_RESIDENCY"); res_env != nullptr && *res_env != '\\0') {
            run_gpu_residency_probe(*assets);
        }'''
assert anchor2 in src
a.write_text(src.replace(anchor2, gate), encoding="utf-8")
print("assets.cpp gates the residency probe")
