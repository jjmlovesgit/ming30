from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

start = text.index("void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {")
end = text.index("void dump_thinker_probe(const MingTinyAssets & assets")
body = text[start:end]

replaced = body.replace("kHidden", "kHiddenLlm")
replaced = replaced.replace(
    "void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {",
    "void run_gpu_layer_benchmark(const MingTinyAssets & assets, int iterations) {\n    constexpr size_t kHiddenLlm = 1536;",
)
text = text[:start] + replaced + text[end:]
p.write_text(text, encoding="utf-8")
print("benchmark now uses a local hidden-size constant:", replaced.count("kHiddenLlm"), "references")
