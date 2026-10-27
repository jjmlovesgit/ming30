from pathlib import Path

p = Path(r"C:\Projects\audio.cpp\CMakeLists.txt")
lines = [line for line in p.read_text(encoding="utf-8").splitlines()]

marker = "    target_sources(engine_model_ming3_tts_tiny PRIVATE src/models/ming3_tts_tiny/gpu_bench.cu)"
if marker not in lines:
    index = next(i for i, line in enumerate(lines) if line == "audiocpp_add_model(ming3_tts_tiny")
    end = index
    while lines[end].strip() != ")":
        end += 1
    block = [
        "",
        "if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)",
        marker,
        "    set_source_files_properties(",
        "        src/models/ming3_tts_tiny/gpu_bench.cu",
        "        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)",
        "    target_compile_definitions(engine_model_ming3_tts_tiny PRIVATE ENGINE_HAS_MING_GPU_BENCH=1)",
        "    target_link_libraries(engine_model_ming3_tts_tiny PRIVATE CUDA::cudart CUDA::cublas)",
        "endif()",
    ]
    lines[end + 1:end + 1] = block
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cuda block inserted after the model block ending at line {end + 1}")
else:
    print("cuda block already present")

for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
    if "gpu_bench" in line or "ENGINE_HAS_MING_GPU_BENCH" in line or "CUDA::cublas" in line:
        print(f"  L{number}: {line.strip()}")
