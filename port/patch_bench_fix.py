from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

if '#include "engine/models/ming3_tts_tiny/gpu_bench.h"' not in text:
    text = text.replace('#include "engine/models/ming3_tts_tiny/thinker.h"',
                        '#include "engine/models/ming3_tts_tiny/thinker.h"\n#include "engine/models/ming3_tts_tiny/gpu_bench.h"')

# drop the stale declaration of the previous benchmark symbol
stale = '''#if defined(ENGINE_HAS_MING_GPU_BENCH)
extern "C" int ming_gpu_layer_benchmark(const float * q_w, const float * k_w, const float * v_w,
    const float * f_w, const float * g_w, const float * b_w, const float * o_w, const float * x,
    int iterations, double * out);
#endif

'''
if stale in text:
    text = text.replace(stale, "")

text = text.replace(
    "        if (std::abs(difference) > 1e-4 * std::max(1.0, std::abs(reference[index]))) ++mismatched;",
    "        const double tolerance = 1e-4 * std::max(1.0, static_cast<double>(std::abs(reference[index])));\n"
    "        if (std::abs(difference) > tolerance) ++mismatched;",
)
p.write_text(text, encoding="utf-8")
print("include added, stale declaration removed, tolerance fixed")
