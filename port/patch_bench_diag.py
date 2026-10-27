from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")
old = """    std::printf("    gpu checksum  : %12.4f   cpu checksum: %12.4f\\n", timings[5],
        std::accumulate(cpu_out.begin(), cpu_out.end(), 0.0));"""
new = """    {
        double weight_sum = 0.0, weight_sq = 0.0, out_sq = 0.0;
        for (float value : q_w) { weight_sum += value; weight_sq += static_cast<double>(value) * value; }
        for (float value : cpu_out) { out_sq += static_cast<double>(value) * value; }
        std::printf("    diagnostics   : |q_w| rms=%.6f mean=%.6f |cpu_out| rms=%.6f |gpu_out| rms=%.6f\\n",
            std::sqrt(weight_sq / q_w.size()), weight_sum / q_w.size(),
            std::sqrt(out_sq / cpu_out.size()),
            std::sqrt(static_cast<double>(timings[5]) * timings[5] / cpu_out.size()));
    }
    std::printf("    gpu checksum  : %12.4f   cpu checksum: %12.4f\\n", timings[5],
        std::accumulate(cpu_out.begin(), cpu_out.end(), 0.0));"""
assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("diagnostics added")
