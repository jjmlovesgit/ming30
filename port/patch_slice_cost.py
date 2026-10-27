from pathlib import Path

p = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\thinker.cpp")
text = p.read_text(encoding="utf-8")

old = """    for (size_t slot = 0; slot < 8; ++slot) {
        const size_t expert = static_cast<size_t>(experts[slot]);
        const auto gate_w = slice(gate_bank, expert, kInter * kHiddenLlm);
        const auto up_w = slice(up_bank, expert, kInter * kHiddenLlm);
        const auto down_w = slice(down_bank, expert, kHiddenLlm * kInter);"""
new = """    double slice_ms = 0.0;
    for (size_t slot = 0; slot < 8; ++slot) {
        const size_t expert = static_cast<size_t>(experts[slot]);
        const auto slice_start = std::chrono::steady_clock::now();
        const auto gate_w = slice(gate_bank, expert, kInter * kHiddenLlm);
        const auto up_w = slice(up_bank, expert, kInter * kHiddenLlm);
        const auto down_w = slice(down_bank, expert, kHiddenLlm * kInter);
        slice_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - slice_start).count();"""
assert old in text
text = text.replace(old, new)

old = """    std::printf("    8 experts     : compute %.3f ms  transfer %.3f ms  sum %.3f ms\\n", total_gemm,
        total_xfer, total_gemm + total_xfer);"""
new = """    std::printf("    8 experts     : compute %.3f ms  transfer %.3f ms  sum %.3f ms\\n", total_gemm,
        total_xfer, total_gemm + total_xfer);
    std::printf("    host slicing  : %.3f ms to extract 8 experts x 3 matrices (f32)\\n", slice_ms);
    std::printf("    hybrid note   : slicing + f32 upload per layer would cost %.3f ms per MoE layer\\n",
        slice_ms + total_xfer);"""
assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("slicing cost instrumented")
