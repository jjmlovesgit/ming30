#include "engine/models/ming3_tts_tiny/gpu_residency.h"

#include <cuda_runtime.h>

#include <vector>

namespace {

struct State {
    std::vector<void *> buffers;
    size_t total = 0;
    size_t peak_used = 0;
};

State & state() {
    static State instance;
    return instance;
}

void sample_peak() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
    const size_t used = total_bytes - free_bytes;
    State & self = state();
    if (used > self.peak_used) self.peak_used = used;
}

}  // namespace

extern "C" int ming_gpu_residency_begin(void) {
    State & self = state();
    self.buffers.clear();
    self.total = 0;
    sample_peak();
    self.peak_used = 0;
    sample_peak();
    return 0;
}

extern "C" int ming_gpu_residency_add(const void * bytes, size_t size) {
    State & self = state();
    void * buffer = nullptr;
    if (cudaMalloc(&buffer, size) != cudaSuccess) return 1;
    if (cudaMemcpy(buffer, bytes, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(buffer);
        return 2;
    }
    self.buffers.push_back(buffer);
    self.total += size;
    sample_peak();
    return 0;
}

extern "C" int ming_gpu_residency_end(double * out) {
    State & self = state();
    sample_peak();
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const double mib = 1024.0 * 1024.0;
    out[0] = static_cast<double>(self.total) / mib;
    out[1] = static_cast<double>(self.peak_used) / mib;
    out[2] = static_cast<double>(free_bytes) / mib;
    return 0;
}

extern "C" void ming_gpu_residency_release(void) {
    State & self = state();
    for (void * buffer : self.buffers) cudaFree(buffer);
    self.buffers.clear();
    self.total = 0;
    self.peak_used = 0;
}
