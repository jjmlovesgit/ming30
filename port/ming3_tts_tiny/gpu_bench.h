#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// One projection: y[out] = W[out, in] * x[in], computed with cuBLAS on the device.
// Returns 0 on success and writes the device result back into y_host for comparison.
//   timings[0] weight upload ms (one-time)
//   timings[1] activation transfer ms (per call, H2D x + D2H y)
//   timings[2] GEMM ms (per call)
//   timings[3] peak VRAM delta MiB
int ming_gpu_gemm(const float * weight, const float * x, int out, int in, int iterations, float * y_host,
    double * timings);

#ifdef __cplusplus
}
#endif
