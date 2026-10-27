#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Benchmark one KDA layer's projection GEMMs (one-token decode shape) on GPU with
// cuBLAS, reporting where the time goes. Returns 0 on success.
//   out[0] weight upload ms (one-time)
//   out[1] activation transfer ms (per call)
//   out[2] compute ms (per call)
//   out[3] total per-call ms
//   out[4] peak VRAM delta MiB
//   out[5] checksum of the GPU result (sum)
int ming_gpu_layer_benchmark(const float * q_w, const float * k_w, const float * v_w, const float * f_w,
    const float * g_w, const float * b_w, const float * o_w, const float * x, int iterations, double * out);

#ifdef __cplusplus
}
#endif
