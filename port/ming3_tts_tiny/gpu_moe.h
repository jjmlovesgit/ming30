#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// One routed expert, computed entirely on the device:
//   act = silu(gate_w * x) * (up_w * x);  y = down_w * act
// gate_w [inter, hidden], up_w [inter, hidden], down_w [hidden, inter], x [hidden], y [hidden].
// timings[0] upload ms (one-time), timings[1] transfer ms (per call),
// timings[2] compute ms (per call, includes kernel + GEMMs), timings[3] VRAM MiB.
int ming_gpu_expert(const float * gate_w, const float * up_w, const float * down_w, const float * x,
    int hidden, int inter, int iterations, float * y_host, double * timings);

#ifdef __cplusplus
}
#endif
