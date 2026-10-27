#include "engine/models/ming3_tts_tiny/gpu_moe.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>

namespace {

using clock_type = std::chrono::steady_clock;

double millis(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// act[i] = silu(gate[i]) * up[i], kept on the device so the expert needs no host round trip
// between its three GEMMs.
__global__ void silu_gate_kernel(const float * gate, const float * up, float * act, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = gate[index];
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    act[index] = value * sigmoid * up[index];
}

void gemm(cublasHandle_t handle, const float * weight, const float * input, float * output, int out, int in) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, 1, in, &alpha, weight, in, input, in, &beta, output, out);
}

}  // namespace

extern "C" int ming_gpu_expert(const float * gate_w, const float * up_w, const float * down_w, const float * x,
    int hidden, int inter, int iterations, float * y_host, double * timings) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;

    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    float * device_gate = nullptr, * device_up = nullptr, * device_down = nullptr;
    float * device_x = nullptr, * device_g = nullptr, * device_u = nullptr, * device_act = nullptr;
    float * device_y = nullptr;

    const auto upload_start = clock_type::now();
    cudaMalloc(&device_gate, static_cast<size_t>(inter) * hidden * sizeof(float));
    cudaMalloc(&device_up, static_cast<size_t>(inter) * hidden * sizeof(float));
    cudaMalloc(&device_down, static_cast<size_t>(hidden) * inter * sizeof(float));
    cudaMalloc(&device_x, static_cast<size_t>(hidden) * sizeof(float));
    cudaMalloc(&device_g, static_cast<size_t>(inter) * sizeof(float));
    cudaMalloc(&device_u, static_cast<size_t>(inter) * sizeof(float));
    cudaMalloc(&device_act, static_cast<size_t>(inter) * sizeof(float));
    cudaMalloc(&device_y, static_cast<size_t>(hidden) * sizeof(float));
    cudaMemcpy(device_gate, gate_w, static_cast<size_t>(inter) * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_up, up_w, static_cast<size_t>(inter) * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_down, down_w, static_cast<size_t>(hidden) * inter * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const double upload_ms = millis(upload_start, clock_type::now());

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    const int threads = 256;
    const int blocks = (inter + threads - 1) / threads;
    double transfer_ms = 0.0, compute_ms = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto transfer_start = clock_type::now();
        cudaMemcpy(device_x, x, static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        transfer_ms += millis(transfer_start, clock_type::now());

        const auto compute_start = clock_type::now();
        gemm(handle, device_gate, device_x, device_g, inter, hidden);
        gemm(handle, device_up, device_x, device_u, inter, hidden);
        silu_gate_kernel<<<blocks, threads>>>(device_g, device_u, device_act, inter);
        gemm(handle, device_down, device_act, device_y, hidden, inter);
        cudaDeviceSynchronize();
        compute_ms += millis(compute_start, clock_type::now());

        const auto download_start = clock_type::now();
        cudaMemcpy(y_host, device_y, static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        transfer_ms += millis(download_start, clock_type::now());
    }

    const int count = iterations > 0 ? iterations : 1;
    timings[0] = upload_ms;
    timings[1] = transfer_ms / count;
    timings[2] = compute_ms / count;
    timings[3] = static_cast<double>(free_before - free_after) / (1024.0 * 1024.0);

    cudaFree(device_gate);
    cudaFree(device_up);
    cudaFree(device_down);
    cudaFree(device_x);
    cudaFree(device_g);
    cudaFree(device_u);
    cudaFree(device_act);
    cudaFree(device_y);
    cublasDestroy(handle);
    return 0;
}
