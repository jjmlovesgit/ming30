#include "engine/models/ming3_tts_tiny/gpu_bench.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

namespace {

constexpr int kHidden = 1536;
constexpr int kProjection = 2048;
constexpr int kHeads = 16;

struct Device {
    float * q = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * f = nullptr;
    float * g = nullptr;
    float * b = nullptr;
    float * o = nullptr;
    float * x = nullptr;
    float * out = nullptr;  // [kProjection * 6 + kHeads + kHidden]
};

// y[out] = W[out, in] * x[in], the same orientation the CPU path uses.
void gemm(cublasHandle_t handle, const float * weight, const float * input, float * output, int out,
    int in) {
    const float alpha = 1.0f, beta = 0.0f;
    // weight is row-major [out, in]; cuBLAS is column-major, so compute
    // out = W * x as out^T = x^T * W^T with W treated as [in, out] transposed.
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, 1, in, &alpha, weight, in, input, in, &beta, output,
        out);
}

}  // namespace

extern "C" int ming_gpu_layer_benchmark(const float * q_w, const float * k_w, const float * v_w,
    const float * f_w, const float * g_w, const float * b_w, const float * o_w, const float * x,
    int iterations, double * out) {
    Device device;
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;

    const size_t projection_weights = static_cast<size_t>(kProjection) * kHidden * sizeof(float);
    const size_t output_size = (6 * kProjection + kHeads + kHidden) * sizeof(float);
    const size_t input_size = kHidden * sizeof(float);

    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    const auto upload_start = std::chrono::steady_clock::now();
    cudaMalloc(&device.q, projection_weights);
    cudaMalloc(&device.k, projection_weights);
    cudaMalloc(&device.v, projection_weights);
    cudaMalloc(&device.f, projection_weights);
    cudaMalloc(&device.g, projection_weights);
    cudaMalloc(&device.b, static_cast<size_t>(kHeads) * kHidden * sizeof(float));
    cudaMalloc(&device.o, static_cast<size_t>(kHidden) * kProjection * sizeof(float));
    cudaMalloc(&device.x, input_size);
    cudaMalloc(&device.out, output_size);
    cudaMemcpy(device.q, q_w, projection_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(device.k, k_w, projection_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(device.v, v_w, projection_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(device.f, f_w, projection_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(device.g, g_w, projection_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(device.b, b_w, static_cast<size_t>(kHeads) * kHidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device.o, o_w, static_cast<size_t>(kHidden) * kProjection * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_start).count();

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    // f32 copy of the activations, standing in for one decode step's hidden row
    std::vector<float> host_out(6 * kProjection + kHeads + kHidden, 0.0f);
    double transfer_ms = 0.0, compute_ms = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto transfer_start = std::chrono::steady_clock::now();
        cudaMemcpy(device.x, x, input_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        transfer_ms += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - transfer_start)
                           .count();

        auto compute_start = std::chrono::steady_clock::now();
        float * base = device.out;
        gemm(handle, device.q, device.x, base + 0 * kProjection, kProjection, kHidden);
        gemm(handle, device.k, device.x, base + 1 * kProjection, kProjection, kHidden);
        gemm(handle, device.v, device.x, base + 2 * kProjection, kProjection, kHidden);
        gemm(handle, device.f, device.x, base + 3 * kProjection, kProjection, kHidden);
        gemm(handle, device.g, device.x, base + 4 * kProjection, kProjection, kHidden);
        gemm(handle, device.b, device.x, base + 5 * kProjection, kHeads, kHidden);
        gemm(handle, device.o, base + 0 * kProjection, base + 5 * kProjection + kHeads, kHidden, kProjection);
        cudaDeviceSynchronize();
        compute_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - compute_start)
                          .count();

        transfer_start = std::chrono::steady_clock::now();
        cudaMemcpy(host_out.data(), base + 5 * kProjection + kHeads, static_cast<size_t>(kHidden) * sizeof(float),
            cudaMemcpyDeviceToHost);
        transfer_ms += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - transfer_start)
                           .count();
    }

    double checksum = 0.0;
    for (float value : host_out) checksum += value;

    const int count = iterations > 0 ? iterations : 1;
    out[0] = upload_ms;
    out[1] = transfer_ms / count;
    out[2] = compute_ms / count;
    out[3] = out[1] + out[2];
    out[4] = static_cast<double>(free_before - free_after) / (1024.0 * 1024.0);
    out[5] = checksum;

    cudaFree(device.q);
    cudaFree(device.k);
    cudaFree(device.v);
    cudaFree(device.f);
    cudaFree(device.g);
    cudaFree(device.b);
    cudaFree(device.o);
    cudaFree(device.x);
    cudaFree(device.out);
    cublasDestroy(handle);
    return 0;
}
