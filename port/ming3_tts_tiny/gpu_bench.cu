#include "engine/models/ming3_tts_tiny/gpu_bench.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>

namespace {

using clock_type = std::chrono::steady_clock;

double millis(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

}  // namespace

extern "C" int ming_gpu_gemm(const float * weight, const float * x, int out, int in, int iterations,
    float * y_host, double * timings) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;

    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    float * device_weight = nullptr;
    float * device_x = nullptr;
    float * device_y = nullptr;
    const auto upload_start = clock_type::now();
    cudaMalloc(&device_weight, static_cast<size_t>(out) * in * sizeof(float));
    cudaMalloc(&device_x, static_cast<size_t>(in) * sizeof(float));
    cudaMalloc(&device_y, static_cast<size_t>(out) * sizeof(float));
    cudaMemcpy(device_weight, weight, static_cast<size_t>(out) * in * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const double upload_ms = millis(upload_start, clock_type::now());

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    // Row-major W[out, in] and x[in] as a column of length in.
    // cuBLAS is column-major: C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C.
    // Take m = out, n = 1, k = in, A = W in column-major, which is the same memory as
    // row-major W[out, in] read as [in, out], hence op(A) = A^T with lda = in.
    const float alpha = 1.0f, beta = 0.0f;
    double transfer_ms = 0.0, gemm_ms = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto transfer_start = clock_type::now();
        cudaMemcpy(device_x, x, static_cast<size_t>(in) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_host, device_y, static_cast<size_t>(out) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        transfer_ms += millis(transfer_start, clock_type::now());

        const auto gemm_start = clock_type::now();
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, 1, in, &alpha, device_weight, in, device_x, in,
            &beta, device_y, out);
        cudaDeviceSynchronize();
        gemm_ms += millis(gemm_start, clock_type::now());

        cudaMemcpy(y_host, device_y, static_cast<size_t>(out) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    const int count = iterations > 0 ? iterations : 1;
    timings[0] = upload_ms;
    timings[1] = transfer_ms / count;
    timings[2] = gemm_ms / count;
    timings[3] = static_cast<double>(free_before - free_after) / (1024.0 * 1024.0);

    cudaFree(device_weight);
    cudaFree(device_x);
    cudaFree(device_y);
    cublasDestroy(handle);
    return 0;
}
