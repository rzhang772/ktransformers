#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// 简单的耗时计算 kernel
__global__ void busyKernel(float *data, int N, int loops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        for (int i = 0; i < loops; i++) {
            x = sinf(x) * cosf(x) + 1.0f;
        }
        data[idx] = x;
    }
}

int main() {
    const int N = 1 << 20;
    const int loops = 100000; // 增大循环让 kernel 耗时
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // 分配 pinned 内存用于 H2D/D2H
    const size_t copySize = 200 * 1024 * 1024; // 200MB
    float *h_pinned, *d_buffer;
    cudaMallocHost(&h_pinned, copySize);
    cudaMalloc(&d_buffer, copySize);

    // 创建一个非默认 stream
    cudaStream_t s2;
    cudaStreamCreate(&s2);

    // --------------- case 1: 只有 kernel ----------------
    auto t1 = std::chrono::high_resolution_clock::now();
    busyKernel<<<(N + 255) / 256, 256>>>(d_data, N, loops);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double kernel_only = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "Case1: Kernel only = " << kernel_only << " sec" << std::endl;

    // --------------- case 2: kernel + memcpyAsync ----------------
    auto t3 = std::chrono::high_resolution_clock::now();
    // 在 s2 上启动一个异步大拷贝
    cudaMemcpyAsync(d_buffer, h_pinned, copySize, cudaMemcpyHostToDevice, s2);

    // 同时在 默认 stream 上启动 kernel
    busyKernel<<<(N + 255) / 256, 256>>>(d_data, N, loops);

    cudaDeviceSynchronize(); // 等待所有任务完成
    auto t4 = std::chrono::high_resolution_clock::now();
    double kernel_with_copy = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "Case2: Kernel + memcpyAsync = " 
              << kernel_with_copy << " sec" << std::endl;

    // 清理
    cudaStreamDestroy(s2);
    cudaFree(d_data);
    cudaFree(d_buffer);
    cudaFreeHost(h_pinned);

    return 0;
}