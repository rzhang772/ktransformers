#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdio>

namespace py = pybind11;

void gpu_memcpy(uint64_t dst_ptr_val, size_t n) {
    // 将传入的 uint64_t 转成 GPU 指针
    void* dst = reinterpret_cast<void*>(dst_ptr_val);

    // 创建 CPU 数据
    float* h = new float[n];
    for(size_t i = 0; i < n; i++) h[i] = static_cast<float>(i);

    // 同步 memcpy
    cudaError_t err = cudaMemcpy(dst, h, n * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("Memcpy failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Memcpy success!\n");
    }

    delete[] h;
}

PYBIND11_MODULE(gpu_memcpy_test, m) {
    m.def("gpu_memcpy", &gpu_memcpy, "Memcpy data to GPU pointer",
          py::arg("dst_ptr_val"), py::arg("n"));
}