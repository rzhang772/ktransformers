import torch
import gpu_memcpy_test

# 分配 GPU tensor
n = 1024
t = torch.zeros(n, dtype=torch.float32, device="cuda")

print("Python GPU tensor ptr:", t.data_ptr())

# 调用 C++ memcpy
gpu_memcpy_test.gpu_memcpy(t.data_ptr(), n)

# 回到 Python 检查前10个元素
t_cpu = t.cpu()
print("前10个元素:", t_cpu[:10])