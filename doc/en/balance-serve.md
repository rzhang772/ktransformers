# Balance Serve backend (multi-concurrency) for ktransformers

## KTransformers v0.2.4 Release Notes
We are excited to announce the official release of the long-awaited **KTransformers v0.2.4**!
In this version, we’ve added highly desired **multi-concurrency** support to the community through a major refactor of the whole architecture, updating more than 10,000 lines of code.
By drawing inspiration from the excellent architecture of sglang, we have implemented high-performance asynchronous concurrent scheduling in C++, including features like continuous batching, chunked prefill, and more. Thanks to GPU sharing in concurrent scenarios, overall throughput is also improved to a certain extent. The following is a demonstration:



https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

</p>

### 🚀 Key Updates
1. Multi-Concurrency Support
   - Added capability to handle multiple concurrent inference requests. Supports receiving and executing multiple tasks simultaneously.
   - We implemented [custom_flashinfer](https://github.com/kvcache-ai/custom_flashinfer/tree/fix-precision-mla-merge-main) based on the high-performance and highly flexible operator library [flashinfer](https://github.com/flashinfer-ai/flashinfer/), and achieved a variable batch size CUDA Graph, which further enhances flexibility while reducing memory and padding overhead.
   - In our benchmarks, overall throughput improved by approximately 130% under 4-way concurrency.
   - With support from Intel, we tested KTransformers v0.2.4 on the latest Xeon6 + MRDIMM-8800 platform. By increasing concurrency, the total output throughput increased from 17 tokens/s to 40 tokens/s. We observed that the bottleneck has now shifted to the GPU. Using a higher-end GPU than the 4090D could further improve performance.
2. Engine Architecture Optimization
 ![image](https://github.com/user-attachments/assets/f5f001fa-dca7-4377-a01a-32192902aa47)
  Inspired by the scheduling framework of sglang, we refactored KTransformers with a clearer three-layer architecture through an update of 11,000 lines of code, now supporting full multi-concurrency:
   - Server：Handles user requests and serves the OpenAI-compatible API.
   - Inference Engine：Executes model inference and supports chunked prefill.
   - Scheduler：Manages task scheduling and requests orchestration. Supports continuous batching by organizing queued requests into batches in a FCFS manner and sending them to the inference engine.
3. Project Structure Reorganization
All C/C++ code is now centralized under the /csrc directory.
4. Parameter Adjustments
Removed some legacy and deprecated launch parameters for a cleaner configuration experience.
We plan to provide a complete parameter list and detailed documentation in future releases to facilitate flexible configuration and debugging.
### 📚 Upgrade Notes
- Due to parameter changes, users who have installed previous versions are advised to delete the ~/.ktransformers directory and reinitialize.
- To enable multi-concurrency, please refer to the latest documentation for configuration examples.
### What's Changed
Implemented **custom_flashinfer** @Atream @ovowei @qiyuxinlin
Implemented **balance_serve** engine based on **FlashInfer** @qiyuxinlin @ovowei 
Implemented a **continuous batching** scheduler in C++ @ErvinXie 
release: bump version v0.2.4 by @Atream @Azure-Tang @ErvinXie  @qiyuxinlin @ovowei @KMSorSMS @SkqLiao 



## Installation Guide
⚠️ Please note that installing this project will replace flashinfer in your environment. It is strongly recommended to create a new conda environment!!!

⚠️ Please note that installing this project will replace flashinfer in your environment. It is strongly recommended to create a new conda environment!!!

⚠️ Please note that installing this project will replace flashinfer in your environment. It is strongly recommended to create a new conda environment!!!
### 1. Set Up Conda Environment

We recommend using Miniconda3/Anaconda3 for environment management:

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create --name ktransformers python=3.11
conda activate ktransformers

# Install required libraries
conda install -c conda-forge libstdcxx-ng

# Verify GLIBCXX version (should include 3.4.32)
strings ~/anaconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
```

> **Note:** Adjust the Anaconda path if your installation directory differs from `~/anaconda3`

### 2. Install dependencies

```bash
sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libfmt-dev libgflags-dev zlib1g-dev patchelf
```

### 3. Build ktransformers

```bash
# Clone repository
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive


# Install single NUMA dependencies
sudo env USE_BALANCE_SERVE=1 PYTHONPATH="$(which python)" PATH="$(dirname $(which python)):$PATH" bash ./install.sh
# Or Install Dual NUMA dependencies
sudo env USE_BALANCE_SERVE=1 USE_NUMA=1 PYTHONPATH="$(which python)" PATH="$(dirname $(which python)):$PATH" bash ./install.sh
```

## Running DeepSeek-R1-Q4KM Models

### 1. Run for 24GB VRAM GPUs

Use our optimized configuration for constrained VRAM:

```bash
python ktransformers/server/main.py \
  --port 10002
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

It features the following arguments:

- `--max_new_tokens`: Maximum number of tokens generated per request.
- `--cache_lens`: Total length of kvcache allocated by the scheduler. All requests share a kvcache space.
- `--chunk_size`: Maximum number of tokens processed in a single run by the engine.
  corresponding to 32768 tokens, and the space occupied will be released after the requests are completed.
- `--max_batch_size`: Maximum number of requests (prefill + decode) processed in a single run by the engine. (Supported only by `balance_serve`)
- `--backend_type`: `balance_serve` is a multi-concurrency backend engine introduced in version v0.2.4. The original single-concurrency engine is `ktransformers`.

### 2. access server
```
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "DeepSeek-R1",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```
