import subprocess
import re
import csv
import os
import time

# 参数范围
cpu_infer_range = range(8, 11, 2)
prefetch_num_range = range(-1, 9)
gpu_compute_max_num_range = range(8, 9)
prefetch_method = 0  # 0: token prefetch, 1: layer prefetch
prefetch_strategy = 0  # 0: 固定传输数量， 1: 动态传输数量
skip_layer = 1  # 测试传统的skip layer prefetch方法，其中提前多少层。例如当为1时，在i层开始prefetch i+1层

run_time = time.strftime("%Y%m%d-%H%M%S")
# 输出文件
if prefetch_method == 0:
    output_file = f"./expirments/{run_time}_results_token_prefetch_strategy{prefetch_strategy}.csv"
elif prefetch_method == 1:
    output_file = f"./expirments/{run_time}_results_layer_prefetch_strategy{prefetch_strategy}.csv"
else:
    output_file = f"./expirments/{run_time}_experiment_results.csv"

# 日志目录
log_dir = "./expirments/experiment_results_logs"
os.makedirs(log_dir, exist_ok=True)

# CSV 表头
headers = ["cpu_infer", "prefetch_num", "eval_count", "eval_duration", "eval_rate", "hit_rate", "gpu_compute_max_num"]

# 正则匹配模式（使用多行模式并锚定到行首，避免匹配到 'prompt eval ...'）
patterns = {
    "eval_count": re.compile(r"(?m)^\s*eval count:\s*(\d+)"),
    "eval_duration": re.compile(r"(?m)^\s*eval duration:\s*([\d\.eE+-]+)s?"),
    "eval_rate": re.compile(r"(?m)^\s*eval rate:\s*([\d\.eE+-]+)"),
    "hit_rate": re.compile(r"(?m)^\s*hit rate:\s*([\d\.eE+-]+)"),
}

def run_experiment(cpu_infer, prefetch_num, gpu_compute_max_num):
    # 使用 -u 以强制无缓冲输出（对于 python 脚本的实时日志很重要）
    cmd = [
        "python", "-u", "./ktransformers/local_chat.py",
        "--model_path", "/mnt/incontainer/models/DeepSeek-V3/DeepSeek-V3-0324-config/",
        "--gguf_path", "/mnt/incontainer/models/deepseek-ai_DeepSeek-V3-0324-GGUF/deepseek-ai_DeepSeek-V3-0324-IQ4_XS/",
        "--prompt_file", "./myprompt.txt",
        "--max_new_tokens", "100",
        "--cpu_infer", str(cpu_infer),
        "--prefetch_num", str(prefetch_num),
        "--prefetch_method", "0",  # 0: token prefetch, 1: layer prefetch
        "--gpu_compute_max_num", str(gpu_compute_max_num),
        # "--prefetch_strategy", "0",  # 0: 固定传输数量 1: 动态传输数量
        # "--skip_layer", "1",  # 测试传统的skip layer prefetch方法，其中提前多少层。例如当为1时，在i层开始prefetch i+1层
    ]

    if prefetch_method == 0:
        log_file = os.path.join(log_dir, f"{run_time}_output_token_prefetch_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")
    elif prefetch_method == 1:
        log_file = os.path.join(log_dir, f"{run_time}_output_layer_prefetch_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")
    else:
        log_file = os.path.join(log_dir, f"{run_time}_output_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")
    output_lines = []

    try:
        # redirect stderr -> stdout，这样日志里能保留错误信息的顺序
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        # 实时读取输出并写入日志
        with open(log_file, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                output_lines.append(line)
            proc.wait()
            f.write(f"\n=== PROCESS RETURN CODE: {proc.returncode} ===\n")
    except Exception as e:
        # 如果 Popen 抛错，记录到日志并返回空指标
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n=== EXCEPTION DURING RUN ===\n")
            f.write(str(e) + "\n")
        return {k: None for k in patterns.keys()}

    combined_output = "".join(output_lines)

    # 提取指标（并转换为合适类型）
    metrics = {}
    for key, pattern in patterns.items():
        match = pattern.search(combined_output)
        if match:
            val = match.group(1)
            try:
                if key == "eval_count":
                    metrics[key] = int(val)
                else:
                    metrics[key] = float(val)
            except Exception:
                metrics[key] = val  # fallback: 原始字符串
        else:
            metrics[key] = None

    return metrics

# 写入 CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for cpu_infer in cpu_infer_range:
        for gpu_compute_max_num in gpu_compute_max_num_range:
            for prefetch_num in prefetch_num_range:
                print(f"Running: cpu_infer={cpu_infer}, gpu_compute_max_num={gpu_compute_max_num}, prefetch_num={prefetch_num}")
                metrics = run_experiment(cpu_infer, prefetch_num, gpu_compute_max_num)
                row = {
                    "cpu_infer": cpu_infer,
                    "prefetch_num": prefetch_num,
                    "eval_count": metrics["eval_count"],
                    "eval_duration": metrics["eval_duration"],
                    "eval_rate": metrics["eval_rate"],
                    "hit_rate": metrics["hit_rate"],
                    "gpu_compute_max_num": gpu_compute_max_num,
                }
                writer.writerow(row)