import subprocess
import re
import csv
import os
import time

# 参数范围
baseline = 0  # 1:原始KT， 0:修改后的KT
cpu_infer_range = range(10, 11, 2)
prefetch_num_range = [-1, 4] #range(-1, 5)
gpu_compute_max_num_range = range(7, 8)
prefetch_method = 0  # 0: token prefetch, 1: layer prefetch
prefetch_strategy = 0
skip_layer = 1
prefetch_start_layer = 0

run_time = time.strftime("%Y%m%d-%H%M%S")
print(f"\n实验开始时间: {run_time}, \n参数范围: \ncpu_infer={list(cpu_infer_range)}, \nprefetch_num={list(prefetch_num_range)}, \ngpu_compute_max_num={list(gpu_compute_max_num_range)}, \nprefetch_method={prefetch_method}, \nprefetch_strategy={prefetch_strategy}, \nskip_layer={skip_layer}, \nprefetch_start_layer={prefetch_start_layer}")
# 输出文件
if prefetch_method == 0:
    output_file = f"./expirments/{run_time}_results_token_strategy{prefetch_strategy}_start{prefetch_start_layer}.csv"
elif prefetch_method == 1:
    output_file = f"./expirments/{run_time}_results_layer_strategy{prefetch_strategy}.csv"
else:
    output_file = f"./expirments/{run_time}_experiment_results.csv"

# 日志目录
log_dir = "./expirments/experiment_results_logs"
os.makedirs(log_dir, exist_ok=True)

# CSV 表头（包含所有指标）
headers = [
    "cpu_infer", "prefetch_num", "gpu_compute_max_num",
    "dataset_name", "file_name",
    "prompt_eval_count", "prompt_eval_duration", "prompt_eval_rate",
    "eval_count", "eval_duration", "eval_rate",
    "hit_rate", "prefetch_hit_rate", "prefetch_start_layer",
]

# 正则匹配模式
block_pattern = re.compile(
    r"dataset name:\s*(\S+).*?"
    r"file name:\s*(\S+).*?"
    r"prompt eval count:\s*(\d+).*?"
    r"prompt eval duration:\s*([\d\.eE+-]+)s.*?"
    r"prompt eval rate:\s*([\d\.eE+-]+).*?"
    r"eval count:\s*(\d+).*?"
    r"eval duration:\s*([\d\.eE+-]+)s.*?"
    r"eval rate:\s*([\d\.eE+-]+).*?"
    r"hit rate:\s*([\d\.eE+-]+).*?"
    r"prefetch hit rate:\s*([\d\.eE+-]+)",
    re.DOTALL
)

def run_experiment(cpu_infer, prefetch_num, gpu_compute_max_num):
    cmd = [
        "python", "-u", "./ktransformers/local_chat.py",
        "--model_path", "/mnt/incontainer/models/DeepSeek-V3/DeepSeek-V3-0324-config/",
        "--gguf_path", "/mnt/incontainer/models/deepseek-ai_DeepSeek-V3-0324-GGUF/deepseek-ai_DeepSeek-V3-0324-IQ4_XS/",
        "--prompt_file", "./myprompt.txt",
        "--max_new_tokens", "100",
        "--cpu_infer", str(cpu_infer),
        "--baseline", str(baseline),
        "--prefetch_num", str(prefetch_num),
        "--prefetch_method", str(prefetch_method),
        "--gpu_compute_max_num", str(gpu_compute_max_num),
        "--prefetch_strategy", str(prefetch_strategy),
        "--skip_layer", str(skip_layer),
        "--prefetch_start_layer", str(prefetch_start_layer)
    ]

    # 日志文件路径
    if prefetch_method == 0:
        log_file = os.path.join(log_dir, f"{run_time}_output_token_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")
    elif prefetch_method == 1:
        log_file = os.path.join(log_dir, f"{run_time}_output_layer_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")
    else:
        log_file = os.path.join(log_dir, f"{run_time}_output_cpu{cpu_infer}_gpuLimit{gpu_compute_max_num}_prefetch{prefetch_num}.log")

    output_lines = []

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        with open(log_file, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                output_lines.append(line)
            proc.wait()
            f.write(f"\n=== PROCESS RETURN CODE: {proc.returncode} ===\n")
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n=== EXCEPTION DURING RUN ===\n")
            f.write(str(e) + "\n")
        return []

    combined_output = "".join(output_lines)

    # 查找所有匹配的结果块
    matches = block_pattern.findall(combined_output)
    results = []
    if not matches:
        print("⚠️ 未检测到任何结果块，可能程序输出格式改变或正则匹配失败。")
    else:
        print(f"✅ 检测到 {len(matches)} 条结果。")

    for i, match in enumerate(matches, start=1):
        (
            dataset_name, file_name,
            prompt_eval_count, prompt_eval_duration, prompt_eval_rate,
            eval_count, eval_duration, eval_rate,
            hit_rate, prefetch_hit_rate
        ) = match

        print(f"  ➤ 第 {i} 条结果: dataset={dataset_name}, file={file_name}, hit_rate={hit_rate}, prefetch_hit_rate={prefetch_hit_rate}")

        results.append({
            "cpu_infer": cpu_infer,
            "prefetch_num": prefetch_num,
            "gpu_compute_max_num": gpu_compute_max_num,
            "dataset_name": dataset_name,
            "file_name": file_name,
            "prompt_eval_count": int(prompt_eval_count),
            "prompt_eval_duration": float(prompt_eval_duration),
            "prompt_eval_rate": float(prompt_eval_rate),
            "eval_count": int(eval_count),
            "eval_duration": float(eval_duration),
            "eval_rate": float(eval_rate),
            "hit_rate": float(hit_rate),
            "prefetch_hit_rate": float(prefetch_hit_rate),
            "prefetch_start_layer": prefetch_start_layer,
        })
    return results


# 写入 CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for cpu_infer in cpu_infer_range:
        for gpu_compute_max_num in gpu_compute_max_num_range:
            for prefetch_num in prefetch_num_range:
                print(f"\n🚀 Running: cpu_infer={cpu_infer}, gpu_compute_max_num={gpu_compute_max_num}, prefetch_num={prefetch_num}")
                records = run_experiment(cpu_infer, prefetch_num, gpu_compute_max_num)

                if not records:
                    print("⚠️ 无有效结果，跳过写入。")
                    continue

                for row in records:
                    writer.writerow(row)

                print(f"✅ 已写入 {len(records)} 条结果到 CSV。")