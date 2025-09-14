import subprocess
import re
import csv

# 参数范围
cpu_infer_range = range(1, 11)
prefetch_num_range = range(0, 5)

# 输出文件
output_file = "./experiment_results.csv"

# CSV 表头
headers = ["cpu_infer", "prefetch_num", "eval_count", "eval_duration", "eval_rate", "hit_rate"]

# 正则匹配模式
patterns = {
    "eval_count": re.compile(r"eval count:\s+(\d+)"),
    "eval_duration": re.compile(r"eval duration:\s+([\d\.]+)s"),
    "eval_rate": re.compile(r"eval rate:\s+([\d\.]+)"),
    "hit_rate": re.compile(r"hit rate:\s+([\d\.]+)"),
}

def run_experiment(cpu_infer, prefetch_num):
    cmd = [
        "python", "./ktransformers/local_chat.py",
        "--model_path", "/mnt/incontainer/models/DeepSeek-V3/DeepSeek-V3-0324-config/",
        "--gguf_path", "/mnt/incontainer/models/deepseek-ai_DeepSeek-V3-0324-GGUF/deepseek-ai_DeepSeek-V3-0324-IQ4_XS/",
        "--prompt_file", "./myprompt.txt",
        "--max_new_tokens", "100",
        "--cpu_infer", str(cpu_infer),
        "--prefetch_num", str(prefetch_num)
    ]

    # 捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # 提取指标
    metrics = {}
    for key, pattern in patterns.items():
        match = pattern.search(output)
        if match:
            metrics[key] = match.group(1)
        else:
            metrics[key] = None  # 没匹配到就写 None

    return metrics

# 写入 CSV
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for cpu_infer in cpu_infer_range:
        for prefetch_num in prefetch_num_range:
            print(f"Running: cpu_infer={cpu_infer}, prefetch_num={prefetch_num}")
            metrics = run_experiment(cpu_infer, prefetch_num)
            row = {
                "cpu_infer": cpu_infer,
                "prefetch_num": prefetch_num,
                "eval_count": metrics["eval_count"],
                "eval_duration": metrics["eval_duration"],
                "eval_rate": metrics["eval_rate"],
                "hit_rate": metrics["hit_rate"],
            }
            writer.writerow(row)