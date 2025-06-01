import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from pathlib import Path
import pandas as pd


num_layers = 58
num_experts = 256

# ---------- 工具函数 ----------
def load_data(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def plot_expert_activation_heatmaps(df, output_root='expert_activation_heatmaps', 
                                     max_tokens=1000, num_experts=256, 
                                     blur_sigma=5.0, plots_per_page=36):
    os.makedirs(output_root, exist_ok=True)

    # 遍历所有 layer
    for layer in tqdm(range(58), desc="Processing layers"):
        layer_df = df[df['layer_idx'] == layer]

        if layer_df.empty:
            continue

        # 创建当前层的输出文件夹
        layer_dir = os.path.join(output_root, f'layer_{layer}')
        os.makedirs(layer_dir, exist_ok=True)

        # 遍历每个 dataset 和 file_name 组合
        grouped = layer_df.groupby(['dataset', 'file_name'])
        heatmaps = {}

        for (dataset, file_name), group in grouped:
            # 初始化 token x expert 的矩阵
            heatmap = np.zeros((max_tokens, num_experts), dtype=np.float32)

            for _, row in group.iterrows():
                token_idx = int(row['token_idx'])
                topk_ids = row['topk_idx']
                for expert_id in topk_ids:
                    heatmap[token_idx, expert_id] += 1

            # 加高斯模糊
            heatmap_blurred = gaussian_filter(heatmap, sigma=blur_sigma)
            heatmaps[(dataset, file_name)] = heatmap_blurred

        # 将heatmaps分成每页36个子图
        items = list(heatmaps.items())
        for page_idx in range(0, len(items), plots_per_page):
            fig, axes = plt.subplots(6, 6, figsize=(24, 24))
            for ax, ((dataset, file_name), heatmap) in zip(axes.flat, items[page_idx:page_idx + plots_per_page]):
                sns.heatmap(heatmap, ax=ax, cbar=False)
                ax.set_title(f'{dataset} | {file_name}', fontsize=8)
                ax.set_xlabel('Expert ID')
                ax.set_ylabel('Token ID')
            # 填充多余的子图为空
            for ax in axes.flat[len(items[page_idx:page_idx + plots_per_page]):]:
                ax.axis('off')

            plt.tight_layout()
            out_path = os.path.join(layer_dir, f'{dataset}_{file_name[:-4]}layer_{layer}_page_{page_idx // plots_per_page}.png')
            plt.savefig(out_path)
            plt.close()


def main():
    input_base_dir = './moe_analysis/output_test/'
    output_base_dir = './moe_analysis/figures/'

    dataset_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

    all_data = pd.DataFrame(columns=['dataset', 'file_name', 'token_idx', 'layer_idx', 'topk_idx'])
    
    for dataset in tqdm(dataset_dirs, desc="Processing Datasets"):
        file_list = [f for f in os.listdir(os.path.join(input_base_dir, dataset)) if f.endswith('.json')]
        max_tokens = 0
        for file in file_list:
            input_path = os.path.join(input_base_dir, dataset, file)
            data = load_data(input_path)
            df = pd.DataFrame(data)
            df['dataset'] = dataset
            df['file_name'] = file
            df = df[['dataset', 'file_name', 'token_idx', 'layer_idx', 'topk_idx']]
            if len(df)/58 > max_tokens:
                max_tokens = int(len(df)/58)
            print(f"Loaded {len(df)} rows from {dataset}/{file}. Tokens: {len(df)/58}")
            all_data = pd.concat([all_data, df], ignore_index=True)
        #     break
        # break
    plot_expert_activation_heatmaps(all_data, output_root=output_base_dir, max_tokens=max_tokens)
    print(f"Total data: {len(all_data)} rows.")
    print(all_data.info())
            

        

if __name__ == "__main__":
    main()