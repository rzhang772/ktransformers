import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------- 工具函数 ----------

def load_data(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def organize_data(data, num_layers=58):
    token_data = defaultdict(lambda: [None] * num_layers)
    for entry in data:
        token = entry['token_idx']
        layer = entry['layer_idx']
        experts = entry['topk_idx'][0]
        token_data[token][layer] = experts
    return token_data

# ---------- 图1：Token-wise Heatmap ----------

def plot_token_expert_heatmaps(dataset_name, file_name, start_id, tokens_per_figure, n_cols, token_data_dict, tokens, num_layers, num_experts, smooth=False, out_path=None):
    fig, axes = plt.subplots(6, 6, figsize=(20, 16))
    axes = axes.flatten()

    for idx, token in enumerate(tokens):
        ax = axes[idx]
        heatmap = np.zeros((num_layers, num_experts))
        for layer in range(num_layers):
            experts = token_data_dict[token][layer]
            if experts is not None:
                for expert in experts:
                    heatmap[layer, expert] = 1
        if smooth:
            heatmap = gaussian_filter(heatmap, sigma=5)
        sns.heatmap(heatmap, ax=ax, cmap='Blues', cbar=False)
        ax.set_title(f"Token {token}")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Layer ID")

    # 删除多余子图
    for j in range(len(tokens), len(axes)):
        fig.delaxes(axes[j])

    # 设置标题
    fig.suptitle(f"{dataset_name}_{file_name}_Token_Wise_Heatmap ({start_id} to {start_id + len(tokens) - 1}) (Smoothed)" if smooth else f"{dataset_name}_{file_name}_Token_Wise_Heatmap ({start_id} to {start_id + len(tokens) - 1})", fontsize=18)
    # fig.suptitle(f"Expert Activations per Token ({dataset_name}) (Smoothed)" if smooth else f"Expert Activations per Token ({dataset_name})", fontsize=18)
    # fig.suptitle("Expert Activations per Token (Smoothed)" if smooth else "Expert Activations per Token", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    output_path = os.path.join(out_path, f"{dataset_name}_{file_name[:-5]}_token_wise_{start_id}_{start_id + len(tokens) - 1}.png")
    plt.savefig(output_path)
    plt.close()

def plot_token_wise_heatmap(input_path, out_path):
    data = load_data(input_path)
    num_layers = 58
    num_experts = 256
    tokens_per_figure = 36
    n_cols = 6

    token_data = organize_data(data, num_layers=num_layers)
    all_tokens = sorted(token_data.keys())
    file_name = os.path.basename(input_path)
    dataset_name = os.path.basename(os.path.dirname(input_path))

    for i in tqdm(range(0, len(all_tokens), tokens_per_figure), desc="Token-wise Heatmaps"):
        tokens_batch = all_tokens[i:i+tokens_per_figure]
        plot_token_expert_heatmaps(dataset_name, file_name, i, tokens_per_figure, n_cols, token_data, tokens_batch, num_layers, num_experts, smooth=True, out_path=out_path)

# ---------- 图2：Layer-wise Heatmap ----------

def organize_heatmap_per_layer(data, num_layers=58, num_experts=256):
    layer_maps = defaultdict(lambda: defaultdict(list))
    for entry in data:
        token = entry['token_idx']
        layer = entry['layer_idx']
        experts = entry['topk_idx'][0]
        layer_maps[layer][token] = experts

    all_tokens = sorted({entry['token_idx'] for entry in data})
    num_tokens = max(all_tokens) + 1

    layer_matrices = {}
    for layer in range(num_layers):
        mat = np.zeros((num_tokens, num_experts))
        for token in range(num_tokens):
            experts = layer_maps[layer].get(token, [])
            for e in experts:
                mat[token, e] = 1
        layer_matrices[layer] = mat

    return layer_matrices, all_tokens

def plot_layerwise_heatmaps(dataset_name, file_name, layer_matrices, smooth=False, out_path=None):
    num_layers = len(layer_matrices)
    ncols = 6
    nrows = int(np.ceil(num_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        mat = layer_matrices[layer_idx]
        if smooth:
            mat = gaussian_filter(mat, sigma=5)
        sns.heatmap(mat, ax=ax, cmap='Blues', cbar=False)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Token ID")

    for j in range(num_layers, len(axes)):
        fig.delaxes(axes[j])

    # 设置标题
    fig.suptitle(f"{dataset_name}_{file_name}_Layer_Wise_Heatmap (Smooth)" if smooth else f"{dataset_name}_{file_name}_Layer_Wise_Heatmap", fontsize=18)
    # fig.suptitle(f"Expert Activations per Layer ({dataset_name}) (Smooth)" if smooth else f"Expert Activations per Layer ({dataset_name})", fontsize=18)
    # fig.suptitle("Expert Activations per Layer (Smooth)" if smooth else "Expert Activations per Layer", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    output_path = os.path.join(out_path, f"{dataset_name}_{file_name[:-5]}_layer_wise_heatmap.png")
    plt.savefig(output_path)
    plt.close()

def plot_layer_wise_heatmaps(input_path, out_path=None):
    data = load_data(input_path)
    num_layers = 58
    num_experts = 256

    dataset_name = os.path.basename(os.path.dirname(input_path))
    file_name = os.path.basename(input_path)
    layer_matrices, _ = organize_heatmap_per_layer(data, num_layers=num_layers, num_experts=num_experts)
    plot_layerwise_heatmaps(dataset_name, file_name, layer_matrices, smooth=True, out_path=out_path)

# ---------- 主程序入口 ----------

def main():
    input_base_dir = './moe_analysis/outputs/'
    output_base_dir = './moe_analysis/figures/'

    dataset_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

    for dataset in tqdm(dataset_dirs, desc="Processing Datasets"):
        dataset_output_dir = os.path.join(output_base_dir, dataset)
        token_wise_dir = os.path.join(dataset_output_dir, 'token_wise')
        layer_wise_dir = os.path.join(dataset_output_dir, 'layer_wise')
        
        os.makedirs(token_wise_dir, exist_ok=True)
        os.makedirs(layer_wise_dir, exist_ok=True)

        file_list = [f for f in os.listdir(os.path.join(input_base_dir, dataset)) if f.endswith('.json')]

        for file in tqdm(file_list, desc=f"[{dataset}] Files", leave=False):
            input_path = os.path.join(input_base_dir, dataset, file)

            # 绘图
            plot_token_wise_heatmap(input_path, token_wise_dir)
            plot_layer_wise_heatmaps(input_path, layer_wise_dir)

        print(f"完成绘图：{dataset}，输出目录：{token_wise_dir} 和 {layer_wise_dir}")

if __name__ == "__main__":
    main()