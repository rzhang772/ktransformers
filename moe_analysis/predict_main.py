import os
import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
import ast
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt

from predictor_dataset import TopKPredictionDataset
from predictor_model import TopkPredictor

# print(pd.__version__)
num_layers = 58
num_experts = 256

def load_data(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


def add_next_token_topk(df):
    '''
    为 DataFrame 添加 next_token_topk 列，表示每个 token 在下一行的 topk_idx。'''
    # 对 layer 分组处理
    def process_group(group):
        group = group.sort_values('token_idx').reset_index(drop=True)
        group['next_token_topk'] = group['topk_idx'].shift(-1)
        return group[:-1]
    
    # 对每个 layer_idx 分组并处理
    df_processed = df.groupby('layer_idx', group_keys=False).apply(process_group)
    return df_processed

def get_data_for_layer(base_dir, dataset_name, layer_idx = None):
    '''
    从指定目录加载数据,生成next_token_topk_idx, 并返回指定层的数据。
    base_dir: 数据所在的基础目录
    dataset_name: 数据集名称列表
    layer_idx: 指定的层索引，如果为 None,则返回所有层的数据
    '''
    all_data = pd.DataFrame(columns=['dataset', 'file_name', 'mode', 'token_idx', 'layer_idx', 'hidden_states', 'topk_idx', 'next_token_topk'])
    for dataset in dataset_name:
        dataset_dir = os.path.join(base_dir, dataset)
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
        print(f"Processing dataset: {dataset}, found {len(files)} files.")

        for file in files:
            print(f"Processing {dataset} file: {file}")
            file_path = os.path.join(dataset_dir, file)
            data = load_data(file_path)
            df = pd.DataFrame(data)
            df['dataset'] = dataset
            df['file_name'] = file

            print(f"Loaded {len(df)} rows from {dataset}/{file}. Tokens: {len(df)/num_layers}")

            # 添加 next_token_topk 列
            df = add_next_token_topk(df)
            all_data = pd.concat([all_data, df], ignore_index=True)
            # break
    print(f"Total data: {len(all_data)} rows.")
    print(all_data.info())
    if layer_idx is not None:
        # 过滤出指定层的数据
        layer_data = all_data[all_data['layer_idx'] == layer_idx].reset_index(drop=True)
        print(f"Filtered data for layer {layer_idx}: {len(layer_data)} rows.")
        return layer_data
    else:
        print("No layer index specified, returning all data.")
        return all_data


from torch.utils.data import Dataset, DataLoader, random_split
def prepare_dataloaders(df, batch_size=64, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    dataset = TopKPredictionDataset(df)
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def record_prediction_errors(model, dataloader, device, writer, epoch, top_k=8):
    model.eval()
    wrong_tokens = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            hidden_states = batch_x.to(device)
            true_labels = batch_y.cpu().numpy()
            logits = model(hidden_states)
            preds = torch.topk(logits, k=top_k, dim=-1).indices.cpu().numpy()

            for i in range(len(true_labels)):
                true_indices = set(np.where(true_labels[i] == 1)[0])
                pred_indices = set(preds[i].flatten())
                wrong = pred_indices - true_indices
                wrong_tokens.extend(list(wrong))

    if wrong_tokens:
        token_counts = Counter(wrong_tokens)
        labels, values = zip(*token_counts.most_common(10))

        plt.figure(figsize=(8, 4))
        plt.bar(range(len(values)), values, tick_label=labels)
        plt.title(f"Top Wrong Predictions @ Epoch {epoch}")
        plt.xlabel("Token Index")
        plt.ylabel("Frequency")

        # 写入 TensorBoard
        writer.add_figure("Wrong_Token_Distribution", plt.gcf(), global_step=epoch)
        plt.close()

def train_model(layer_idx, model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, 
                epochs=10, 
                log_dir="./moe_analysis/runs/llm_style_topk", 
                early_stopping_patience=4):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_recall = 0.0
    best_epoch = 0
    best_model = None

    early_stop_counter = 0
    last_val_recall = 0.0

    global_step = 0 # 即batch步数
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            hidden_states = batch_x.to(device)
            labels = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(hidden_states).squeeze(1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            running_loss += loss.item()
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        train_recall = evaluate(model, train_loader, device)
        val_recall = evaluate(model, val_loader, device)
        test_recall = evaluate(model, test_loader, device)

        # 记录最佳模型
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch
            best_model = model.state_dict()
        
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Recall@8/train", train_recall, epoch)
        writer.add_scalar("Recall@8/val", val_recall, epoch)
        writer.add_scalar("Recall@8/test", test_recall, epoch)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_train_loss:.4f}, Val R@8={val_recall:.4f}, Test R@8={test_recall:.4f}")

        # 早停机制
        if epoch > 0 and val_recall < last_val_recall:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stopping_patience}")
            if early_stop_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
        else:
            early_stop_counter = 0
        last_val_recall = val_recall
        # 可选：记录错误 token 分布
        # record_prediction_errors(model, val_loader, device, writer, epoch)
    # 保存最佳模型
    if best_model is not None:
        model_path = os.path.join(log_dir, f"best_model_layer_{layer_idx}.pth")
        torch.save(best_model, model_path)
        print(f"Best model saved at {model_path} (Epoch {best_epoch+1}, Val R@8={best_val_recall:.4f})")
    writer.close()

def evaluate(model, dataloader, device, top_k=32):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)  # [B, 256]
            topk_preds = torch.topk(logits, k=top_k, dim=-1).indices  # [B, 8] top-k 预测索引

            for i in range(batch_x.size(0)):
                true_labels = torch.nonzero(batch_y[i]).flatten() # 正类标签索引
                pred_labels = topk_preds[i].flatten()  # 预测的 top-k 标签索引
                # print(f"True labels: {true_labels.tolist()}, Predicted labels: {pred_labels.tolist()}")
                hits = len(set(true_labels.tolist()) & set(pred_labels.tolist()))
                correct += hits
                total += len(true_labels)

    recall_at_k = correct / total if total > 0 else 0.0
    return recall_at_k

def main():
    base_dir = "./moe_analysis/outputs/"
    dataset_name = [
        # "testset",
        "flores_101", 
        # "gsm8k", 
        # "humaneval", 
        # "triviaqa", 
        # "xsum"
        ]
    all_data = get_data_for_layer(base_dir, dataset_name)
    print(f"Total data loaded: {len(all_data)} rows.")


    for layer_idx in range(num_layers):
        # layer_idx = 10  # 指定要处理的层索引
        
        layer_data = all_data[all_data['layer_idx'] == layer_idx].reset_index(drop=True)
        print(f"Layer {layer_idx} data: {len(layer_data)} rows.")

        train_loader, val_loader, test_loader = prepare_dataloaders(layer_data, batch_size=64)
        
        # 初始化模型
        input_dim = 7168  # 输入维度
        vocab_size = 256  # 词汇表大小
        model = TopkPredictor(input_dim=input_dim, vocab_size=vocab_size)
        print(model)
        # 训练模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        
        train_model(layer_idx, model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, epochs=500, log_dir=f"./moe_analysis/runs/llm_style_topk/layer_{layer_idx}")
    

if __name__ == "__main__":
    main()