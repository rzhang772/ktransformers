import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import ast
import numpy as np

class TopKPredictionDataset(Dataset):
    def __init__(self, df, input_dim=7168, vocab_size=256):
        self.vocab_size = vocab_size
        self.inputs = torch.tensor(df['hidden_states'].to_list(), dtype=torch.float32)
        self.labels_raw = torch.tensor(df['next_token_topk'].to_list(), dtype=torch.int64)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y_idx = self.labels_raw[idx]
        y = torch.zeros(self.vocab_size)
        y[y_idx] = 1.0  # multi-hot vector
        return x, y