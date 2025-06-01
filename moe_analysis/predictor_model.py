import torch
import torch.nn as nn
import torch.nn.functional as F

class TopkPredictor(nn.Module):
    def __init__(self, input_dim=7168, vocab_size=256, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        return self.net(x)  # shape: [batch_size, 256]
