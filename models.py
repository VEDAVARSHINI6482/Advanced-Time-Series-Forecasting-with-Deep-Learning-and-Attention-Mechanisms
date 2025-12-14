import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        w = torch.softmax(self.score(h), dim=1)
        return torch.sum(w * h, dim=1), w

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h, _ = self.lstm(x)
        ctx, w = self.attn(h)
        return self.fc(ctx), w
