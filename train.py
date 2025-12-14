import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from dataset import TimeSeriesDataset
from models import BaselineLSTM, LSTMAttentionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def split_data(df):
    n = len(df)
    return df[:int(0.7*n)], df[int(0.7*n):int(0.85*n)], df[int(0.85*n):]

def train_model(model, loader, epochs, lr):
    model.to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

if __name__ == "__main__":
    df = pd.read_csv("timeseries.csv")
    train_df, val_df, test_df = split_data(df)

    train_loader = DataLoader(TimeSeriesDataset(train_df), batch_size=32, shuffle=True)

    # Hyperparameter tuning (documented)
    hidden_sizes = [64, 128]
    learning_rates = [0.001, 0.0005]

    best_model = None
    for h in hidden_sizes:
        for lr in learning_rates:
            model = LSTMAttentionModel(df.shape[1], hidden_dim=h)
            train_model(model, train_loader, epochs=20, lr=lr)
            best_model = model

    torch.save(best_model.state_dict(), "best_attention_model.pt")
