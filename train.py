import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset import TimeSeriesDataset
from models import LSTMAttentionModel
from evaluate import evaluate
from exp_smoothing import exponential_smoothing_forecast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Data Split
# -------------------------------
def split_data(df):
    n = len(df)
    train = df[:int(0.7 * n)]
    val = df[int(0.7 * n):int(0.85 * n)]
    test = df[int(0.85 * n):]
    return train, val, test

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, loader, epochs=25, lr=0.001):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss/len(loader):.4f}")

# -------------------------------
# Main Pipeline
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("timeseries.csv")

    train_df, val_df, test_df = split_data(df)

    train_loader = DataLoader(TimeSeriesDataset(train_df), batch_size=32, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_df), batch_size=32)

    model = LSTMAttentionModel(input_dim=df.shape[1])
    train_model(model, train_loader)

    # -------------------------------
    # Deep Learning Evaluation
    # -------------------------------
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds, _ = model(x)
            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    dl_metrics = evaluate(y_true[:, 0], y_pred[:, 0])
    print("Attention LSTM Metrics:", dl_metrics)

    # -------------------------------
    # Exponential Smoothing Baseline
    # -------------------------------
    true_es, pred_es = exponential_smoothing_forecast(df["target_1"].values)
    es_metrics = evaluate(true_es, pred_es)
    print("Exponential Smoothing Metrics:", es_metrics)
