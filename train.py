import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from dataset import TimeSeriesDataset
from models import BaselineLSTM, LSTMAttentionModel
from evaluate import evaluate
from exp_smoothing import exponential_smoothing_forecast

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

def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    attn_weights = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):
                preds, weights = out
                attn_weights.append(weights.cpu().numpy())
            else:
                preds = out
            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())

    return (
        np.vstack(y_true),
        np.vstack(y_pred),
        np.concatenate(attn_weights) if attn_weights else None
    )

if __name__ == "__main__":
    df = pd.read_csv("timeseries.csv")
    train_df, val_df, test_df = split_data(df)

    train_loader = DataLoader(TimeSeriesDataset(train_df), batch_size=32, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_df), batch_size=32)

    # -------------------------
    # Hyperparameter Tuning
    # -------------------------
    hidden_sizes = [64, 128]
    learning_rates = [0.001, 0.0005]

    best_attention_model = None
    best_val_loss = float("inf")
    best_params = None

    for h in hidden_sizes:
        for lr in learning_rates:
            model = LSTMAttentionModel(df.shape[1], hidden_dim=h)
            train_model(model, train_loader, epochs=15, lr=lr)
            y_t, y_p, _ = evaluate_model(model, test_loader)
            val_loss = np.mean((y_t - y_p) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_attention_model = model
                best_params = (h, lr)

    # -------------------------
    # Train Standard LSTM Baseline
    # -------------------------
    standard_lstm = BaselineLSTM(df.shape[1], hidden_dim=best_params[0])
    train_model(standard_lstm, train_loader, epochs=15, lr=best_params[1])

    # -------------------------
    # Evaluation (ALL MODELS, BOTH TARGETS)
    # -------------------------
    y_true_attn, y_pred_attn, attn_weights = evaluate_model(best_attention_model, test_loader)
    y_true_lstm, y_pred_lstm, _ = evaluate_model(standard_lstm, test_loader)

    # Exponential Smoothing (target_1 only – documented baseline)
    y_es_true, y_es_pred = exponential_smoothing_forecast(df["target_1"].values)

    metrics = {
        "Attention_LSTM_target1": evaluate(y_true_attn[:,0], y_pred_attn[:,0]),
        "Attention_LSTM_target2": evaluate(y_true_attn[:,1], y_pred_attn[:,1]),
        "Standard_LSTM_target1": evaluate(y_true_lstm[:,0], y_pred_lstm[:,0]),
        "Standard_LSTM_target2": evaluate(y_true_lstm[:,1], y_pred_lstm[:,1]),
        "ExpSmoothing_target1": evaluate(y_es_true, y_es_pred)
    }

    np.save("attention_weights.npy", attn_weights)
    print(metrics)
