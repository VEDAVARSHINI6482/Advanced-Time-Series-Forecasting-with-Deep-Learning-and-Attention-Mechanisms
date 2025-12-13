import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from dataset import TimeSeriesDataset
from models import BaselineLSTM, LSTMAttentionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, loader, epochs=20, lr=0.001):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    df = pd.read_csv("timeseries.csv")
    dataset = TimeSeriesDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    baseline = BaselineLSTM(df.shape[1])
    attention_model = LSTMAttentionModel(df.shape[1])

    train_model(baseline, loader)
    train_model(attention_model, loader)

    torch.save(baseline.state_dict(), "baseline_lstm.pt")
    torch.save(attention_model.state_dict(), "attention_lstm.pt")
