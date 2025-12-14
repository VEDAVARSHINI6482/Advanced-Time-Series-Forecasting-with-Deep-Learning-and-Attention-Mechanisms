import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=40):
        self.X = data.values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_len]
        y = self.X[idx+self.seq_len, :2]  # target_1, target_2
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
