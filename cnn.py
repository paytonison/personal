import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, data, targets, vocab_size, seq_length):
        self.data = data
        self.targets = targets
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


vocab_size = 5000
sq_length = 100
num_classes = 2

train_data = [[1, 2, 3, 4, 5] * 20 for _ in range(1000)]
