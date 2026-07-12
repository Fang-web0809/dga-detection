"""Character-level LSTM classifier."""
import torch
import torch.nn as nn

from common import VOCAB_SIZE


class CharLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=32, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        e = self.emb(x)
        out, (h, c) = self.lstm(e)
        return self.fc(h[-1]).squeeze(-1)  # (B,) logit
