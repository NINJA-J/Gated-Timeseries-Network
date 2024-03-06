import math

import torch
import torch.nn.functional as F
from torch.nn import Module


class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h).to(device)
        self.W_k = torch.nn.Linear(d_model, q * h).to(device)
        self.W_v = torch.nn.Linear(d_model, v * h).to(device)

        self.W_o = torch.nn.Linear(v * h, d_model).to(device)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0).to(self.device)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0).to(self.device)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0).to(self.device)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score
