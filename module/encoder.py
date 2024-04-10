import torch
from torch.nn import Module, ModuleList

from .feedForward import FeedForward
from .multiHeadAttention import MultiHeadAttention


class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = False,
                 dropout: float = 0.1,
                 sub_name="none"):
        super(Encoder, self).__init__()

        self.sub_name = sub_name
        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, dropout=dropout, sub_name=sub_name)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden, sub_name=sub_name)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):
        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class EncoderList(Module):
    def __init__(self, d_model, d_hidden, q, v, h, N, mask=False, dropout: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list = ModuleList([Encoder(d_model=d_model,
                                        d_hidden=d_hidden,
                                        q=q,
                                        v=v,
                                        h=h,
                                        mask=mask,
                                        dropout=dropout) for _ in range(N)])

    def forward(self, x, stage):
        score = None
        for m in self.list:
            x, score = m(x, stage)
        return x, score
