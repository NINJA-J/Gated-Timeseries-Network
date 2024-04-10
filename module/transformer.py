import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from torch.nn import Module

from .encoder import EncoderList


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False, ):
        super(Transformer, self).__init__()

        self.encoder_input = EncoderList(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, N=N,
                                         mask=mask, dropout=dropout)

        self.encoder_channel = EncoderList(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, N=N,
                                           dropout=dropout)

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        encoding_1 = self.embedding_channel(x)

        # if self.pe:
        #     pe = torch.ones_like(encoding_1[0])
        #     position = torch.arange(0, self._d_input).unsqueeze(-1)
        #     temp = torch.Tensor(range(0, self._d_model, 2))
        #     temp = temp * -(math.log(10000) / self._d_model)
        #     temp = torch.exp(temp).unsqueeze(0)
        #     temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        #     pe[:, 0::2] = torch.sin(temp)
        #     pe[:, 1::2] = torch.cos(temp)
        #
        #     encoding_1 = encoding_1 + pe
        with record_function("_train.e_input"):
            encoding_1, score_input = self.encoder_input(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        encoding_2 = self.embedding_input(x.transpose(-1, -2))

        with record_function("_train.e_channel"):
            encoding_2, score_channel = self.encoder_channel(encoding_2, stage)

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        return self.output_linear(encoding)
