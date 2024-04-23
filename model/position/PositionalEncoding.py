import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.embedding.Embedding import Embedding


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# d_model = 512
# max_len = 1000
#
# em = Embedding(d_model, max_len)
# em_result = em(x)
# print('Embedding:', em_result)
# print(em_result.shape)
#
# dropout = 0.1
# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(em_result)
# print('PositionalEncoding:', pe_result)
# print(pe_result.shape)
