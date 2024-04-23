import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import Common
from model.embedding.Embedding import Embedding
from model.position.PositionalEncoding import PositionalEncoding


class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


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
#
# query = key = value = pe_result
# attn, p_attn = Common.attention(query, key, value)
# print(attn)
# print(attn.shape)
# print('*****')
# print(p_attn)
# print(p_attn.shape)
#
# d_ff = 64
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# ff_result = ff(pe_result)
# print('PositionalwiseFeedForward:', ff_result)
# print(ff_result.shape)
