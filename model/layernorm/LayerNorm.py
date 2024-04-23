import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Common
from model.embedding.Embedding import Embedding
from model.feedforward.PositionalwiseFeedForward import PositionalwiseFeedForward
from model.position.PositionalEncoding import PositionalEncoding


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


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
#
# features = d_model = 512
# eps = 1e-6
# ln = LayerNorm(features, eps)
# ln_result = ln(ff_result)
# print('LayerNorm:', ln_result)
# print(ln_result.shape)
