import torch
import torch.nn as nn
from torch.autograd import Variable

from model.multiheadattention.MultiHeadAttention import MultiHeadAttention
from model.embedding.Embedding import Embedding
from model.layernorm.LayerNorm import LayerNorm
from model.position.PositionalEncoding import PositionalEncoding


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


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
# size = d_model = 512
# head = 8
# mask = Variable(torch.zeros(8, 4, 4))
# self_attn = MultiHeadAttention(head, d_model)
# sublayer = lambda x: self_attn(x, x, x, mask)
#
# sc = SubLayerConnection(size, dropout)
# sc_result = sc(pe_result, sublayer)
# print('SubLayerConnection:', sc_result)
# print(sc_result.shape)
