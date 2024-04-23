import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Common
from model.SubLayerConnection import SubLayerConnection
from model.embedding.Embedding import Embedding
from model.feedforward.PositionalwiseFeedForward import PositionalwiseFeedForward
from model.multiheadattention.MultiHeadAttention import MultiHeadAttention
from model.position.PositionalEncoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = Common.clone(SubLayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# size = 512
# head = 8
# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# d_model = 512
# max_len = 1000
# d_ff = 64
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
# dropout = 0.2
# self_attn = MultiHeadAttention(head, d_model)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))
#
# el = EncoderLayer(size, self_attn, ff, dropout)
# el_result = el(pe_result, mask)
# print(el_result)
# print(el_result.shape)