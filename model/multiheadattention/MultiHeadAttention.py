import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Common
from model.Common import attention
from model.embedding.Embedding import Embedding
from model.position.PositionalEncoding import PositionalEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embedding_dim // head

        self.linears = Common.clone(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)


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
# head = 8
# embedding_dim = d_model = 512
# query = key = value = pe_result
# mask = Variable(torch.zeros(8, 4, 4))
#
# mha = MultiHeadAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print("MultiHeadAttention:", mha_result)
# print(mha_result.shape)
