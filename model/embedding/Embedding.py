import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.lut = nn.Embedding(vocab, d_model)

    def forward(self, x):
        y = self.lut(x) * math.sqrt(self.d_model)
        # print(y)
        # print(y.shape)
        return y


# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# d_model = 512
# vocab = 1000
# emb = Embedding(d_model, vocab)
# res = emb(x)
# print(res)
# print(res.shape)