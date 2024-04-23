import torch.nn as nn
import torch
import copy

from EncoderDecoder import EncoderDecoder
from model.Decoder import Decoder
from model.DecoderLayer import DecoderLayer
from model.Encoder import Encoder
from model.EncoderLayer import EncoderLayer
from model.embedding.Embedding import Embedding
from model.feedforward.PositionalwiseFeedForward import PositionalwiseFeedForward
from model.generator.Generator import Generator
from model.multiheadattention.MultiHeadAttention import MultiHeadAttention
from model.position.PositionalEncoding import PositionalEncoding


def make_model(source_vocab, target_vocab, N=6, d_model=512,
               d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model, dropout)
    ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(position)),
        nn.Sequential(Embedding(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


source_vocab = 11
target_vocab = 11
N = 6

if __name__ == '__main__':
    print(torch.__version__)
    res = make_model(source_vocab, target_vocab, N)
    print(res)


