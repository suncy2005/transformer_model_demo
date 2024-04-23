import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext.data.utils import get_tokenizer
from pyitcast.transformer import TransformerModel

TEXT = torchtext.data.Field(tokenizer=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            oes_token='<eos>',
                            lower=True)
print(TEXT)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
print(test_txt.examples[0].text[:10])

TEXT.build_vocab(train_txt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    data = TEXT.numbericalize([data.examples[0].text])
    nbatch = data.size(0) / bsz
    data = data.narrow(0, 0, nbatch*bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, batch_size)
test_data = batchify(test_txt, batch_size)


bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target




