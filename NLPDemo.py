import math
import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.nn.modules.multiheadattention import generate_square_subsequent_mask
from torchtext.vocab import build_vocab_from_iterator

import ModelMain

train_iter = WikiText2("G:\\project\\transformer_model_demo", split='train')  # 训练数据迭代器
tokenizer = get_tokenizer('basic_english')  # 基本的英文分词器
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])  # 设置单词索引，当某个单词不在词汇表中，则返回0


# print(vocab(tokenizer('here is an example')))   # [1291, 23, 30, 617]


def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# print(data_process(['here is an example']))  # tensor([1291,   23,   30,  617])

# train_iter在构建词表的时候用掉了，这里再次创建
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data, bsz):
    """
    分割数据，并且移除多余数据
    :param data: Tensor, shape [N] 文本数据 train_data、val_data、test_data
    :param bsz: int, batch_size，每次模型更新参数的数据量
    :return: Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # [seq_len, batch_size] 句子是竖着的
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# print(train_data.shape) # torch.Size([102499, 20])
# 每个句子长度为102499，明显不科学，我们要限制句子的长度

bptt = 35  # 句子最大长度


def get_batch(source, i):
    """
    :param source: Tensor, shape [full_seq_len, batch_size]
    :param i: 批次数
    :return: tuple (data, target), where data has shape [seq_len, batch_size] and
             target has shape [seq_len * batch_size]
    """
    # 前面的批次都会是bptt的值, 只不过最后一个批次中
    # 句子长度可能不够bptt的35个, 因此会变为len(source) - 1 - i的值
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


source = test_data
i = 1
data, target = get_batch(source, i)
print(data.shape)  # torch.Size([35, 10])
print(target.shape)  # to# 超参数定义
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
# model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)rch.Size([350])
model = ModelMain.make_model(11, 11, N=2)


import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def train(model):
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data):
    model.eval()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')  # 保存最低的loss
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)  # 困惑度，越低越好
    elapsed = time.time() - epoch_start_time

    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()

