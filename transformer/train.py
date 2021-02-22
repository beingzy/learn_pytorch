import time
import math

import torch
import torch.nn as nn

from .model import TransformerModel
from .process_data import (
    tokenizer,
    vocab,
    device,
    bptt,
    train_data,
    val_data,
    test_data,
    get_batch,
)


ntokens = len(vocab.stoi)
emb_size = 200 # embedding dimension
nhid = 200 # dimension of the feedforward network model in `nn.TransformerEncoder`
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads
dropout = 0.2

model = TransformerModel(
    ntokens, emb_size, nhid, nlayers, nhead, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # initial learning rate, will update with `StepLR` over epochs
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model):
    # Turn on the train mode
    model.train()
    total_loss = 0
    start_time = time.time()
    src_mask = model.gen_square_subsequent_mask(bptt).to(device)

    def _gen_train_info():
        return ('| epoch {:3d} | {:5d}/{:5d} batches | ' + 
            'lr {:02.2f} | ms/batch {:5.2f} | ' + 
            'loss {:5.2f} | ppl {:8.2f}').format(
                epoch, 
                batch, 
                len(train_data) // bptt, 
                scheduler.get_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss))

    for batch, ii in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, ii)

        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.gen_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(_gen_train_info())
            total_loss = 0
            start_time = time.time()
        

def evaluate(model, data_src):
    model.eval()
    total_loss = 0
    src_mask = model.gen_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for ii in range(0, data_src.size(0) - 1, bptt):
            data, targets = get_batch(data_src, ii)
            if data.size(0) != bptt:
                src_mask = model.gen_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_src) - 1)


def _gen_line_separator():
    return "-" * 89


def _gen_epoch_info():
    return ('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' + 
        'valid ppl {:8.2f}').format(
            epoch, 
            (time.time() - epoch_start_time),
            val_loss, 
            math.exp(val_loss),
        )


best_val_loss = float("inf")
epochs = 3
best_model = None
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    print(_gen_line_separator())
    print(_gen_epoch_info())
    print(_gen_line_separator())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    
    scheduler.step()

# reporting best model performance:
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)