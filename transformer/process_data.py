import io
import torch
from torchtext.utils import (
    download_from_url,
    extract_archive,
)
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(
    download_from_url(url))

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    map(tokenizer, iter(io.open(train_filepath, encoding='utf8')))
)


def data_process(raw_text_iter):
    data = [torch.tensor(
        [vocab[token] for token in tokenizer(item)], dtype=torch.long)
        for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz, device=device):
    n_batch = data.size(0) // bsz
    data = data.narrow(0, 0, n_batch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


bptt=35
def get_batch(src, i, bptt=bptt):
    seq_len = min(bptt, len(src) - 1 - i)
    data = src[i:i+seq_len]
    target = src[i+1:i+1+seq_len].reshape(-1)
    return data, target


train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
val_data = data_process(iter(io.open(val_filepath, encoding="utf8")))
test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
