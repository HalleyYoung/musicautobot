import torch
import torch.nn as nn
import pickle
import numpy as np
import random
import math
import time
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ninp=12, nhead=2, nhid=8, nlayers=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ninp)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = nn.Sigmoid()(self.decoder(output))
        return output

bptt = 20
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].view(seq_len, 12)
    target = source[i+1:i+1+seq_len].view(seq_len, 12)
    return data, target

"""
profiles = pickle.load(open("../pitchprofiles.pcl", "rb"))

dataset = []
for profile in profiles:
    index = 0
    batch = get_batch(torch.from_numpy(profile).float().softmax(1), index)
    while batch[0].shape[0] == bptt:
        dataset.append(batch)
        index += 1
        batch = get_batch(torch.from_numpy(profile).float().softmax(1), index)


print(len(dataset))
model = TransformerModel()

criterion = nn.BCELoss(reduction="mean")
lr = 5e-1 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)


random.shuffle(dataset)

trainloader = DataLoader(dataset[:-1000], shuffle = True, batch_size = 8)
testloader = DataLoader(dataset[-1000:], shuffle = True, batch_size = 8)


for epoch in range(2000):
    model.train()
    total_loss = 0
    start_time = time.time()
    for (batch_ind, i) in enumerate(trainloader):
        (data, targets) = i 
        optimizer.zero_grad()
        output = model(data)
        #print((data.shape, output.shape))
        loss = criterion(output.view(-1, 12), targets.view(-1,12))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch_ind % log_interval == 0 and batch_ind > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_ind, len(trainloader) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            torch.save(model.state_dict(), "transformernn/predict_space_ref_ahead" + str(epoch) + ".pth")
    scheduler.step()
"""
