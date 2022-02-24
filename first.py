# Train a transformer from scratch on FIRST, using fixed positional encodings.

import torch
import encoder
import math
import random
import sys
import argparse

log_sigmoid = torch.nn.LogSigmoid()

class PositionEncoding(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, n):
        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pe = torch.stack([pos == 1] + [zero]*(self.size-1), dim=1)
        return pe

ap = argparse.ArgumentParser()
ap.add_argument('--train_length', type=int, default=50)
ap.add_argument('--test_length', type=int, default=1000)
ap.add_argument('--trial', type=int, default=0)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--layers', dest='layers', type=int, default=2)
ap.add_argument('--heads', dest='heads', type=int, default=1)
ap.add_argument('--d_model', type=int, default=16)
ap.add_argument('--d_ffnn', type=int, default=64)
ap.add_argument('--scaled', type=bool, default=False, help='log-length scaled attention')
ap.add_argument('--eps', type=float, default=1e-5, help='Value added to denominator in layer normalization')
args = ap.parse_args()

alphabet = ["0", "1", "$"]
alphabet_index = {a:i for i,a in enumerate(alphabet)}

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        super().__init__()
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)
        self.pos_encoding = PositionEncoding(d_model)

        if scaled:
            encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = encoder.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_layer = torch.nn.Linear(d_model, 1)

    def forward(self, w):
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        y = y[0]
        z = self.output_layer(y)
        return z

model = Model(len(alphabet), args.layers, args.heads, args.d_model, args.d_ffnn, args.scaled, args.eps)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(args.epochs):
    train_loss = train_correct = train_num = 0
    train_alpha = torch.zeros(2)
    for step in range(args.steps):
        n = args.train_length
        w = torch.tensor([alphabet_index['$']] + [alphabet_index[str(random.randrange(2))] for i in range(n)])
        label = w[1] == alphabet_index['1']
        output = model(w)
        for l in range(2):
            train_alpha[l] += model.encoder.layers[l].last_weights[0][0][1].detach()
        if not label: output = -output
        if output > 0: train_correct += 1
        train_num += 1
        loss = -log_sigmoid(output)
        train_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        test_loss = test_num = test_correct = 0
        test_alpha = torch.zeros(len(model.encoder.layers))
        for step in range(args.steps):
            n = args.test_length
            w = torch.tensor([2] + [random.randrange(2) for i in range(n)])
            label = w[1] == alphabet_index['1']
            output = model(w)
            for l, layer in enumerate(model.encoder.layers):
                # weight of CLS attending to first symbol
                test_alpha[l] += model.encoder.layers[l].last_weights[0][0][1]
            if not label: output = -output
            if output > 0: test_correct += 1
            test_num += 1
            loss = -log_sigmoid(output)
            test_loss += loss.item()

    print(args.train_length,
          args.test_length,
          args.trial+1,
          epoch+1,
          train_loss,
          train_correct/train_num,
          ' '.join(str(a.item()/test_num) for a in train_alpha),
          test_loss,
          test_correct/test_num,
          ' '.join(str(a.item()/test_num) for a in test_alpha),
          flush=True)
