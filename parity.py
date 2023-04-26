import torch
import math
import random
import encoder
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--train_length', dest='train_length', type=int, default=100)
ap.add_argument('--test_length', dest='test_length', type=int, default=100)
ap.add_argument('--epochs', dest='epochs', type=int, default=100)
ap.add_argument('--steps', dest='steps', type=int, default=100)
ap.add_argument('--layers', dest='layers', type=int, default=2)
ap.add_argument('--heads', dest='heads', type=int, default=2)
ap.add_argument('--d_model', type=int, default=16)
ap.add_argument('--d_ffnn', type=int, default=64)
ap.add_argument('--scaled', type=bool, default=False, help='log-length scaled attention')
ap.add_argument('--eps', type=float, default=1e-5, help='Value added to denominator in layer normalization')
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

class PositionEncoding(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size % 2 == 0
        self.size = size
        self.scales = torch.nn.Parameter(torch.normal(0, 1., (size,)))

    def forward(self, n):
        p = torch.arange(0, n).to(torch.float).unsqueeze(1)
        pe = torch.cat([
            p / n * torch.exp(self.scales[:self.size//2]),
            torch.cos(p*math.pi * torch.exp(self.scales[self.size//2:])),
        ], dim=1)
        return pe

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        super().__init__()

        self.pos_encoding = PositionEncoding(d_model)
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)

        if scaled:
            encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = encoder.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_layer = torch.nn.Linear(d_model, 1)

    def forward(self, w):
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[-1])
        return z

model = Model(3, args.layers, args.heads, args.d_model, args.d_ffnn, args.scaled, args.eps)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(args.epochs):
    train_loss = 0
    train_steps = 0
    train_correct = 0
    
    for step in range(args.steps):
        n = args.train_length
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        label = len([a for a in w if a == 1]) % 2 == 1
        output = model(w)
        if not label: output = -output
        if output > 0: train_correct += 1
        loss = -log_sigmoid(output)
        train_loss += loss.item()
        train_steps += 1
        optim.zero_grad()
        loss.backward()
        optim.step()

    test_loss = 0
    test_steps = 0
    test_correct = 0
    for step in range(args.steps):
        n = args.test_length
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        label = len([a for a in w if a == 1]) % 2 == 1
        output = model(w)
        if not label: output = -output
        if output > 0: test_correct += 1
        loss = -log_sigmoid(output)
        test_loss += loss.item()
        test_steps += 1
        
    print(f'train_length={args.train_length} train_ce={train_loss/train_steps/math.log(2)} train_acc={train_correct/train_steps} test_ce={test_loss/test_steps/math.log(2)} test_acc={test_correct/test_steps}', flush=True)
