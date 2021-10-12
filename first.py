import torch
import encoder
import math
import random
import sys
import argparse

log_sigmoid = torch.nn.LogSigmoid()

ap = argparse.ArgumentParser()
ap.add_argument('--train_length', type=int, default=50)
ap.add_argument('--test_length', type=int, default=1000)
ap.add_argument('--trial', type=int, default=0)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--steps', type=int, default=100)
args = ap.parse_args()

alphabet = ["0", "1", "$"]
alphabet_index = {a:i for i,a in enumerate(alphabet)}
max_pos = 10000
size = 16

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, size):
        super().__init__()
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=size)
        self.pos_embedding = torch.stack([
            torch.arange(0, max_pos, dtype=torch.float) == 0,
            torch.arange(0, max_pos, dtype=torch.float) == 1,
            torch.arange(0, max_pos, dtype=torch.float) >= 2,
        ], dim=1).to(torch.float)
        self.pos_adapter = torch.nn.Linear(self.pos_embedding.size()[1], size)

        encoder_layer = encoder.PostnormTransformerEncoderLayer(d_model=size, nhead=1, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=size, nhead=1, dim_feedforward=size*4, dropout=0.)
        #encoder_layer.norm1.eps = encoder_layer.norm2.eps = 0.
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_layer = torch.nn.Linear(size, 1)

    def forward(self, w):
        x = self.word_embedding(w) + self.pos_adapter(self.pos_embedding[:len(w)])
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        y = y[0]
        z = self.output_layer(y)
        return z

model = Model(len(alphabet), size)
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
