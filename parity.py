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
ap.add_argument('--size', dest='size', type=int, default=64)
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, size):
        super().__init__()
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=size)
        n_max = max(args.train_length, args.test_length)
        self.pos_embedding = torch.stack([
            torch.arange(0, n_max+1, dtype=torch.float),
            torch.cos(torch.arange(0, n_max+1, dtype=torch.float)*math.pi)
        ], dim=1)
        self.pos_adapter = torch.nn.Linear(self.pos_embedding.size()[1], size)

        encoder_layer = encoder.PostnormTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.SigmoidTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_layer = torch.nn.Linear(size, 1)

    def forward(self, w):
        p = self.pos_embedding[:len(w)].clone()
        p[:,0] /= len(w)
        x = self.word_embedding(w) + self.pos_adapter(p)
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        y = y[-1]
        return self.output_layer(y)

model = Model(3, args.size)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)

best_train_loss = float('inf')
no_improvement = 0

for epoch in range(args.epochs):
    train_loss = 0
    train_steps = 0
    train_correct = 0
    
    for step in range(args.steps):
        n = args.train_length
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        """n0 = min(n, random.randrange(0, n0_max+1))
        w = [1]*n0 + [0]*(n-n0)
        random.shuffle(w)
        w.append(2)
        w = torch.tensor(w)"""
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

    """if train_loss < best_train_loss:
        best_train_loss = train_loss
        no_improvement = 0
    else:
        no_improvement += 1
        if False and no_improvement >= 10:
            optim.param_groups[0]['lr'] *= 0.5
            print(f"lr={optim.param_groups[0]['lr']}")
            no_improvement = 0"""

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

    """if n0_max < n_max:
        #if train_loss/train_steps/math.log(2) < 0.1:
        if train_correct/train_steps > 0.9:
            n0_max += 1
            no_improvement = 0
            #optim = torch.optim.Adam(model.parameters(), lr=0.0003)
            #optim = torch.optim.Adagrad(model.parameters(), lr=0.001) # reset optim"""
