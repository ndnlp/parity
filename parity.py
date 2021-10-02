import torch
import math
import random
import encoder

alphabet = ["0", "1", "$"]
max_pos = 10000
size = 16

log_sigmoid = torch.nn.LogSigmoid()

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, size):
        super().__init__()
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=size)
        self.pos_embedding = torch.stack([
            torch.arange(0, max_pos, dtype=torch.float),
            torch.cos(torch.arange(0, max_pos, dtype=torch.float)*math.pi)
        ], dim=1)
        self.pos_adapter = torch.nn.Linear(self.pos_embedding.size()[1], size)

        #encoder_layer = torch.nn.TransformerEncoderLayer(d_model=size, nhead=2, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=size, nhead=2, dim_feedforward=size*4, dropout=0.)
        encoder_layer = encoder.SigmoidTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_layer = torch.nn.Linear(size, 1)

    def forward(self, w):
        p = self.pos_embedding[:len(w)].clone()
        #p[:,0] /= len(w)
        x = self.word_embedding(w) + self.pos_adapter(p)
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        y = y[-1]
        return self.output_layer(y)

model = Model(len(alphabet), size)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)
#optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
#optim = torch.optim.SGD(model.parameters(), lr=0.001)

best_epoch_loss = float('inf')
no_improvement = 0
n0_max = 0
n_max = 100

for epoch in range(10000):
    epoch_loss = 0
    epoch_steps = 0
    epoch_correct = 0
    for step in range(1000):
        n = random.randrange(1, n_max+1)
        n0 = min(n, random.randrange(0, n0_max+1))
        w = [0]*n0 + [1]*(n-n0)
        random.shuffle(w)
        w.append(2)
        w = torch.tensor(w)
        o = len([a for a in w if a == 1]) % 2 == 1
        #print(''.join(alphabet[a] for a in w), o)
        p = model(w)
        if not o: p = -p
        if p > 0: epoch_correct += 1
        # cross-entropy
        loss = -log_sigmoid(p)
        # perceptron
        #loss = torch.maximum(-p, torch.Tensor([0.]))
        epoch_loss += loss.item()
        epoch_steps += 1
        optim.zero_grad()
        #(loss*n).backward()
        loss.backward()
        optim.step()

    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        no_improvement = 0
    else:
        no_improvement += 1
        if False and no_improvement >= 10:
            optim.param_groups[0]['lr'] *= 0.5
            print(f"lr={optim.param_groups[0]['lr']}")
            no_improvement = 0

    valid_loss = 0
    valid_steps = 0
    valid_correct = 0
    for step in range(100):
        n = random.randrange(1, n_max+1)
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        o = len([a for a in w if a == 1]) % 2 == 1
        #print(''.join(alphabet[a] for a in w), o)
        p = model(w)
        if not o: p = -p
        if p > 0: valid_correct += 1
        # cross-entropy
        loss = -log_sigmoid(p)
        # perceptron
        #loss = torch.maximum(-p, torch.Tensor([0.]))
        valid_loss += loss.item()
        valid_steps += 1
        
    print(f'n0_max={n0_max} train_ce={epoch_loss/epoch_steps/math.log(2)} train_acc={epoch_correct/epoch_steps} valid_ce={valid_loss/valid_steps/math.log(2)} valid_acc={valid_correct/valid_steps}')

    if n0_max < n_max:
        #if epoch_loss/epoch_steps/math.log(2) < 0.1:
        if epoch_correct/epoch_steps > 0.9:
            n0_max += 1
            no_improvement = 0
            #optim = torch.optim.Adam(model.parameters(), lr=0.0003)
            #optim = torch.optim.Adagrad(model.parameters(), lr=0.001) # reset optim
