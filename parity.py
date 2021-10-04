import torch
import math
import random
import encoder

log_sigmoid = torch.nn.LogSigmoid()

size = 64

n_max = 100
n0_max = 1
num_epochs = 1000000
num_steps = 100

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, size):
        super().__init__()
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=size)
        self.pos_embedding = torch.stack([
            torch.arange(0, n_max+1, dtype=torch.float),
            torch.cos(torch.arange(0, n_max+1, dtype=torch.float)*math.pi)
        ], dim=1)
        self.pos_adapter = torch.nn.Linear(self.pos_embedding.size()[1], size)

        encoder_layer = encoder.PostnormTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        #encoder_layer = encoder.SigmoidTransformerEncoderLayer(d_model=size, nhead=4, dim_feedforward=size*4, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_layer = torch.nn.Linear(size, 1)

    def forward(self, w):
        p = self.pos_embedding[:len(w)].clone()
        p[:,0] /= len(w)
        x = self.word_embedding(w) + self.pos_adapter(p)
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        y = y[-1]
        return self.output_layer(y)

model = Model(3, size)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)
#optim = torch.optim.Adagrad(model.parameters(), lr=0.01)

best_epoch_loss = float('inf')
no_improvement = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_steps = 0
    epoch_correct = 0
    
    for step in range(num_steps):
        n = random.randrange(1, n_max+1)
        n0 = min(n, random.randrange(0, n0_max+1))
        w = [1]*n0 + [0]*(n-n0)
        random.shuffle(w)
        w.append(2)
        w = torch.tensor(w)
        
        label = len([a for a in w if a == 1]) % 2 == 1
        output = model(w)
        if not label: output = -output
        if output > 0: epoch_correct += 1
        loss = -log_sigmoid(output)
        epoch_loss += loss.item()
        epoch_steps += 1
        optim.zero_grad()
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

    test_loss = 0
    test_steps = 0
    test_correct = 0
    for step in range(num_steps):
        n = random.randrange(1, n_max+1)
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        label = len([a for a in w if a == 1]) % 2 == 1
        output = model(w)
        if not label: output = -output
        if output > 0: test_correct += 1
        loss = -log_sigmoid(output)
        test_loss += loss.item()
        test_steps += 1
        
    print(f'n0_max={n0_max} train_ce={epoch_loss/epoch_steps/math.log(2)} train_acc={epoch_correct/epoch_steps} test_ce={test_loss/test_steps/math.log(2)} test_acc={test_correct/test_steps}')

    if n0_max < n_max:
        #if epoch_loss/epoch_steps/math.log(2) < 0.1:
        if epoch_correct/epoch_steps > 0.9:
            n0_max += 1
            no_improvement = 0
            #optim = torch.optim.Adam(model.parameters(), lr=0.0003)
            #optim = torch.optim.Adagrad(model.parameters(), lr=0.001) # reset optim
