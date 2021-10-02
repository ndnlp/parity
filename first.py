import torch
import encoder
import math
import random
import tqdm
import sys

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
        z = torch.sigmoid(self.output_layer(y))
        return z

n_trials = 20
n_epochs = 100
n_steps = 100

train_graph = torch.zeros(n_trials, n_epochs, requires_grad=False)
valid_graph = torch.zeros(n_trials, n_epochs, requires_grad=False)
valid_acc_graph = torch.zeros(n_trials, n_epochs, requires_grad=False)
valid_att_graph = torch.zeros(n_trials, n_epochs, 2, requires_grad=False)

with open(sys.argv[1], 'w') as outfile:

    for n in range(10, 101, 10):
    
        for trial in tqdm.tqdm(range(n_trials)):
            model = Model(len(alphabet), size)
            optim = torch.optim.Adam(model.parameters(), lr=0.0003)

            for epoch in range(n_epochs):

                epoch_loss = 0
                for step in range(n_steps):
                    #n = random.randrange(1, 101)
                    n = 10 * (trial+1)
                    w = torch.tensor([alphabet_index['$']] + [alphabet_index[str(random.randrange(2))] for i in range(n)])
                    o = w[1] == alphabet_index['1']
                    p = model(w)
                    if not o: p = 1-p
                    loss = -torch.log(p)
                    epoch_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                train_graph[trial, epoch] = epoch_loss

                valid_loss = 0
                valid_num = 0
                valid_correct = 0
                alpha = torch.zeros(2)
                for step in range(n_steps):
                    #n = random.randrange(101, 1001)
                    n = 1000
                    w = torch.tensor([2] + [random.randrange(2) for i in range(n)])
                    o = w[1] == alphabet_index['1']
                    p = model(w)
                    for l in range(2):
                        alpha[l] += model.encoder.layers[l].last_weights[0][0][1].detach()
                    if not o: p = 1-p
                    if p > 0.5: valid_correct += 1
                    valid_num += 1
                    loss = -torch.log(p)
                    valid_loss += loss.item()
                valid_graph[trial, epoch] = valid_loss
                valid_acc_graph[trial, epoch] = valid_correct/valid_num
                valid_att_graph[trial, epoch] = alpha/valid_num

        print(n,
              train_graph[:,-1].mean().item(),
              train_graph[:,-1].std().item(),
              valid_graph[:,-1].mean().item(),
              valid_graph[:,-1].std().item(),
              valid_acc_graph[:,-1].mean().item(),
              valid_acc_graph[:,-1].std().item(),
              valid_att_graph[:,-1,0].mean().item(),
              valid_att_graph[:,-1,0].std().item(),
              valid_att_graph[:,-1,1].mean().item(),
              valid_att_graph[:,-1,1].std().item(),
              file=outfile)


        """for epoch in range(n_epochs):
            print(epoch+1,
                  train_graph[:,epoch].mean().item(),
                  train_graph[:,epoch].std().item(),
                  valid_graph[:,epoch].mean().item(),
                  valid_graph[:,epoch].std().item(),
                  valid_acc_graph[:,epoch].mean().item(),
                  valid_acc_graph[:,epoch].std().item(),
                  valid_att_graph[:,epoch,0].mean().item(),
                  valid_att_graph[:,epoch,0].std().item(),
                  valid_att_graph[:,epoch,1].mean().item(),
                  valid_att_graph[:,epoch,1].std().item(),
                  file=outfile)"""
