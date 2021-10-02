import torch
import encoder
import math
import random
import tqdm
import sys

alphabet = ["0", "1", "$"]
alphabet_index = {a:i for i,a in enumerate(alphabet)}
max_pos = 10000
big = 10

class FirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(6, 1, 1, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.zeros(18,6))
        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(18))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.zeros(6,6))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(6))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [0,1,0,1,0,0],
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.tensor([-1], dtype=torch.float))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0]]*4 +
            [[1],
             [0]],
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.tensor(
            [0,0,0,0,0,0],
            dtype=torch.float))
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        #src2 = self.norm1(src2) # norm before residual
        src = src + self.dropout1(src2)
        #src = self.norm1(src) # norm after residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        #src = self.norm2(src)
        return src

class SecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(6, 1, 1, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # W^Q
            [[0,0,big,0,0,0]] +
            [[0]*6]*5 +
            # W^K
            [[0,0,0,1,0,0]] +
            [[0]*6]*5 +
            # W^V
            [[0]*6]*5 +
            [[0,0,0,-0.5,1,0]],
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(18))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*6]*5 +
            [[0,0,0,0,0,1]],
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(6))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(1,6))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(1))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(6,1))
        self.linear2.bias = torch.nn.Parameter(torch.tensor(
            [0,0,0,0,0,0],
            dtype=torch.float))

    forward = FirstLayer.forward

class MyTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstLayer(),
            SecondLayer(),
        ])
        self.num_layers = len(self.layers)
        self.norm = None

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.word_embedding = torch.eye(3, 6)
        self.pos_embedding = torch.stack(
            [torch.zeros(max_pos)]*3 +
            [torch.arange(0, max_pos, dtype=torch.float) == 1] +
            [torch.zeros(max_pos)]*2,
            dim=1)

        self.transformer_encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(6, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,1]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        print(w)
        x = self.word_embedding[w] + self.pos_embedding[:len(w)]
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        print(z)
        return torch.sigmoid(z)

model = Model()

valid_loss = 0
valid_num = 0
valid_correct = 0
n_steps = 1000
for step in range(n_steps):
    n = random.randrange(1, 11)
    w = torch.tensor([alphabet_index['$']] + [alphabet_index[str(random.randrange(2))] for i in range(n)])
    o = w[1] == alphabet_index['1']
    p = model(w)
    if not o: p = 1-p
    if p > 0.5:
        valid_correct += 1
    valid_num += 1
    loss = -torch.log(p)
    valid_loss += loss.item()
print(f'ce={valid_loss/valid_num/math.log(2)} acc={valid_correct/valid_num}')

