import torch
import math
import random
import encoder

#torch.autograd.set_detect_anomaly(True)

loss_function = "cross_entropy"
#loss_function = "perceptron"

alphabet = ["0", "1", "$"]

big = 1

log_sigmoid = torch.nn.LogSigmoid()

class FirstLayer(encoder.SigmoidTransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # First head attends to all symbols,
            # second head does nothing.
            # W^Q
            [[big]*10]*10 +
            # W^K
            [[big]*10]*10 +
            # W^V
            [[0,1,0,0,0,0,0,0,0,0],   # count 1s  (k)
             [0,0,1,0,0,0,0,0,0,0]] + # count CLS (1)
            [[0]*10]*8,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*10]*5 +
            [[1,0,0,0,0,0,0,0,0,0],   # put new values into dims 5-6
             [0,1,0,0,0,0,0,0,0,0]] +
            [[0]*10]*3,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [0,0,0,-1,0,1,-1,0,0,0],  # k-i-1
            [0,0,0,-1,0,1, 0,0,0,0],  # k-i
            [0,0,0,-1,0,1, 1,0,0,0],  # k-i+1
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0, 0, 0]]*7 +
            [[1,-2, 1],  # put I[i=c1] in dim 7
             [0, 0, 0],
             [0, 0, 0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        #src2 = self.norm1(src2) # norm before residual doesn't work
        src = src + self.dropout1(src2)
        #src = self.norm1(src) # norm after residual does work
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        #src = self.norm2(src)
        return src

class SecondLayer(encoder.SigmoidTransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # W^Q
            # Heads 1 and 2 attend from CLS
            [[0,0,big,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            [[0,0,big,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            # W^K
            # Head 1 attends to odd positions
            [[0,0,0,0, big,0,0,0,0,0]] +
            [[0]*10]*4 +
            # Head 2 attends to even positions
            [[0,0,0,0,-big,0,0,0,0,0]] +
            [[0]*10]*4 +
            # W^V
            # Heads 1 and 2 average dim 7
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4 +
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            # Even positions minus odd positions
            # Place in dim 8
            [[0]*10]*8 +
            [[-1,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0]],
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(3,10))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(10,3))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        q = src
        v = src
        src2 = self.self_attn(q, src, v)[0]
        #src2 = self.norm1(src2)
        src = src + self.dropout1(src2)
        #src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        #src = self.norm2(src)
        return src
    #forward = FirstLayer.forward

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
        self.word_embedding = torch.eye(3, 10)
        self.transformer_encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(10, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,big,0]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        p = torch.stack([torch.zeros(len(w))]*3 +
                        [torch.arange(0, len(w), dtype=torch.float),
                         torch.cos(torch.arange(0, len(w), dtype=torch.float)*math.pi)] +
                        [torch.zeros(len(w))]*5,
                        dim=1)
        x = self.word_embedding[w] + p
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[-1])
        return z

model = Model()
#optim = torch.optim.Adam(model.parameters(), lr=3e-4)
#optim = torch.optim.SGD(model.parameters(), lr=0.001)
optim = torch.optim.Adagrad(model.parameters(), lr=0.01)

n_max = 100

best_epoch_loss = float('inf')
no_improvement = 0

graph_s = {}
graph_n = {}

for epoch in range(10000):
    epoch_loss = 0
    epoch_steps = 0
    epoch_correct = 0
    # Perturb parameters
    if True:
        with torch.no_grad():
            for p in model.parameters():
                p += torch.randn(p.size()) * 1e-2 # 1e-2 works, 1e-1 is too hard
    for step in range(1000):
        n = random.randrange(1, n_max+1)
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        o = len([a for a in w if a == 1]) % 2 == 1
        y = model(w)
        if not o: y = -y
        if y > 0: epoch_correct += 1
        graph_s.setdefault(n, 0.)
        graph_n.setdefault(n, 0.)
        graph_s[n] += y.item()**2
        graph_n[n] += 1

        if loss_function == "cross_entropy":
            loss = -log_sigmoid(y)
        elif loss_function == "perceptron":
            loss = torch.maximum(-y, torch.Tensor([0.]))
        else:
            raise NotImplementedError()
            
        epoch_loss += loss.item()
        epoch_steps += 1
        optim.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optim.step()

    """with open('graph', 'w') as outfile:
        for n in sorted(graph_s):
            print(n, (graph_s[n] / graph_n[n])**0.5, file=outfile)"""

    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= 10:
            optim.param_groups[0]['lr'] *= 0.5
            print(f"lr={optim.param_groups[0]['lr']}")
            no_improvement = 0
        
    valid_loss = 0
    valid_steps = 0
    valid_correct = 0
    for step in range(100):
        n = random.randrange(1, n_max+1)
        #n = random.randrange(n_max+1, 10*n_max+1)
        w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
        o = len([a for a in w if a == 1]) % 2 == 1
        y = model(w)
        if not o: y = -y
        if y > 0: valid_correct += 1
        
        if loss_function == "cross_entropy":
            loss = -log_sigmoid(y)
        elif loss_function == "perceptron":
            loss = torch.maximum(-y, torch.Tensor([0.]))
        else:
            raise NotImplementedError()
        
        valid_loss += loss.item()
        valid_steps += 1

    print(f'n_max={n_max} train_ce={epoch_loss/epoch_steps/math.log(2)} train_acc={epoch_correct/epoch_steps} valid_ce={valid_loss/valid_steps/math.log(2)} valid_acc={valid_correct/valid_steps}')
