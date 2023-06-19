
from torch import nn
from Model.activations import (
    SiLU,
    Hardswish,
    MemoryEfficientSwish,
    Mish,
    MemoryEfficientMish,
    FReLU
)

class FNNBlock(nn.Module):

    def __init__(self, dim_in, dim_out, activation=None, add_drop=False):
        super(FNNBlock, self).__init__()

        # set parameters
        self.dim_in = dim_in
        self.dim_out = dim_out

        # create linear layers
        self.Linear = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(0.05) if add_drop else None

        # choose activation function = Tanh
        if activation is None:
            self.activation = activation
        else:
            if activation.lower() == 'tanh':
                self.activation = nn.Tanh()
            elif activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation.lower() == 'silu':
                self.activation = SiLU()
            elif activation.lower() == 'hard_swish':
                self.activation = Hardswish()
            elif activation.lower() == 'me_swish':
                self.activation = MemoryEfficientSwish()
            elif activation.lower() == 'mish':
                self.activation = Mish()
            elif activation.lower() == 'me_mish':
                self.activation = MemoryEfficientMish()
            elif activation.lower() == 'frelu':
                self.activation = FReLU()

    def forward(self, x):

        x = self.Linear(x)
        if self.dropout is not None:
            x = self.dropout(x)

        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class FNN(nn.Module):

    def __init__(self, dim_in, width, dim_out, depth,
                 activation={'in': 'tanh', 'hid': 'tanh', 'out': None}):
        super(FNN, self).__init__()

        # set parameters
        self.dim_in = dim_in
        self.width = width
        self.dim_out = dim_out
        self.depth = depth

        # creat a list for holding all blocks/layers
        self.stack = nn.ModuleList()
                
        # input layer
        self.stack.append(FNNBlock(dim_in, width, activation=activation['in'], add_drop=False))
        # add hidden blocks
        for i in range(depth):
            self.stack.append(FNNBlock(width, width, activation=activation['hid'], add_drop=False))        
        # output layer
        self.stack.append(FNNBlock(width, dim_out, activation=activation['out']))

    def forward(self, x):
        
        for layer in self.stack:
            x = layer(x)

        return x

    def Xavier_initi(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()  
