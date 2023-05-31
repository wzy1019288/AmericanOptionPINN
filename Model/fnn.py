
from torch import nn


class FNNBlock(nn.Module):

    def __init__(self, dim_in, dim_out, activation=None):
        super(FNNBlock, self).__init__()

        # set parameters
        self.dim_in = dim_in
        self.dim_out = dim_out

        # create linear layers
        self.Linear = nn.Linear(dim_in, dim_out)
        # choose activation function = Tanh
        if activation is None:
            self.activation = activation
        else:
            if activation.lower() == 'tanh':
                self.activation = nn.Tanh()

    def forward(self, x):
        
        if self.activation is not None:
            return self.activation(self.Linear(x))
        else:
            return self.Linear(x)

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
        self.stack.append(FNNBlock(dim_in, width, activation=activation['in']))
        # add hidden blocks
        for i in range(depth):
            self.stack.append(FNNBlock(width, width, activation=activation['hid']))        
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
