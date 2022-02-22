import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayers(nn.Module):
    def __init__(self, layers, input_dim, h_units, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, h_units[i]))
            else:
                self.layers.append(nn.Linear(h_units[i-1], h_units[i]))

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i < len(self.layers)-1:
                output = self.activation(output)
        return output