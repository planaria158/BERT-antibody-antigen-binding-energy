
import pickle as pk
from pathlib import Path
import os
import time
import torch
from torch import nn
# from pytorch_lightning.core import LightningModule

class Layer_simple(nn.Module):
    """
        A single layer of an MLP type network
        Args:
            in_dim: input dimension
            out_dim: output dimension
            dropout: dropout rate
            normalize: whether to apply batch normalization
            activation: whether to apply GELU activation
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, normalize=True, activation=True):
        super(Layer_simple, self).__init__()
        self.normalize = normalize
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.activation(self.norm(self.linear(x))))
        return out

class ResidualMLP(nn.Module):
    """
        An MLP model that uses resiual-like connections between layers

        Args:
            config: dictionary of config parameters
            input_dim: input dimension
    """
    def __init__(self, config, input_dim, num_layers=4):
        super(ResidualMLP, self).__init__()
        print('Regression head is ResidualMLP')
        self.in_dim = input_dim
        self.out_dim = input_dim  # output dim each layer. 
        self.num_layers = num_layers      

        #  may look something like this: (assuming input_dim = 256, for example)
        #  layers: [[256,256], [256,256], [256,256], [256,1]]
        self.net = nn.ModuleList()
        in_dim = self.in_dim
        for idx in range(self.num_layers-1):
            print('making residual mlp layer in_dim, out_dim:', in_dim, self.out_dim)
            layer = Layer_simple(in_dim, self.out_dim, config['mlp_dropout'])
            self.net.append(layer)

        # For regression, the very last Layer has no normalization or activation
        print('making final dense mlp layer in_dim, out_dim:', in_dim, 1)
        layer = Layer_simple(self.out_dim, 1, config['mlp_dropout'], normalize=False, activation=False)
        self.net.append(layer)

    def forward(self, x_in):
        x = x_in
        for layer in self.net[:-1]:
            x = layer(x) + x

        logits = self.net[-1](x)

        return logits
