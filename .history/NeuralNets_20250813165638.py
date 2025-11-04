import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.animation import FuncAnimation

import numpy as np
import time
import copy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import gridspec
import pandas as pd

class RNNBlock(nn.Module):

    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(RNNBlock, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(torch.sin(self.L1(x)))) + x
        else:
            a = self.act(self.L1(self.act(self.L2(x)))) + x
        return a


class FCBlock(nn.Module):
    """
    Fully Connected Block with configurable hidden layers and dropout
    """

    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(FCBlock, self).__init__()
        
        # Get network architecture parameters
        hidden_layers = params.get("hidden_layers", [inputs, inputs])  # Default to [inputs, inputs] for backward compatibility
        dropout_rate = params.get("dropout_rate", 0.0)
        
        # Build the fully connected network
        layers = []
        prev_dim = inputs
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization for better training
            
            # Add activation function
            if activation == "Sin" or activation == "sin":
                layers.append(nn.SiLU())  # Use SiLU as approximation for sin activation
            elif activation == "Tanh":
                layers.append(nn.Tanh())
            elif activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif activation == "GELU":
                layers.append(nn.GELU())
            else:
                # Use the activation from params if specified
                if activation in params:
                    layers.append(params[activation])
                else:
                    layers.append(nn.Tanh())  # Default to Tanh
            
            # Add dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final layer to map back to input dimension for residual connection
        layers.append(nn.Linear(prev_dim, inputs))
        
        self.network = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x):
        # Apply the fully connected network
        output = self.network(x)
        
        # Add residual connection
        return output + x


class Network(nn.Module):

    def __init__(self, params, penalty=None):
        super(Network, self).__init__()
        self.params = params
        self.first = nn.Linear(self.params["inputs"], self.params["width"])
        self.last = nn.Linear(self.params["width"], self.params["output"])
        
        # Choose block type based on parameters
        block_type = params.get("block_type", "rnn")  # Default to original RNN block
        if block_type == "fc":
            # Use fully connected block
            blocks = [FCBlock(self.params["width"], self.params, self.params["activation"])] * self.params["depth"]
        else:
            # Use original RNN block
            blocks = [RNNBlock(self.params["width"], self.params["params_act"], self.params["activation"])] * self.params["depth"]
        
        self.network = nn.Sequential(*[
            self.first,
            *blocks,
            self.last
        ])
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        if self.penalty is not None:
            if self.penalty == "Sigmoid":
                sigmoid = nn.Sigmoid()
                return sigmoid(self.network(x)) * self.bound
            elif self.penalty == "Tanh":
                tanh = nn.Tanh()
                return tanh(self.network(x)) * self.bound
            else:
                raise RuntimeError("Other penalty has not bee implemented!")
        else:
            return self.network(x)
