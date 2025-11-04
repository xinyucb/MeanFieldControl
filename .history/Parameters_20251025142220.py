
import torch
import numpy as np
import torch.nn as nn

# ============================================================================
# Define all parameters 
# ============================================================================

"""
control_types = ["B", "X", "A"] #Brownian motion, path of X, path of average
"""

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DefaultParams:
    N = 100
    input_param = (N + 2)*d

    params = {
        # Parameters of the game setup
        "equation": {
                "state_dim": d, "BM_dim": d, "u_dim": d, 
                "M": 500, # number of particles
                "N": N, # number of time intervals
                "T": 1,
                "control_type": control_type
            },
        "train": {
            "lr_actor": 1e-3, "gamma_actor":0.8, "milestones_actor": [30000, 40000],
            "iteration": 50, "epochs": 50, "device": device,  
        },
        "net": {
            # Choose block type: "res" for ResNet block, "fc" for fully connected block
            "block_type": ["fc", "rnn"][1],
            "inputs": input_param, "output": d, 


            # RNN network parameters(only used when block_type="rnn")
            "width": 100, "depth": 4, #N + 10, "depth": 4, 
            "activation": "ReLU", "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
            
            
            # Fully connected network parameters (only used when block_type="fc")
            "hidden_layers": [20, 20], #[N + 10, N + 10, N + 10],  #widths of hidden layers 
            "dropout_rate": 0.1,  # Add dropout for regularization
        },
    }






