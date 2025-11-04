
import torch
import numpy as np
import torch.nn as nn






class DefaultParams:
    def __init__(self, d = 20, control_type='B', N=50, u_dim = 2, BM_dim = 2):

        input_param_dic = {
            'B': 1 + d + BM_dim * N,
            "X": (N+1) * d + 1,
            "Xt": d + 1
            } 

        input_param = input_param_dic[control_type]

        params = {
                    # Parameters of the game setup
                    "equation": {
                            "state_dim": d, "BM_dim": BM_dim, "u_dim": u_dim, 
                            "M": 500, # number of particles
                            "N": N, # number of time intervals
                            "T": 1,
                            "control_type": control_type
                        },
                    "train": {
                        "lr_actor": 1e-3, "gamma_actor": 0.8, "milestones_actor": [30000, 40000],
                        "iteration": 5, "epochs": 50, "device": "cpu",  
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

