import numpy as np
import torch
import torch.nn as nn


class PathModel():
    def __init__(self, params_equ, device, kappa=2, sigma=1, K=1.25, T=0.2,  c=2, \
        mu_0 = 0, sigma_0 = 1,
        BM_dim=None, u_dim=None) -> None:
        self.terminal = params_equ["T"]
        self.sigma_ = sigma
        self.kappa = kappa
        self.K = K
        self.c = c



        self.d = params_equ["state_dim"] # dimension of Xt
        # dim of Brownian motion
        self.BM_dim = params_equ["BM_dim"]
        # dim of control 
        self.u_dim = params_equ["u_dim"]
        # number of samples
        self.M = params_equ["M"]
        self.device = device

        # default mean and covariance for the initial random variable
        self.mu0 = mu_0 * torch.tensor(np.ones(self.d)).float().to(device)
        self.var0 = sigma_0**2 * torch.tensor(np.eye(self.d)).float().to(device) 
    
    def change_mean_var(self, mu0, var0):
        """
        mu0: numpy array [d,]
        sigma0: numpy matrix [d * d]
        """
        self.mu0 = torch.tensor(mu0).float().to(self.device)
        self.var0 = torch.tensor(var0).float().to(self.device)


    
    def generate_initial_condition(self):
        # Shape: (M, d)
        L = torch.linalg.cholesky(self.var0)
        Gaussian_noise = torch.randn(self.M, self.d, device=self.device).float()
        x = self.mu0 + Gaussian_noise @ L.T 
        return x #x.view(self.d*self.M)

    
    def mu(self, t, Xt, control):
        """
        Xt: Batch size (M) * d  
        control: (M, d)

        return: (M,d)
        """
        return self.kappa * Xt + control

    def sigma(self, t, Xt, control):
        return self.sigma_ * Xt
        
    
    def get_mean_expanded(self, Xt):
        """
        Xt: Batch size (M) * d  

        return: M copies of mu_t in shape of (M,d)
        """
        return torch.mean(Xt, axis=0).expand(self.M,-1) # (M,d)

    def get_empirical_prob(self, Xt):
        return  torch.mean(1.0*(Xt >= 0.05), axis=0).expand(self.M, -1)

    def f_running(self, t, X_path, control):
        return 0.01 * torch.mean(torch.bmm(control.unsqueeze(1), control.unsqueeze(2))) 
 
    def g_terminal(self, X_path):
        payoff =torch.abs(torch.mean(X_path, dim=1) - self.K)
        mean_payoff = torch.mean(payoff)
        var_payoff = torch.var(payoff)
        return  mean_payoff + self.c * var_payoff


    def default_params(d, N, T, control_type, device, ):
        # ============================================================================
        # Define all parameters 
        # ============================================================================
        if type(control_type) == int:
            control_type = ["B", "X", "A"][control_type]
        elif type(control_type) == str:
            if "path" in control_type.lower():
                control_type = "path_dependent"
            if "markov" in control_type.lower():
                control_type = "" 
        input_param = (N * (control_type == "path_dependent") + 2)*d
        
        params = {
            # Parameters of the game setup
            "equation": {
                    "state_dim": d, "BM_dim": d, "u_dim": d, 
                    "M": 500, # number of particles
                    "N": N,# number of time intervals
                    "T": T,
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

        return params





