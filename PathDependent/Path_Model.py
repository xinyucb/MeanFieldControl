import numpy as np
import torch
import torch.nn as nn


class MVPathModel():
    def __init__(self, params, kappa=0.6, sigma=1, eta=0.01, K=1.25, T=1,  c=1, \
        mu_0 = 0, sigma_0 = 0.5,
        BM_dim=None, u_dim=None) -> None:
        params_equ = params["equation"]
        params_train = params["train"]

        self.terminal = params_equ["T"]
        self.sigma_ = sigma
        self.kappa = kappa
        self.K = K
        self.c = c
        self.eta = eta


        self.d = params_equ["state_dim"] # dimension of Xt
        # dim of Brownian motion
        self.BM_dim = params_equ["BM_dim"]
        # dim of control 
        self.u_dim = params_equ["u_dim"]
        # number of samples
        self.M = params_equ["M"]
        self.device = params_train["device"]

        # default mean and covariance for the initial random variable
        self.mu0 = mu_0 * torch.tensor(np.ones(self.d)).float().to(self.device)
        self.var0 = sigma_0**2 * torch.tensor(np.eye(self.d)).float().to(self.device) 
    
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
        return self.eta * torch.mean(torch.bmm(control.unsqueeze(1), control.unsqueeze(2))) 
 
    def g_terminal(self, X_path):
        payoff =torch.abs(torch.mean(X_path, dim=1) - self.K)
        mean_payoff = torch.mean(payoff)
        var_payoff = torch.var(payoff)
        return  mean_payoff + self.c * var_payoff + mean_payoff

