import numpy as np
import torch

class Systematic():
    def __init__(self, params_equ, device, kappa=0.6, sigma=1, q=0.8, T=0.2, eta=2, c=2, \
        mu_0 = 0, sigma_0 = 0.2,
        BM_dim=None, u_dim=None) -> None:
        self.terminal = params_equ["T"]
        self.sigma_ = sigma
        self.kappa = kappa
        self.q = q
        self.eta = eta
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
        mu_t = self.get_mean_expanded(Xt)
        return self.kappa * (mu_t - Xt) + control

    def sigma(self, t, Xt, control):
        return self.sigma_ 
        
    
    def get_mean_expanded(self, Xt):
        """
        Xt: Batch size (M) * d  

        return: M copies of mu_t in shape of (M,d)
        """
        return torch.mean(Xt, axis=0).expand(self.M,-1) # (M,d)
            
    def f_running(self, t, Xt, control):
        mu_t = self.get_mean_expanded(Xt)
        # return 0
        return  0.5 * torch.mean(torch.bmm(control.unsqueeze(1), control.unsqueeze(2))) \
            - self.q * torch.mean(torch.bmm(control.unsqueeze(1), (mu_t - Xt).unsqueeze(2))) \
            + 0.5* self.eta * torch.mean(torch.bmm( (mu_t - Xt).unsqueeze(1), (mu_t - Xt).unsqueeze(2)))
            
    def g_terminal(self, XT):
        mu_T = self.get_mean_expanded(XT)
        return self.c/2 * torch.mean(torch.bmm((mu_T - XT).unsqueeze(1), (mu_T - XT).unsqueeze(2)))

    # def get_theoretical_sol(self):
    #     Qt = 
        