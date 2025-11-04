import numpy as np
import torch
import torch.nn.functional as F

class HighDimPathModel():
    def __init__(self, params, kappa=0.6, sigma=0.1, eta=0.01, K=1.25, T=1,  c=1, \
        mu_0 = 0, sigma_0 = 0.5) -> None:
        params_equ = params["equation"]
        params_train = params["train"]

        self.d = params_equ["state_dim"] # dimension of Xt
        # dim of Brownian motion
        self.BM_dim = params_equ["BM_dim"]
        # dim of control 
        self.u_dim = params_equ["u_dim"]
        # number of samples
        self.M = params_equ["M"]
        self.N = params_equ["N"]
        self.device = params_train["device"]

        self.terminal = params_equ["T"]
        sigma = 1/(self.d * self.u_dim) * np.ones([self.d, self.u_dim]) + np.random.rand(self.d, self.u_dim) * sigma
        self.sigma_ = sigma 

        self.kappa = kappa
        self.K = K
        self.c = c
        self.eta = eta
        self.delay = 4


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

    
    def mu(self, j, dt, t, X_path, control):
        """
        X:  (M, j+1, d)
        control: (M, d)

        return: (M,d)
        """
        Xt = X_path[:, -1]
        if self.delay > 0:
            if j <= self.delay: return 0 * Xt # just to match the dimension
            Xt_minus_2 = X_path[:, -3]
            return  (Xt - Xt_minus_2 +  control) * dt
        else:
            return  (Xt  +  control) * dt

    def sigma(self, DW_t, t, X_path, control):
        # if t[0,0] <= self.delay: return 0
        sigma_torch = torch.from_numpy(self.sigma_).to(DW_t.device, dtype=DW_t.dtype)
        
        mat1 = sigma_torch.unsqueeze(0).expand(self.M, self.d, self.BM_dim)  # (500, 100, 2)
        # print("mat1", np.shape(mat1))
        mat2 = DW_t.unsqueeze(2)                      # (500, 2, 1)
        # print("mat2", np.shape(mat2))
        result = torch.bmm(mat1, mat2).squeeze(-1)                # (500, 100, 2)


        return result
        
    
    def get_mean_expanded(self, Xt):
        """
        Xt: Batch size (M) * d  

        return: M copies of mu_t in shape of (M,d)
        """
        return torch.mean(Xt, axis=0).expand(self.M,-1) # (M,d)

    def get_empirical_prob(self, Xt):
        return  torch.mean(1.0*(Xt >= 0.05), axis=0).expand(self.M, -1)

    def f_running(self, j, t, X_path, control):
        """
        j: an integer in [0, 1, ..., N-1]
        t: tensor(constant)
        """
        # print("time,", t[0])
        if t[0,0] <= self.delay:
            return 0
        return self.eta * torch.mean(torch.bmm(control.unsqueeze(1), control.unsqueeze(2))) 
 
    def g_terminal(self, XT):
        #X_path: (M, T+1, d)
        AT = torch.mean(XT, dim=1)
        var_payoff = torch.var(AT - self.K)
        return self.c * var_payoff

    def get_control(self, j, t_, X_path, dw, control_type):
        """
        j: an integer in [0, 1, ..., N-1]
        t: (M, 1)
        X_path: (M, j+1, d)
        """
        X0 = X_path[:,0,:]
        Xt = X_path[:,j,:]  
    

  
        hist_len = self.N + 1
        pad_len = hist_len - X_path.size(1)
        X_path_padded = F.pad(X_path, (0, 0, 0, pad_len), mode='constant', value=0.0) # [N+1, d]
        # 
        ## Generate the control based on control type
        if control_type == "B":     
            net_input = torch.cat([t_,  X0, dw], dim=1)
            
        elif control_type == "X": 
            net_input = torch.cat([t_,  X_path_padded.view(self.M,-1)], dim=1)       # (M, hist_len, d)
            net_input = net_input.view(self.M,-1)
            

        elif control_type == "Xt":

            net_input = torch.cat([t_, Xt], dim=1)
            net_input = net_input.view(self.M,-1)
                
        return net_input

