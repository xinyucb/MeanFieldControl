
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.animation import FuncAnimation
import torch.nn.functional as F

import numpy as np
import time
import copy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import gridspec
import pandas as pd

from NeuralNets import Network
import os

class PathMKV(nn.Module): 
    def __init__(self, setup, params, saved_NN_file=""):
        super().__init__()
        self.setup = setup
        # Save equation, training, and network parameters
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        self.params_net = params["net"]
        self.device = self.params_train["device"]  # Device (CPU/GPU)


        # # Problem-specific constants
        self.d = self.params_equ["state_dim"]
        self.T = self.params_equ["T"]      # Terminal time (end of time horizon)
        self.M = self.params_equ["M"]      # Number of trajectories (samples/paths)
        self.N = self.params_equ["N"]      # Number of time snapshots (time steps)
        self.state_dim = self.params_equ["state_dim"]
        self.bm_dim = self.params_equ["BM_dim"]
        self.u_dim = self.params_equ["u_dim"]
        
        ## Initialize control network
        net_control = Network(params["net"]) 
        filepath = os.path.join("outputNN", saved_NN_file + ".pth")

        # Check if file exists
        if os.path.exists(filepath):
            print(f"Loading pre-trained model from {filepath}")
            loaded = torch.load(filepath, map_location='cpu')  # or map_location=device
            net_control.load_state_dict(loaded['net_actor'])
            print("Model loaded successfully.")
        else:
            print(f"⚠️ Pre-trained model not found at {filepath}. Starting from scratch.")

        # Create a multiprocessing manager to hold shared objects
        manager = mp.Manager()
        self.dict = manager.dict()  # Shared dict to store networks, optimizers, schedulers
        self.dict['net_actor'] = net_control.state_dict()
        self.control_type = self.params_equ["control_type"]
     
        self.training_cost_list = []
        self.validation_list = []
        # Time step size
        self.dt = self.T / self.N
        

        filename= f"path_{self.control_type}_d_{self.d}"
        self.filename = filename 
        print("current filename:", filename)


  
    def fetch_minibatch(self):
        """
        Sample mini-batch of stochastic processes for M trajectories over N time steps:
            Brownian motions of each player, common Brownian motion,

        Inputs:
        - None (samples from internal parameters and random generators).

        Outputs: t, W, B, P, P0
        - t: Tensor (M, N+1, 1) of cumulative time grid.
        - W: Tensor (M, N+1, D*bm_dim) of idiosyncratic Brownian motions for each player.
        - B: Tensor (M, N+1, bm_dim) of common Brownian motion.
    
        Note:
        - No control inputs involved; only noise processes are sampled.
        """

        # Initialize empty arrays for time steps (Dt), Brownian increments (DW, DB)
        # Dt: M trajectories, N+1 time points (including t=0), 1-dim time step at each point
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.bm_dim))  # individual Brownian motion per player

        # Set constant dt for time steps except the initial point (t=0)
        Dt[:, 1:, :] = self.dt

        # Sample Brownian increments ~ N(0, dt)
        DW[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, self.bm_dim))

        # Integrate to get actual Brownian paths and time grid
        t = np.cumsum(Dt, axis=1)    # cumulative sum of time steps → time at each node
        W = np.cumsum(DW, axis=1)    # cumulative sum of Brownian increments → Brownian paths

 

        # Return tensors on the correct device
        # B is repeated along last axis to match dimensionality with W (D components)
        return (
            torch.from_numpy(t).float().to(self.device),                     # time grid (M x N+1 x 1)
            torch.from_numpy(W).float().to(self.device),                     # Brownian motion (M x N+1 x D)
            torch.from_numpy(DW[:,1:,:]).float().to(self.device)
          )

    def DW_mask(self, dw, j):
        dw_mask = dw.clone().detach()
        dw_mask[:, j:, :] = 0
        return dw_mask


    def loss_function(
        self, net_actor, t, DW, X0):
        """
        Compute the loss for the critic and actor networks over a batch of trajectories.

        Inputs:
        - net_actor: neural net producing control
        - t: Tensor (M, N+1, 1), time grid for each sample path.
        - DW: Tensor (M, N, D), Brownian motion intervals.
        - X0: Tensor (M, state_dim), initial state for all the players.
        - control_type: str, either "path_dependent" or "markovian" 
        Returns:
        - loss_actor: scalar tensor of actor loss.
        - X: Tensor of simulated state trajectories (M, n+1, state_dim).
        """

        # Initialize loss and buffers
        
        # Require gradients for X0 for potential derivative computations
        Xt = X0
        At = X0.clone().detach()
        Xt.requires_grad = True
        At.requires_grad = True
        X0_block = X0.unsqueeze(1) 
        reward = 0
        X_buffer = [Xt]   # to store states (X) along path
        A_buffer = [At]
        # Loop over n time steps for this loss segment
        for j in range(0, self.N ):
        
            t_= t[:, j, :]
            DW_t = DW[:, j, :]
            t_block = t_.unsqueeze(1).expand(-1, 1, self.state_dim)  #  t0 -> (M, 1, d): duplicate the single column to state_dim

            X_path = torch.stack(X_buffer, dim=1)     # (M, j+1, d)
            A_path = torch.stack(A_buffer, dim=1)
            hist_len = self.N + 1
            pad_len = hist_len - X_path.size(1)
            X_path_padded = F.pad(X_path, (0, 0, 0, pad_len), mode='constant', value=0.0) # [N+1, d]
            A_path_padded = F.pad(A_path, (0, 0, 0, pad_len), mode='constant', value=0.0) # [N+1, d]
            
            # print(self.control_type)
            ## Generate the control based on control type
            if self.control_type == "B":
                #  control: depends on time, initial state, and Brownian path
                net_input = torch.cat([t_block,  X0_block, self.DW_mask(DW, j)], dim=1)   # (M, N+2, d)
                net_input = net_input.view(self.M,-1)
                
            elif self.control_type == "X": 
                net_input =torch.cat([t_block,  X_path_padded], dim=1)       # (M, hist_len, d)
                net_input = net_input.view(self.M,-1)
                
            elif self.control_type == "A":
                net_input =torch.cat([t_block,  A_path_padded], dim=1)       # (M, hist_len, d)
                net_input = net_input.view(self.M,-1)
                

            elif self.control_type == "XtAt":
                net_input = torch.cat([t_block, At.unsqueeze(1), Xt.unsqueeze(1)], dim=1)
                net_input = net_input.view(self.M,-1)

            elif self.control_type == "Xt":
                net_input = torch.cat([t_block, Xt.unsqueeze(1)], dim=1)
                net_input = net_input.view(self.M,-1)
                
            # get current reward
            control = net_actor(net_input) 

            reward += self.setup.f_running(t_, X_path, control) * self.dt
            
            
            # Forward simulate SDE dynamics (Euler + jump terms)
            drift = self.setup.mu(t_, Xt, control)
            violatility = self.setup.sigma(t_, Xt, control)
            Xt = Xt + drift * self.dt + violatility * DW_t
            At = (j+1)/(j+2) * At + 1/(j+2) * Xt
            # Add Xt to the path history
            X_buffer.append(Xt)
            A_buffer.append(At)

        X_path_complete = torch.stack(X_buffer, dim=1)  # [X0, X1, ..., XT]
        
        # Calculate the terminal cost
        reward += self.setup.g_terminal(X_path_complete) 
        return X_buffer, reward

    def Actor_step(self, net_actor, optimizer_actor):
        # Fetch a minibatch of simulated noise trajectories:
        # t: time grid, W: Brownian motions, B: common noise, P: individual Poisson jumps, P0: common Poisson jumps
        t, W, DW = self.fetch_minibatch()

        # Xi is the initial condition
        X0 = self.setup.generate_initial_condition()
        
        # Call the loss function:
        X_buffer, loss = self.loss_function(net_actor, t, DW, X0)

        # Reset gradients before backward pass
        optimizer_actor.zero_grad()
        # Compute gradients of actor loss w.r.t. actor parameters
        loss.backward()
        optimizer_actor.step()


        # Return the actor loss for monitoring
        return loss.to(self.device)

    def train_players(self):
        """
        Train player k during iteration it.

        Parameters:
        - it: current outer iteration (int)
        - k: index of the player to train (int)

        This function:
        - Initializes terminal states if it's the first iteration.
        - Creates local copies of critic and actor networks for all players and loads their states.
        - Applies dynamic learning rate scheduling.
        - Trains player k's critic and actor over several epochs.
        - Updates the stored network states.
        """
        self.start_time = time.time()


        # Create local networks for all players and load stored parameters

        net_actor = Network(self.params_net, penalty=self.params_net["penalty"]).to(self.device)

            
        all_weights = []

        # Load previously saved parameters into local networks
        net_actor.load_state_dict(self.dict['net_actor'])
        all_weights += list(net_actor.parameters())

        # Set up Adam optimizers for player k's actor
        optimizer_actor = optim.Adam(all_weights, lr=self.params_train["lr_actor"])
        scheduler = ReduceLROnPlateau(optimizer_actor, mode='min', patience=50, factor=self.params_train["gamma_actor"])

        for it in range(self.params_train["iteration"]):
            # Train over specified number of epochs
            for _ in tqdm(range(self.params_train["epochs"])):
                
                # Perform actor update step for all players
                # Get control type from parameters, default to path_dependent for backward compatibility
                loss_actor = self.Actor_step(net_actor, optimizer_actor)
                
                # Save detached losses for monitoring
                self.dict[f"loss_actor"] = loss_actor.detach().cpu().clone()
                self.training_cost_list.append(loss_actor.detach().cpu().clone())

            self.dict[f'net_actor'] = net_actor.state_dict()

            ## Validation
            _,  loss, _ = self.simulation_paths()
            scheduler.step(loss) # validation  loss
            self.validation_list.append(loss)
           
            print('It: %d, loss_training: %.4f, loss_validation: %.4f' % (it, self.dict["loss_actor"].item(), loss))
            # Then print the updated lr
            for param_group in optimizer_actor.param_groups:
                print("Updated learning rate:", param_group['lr'])


        # After training, save updated networks back to the shared dictionary
        self.dict[f'net_actor'] = net_actor.state_dict()







    def simulation_paths(self):

        # Fetch a minibatch of simulated noise trajectories:
        # t: time grid, W: Brownian motions, B: common noise
        t, W, DW = self.fetch_minibatch()
        
        # Xi is the initial condition, repeat it M times for the batch
        X0 = self.setup.generate_initial_condition()


        # Create local networks for all players and load stored parameters

        net_actor = Network(self.params_net, penalty=self.params_net["penalty"]).to(self.device)
        

        # Load previously saved parameters into local networks
        net_actor.load_state_dict(self.dict['net_actor'])

        # Call the loss function:
        # Get control type from parameters, default to path_dependent for backward compatibility
        X_buffer, loss = self.loss_function(net_actor, t, DW, X0)
        return X_buffer, loss, net_actor

    
    

    def save_NN(self, filename = None):
        if filename == None: filename = self.filename
        X_buffer, loss, net_actor= self.simulation_paths()
        
        # Save all models in one file
        torch.save({'net_actor': net_actor.state_dict()}, \
                   'outputNN/' + filename + '.pth')
        print("The NN weights are saved in")
        print('outputNN/' + filename + '.pth')

    def save_loss(self, filename = None):
        if filename == None: filename = self.filename
        
        # Use the training cost list instead of a single loss value
        if hasattr(self, 'training_cost_list') and len(self.training_cost_list) > 0:
            loss_data = [t.numpy() for t in self.training_cost_list]
            df = pd.DataFrame(loss_data, columns=["val"])
            df.to_csv(f'outputLoss/{filename}_val_loss.csv', index=False)
            print("The loss data is saved in")
            print(f'outputLoss/{filename}_val_loss.csv')
        else:
            print("No training cost data available to save.")

    def get_network_input_size(self, control_type="path_dependent"):
        """
        Calculate the required network input size based on control type
        
        Args:
            control_type: str, either "path_dependent" or "markovian"
            
        Returns:
            int: required input size for the network
        """
        if control_type == "path_dependent":
            # Original: time + current state + Brownian path
            return (self.N + 2) * self.state_dim
        elif control_type == "markovian":
            # New: only time + current state
            return 2 * self.state_dim
        else:
            raise ValueError(f"Unknown control_type: {control_type}. Use 'path_dependent' or 'markovian'")
    
    def switch_control_type(self, new_control_type):
        """
        Switch between control types and adjust network accordingly
        
        Args:
            new_control_type: str, either "path_dependent" or "markovian"
        """
        if new_control_type not in ["path_dependent", "markovian"]:
            raise ValueError(f"Unknown control_type: {new_control_type}. Use 'path_dependent' or 'markovian'")
        
        # Update the control type in parameters
        self.params_equ["control_type"] = new_control_type
        
        # Calculate new input size
        new_input_size = self.get_network_input_size(new_control_type)
        
        # Update network parameters
        self.params_net["inputs"] = new_input_size
        
        print(f"Switched to {new_control_type} control")
        print(f"Network input size updated to: {new_input_size}")
        print(f"Input size breakdown:")
        if new_control_type == "path_dependent":
            print(f"  - Time: {self.state_dim}")
            print(f"  - Current state: {self.state_dim}")
            print(f"  - Brownian path: {self.N * self.state_dim}")
        else:  # markovian
            print(f"  - Time: {self.state_dim}")
            print(f"  - Current state: {self.state_dim}")
        
        return new_input_size