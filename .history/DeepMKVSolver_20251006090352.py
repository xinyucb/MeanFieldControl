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

from NeuralNets import Network

class deepPG(nn.Module):  # Solver for deep potential game
    def __init__(self, net_control, setup, params):
        super().__init__()
        self.setup = setup
        # Save equation, training, and network parameters
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        self.params_net = params["net"]
        self.device = self.params_train["device"]  # Device (CPU/GPU)


        # # Problem-specific constants
        # self.Xi = self.params_equ["Xi"]    # Initial point of state dynamics (starting state)
        # self.Yi = self.params_equ["Yi"]    # Initial point of sensitivity process (starting state)
        self.T = self.params_equ["T"]      # Terminal time (end of time horizon)
        self.M = self.params_equ["M"]      # Number of trajectories (samples/paths)
        self.N = self.params_equ["N"]      # Number of time snapshots (time steps)
        # self.D = self.params_equ["D"]      # Number of players (for multi-agent problems)
        # self.Q = torch.from_numpy(self.params_equ["Q"]).float().to(self.device)     # Weight matrix
        self.state_dim = self.params_equ["state_dim"]
        self.bm_dim = self.params_equ["BM_dim"]
        self.u_dim = self.params_equ["u_dim"]
        # self.jump_dim =self.params_equ["jump_dim"]

        # Create a multiprocessing manager to hold shared objects
        manager = mp.Manager()
        self.dict = manager.dict()  # Shared dict to store networks, optimizers, schedulers
        self.dict['net_actor'] = net_control.state_dict()
        self.control_type = self.params_equ["control_type"]
        # self.val_loss = np.inf
        # self.game_type, self.control_type,self.random_initial_state = self.params_equ["game_type"], self.params_equ["control_type"], self.params_equ["random_initial_state"]
        # self.sigma_square = self.params_equ["f_sigma_square"]
        # self.fi_param_u = self.params_equ["fi_param_u"]
        # self.fi_param_x = self.params_equ["fi_param_x"]
         
        # # For each player (or agent):
        # for i in range(self.D):
        #     # Store control network weights

        #     # Create optimizer for the actor
        #     self.dict[f'optimizer_actor_{i}'] = optim.Adam(
        #         net_control[i].parameters(), # network structure
        #         lr=self.params_train["lr_actor"]
        #     )

        #     # Learning rate scheduler for actor optimizer
        #     self.dict[f'scheduler_actor_{i}'] = optim.lr_scheduler.MultiStepLR(
        #         self.dict[f'optimizer_actor_{i}'],
        #         milestones=self.params_train["milestones_actor"],
        #         gamma=self.params_train["gamma_actor"]
        #     )

        # # Terminal condition of the game/dynamics (e.g., terminal cost)
        # self.terminal = self.params_equ["terminal"].to(self.device)
        # self.origin = self.params_equ["origin"].to(self.device)
        # self.num_of_dest = self.params_equ["num_of_dest"]
        # self.num_of_orig = self.params_equ["num_of_orig"]
        # self.Q_type = self.params_equ['Q_type']

        # # Lists to log errors and costs during training
        # self.Y_error_list = []
        # self.control_error_list = []
        self.training_cost_list = []

        # Time step size
        self.dt = self.T / self.N
        
        

        # filename= f"_{self.game_type}_{self.control_type}_D{self.D}_d{self.state_dim}_dest{self.num_of_dest}"
        # filename+=f"_orig{self.num_of_orig}_{self.Q_type}_fiu{self.fi_param_u}_fix{self.fi_param_x}_gi{self.params_equ['gi_param']}_Depth{self.params_net['depth']}"
        # if self.game_type=="aversion": filename += f"_fsigmasq{self.sigma_square}"
        # filename = filename.replace(".","_")
        # self.filename = filename


  
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
        - control_type: str, either "path_dependent" (original) or "markovian" (new)
        Returns:
        - loss_actor: scalar tensor of actor loss.
        - X: Tensor of simulated state trajectories (M, n+1, state_dim).
        """

        # Initialize loss and buffers
        X_buffer = [X0]   # to store states (X) along path

        # Require gradients for X0 for potential derivative computations
        X0.requires_grad = True

        reward = 0
        # Loop over n time steps for this loss segment
        for j in range(0, self.N):
            # current time and noise values
            t0, DW0 = t[:, j, :], DW[:, j, :]
            ## Generate the control based on control type
            if self.control_type == "path_dependent":
                # Original control: depends on time, current state, and Brownian path
                t0_block = t0.unsqueeze(1).expand(-1, 1, self.state_dim)  #  t0 -> (B, 1, d): duplicate the single column to state_dim
                # Xt
                X0_block = X0.unsqueeze(1) #xi -> (B, 1, d): just add a time dimension
                # Concatenate along time dimension
                net_input = torch.cat([t0_block,  X0_block, self.DW_mask(DW, j)], dim=1)   # (B, N+2, d)
                net_input = net_input.view(self.M,-1)
                control = net_actor(net_input) # batch * d
                
            elif self.control_type == "markovian":
                # New Markovian control: depends only on time and current state
                t0_block = t0.unsqueeze(1).expand(-1, 1, self.state_dim)  #  t0 -> (B, 1, d)
                X0_block = X0.unsqueeze(1) # Xt -> (B, 1, d)
                # Only concatenate time and current state, no Brownian path
                net_input = torch.cat([t0_block, X0_block], dim=1)   # (B, 2, d)
                net_input = net_input.view(self.M, -1)
                control = net_actor(net_input) # batch * d
                
            else:
                raise ValueError(f"Unknown control_type: {control_type}. Use 'path_dependent' or 'markovian'")

            # get current reward
            reward += self.setup.f_running(t0, X0, control)* self.dt
            
            # Forward simulate SDE dynamics (Euler + jump terms)
            drift = self.setup.mu(t0, X0, control)
            violatility = self.setup.sigma(t0, X0, control)
            X0 = X0 + drift * self.dt + violatility * DW0
            X_buffer.append(X0)

        # Calculate the terminal cost
        reward += self.setup.g_terminal(X0) 
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
        scheduler = ReduceLROnPlateau(optimizer_actor, mode='min', patience=5, factor=self.params_train["gamma_actor"])

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
            # if loss < self.val_loss:
            #     self.val_loss= loss
            #     # record the network weights
            #     for d in range(self.D):
            #         self.dict[f'best_net_actor_{d}'] = net_actor[d].state_dict()

            print('It: %d, loss_training: %.4f, loss_validation: %.4f' % (it, self.dict["loss_actor"].item(), loss))
            # Then print the updated lr
            for param_group in optimizer_actor.param_groups:
                print("Updated learning rate:", param_group['lr'])
            # if it%5 == 0:
                # if self.state_dim==1: self.generate_1d_trajectory()
                # else:self.generate_2d_trajectory()

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