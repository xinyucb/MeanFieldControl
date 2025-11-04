#!/usr/bin/env python3
"""
Simple Path Dependent McKean Vlasov Control Solver Test
"""

from tokenize import PlainToken
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import time
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from NeuralNets import Network
import os
# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set default tensor type
d_type = torch.float32
torch.set_default_dtype(d_type)
import warnings
warnings.filterwarnings("ignore")
from Param import DefaultParams


def main():
    """Main function to run the MKV solver test."""
    
    print("=" * 50)
    print("Path Dependent Solver Test (Low Dimension)")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # Parameters Setup 
    # ============================================================================
    params = DefaultParams(d=2, control_type='X').params
    setup = HighDimPathModel(params) 
    
   


    
    # ============================================================================
    # Create control networks
    # ============================================================================
    net_control = [Network(params["net"]) for _ in range(D)]  # u=(u1, u2, ..., uD)
    
    # ============================================================================
    # Create and configure the model
    # ============================================================================
    model = deepPG(net_control, params).to(device)
    
    # Update filename with diffusion parameters
    model.filename = model.filename + f"sigma{sigma}_sigma0{sigma0}_gamma{gamma}_gamma0{gamma0}".replace(".", "_")
    
    print(f"Model filename: {model.filename}")

    # ============================================================================
    # Check for pretrained models and load if available
    # ============================================================================

    
    # Check for the specific pretrained model file
    model_file_path = f"outputNN/{model.filename}.pth"
    
    if os.path.exists(model_file_path):
        print(f"\nFound pretrained model: {model_file_path}")
        print(f"Loading pretrained model: {os.path.basename(model_file_path)}")
        
        try:
            loaded_state = torch.load(model_file_path, map_location=device)
            
            # Load state dict for each player's network
            for d in range(D):
                if f'net_actor_{d}' in loaded_state:
                    net_control[d].load_state_dict(loaded_state[f'net_actor_{d}'])
                    print(f"  Loaded network for player {d+1}")
                else:
                    print(f"  Warning: No saved state found for player {d+1}")
            
            print("Pretrained model loaded successfully!")
            skip_training = True
            
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Will train from scratch instead.")
            skip_training = False
    else:
        print(f"\nNo pretrained model found at: {model_file_path}")
        print("Will train from scratch.")
        skip_training = False

    # ============================================================================
    # Train the model (only if no pretrained model was loaded)
    # ============================================================================
    if not skip_training:
        print("\nStarting training...")
        model.train_players()
    else:
        print("\nSkipping training - using pretrained model.")

    
    # ============================================================================
    # Generate trajectories
    # ============================================================================
    print("\nGenerating trajectories...")
    if state_dim == 1:
        _ = model.generate_1d_trajectory(plot_mean=True)
        model.generate_animation(plot_mean=True)
        model.generate_animation(plot_mean=False)
    else:
        _ = model.generate_2d_trajectory(plot_mean=False, save=False, filename=model.filename + f"sample{1}")
        model.generate_animation(plot_mean=False, filename=model.filename + f"sample{1}")
        model.generate_animation(plot_mean=True)

    # ============================================================================
    # Save the model
    # ============================================================================
    print("\nSaving model...")
    model.save_NN()
    
    # ============================================================================
    # Plot and save results
    # ============================================================================
    print("\nPlotting loss results...")
    model.save_loss()
    
    
    print(f"\nFlocking game test completed.")
    print(f"Loss plot saved as: output/loss{model.filename}.png")
    print(f"Loss data saved as: outputLoss/{model.filename}_val_loss.csv")
    print(f"Model saved as: outputNN/{model.filename}")

if __name__ == "__main__":
    main() 