#!/usr/bin/env python3
"""
Example script demonstrating both control types in the McKean-Vlasov Solver
"""

import torch
import torch.nn as nn
from DeepMKVSolver import deepMKV
from NeuralNets import Network
from Parameters import GameConfig
import numpy as np

def example_path_dependent_control():
    """Example using the original path-dependent control"""
    print("=" * 60)
    print("Example 1: Path-Dependent Control (Original)")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    N = 50  # number of time discretization intervals
    d = 2   # dimension of Xt
    M = 100 # number of particles (reduced for demo)
    
    # Network parameters for path-dependent control
    params = {
        "equation": {
            "state_dim": d, 
            "BM_dim": d, 
            "u_dim": d, 
            "M": M,
            "N": N,
            "T": 1,
            "control_type": "path_dependent"  # Original control type
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 2,  # Reduced for demo
            "epochs": 10,     # Reduced for demo
            "device": device,  
        },
        "net": {
            "block_type": "fc",  # Use fully connected blocks
            "inputs": (N + 2) * d,  # Time + State + Brownian path
            "output": d, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
            "hidden_layers": [d * 4, d * 4],  # Simple hidden layers
            "dropout_rate": 0.1,
        },
    }
    
    print(f"Network input size: {params['net']['inputs']}")
    print(f"Input breakdown:")
    print(f"  - Time: {d}")
    print(f"  - Current state: {d}")
    print(f"  - Brownian path: {N * d}")
    print()
    
    # Create control network
    net_control = Network(params["net"])
    print(f"Network created with {sum(p.numel() for p in net_control.parameters()):,} parameters")
    
    # Create setup (you'll need to implement this based on your Systematic class)
    # setup = Systematic(params["equation"], device)
    
    # Create solver
    # model = deepMKV(net_control, setup, params)
    
    print("Path-dependent control setup complete!")
    print("This control depends on:")
    print("  - Current time t")
    print("  - Current state Xt")
    print("  - Full Brownian motion path up to current time")
    print()

def example_markovian_control():
    """Example using the new Markovian control"""
    print("=" * 60)
    print("Example 2: Markovian Control (New)")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    N = 50  # number of time discretization intervals
    d = 2   # dimension of Xt
    M = 100 # number of particles (reduced for demo)
    
    # Network parameters for Markovian control
    params = {
        "equation": {
            "state_dim": d, 
            "BM_dim": d, 
            "u_dim": d, 
            "M": M,
            "N": N,
            "T": 1,
            "control_type": "markovian"  # New control type
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 2,  # Reduced for demo
            "epochs": 10,     # Reduced for demo
            "device": device,  
        },
        "net": {
            "block_type": "fc",  # Use fully connected blocks
            "inputs": 2 * d,     # Only Time + State (much smaller!)
            "output": d, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
            "hidden_layers": [d * 4, d * 4],  # Simple hidden layers
            "dropout_rate": 0.1,
        },
    }
    
    print(f"Network input size: {params['net']['inputs']}")
    print(f"Input breakdown:")
    print(f"  - Time: {d}")
    print(f"  - Current state: {d}")
    print(f"  - Brownian path: 0 (not used)")
    print()
    
    # Create control network
    net_control = Network(params["net"])
    print(f"Network created with {sum(p.numel() for p in net_control.parameters()):,} parameters")
    
    # Create setup (you'll need to implement this based on your Systematic class)
    # setup = Systematic(params["equation"], device)
    
    # Create solver
    # model = deepMKV(net_control, setup, params)
    
    print("Markovian control setup complete!")
    print("This control depends only on:")
    print("  - Current time t")
    print("  - Current state Xt")
    print("  - No Brownian motion path dependency")
    print()

def example_switching_control_types():
    """Example showing how to switch between control types"""
    print("=" * 60)
    print("Example 3: Switching Between Control Types")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    N = 50  # number of time discretization intervals
    d = 2   # dimension of Xt
    M = 100 # number of particles
    
    # Start with path-dependent control
    params = {
        "equation": {
            "state_dim": d, 
            "BM_dim": d, 
            "u_dim": d, 
            "M": M,
            "N": N,
            "T": 1,
            "control_type": "path_dependent"  # Start with path-dependent
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 2,
            "epochs": 10,
            "device": device,  
        },
        "net": {
            "block_type": "fc",
            "inputs": (N + 2) * d,  # Path-dependent input size
            "output": d, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
            "hidden_layers": [d * 4, d * 4],
            "dropout_rate": 0.1,
        },
    }
    
    print("Starting with path-dependent control...")
    print(f"Initial network input size: {params['net']['inputs']}")
    
    # Create control network
    net_control = Network(params["net"])
    print(f"Initial network parameters: {sum(p.numel() for p in net_control.parameters()):,}")
    
    # Create setup (you'll need to implement this based on your Systematic class)
    # setup = Systematic(params["equation"], device)
    
    # Create solver
    # model = deepMKV(net_control, setup, params)
    
    print("\nNow switching to Markovian control...")
    
    # Switch to Markovian control
    params["equation"]["control_type"] = "markovian"
    params["net"]["inputs"] = 2 * d  # Update input size
    
    print(f"Updated network input size: {params['net']['inputs']}")
    print("Note: You would need to recreate the network with the new input size")
    print("or use the switch_control_type method if available")
    
    print("\nControl type switching demonstration complete!")

def main():
    """Run all examples"""
    print("McKean-Vlasov Solver: Control Types Examples")
    print("=" * 80)
    
    example_path_dependent_control()
    example_markovian_control()
    example_switching_control_types()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("\nKey differences between control types:")
    print("1. Path-Dependent Control:")
    print("   - Input: time + current state + Brownian path")
    print("   - Input size: (N + 2) * state_dim")
    print("   - Can use full information about noise history")
    print("   - More complex, potentially better performance")
    print()
    print("2. Markovian Control:")
    print("   - Input: time + current state only")
    print("   - Input size: 2 * state_dim")
    print("   - Simpler, more efficient")
    print("   - Standard in many control applications")
    print()
    print("To use in your main scripts:")
    print("1. Set 'control_type': 'path_dependent' or 'markovian' in params['equation']")
    print("2. Adjust 'inputs' in params['net'] accordingly")
    print("3. The solver will automatically use the correct control generation method")

if __name__ == "__main__":
    main() 