#!/usr/bin/env python3
"""
Configuration file for the main.ipynb notebook
Provides parameter configurations for both control types
"""

import torch
import torch.nn as nn

def get_params(d, N, device, control_type="path_dependent", block_type="fc"):
    """
    Get complete parameter configuration for the McKean-Vlasov Solver
    
    Args:
        d: state dimension
        N: number of time intervals
        device: torch device
        control_type: "path_dependent" or "markovian"
        block_type: "fc" or "rnn"
    
    Returns:
        dict: complete parameter configuration
    """
    
    params = {
        # Parameters of the game setup
        "equation": {
            "state_dim": d, 
            "BM_dim": d, 
            "u_dim": d, 
            "M": 500,  # number of particles
            "N": N,    # number of time intervals
            "T": 1,
            "control_type": control_type,  # New control type parameter
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 5,  # Reduced for demo
            "epochs": 50, 
            "device": device,  
        },
        "net": {
            # Choose block type: "rnn" for original RNN block, "fc" for fully connected block
            "block_type": block_type,
            
            # Network architecture - automatically adjusted based on control type
            "inputs": (N + 2) * d,  # Will be adjusted based on control_type
            "output": d, 
            
            # RNN network parameters (only used when block_type="rnn")
            "width": N + 10, 
            "depth": 4, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
            
            # Fully connected network parameters (only used when block_type="fc")
            "hidden_layers": [d * 4, d * 4, d * 4],  # Simplified hidden layers
            "dropout_rate": 0.1,  # Add dropout for regularization
        },
    }
    
    # Adjust network input size based on control type
    if control_type == "markovian":
        # For Markovian control: only time + current state
        params["net"]["inputs"] = 2 * d
    else:
        # For path-dependent control: time + current state + Brownian path
        params["net"]["inputs"] = (N + 2) * d
    
    return params

def print_config_summary(params, d, N):
    """Print a summary of the current configuration"""
    
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    
    control_type = params["equation"]["control_type"]
    block_type = params["net"]["block_type"]
    inputs = params["net"]["inputs"]
    
    print(f"Control Type: {control_type.upper()}")
    print(f"Block Type: {block_type.upper()}")
    print(f"State Dimension: {d}")
    print(f"Time Intervals: {N}")
    print(f"Network Inputs: {inputs}")
    
    if control_type == "markovian":
        print(f"  - Time: {d}")
        print(f"  - Current State: {d}")
        print(f"  - Brownian Path: 0 (not used)")
    else:
        print(f"  - Time: {d}")
        print(f"  - Current State: {d}")
        print(f"  - Brownian Path: {N * d}")
    
    print(f"\nTraining Parameters:")
    print(f"  - Iterations: {params['train']['iteration']}")
    print(f"  - Epochs per iteration: {params['train']['epochs']}")
    print(f"  - Learning rate: {params['train']['lr_actor']}")
    
    print(f"\nNetwork Parameters:")
    if block_type == "fc":
        print(f"  - Hidden layers: {params['net']['hidden_layers']}")
        print(f"  - Dropout rate: {params['net']['dropout_rate']}")
    else:
        print(f"  - Width: {params['net']['width']}")
        print(f"  - Depth: {params['net']['depth']}")
    
    print("=" * 60)

def get_path_dependent_config(d, N, device, block_type="fc"):
    """Get configuration for path-dependent control"""
    return get_params(d, N, device, "path_dependent", block_type)

def get_markovian_config(d, N, device, block_type="fc"):
    """Get configuration for Markovian control"""
    return get_params(d, N, device, "markovian", block_type)

def switch_control_type(params, new_control_type, d, N):
    """
    Switch between control types and update parameters accordingly
    
    Args:
        params: existing parameter dictionary
        new_control_type: "path_dependent" or "markovian"
        d: state dimension
        N: number of time intervals
    
    Returns:
        dict: updated parameters
    """
    
    if new_control_type not in ["path_dependent", "markovian"]:
        raise ValueError(f"Unknown control_type: {new_control_type}. Use 'path_dependent' or 'markovian'")
    
    # Update control type
    params["equation"]["control_type"] = new_control_type
    
    # Update network input size
    if new_control_type == "markovian":
        params["net"]["inputs"] = 2 * d
    else:
        params["net"]["inputs"] = (N + 2) * d
    
    print(f"Switched to {new_control_type} control")
    print(f"Network input size updated to: {params['net']['inputs']}")
    
    return params

# Example usage configurations
if __name__ == "__main__":
    # Example parameters
    d = 2
    N = 50
    device = "cpu"
    
    print("Example Configurations")
    print("=" * 40)
    
    # Path-dependent control
    print("\n1. Path-Dependent Control:")
    params_pd = get_path_dependent_config(d, N, device)
    print_config_summary(params_pd, d, N)
    
    # Markovian control
    print("\n2. Markovian Control:")
    params_mk = get_markovian_config(d, N, device)
    print_config_summary(params_mk, d, N)
    
    # Switch control type
    print("\n3. Switching Control Type:")
    params_switched = switch_control_type(params_pd.copy(), "markovian", d, N)
    print_config_summary(params_switched, d, N) 