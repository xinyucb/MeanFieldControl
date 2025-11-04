#!/usr/bin/env python3
"""
Example: Using Markovian Control in the McKean-Vlasov Solver
This demonstrates the new control type that only depends on current state Xt
without the Brownian motion path dependency
"""

import torch
import torch.nn as nn
from DeepMKVSolver import deepMKV
from NeuralNets import Network
import numpy as np

def markovian_control_example():
    """Example using the new Markovian control"""
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    N = 50  # number of time discretization intervals
    d = 2   # dimension of Xt
    M = 100 # number of particles (reduced for demo)
    
    print("=" * 60)
    print("Markovian Control Example")
    print("=" * 60)
    
    # Create parameters for Markovian control
    params_markovian = {
        "equation": {
            "state_dim": d, 
            "BM_dim": d, 
            "u_dim": d, 
            "M": M,
            "N": N,
            "T": 1,
            "control_type": "markovian"  # New control type!
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
            "block_type": "fc",
            "inputs": 2 * d,  # Much smaller! Only time + current state
            "output": d, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
            "hidden_layers": [d * 4, d * 4], 
            "dropout_rate": 0.1,
        },
    }
    
    print(f"Original path-dependent input size: {(N + 2) * d}")
    print(f"New Markovian input size: {2 * d}")
    print(f"Reduction: {((N + 2) * d - 2 * d) / ((N + 2) * d) * 100:.1f}%")
    
    # Create control network for Markovian control
    net_control_markovian = Network(params_markovian["net"])
    print(f"\nMarkovian network parameters: {sum(p.numel() for p in net_control_markovian.parameters()):,}")
    
    # Create setup and solver (commented out as you'll need to implement this)
    # setup_markovian = Systematic(params_markovian["equation"], device)
    # model_markovian = deepMKV(net_control_markovian, setup_markovian, params_markovian)
    
    print("\nMarkovian control setup complete!")
    print("Key differences:")
    print("  - Input: time + current state only (no Brownian path)")
    print("  - Much smaller network input size")
    print("  - Standard Markovian control structure")
    print("  - Potentially faster training and inference")
    print("  - May have different performance characteristics")
    
    return params_markovian, net_control_markovian

def control_types_comparison():
    """Compare both control types"""
    
    print("\n" + "=" * 60)
    print("Control Types Comparison")
    print("=" * 60)
    
    N = 50
    d = 2
    
    print("1. Path-Dependent Control (Original):")
    print(f"   - Input size: {(N + 2) * d}")
    print("   - Dependencies: time + current state + Brownian path")
    print("   - Use case: When you need full noise history information")
    print("   - Complexity: Higher, more parameters")
    print("   - Network input: [t, Xt, W0, W1, ..., WN]")
    
    print("\n2. Markovian Control (New):")
    print(f"   - Input size: {2 * d}")
    print("   - Dependencies: time + current state only")
    print("   - Use case: Standard control problems, efficiency-focused")
    print("   - Complexity: Lower, fewer parameters")
    print("   - Network input: [t, Xt]")
    
    print("\nTo switch between them in your code:")
    print("1. Change 'control_type' in params['equation']")
    print("2. Adjust 'inputs' in params['net'] accordingly")
    print("3. The solver automatically uses the correct method")

def usage_instructions():
    """Show how to use both control types"""
    
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    
    print("1. Path-Dependent Control (Original):")
    print("   params = {")
    print('       "equation": {')
    print('           "control_type": "path_dependent",  # or omit (default)')
    print('           # ... other parameters')
    print('       },')
    print('       "net": {')
    print(f'           "inputs": (N + 2) * d,  # Time + State + Brownian path')
    print('           # ... other parameters')
    print('       }')
    print('   }')
    
    print("\n2. Markovian Control (New):")
    print("   params = {")
    print('       "equation": {')
    print('           "control_type": "markovian",  # New control type')
    print('           # ... other parameters')
    print('       },')
    print('       "net": {')
    print(f'           "inputs": 2 * d,  # Only Time + State')
    print('           # ... other parameters')
    print('       }')
    print('   }')
    
    print("\n3. The solver automatically detects the control type and:")
    print("   - Uses the appropriate input processing")
    print("   - Generates controls with or without Brownian path")
    print("   - Maintains backward compatibility")

def main():
    """Run the complete example"""
    
    print("McKean-Vlasov Solver: Markovian Control Example")
    print("=" * 80)
    
    # Run the Markovian control example
    params, network = markovian_control_example()
    
    # Show comparison
    control_types_comparison()
    
    # Show usage instructions
    usage_instructions()
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Implement your Systematic class")
    print("2. Create the setup object")
    print("3. Run the solver with the new control type")
    print("4. Compare performance between control types")

if __name__ == "__main__":
    main() 