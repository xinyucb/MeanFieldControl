#!/usr/bin/env python3
"""
Example usage of both Block types in the McKean-Vlasov Solver
"""

import torch
import torch.nn as nn
from DeepMKVSolver import Block, FCBlock, Network

def example_rnn_network():
    """Example of using the original RNN Block"""
    print("Example 1: RNN Block Network")
    print("-" * 40)
    
    # Network parameters for RNN block
    params_rnn = {
        "inputs": 10,
        "width": 10,
        "depth": 3,
        "output": 5,
        "activation": "Tanh",
        "params_act": {"Tanh": nn.Tanh()},
        "block_type": "rnn"  # Use original RNN block
    }
    
    # Create network
    network = Network(params_rnn)
    
    # Test input
    x = torch.randn(2, 10)  # batch_size=2, input_dim=10
    
    # Forward pass
    with torch.no_grad():
        output = network(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()

def example_fc_network():
    """Example of using the new FCBlock"""
    print("Example 2: Fully Connected Block Network")
    print("-" * 40)
    
    # Network parameters for FC block
    params_fc = {
        "inputs": 10,
        "width": 10,
        "depth": 3,
        "output": 5,
        "activation": "ReLU",
        "params_act": {"ReLU": nn.ReLU()},
        "block_type": "fc",  # Use fully connected block
        "hidden_layers": [20, 30, 20],  # Custom hidden layer sizes
        "dropout_rate": 0.1  # Add dropout for regularization
    }
    
    # Create network
    network = Network(params_fc)
    
    # Test input
    x = torch.randn(2, 10)  # batch_size=2, input_dim=10
    
    # Forward pass
    with torch.no_grad():
        output = network(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()

def example_comparison():
    """Compare both approaches"""
    print("Example 3: Comparison of Both Approaches")
    print("-" * 40)
    
    # Common parameters
    inputs = 8
    width = 8
    depth = 2
    output = 4
    
    # RNN Network
    params_rnn = {
        "inputs": inputs,
        "width": width,
        "depth": depth,
        "output": output,
        "activation": "Tanh",
        "params_act": {"Tanh": nn.Tanh()},
        "block_type": "rnn"
    }
    
    # FC Network
    params_fc = {
        "inputs": inputs,
        "width": width,
        "depth": depth,
        "output": output,
        "activation": "ReLU",
        "params_act": {"ReLU": nn.ReLU()},
        "block_type": "fc",
        "hidden_layers": [16, 16],  # Two hidden layers of size 16
        "dropout_rate": 0.1
    }
    
    # Create both networks
    network_rnn = Network(params_rnn)
    network_fc = Network(params_fc)
    
    # Test input
    x = torch.randn(3, inputs)
    
    # Forward pass
    with torch.no_grad():
        output_rnn = network_rnn(x)
        output_fc = network_fc(x)
    
    print("RNN Block Network:")
    print(f"  Parameters: {sum(p.numel() for p in network_rnn.parameters()):,}")
    print(f"  Output shape: {output_rnn.shape}")
    
    print("\nFC Block Network:")
    print(f"  Parameters: {sum(p.numel() for p in network_fc.parameters()):,}")
    print(f"  Output shape: {output_fc.shape}")
    
    # Check if outputs are different (they should be due to different architectures)
    diff = torch.abs(output_rnn - output_fc).max().item()
    print(f"\nOutput difference: {diff:.6f}")

def main():
    """Run all examples"""
    print("McKean-Vlasov Solver: Block Type Examples")
    print("=" * 50)
    
    example_rnn_network()
    example_fc_network()
    example_comparison()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use in your main scripts:")
    print("1. For RNN blocks: set 'block_type': 'rnn' (default)")
    print("2. For FC blocks: set 'block_type': 'fc' and add 'hidden_layers' and 'dropout_rate'")

if __name__ == "__main__":
    main() 