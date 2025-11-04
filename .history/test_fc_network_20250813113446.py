#!/usr/bin/env python3
"""
Test script for the new fully connected Block class
"""

import torch
import torch.nn as nn
from DeepMKVSolver import Block

def test_fc_block():
    """Test the fully connected Block class"""
    
    print("Testing Fully Connected Block Class")
    print("=" * 40)
    
    # Test parameters
    inputs = 10
    batch_size = 5
    
    # Test different configurations
    test_configs = [
        {
            "name": "Default configuration (backward compatible)",
            "params": {},
            "activation": "Tanh"
        },
        {
            "name": "Custom hidden layers",
            "params": {
                "hidden_layers": [20, 30, 20],
                "dropout_rate": 0.1
            },
            "activation": "ReLU"
        },
        {
            "name": "Deep network with dropout",
            "params": {
                "hidden_layers": [32, 64, 128, 64, 32],
                "dropout_rate": 0.2
            },
            "activation": "GELU"
        }
    ]
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        print(f"  Activation: {config['activation']}")
        print(f"  Params: {config['params']}")
        
        try:
            # Create block
            block = Block(inputs, config['params'], config['activation'])
            
            # Create test input
            x = torch.randn(batch_size, inputs)
            
            # Forward pass
            with torch.no_grad():
                output = block(x)
            
            # Check output shape
            expected_shape = (batch_size, inputs)
            if output.shape == expected_shape:
                print(f"  ✓ Output shape: {output.shape}")
            else:
                print(f"  ✗ Expected shape: {expected_shape}, got: {output.shape}")
            
            # Check that residual connection works
            residual_diff = torch.abs(output - x).max().item()
            if residual_diff < 1e-3:
                print(f"  ✗ Residual connection may not be working (diff: {residual_diff:.6f})")
            else:
                print(f"  ✓ Residual connection working (diff: {residual_diff:.6f})")
            
            # Print network structure
            total_params = sum(p.numel() for p in block.parameters())
            print(f"  Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 40)
    print("Test completed!")

if __name__ == "__main__":
    test_fc_block() 