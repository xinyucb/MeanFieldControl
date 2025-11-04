#!/usr/bin/env python3
"""
Test script for both the original RNN Block and new FCBlock classes
"""

import torch
import torch.nn as nn
from DeepMKVSolver import Block, FCBlock, Network

def test_blocks():
    """Test both Block and FCBlock classes"""
    
    print("Testing Block Classes")
    print("=" * 50)
    
    # Test parameters
    inputs = 10
    batch_size = 5
    
    # Test original RNN Block
    print("\n1. Testing Original RNN Block:")
    print("-" * 30)
    
    try:
        # Create RNN block
        rnn_block = Block(inputs, {"Tanh": nn.Tanh()}, "Tanh")
        
        # Create test input
        x = torch.randn(batch_size, inputs)
        
        # Forward pass
        with torch.no_grad():
            output = rnn_block(x)
        
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
        total_params = sum(p.numel() for p in rnn_block.parameters())
        print(f"  Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test new FCBlock
    print("\n2. Testing New FCBlock:")
    print("-" * 30)
    
    test_configs = [
        {
            "name": "Default configuration",
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
        print(f"\n  {config['name']}:")
        print(f"    Activation: {config['activation']}")
        print(f"    Params: {config['params']}")
        
        try:
            # Create FC block
            fc_block = FCBlock(inputs, config['params'], config['activation'])
            
            # Create test input
            x = torch.randn(batch_size, inputs)
            
            # Forward pass
            with torch.no_grad():
                output = fc_block(x)
            
            # Check output shape
            expected_shape = (batch_size, inputs)
            if output.shape == expected_shape:
                print(f"    ✓ Output shape: {output.shape}")
            else:
                print(f"    ✗ Expected shape: {expected_shape}, got: {output.shape}")
            
            # Check that residual connection works
            residual_diff = torch.abs(output - x).max().item()
            if residual_diff < 1e-3:
                print(f"    ✗ Residual connection may not be working (diff: {residual_diff:.6f})")
            else:
                print(f"    ✓ Residual connection working (diff: {residual_diff:.6f})")
            
            # Print network structure
            total_params = sum(p.numel() for p in fc_block.parameters())
            print(f"    Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Test Network class with both block types
    print("\n3. Testing Network Class:")
    print("-" * 30)
    
    network_configs = [
        {
            "name": "RNN Block Network",
            "params": {
                "inputs": inputs,
                "width": inputs,
                "depth": 2,
                "output": inputs,
                "activation": "Tanh",
                "params_act": {"Tanh": nn.Tanh()},
                "block_type": "rnn"
            }
        },
        {
            "name": "FC Block Network",
            "params": {
                "inputs": inputs,
                "width": inputs,
                "depth": 2,
                "output": inputs,
                "activation": "ReLU",
                "params_act": {"ReLU": nn.ReLU()},
                "block_type": "fc",
                "hidden_layers": [20, 20],
                "dropout_rate": 0.1
            }
        }
    ]
    
    for config in network_configs:
        print(f"\n  {config['name']}:")
        print(f"    Block type: {config['params'].get('block_type', 'rnn')}")
        
        try:
            # Create network
            network = Network(config['params'])
            
            # Create test input
            x = torch.randn(batch_size, inputs)
            
            # Forward pass
            with torch.no_grad():
                output = network(x)
            
            # Check output shape
            expected_shape = (batch_size, inputs)
            if output.shape == expected_shape:
                print(f"    ✓ Output shape: {output.shape}")
            else:
                print(f"    ✗ Expected shape: {expected_shape}, got: {output.shape}")
            
            # Print network structure
            total_params = sum(p.numel() for p in network.parameters())
            print(f"    Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_blocks() 