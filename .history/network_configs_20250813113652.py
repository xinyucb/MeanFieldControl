#!/usr/bin/env python3
"""
Network configuration examples for the McKean-Vlasov Solver
Shows how to configure both RNN and FC block types
"""

import torch.nn as nn

def get_rnn_network_config(D, state_dim, game_type="flocking"):
    """
    Get network configuration for RNN blocks (original approach)
    
    Args:
        D: number of players
        state_dim: state dimension
        game_type: type of game ("flocking" or "aversion")
    
    Returns:
        dict: network configuration
    """
    return {
        "inputs": 2 * D * state_dim + 1,  # time + X + Y
        "width": D * state_dim * 1 + 10,   # hidden layer width
        "depth": 4,                         # number of blocks
        "output": state_dim,                # control dimension
        "activation": "ReLU",
        "penalty": "Tanh",
        "block_type": "rnn",               # Use original RNN blocks
        "params_act": {
            "Tanh": nn.Tanh(),
            "Tanhshrink": nn.Tanhshrink(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6()
        }
    }

def get_fc_network_config(D, state_dim, game_type="flocking", 
                         hidden_layers=None, dropout_rate=0.1):
    """
    Get network configuration for FC blocks (new approach)
    
    Args:
        D: number of players
        state_dim: state dimension
        game_type: type of game ("flocking" or "aversion")
        hidden_layers: list of hidden layer sizes (optional)
        dropout_rate: dropout rate for regularization (optional)
    
    Returns:
        dict: network configuration
    """
    if hidden_layers is None:
        # Default hidden layers: 3 layers with increasing width
        hidden_layers = [
            D * state_dim * 2 + 20,
            D * state_dim * 2 + 20,
            D * state_dim * 2 + 20
        ]
    
    return {
        "inputs": 2 * D * state_dim + 1,  # time + X + Y
        "width": D * state_dim * 1 + 10,   # hidden layer width
        "depth": 4,                         # number of blocks
        "output": state_dim,                # control dimension
        "activation": "ReLU",
        "penalty": "Tanh",
        "block_type": "fc",                # Use fully connected blocks
        "params_act": {
            "Tanh": nn.Tanh(),
            "Tanhshrink": nn.Tanhshrink(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6()
        },
        # FC-specific parameters
        "hidden_layers": hidden_layers,
        "dropout_rate": dropout_rate
    }

def get_lightweight_fc_config(D, state_dim):
    """Lightweight FC configuration with fewer parameters"""
    return get_fc_network_config(
        D, state_dim,
        hidden_layers=[D * state_dim + 10, D * state_dim + 10],
        dropout_rate=0.05
    )

def get_deep_fc_config(D, state_dim):
    """Deep FC configuration with many layers"""
    return get_fc_network_config(
        D, state_dim,
        hidden_layers=[32, 64, 128, 256, 128, 64, 32],
        dropout_rate=0.2
    )

def get_wide_fc_config(D, state_dim):
    """Wide FC configuration with large hidden layers"""
    return get_fc_network_config(
        D, state_dim,
        hidden_layers=[D * state_dim * 4, D * state_dim * 4, D * state_dim * 4],
        dropout_rate=0.15
    )

# Example usage configurations
if __name__ == "__main__":
    D = 4
    state_dim = 2
    
    print("Network Configuration Examples")
    print("=" * 50)
    
    print("\n1. RNN Network (Original):")
    rnn_config = get_rnn_network_config(D, state_dim)
    print(f"   Block type: {rnn_config['block_type']}")
    print(f"   Depth: {rnn_config['depth']}")
    print(f"   Width: {rnn_config['width']}")
    
    print("\n2. FC Network (Default):")
    fc_config = get_fc_network_config(D, state_dim)
    print(f"   Block type: {fc_config['block_type']}")
    print(f"   Depth: {fc_config['depth']}")
    print(f"   Hidden layers: {fc_config['hidden_layers']}")
    print(f"   Dropout rate: {fc_config['dropout_rate']}")
    
    print("\n3. Lightweight FC Network:")
    light_config = get_lightweight_fc_config(D, state_dim)
    print(f"   Hidden layers: {light_config['hidden_layers']}")
    print(f"   Dropout rate: {light_config['dropout_rate']}")
    
    print("\n4. Deep FC Network:")
    deep_config = get_deep_fc_config(D, state_dim)
    print(f"   Hidden layers: {deep_config['hidden_layers']}")
    print(f"   Dropout rate: {deep_config['dropout_rate']}")
    
    print("\n5. Wide FC Network:")
    wide_config = get_wide_fc_config(D, state_dim)
    print(f"   Hidden layers: {wide_config['hidden_layers']}")
    print(f"   Dropout rate: {wide_config['dropout_rate']}")
    
    print("\n" + "=" * 50)
    print("To use these configurations in your main scripts:")
    print("1. Import this file: from network_configs import get_rnn_network_config, get_fc_network_config")
    print("2. Choose your config: params['net'] = get_fc_network_config(D, state_dim)")
    print("3. Or customize: params['net'] = get_fc_network_config(D, state_dim, hidden_layers=[64, 64], dropout_rate=0.1)") 