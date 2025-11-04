# Block Types in McKean-Vlasov Solver

This document explains the two different block types available in the McKean-Vlasov Solver and how to use them.

## Overview

The solver now supports two types of neural network blocks:

1. **ResNet Block** (`Block`) - The original implementation with two linear layers and activation functions
2. **FC Block** (`FCBlock`) - A new fully connected block with configurable hidden layers and dropout

## Block Types

### 1. ResNet Block (Original)

The original `Block` class implements a residual connection with two linear layers:

```python
class Block(nn.Module):
    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(Block, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(torch.sin(self.L1(x)))) + x
        else:
            a = self.act(self.L1(self.act(self.L2(x)))) + x
        return a
```

**Characteristics:**
- Fixed architecture: 2 linear layers with same input/output dimensions
- Residual connection: `output = f(x) + x`
- Supports various activation functions including custom ones
- Lightweight and fast

### 2. FC Block (New)

The new `FCBlock` class implements a fully connected network with configurable architecture:

```python
class FCBlock(nn.Module):
    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        # Configurable hidden layers
        hidden_layers = params.get("hidden_layers", [inputs, inputs])
        dropout_rate = params.get("dropout_rate", 0.0)
        
        # Build network with batch normalization and dropout
        # ... (see implementation for details)
        
    def forward(self, x):
        output = self.network(x)
        return output + x  # Residual connection
```

**Characteristics:**
- Configurable hidden layer sizes
- Batch normalization for better training
- Dropout for regularization
- Residual connection: `output = f(x) + x`
- More flexible architecture
- Potentially better performance with proper tuning

## Usage

### Choosing Block Type

In your network configuration, specify the block type:

```python
params = {
    "net": {
        # ... other parameters ...
        "block_type": "rnn",  # Use RNN blocks (default)
        # OR
        "block_type": "fc",   # Use FC blocks
    }
}
```

### ResNet Block Configuration

```python
params["net"] = {
    "inputs": 2 * D * state_dim + 1,
    "width": D * state_dim * 1 + 10,
    "depth": 4,
    "output": state_dim,
    "activation": "ReLU",
    "penalty": "Tanh",
    "block_type": "rnn",  # Use RNN blocks
    "params_act": {
        "Tanh": nn.Tanh(),
        "ReLU": nn.ReLU(),
        # ... other activations
    }
}
```

### FC Block Configuration

```python
params["net"] = {
    "inputs": 2 * D * state_dim + 1,
    "width": D * state_dim * 1 + 10,
    "depth": 4,
    "output": state_dim,
    "activation": "ReLU",
    "penalty": "Tanh",
    "block_type": "fc",  # Use FC blocks
    "params_act": {
        "ReLU": nn.ReLU(),
        # ... other activations
    },
    # FC-specific parameters
    "hidden_layers": [32, 64, 32],  # Custom hidden layer sizes
    "dropout_rate": 0.1,            # Dropout rate
}
```

## Configuration Examples

### Using the Configuration Helper

```python
from network_configs import get_rnn_network_config, get_fc_network_config

# RNN network
params["net"] = get_rnn_network_config(D, state_dim)

# FC network (default)
params["net"] = get_fc_network_config(D, state_dim)

# FC network (custom)
params["net"] = get_fc_network_config(
    D, state_dim,
    hidden_layers=[64, 128, 64],
    dropout_rate=0.2
)
```

### Predefined FC Configurations

```python
from network_configs import (
    get_lightweight_fc_config,
    get_deep_fc_config,
    get_wide_fc_config
)

# Lightweight FC network
params["net"] = get_lightweight_fc_config(D, state_dim)

# Deep FC network
params["net"] = get_deep_fc_config(D, state_dim)

# Wide FC network
params["net"] = get_wide_fc_config(D, state_dim)
```

## Migration Guide

### From RNN to FC Blocks

1. **Change block type:**
   ```python
   "block_type": "rnn" → "block_type": "fc"
   ```

2. **Add FC-specific parameters:**
   ```python
   "hidden_layers": [32, 64, 32],  # Customize as needed
   "dropout_rate": 0.1,            # Adjust regularization
   ```

3. **Remove RNN-specific parameters** (if any):
   ```python
   # These are still needed for backward compatibility
   "params_act": {...}
   ```

### Backward Compatibility

- If `block_type` is not specified, it defaults to `"rnn"`
- All existing configurations will continue to work without changes
- The `params_act` parameter is still required for both block types

## Performance Considerations

### RNN Blocks
- **Pros:** Fast, lightweight, proven performance
- **Cons:** Fixed architecture, limited flexibility
- **Best for:** Quick prototyping, resource-constrained environments

### FC Blocks
- **Pros:** Flexible architecture, better regularization, potentially better performance
- **Cons:** More parameters, requires tuning, slower training
- **Best for:** Production systems, when you need better performance

## Testing

Run the test scripts to verify both block types work correctly:

```bash
# Test both block types
python test_fc_network.py

# Run examples
python example_usage.py

# View configurations
python network_configs.py
```

## Troubleshooting

### Common Issues

1. **Missing `params_act`:**
   ```
   Error: 'params_act' key not found
   ```
   **Solution:** Always include `params_act` with your activation functions

2. **Invalid `block_type`:**
   ```
   Error: Unknown block_type
   ```
   **Solution:** Use only `"rnn"` or `"fc"`

3. **FC parameters ignored:**
   ```
   Warning: hidden_layers parameter ignored for RNN blocks
   ```
   **Solution:** Set `block_type: "fc"` to use FC-specific parameters

### Performance Tips

1. **Start with RNN blocks** for initial development
2. **Switch to FC blocks** when you need better performance
3. **Tune hidden layers** based on your problem complexity
4. **Adjust dropout rate** based on overfitting/underfitting
5. **Monitor parameter count** to avoid excessive memory usage

## Examples in Main Scripts

The main test scripts have been updated to use FC blocks by default:

- `main_test_flocking_game.py` - Uses FC blocks
- `main_test_aversion_game.py` - Uses FC blocks

To switch back to RNN blocks, change:
```python
"block_type": "fc" → "block_type": "rnn"
``` 