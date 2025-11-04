# Markovian Control Feature

This document explains the new **Markovian Control** feature that has been added to your McKean-Vlasov Solver while keeping your existing **Path-Dependent Control** intact.

## Overview

Your solver now supports two control types:

1. **Path-Dependent Control** (Original) - Control depends on time, current state, and full Brownian motion path
2. **Markovian Control** (New) - Control depends only on time and current state

## What Was Added

### 1. New Control Type Parameter

In your `params["equation"]`, you can now specify:

```python
params = {
    "equation": {
        # ... other parameters ...
        "control_type": "markovian",  # New! Use "path_dependent" for original
    }
}
```

### 2. Automatic Control Generation

The solver automatically detects the control type and:
- **Path-Dependent**: Uses time + current state + Brownian path
- **Markovian**: Uses only time + current state

### 3. Network Input Size Adjustment

The network input size automatically adjusts:
- **Path-Dependent**: `(N + 2) * state_dim` (time + state + Brownian path)
- **Markovian**: `2 * state_dim` (time + state only)

## Key Benefits

### Markovian Control Advantages:
- **Smaller networks**: Significantly fewer input parameters
- **Faster training**: Less complex input processing
- **Standard structure**: Follows classical control theory
- **Efficient inference**: Faster forward passes

### Path-Dependent Control Advantages:
- **Full information**: Access to complete noise history
- **Potentially better performance**: Can exploit noise patterns
- **Proven approach**: Your existing implementation

## Implementation Details

### Modified Files:

1. **`DeepMKVSolver.py`**:
   - Added `control_type` parameter to `loss_function`
   - Updated `Actor_step` method
   - Updated `train_players` method
   - Updated `simulation_paths` method
   - Added helper methods for control type management

2. **New Example Files**:
   - `example_control_types.py` - Comprehensive examples
   - `markovian_control_example.py` - Focused Markovian example
   - `test_fc_network.py` - Testing both block types

### How It Works:

```python
def loss_function(self, net_actor, t, DW, X0, control_type="path_dependent"):
    # ... existing code ...
    
    if control_type == "path_dependent":
        # Original: time + state + Brownian path
        net_input = torch.cat([t0_block, X0_block, self.DW_mask(DW, j)], dim=1)
    elif control_type == "markovian":
        # New: only time + state
        net_input = torch.cat([t0_block, X0_block], dim=1)
    
    # ... rest of existing code ...
```

## Usage Examples

### 1. Path-Dependent Control (Original):

```python
params = {
    "equation": {
        "control_type": "path_dependent",  # or omit (default)
        # ... other parameters
    },
    "net": {
        "inputs": (N + 2) * d,  # Time + State + Brownian path
        # ... other parameters
    }
}
```

### 2. Markovian Control (New):

```python
params = {
    "equation": {
        "control_type": "markovian",  # New control type
        # ... other parameters
    },
    "net": {
        "inputs": 2 * d,  # Only Time + State
        # ... other parameters
    }
}
```

## Backward Compatibility

- **No breaking changes**: Your existing code continues to work
- **Default behavior**: If `control_type` is not specified, it defaults to `"path_dependent"`
- **Same interface**: All existing methods work unchanged
- **Automatic detection**: The solver automatically uses the correct control generation

## Performance Comparison

### Input Size Reduction:
- **N=50, d=2**: 
  - Path-dependent: `(50+2) * 2 = 104` inputs
  - Markovian: `2 * 2 = 4` inputs
  - **Reduction: 96.2%**

### Network Complexity:
- **Path-dependent**: More parameters, potentially better performance
- **Markovian**: Fewer parameters, faster training/inference

## Testing

Run the example scripts to verify both control types work:

```bash
# Test both control types
python example_control_types.py

# Test Markovian control specifically
python markovian_control_example.py

# Test network functionality
python test_fc_network.py
```

## Migration Guide

### From Path-Dependent to Markovian:

1. **Add control type**:
   ```python
   params["equation"]["control_type"] = "markovian"
   ```

2. **Reduce input size**:
   ```python
   params["net"]["inputs"] = 2 * state_dim  # Instead of (N + 2) * state_dim
   ```

3. **Recreate network** (if needed):
   ```python
   net_control = Network(params["net"])
   ```

### From Markovian to Path-Dependent:

1. **Change control type**:
   ```python
   params["equation"]["control_type"] = "path_dependent"
   ```

2. **Increase input size**:
   ```python
   params["net"]["inputs"] = (N + 2) * state_dim
   ```

## Advanced Features

### 1. Dynamic Control Type Switching:

```python
# Switch control type during runtime
model.switch_control_type("markovian")
```

### 2. Automatic Input Size Calculation:

```python
# Get required input size for any control type
input_size = model.get_network_input_size("markovian")
```

### 3. Control Type Validation:

The solver automatically validates control types and provides helpful error messages.

## Troubleshooting

### Common Issues:

1. **"Unknown control_type"**:
   - Use only `"path_dependent"` or `"markovian"`

2. **Input size mismatch**:
   - Ensure `params["net"]["inputs"]` matches your control type
   - Use the helper methods to calculate correct sizes

3. **Performance differences**:
   - Markovian control may have different convergence properties
   - Monitor training loss and adjust hyperparameters if needed

## Future Enhancements

Potential additions:
- Hybrid control types
- Adaptive control type selection
- Performance comparison tools
- Control type-specific hyperparameter optimization

## Summary

You now have a **dual-control system** that maintains all your existing functionality while adding a new, efficient Markovian control option. The implementation is:

- **Non-breaking**: Your existing code works unchanged
- **Flexible**: Easy to switch between control types
- **Efficient**: Markovian control reduces computational complexity
- **Well-tested**: Comprehensive examples and testing included

Use the control type that best fits your specific problem requirements! 