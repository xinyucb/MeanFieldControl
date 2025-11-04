# Notebook Setup Guide

This guide explains how to make your `main.ipynb` notebook runnable with both control types.

## Quick Setup

### Option 1: Use the Configuration Helper (Recommended)

1. **Copy the configuration file**:
   ```bash
   # The file notebook_config.py should be in your workspace
   ```

2. **Add this cell to your notebook** (after your imports):
   ```python
   from notebook_config import get_params, print_config_summary, switch_control_type
   ```

3. **Replace your params definition with**:
   ```python
   # Choose your control type
   CONTROL_TYPE = "path_dependent"  # or "markovian"
   BLOCK_TYPE = "fc"  # or "rnn"
   
   # Get complete configuration
   params = get_params(d, N, device, CONTROL_TYPE, BLOCK_TYPE)
   
   # Print summary
   print_config_summary(params, d, N)
   ```

### Option 2: Simple Inline Configuration

If you prefer not to use external files, replace your params with:

```python
params = {
    "equation": {
        "state_dim": d, 
        "BM_dim": d, 
        "u_dim": d, 
        "M": 500,
        "N": N,
        "T": 1,
        "control_type": "path_dependent",  # Change to "markovian" for new control
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
        "block_type": "fc",  # or "rnn"
        "inputs": (N + 2) * d if params["equation"]["control_type"] == "path_dependent" else 2 * d,
        "output": d, 
        "activation": "ReLU", 
        "penalty": "Tanh",
        "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
        "hidden_layers": [d * 4, d * 4, d * 4],
        "dropout_rate": 0.1,
    },
}
```

## Complete Working Example

Here's a complete cell sequence that will work:

```python
# Cell 1: Choose control type
CONTROL_TYPE = "path_dependent"  # or "markovian"
BLOCK_TYPE = "fc"  # or "rnn"

# Cell 2: Get parameters
params = get_params(d, N, device, CONTROL_TYPE, BLOCK_TYPE)
print_config_summary(params, d, N)

# Cell 3: Create network
net_control = Network(params["net"])

# Cell 4: Create setup and solver
setup = Systematic(params["equation"], device)
model = deepMKV(net_control, setup, params)

# Cell 5: Test setup
t, W, DW = model.fetch_minibatch()
X0 = setup.generate_initial_condition()
X_buffer, loss = model.loss_function(net_control, t, DW, X0)

# Cell 6: Run simulation
X_buffer, loss, net_actor = model.simulation_paths()

# Cell 7: Train
model.train_players()

# Cell 8: Plot results
plt.plot(model.training_cost_list)
plt.title(f'Training Loss - {CONTROL_TYPE.upper()} Control')
plt.show()
```

## Control Type Comparison

| Control Type | Input Size | Dependencies | Use Case |
|--------------|------------|--------------|----------|
| **Path-Dependent** | `(N + 2) * d` | Time + State + Brownian Path | Full information, potentially better performance |
| **Markovian** | `2 * d` | Time + State only | Efficiency, standard control theory |

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure `notebook_config.py` is in the same directory
2. **Parameter mismatch**: Check that `control_type` is set correctly
3. **Network size**: Ensure `inputs` matches your control type

### Testing Your Setup:

```python
# Test if everything is working
try:
    from notebook_config import get_params
    print("✓ Configuration helper imported")
    
    test_params = get_params(2, 50, "cpu", "markovian")
    print("✓ Parameters generated successfully")
    
    print(f"Control type: {test_params['equation']['control_type']}")
    print(f"Network inputs: {test_params['net']['inputs']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
```

## Switching Control Types

To switch between control types during development:

```python
# Switch to Markovian control
params = switch_control_type(params, "markovian", d, N)

# Switch back to path-dependent
params = switch_control_type(params, "path_dependent", d, N)

# Recreate network with new parameters
net_control = Network(params["net"])
```

## Performance Tips

1. **Start with Markovian control** for faster development
2. **Use path-dependent control** when you need maximum performance
3. **Reduce training iterations** during testing (`iteration: 5`)
4. **Monitor input sizes** to ensure consistency

## Next Steps

1. **Test the setup** with the provided cells
2. **Choose your control type** based on your needs
3. **Adjust hyperparameters** as needed
4. **Run training** and monitor results
5. **Compare performance** between control types

Your notebook should now be fully runnable with both control types! 🚀 