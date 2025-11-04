# ============================================================================
# NOTEBOOK CELLS - Copy and paste these into your main.ipynb
# ============================================================================

# CELL 1: Import the configuration helper
from notebook_config import get_params, print_config_summary, switch_control_type

# CELL 2: Choose your control type and get parameters
# Choose one of these options:

# Option A: Path-Dependent Control (Original)
CONTROL_TYPE = "path_dependent"
BLOCK_TYPE = "fc"  # or "rnn"

# Option B: Markovian Control (New)
# CONTROL_TYPE = "markovian"
# BLOCK_TYPE = "fc"  # or "rnn"

# Get the complete parameter configuration
params = get_params(d, N, device, CONTROL_TYPE, BLOCK_TYPE)

# Print configuration summary
print_config_summary(params, d, N)

# CELL 3: Create control network
net_control = Network(params["net"])
print(f"\nNetwork created with {sum(p.numel() for p in net_control.parameters()):,} parameters")

# CELL 4: Create setup and solver
setup = Systematic(params["equation"], device)
model = deepMKV(net_control, setup, params)

# CELL 5: Test the setup
t, W, DW = model.fetch_minibatch()
X0 = setup.generate_initial_condition()
X_buffer, loss = model.loss_function(net_control, t, DW, X0)

print(f"Setup test successful!")
print(f"Time grid shape: {t.shape}")
print(f"Brownian motion shape: {W.shape}")
print(f"Brownian increments shape: {DW.shape}")
print(f"Initial condition shape: {X0.shape}")
print(f"Loss: {loss.item():.4f}")

# CELL 6: Run simulation
X_buffer, loss, net_actor = model.simulation_paths()
print(f"Simulation completed! Loss: {loss.item():.4f}")

# CELL 7: Train the model
print("Starting training...")
model.train_players()

# CELL 8: Plot training results
plt.figure(figsize=(10, 6))
plt.plot(model.training_cost_list)
plt.title(f'Training Loss - {CONTROL_TYPE.upper()} Control')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# CELL 9: Switch control type (optional)
# Uncomment to switch to the other control type
# if CONTROL_TYPE == "path_dependent":
#     new_params = switch_control_type(params.copy(), "markovian", d, N)
#     print("\nSwitched to Markovian control!")
# else:
#     new_params = switch_control_type(params.copy(), "path_dependent", d, N)
#     print("\nSwitched to Path-dependent control!")

# ============================================================================
# ALTERNATIVE: Simple parameter configuration without external file
# ============================================================================

# If you prefer not to use the external config file, use this instead:

# CELL A: Simple parameter configuration
params_simple = {
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
        "inputs": (N + 2) * d if params_simple["equation"]["control_type"] == "path_dependent" else 2 * d,
        "output": d, 
        "activation": "ReLU", 
        "penalty": "Tanh",
        "params_act": {"Tanh": nn.Tanh(), "ReLU": nn.ReLU()},
        "hidden_layers": [d * 4, d * 4, d * 4],
        "dropout_rate": 0.1,
    },
}

# CELL B: Print configuration
control_type = params_simple["equation"]["control_type"]
inputs = params_simple["net"]["inputs"]
print(f"Control Type: {control_type}")
print(f"Network Inputs: {inputs}")
if control_type == "markovian":
    print(f"  - Time: {d}")
    print(f"  - Current State: {d}")
    print(f"  - Brownian Path: 0 (not used)")
else:
    print(f"  - Time: {d}")
    print(f"  - Current State: {d}")
    print(f"  - Brownian Path: {N * d}")

# ============================================================================
# TROUBLESHOOTING CELLS
# ============================================================================

# CELL T1: Check if all required modules are available
try:
    from DeepMKVSolver import deepMKV
    print("✓ DeepMKVSolver imported successfully")
except ImportError as e:
    print(f"✗ Error importing DeepMKVSolver: {e}")

try:
    from NeuralNets import Network
    print("✓ NeuralNets imported successfully")
except ImportError as e:
    print(f"✗ Error importing NeuralNets: {e}")

try:
    from SystematicRisk import Systematic
    print("✓ SystematicRisk imported successfully")
except ImportError as e:
    print(f"✗ Error importing SystematicRisk: {e}")

# CELL T2: Check parameter consistency
def check_params(params):
    """Check if parameters are consistent"""
    try:
        # Check required keys
        required_keys = ["equation", "train", "net"]
        for key in required_keys:
            if key not in params:
                print(f"✗ Missing key: {key}")
                return False
        
        # Check equation parameters
        eq_keys = ["state_dim", "BM_dim", "u_dim", "M", "N", "T", "control_type"]
        for key in eq_keys:
            if key not in params["equation"]:
                print(f"✗ Missing equation key: {key}")
                return False
        
        # Check network parameters
        net_keys = ["block_type", "inputs", "output", "activation", "penalty", "params_act"]
        for key in net_keys:
            if key not in params["net"]:
                print(f"✗ Missing network key: {key}")
                return False
        
        print("✓ All required parameters are present")
        return True
        
    except Exception as e:
        print(f"✗ Error checking parameters: {e}")
        return False

# Run the check
check_params(params) 