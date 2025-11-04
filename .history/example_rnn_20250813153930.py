import torch
import torch.nn as nn
from typing import Tuple

class ControlNetRNN(nn.Module):
    """
    Control policy that is 𝔽_n-adapted by construction.
    Inputs:
      - t0:  (B, 1)
      - DW:  (B, N, state_dim)  Brownian increments over horizon
      - xi:  (B, state_dim)     initial state
    Output:
      - A:   (B, N, action_dim) control at each time step

    Mechanism:
      - Embed t0 and xi to initialize RNN hidden state
      - Feed the sequence of dW_k step by step into a GRU (causal)
      - At each step, map hidden -> action
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        # Embeddings for (t0, xi)
        self.t0_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.xi_embed = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )

        # Project each dW_k into hidden space before RNN
        self.dw_proj = nn.Linear(state_dim, hidden)

        # Causal RNN over time (uses only past dW)
        self.rnn = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=num_layers, batch_first=True)

        # Head to actions
        self.to_action = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, action_dim)
        )

    def forward(self, t0: torch.Tensor, DW: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        t0:  (B,1)
        DW:  (B,N,d)
        xi:  (B,d)
        returns actions A: (B,N,m)
        """
        B, N, d = DW.shape
        assert d == self.state_dim

        # Initial hidden from (t0, xi)
        h0 = self.t0_embed(t0) + self.xi_embed(xi)  # (B, hidden)
        h0 = h0.unsqueeze(0)  # (num_layers=1, B, hidden)

        # Project inputs and run GRU
        X = self.dw_proj(DW)  # (B,N,hidden)
        H, _ = self.rnn(X, h0)  # (B,N,hidden)

        # Map hidden states to actions per step
        A = self.to_action(H)  # (B,N,action_dim)
        return A


# ------------------
# Minimal example
# ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, d, m = 4, 50, 3, 2 

    t0 = torch.zeros(B, 1, device=device)           # e.g., all start at 0.0
    DW = torch.randn(B, N, d, device=device)        # Brownian increments
    xi = torch.randn(B, d, device=device)           # initial states

    net = ControlNetRNN(state_dim=d, action_dim=m, hidden=128).to(device)
    A = net(t0, DW, xi)
    print("actions shape:", A.shape)  # (B, N, m)