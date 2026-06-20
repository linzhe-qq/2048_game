import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int,
                 hidden_size: int = 256, num_hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sync_target(self, source: "DQNNetwork"):
        self.load_state_dict(source.state_dict())
