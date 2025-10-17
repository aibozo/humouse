"""Shared neural network components."""
from __future__ import annotations

import torch
from torch import nn


def mlp(*layers: int, activation: type[nn.Module] | None = nn.ReLU) -> nn.Sequential:
    modules: list[nn.Module] = []
    for idx, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
        modules.append(nn.Linear(in_dim, out_dim))
        if activation is not None and idx < len(layers) - 2:
            modules.append(activation())
    return nn.Sequential(*modules)


class ResidualTCN(nn.Module):
    """Temporal convolutional network with residual connections."""

    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden_channels, kernel_size, padding='same'),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = hidden_channels
        self.layers = nn.ModuleList(layers)
        self.final_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding='same')
        self.res_proj = nn.Conv1d(input_channels, hidden_channels, kernel_size=1) if input_channels != hidden_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        return torch.relu(out + residual)

