from __future__ import annotations

from typing import cast

import torch
from torch import nn


class GatedResidualNetwork(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(size, size), nn.ELU(), nn.Linear(size, size))
        self.gate = nn.Sequential(nn.Linear(size, size), nn.Sigmoid())
        self.norm = nn.LayerNorm(size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(values)
        return cast(torch.Tensor, self.norm(values + self.gate(values) * hidden))


class TftStyleForecaster(nn.Module):
    """Memory-bounded TFT with variable selection, GRN, recurrence and temporal attention."""

    def __init__(self, input_size: int, hidden_size: int = 32, horizons: int = 2, quantiles: int = 3) -> None:
        super().__init__()
        self.variable_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, input_size), nn.Softmax(dim=-1)
        )
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_grn = GatedResidualNetwork(hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.output_grn = GatedResidualNetwork(hidden_size)
        self.head = nn.Linear(hidden_size, horizons * quantiles)
        self.horizons, self.quantiles = horizons, quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.input_grn(self.input_projection(x * self.variable_gate(x)))
        encoded, _ = self.encoder(gated)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        output = self.output_grn(attended[:, -1] + encoded[:, -1])
        return cast(torch.Tensor, self.head(output).view(x.shape[0], self.horizons, self.quantiles))
