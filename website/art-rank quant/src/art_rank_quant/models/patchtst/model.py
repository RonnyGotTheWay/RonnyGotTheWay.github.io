from __future__ import annotations

from typing import cast

import torch
from torch import nn

from art_rank_quant.models.patchtst.dataset import patch_sequence


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int = 10,
        stride: int = 5,
        embedding_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size, self.stride = patch_size, stride
        self.projection = nn.Linear(channels * patch_size, embedding_dim)
        block = nn.TransformerEncoderLayer(embedding_dim, heads, embedding_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = patch_sequence(values, self.patch_size, self.stride).flatten(2)
        encoded = self.encoder(self.projection(patches))
        return cast(torch.Tensor, encoded.mean(dim=1))
