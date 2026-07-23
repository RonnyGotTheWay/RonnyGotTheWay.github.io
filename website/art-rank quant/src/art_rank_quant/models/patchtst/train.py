from __future__ import annotations

import torch
from torch import nn


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (prediction - target).pow(2)
    return loss.masked_select(mask).mean() if mask.any() else nn.functional.mse_loss(prediction, target)
