from __future__ import annotations

import torch


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    return torch.maximum((quantiles - 1) * error, quantiles * error).mean()
