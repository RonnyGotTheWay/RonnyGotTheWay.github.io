from __future__ import annotations

import torch


def sliding_sequences(values: torch.Tensor, encoder_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[0] <= encoder_length:
        raise ValueError("Expected [time, features] with enough history")
    sequences = values.unfold(0, encoder_length, 1).permute(0, 2, 1)
    mask = torch.isfinite(sequences)
    return torch.nan_to_num(sequences), mask
