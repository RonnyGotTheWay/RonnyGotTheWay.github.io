from __future__ import annotations

import torch


def patch_sequence(values: torch.Tensor, patch_size: int = 10, stride: int = 5) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("Expected [batch, time, channels]")
    if values.shape[1] < patch_size:
        raise ValueError("Sequence shorter than patch size")
    return values.unfold(1, patch_size, stride).permute(0, 1, 3, 2)
