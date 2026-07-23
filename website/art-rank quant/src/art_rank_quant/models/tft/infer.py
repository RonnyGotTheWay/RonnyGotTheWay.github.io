from __future__ import annotations

from typing import cast

import torch

from art_rank_quant.models.tft.model import TftStyleForecaster


@torch.inference_mode()
def forecast(model: TftStyleForecaster, values: torch.Tensor) -> torch.Tensor:
    model.eval()
    return cast(torch.Tensor, model(values))
