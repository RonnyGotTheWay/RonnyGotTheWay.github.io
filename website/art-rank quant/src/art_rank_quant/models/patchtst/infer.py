from __future__ import annotations

from typing import cast

import torch

from art_rank_quant.models.patchtst.model import PatchTSTEncoder


@torch.inference_mode()
def temporal_embedding(model: PatchTSTEncoder, values: torch.Tensor) -> torch.Tensor:
    model.eval()
    return cast(torch.Tensor, model(values))
