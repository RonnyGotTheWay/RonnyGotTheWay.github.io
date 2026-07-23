import torch

from art_rank_quant.models.patchtst.model import PatchTSTEncoder
from art_rank_quant.models.tft.model import TftStyleForecaster


def test_deep_model_output_shapes() -> None:
    assert TftStyleForecaster(6)(torch.randn(4, 60, 6)).shape == (4, 2, 3)
    assert PatchTSTEncoder(6)(torch.randn(4, 120, 6)).shape == (4, 64)
