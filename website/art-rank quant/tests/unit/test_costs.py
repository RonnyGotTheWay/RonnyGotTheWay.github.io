from art_rank_quant.backtest.costs import CostModel


def test_sell_cost_includes_stamp_duty() -> None:
    model = CostModel()
    assert model.estimate(10000, "sell") > model.estimate(10000, "buy")
