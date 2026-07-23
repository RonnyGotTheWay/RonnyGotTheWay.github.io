from art_rank_quant.models.baseline.lightgbm_ranker import LambdaRankBaseline


class ArtRanker(LambdaRankBaseline):
    """LambdaRank model receiving PIT, temporal and lagged market-context features."""
