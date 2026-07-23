from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb

from art_rank_quant.models.base import Model


class LambdaRankBaseline(Model):
    def __init__(self, **params: Any) -> None:
        defaults = {"objective": "lambdarank", "metric": "ndcg", "n_estimators": 300, "learning_rate": 0.03}
        self.params = defaults | params
        self.model = lgb.LGBMRanker(**self.params)  # type: ignore[arg-type]

    def fit(self, features: Any, target: Any, **kwargs: Any) -> LambdaRankBaseline:
        group = kwargs.get("group")
        if group is None:
            raise ValueError("LambdaRank requires per-date group sizes")
        self.model.fit(features, target, group=group)
        return self

    def predict(self, features: Any) -> Any:
        return self.model.predict(features)

    def save(self, path: Path) -> None:
        self.model.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> LambdaRankBaseline:
        instance = cls()
        instance.model = lgb.LGBMRanker()
        instance.model._Booster = lgb.Booster(model_file=str(path))
        return instance

    def metadata(self) -> dict[str, Any]:
        return {"name": "lightgbm_lambdarank", "params": self.params}
