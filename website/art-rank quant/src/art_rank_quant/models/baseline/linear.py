from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.linear_model import Ridge

from art_rank_quant.models.base import Model


class LinearBaseline(Model):
    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, features: Any, target: Any, **kwargs: Any) -> LinearBaseline:
        self.model.fit(features, target)
        return self

    def predict(self, features: Any) -> Any:
        return self.model.predict(features)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self.model))

    @classmethod
    def load(cls, path: Path) -> LinearBaseline:
        instance = cls()
        instance.model = pickle.loads(path.read_bytes())
        return instance

    def metadata(self) -> dict[str, Any]:
        return {"name": "ridge", "alpha": self.model.alpha}
