from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from art_rank_quant.models.base import Model


class RegimeHMM(Model):
    def __init__(self, n_components: int = 4, random_state: int = 474) -> None:
        self.model = GaussianHMM(
            n_components=n_components, covariance_type="full", n_iter=200, random_state=random_state
        )
        self._state_order: np.ndarray | None = None

    def fit(self, features: Any, target: Any = None, **kwargs: Any) -> RegimeHMM:
        values = np.asarray(features, dtype=np.float64)
        self.model.fit(values)
        self._state_order = np.argsort(self.model.means_[:, 0])
        return self

    def predict(self, features: Any) -> np.ndarray:
        probabilities = self.model.predict_proba(np.asarray(features, dtype=np.float64))
        ordered = probabilities[:, self._state_order] if self._state_order is not None else probabilities
        return np.asarray(ordered, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps((self.model, self._state_order)))

    @classmethod
    def load(cls, path: Path) -> RegimeHMM:
        model, state_order = pickle.loads(path.read_bytes())
        instance = cls(model.n_components)
        instance.model, instance._state_order = model, state_order
        return instance

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "gaussian_hmm",
            "states": self.model.n_components,
            "state_order": self._state_order.tolist() if self._state_order is not None else None,
        }
