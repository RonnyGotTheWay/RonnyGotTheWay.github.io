from __future__ import annotations

import numpy as np

from art_rank_quant.models.hmm.model import RegimeHMM


def infer_regime(model: RegimeHMM, features: np.ndarray) -> dict[str, object]:
    probabilities = model.predict(features)[-1]
    return {"state_id": int(np.argmax(probabilities)), "state_probabilities": probabilities.tolist()}
