from __future__ import annotations

import numpy as np


def percentile_calibrate(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order / max(1, len(values) - 1)
