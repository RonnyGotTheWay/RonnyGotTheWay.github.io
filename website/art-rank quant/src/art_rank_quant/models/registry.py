from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ModelRecord:
    name: str
    version: str
    stage: str
    trained_from: date
    trained_to: date
    data_version: str
    feature_version: str
    artifact_sha256: str
    git_commit: str
