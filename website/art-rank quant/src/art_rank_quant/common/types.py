from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import NewType

Symbol = NewType("Symbol", str)
DataVersion = NewType("DataVersion", str)
FeatureVersion = NewType("FeatureVersion", str)
ModelVersion = NewType("ModelVersion", str)
RunId = NewType("RunId", str)


@dataclass(frozen=True, slots=True)
class VersionContext:
    as_of_date: date
    generated_at: datetime
    data_version: DataVersion
    feature_version: FeatureVersion
    model_version: ModelVersion
    run_id: RunId
    is_stale: bool = False
