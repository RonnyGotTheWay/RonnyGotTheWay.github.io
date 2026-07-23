from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class VersionMetadata(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data_version: str = "synthetic-v1"
    feature_version: str = "features-v1"
    model_version: str = "baseline-v1"
    run_id: str = "demo-run"
    is_stale: bool = False


class Envelope[T](BaseModel):
    meta: VersionMetadata = Field(default_factory=VersionMetadata)
    data: T
