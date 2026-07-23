from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def purged_split(
    train_start: date,
    train_end: date,
    validation_start: date,
    validation_end: date,
    test_start: date,
    test_end: date,
    embargo_days: int,
) -> TimeSplit:
    purged_train_end = min(train_end, validation_start - timedelta(days=embargo_days + 1))
    purged_validation_end = min(validation_end, test_start - timedelta(days=embargo_days + 1))
    if purged_train_end < train_start or purged_validation_end < validation_start:
        raise ValueError("Insufficient data after embargo")
    return TimeSplit(train_start, purged_train_end, validation_start, purged_validation_end, test_start, test_end)
