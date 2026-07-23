from __future__ import annotations

from datetime import date, timedelta

from art_rank_quant.validation.purged_split import TimeSplit, purged_split


def walk_forward_splits(
    start: date, end: date, train_days: int = 730, validation_days: int = 90, test_days: int = 30, embargo_days: int = 5
) -> list[TimeSplit]:
    splits: list[TimeSplit] = []
    cursor = start + timedelta(days=train_days)
    while cursor + timedelta(days=validation_days + test_days) <= end:
        splits.append(
            purged_split(
                start,
                cursor,
                cursor + timedelta(days=embargo_days + 1),
                cursor + timedelta(days=validation_days),
                cursor + timedelta(days=validation_days + embargo_days + 1),
                cursor + timedelta(days=validation_days + test_days),
                embargo_days,
            )
        )
        cursor += timedelta(days=test_days)
    return splits
