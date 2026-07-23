from datetime import date

from art_rank_quant.validation.purged_split import purged_split


def test_embargo_separates_train_and_validation() -> None:
    split = purged_split(
        date(2020, 1, 1),
        date(2022, 1, 31),
        date(2022, 2, 1),
        date(2022, 5, 31),
        date(2022, 6, 15),
        date(2022, 7, 15),
        5,
    )
    assert (split.validation_start - split.train_end).days >= 6
