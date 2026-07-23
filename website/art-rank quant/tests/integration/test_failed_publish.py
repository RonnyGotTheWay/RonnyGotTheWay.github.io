import pytest

from art_rank_quant.pipelines.daily_publish import publish_snapshot


def test_immutable_snapshot_cannot_be_overwritten(tmp_path) -> None:
    publish_snapshot({"ok": True}, tmp_path, "v1")
    with pytest.raises(FileExistsError):
        publish_snapshot({"ok": False}, tmp_path, "v1")
