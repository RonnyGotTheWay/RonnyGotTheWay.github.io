from datetime import date

from art_rank_quant.pipelines.daily import run_daily


def test_daily_pipeline_is_resumable_and_idempotent(tmp_path) -> None:
    calls = []
    result = run_daily(date(2024, 1, 2), tmp_path, {"a": lambda: calls.append("a"), "b": lambda: calls.append("b")})
    repeated = run_daily(date(2024, 1, 2), tmp_path, {"a": lambda: calls.append("again")})
    assert result.status == repeated.status == "success"
    assert calls == ["a", "b"]
