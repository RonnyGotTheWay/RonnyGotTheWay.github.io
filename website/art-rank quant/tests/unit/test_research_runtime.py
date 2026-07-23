import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from art_rank_quant.api.schemas.models import TrainingRunSummary
from art_rank_quant.api.services.research_runtime import ResearchRuntimeService
from art_rank_quant.common.config import Settings


def runtime(tmp_path: Path) -> ResearchRuntimeService:
    return ResearchRuntimeService(
        Settings(
            data_root=tmp_path / "data",
            artifact_root=tmp_path / "artifacts",
            eastmoney_permission_confirmed=True,
        )
    )


def test_refresh_submission_is_deduplicated(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    first = service.start_refresh()
    second = service.start_refresh()

    assert first.status == "queued"
    assert second.status == "running"
    assert second.run_id == first.run_id


def test_unvalidated_models_block_prediction_publication(tmp_path: Path) -> None:
    service = runtime(tmp_path)

    prediction, metadata = service.latest_prediction()
    monitoring = service.monitoring()

    assert prediction.status == "blocked"
    assert prediction.candidates == []
    assert metadata.is_stale is True
    assert {component.component for component in monitoring.components} == {"hmm", "tft", "patchtst", "ranker"}
    assert all(component.status == "not_started" for component in monitoring.components)
    assert monitoring.ensemble.status == "not_started"
    assert monitoring.ensemble.publish_gate_passed is False
    assert "门禁尚未执行" in monitoring.message


def test_stale_run_does_not_leave_monitoring_in_training_state(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    run_root = tmp_path / "artifacts" / "training_runs"
    run_root.mkdir(parents=True)
    run = TrainingRunSummary(run_id="bootstrap-stale", status="running", stage="daily")
    path = run_root / f"{run.run_id}.json"
    path.write_text(json.dumps(run.model_dump(mode="json")), encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))

    monitoring = service.monitoring()

    assert monitoring.status != "training"
    assert monitoring.training_runs[0].status == "failed"
    assert monitoring.training_runs[0].error == "stale_training_run"
    assert monitoring.ensemble.status == "not_started"


def test_missing_market_snapshot_is_explicit(tmp_path: Path) -> None:
    snapshot, metadata = runtime(tmp_path).latest_market()

    assert snapshot.status == "not_available"
    assert snapshot.quotes == []
    assert metadata.is_stale is True


def test_previous_day_market_snapshot_is_reclassified_as_stale(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    previous = now - timedelta(days=1)
    target = tmp_path / "artifacts" / "published" / "market" / "stable.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "meta": {
                    "as_of_date": previous.date().isoformat(),
                    "generated_at": previous.isoformat(),
                    "data_version": "previous-day",
                    "feature_version": "market-snapshot-v2",
                    "model_version": "observation-only",
                    "run_id": "refresh-old",
                    "is_stale": False,
                },
                "data": {
                    "status": "live",
                    "market_session": "trading",
                    "message": "old live state",
                    "trade_date": previous.date().isoformat(),
                    "fetched_at": previous.isoformat(),
                    "quotes": [],
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot, metadata = service.latest_market()

    assert snapshot.status == "stale"
    assert metadata.is_stale is True
    assert snapshot.message == "当前显示最近一次已校验快照，数据已过实时有效期。"
