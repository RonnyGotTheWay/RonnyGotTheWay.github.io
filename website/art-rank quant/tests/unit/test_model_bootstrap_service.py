import json
import os
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from art_rank_quant.api.schemas.models import TrainingRunSummary
from art_rank_quant.api.services.model_bootstrap import (
    ModelBootstrapService,
    ShortTermBootstrapService,
    TrainingConflictError,
    load_training_run,
)
from art_rank_quant.common.config import Settings


class FakeProcess:
    pid = 474


def test_bootstrap_submission_is_detached_and_deduplicated(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap.subprocess.Popen", fake_popen)
    service = ModelBootstrapService(Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data"))

    first = service.start()
    second = service.start()

    assert first.status == "queued"
    assert second.status == "running"
    assert second.run_id == first.run_id
    assert len(calls) == 1
    assert calls[0][-2:] == ["--run-id", first.run_id]


def test_long_and_short_training_are_mutually_exclusive(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap.subprocess.Popen", fake_popen)
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    long_service = ModelBootstrapService(settings)
    short_service = ShortTermBootstrapService(settings)

    long_run = long_service.start()
    with pytest.raises(TrainingConflictError) as captured:
        short_service.start()

    assert captured.value.detail["code"] == "long_term_training_active"
    assert captured.value.detail["blocking_run_id"] == long_run.run_id
    assert len(calls) == 1


def test_short_submission_uses_isolated_pipeline(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap.subprocess.Popen", fake_popen)
    service = ShortTermBootstrapService(Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data"))

    accepted = service.start()
    stored = service.get(accepted.run_id)

    assert accepted.run_id.startswith("short-term-")
    assert stored.mode == "short_term"
    assert calls[0][2:4] == ["art_rank_quant.pipelines.short_term", "--run-id"]
    assert service.root == tmp_path / "artifacts" / "training_runs" / "short_term"
    assert not (tmp_path / "artifacts" / "training_runs" / "long_term").exists()


def test_cancel_request_is_mode_scoped_and_persistent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "art_rank_quant.api.services.model_bootstrap.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    service = ShortTermBootstrapService(
        Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    )
    accepted = service.start()

    cancelled = service.cancel(accepted.run_id)

    assert cancelled.mode == "short_term"
    assert cancelled.cancel_requested is True
    assert (service.root / f"{accepted.run_id}.cancel").exists()


def test_stale_training_run_is_recovered_before_new_submission(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap.subprocess.Popen", fake_popen)
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        model_run_stale_minutes=5,
    )
    service = ModelBootstrapService(settings)
    service.root.mkdir(parents=True)
    stale = TrainingRunSummary(run_id="bootstrap-stale", status="running", stage="daily")
    stale_path = service.root / f"{stale.run_id}.json"
    stale_path.write_text(json.dumps(stale.model_dump(mode="json")), encoding="utf-8")
    old = time.time() - 600
    os.utime(stale_path, (old, old))

    accepted = service.start()
    recovered = TrainingRunSummary.model_validate_json(stale_path.read_text(encoding="utf-8"))

    assert accepted.run_id != stale.run_id
    assert recovered.status == "failed"
    assert recovered.error == "stale_training_run"
    assert len(calls) == 1


def test_unresponsive_worker_is_terminated_and_releases_lock(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "artifacts" / "training_runs" / "long_term"
    root.mkdir(parents=True)
    run = TrainingRunSummary(run_id="bootstrap-hung", status="running", mode="long_term", pid=474)
    path = root / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(), encoding="utf-8")
    heartbeat = datetime.now(UTC) - timedelta(seconds=120)
    path.with_suffix(".heartbeat.json").write_text(
        json.dumps({"heartbeat_at": heartbeat.isoformat()}), encoding="utf-8"
    )
    lock = root.parent / "active-model-training.json"
    lock.write_text(json.dumps({"run_id": run.run_id}), encoding="utf-8")
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap._process_is_alive", lambda _pid: True)
    monkeypatch.setattr("art_rank_quant.api.services.model_bootstrap._is_expected_worker", lambda *_args: True)
    monkeypatch.setattr(
        "art_rank_quant.api.services.model_bootstrap.os.kill",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    recovered = load_training_run(path, stale_minutes=30, unresponsive_seconds=90)

    assert recovered.status == "failed"
    assert recovered.error == "unresponsive_training_run"
    assert signals == [(474, signal.SIGTERM)]
    assert not lock.exists()
