from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

from art_rank_quant.api.schemas.models import BootstrapAccepted, TrainingRunSummary
from art_rank_quant.common.config import Settings, get_settings

TrainingMode = Literal["long_term", "short_term"]
ACTIVE_STATUSES = {"queued", "running"}


class TrainingConflictError(RuntimeError):
    def __init__(self, holder: TrainingRunSummary) -> None:
        self.holder = holder
        code = "long_term_training_active" if holder.mode == "long_term" else "short_term_training_active"
        label = "长期拟合" if holder.mode == "long_term" else "短期预测"
        self.detail = {
            "code": code,
            "message": f"{label}正在运行，本次任务未启动。",
            "blocking_run_id": holder.run_id,
        }
        super().__init__(self.detail["message"])


class TrainingCancelled(RuntimeError):
    pass


class TrainingMemoryLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LockHolder:
    run_id: str
    mode: TrainingMode


def training_run_root(settings: Settings, mode: TrainingMode) -> Path:
    return settings.artifact_root / "training_runs" / mode


def legacy_training_root(settings: Settings) -> Path:
    return settings.artifact_root / "training_runs"


def resolve_training_run_path(settings: Settings, run_id: str, mode: TrainingMode) -> Path:
    modern = training_run_root(settings, mode) / f"{run_id}.json"
    if modern.exists():
        return modern
    return legacy_training_root(settings) / f"{run_id}.json"


class _BootstrapService:
    mode: TrainingMode
    prefix: str
    module: str
    accepted_message: str
    queued_message: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def root(self) -> Path:
        return training_run_root(self.settings, self.mode)

    def start(self) -> BootstrapAccepted:
        self.root.mkdir(parents=True, exist_ok=True)
        same_mode = active_training_run(self.settings, self.mode)
        if same_mode is not None:
            return BootstrapAccepted(run_id=same_mode.run_id, status="running", message="同类训练任务正在运行。")
        other_mode: TrainingMode = "short_term" if self.mode == "long_term" else "long_term"
        other = active_training_run(self.settings, other_mode)
        if other is not None:
            raise TrainingConflictError(other)

        run_id = f"{self.prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
        holder = acquire_training_lock(self.settings, run_id, self.mode)
        if holder.run_id != run_id:
            existing = _load_holder_run(self.settings, holder)
            if existing.mode != self.mode:
                raise TrainingConflictError(existing)
            return BootstrapAccepted(run_id=existing.run_id, status="running", message="同类训练任务正在运行。")

        now = datetime.now(UTC)
        run = TrainingRunSummary(
            run_id=run_id,
            status="queued",
            stage="queued",
            message=self.queued_message,
            mode=self.mode,
            heartbeat_at=now,
        )
        _write_run(self.root, run)
        log_path = self.root / f"{run_id}.log"
        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(  # noqa: S603
                    [sys.executable, "-m", self.module, "--run-id", run_id],
                    cwd=Path.cwd(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=os.environ.copy(),
                )
        except Exception:
            release_training_lock(self.settings, run_id)
            raise
        run.pid = process.pid
        run.message = f"独立任务已启动，pid={process.pid}。"
        _write_run(self.root, run)
        _update_lock(self.settings, run)
        return BootstrapAccepted(run_id=run_id, status="queued", message=self.accepted_message)

    def get(self, run_id: str) -> TrainingRunSummary:
        self._validate_run_id(run_id)
        path = resolve_training_run_path(self.settings, run_id, self.mode)
        if not path.exists():
            raise FileNotFoundError(run_id)
        run = load_training_run(
            path,
            self.settings.model_run_stale_minutes,
            self.settings.training_unresponsive_seconds,
        )
        if run.mode != self.mode:
            raise ValueError("Training run mode mismatch")
        return run

    def cancel(self, run_id: str) -> TrainingRunSummary:
        run = self.get(run_id)
        if run.status not in ACTIVE_STATUSES:
            return run
        marker = _cancel_path(resolve_training_run_path(self.settings, run_id, self.mode))
        marker.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        run.cancel_requested = True
        run.message = "已请求取消，任务将在下一个安全检查点停止。"
        _write_run(marker.parent, run)
        return run

    def _validate_run_id(self, run_id: str) -> None:
        if not run_id.startswith(f"{self.prefix}-") or "/" in run_id or ".." in run_id:
            raise ValueError(f"Invalid {self.mode} run id")


class ModelBootstrapService(_BootstrapService):
    mode: TrainingMode = "long_term"
    prefix = "bootstrap"
    module = "art_rank_quant.pipelines.model_bootstrap"
    accepted_message = "两年回填与完整训练任务已启动。"
    queued_message = "长期任务已提交。"


class ShortTermBootstrapService(_BootstrapService):
    mode: TrainingMode = "short_term"
    prefix = "short-term"
    module = "art_rank_quant.pipelines.short_term"
    accepted_message = "15日实验训练任务已启动。"
    queued_message = "15日实验训练任务已提交。"


@lru_cache
def get_model_bootstrap_service() -> ModelBootstrapService:
    return ModelBootstrapService(get_settings())


@lru_cache
def get_short_term_bootstrap_service() -> ShortTermBootstrapService:
    return ShortTermBootstrapService(get_settings())


def acquire_training_lock(settings: Settings, run_id: str, mode: TrainingMode = "long_term") -> LockHolder:
    root = legacy_training_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "active-model-training.json"
    for _ in range(2):
        try:
            now = datetime.now(UTC).isoformat()
            with lock.open("x", encoding="utf-8") as handle:
                json.dump({"run_id": run_id, "mode": mode, "created_at": now, "heartbeat_at": now}, handle)
            return LockHolder(run_id, mode)
        except FileExistsError:
            try:
                body = json.loads(lock.read_text(encoding="utf-8"))
                holder = LockHolder(str(body["run_id"]), _infer_mode(str(body["run_id"]), body.get("mode")))
                run = _load_holder_run(settings, holder)
                if run.status in ACTIVE_STATUSES:
                    return holder
            except (KeyError, ValueError, OSError, FileNotFoundError):
                pass
            lock.unlink(missing_ok=True)
    raise RuntimeError("Unable to acquire model training lock")


def release_training_lock(settings: Settings, run_id: str) -> None:
    lock = legacy_training_root(settings) / "active-model-training.json"
    if not lock.exists():
        return
    try:
        holder = str(json.loads(lock.read_text(encoding="utf-8"))["run_id"])
    except (KeyError, ValueError, OSError):
        holder = run_id
    if holder == run_id:
        lock.unlink(missing_ok=True)


def active_training_run(settings: Settings, mode: TrainingMode) -> TrainingRunSummary | None:
    paths = list(training_run_root(settings, mode).glob("*.json"))
    paths.extend(legacy_training_root(settings).glob("*.json"))
    for path in sorted(paths, reverse=True):
        if path.name == "active-model-training.json" or path.name.endswith(".heartbeat.json"):
            continue
        try:
            run = load_training_run(
                path,
                settings.model_run_stale_minutes,
                settings.training_unresponsive_seconds,
            )
        except (ValueError, OSError):
            continue
        if run.mode == mode and run.status in ACTIVE_STATUSES:
            return run
    return None


def load_training_run(
    path: Path,
    stale_minutes: int,
    unresponsive_seconds: int | None = None,
) -> TrainingRunSummary:
    """Read a run and recover state left by a dead or unresponsive worker."""

    run = TrainingRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
    heartbeat_path = _heartbeat_path(path)
    if heartbeat_path.exists():
        try:
            heartbeat = datetime.fromisoformat(json.loads(heartbeat_path.read_text(encoding="utf-8"))["heartbeat_at"])
            run.heartbeat_at = heartbeat
        except (KeyError, ValueError, OSError):
            pass
    if run.status not in ACTIVE_STATUSES:
        return run

    now = datetime.now(UTC)
    updated_at = run.heartbeat_at or datetime.fromtimestamp(path.stat().st_mtime, UTC)
    threshold = timedelta(seconds=unresponsive_seconds) if unresponsive_seconds else timedelta(minutes=stale_minutes)
    age = now - updated_at
    alive = run.pid is not None and _process_is_alive(run.pid)
    if age <= timedelta(seconds=10):
        return run
    if age <= threshold and (run.pid is None or alive):
        return run

    if alive and run.pid is not None and _is_expected_worker(run.pid, run.run_id):
        os.kill(run.pid, signal.SIGTERM)
    run.status = "failed"
    run.stage = "failed"
    run.error = "unresponsive_training_run" if alive else "stale_training_run"
    run.message = "训练心跳超时，已停止任务并释放资源；可从已校验缓存继续。"
    run.finished_at = now
    run.elapsed_seconds = _elapsed(run, now)
    run.duration_seconds = run.elapsed_seconds
    _write_run(path.parent, run)
    lock_root = path.parent.parent if path.parent.name in {"long_term", "short_term"} else path.parent
    lock = lock_root / "active-model-training.json"
    try:
        if json.loads(lock.read_text(encoding="utf-8")).get("run_id") == run.run_id:
            lock.unlink(missing_ok=True)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass
    return run


class TrainingHeartbeat:
    """Sidecar heartbeat that never races with the main run JSON writer."""

    def __init__(self, settings: Settings, run_path: Path, run_id: str, mode: TrainingMode) -> None:
        self.settings = settings
        self.run_path = run_path
        self.run_id = run_id
        self.mode = mode
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"heartbeat-{run_id}", daemon=True)

    def __enter__(self) -> TrainingHeartbeat:
        self.beat()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def beat(self) -> None:
        now = datetime.now(UTC).isoformat()
        body = {"run_id": self.run_id, "mode": self.mode, "heartbeat_at": now, "pid": os.getpid()}
        _atomic_json(_heartbeat_path(self.run_path), body)
        lock = legacy_training_root(self.settings) / "active-model-training.json"
        if lock.exists():
            try:
                current = json.loads(lock.read_text(encoding="utf-8"))
                if current.get("run_id") == self.run_id:
                    current.update(body)
                    _atomic_json(lock, current)
            except (ValueError, OSError):
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self.settings.training_heartbeat_seconds):
            self.beat()


def check_run_control(settings: Settings, run_path: Path, run: TrainingRunSummary) -> None:
    now = datetime.now(UTC)
    run.heartbeat_at = now
    run.elapsed_seconds = _elapsed(run, now)
    run.memory_rss_mb = _peak_memory_mb()
    if _cancel_path(run_path).exists() or run.cancel_requested:
        raise TrainingCancelled("任务已由用户取消")
    if run.memory_rss_mb > settings.model_max_memory_gb * 1024:
        raise TrainingMemoryLimitExceeded(
            f"训练内存 {run.memory_rss_mb:.0f} MB 超过限制 {settings.model_max_memory_gb:.1f} GB"
        )
    soft_limit = settings.model_max_memory_gb * 1024 * 0.8
    if run.memory_rss_mb > soft_limit and run.message and "内存预警" not in run.message:
        run.message = f"{run.message} 内存预警：已使用 {run.memory_rss_mb:.0f} MB，已接近任务上限。"


def _load_holder_run(settings: Settings, holder: LockHolder) -> TrainingRunSummary:
    path = resolve_training_run_path(settings, holder.run_id, holder.mode)
    if not path.exists():
        raise FileNotFoundError(holder.run_id)
    return load_training_run(path, settings.model_run_stale_minutes, settings.training_unresponsive_seconds)


def _update_lock(settings: Settings, run: TrainingRunSummary) -> None:
    lock = legacy_training_root(settings) / "active-model-training.json"
    if not lock.exists():
        return
    try:
        body = json.loads(lock.read_text(encoding="utf-8"))
        if body.get("run_id") != run.run_id:
            return
        body.update({"mode": run.mode, "pid": run.pid, "heartbeat_at": datetime.now(UTC).isoformat()})
        _atomic_json(lock, body)
    except (ValueError, OSError):
        return


def _infer_mode(run_id: str, value: object = None) -> TrainingMode:
    if value in {"long_term", "short_term"}:
        return value  # type: ignore[return-value]
    return "short_term" if run_id.startswith("short-term-") else "long_term"


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_expected_worker(pid: int, run_id: str) -> bool:
    try:
        result = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return run_id in result.stdout and "art_rank_quant.pipelines" in result.stdout


def _heartbeat_path(path: Path) -> Path:
    return path.with_suffix(".heartbeat.json")


def _cancel_path(path: Path) -> Path:
    return path.with_suffix(".cancel")


def _elapsed(run: TrainingRunSummary, now: datetime) -> float | None:
    return max(0.0, (now - run.started_at).total_seconds()) if run.started_at else None


def _peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _write_run(root: Path, run: TrainingRunSummary) -> None:
    _atomic_json(root / f"{run.run_id}.json", run.model_dump(mode="json"))


def _atomic_json(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
