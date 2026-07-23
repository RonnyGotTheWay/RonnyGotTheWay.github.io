from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelHealth(BaseModel):
    name: str
    version: str
    stage: str
    rank_ic_60d: float
    drift_psi: float
    status: str


class ComponentHealth(BaseModel):
    component: Literal["hmm", "tft", "patchtst", "ranker"]
    display_name: str
    status: Literal["not_started", "running", "healthy", "warning", "failed", "blocked"]
    version: str | None = None
    progress: int = 0
    last_trained_at: datetime | None = None
    metrics: dict[str, float | int | str | bool | list[int]] = Field(default_factory=dict)
    checkpoint: str | None = None
    artifact_sha256: str | None = None
    error: str | None = None


class EnsembleHealth(BaseModel):
    status: Literal["not_started", "running", "healthy", "warning", "failed", "blocked"]
    champion_version: str | None = None
    weight_sum: float | None = None
    lagged_rank_ic: float | None = None
    horizon_consistent: bool = False
    publish_gate_passed: bool = False
    blocked_reasons: list[str] = Field(default_factory=list)


class SourceCheck(BaseModel):
    provider: str
    datasets: list[str] = Field(default_factory=list)
    status: Literal["ready", "partial", "unavailable"]
    message: str


class TrainingRunSummary(BaseModel):
    run_id: str
    status: str
    pid: int | None = None
    stage: str | None = None
    stage_label: str | None = None
    stage_completed_items: int = 0
    stage_total_items: int = 0
    progress: int = 0
    current_symbol: str | None = None
    completed_symbols: int = 0
    total_symbols: int = 0
    daily_rows: int = 0
    intraday_rows: int = 0
    retries: int = 0
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None
    mode: Literal["long_term", "short_term"] = "long_term"
    heartbeat_at: datetime | None = None
    last_success_at: datetime | None = None
    provider: str | None = None
    failed_items: int = 0
    memory_rss_mb: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    cancel_requested: bool = False


class ModelMonitoring(BaseModel):
    status: Literal["idle", "training", "warning", "failed", "blocked"]
    message: str
    data_coverage_pct: float
    drift_psi: float | None = None
    latest_prediction_status: str
    daily_history_days: int = 0
    intraday_history_days: int = 0
    intraday_coverage_pct: float = 0.0
    components: list[ComponentHealth]
    ensemble: EnsembleHealth
    source_checks: list[SourceCheck] = Field(default_factory=list)
    training_runs: list[TrainingRunSummary] = Field(default_factory=list)
    mode: Literal["long_term", "short_term"] = "long_term"
    experimental: bool = False


class BootstrapAccepted(BaseModel):
    run_id: str
    status: Literal["queued", "running"]
    message: str
