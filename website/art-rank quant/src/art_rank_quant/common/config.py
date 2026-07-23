from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    database_url: str = "sqlite:///art_rank.sqlite3"
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    model_stage: str = "champion"
    log_level: str = "INFO"
    random_seed: int = Field(default=474, ge=0)
    market_data_provider: str = "eastmoney"
    tushare_token: SecretStr | None = None
    tushare_api_url: str = "https://api.tushare.pro"
    realtime_symbols: str = "3*.SZ,6*.SH,0*.SZ,9*.BJ"
    realtime_cache_seconds: int = Field(default=600, ge=5, le=3600)
    realtime_request_timeout_seconds: float = Field(default=8.0, gt=0, le=30)
    candidate_min_amount_cny: float = Field(default=50_000_000, ge=0)
    eastmoney_permission_confirmed: bool = False
    eastmoney_endpoint: str = "https://push2.eastmoney.com/webguest/api/qt/clist/get"
    eastmoney_page_size: int = Field(default=100, ge=1, le=100)
    eastmoney_request_delay_seconds: float = Field(default=0.25, ge=0, le=10)
    eastmoney_request_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    baostock_enabled: bool = False
    model_history_years: int = Field(default=2, ge=2, le=10)
    model_min_daily_days: int = Field(default=480, ge=252)
    model_min_intraday_coverage: float = Field(default=0.95, ge=0.5, le=1.0)
    model_max_memory_gb: float = Field(default=12.0, ge=2, le=64)
    model_run_stale_minutes: int = Field(default=30, ge=5, le=1440)
    training_heartbeat_seconds: int = Field(default=15, ge=5, le=60)
    training_unresponsive_seconds: int = Field(default=90, ge=30, le=600)
    frontend_api_timeout_ms: int = Field(default=10_000, ge=1_000, le=120_000)
    long_term_intraday_provider: Literal["eastmoney", "akshare_sina", "baostock"] = "eastmoney"
    short_term_history_provider: Literal["eastmoney", "akshare_sina", "baostock"] = "eastmoney"
    history_provider_fallbacks: str = "akshare_sina"
    short_term_history_workers: int = Field(default=4, ge=1, le=8)
    short_term_secondary_workers: int = Field(default=2, ge=1, le=4)
    short_term_secondary_delay_seconds: float = Field(default=0.5, ge=0.1, le=10)
    short_term_failure_circuit: int = Field(default=20, ge=3, le=100)
    short_term_daily_coverage: float = Field(default=0.95, ge=0.5, le=1.0)
    short_term_intraday_coverage: float = Field(default=0.85, ge=0.5, le=1.0)

    @property
    def history_provider_fallback_list(self) -> list[str]:
        return [value.strip() for value in self.history_provider_fallbacks.split(",") if value.strip()]

    @property
    def realtime_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.realtime_symbols.split(",") if symbol.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded
