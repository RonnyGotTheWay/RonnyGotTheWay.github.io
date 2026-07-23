from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from art_rank_quant.data.clients.synthetic import SyntheticAshareClient
from art_rank_quant.data.processing.adjust_prices import add_adjusted_prices
from art_rank_quant.data.repositories.manifest import DatasetManifest
from art_rank_quant.data.storage.parquet import file_sha256, write_snapshot
from art_rank_quant.data.validation.quality_checks import validate_stock_daily


def _date_range(frame: pl.DataFrame) -> tuple[str, str]:
    column = "trade_date" if "trade_date" in frame.columns else "event_time"
    return str(frame[column].min()), str(frame[column].max())


def _write_with_manifest(
    frame: pl.DataFrame,
    root: Path,
    layer: str,
    dataset: str,
    version: str,
    schema_version: str,
) -> None:
    versioned = frame.with_columns(pl.lit(version).alias("data_version"))
    path = write_snapshot(versioned, root, layer, dataset, version)
    minimum, maximum = _date_range(versioned)
    manifest = DatasetManifest.create(
        dataset,
        version,
        schema_version,
        versioned.height,
        minimum,
        maximum,
        file_sha256(path),
    )
    manifest.write(path.with_name("manifest.json"))


def backfill_synthetic(
    start: date,
    end: date,
    root: Path,
    seed: int = 474,
    symbols: int = 300,
) -> str:
    client = SyntheticAshareClient(seed=seed, symbol_count=symbols)
    version = f"synthetic-v1-{client.fingerprint(start, end)}"
    batches = client.fetch(start, end)
    for name, batch in batches.items():
        _write_with_manifest(batch.frame, root, "raw", name, version, batch.schema_version)

    market = batches["stock_daily"]
    report = validate_stock_daily(market.frame)
    if not report.passed:
        raise ValueError("; ".join(report.errors))
    processed = {
        name: add_adjusted_prices(batch.frame) if name == "stock_daily" else batch.frame
        for name, batch in batches.items()
    }
    for name, frame in processed.items():
        _write_with_manifest(frame, root, "processed", name, version, batches[name].schema_version)
    return version
