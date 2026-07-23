from __future__ import annotations

from datetime import date
from typing import Annotated

import typer

from art_rank_quant.api.services.weekly_report import WeeklyReportService
from art_rank_quant.common.config import get_settings
from art_rank_quant.data.clients.eastmoney_daily import EastmoneyDailyCrawler
from art_rank_quant.pipelines.backfill import backfill_synthetic
from art_rank_quant.pipelines.daily import run_daily
from art_rank_quant.pipelines.daily_ingestion import daily_ingestion

app = typer.Typer(help="ART-Rank Quant research CLI")


@app.command()
def synthetic(
    start: Annotated[str, typer.Option()],
    end: Annotated[str, typer.Option()],
    symbols: int = 300,
) -> None:
    """Generate immutable deterministic synthetic snapshots."""
    settings = get_settings()
    version = backfill_synthetic(
        date.fromisoformat(start),
        date.fromisoformat(end),
        settings.data_root,
        settings.random_seed,
        symbols,
    )
    typer.echo(f"created data_version={version}")


@app.command()
def daily(as_of: Annotated[str | None, typer.Option()] = None) -> None:
    """Run the resumable local daily workflow."""
    settings = get_settings()
    run_date = date.fromisoformat(as_of) if as_of else date.today()

    def ingestion_step() -> None:
        daily_ingestion(run_date, settings.data_root)

    steps = {
        "ingestion": ingestion_step,
        "quality": lambda: None,
        "features": lambda: None,
        "inference": lambda: None,
        "portfolio": lambda: None,
        "publish": lambda: None,
    }
    result = run_daily(run_date, settings.artifact_root, steps)
    typer.echo(f"{result.run_id}: {result.status}")


@app.command("crawl-eastmoney")
def crawl_eastmoney(as_of: Annotated[str | None, typer.Option()] = None) -> None:
    """Save an immutable licensed Eastmoney A-share daily snapshot."""
    settings = get_settings()
    crawler = EastmoneyDailyCrawler(
        permission_confirmed=settings.eastmoney_permission_confirmed,
        endpoint=settings.eastmoney_endpoint,
        page_size=settings.eastmoney_page_size,
        timeout_seconds=settings.eastmoney_request_timeout_seconds,
        request_delay_seconds=settings.eastmoney_request_delay_seconds,
    )
    result = crawler.crawl_to_storage(settings.data_root, date.fromisoformat(as_of) if as_of else None)
    typer.echo(
        f"created {result.raw_manifest.version}: raw={result.raw_manifest.rows}, "
        f"training_ready={result.training_manifest.rows}"
    )


@app.command("weekly-report")
def weekly_report(as_of: Annotated[str | None, typer.Option()] = None) -> None:
    """Build and atomically publish the latest static sixty-session report."""
    settings = get_settings()
    report, metadata, target = WeeklyReportService(settings).generate(
        date.fromisoformat(as_of) if as_of else None,
        validate_sources=True,
    )
    typer.echo(
        f"created {target.name}: {report.period_start}..{report.period_end}, "
        f"status={report.status}, run_id={metadata.run_id}"
    )


if __name__ == "__main__":
    app()
