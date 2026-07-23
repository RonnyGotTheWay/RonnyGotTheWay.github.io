from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from art_rank_quant.pipelines.state import PipelineRun, RunStore


def run_daily(
    as_of: date,
    root: Path,
    steps: dict[str, Callable[[], None]],
    run_id: str | None = None,
) -> PipelineRun:
    run_id = run_id or f"daily-{as_of.isoformat()}"
    store = RunStore(root)
    run = store.load(run_id) or PipelineRun(run_id, as_of.isoformat())
    if run.status == "success":
        return run
    run.status = "running"
    store.save(run)
    completed = list(run.completed_steps)
    try:
        for name, action in steps.items():
            if name in completed:
                continue
            action()
            completed.append(name)
            run.completed_steps = tuple(completed)
            store.save(run)
        run.status = "success"
        run.error = None
    except Exception as exception:
        run.status = "failed"
        run.error = f"{type(exception).__name__}: {exception}"
    store.save(run)
    return run
