from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class PipelineRun:
    run_id: str
    as_of_date: str
    status: str = "pending"
    completed_steps: tuple[str, ...] = ()
    error: str | None = None
    updated_at: str = ""


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, run_id: str) -> Path:
        return self.root / "pipeline_runs" / f"{run_id}.json"

    def load(self, run_id: str) -> PipelineRun | None:
        path = self.path(run_id)
        return PipelineRun(**json.loads(path.read_text())) if path.exists() else None

    def save(self, run: PipelineRun) -> None:
        path = self.path(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        run.updated_at = datetime.now(UTC).isoformat()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
