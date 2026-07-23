from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def publish_snapshot(payload: dict[str, Any], root: Path, version: str) -> Path:
    snapshots = root / "published"
    snapshots.mkdir(parents=True, exist_ok=True)
    target = snapshots / f"snapshot-{version}.json"
    if target.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {version}")
    body = payload | {"version": version, "published_at": datetime.now(UTC).isoformat(), "is_stale": False}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)
    stable = snapshots / "stable.json"
    stable_tmp = snapshots / "stable.tmp"
    stable_tmp.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    stable_tmp.replace(stable)
    return target
