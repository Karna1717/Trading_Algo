"""Runtime status persistence for dashboard consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_system.config import LOGS_DIR, STATUS_PATH


def write_runtime_status(payload: dict[str, Any], status_path: Path = STATUS_PATH) -> None:
    """Persist runtime status to a JSON file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


def read_runtime_status(status_path: Path = STATUS_PATH) -> dict[str, Any]:
    """Read runtime status from disk."""
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))

