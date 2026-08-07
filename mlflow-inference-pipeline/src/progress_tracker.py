"""
progress_tracker.py

Tracks per-item status (completed / failed / skipped) in a JSON file
next to the outputs, so a run can be safely interrupted and resumed
without repeating already-finished work.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class ProgressTracker:
    def __init__(self, status_file: Path):
        self.status_file = Path(status_file)
        self.status: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if self.status_file.exists():
            with open(self.status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(self.status, f, indent=2)

    def is_done(self, key: str) -> bool:
        return self.status.get(key, {}).get("status") in ("completed", "skipped")

    def mark(self, key: str, status: str, error: Optional[str] = None) -> None:
        entry = {"status": status, "timestamp": datetime.now().isoformat()}
        if error:
            entry["error"] = error
        self.status[key] = entry
        self.save()

    def summary(self) -> Dict[str, int]:
        counts = {"completed": 0, "failed": 0, "skipped": 0}
        for entry in self.status.values():
            s = entry.get("status")
            if s in counts:
                counts[s] += 1
        return counts
