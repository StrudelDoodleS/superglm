from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class REMLIterRow:
    run_id: str
    iteration: int
    objective_before: float
    objective_after: float
    lambda_max_delta: float


class REMLDebugRecorder:
    def __init__(self, enabled_level: int, base_dir: Path, run_id: str):
        self.enabled_level = enabled_level
        self.base_dir = base_dir
        self.run_id = run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, payload: dict) -> None:
        (self.base_dir / f"{self.run_id}_run.json").write_text(json.dumps(payload, indent=2))

    def append_jsonl(self, suffix: str, payload: dict) -> None:
        path = self.base_dir / f"{self.run_id}_{suffix}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
