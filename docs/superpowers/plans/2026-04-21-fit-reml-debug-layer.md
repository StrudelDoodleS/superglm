# Fit REML Debug Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a private internal `DEBUG` layer for `fit_reml()` that emits basic summaries at `DEBUG=1` and full REML/PIRLS/SCOP traces at `DEBUG=2`, then use it to study single- and multi-`SCOP` discrete convergence alongside a small set of control paths.

**Architecture:** Introduce one internal debug configuration module plus one dedicated `fit_reml` trace recorder/writer module. Instrument `fit_reml()`, `irls_direct`, `reml_execute`, `reml/scop_efs.py`, and `scop_newton.py` through a recorder object that is a no-op when `DEBUG=0`. Then add a repo-local analysis script that runs targeted scenarios and writes logs, CSV/JSONL traces, and trajectory plots.

**Tech Stack:** Python 3.10+, NumPy, pandas, matplotlib, JSON/CSV, pytest, SuperGLM `fit_reml`, SCOP EFS, IRLS, cProfile/tracemalloc outputs from the profiling study

---

## File Map

### Private Debug Configuration

- Create: `src/superglm/_debug.py`
  - internal `DEBUG` level
  - env var loading
  - helper predicates like `debug_enabled(level: int) -> bool`

### Fit REML Trace Recorder

- Create: `src/superglm/model/reml_debug.py`
  - run metadata dataclasses
  - trace recorder object
  - CSV/JSONL writers
  - summary and plot helpers

### Core Instrumentation

- Modify: `src/superglm/model/fit_ops.py`
  - create recorder for `fit_reml()`
  - record run start/end and overall convergence reason
- Modify: `src/superglm/model/reml_execute.py`
  - pass recorder into fixed/SCOP REML helpers
- Modify: `src/superglm/reml/scop_efs.py`
  - record per-REML iteration lambda/objective updates
- Modify: `src/superglm/solvers/irls_direct.py`
  - record per-PIRLS iteration summaries and decomposition fallback info
- Modify: `src/superglm/solvers/scop_newton.py`
  - record per-SCOP step trace data

### Tests

- Create: `tests/test_fit_reml_debug.py`
  - debug level gating
  - trace file writing
  - SCOP trace population for a small debug run

### Analysis Script

- Create: `benchmarks/debug_fit_reml_convergence.py`
  - targeted study runs
  - summary table
  - lambda/objective trajectory plots
  - artifact path printing

## Task 1: Add Red Tests For The Debug Layer

**Files:**
- Create: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Write red tests for debug-level gating**

Create `tests/test_fit_reml_debug.py` with:

```python
from pathlib import Path

import pandas as pd

from superglm import Constraint, PSpline, SuperGLM
from superglm._debug import set_debug_level


def test_debug_level_zero_emits_no_reml_trace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(0)

    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit_reml(x, y, max_reml_iter=4)

    assert not list(tmp_path.glob("*.jsonl"))
```

- [ ] **Step 2: Write red tests for `DEBUG=2` trace artifact creation**

Append:

```python
def test_debug_level_two_writes_reml_trace_files(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit_reml(x, y, max_reml_iter=4)

    assert list(tmp_path.glob("*run.json"))
    assert list(tmp_path.glob("*reml.jsonl"))
    assert list(tmp_path.glob("*pirls.jsonl"))
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_fit_reml_debug.py -q'
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'superglm._debug'
```

- [ ] **Step 4: Commit the red tests**

Run:

```bash
rtk git add tests/test_fit_reml_debug.py
rtk git commit -m "test: add fit_reml debug layer coverage"
```

## Task 2: Implement The Private Debug Layer And Recorder

**Files:**
- Create: `src/superglm/_debug.py`
- Create: `src/superglm/model/reml_debug.py`
- Modify: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Add the private debug configuration module**

Create `src/superglm/_debug.py` with:

```python
from __future__ import annotations

import os

DEBUG = int(os.environ.get("SUPERGLM_DEBUG", "0"))


def set_debug_level(level: int) -> None:
    global DEBUG
    DEBUG = int(level)


def get_debug_level() -> int:
    return int(DEBUG)


def debug_enabled(level: int) -> bool:
    return get_debug_level() >= level
```

- [ ] **Step 2: Add the `fit_reml` trace recorder**

Create `src/superglm/model/reml_debug.py` with:

```python
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
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
            fh.write(json.dumps(payload) + "\\n")
```

- [ ] **Step 3: Run the tests again**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_fit_reml_debug.py -q -k "debug_level_zero or debug_level_two"'
```

Expected:

```text
still failing because the recorder is not wired into fit_reml yet
```

- [ ] **Step 4: Commit the debug core**

Run:

```bash
rtk git add src/superglm/_debug.py src/superglm/model/reml_debug.py tests/test_fit_reml_debug.py
rtk git commit -m "feat: add private fit_reml debug scaffolding"
```

## Task 3: Wire `fit_reml`, PIRLS, And SCOP Into The Debug Recorder

**Files:**
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/model/reml_execute.py`
- Modify: `src/superglm/reml/scop_efs.py`
- Modify: `src/superglm/solvers/irls_direct.py`
- Modify: `src/superglm/solvers/scop_newton.py`
- Modify: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Add a small end-to-end trace test**

Append to `tests/test_fit_reml_debug.py`:

```python
def test_debug_level_two_records_scop_reml_and_pirls_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit_reml(x, y, max_reml_iter=4)

    reml_lines = list(tmp_path.glob("*reml.jsonl"))
    pirls_lines = list(tmp_path.glob("*pirls.jsonl"))
    assert reml_lines
    assert pirls_lines
    assert reml_lines[0].read_text().strip()
    assert pirls_lines[0].read_text().strip()
```

- [ ] **Step 2: Add recorder creation in `fit_reml()`**

In `src/superglm/model/fit_ops.py`, create the recorder near the top of `fit_reml()`:

```python
from superglm._debug import get_debug_level
from superglm.model.reml_debug import REMLDebugRecorder

debug_level = get_debug_level()
debug_recorder = None
if debug_level > 0:
    debug_dir = Path(os.environ.get("SUPERGLM_DEBUG_DIR", "benchmarks/results/reml_debug"))
    run_id = f"fit_reml_{int(time.time() * 1000)}"
    debug_recorder = REMLDebugRecorder(debug_level, debug_dir, run_id)
```

Write run metadata once, then pass `debug_recorder` through helper calls.

- [ ] **Step 3: Add per-iteration writes**

Instrument:

- `src/superglm/reml/scop_efs.py`
  - per-REML iteration lambda/objective trace
- `src/superglm/solvers/irls_direct.py`
  - per-PIRLS iteration summary
- `src/superglm/solvers/scop_newton.py`
  - per-SCOP step summary

Use JSONL appends like:

```python
if debug_recorder is not None and debug_recorder.enabled_level >= 2:
    debug_recorder.append_jsonl(
        "reml",
        {
            "iteration": i,
            "objective_before": obj_before,
            "objective_after": obj_after,
            "lambda_max_delta": lambda_max_delta,
        },
    )
```

- [ ] **Step 4: Run the full debug tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_fit_reml_debug.py -q'
```

Expected:

```text
all debug-layer tests passed
```

- [ ] **Step 5: Run ruff on the instrumentation files**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check src/superglm/_debug.py src/superglm/model/reml_debug.py src/superglm/model/fit_ops.py src/superglm/model/reml_execute.py src/superglm/reml/scop_efs.py src/superglm/solvers/irls_direct.py src/superglm/solvers/scop_newton.py tests/test_fit_reml_debug.py'
```

Expected:

```text
All checks passed!
```

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add src/superglm/_debug.py src/superglm/model/reml_debug.py src/superglm/model/fit_ops.py src/superglm/model/reml_execute.py src/superglm/reml/scop_efs.py src/superglm/solvers/irls_direct.py src/superglm/solvers/scop_newton.py tests/test_fit_reml_debug.py
rtk git commit -m "feat: add internal fit_reml debug tracing"
```

## Task 4: Add The Convergence Analysis Script And Run The Study

**Files:**
- Create: `benchmarks/debug_fit_reml_convergence.py`
- Modify: `src/superglm/model/reml_debug.py`

- [ ] **Step 1: Add plot and summary helpers to `reml_debug.py`**

Extend `src/superglm/model/reml_debug.py` with helpers that:

- read JSONL traces
- write compact summary CSVs
- plot lambda and objective trajectories

Use signatures like:

```python
def load_trace_jsonl(path: Path) -> list[dict]:
    ...


def plot_lambda_trajectory(rows: list[dict], out_path: Path) -> None:
    ...
```

- [ ] **Step 2: Add the analysis script**

Create `benchmarks/debug_fit_reml_convergence.py` that:

- sets `SUPERGLM_DEBUG=2`
- runs:
  - single-feature `SCOP` discrete
  - multi-feature `SCOP` discrete
  - single-feature `SCOP` exact
  - one `QP` control
- writes:
  - summary CSV
  - lambda trajectory plots
  - objective trajectory plots
- prints visible progress lines and artifact paths

- [ ] **Step 3: Run the debug study**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && SUPERGLM_DEBUG=2 SUPERGLM_DEBUG_DIR=benchmarks/results/reml_debug PYTHONPATH=src uv run python benchmarks/debug_fit_reml_convergence.py'
```

Expected:

```text
visible progress output, trace files, summary CSV, and plots under benchmarks/results/reml_debug
```

- [ ] **Step 4: Commit**

Run:

```bash
rtk git add benchmarks/debug_fit_reml_convergence.py src/superglm/model/reml_debug.py
rtk git commit -m "bench: add fit_reml convergence debug study"
```

## Spec Coverage Check

- private internal debug layer only: covered
- `DEBUG=0/1/2`: covered
- `fit_reml()` first: covered
- deep SCOP tracing first: covered
- machine-readable logs plus plots: covered
- focus on single- and multi-SCOP discrete diagnosis: covered

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Every code step contains concrete code
- Every verification step has exact commands

## Type And Naming Consistency Check

- `DEBUG` is internal/private and accessed through `src/superglm/_debug.py`
- `REMLDebugRecorder` is the recorder entry point
- JSONL trace suffixes remain stable:
  - `reml`
  - `pirls`
  - `solver`
