# SCOP Lambda Sensitivity Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether `SCOP` lambdas materially change constrained term curves and full-model predictions by comparing integrated `SCOP` REML, passthrough-style lambdas, and a broad fixed-lambda sensitivity grid on single- and multi-`SCOP` exact and `discrete=True` scenarios.

**Architecture:** Keep everything benchmark-local and internal. Build one comparison harness that can produce the integrated `SCOP` REML fit, a passthrough-style constrained refit using unconstrained lambdas, and a fixed-lambda grid of constrained refits. Use that harness to compute curve/prediction similarity metrics and generate plots for `freMTPL` single- and multi-`SCOP` cases in both exact and discrete modes.

**Tech Stack:** Python 3.10+, NumPy, pandas, matplotlib, SuperGLM benchmark scripts, repo-local comparison utilities

---

## File Map

### Benchmark Harness

- Create: `benchmarks/scop_lambda_sensitivity.py`
  - scenario definitions
  - integrated `SCOP` REML baseline
  - passthrough-style lambda comparison
  - fixed-lambda sensitivity sweep
  - similarity metrics
  - curve/prediction plots

### Tests

- Create: `tests/test_scop_lambda_sensitivity.py`
  - internal helper tests
  - metric/sweep schema tests
  - optional monkeypatch regression around the bypass helper

## Task 1: Add Red Tests For The Sensitivity Harness

**Files:**
- Create: `tests/test_scop_lambda_sensitivity.py`

- [ ] **Step 1: Write red tests for the lambda-grid helper**

Create `tests/test_scop_lambda_sensitivity.py` with:

```python
import numpy as np

from benchmarks.scop_lambda_sensitivity import build_lambda_grid


def test_build_lambda_grid_is_log_symmetric_around_baseline():
    values = build_lambda_grid(0.5)
    assert np.allclose(
        values,
        0.5 * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
    )
```

- [ ] **Step 2: Write red tests for curve similarity metrics**

Append:

```python
from benchmarks.scop_lambda_sensitivity import curve_similarity_metrics


def test_curve_similarity_metrics_match_identical_curves():
    x = np.linspace(0.0, 1.0, 50)
    y = x**2
    metrics = curve_similarity_metrics(x, y, y)
    assert metrics["r2"] == 1.0
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["rmse"] == 0.0
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_scop_lambda_sensitivity.py -q'
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'benchmarks.scop_lambda_sensitivity'
```

- [ ] **Step 4: Commit the red tests**

Run:

```bash
rtk git add tests/test_scop_lambda_sensitivity.py
rtk git commit -m "test: add SCOP lambda sensitivity coverage"
```

## Task 2: Add The Benchmark-Local Comparison Helpers

**Files:**
- Create: `benchmarks/scop_lambda_sensitivity.py`
- Modify: `tests/test_scop_lambda_sensitivity.py`

- [ ] **Step 1: Implement lambda-grid and metric helpers**

Create `benchmarks/scop_lambda_sensitivity.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_lambda_grid(baseline: float) -> np.ndarray:
    return baseline * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def curve_similarity_metrics(x: np.ndarray, y_true: np.ndarray, y_other: np.ndarray) -> dict[str, float]:
    residual = y_true - y_other
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "r2": float(r2),
        "max_abs_diff": float(np.max(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }
```

- [ ] **Step 2: Run the helper tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_scop_lambda_sensitivity.py -q'
```

Expected:

```text
2 passed
```

- [ ] **Step 3: Run ruff**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check benchmarks/scop_lambda_sensitivity.py tests/test_scop_lambda_sensitivity.py'
```

Expected:

```text
All checks passed!
```

- [ ] **Step 4: Commit**

Run:

```bash
rtk git add benchmarks/scop_lambda_sensitivity.py tests/test_scop_lambda_sensitivity.py
rtk git commit -m "feat: add SCOP lambda sensitivity helpers"
```

## Task 3: Implement The Benchmark Harness

**Files:**
- Modify: `benchmarks/scop_lambda_sensitivity.py`
- Modify: `tests/test_scop_lambda_sensitivity.py`

- [ ] **Step 1: Add scenario and dataset helpers**

Extend `benchmarks/scop_lambda_sensitivity.py` with:

```python
from dataclasses import dataclass, asdict

from superglm import Constraint, PSpline, SuperGLM


@dataclass(frozen=True)
class SensitivityScenario:
    name: str
    discrete: bool
    n_constrained: int
```

Add `freMTPL` loading helpers using the same path logic as the existing benchmark scripts.

- [ ] **Step 2: Add an output-schema test**

Append to `tests/test_scop_lambda_sensitivity.py`:

```python
from benchmarks.scop_lambda_sensitivity import summarize_result_rows


def test_summarize_result_rows_has_expected_columns():
    df = summarize_result_rows(
        [
            {
                "scenario": "demo",
                "comparison": "baseline",
                "target": "curve",
                "r2": 1.0,
                "max_abs_diff": 0.0,
                "rmse": 0.0,
            }
        ]
    )
    assert list(df.columns) == [
        "scenario",
        "comparison",
        "target",
        "r2",
        "max_abs_diff",
        "rmse",
    ]
```

- [ ] **Step 3: Add the actual comparison harness**

Continue the benchmark file with helpers that:

- run integrated `SCOP` REML
- run passthrough-style lambda constrained refit
- run fixed-lambda constrained refits over the grid
- reconstruct constrained term curves
- compare predictions
- return machine-readable rows

Important:
- keep the bypass internal and benchmark-local
- no public API changes

- [ ] **Step 4: Smoke-run one scenario**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/scop_lambda_sensitivity.py --scenario single_scop_discrete --grid-limit 5'
```

Expected:

```text
visible progress output plus one or more CSV/PNG artifacts
```

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add benchmarks/scop_lambda_sensitivity.py tests/test_scop_lambda_sensitivity.py
rtk git commit -m "feat: add SCOP lambda sensitivity benchmark harness"
```

## Task 4: Run The Full Experiment Matrix And Summarize It

**Files:**
- Modify: `benchmarks/scop_lambda_sensitivity.py`

- [ ] **Step 1: Support the full matrix**

Make sure the harness runs:

- single-`SCOP` exact
- single-`SCOP` discrete
- multi-`SCOP` exact
- multi-`SCOP` discrete

and for each:

- integrated `SCOP` REML baseline
- passthrough-style lambda refit
- fixed-lambda grid sweep

- [ ] **Step 2: Add output writing**

Write:

- summary CSV
- per-feature curve comparison CSV
- prediction comparison CSV
- curve overlay plots
- difference plots
- prediction-comparison plots

- [ ] **Step 3: Run the experiment**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/scop_lambda_sensitivity.py'
```

Expected:

```text
visible progress output and artifact paths for all exact/discrete single- and multi-SCOP scenarios
```

- [ ] **Step 4: Commit**

Run:

```bash
rtk git add benchmarks/scop_lambda_sensitivity.py
rtk git commit -m "bench: add SCOP lambda sensitivity experiment"
```

## Spec Coverage Check

- integrated baseline: covered
- passthrough-style lambda comparison: covered
- broad fixed-lambda grid: covered
- exact and discrete: covered
- single- and multi-SCOP: covered
- curve and prediction metrics: covered
- visual outputs: covered

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Every code step contains concrete code
- Every verification step has exact commands

## Type And Naming Consistency Check

- `build_lambda_grid`
- `curve_similarity_metrics`
- `SensitivityScenario`
- `summarize_result_rows`

These names stay stable across the plan.
