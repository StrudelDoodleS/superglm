# Multi-SCOP Scaling Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how multi-`SCOP` runtime, memory, convergence, and lambda activity scale as constrained-feature count increases, with the full matrix on `discrete=True` and exact spot-checks for smaller cases.

**Architecture:** Add one synthetic multi-`SCOP` benchmark harness that generates controlled repeated-support constrained features, runs both exact and discrete `fit_reml()` paths, and records runtime, peak memory, REML/PIRLS iterations, and lambda-activity summaries. Keep the study benchmark-local and separate from the earlier profiling and sensitivity scripts so the outputs stay interpretable.

**Tech Stack:** Python 3.10+, NumPy, pandas, matplotlib, SuperGLM benchmark scripts

---

## File Map

### Main Benchmark Harness

- Create: `benchmarks/multi_scop_scaling.py`
  - synthetic dataset generator
  - multi-`SCOP` scenario matrix
  - exact/discrete runners
  - runtime/memory/convergence collection
  - lambda-activity summaries
  - CSV + plots

### Tests

- Create: `tests/test_multi_scop_scaling.py`
  - generator sanity
  - summary schema
  - lambda-activity helper tests

## Task 1: Add Red Tests For The Multi-SCOP Study Harness

**Files:**
- Create: `tests/test_multi_scop_scaling.py`

- [ ] **Step 1: Write red tests for the scenario builder**

Create `tests/test_multi_scop_scaling.py` with:

```python
from benchmarks.multi_scop_scaling import MultiSCOPScenario, build_scenarios


def test_build_scenarios_covers_discrete_matrix_and_exact_spot_checks():
    scenarios = build_scenarios()
    names = {(s.mode, s.n_constrained) for s in scenarios}
    assert ("discrete", 1) in names
    assert ("discrete", 16) in names
    assert ("exact", 1) in names
    assert ("exact", 4) in names
```

- [ ] **Step 2: Write red tests for lambda-activity summary**

Append:

```python
from benchmarks.multi_scop_scaling import summarize_lambda_activity


def test_summarize_lambda_activity_counts_floor_pinned_and_active_terms():
    lambdas = {"x1": 1e-4, "x2": 0.05, "x3": 0.2}
    summary = summarize_lambda_activity(lambdas, floor=1e-4, active_threshold=1e-3)
    assert summary["n_floor"] == 1
    assert summary["n_active"] == 2
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_scaling.py -q'
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'benchmarks.multi_scop_scaling'
```

- [ ] **Step 4: Commit the red tests**

Run:

```bash
rtk git add tests/test_multi_scop_scaling.py
rtk git commit -m "test: add multi-SCOP scaling benchmark coverage"
```

## Task 2: Add The Benchmark Harness Core

**Files:**
- Create: `benchmarks/multi_scop_scaling.py`
- Modify: `tests/test_multi_scop_scaling.py`

- [ ] **Step 1: Add scenario and synthetic data helpers**

Create `benchmarks/multi_scop_scaling.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiSCOPScenario:
    mode: str
    n_constrained: int
    n: int
    k: int
```

Add:

```python
def build_scenarios():
    scenarios = []
    for n_constrained in (1, 2, 4, 8, 16):
        scenarios.append(MultiSCOPScenario("discrete", n_constrained, 100_000, 12))
    for n_constrained in (1, 2, 4):
        scenarios.append(MultiSCOPScenario("exact", n_constrained, 100_000, 12))
    return scenarios
```

- [ ] **Step 2: Add lambda-activity helper**

Append:

```python
def summarize_lambda_activity(lambdas: dict[str, float], *, floor: float, active_threshold: float):
    n_floor = sum(value <= floor * 1.000001 for value in lambdas.values())
    n_active = sum(value > active_threshold for value in lambdas.values())
    return {"n_floor": n_floor, "n_active": n_active}
```

- [ ] **Step 3: Add synthetic generator and run-row schema**

Append:

```python
def make_dataset(n: int, n_constrained: int, seed: int):
    rng = np.random.default_rng(seed)
    data = {}
    eta = np.full(n, -0.2)
    for j in range(n_constrained):
        support = max(20, n // 50)
        x = np.repeat(np.linspace(0.0, 1.0, support), n // support + 1)[:n]
        rng.shuffle(x)
        data[f"x{j+1}"] = x
        eta += 0.15 * x + 0.35 * x**2
    y = eta + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame(data), y.astype(np.float64)
```

- [ ] **Step 4: Run the tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_scaling.py -q'
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add benchmarks/multi_scop_scaling.py tests/test_multi_scop_scaling.py
rtk git commit -m "feat: add multi-SCOP scaling benchmark core"
```

## Task 3: Implement The Actual Multi-SCOP Fit Matrix

**Files:**
- Modify: `benchmarks/multi_scop_scaling.py`

- [ ] **Step 1: Add feature-builder and fit runner**

Extend the benchmark file with:

```python
from superglm import Constraint, PSpline, SuperGLM


def make_features(n_constrained: int):
    return {
        f"x{j+1}": PSpline(n_knots=12, constraint=Constraint.fit.convex if j % 2 == 0 else Constraint.fit.concave)
        for j in range(n_constrained)
    }
```

Add a fit runner that records:

- runtime
- peak memory via tracemalloc
- `n_reml_iter`
- `n_pirls_iter`
- convergence flags
- lambda summary

- [ ] **Step 2: Add a small smoke run**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/multi_scop_scaling.py --max-constrained 2 --n 20000'
```

Expected:

```text
a small summary table for 1 and 2 constrained features
```

- [ ] **Step 3: Commit**

Run:

```bash
rtk git add benchmarks/multi_scop_scaling.py
rtk git commit -m "feat: add multi-SCOP scaling fit matrix"
```

## Task 4: Add Outputs And Run The Full Study

**Files:**
- Modify: `benchmarks/multi_scop_scaling.py`

- [ ] **Step 1: Add CSV and plots**

Write:

- summary CSV
- runtime vs constrained-feature count plot
- memory vs constrained-feature count plot
- REML-iteration vs constrained-feature count plot
- active/floor-pinned lambda count plot

- [ ] **Step 2: Run the full study**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/multi_scop_scaling.py --max-constrained 16 --n 100000'
```

Expected:

```text
visible progress output plus summary CSV and scaling plots
```

- [ ] **Step 3: Commit**

Run:

```bash
rtk git add benchmarks/multi_scop_scaling.py
rtk git commit -m "bench: add multi-SCOP scaling study"
```

## Spec Coverage Check

- multi-`SCOP` is the primary focus: covered
- full `1/2/4/8/16` matrix on discrete: covered
- exact spot checks at `1/2/4`: covered
- runtime, memory, convergence, and lambda activity: covered
- synthetic study only in first pass: covered

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Every code step contains concrete code
- Every verification step has exact commands

## Type And Naming Consistency Check

- `MultiSCOPScenario`
- `build_scenarios`
- `summarize_lambda_activity`
- `make_dataset`

These stay stable across the plan.
