# PR105 Multi-SCOP Discrete Wall-Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine why the current multi-SCOP discrete cleanup path does not reduce wall time and make one narrow internal change, if justified by the measurements, that improves repeated `freMTPL2` wall time without materially changing predictions or lambdas.

**Architecture:** Keep PR105 internal and benchmark-driven. First strengthen the benchmark harness so it measures gate activation, actual freezing, and repeated-run wall time without order bias. Then add targeted debug instrumentation to the managed cleanup path so we can see whether names freeze, when they freeze, and whether that changes aggregate work. Only after that evidence exists should we touch the solver logic, and even then only in the narrowest place justified by the measurements.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, existing SuperGLM SCOP REML path, benchmark harnesses, `freMTPL2freq.parquet`

---

## File Map

### Benchmark Harness

- Modify: `benchmarks/benchmark_multi_scop_discrete_convergence.py`
  - repeated runs
  - per-run order control
  - freeze / gate metrics in CSV
  - richer summary output

### Cleanup Diagnostics

- Modify: `src/superglm/reml/scop_efs.py`
  - expose managed cleanup diagnostics in debug rows / result metadata
  - keep changes internal only

### Result / Debug Summaries

- Modify: `src/superglm/reml/result.py`
  - add optional fields for managed cleanup diagnostics if needed
- Modify: `src/superglm/model/reml_debug.py`
  - extend summary loading/writing to include managed cleanup metrics if helpful

### Regression Tests

- Create: `tests/test_multi_scop_discrete_walltime.py`
  - benchmark-helper coverage
  - cleanup diagnostic metadata coverage
  - no public API changes

## Task 1: Strengthen The Benchmark Harness Before Changing Solver Logic

**Files:**
- Modify: `benchmarks/benchmark_multi_scop_discrete_convergence.py`
- Create: `tests/test_multi_scop_discrete_walltime.py`

- [ ] **Step 1: Add red tests for repeated-run benchmark structure**

Create `tests/test_multi_scop_discrete_walltime.py` with:

```python
from pathlib import Path

import pandas as pd

from benchmarks import benchmark_multi_scop_discrete_convergence as bench


def test_prediction_metrics_zero_for_identical_inputs():
    out = bench._prediction_metrics(
        reference=pd.Series([1.0, 2.0, 3.0]).to_numpy(),
        other=pd.Series([1.0, 2.0, 3.0]).to_numpy(),
    )
    assert out["rmse"] == 0.0
    assert out["max_abs_diff"] == 0.0


def test_summary_csv_columns_include_gate_and_order_fields():
    row = bench.SummaryRow(
        dataset="demo",
        n_rows=10,
        execution_order="baseline->optimized",
        baseline_runtime_s=1.0,
        optimized_runtime_s=0.9,
        speedup_x=1.1111111111111112,
        baseline_n_reml_iter=4,
        optimized_n_reml_iter=3,
        baseline_n_pirls_iter=12,
        optimized_n_pirls_iter=10,
        baseline_converged=True,
        optimized_converged=True,
        baseline_cleanup_gate_calls=1,
        optimized_cleanup_gate_calls=1,
        baseline_cleanup_gate_true_count=0,
        optimized_cleanup_gate_true_count=1,
        pred_rmse=0.0,
        pred_max_abs_diff=0.0,
        lambda_max_abs_diff=0.0,
        lambda_keys_match=True,
        baseline_lambdas_json=\"{}\",
        optimized_lambdas_json=\"{}\",
    )
    frame = pd.DataFrame([row.__dict__])
    assert \"execution_order\" in frame.columns
    assert \"optimized_cleanup_gate_true_count\" in frame.columns
```

- [ ] **Step 2: Add repeated-run support and freeze metrics to the harness**

In `benchmarks/benchmark_multi_scop_discrete_convergence.py`, extend the dataclasses and summary logic to include:

```python
repeats: int
baseline_frozen_count: int
optimized_frozen_count: int
baseline_freeze_iter: int | None
optimized_freeze_iter: int | None
```

and add a `--repeats` CLI argument with default `3`.

- [ ] **Step 3: Run the tests to confirm the new benchmark fields are wired**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_walltime.py -q'
```

Expected:

```text
tests should fail or be incomplete until the repeated-run / summary changes are fully wired
```

- [ ] **Step 4: Implement repeated-run execution with alternating order**

In the benchmark script:

- run both `baseline->optimized` and `optimized->baseline`
- aggregate over repeated runs
- write median wall times and the freeze/gate metrics to CSV
- keep the current prediction/lambda parity metrics

Use the existing `_fit_variant(...)` shape, but return enough information to summarize:

```python
{
    "runtime_s": ...,
    "n_reml_iter": ...,
    "n_pirls_iter": ...,
    "cleanup_gate_calls": ...,
    "cleanup_gate_true_count": ...,
    "frozen_count": ...,
    "freeze_iter": ...,
    ...
}
```

- [ ] **Step 5: Run the benchmark harness**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/benchmark_multi_scop_discrete_convergence.py --repeats 3'
```

Expected:

```text
visible repeated-run summary for synthetic and freMTPL2 plus a refreshed CSV with gate/freeze fields
```

- [ ] **Step 6: Commit**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add benchmarks/benchmark_multi_scop_discrete_convergence.py tests/test_multi_scop_discrete_walltime.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "bench: strengthen multi-SCOP discrete wall-time harness"
```

## Task 2: Add Cleanup Diagnostics To The SCOP REML Path

**Files:**
- Modify: `src/superglm/reml/scop_efs.py`
- Modify: `src/superglm/reml/result.py`
- Modify: `tests/test_multi_scop_discrete_walltime.py`

- [ ] **Step 1: Add a red test for managed cleanup diagnostics**

Append to `tests/test_multi_scop_discrete_walltime.py`:

```python
import numpy as np
import pandas as pd

from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM


def _make_multi_scop_data(n: int = 1500, seed: int = 123):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "DrivAge": rng.uniform(18.0, 85.0, size=n),
            "VehAge": rng.uniform(0.0, 20.0, size=n),
            "BonusMalus": rng.uniform(50.0, 150.0, size=n),
            "LogDensity": np.log(rng.uniform(1.0, 5000.0, size=n)),
            "Area": rng.choice(["A", "B", "C"], size=n),
        }
    )
    eta = (
        -2.4
        - 0.018 * (x["DrivAge"] - 45.0) ** 2 / 25.0
        - 0.0015 * (x["BonusMalus"] - 90.0) ** 2 / 12.0
        + 0.02 * np.sin(x["VehAge"] / 3.0)
        + 0.08 * x["LogDensity"]
    )
    exposure = rng.uniform(0.2, 1.5, size=n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float) / exposure
    return x, y.astype(float), exposure.astype(float)


def test_reml_result_exposes_multi_scop_cleanup_metrics():
    X, y, w = _make_multi_scop_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "VehAge": CubicRegressionSpline(n_knots=8),
            "BonusMalus": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "LogDensity": CubicRegressionSpline(n_knots=8),
            "Area": Categorical(base="most_exposed"),
        },
    )
    model.fit_reml(X, y, sample_weight=w, max_reml_iter=12)

    reml_result = model._reml_result
    assert hasattr(reml_result, "managed_cleanup_names")
    assert hasattr(reml_result, "managed_cleanup_frozen_names")
```

- [ ] **Step 2: Add optional cleanup-diagnostic fields to `REMLResult`**

In `src/superglm/reml/result.py`, extend `REMLResult` with optional fields:

```python
managed_cleanup_names: list[str] | None = None
managed_cleanup_frozen_names: list[str] | None = None
managed_cleanup_freeze_iter: int | None = None
managed_cleanup_active_history: list[list[str]] | None = None
managed_cleanup_frozen_history: list[list[str]] | None = None
```

- [ ] **Step 3: Populate these fields in `optimize_scop_efs_reml()`**

In `src/superglm/reml/scop_efs.py`, record:

- which names are under managed cleanup
- when names freeze
- per-iteration active/frozen history

and return them through `REMLResult`.

- [ ] **Step 4: Run the tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_walltime.py -q'
```

Expected:

```text
all cleanup-diagnostic tests pass
```

- [ ] **Step 5: Commit**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add src/superglm/reml/scop_efs.py src/superglm/reml/result.py tests/test_multi_scop_discrete_walltime.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "feat: expose multi-SCOP cleanup diagnostics"
```

## Task 3: Use The Diagnostics To Make One Narrow Solver Adjustment

**Files:**
- Modify: `src/superglm/reml/scop_efs.py`
- Modify: `tests/test_multi_scop_discrete_walltime.py`

- [ ] **Step 1: Add a red benchmark/diagnostic regression**

Append a focused test that locks in one intended improvement, for example:

```python
def test_managed_cleanup_can_freeze_floor_pinned_lambda():
    X, y, w = _make_multi_scop_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "VehAge": CubicRegressionSpline(n_knots=8),
            "BonusMalus": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "LogDensity": CubicRegressionSpline(n_knots=8),
            "Area": Categorical(base="most_exposed"),
        },
    )
    model.fit_reml(X, y, sample_weight=w, max_reml_iter=12)
    assert model._reml_result.managed_cleanup_names
    assert model._reml_result.managed_cleanup_frozen_names is not None
```

- [ ] **Step 2: Make one narrow internal change only if diagnostics justify it**

Use the repeated-run benchmark plus the new cleanup diagnostics to choose one
targeted change.

Examples of acceptable scope:

- freeze earlier once a managed name is consecutively floor-pinned
- avoid bookkeeping on empty managed sets
- skip helper checks after all managed names are frozen

Do **not** add a public parameter.

- [ ] **Step 3: Rerun the benchmark harness**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/benchmark_multi_scop_discrete_convergence.py --repeats 3'
```

Expected:

```text
repeated-run summary with freeze metrics showing whether the change actually reduces wall time
```

- [ ] **Step 4: Commit**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add src/superglm/reml/scop_efs.py tests/test_multi_scop_discrete_walltime.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "fix: tighten multi-SCOP discrete wall-time path"
```

## Task 4: Final Verification And Branch Readout

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused PR105 test slice**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_walltime.py tests/test_fit_reml_debug.py tests/test_discretize_fit.py -q'
```

Expected:

```text
selected tests pass
```

- [ ] **Step 2: Run lint on touched files**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check benchmarks/benchmark_multi_scop_discrete_convergence.py src/superglm/reml/scop_efs.py src/superglm/reml/result.py tests/test_multi_scop_discrete_walltime.py'
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Inspect the benchmark CSV**

Run:

```bash
rtk read /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints/benchmarks/results/multi_scop_discrete_convergence.csv
```

Expected:

```text
repeated-run rows including order, gate, freeze, runtime, and parity metrics
```

- [ ] **Step 4: Commit the verification checkpoint**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit --allow-empty -m "chore: verify PR105 multi-SCOP discrete wall-time pass"
```

## Spec Coverage Check

- wall time is the primary metric: covered by Task 1 repeated benchmark harness
- iterations only diagnostic: covered by benchmark/diagnostic fields rather than success criteria
- synthetic + freMTPL2: covered by Task 1 and Task 3 benchmark runs
- no public API changes: no tasks introduce public API
- no canonical-runtime work mixed in: file map and tasks stay within benchmark + SCOP REML internals

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Every code step contains concrete files and concrete behavior to implement
- Every verification step has exact commands and expected outcomes

## Type And Naming Consistency Check

- benchmark CSV remains `benchmarks/results/multi_scop_discrete_convergence.csv`
- cleanup diagnostics live on `REMLResult`
- no new public method names are introduced
