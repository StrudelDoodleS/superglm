# Multi-SCOP Discrete Convergence Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce unnecessary REML outer iterations for `multi-SCOP` `discrete=True` fits by adding internal plateau detection and conservative active-lambda freezing without changing fitted curves or public API.

**Architecture:** Keep the core behavior change isolated to `src/superglm/reml/scop_efs.py`. Replace the current ad hoc staged-freeze logic with explicit private helpers that only activate for `multi-SCOP discrete`, then verify the new behavior with a dedicated regression test module and a targeted benchmark script that compares optimized vs baseline behavior by monkeypatching the private activation helper.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, SuperGLM `fit_reml`, `PSpline` SCOP constraints, existing benchmark utilities and `freMTPL2freq.parquet`

---

## File Map

### Core Solver Logic

- Modify: `src/superglm/reml/scop_efs.py`
  - add private activation helper for `multi-SCOP discrete`
  - add private helpers for stability counting, freezing, and plateau checks
  - replace the current generic staged-freeze branch with a `multi-SCOP discrete`-only path
  - record `active_names` and `frozen_names` in debug traces

### Regression Tests

- Create: `tests/test_multi_scop_discrete_convergence.py`
  - unit tests for the new private helper functions
  - integration regression comparing optimized behavior to baseline behavior with cleanup disabled
  - guardrail that single-SCOP and exact cases do not activate the cleanup path

### Benchmark Verification

- Create: `benchmarks/benchmark_multi_scop_discrete_convergence.py`
  - run baseline vs optimized `multi-SCOP discrete` fits on synthetic data and full `freMTPL2`
  - write summary CSV under `benchmarks/results/`
  - print runtime, REML iterations, and prediction-difference metrics

## Task 1: Add Red Tests For The New Convergence Helpers

**Files:**
- Create: `tests/test_multi_scop_discrete_convergence.py`

- [ ] **Step 1: Write helper-level red tests**

Create `tests/test_multi_scop_discrete_convergence.py` with:

```python
import numpy as np
import pandas as pd

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM


def test_cleanup_enabled_only_for_multi_scop_discrete():
    assert scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=2)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=1)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=False, scop_term_count=2)


def test_floor_pinned_lambda_freezes_after_stability_window():
    stable_counts = {"DrivAge": 0, "BonusMalus": 2}
    active_names = {"DrivAge", "BonusMalus"}
    frozen_names = set()

    stable_counts = scop_efs._update_multi_scop_discrete_stability_counts(
        lambdas_old={"DrivAge": 0.12, "BonusMalus": 1.0e-4},
        lambdas_new={"DrivAge": 0.118, "BonusMalus": 1.0e-4},
        active_names=active_names,
        stable_counts=stable_counts,
    )
    active_names, frozen_names = scop_efs._freeze_multi_scop_discrete_lambdas(
        active_names=active_names,
        frozen_names=frozen_names,
        lambdas_new={"DrivAge": 0.118, "BonusMalus": 1.0e-4},
        stable_counts=stable_counts,
    )

    assert active_names == {"DrivAge"}
    assert frozen_names == {"BonusMalus"}
```

- [ ] **Step 2: Write an integration red test comparing optimized vs baseline behavior**

Append:

```python
def _make_multi_scop_data(n: int = 1500, seed: int = 42):
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18.0, 85.0, size=n)
    veh_age = rng.uniform(0.0, 20.0, size=n)
    bonus_malus = rng.uniform(50.0, 150.0, size=n)
    density = rng.uniform(10.0, 5000.0, size=n)
    area = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    eta = (
        -2.3
        - 0.018 * (driv_age - 45.0) ** 2 / 25.0
        - 0.0015 * (bonus_malus - 90.0) ** 2 / 12.0
        + 0.02 * np.sin(veh_age / 3.0)
        + 0.08 * np.log(density)
        + np.where(area == "B", 0.1, 0.0)
        + np.where(area == "C", -0.08, 0.0)
    )
    exposure = rng.uniform(0.2, 1.5, size=n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float) / exposure
    X = pd.DataFrame(
        {
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "BonusMalus": bonus_malus,
            "LogDensity": np.log(density),
            "Area": area,
        }
    )
    return X, y, exposure.astype(float)


def _make_model() -> SuperGLM:
    return SuperGLM(
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


def test_multi_scop_discrete_cleanup_preserves_predictions(monkeypatch):
    X, y, w = _make_multi_scop_data()

    optimized = _make_model()
    optimized.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_cleanup_enabled",
        lambda *, discrete, scop_term_count: False,
    )
    baseline = _make_model()
    baseline.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

    pred_opt = optimized.predict(X)
    pred_base = baseline.predict(X)
    np.testing.assert_allclose(pred_opt, pred_base, rtol=1e-4, atol=1e-6)
    assert set(optimized._reml_lambdas) == set(baseline._reml_lambdas)
```

- [ ] **Step 3: Run the new tests to confirm they fail**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_convergence.py -q'
```

Expected:

```text
FAIL because superglm.reml.scop_efs does not yet define the new private helpers
```

- [ ] **Step 4: Commit the red tests**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add tests/test_multi_scop_discrete_convergence.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "test: add multi-SCOP discrete convergence coverage"
```

## Task 2: Add Private Multi-SCOP Discrete Convergence Helpers

**Files:**
- Modify: `src/superglm/reml/scop_efs.py`

- [ ] **Step 1: Add private internal thresholds and activation helper**

At the top of `src/superglm/reml/scop_efs.py`, add:

```python
_MULTI_SCOP_DISCRETE_LAMBDA_FLOOR = 1.0e-4
_MULTI_SCOP_DISCRETE_FLOOR_FACTOR = 1.05
_MULTI_SCOP_DISCRETE_LOG_STEP_TOL = 1.0e-3
_MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS = 3
_MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL = 5.0e-3
_MULTI_SCOP_DISCRETE_OBJ_REL_TOL = 1.0e-6


def _multi_scop_discrete_cleanup_enabled(*, discrete: bool, scop_term_count: int) -> bool:
    return bool(discrete and scop_term_count > 1)
```

- [ ] **Step 2: Add stability-count and freeze helpers**

In the same file, add:

```python
def _update_multi_scop_discrete_stability_counts(
    *,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
    stable_counts: dict[str, int],
) -> dict[str, int]:
    updated = dict(stable_counts)
    for name in active_names:
        lam_old = max(lambdas_old[name], 1.0e-10)
        lam_new = max(lambdas_new[name], 1.0e-10)
        log_step = abs(np.log(lam_new) - np.log(lam_old))
        near_floor = lam_new <= _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
        if near_floor or log_step < _MULTI_SCOP_DISCRETE_LOG_STEP_TOL:
            updated[name] = updated.get(name, 0) + 1
        else:
            updated[name] = 0
    return updated


def _freeze_multi_scop_discrete_lambdas(
    *,
    active_names: set[str],
    frozen_names: set[str],
    lambdas_new: dict[str, float],
    stable_counts: dict[str, int],
) -> tuple[set[str], set[str]]:
    active_out = set(active_names)
    frozen_out = set(frozen_names)
    for name in list(active_names):
        lam_new = lambdas_new[name]
        near_floor = lam_new <= _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
        stable_long_enough = stable_counts.get(name, 0) >= _MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS
        if near_floor and stable_long_enough:
            active_out.discard(name)
            frozen_out.add(name)
    return active_out, frozen_out
```

- [ ] **Step 3: Add active-set plateau helper**

Still in `src/superglm/reml/scop_efs.py`, add:

```python
def _multi_scop_discrete_plateau_converged(
    *,
    obj_rel_change: float,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
) -> bool:
    if not active_names:
        return True
    active_changes = [
        abs(np.log(max(lambdas_new[name], 1.0e-10)) - np.log(max(lambdas_old[name], 1.0e-10)))
        for name in active_names
    ]
    max_active_change = max(active_changes) if active_changes else 0.0
    return (
        obj_rel_change < _MULTI_SCOP_DISCRETE_OBJ_REL_TOL
        and max_active_change < _MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL
    )
```

- [ ] **Step 4: Run the helper tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_convergence.py -q -k "cleanup_enabled_only or floor_pinned_lambda_freezes"'
```

Expected:

```text
2 passed
1 deselected
```

- [ ] **Step 5: Commit the helper layer**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add src/superglm/reml/scop_efs.py tests/test_multi_scop_discrete_convergence.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "feat: add multi-SCOP discrete convergence helpers"
```

## Task 3: Wire The Helpers Into `optimize_scop_efs_reml`

**Files:**
- Modify: `src/superglm/reml/scop_efs.py`
- Modify: `tests/test_multi_scop_discrete_convergence.py`
- Test: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Narrow activation to `multi-SCOP discrete`**

Inside `optimize_scop_efs_reml`, replace the current unconditional staged-freeze setup with:

```python
scop_term_count = sum(1 for group in groups if group.monotone_engine == "scop")
has_discrete_scop = any(st.get("bin_idx") is not None for st in boot_scop_states.values())
enable_multi_scop_discrete_cleanup = _multi_scop_discrete_cleanup_enabled(
    discrete=has_discrete_scop,
    scop_term_count=scop_term_count,
)

active_names: set[str] = set(estimated_names)
frozen_names: set[str] = set()
stable_counts: dict[str, int] = {name: 0 for name in estimated_names}
```

- [ ] **Step 2: Replace the current staged freeze / unfreeze branch**

In the main outer loop, replace the existing `freeze_threshold`, `_unfreeze_pending`, and `_unfreeze_scheduled` logic with:

```python
if enable_multi_scop_discrete_cleanup:
    stable_counts = _update_multi_scop_discrete_stability_counts(
        lambdas_old=lambdas,
        lambdas_new=lambdas_new,
        active_names=active_names,
        stable_counts=stable_counts,
    )
    active_names, frozen_names = _freeze_multi_scop_discrete_lambdas(
        active_names=active_names,
        frozen_names=frozen_names,
        lambdas_new=lambdas_new,
        stable_counts=stable_counts,
    )
else:
    active_names = set(estimated_names)
    frozen_names.clear()
```

- [ ] **Step 3: Base plateau checks and debug rows on the active set**

Update convergence and debug logging to use the helper:

```python
strict_converged = max_change < reml_tol
plateau_converged = False
if enable_multi_scop_discrete_cleanup and n_reml_iter >= 3:
    plateau_converged = _multi_scop_discrete_plateau_converged(
        obj_rel_change=obj_rel_change,
        lambdas_old=lambdas,
        lambdas_new=lambdas_new,
        active_names=active_names,
    )
elif n_reml_iter >= 3:
    plateau_converged = obj_rel_change < 1e-6 and max_change < 0.01

if debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2:
    debug_recorder.append_jsonl(
        "reml",
        {
            "iteration": n_reml_iter,
            "objective_before": float(obj_curr),
            "objective_after": float(obj_after),
            "lambda_max_delta": float(max_change),
            "objective_relative_change": float(obj_rel_change),
            "strict_converged": bool(strict_converged),
            "plateau_converged": bool(plateau_converged),
            "active_names": sorted(active_names),
            "frozen_names": sorted(frozen_names),
            "lambdas": {name: float(value) for name, value in lambdas_new.items()},
        },
    )
```

- [ ] **Step 4: Run the integration regression and debug tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_convergence.py tests/test_fit_reml_debug.py -q'
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 5: Run the discretized REML regression checks**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_discretize_fit.py -q -k "reml_discrete or freml_lambdas_close_to_exact"'
```

Expected:

```text
selected discretized REML tests passed
```

- [ ] **Step 6: Run ruff on the touched files**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check src/superglm/reml/scop_efs.py tests/test_multi_scop_discrete_convergence.py tests/test_fit_reml_debug.py'
```

Expected:

```text
All checks passed!
```

- [ ] **Step 7: Commit the solver wiring**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add src/superglm/reml/scop_efs.py tests/test_multi_scop_discrete_convergence.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "feat: tighten multi-SCOP discrete convergence"
```

## Task 4: Add A Targeted Benchmark Script And Verify Synthetic + `freMTPL2`

**Files:**
- Create: `benchmarks/benchmark_multi_scop_discrete_convergence.py`

- [ ] **Step 1: Add a benchmark harness that compares baseline vs optimized behavior**

Create `benchmarks/benchmark_multi_scop_discrete_convergence.py` with:

```python
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "multi_scop_discrete_convergence.csv"


def _make_model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": PSpline(n_knots=12, penalty="ssp", constraint=Constraint.fit.concave),
            "VehAge": CubicRegressionSpline(n_knots=10),
            "BonusMalus": PSpline(n_knots=12, penalty="ssp", constraint=Constraint.fit.concave),
            "LogDensity": CubicRegressionSpline(n_knots=10),
            "Area": Categorical(base="most_exposed"),
        },
    )


def _fit_variant(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, cleanup_enabled: bool) -> dict:
    original = scop_efs._multi_scop_discrete_cleanup_enabled
    if not cleanup_enabled:
        scop_efs._multi_scop_discrete_cleanup_enabled = (
            lambda *, discrete, scop_term_count: False
        )
    try:
        model = _make_model()
        t0 = time.perf_counter()
        model.fit_reml(X, y, sample_weight=w, max_reml_iter=20)
        runtime_s = time.perf_counter() - t0
    finally:
        scop_efs._multi_scop_discrete_cleanup_enabled = original

    return {
        "runtime_s": runtime_s,
        "n_reml_iter": int(model._reml_result.n_reml_iter),
        "n_pirls_iter": int(model._result.n_iter),
        "pred": np.asarray(model.predict(X), dtype=float),
        "lambdas": {k: float(v) for k, v in model._reml_lambdas.items()},
    }
```

- [ ] **Step 2: Add synthetic and `freMTPL2` runners**

Extend the script with:

```python
def _prediction_metrics(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = other - reference
    return {
        "prediction_rmse": float(np.sqrt(np.mean(diff**2))),
        "prediction_max_abs": float(np.max(np.abs(diff))),
    }


def _load_fremtpl() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = Path("data/freMTPL2freq.parquet")
    df = pd.read_parquet(path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    df["LogDensity"] = np.log(df["Density"].clip(lower=1.0))
    X = df[["DrivAge", "VehAge", "BonusMalus", "LogDensity", "Area"]].copy()
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=float)
    w = df["Exposure"].to_numpy(dtype=float)
    return X, y, w
```

- [ ] **Step 3: Run the benchmark script**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/benchmark_multi_scop_discrete_convergence.py'
```

Expected:

```text
summary rows printed for synthetic and freMTPL2, and benchmarks/results/multi_scop_discrete_convergence.csv written
```

- [ ] **Step 4: Commit the benchmark harness**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints add benchmarks/benchmark_multi_scop_discrete_convergence.py
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit -m "bench: add multi-SCOP discrete convergence check"
```

## Task 5: Final Verification And Branch Summary

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused test suite**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_multi_scop_discrete_convergence.py tests/test_fit_reml_debug.py tests/test_discretize_fit.py -q -k "multi_scop_discrete_convergence or reml_discrete or freml_lambdas_close_to_exact or debug_level"'
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 2: Run a final lint pass**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check src/superglm/reml/scop_efs.py tests/test_multi_scop_discrete_convergence.py benchmarks/benchmark_multi_scop_discrete_convergence.py'
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
optimized rows with `n_reml_iter` less than or equal to baseline and tiny prediction differences
```

- [ ] **Step 4: Commit the final verification state**

Run:

```bash
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints status --short
rtk git -C /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints commit --allow-empty -m "chore: verify multi-SCOP discrete convergence cleanup"
```

## Spec Coverage Check

- `multi-SCOP discrete` only: covered by activation helper and integration test
- no public tolerance knob: covered by private constants and no API edits
- plateau-based stop: covered by `_multi_scop_discrete_plateau_converged`
- floor-pinned / inactive lambda freezing: covered by stability-count and freeze helpers
- no change to fitted curves: covered by baseline-vs-optimized prediction comparison
- synthetic + `freMTPL2` verification: covered by the benchmark script

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Every code-modifying step includes concrete code
- Every verification step has exact commands and expected outcomes

## Type And Naming Consistency Check

- private activation helper: `_multi_scop_discrete_cleanup_enabled`
- stability helper: `_update_multi_scop_discrete_stability_counts`
- freeze helper: `_freeze_multi_scop_discrete_lambdas`
- plateau helper: `_multi_scop_discrete_plateau_converged`
- benchmark artifact: `benchmarks/results/multi_scop_discrete_convergence.csv`
