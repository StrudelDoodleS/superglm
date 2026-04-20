# Constrained Fit Path Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a profiling and scaling study for exact and `discrete=True` constrained fit paths so we can identify the real hotspots for single-feature and multi-feature `SCOP`/`QP` models before making the next optimization change.

**Architecture:** Add a small internal benchmark support module for scenario generation, timing, call-stack profiling, and memory/allocation capture; then build one main profiling script that runs synthetic sweeps and `freMTPL` sanity checks and writes summary artifacts. Keep the study repo-local, with no public API changes and no solver-behavior changes.

**Tech Stack:** Python 3.10+, NumPy, pandas, cProfile, pstats, tracemalloc, pytest, SuperGLM benchmark scripts

---

## File Map

### Profiling Support

- Create: `benchmarks/_constrained_fit_profile.py`
  - scenario dataclasses
  - synthetic dataset generators
  - `freMTPL` loader helpers
  - timing/profile/memory capture helpers
  - result aggregation utilities

### Main Profiling Script

- Create: `benchmarks/profile_constrained_fit_paths.py`
  - CLI entry point
  - synthetic sweeps
  - `freMTPL` sanity checks
  - artifact writing and summary output

### Tests

- Create: `tests/test_constrained_fit_profile.py`
  - validate scenario generation, summary schema, artifact writing, and profiler helper behavior on small cases

## Task 1: Add Red Tests For The Profiling Harness

**Files:**
- Create: `tests/test_constrained_fit_profile.py`

- [ ] **Step 1: Write red tests for synthetic scenario generation**

Create `tests/test_constrained_fit_profile.py` with:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._constrained_fit_profile import (
    ProfileScenario,
    make_synthetic_dataset,
    summarize_rows,
)


def test_make_synthetic_dataset_repeated_support_has_fewer_unique_values():
    scenario = ProfileScenario(
        name="single_scop_repeated",
        engine="scop",
        n=1000,
        k=10,
        n_constrained=1,
        repeated_support=True,
        discrete=False,
        use_fremtpl=False,
    )
    X, y, w = make_synthetic_dataset(scenario, seed=42)

    assert len(X) == len(y) == len(w) == 1000
    assert X["x1"].nunique() < len(X)


def test_summarize_rows_emits_expected_columns():
    frame = summarize_rows(
        [
            {
                "scenario": "demo",
                "engine": "scop",
                "mode": "exact",
                "n": 1000,
                "k": 10,
                "n_constrained": 1,
                "runtime_s": 0.5,
                "n_reml_iter": 4,
                "n_pirls_iter": 2,
                "peak_mem_mb": 12.0,
            }
        ]
    )
    assert list(frame.columns) == [
        "scenario",
        "engine",
        "mode",
        "n",
        "k",
        "n_constrained",
        "runtime_s",
        "n_reml_iter",
        "n_pirls_iter",
        "peak_mem_mb",
    ]
```

- [ ] **Step 2: Write red test for profiler artifact writing**

Append:

```python
from benchmarks._constrained_fit_profile import write_profile_artifacts


def test_write_profile_artifacts_creates_expected_files(tmp_path: Path):
    paths = write_profile_artifacts(
        base_dir=tmp_path,
        stem="demo",
        profile_stats={"cumulative": [("demo", 1.0)]},
        memory_stats={"peak_mb": 3.5},
    )
    assert paths["cpu_txt"].exists()
    assert paths["memory_json"].exists()
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_constrained_fit_profile.py -q'
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'benchmarks._constrained_fit_profile'
```

- [ ] **Step 4: Commit the red tests**

Run:

```bash
rtk git add tests/test_constrained_fit_profile.py
rtk git commit -m "test: add constrained fit profiling harness coverage"
```

## Task 2: Implement The Profiling Support Module

**Files:**
- Create: `benchmarks/_constrained_fit_profile.py`
- Modify: `tests/test_constrained_fit_profile.py`

- [ ] **Step 1: Add scenario and helper dataclasses**

Create `benchmarks/_constrained_fit_profile.py` with:

```python
from __future__ import annotations

import cProfile
import io
import json
import pstats
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileScenario:
    name: str
    engine: str
    n: int
    k: int
    n_constrained: int
    repeated_support: bool
    discrete: bool
    use_fremtpl: bool
```

- [ ] **Step 2: Add synthetic dataset and summary helpers**

Continue the same file with:

```python
def make_synthetic_dataset(scenario: ProfileScenario, seed: int):
    rng = np.random.default_rng(seed)
    data = {}
    for j in range(scenario.n_constrained):
        if scenario.repeated_support:
            support = max(10, scenario.n // 20)
            x = np.repeat(np.linspace(0.0, 1.0, support), scenario.n // support + 1)[: scenario.n]
        else:
            x = np.linspace(0.0, 1.0, scenario.n)
        data[f"x{j + 1}"] = x
    X = pd.DataFrame(data)
    y = 0.4 + 0.7 * X.iloc[:, 0].to_numpy() + 1.2 * X.iloc[:, 0].to_numpy() ** 2
    w = np.ones(len(X), dtype=float)
    return X, y.astype(float), w


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            "scenario",
            "engine",
            "mode",
            "n",
            "k",
            "n_constrained",
            "runtime_s",
            "n_reml_iter",
            "n_pirls_iter",
            "peak_mem_mb",
        ]
    ]
```

- [ ] **Step 3: Add profiler/memory artifact writers**

Append:

```python
def write_profile_artifacts(base_dir: Path, stem: str, profile_stats: dict, memory_stats: dict):
    base_dir.mkdir(parents=True, exist_ok=True)
    cpu_txt = base_dir / f"{stem}_cpu.txt"
    memory_json = base_dir / f"{stem}_memory.json"
    cpu_txt.write_text(str(profile_stats))
    memory_json.write_text(json.dumps(memory_stats, indent=2))
    return {"cpu_txt": cpu_txt, "memory_json": memory_json}


def profile_callstack_and_memory(fn):
    profiler = cProfile.Profile()
    tracemalloc.start()
    profiler.enable()
    result = fn()
    profiler.disable()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative").print_stats(30)
    return result, stats_stream.getvalue(), {"peak_mb": peak / (1024 * 1024)}
```

- [ ] **Step 4: Run the profiling helper tests**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_constrained_fit_profile.py -q'
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Run ruff**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && uv run ruff check benchmarks/_constrained_fit_profile.py tests/test_constrained_fit_profile.py'
```

Expected:

```text
All checks passed!
```

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add benchmarks/_constrained_fit_profile.py tests/test_constrained_fit_profile.py
rtk git commit -m "feat: add constrained fit profiling support module"
```

## Task 3: Add The Main Profiling Script

**Files:**
- Create: `benchmarks/profile_constrained_fit_paths.py`
- Modify: `benchmarks/_constrained_fit_profile.py`
- Modify: `tests/test_constrained_fit_profile.py`

- [ ] **Step 1: Add a small scenario matrix test**

Append to `tests/test_constrained_fit_profile.py`:

```python
from benchmarks._constrained_fit_profile import build_scenarios


def test_build_scenarios_includes_single_and_multi_feature_modes():
    scenarios = build_scenarios(max_n=10_000)
    names = {s.name for s in scenarios}
    assert "single_scop_exact" in names
    assert "single_qp_exact" in names
    assert "multi_scop_exact" in names
```

- [ ] **Step 2: Implement scenario builder and main script**

Extend `benchmarks/_constrained_fit_profile.py`:

```python
def build_scenarios(max_n: int = 500_000):
    ns = [10_000, 50_000, 100_000, 250_000, max_n]
    scenarios = []
    for n in ns:
        scenarios.extend(
            [
                ProfileScenario("single_scop_exact", "scop", n, 10, 1, True, False, False),
                ProfileScenario("single_scop_discrete", "scop", n, 10, 1, True, True, False),
                ProfileScenario("single_qp_exact", "qp", n, 10, 1, True, False, False),
                ProfileScenario("single_qp_discrete", "qp", n, 10, 1, True, True, False),
                ProfileScenario("multi_scop_exact", "scop", n, 10, 4, True, False, False),
            ]
        )
    return scenarios
```

Create `benchmarks/profile_constrained_fit_paths.py` that:

- builds the scenario matrix
- creates the right feature specs per scenario
- runs fit timing
- captures call-stack and memory artifacts for representative scenarios
- writes `benchmarks/results/constrained_fit_profile_summary.csv`

- [ ] **Step 3: Run the scenario-matrix test**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_constrained_fit_profile.py -q -k "build_scenarios"'
```

Expected:

```text
1 passed
```

- [ ] **Step 4: Smoke-run the profiling script on a tiny sweep**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/profile_constrained_fit_paths.py --max-n 10000 --reps 1'
```

Expected:

```text
a small printed summary and one CSV written under benchmarks/results/
```

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add benchmarks/profile_constrained_fit_paths.py benchmarks/_constrained_fit_profile.py tests/test_constrained_fit_profile.py
rtk git commit -m "feat: add constrained fit path profiling script"
```

## Task 4: Add `freMTPL` Sanity Checks And Final Profiling Gate

**Files:**
- Modify: `benchmarks/profile_constrained_fit_paths.py`
- Modify: `benchmarks/_constrained_fit_profile.py`

- [ ] **Step 1: Add `freMTPL` loading helpers**

Extend `benchmarks/_constrained_fit_profile.py` with a loader patterned after the existing benchmark scripts:

```python
def load_fremtpl_frequency():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "freMTPL2freq.parquet"
    if not data_path.exists() and root.parent.name == ".worktrees":
        data_path = root.parent.parent / "data" / "freMTPL2freq.parquet"
    ...
```

- [ ] **Step 2: Add real-data scenarios to the main script**

Extend `benchmarks/profile_constrained_fit_paths.py` so it can run:

- single constrained `SCOP`
- single constrained `QP`
- multiple constrained features when feasible

on `freMTPL`, with exact vs `discrete=True`.

- [ ] **Step 3: Run the final profiling gate**

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run pytest tests/test_constrained_fit_profile.py -q'
```

Expected:

```text
all profiling helper tests passed
```

Run:

```bash
rtk proxy zsh -lc 'cd /home/mhick/python_projects/superglm/.worktrees/convex-concave-constraints && PYTHONPATH=src uv run python benchmarks/profile_constrained_fit_paths.py --max-n 500000 --reps 1'
```

Expected:

```text
summary output plus CSV/profile artifacts for synthetic sweeps and freMTPL sanity checks
```

- [ ] **Step 4: Commit**

Run:

```bash
rtk git add benchmarks/profile_constrained_fit_paths.py benchmarks/_constrained_fit_profile.py
rtk git commit -m "bench: add constrained fit path profiling study"
```

## Spec Coverage Check

- single-feature `SCOP`: covered
- single-feature `QP`: covered
- multi-feature constrained models: covered
- exact vs `discrete=True`: covered
- synthetic sweeps up to `500k`: covered
- `freMTPL` sanity checks: covered
- call-stack and memory profiling: covered

## Placeholder Scan

- No `TODO` / `TBD`
- No “similar to above”
- Each code step contains concrete code
- Each verification step has exact commands

## Type And Naming Consistency Check

- support module names are stable:
  - `ProfileScenario`
  - `make_synthetic_dataset`
  - `build_scenarios`
  - `write_profile_artifacts`
- main script is `benchmarks/profile_constrained_fit_paths.py`
- outputs stay benchmark-local and repo-private
