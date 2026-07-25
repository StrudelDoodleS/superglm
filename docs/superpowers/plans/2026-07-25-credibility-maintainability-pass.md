# Credibility Solver Maintainability Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the RandomEffect/FactorSmooth and REML implementation without changing
public behavior, numerical results, convergence, memory use, or performance.

**Architecture:** Add one pure internal REML-candidate module shared by exact and discrete
optimizers. Locally simplify compact FactorSmooth support and structured cross assembly while
retaining the current Tabmat/Numba boundary. Keep the ultra-fast GAM redesign in the approved
design document only.

**Tech Stack:** Python 3.13, NumPy, SciPy, Numba, Tabmat, pytest, Ruff, cProfile benchmark
harnesses.

---

## File Map

- Create `src/superglm/reml/convergence.py`: pure projected-gradient and evaluated-candidate
  convergence decisions.
- Create `tests/test_reml_convergence.py`: direct unit characterization for the pure helpers.
- Modify `src/superglm/reml/direct.py`: delegate the duplicated pure decisions.
- Modify `src/superglm/reml/discrete.py`: delegate the same decisions without moving tensor or
  cache logic.
- Modify `src/superglm/_group_matrix/_group_matrix_core.py`: centralize discrete-support access
  and clarify cell/cross names.
- Modify `src/superglm/_group_matrix/_group_matrix_kernels.py`: remove the superseded,
  repository-unreferenced sufficient-statistic kernel.
- Modify `src/superglm/solvers/structured.py`: isolate optimized discrete cross selection from
  fallback construction.
- Use
  `docs/superpowers/specs/2026-07-25-credibility-maintainability-pass-design.md`
  as the architecture exploration; do not implement its future module split.

### Task 1: Freeze baseline behavior and performance

**Files:**

- Read: `benchmarks/profile_structured_credibility.py`
- Read: `benchmarks/benchmark_tensor_ti_freq.py`
- Output: `/tmp/superglm-cred-maint-baseline-b5a0cd0-fs/`
- Output: `/tmp/superglm-cred-maint-baseline-b5a0cd0-sz/`

- [ ] **Step 1: Confirm the starting revision and clean worktree**

Run:

```bash
rtk git status --short
rtk git rev-parse --short b5a0cd0
rtk git diff --name-only b5a0cd0..HEAD
```

Expected: clean status, the frozen production base resolves to `b5a0cd0`, and only this
implementation-plan document follows that base.

- [ ] **Step 2: Run the focused characterization suite**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest \
  tests/test_random_effect*.py \
  tests/test_factor_smooth*.py \
  tests/test_structured*.py \
  tests/test_sum_to_zero_structured_factor.py \
  tests/test_block_schur_factor.py \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py -q
```

Expected: all collected tests pass. Preserve the test count in the work log.

- [ ] **Step 3: Capture the million-row FS baseline**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis fs --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-cred-maint-baseline-b5a0cd0-fs
```

Expected: converged structured fit with `score_objective_tolerance`, seven REML iterations,
and a prediction checksum near `1045154.11521279`.

- [ ] **Step 4: Capture the million-row SZ baseline**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-cred-maint-baseline-b5a0cd0-sz
```

Expected: converged structured fit with `score_objective_tolerance`, five REML iterations,
and a prediction checksum near `1045155.20722107`.

- [ ] **Step 5: Record baseline summary fields**

Run:

```bash
rtk proxy jq \
  '{config, timing:.backends.structured.wall_times_s,
    model:(.backends.structured.model |
      {converged,termination_reason,reml_iterations,objective,deviance,
       effective_df,lambdas,prediction_checksum})}' \
  /tmp/superglm-cred-maint-baseline-b5a0cd0-fs/summary.json
rtk proxy jq \
  '{config, timing:.backends.structured.wall_times_s,
    model:(.backends.structured.model |
      {converged,termination_reason,reml_iterations,objective,deviance,
       effective_df,lambdas,prediction_checksum})}' \
  /tmp/superglm-cred-maint-baseline-b5a0cd0-sz/summary.json
```

Expected: finite diagnostics and deterministic numerical fields.

### Task 2: Extract pure REML candidate decisions

**Files:**

- Create: `src/superglm/reml/convergence.py`
- Create: `tests/test_reml_convergence.py`
- Modify: `src/superglm/reml/direct.py`
- Modify: `src/superglm/reml/discrete.py`

- [ ] **Step 1: Write failing tests for projected gradients**

Create `tests/test_reml_convergence.py` with:

```python
from __future__ import annotations

import numpy as np

from superglm.reml.convergence import (
    evaluate_reml_candidate,
    project_reml_gradient,
)


def test_project_reml_gradient_respects_fixed_and_active_bounds() -> None:
    gradient = np.array([-3.0, 4.0, 5.0, 6.0, -7.0])
    rho = np.array([10.0, -10.0, 0.0, 10.0, -10.0])
    estimated = np.array([True, True, False, True, True])

    projected = project_reml_gradient(
        gradient,
        rho,
        estimated,
        log_lower=-10.0,
        log_upper=10.0,
    )

    np.testing.assert_array_equal(projected, np.array([0.0, 0.0, 0.0, 6.0, -7.0]))
    np.testing.assert_array_equal(gradient, np.array([-3.0, 4.0, 5.0, 6.0, -7.0]))


def test_project_reml_gradient_rejects_misaligned_inputs() -> None:
    with np.testing.assert_raises_regex(ValueError, "identical shapes"):
        project_reml_gradient(
            np.ones(2),
            np.ones(3),
            np.ones(2, dtype=bool),
            log_lower=-10.0,
            log_upper=10.0,
        )
```

- [ ] **Step 2: Write failing tests for evaluated-candidate convergence**

Append:

```python
def test_evaluated_candidate_requires_two_evaluations() -> None:
    first = evaluate_reml_candidate(
        iteration=0,
        objective=12.0,
        previous_objective=12.0,
        projected_gradient=np.zeros(2),
        tolerance=1.0,
    )
    second = evaluate_reml_candidate(
        iteration=1,
        objective=12.0,
        previous_objective=12.0,
        projected_gradient=np.zeros(2),
        tolerance=1.0,
    )

    assert not first.converged
    assert np.isinf(first.objective_change)
    assert second.converged
    assert second.objective_change == 0.0
    assert second.projected_gradient_norm == 0.0
    assert second.score_scale == 13.0


def test_evaluated_candidate_requires_both_score_and_objective_tolerance() -> None:
    score_failure = evaluate_reml_candidate(
        iteration=2,
        objective=9.0,
        previous_objective=9.0,
        projected_gradient=np.array([2.0]),
        tolerance=0.1,
    )
    objective_failure = evaluate_reml_candidate(
        iteration=2,
        objective=9.0,
        previous_objective=7.0,
        projected_gradient=np.zeros(1),
        tolerance=0.1,
    )

    assert not score_failure.converged
    assert not objective_failure.converged
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest tests/test_reml_convergence.py -q
```

Expected: collection fails because `superglm.reml.convergence` does not exist.

- [ ] **Step 4: Implement the pure helper module**

Create `src/superglm/reml/convergence.py` with:

```python
"""Pure convergence decisions shared by exact and discrete REML optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class REMLCandidateConvergence:
    """Diagnostics for one fully evaluated REML lambda candidate."""

    projected_gradient_norm: float
    score_scale: float
    objective_change: float
    converged: bool


def project_reml_gradient(
    gradient: NDArray,
    rho: NDArray,
    estimated_mask: NDArray,
    *,
    log_lower: float,
    log_upper: float,
    bound_window: float = 0.01,
) -> NDArray:
    """Project fixed and outward-pointing bound scores to zero."""
    score = np.asarray(gradient, dtype=np.float64)
    log_lambda = np.asarray(rho, dtype=np.float64)
    estimated = np.asarray(estimated_mask, dtype=bool)
    if score.shape != log_lambda.shape or score.shape != estimated.shape:
        raise ValueError("gradient, rho, and estimated_mask must have identical shapes")

    projected = score.copy()
    fixed = ~estimated
    upper_stationary = estimated & (log_lambda >= log_upper - bound_window) & (score < 0.0)
    lower_stationary = estimated & (log_lambda <= log_lower + bound_window) & (score > 0.0)
    projected[fixed | upper_stationary | lower_stationary] = 0.0
    return projected


def evaluate_reml_candidate(
    *,
    iteration: int,
    objective: float,
    previous_objective: float,
    projected_gradient: NDArray,
    tolerance: float,
) -> REMLCandidateConvergence:
    """Evaluate Wood's compound score/objective stopping criterion."""
    projected = np.asarray(projected_gradient, dtype=np.float64)
    gradient_norm = float(np.max(np.abs(projected))) if projected.size else 0.0
    score_scale = max(1.0 + abs(objective), 1.0)
    objective_change = abs(objective - previous_objective) if iteration > 0 else np.inf
    converged = (
        iteration >= 1
        and gradient_norm < tolerance * score_scale
        and objective_change < tolerance * score_scale
    )
    return REMLCandidateConvergence(
        projected_gradient_norm=gradient_norm,
        score_scale=score_scale,
        objective_change=objective_change,
        converged=converged,
    )
```

- [ ] **Step 5: Run the helper tests and verify GREEN**

Run the Step 3 command.

Expected: four tests pass.

- [ ] **Step 6: Replace duplicated exact-REML decisions**

Import:

```python
from superglm.reml.convergence import evaluate_reml_candidate, project_reml_gradient
```

Replace the manual projected-gradient and compound-convergence calculations in
`optimize_direct_reml` with:

```python
proj_grad = project_reml_gradient(
    grad,
    rho_clipped,
    estimated_mask,
    log_lower=log_lo,
    log_upper=log_hi,
)
candidate_convergence = evaluate_reml_candidate(
    iteration=outer,
    objective=obj,
    previous_objective=prev_obj,
    projected_gradient=proj_grad,
    tolerance=_tol,
)
proj_grad_norm = candidate_convergence.projected_gradient_norm
score_scale = candidate_convergence.score_scale
obj_change = candidate_convergence.objective_change
```

Keep the existing verbose output and `prev_obj = obj` ordering. Replace the compound boolean
block with:

```python
if candidate_convergence.converged:
    converged = True
    termination_reason = "score_objective_tolerance"
    break
```

- [ ] **Step 7: Replace duplicated discrete-REML decisions**

Use the same imports. Replace the discrete manual projection and compound check with:

```python
proj_grad_d = project_reml_gradient(
    grad,
    rho_clipped,
    estimated_mask,
    log_lower=log_lo,
    log_upper=log_hi,
)
candidate_convergence = evaluate_reml_candidate(
    iteration=poi_iter,
    objective=obj,
    previous_objective=prev_obj,
    projected_gradient=proj_grad_d,
    tolerance=_tol,
)
score_scale_d = candidate_convergence.score_scale
proj_grad_norm = candidate_convergence.projected_gradient_norm
obj_change = candidate_convergence.objective_change
```

Preserve verbose output, terminal `rho`, timing updates, and `prev_obj` assignment. Use
`candidate_convergence.converged` for the existing terminal branch.

- [ ] **Step 8: Run REML regression tests**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest \
  tests/test_reml_convergence.py \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py \
  tests/test_factor_smooth_reml.py \
  tests/test_factor_smooth_sz_reml.py \
  tests/test_random_effect_reml.py -q
```

Expected: all tests pass with unchanged feature-test warnings.

- [ ] **Step 9: Lint and commit**

Run:

```bash
rtk ruff check src/superglm/reml tests/test_reml_convergence.py
rtk ruff format --check src/superglm/reml tests/test_reml_convergence.py
rtk git diff --check
rtk git add src/superglm/reml/convergence.py src/superglm/reml/direct.py \
  src/superglm/reml/discrete.py tests/test_reml_convergence.py
rtk git commit -m "Simplify REML candidate convergence"
```

Expected: clean checks and one focused commit.

### Task 3: Simplify compact FactorSmooth assembly

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/solvers/structured.py`
- Test: `tests/test_factor_smooth_discrete.py`
- Test: `tests/test_factor_smooth_structured_system.py`
- Test: `tests/test_structured_allocations.py`

- [ ] **Step 1: Prove the superseded kernel is dead**

Run:

```bash
rtk grep '_factor_smooth_support_sufficient_stats' src tests -g '*.py'
```

Expected: exactly one match, its definition in `_group_matrix_kernels.py`.

- [ ] **Step 2: Run the existing characterization tests before refactoring**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_factor_smooth_structured_parity.py \
  tests/test_factor_smooth_sz_reml.py \
  tests/test_structured_allocations.py -q
```

Expected: all tests pass. These are the green characterization tests for the refactor.

- [ ] **Step 3: Remove the dead kernel**

Delete `_factor_smooth_support_sufficient_stats` from
`src/superglm/_group_matrix/_group_matrix_kernels.py`. Do not remove
`_factor_smooth_support_dense_cross`, which is the compact fallback for unsupported or
mismatched cross geometry.

- [ ] **Step 4: Add one private discrete-support accessor**

Add this method to `FactorSmoothGroupMatrix`:

```python
def _discrete_support(self) -> tuple[NDArray, NDArray] | None:
    """Return immutable discrete basis/index support, or ``None`` for exact geometry."""
    if not self.is_discrete:
        return None
    basis = self.B_unique
    support_index = self.bin_idx
    if basis is None or support_index is None:  # pragma: no cover - constructor invariant
        raise RuntimeError("discrete FactorSmooth support is unavailable")
    return basis, support_index
```

Use it in `factor_smooth_discrete_cell_moments`,
`factor_smooth_discrete_dense_cell_cross_gram`, and
`factor_smooth_discrete_shared_bin_cross_gram`. Preserve the existing `ValueError` text for
calling the first two methods on exact geometry. In the shared-bin method, return `None` when
the accessor returns `None`.

Rename only local variables where doing so clarifies coordinates:

```python
dense_cells = _factor_smooth_support_dense_cell_aggregates(...)
raw_cross = basis.T[None, :, :] @ dense_cells
```

- [ ] **Step 5: Extract optimized cross selection**

Add a private helper near `build_block_structured_system`:

```python
def _optimized_discrete_factor_smooth_cross(
    dominant: FactorSmoothGroupMatrix,
    matrix: GroupMatrix,
    weights: NDArray,
    cell_weights: NDArray | None,
) -> NDArray | None:
    """Use compact cell crosses when the small matrix has eligible geometry."""
    if not dominant.is_discrete:
        return None
    if type(matrix) is DenseGroupMatrix:
        return dominant.factor_smooth_discrete_dense_cell_cross_gram(weights, matrix.M)
    if cell_weights is None:  # pragma: no cover - built with discrete moments above
        raise RuntimeError("discrete FactorSmooth cell weights are unavailable")
    return dominant.factor_smooth_discrete_shared_bin_cross_gram(cell_weights, matrix)
```

Replace the nested optimized-cross selection inside the small-matrix loop with:

```python
optimized_cross = _optimized_discrete_factor_smooth_cross(
    dominant,
    matrix,
    weights,
    cell_weights,
)
if optimized_cross is not None:
    cross_blocks.append(optimized_cross)
    continue
```

Keep the exact, SZ-column, `_cross_gram`, mismatched-bin, and subclass fallbacks unchanged.

- [ ] **Step 6: Run the characterization tests after refactoring**

Run the Step 2 command.

Expected: all tests pass with the same count and warnings.

- [ ] **Step 7: Confirm the obsolete symbol is gone and fallbacks remain**

Run:

```bash
rtk grep '_factor_smooth_support_sufficient_stats' src tests -g '*.py'
rtk grep '_factor_smooth_support_dense_cross' src tests -g '*.py'
```

Expected: zero matches for the removed symbol and both a definition and live caller for the
fallback symbol.

- [ ] **Step 8: Lint and commit**

Run:

```bash
rtk ruff check \
  src/superglm/_group_matrix/_group_matrix_core.py \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/solvers/structured.py
rtk ruff format --check \
  src/superglm/_group_matrix/_group_matrix_core.py \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/solvers/structured.py
rtk git diff --check
rtk git add \
  src/superglm/_group_matrix/_group_matrix_core.py \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/solvers/structured.py
rtk git commit -m "Simplify compact FactorSmooth assembly"
```

Expected: clean checks and one focused commit.

### Task 4: Perform the final bounded readability audit

**Files:**

- Review: `src/superglm/reml/convergence.py`
- Review: `src/superglm/reml/direct.py`
- Review: `src/superglm/reml/discrete.py`
- Review: `src/superglm/_group_matrix/_group_matrix_core.py`
- Review: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Review: `src/superglm/solvers/structured.py`

- [ ] **Step 1: Inspect the final production diff**

Run:

```bash
rtk git diff b5a0cd0..HEAD -- src/superglm
rtk git diff --stat b5a0cd0..HEAD -- src/superglm
```

Expected: only the planned internal helper extraction, local simplification, and dead-code
removal.

- [ ] **Step 2: Scan for stale narration and ambiguous coordinate names**

Run:

```bash
rtk grep 'FIXME|XXX|temporary|legacy|raw_|public_|cell_weights|rho_trial|rho_clipped' \
  src/superglm/reml/convergence.py \
  src/superglm/reml/direct.py \
  src/superglm/reml/discrete.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  src/superglm/solvers/structured.py
```

Expected: no FIXME/XXX markers. Every remaining `raw_`, `public_`, and cell/candidate name has
an unambiguous coordinate or state meaning.

- [ ] **Step 3: Make only evidence-backed wording/name corrections**

Use `apply_patch` for corrections that remove duplication or clarify an invariant. Do not move
module boundaries, change public/private method contracts, or alter control flow in this step.

- [ ] **Step 4: Re-run focused tests after any correction**

Run the Task 1 Step 2 focused characterization command.

Expected: all tests pass.

- [ ] **Step 5: Commit only if this audit changed files**

Run:

```bash
rtk git status --short
rtk git diff --check
rtk git add src/superglm
rtk git commit -m "Clarify credibility solver invariants"
```

Expected: no commit when there are no additional evidence-backed changes; otherwise one
wording-only commit.

### Task 5: Verify numerical and performance parity

**Files:**

- Output: `/tmp/superglm-cred-maint-final-fs/`
- Output: `/tmp/superglm-cred-maint-final-sz/`
- Compare:
  `/tmp/superglm-cred-maint-baseline-b5a0cd0-fs/summary.json`
- Compare:
  `/tmp/superglm-cred-maint-baseline-b5a0cd0-sz/summary.json`

- [ ] **Step 1: Run the final million-row FS benchmark**

Repeat Task 1 Step 3, changing only:

```text
--output-dir /tmp/superglm-cred-maint-final-fs
```

Expected: converged in seven REML iterations.

- [ ] **Step 2: Run the final million-row SZ benchmark**

Repeat Task 1 Step 4, changing only:

```text
--output-dir /tmp/superglm-cred-maint-final-sz
```

Expected: converged in five REML iterations.

- [ ] **Step 3: Assert numerical parity and the 5% wall-time gate**

Run:

```bash
rtk proxy \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  - <<'PY'
import json
import statistics
from pathlib import Path

pairs = {
    "fs": (
        Path("/tmp/superglm-cred-maint-baseline-b5a0cd0-fs/summary.json"),
        Path("/tmp/superglm-cred-maint-final-fs/summary.json"),
    ),
    "sz": (
        Path("/tmp/superglm-cred-maint-baseline-b5a0cd0-sz/summary.json"),
        Path("/tmp/superglm-cred-maint-final-sz/summary.json"),
    ),
}
for basis, (before_path, after_path) in pairs.items():
    before = json.loads(before_path.read_text())["backends"]["structured"]
    after = json.loads(after_path.read_text())["backends"]["structured"]
    old = before["model"]
    new = after["model"]
    assert old["termination_reason"] == new["termination_reason"]
    assert old["reml_iterations"] == new["reml_iterations"]
    assert abs(old["objective"] - new["objective"]) < 1e-6
    assert abs(old["deviance"] - new["deviance"]) < 1e-6
    assert abs(old["effective_df"] - new["effective_df"]) < 1e-6
    assert abs(old["prediction_checksum"] - new["prediction_checksum"]) < 1e-5
    assert old["lambdas"].keys() == new["lambdas"].keys()
    assert max(abs(old["lambdas"][key] / new["lambdas"][key] - 1.0) for key in old["lambdas"]) < 1e-9
    old_mean = statistics.mean(before["wall_times_s"])
    new_mean = statistics.mean(after["wall_times_s"])
    assert new_mean <= old_mean * 1.05, (basis, old_mean, new_mean)
    print(basis, old_mean, new_mean, 100.0 * (new_mean / old_mean - 1.0))
PY
```

Expected: both bases pass numerical parity and the mean wall-time gate.

- [ ] **Step 4: Run focused tests, full tests, and lint**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest tests/ -q
rtk ruff check src/ tests/
rtk ruff format --check src/ tests/
rtk git diff --check
```

Expected: complete suite passes, Ruff is clean, formatting is stable, and no whitespace errors
exist.

- [ ] **Step 5: Confirm scope and clean worktree**

Run:

```bash
rtk git status --short
rtk git log --oneline b5a0cd0..HEAD
rtk git diff --stat b5a0cd0..HEAD
rtk git diff --name-only b5a0cd0..HEAD | rtk grep -i 'lss'
```

Expected: only planned credibility/REML code, tests, and workflow documents changed; the LSS
scan prints nothing; the worktree is clean.

### Task 6: Prepare final review handoff

**Files:**

- Read: `docs/superpowers/specs/2026-07-25-credibility-maintainability-pass-design.md`
- Read: `docs/superpowers/plans/2026-07-25-credibility-maintainability-pass.md`

- [ ] **Step 1: Summarize the bounded shipping changes**

Report the removed dead code, extracted helpers, reduced duplicated logic, and unchanged
public/numerical contracts. Do not present the architecture exploration as implemented.

- [ ] **Step 2: Summarize verification evidence**

Report focused/full test counts, Ruff/format results, exact numerical deltas, REML iterations,
and benchmark deltas.

- [ ] **Step 3: Hand off for PR review**

Keep the branch and worktree. Do not push, open a PR, merge, or clean up until the user selects
that integration action explicitly.
