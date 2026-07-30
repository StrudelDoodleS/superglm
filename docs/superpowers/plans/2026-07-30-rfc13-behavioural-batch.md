# RFC-13 Behavioural Batch Implementation Plan

> **SUPERSEDED WHERE IT CONFLICTS WITH THE CODE.** This plan was written before
> implementation and is kept as the record of intent, not as guidance. Task 4
> Step 4 below specifies `_project_feasible -> (beta, feasible)` with
> `QPResult.converged` latched from the projected *starting* point. That was
> implemented, measured at 27/100 spurious `converged=False`, and replaced by
> deriving `converged` from the *returned* point; the feasibility test also
> became relative. Other steps were refined across review rounds. The spec
> (`docs/superpowers/specs/2026-07-30-rfc13-behavioural-batch-design.md`) and
> the shipped code are the authority — do not re-apply the code blocks here.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four RFC-13 correctness-hygiene fixes — `max_iter=0` validation, the polarization merit delta in pirls, rank-policy-routed QP solves with a meaningful `converged` flag, and a warning when the REML W(ρ) correction is silently dropped.

**Architecture:** Four independent changes plus one shared refactor. The merit-delta helper moves from `irls_direct.py` to `irls_state.py` (its shared home) and gains two optional penalty terms so both IRLS orchestrations can use it. Nothing else is shared between items; each lands as its own commit.

**Tech Stack:** Python 3.11+, numpy, scipy, pytest, `uv` for running everything.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-rfc13-behavioural-batch-design.md`. Read it before starting.
- Base branch: `fix/rfc13-behavioural-batch`, branched from `master` at `e8e31f4` (= v0.16.1).
- **Release impact: `patch` → `0.16.2`.** Declared in the PR body, bumped in Task 6 of this same PR.
- Every command runs under `uv`: `uv run pytest ...`, never bare `pytest`.
- **Benchmark/BLAS canon:** if you run any timing, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No task here requires timing.
- **Gitignore trap:** `docs/superpowers/` and `benchmarks/results/*` are ignored but tracked by convention. Any new file there needs `git add -f` followed by a `git log --stat -1` check that it actually landed.
- Protected semantics (CLAUDE.md) that these changes must not disturb: `fit` vs `fit_reml` stay distinct; `discrete=True` never silently drifts from exact; `select=` vs `selection_penalty` stay distinct; the k/k−1 contract; `sample_weight = exposure`.
- `max_iter=1` must remain legal — `reml/discrete.py:551-578` depends on it.
- Run the full suite before opening the PR. Baseline on master at `e8e31f4` is **4818 passed / 152 skipped**.

---

### Task 1: `max_iter` validation in both solver entry points

**Files:**
- Modify: `src/superglm/solvers/irls_direct.py` (insert after the `fit_irls_direct` docstring, currently line 420, before `def run_once`)
- Modify: `src/superglm/solvers/pirls.py` (insert at the top of the `fit_pirls` body, currently line 1463, before `if isinstance(X, DesignMatrix):`)
- Create: `tests/test_solver_max_iter_validation.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks rely on.

**Background:** `_fit_pirls_inner` (`pirls.py:843`) and `_fit_irls_direct_once` (`irls_direct.py:1285`) both read loop-body locals after `for ... in range(max_iter)`. With `max_iter=0` the loop never runs and the post-loop reads raise `UnboundLocalError`. Guard the *public* entry points only — the private `_fit_*` helpers are only reachable through them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solver_max_iter_validation.py`:

```python
"""max_iter must be validated, not crash with UnboundLocalError (audit S13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM


def _tiny_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.3 * df["x0"].to_numpy())).astype(float)
    return df, y


@pytest.mark.parametrize("bad_max_iter", [0, -1])
def test_fit_rejects_non_positive_max_iter(bad_max_iter: int) -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", max_iter=bad_max_iter)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit(df, y)


def test_fit_reml_rejects_zero_max_iter() -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit_reml(df, y)


def test_selection_path_rejects_zero_max_iter() -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", selection_penalty=0.1)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit(df, y, max_iter=0)


def test_fit_pirls_rejects_non_positive_inner_and_outer() -> None:
    from superglm.solvers import fit_pirls

    # Arguments are never reached: validation runs before any array handling.
    with pytest.raises(ValueError, match="max_iter_outer must be at least 1"):
        fit_pirls(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            penalty=None,
            max_iter_outer=0,
        )
    with pytest.raises(ValueError, match="max_iter_inner must be at least 1"):
        fit_pirls(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            penalty=None,
            max_iter_inner=0,
        )


def test_fit_irls_direct_rejects_zero_max_iter() -> None:
    from superglm.solvers import fit_irls_direct

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        fit_irls_direct(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            lambda2=0.0,
            max_iter=0,
        )


def test_max_iter_one_remains_legal() -> None:
    """The discrete POI loop depends on max_iter=1 (reml/discrete.py:551-578)."""
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", max_iter=1)
    model.fit(df, y)
    assert np.all(np.isfinite(model.coefficients_))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_solver_max_iter_validation.py -v`

Expected: the six validation tests FAIL with `UnboundLocalError` (or `TypeError`/`AttributeError` for the direct-solver calls, which reach array handling before the guard exists). `test_max_iter_one_remains_legal` should already PASS.

- [ ] **Step 3: Add the guard to `fit_irls_direct`**

In `src/superglm/solvers/irls_direct.py`, immediately after the `fit_irls_direct` docstring line `"""Fit by direct IRLS, retrying automatic globally-ineligible SZ fits on Gram."""` and before `def run_once(`:

```python
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
```

- [ ] **Step 4: Add the guard to `fit_pirls`**

In `src/superglm/solvers/pirls.py`, as the first statements of the `fit_pirls` body (after its docstring, before `if isinstance(X, DesignMatrix):`):

```python
    if max_iter_outer < 1:
        raise ValueError(f"max_iter_outer must be at least 1, got {max_iter_outer}")
    if max_iter_inner < 1:
        raise ValueError(f"max_iter_inner must be at least 1, got {max_iter_inner}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_solver_max_iter_validation.py -v`
Expected: all 7 PASS.

- [ ] **Step 6: Run the neighbouring suites for regressions**

Run: `uv run pytest tests/test_irls_direct.py tests/test_pirls_composite_optimizer.py tests/test_reml.py -q`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add tests/test_solver_max_iter_validation.py src/superglm/solvers/irls_direct.py src/superglm/solvers/pirls.py
git commit -m "Validate max_iter at the solver entry points (RFC-13, audit S13)

max_iter=0 reached the post-loop reads of loop-body locals and raised
UnboundLocalError through fit, fit_reml, and the selection_penalty
path.  Guard the two public entry points with < 1 (range(-5) fails
identically); max_iter=1 stays legal for the discrete POI loop."
```

---

### Task 2: Move and generalize `_stable_penalized_deviance_delta` (pure refactor)

**Files:**
- Modify: `src/superglm/solvers/irls_direct.py` — delete the function at lines 117-152; drop the now-unused `math` import only if nothing else in the file uses it (check with `grep -n "math\." src/superglm/solvers/irls_direct.py`)
- Modify: `src/superglm/solvers/irls_state.py` — add the function below the `MeritDelta` alias at line 119
- Modify: `tests/test_irls_state.py:14` — change the import source
- Test: `tests/test_irls_state.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `superglm.solvers.irls_state._stable_penalized_deviance_delta(candidate, committed, penalty_matvec=None, nonsmooth_penalty=None) -> float`. `candidate`/`committed` are `_IRLSState`; `penalty_matvec` is `Callable[[NDArray], NDArray] | NDArray | None`; `nonsmooth_penalty` is `Callable[[NDArray], float] | None`. Task 3 consumes it.

**Background:** pirls already imports from `irls_state`; importing this helper from `irls_direct` would invert the dependency direction. `irls_state.py` is where the `MeritDelta` alias it satisfies already lives (line 119). The symbol is private, so no compatibility shim is needed. **This task must not change any numeric result** — it exists so a reviewer can gate the move separately from the behaviour change in Task 3.

- [ ] **Step 1: Write the failing tests for the new optional arguments**

Append to `tests/test_irls_state.py`:

```python
def test_stable_delta_omits_quadratic_when_penalty_matvec_is_none() -> None:
    """With no quadratic penalty the delta is a plain deviance difference."""
    committed = _synthetic_state(0.0, penalized_deviance=40.0)
    proposal = _synthetic_state(0.0, penalized_deviance=17.0)
    delta = _stable_penalized_deviance_delta(proposal, committed)
    assert delta == pytest.approx(proposal.deviance - committed.deviance, abs=1e-15)


def test_stable_delta_includes_nonsmooth_penalty_term() -> None:
    """A non-quadratic penalty enters as two pre-scaled fsum terms."""
    committed = _synthetic_state(0.0, penalized_deviance=40.0)
    proposal = _synthetic_state(0.0, penalized_deviance=17.0)

    def nonsmooth(beta: np.ndarray) -> float:
        return 2.0 * float(np.abs(beta).sum())

    delta = _stable_penalized_deviance_delta(
        proposal, committed, nonsmooth_penalty=nonsmooth
    )
    expected = (
        proposal.deviance
        - committed.deviance
        + nonsmooth(proposal.beta)
        - nonsmooth(committed.beta)
    )
    assert delta == pytest.approx(expected, abs=1e-15)


def test_stable_delta_combines_quadratic_and_nonsmooth_terms() -> None:
    """Both penalty terms compose in a single fsum."""
    penalty = np.array([[2.0, 0.0], [0.0, 3.0]])
    committed_beta = np.array([0.25, -0.5])
    proposal_beta = np.array([0.5, -0.25])

    def state(beta: np.ndarray, deviance: float) -> _IRLSState:
        eta = _immutable_array(np.zeros(1))
        return _IRLSState(
            beta=_immutable_array(beta),
            intercept=0.0,
            eta_unclipped=eta,
            eta=eta,
            mu=eta,
            deviance=deviance,
            penalized_deviance=float(deviance + beta @ penalty @ beta),
        )

    def nonsmooth(beta: np.ndarray) -> float:
        return 2.0 * float(np.abs(beta).sum())

    committed = state(committed_beta, 10.0)
    proposal = state(proposal_beta, 9.0)
    delta = _stable_penalized_deviance_delta(
        proposal, committed, penalty, nonsmooth_penalty=nonsmooth
    )
    expected = (
        9.0
        - 10.0
        + proposal_beta @ penalty @ proposal_beta
        - committed_beta @ penalty @ committed_beta
        + nonsmooth(proposal_beta)
        - nonsmooth(committed_beta)
    )
    assert delta == pytest.approx(expected, rel=1e-12)
```

- [ ] **Step 2: Change the import in the test file**

In `tests/test_irls_state.py`, line 14 currently reads:

```python
from superglm.solvers.irls_direct import _stable_penalized_deviance_delta, fit_irls_direct
```

Split it into:

```python
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import _stable_penalized_deviance_delta
```

Keep whatever other names the file already imports from `irls_state` on the existing `irls_state` import line if one exists — merge, don't duplicate.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_irls_state.py -v`
Expected: FAIL at import time with `ImportError: cannot import name '_stable_penalized_deviance_delta' from 'superglm.solvers.irls_state'`.

- [ ] **Step 4: Add the generalized function to `irls_state.py`**

In `src/superglm/solvers/irls_state.py`, after the `MeritDelta` alias (line 119). Add `import math` to the module's imports if absent.

```python
def _stable_penalized_deviance_delta(
    candidate: _IRLSState,
    committed: _IRLSState,
    penalty_matvec: Callable[[NDArray], NDArray] | NDArray | None = None,
    nonsmooth_penalty: Callable[[NDArray], float] | None = None,
) -> float:
    """Compare penalized deviances without subtracting two large quadratics.

    In an ill-conditioned smooth basis, the two penalty quadratics can each be
    accurately evaluated while their tiny difference loses enough digits to
    reverse the sign of an otherwise safe terminal step.  The polarization
    identity evaluates that difference directly from the coefficient update.

    ``penalty_matvec`` supplies the quadratic penalty ``S`` (a matrix or a
    matvec); pass ``None`` when the fit carries no quadratic penalty.
    ``nonsmooth_penalty`` supplies any non-quadratic penalty term as a
    function of ``beta``, already scaled to match the caller's merit
    convention; its two evaluations enter the same ``math.fsum``.
    """
    terms = [float(candidate.deviance), -float(committed.deviance)]

    if penalty_matvec is not None:
        delta_beta = candidate.beta - committed.beta
        summed_beta = candidate.beta + committed.beta
        penalty_direction = (
            penalty_matvec(summed_beta)
            if callable(penalty_matvec)
            else np.asarray(penalty_matvec, dtype=np.float64) @ summed_beta
        )
        terms.append(
            math.fsum(
                float(delta_value * direction_value)
                for delta_value, direction_value in zip(
                    delta_beta,
                    penalty_direction,
                    strict=True,
                )
            )
        )

    if nonsmooth_penalty is not None:
        terms.append(float(nonsmooth_penalty(candidate.beta)))
        terms.append(-float(nonsmooth_penalty(committed.beta)))

    return float(math.fsum(terms))
```

- [ ] **Step 5: Delete the original and re-point `irls_direct`**

Delete lines 117-152 of `src/superglm/solvers/irls_direct.py` (the whole `_stable_penalized_deviance_delta` definition, up to but not including the `@dataclass(frozen=True)` decorator for `_SCOPGroupSpec`).

Add `_stable_penalized_deviance_delta` to the existing `from superglm.solvers.irls_state import ...` line in `irls_direct.py`. The call site at line 1815 stays exactly as it is — the third positional argument still binds to `penalty_matvec`.

Check whether `math` is still used: `grep -n "math\." src/superglm/solvers/irls_direct.py`. If there are no hits, remove the `import math` line.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_irls_state.py tests/test_irls_direct.py -v`
Expected: all PASS, including the pre-existing `test_stable_composite_merit_accepts_ill_conditioned_penalty_plateau`.

- [ ] **Step 7: Confirm no numeric drift**

Run: `uv run pytest tests/test_reml.py tests/test_reml_convergence.py tests/test_scop_irls_state.py -q`
Expected: no failures. This is a pure move; any change here means the refactor was not neutral.

- [ ] **Step 8: Commit**

```bash
git add src/superglm/solvers/irls_state.py src/superglm/solvers/irls_direct.py tests/test_irls_state.py
git commit -m "Move the stable merit delta to irls_state and make its penalty terms optional

The helper satisfies the MeritDelta alias that already lives in
irls_state; pirls imports from irls_state, so sourcing it from
irls_direct would invert the dependency direction.  Both penalty
contributions become optional so either IRLS orchestration can use it.

Pure refactor: irls_direct's call site and numeric results are
unchanged."
```

---

### Task 3: Wire the polarization merit delta into pirls

**Files:**
- Modify: `src/superglm/solvers/pirls.py:1043-1048` (the `_select_irls_trial` call)
- Test: `tests/test_irls_state.py`, `tests/test_pirls_composite_optimizer.py`

**Interfaces:**
- Consumes: `_stable_penalized_deviance_delta(candidate, committed, penalty_matvec=None, nonsmooth_penalty=None)` from Task 2.
- Produces: nothing later tasks rely on.

**Background:** pirls's merit is `deviance + βʹSβ + 2·penalty.eval(β, groups)` (`pirls.py:756`), but its line search falls through to the raw comparison at `irls_state.py:171`. The audit demonstrated a −1.5e-7 improvement being read as +1.7e-5, i.e. a safe terminal step rejected (finding S3).

Two facts that make the wiring safe:
1. After `pirls.py:695-697`, `S is not None` is exactly equivalent to `has_smooth_penalty`, so passing `S` directly gives the right skip behaviour with no extra condition.
2. The delta must be the difference of exactly what `_state_merit` returns, because `_irls_trial_is_unsafe` compares `delta > roundoff` where `roundoff` is scaled from `_state_merit` magnitudes. The three term groups reproduce `pirls.py:756` exactly.

**Expect this task to change fitted coefficients** for pirls fits in ill-conditioned smooth bases. That is the point of the change and the reason the batch is `release:patch`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_irls_state.py`:

```python
def test_pirls_merit_convention_matches_stable_delta() -> None:
    """The delta must reproduce pirls's merit: D + beta'S beta + 2 * penalty.eval."""
    penalty_matrix = np.array(
        [
            [500000.49999999994, 499999.49999999994],
            [499999.49999999994, 500000.49999999994],
        ]
    )
    beta_committed = np.array([0.7071067882576153, -0.7071067741154796])
    beta_proposal = np.array([0.707106788257686, -0.7071067741154089])
    deviance_committed = 31.358012850845732
    deviance_proposal = 31.35801285084573

    def selection(beta: np.ndarray) -> float:
        return 2.0 * 0.05 * float(np.linalg.norm(beta))

    def state(beta: np.ndarray, deviance: float) -> _IRLSState:
        eta = _immutable_array(np.zeros(1))
        return _IRLSState(
            beta=_immutable_array(beta),
            intercept=0.0,
            eta_unclipped=eta,
            eta=eta,
            mu=eta,
            deviance=deviance,
            penalized_deviance=float(
                deviance + beta @ penalty_matrix @ beta + selection(beta)
            ),
        )

    committed = state(beta_committed, deviance_committed)
    proposal = state(beta_proposal, deviance_proposal)

    # The raw comparison rejects this safe terminal step.
    assert _irls_trial_is_unsafe(proposal, committed)

    delta = _stable_penalized_deviance_delta(
        proposal, committed, penalty_matrix, nonsmooth_penalty=selection
    )
    assert abs(delta) < 1.0e-13

    decision = _select_irls_trial(
        committed=committed,
        proposal=proposal,
        evaluate_state=lambda alpha: pytest.fail(f"unexpected trial at {alpha}"),
        merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
            candidate, base, penalty_matrix, nonsmooth_penalty=selection
        ),
    )
    assert decision == _IRLSStepDecision(1.0, 0, False, trials_attempted=1)
```

Append to `tests/test_pirls_composite_optimizer.py` (adjust the imports at the top of that file to include what this needs):

```python
def test_pirls_ill_conditioned_smooth_basis_has_no_spurious_step_rejection() -> None:
    """Audit S3's open 'Verify:' — pirls with a large S must not reject terminal steps."""
    import numpy as np
    import pandas as pd

    from superglm import PSpline, SuperGLM

    rng = np.random.default_rng(11)
    n = 600
    x = np.sort(rng.uniform(0.0, 1.0, n))
    mu = np.exp(0.5 + np.sin(4.0 * np.pi * x))
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n)})

    model = SuperGLM(
        features={"x": PSpline(n_knots=25)},
        family="poisson",
        selection_penalty=0.01,
    )
    model.fit(df, y, lambda2={"x": 1.0e6})

    diagnostics = model._result.diagnostics
    assert diagnostics is not None
    assert not any(d.step_rejected for d in diagnostics), (
        "pirls rejected a terminal step in an ill-conditioned smooth basis"
    )
```

If `IterationDiagnostics` exposes the rejection under a different attribute name, read `src/superglm/solvers/pirls.py:1126-1180` and use the real one rather than inventing one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_irls_state.py::test_pirls_merit_convention_matches_stable_delta tests/test_pirls_composite_optimizer.py -k "spurious or merit_convention" -v`
Expected: the merit-convention test PASSES already (it only exercises Task 2's helper); the pirls end-to-end test FAILS with at least one `step_rejected`.

If the end-to-end test passes before the change, the chosen `lambda2` is not ill-conditioned enough — raise it (try `1.0e8`) until it fails, and keep the value that reproduces.

- [ ] **Step 3: Wire the merit delta into pirls**

In `src/superglm/solvers/pirls.py`, replace the `_select_irls_trial` call at lines 1043-1048:

```python
        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            max_halving=max_halving,
        )
```

with:

```python
        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            max_halving=max_halving,
            merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                candidate,
                base,
                S,
                nonsmooth_penalty=lambda values: 2.0 * float(penalty.eval(values, groups)),
            ),
        )
```

`S` is `None` whenever `has_smooth_penalty` is false (`pirls.py:695-697`), which is exactly when the quadratic term should be skipped. The `2.0` factor lives here, not in the helper, because it is pirls's merit convention.

Add `_stable_penalized_deviance_delta` to the existing `from superglm.solvers.irls_state import ...` line at `pirls.py:39`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_irls_state.py tests/test_pirls_composite_optimizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Check for fitted-value drift across the selection suites**

Run: `uv run pytest tests/ -q -k "pirls or selection or group_lasso or elastic_net or adaptive"`
Expected: PASS. If a test fails on a *numeric tolerance* rather than a structural assertion, that is the expected behaviour change — inspect the delta, confirm it is the line search now accepting a better step, and update the expected value with a comment naming this task. If a test fails structurally (wrong group selected, non-convergence), stop and investigate.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/solvers/pirls.py tests/test_irls_state.py tests/test_pirls_composite_optimizer.py
git commit -m "Use the polarization merit delta in the pirls line search (RFC-13, audit S3)

pirls compared raw penalized deviances, so quadratic cancellation in an
ill-conditioned smooth basis could reverse the sign of a safe terminal
step -- the audit measured a -1.5e-7 improvement read as +1.7e-5.  Pass
the shared stable delta with S as the quadratic and 2 * penalty.eval as
the non-quadratic term, matching pirls's merit convention exactly.

Answers audit S3's open 'Verify:' note.  May move fitted coefficients
for pirls fits with ill-conditioned smooth bases."
```

---

### Task 4: Route pure-H QP solves through the rank policy and surface `converged`

**Files:**
- Modify: `src/superglm/solvers/constrained_qp.py` (lines 38-54 `_project_feasible`, 91-117 the three pure-H solves, 199 the exhaustion return)
- Modify: `src/superglm/solvers/irls_direct.py:1625-1634` (check the flag)
- Modify: `src/superglm/solvers/scop.py:171` and `:286` (check the flag; add a module logger)
- Test: `tests/test_constrained_qp.py`

**Interfaces:**
- Consumes: `superglm.solvers.rank.decompose_gram(matrix) -> RankDecomposition` with `.solve(rhs) -> NDArray`.
- Produces: `_project_feasible(beta, A, b) -> tuple[NDArray, bool]` (internal to `constrained_qp.py`). `QPResult`'s public shape is unchanged.

**Background:** audit S16. `constrained_qp.py:96,100,117` are the only unguarded dense `np.linalg.solve` calls in the solver subsystem; every other consumer routes through `rank.py`. `QPResult.converged` already exists (line 35) and is already set `False` on `max_iter` exhaustion (line 199), but all three call sites discard it.

Two verified facts:
1. `RankDecomposition.solve` divides by `column_scale` on both the RHS and the solution (`rank.py:168,173`), so it returns in **original coordinates**. The `column_scale` trap recorded in the RFC-12b disposition note does not apply here.
2. `rank.py` imports only numpy and scipy, so importing it from `constrained_qp.py` creates no cycle.

`H` is `XtWX + S` at `irls_direct.py:1622` (PSD) and `XʹX + λP + 1e-8·I` at `scop.py:162-163,279-280` (PD), so `decompose_gram`'s default `allow_indefinite=False` is correct. **Leave the indefinite KKT solve at line 136 alone** — it has its own `lstsq` fallback and is out of scope.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_constrained_qp.py`:

```python
def test_singular_h_returns_finite_solution_instead_of_raising() -> None:
    """Rank-deficient H must go through the rank policy, not raise LinAlgError."""
    H = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    g = np.array([1.0, 1.0])
    A = np.array([[1.0, 0.0]])
    b = np.array([-10.0])

    result = solve_constrained_qp(H, g, A, b)

    assert np.all(np.isfinite(result.beta))
    assert result.converged


def test_well_conditioned_solution_is_unchanged_by_the_rank_policy() -> None:
    """The rank policy must not perturb a well-conditioned unconstrained solve."""
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    g = np.array([1.0, 2.0])
    A = np.array([[1.0, 0.0]])
    b = np.array([-10.0])  # inactive

    result = solve_constrained_qp(H, g, A, b)

    np.testing.assert_allclose(result.beta, np.linalg.solve(H, g), rtol=1e-12)
    assert result.converged


def test_iteration_starved_qp_reports_non_convergence() -> None:
    H = np.eye(3)
    g = np.array([5.0, 5.0, 5.0])
    A = -np.eye(3)
    b = np.array([-0.1, -0.1, -0.1])

    result = solve_constrained_qp(H, g, A, b, max_iter=1)

    assert not result.converged


def test_infeasible_projection_reports_non_convergence() -> None:
    """Mutually contradictory constraints cannot be projected onto."""
    H = np.eye(1)
    g = np.array([0.0])
    A = np.array([[1.0], [-1.0]])
    b = np.array([1.0, 1.0])  # x >= 1 and -x >= 1

    result = solve_constrained_qp(H, g, A, b)

    assert not result.converged
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_constrained_qp.py -v`
Expected: `test_singular_h_...` FAILS with `numpy.linalg.LinAlgError: Singular matrix`; `test_infeasible_projection_...` FAILS because `converged` is still `True`. The other two should already PASS.

- [ ] **Step 3: Route the pure-H solves through the rank policy**

In `src/superglm/solvers/constrained_qp.py`, add to the imports:

```python
from superglm.solvers.rank import decompose_gram
```

Inside `solve_constrained_qp`, after `m = A.shape[0]`, decompose once:

```python
    decomposition = decompose_gram(H)
```

Replace line 96 `beta = np.linalg.solve(H, g)` with `beta = decomposition.solve(g)`.
Replace line 100 `beta_unc = np.linalg.solve(H, g)` with `beta_unc = decomposition.solve(g)`.
Replace line 117 `step = np.linalg.solve(H, g) - beta` with `step = beta_unc - beta` — `beta_unc` is the same quantity and is already computed above the loop, so this also removes a redundant O(p³) per active-set iteration.

- [ ] **Step 4: Report projection failure**

Replace `_project_feasible` (lines 38-54) with:

```python
def _project_feasible(beta: NDArray, A: NDArray, b: NDArray) -> tuple[NDArray, bool]:
    """Project beta onto the feasible set {x : A @ x >= b}.

    Uses iterative constraint-by-constraint projection (Dykstra-like).
    For the small dense problems we handle, this converges quickly.

    Returns the projected point and whether the sweeps reached feasibility;
    a mutually infeasible constraint set exhausts the sweep budget and
    reports ``False``.
    """
    beta = beta.copy()
    for _ in range(100):
        violations = A @ beta - b
        worst = np.argmin(violations)
        if violations[worst] >= -1e-12:
            return beta, True
        # Project onto the violated constraint: a^T x >= b_i
        a = A[worst]
        deficit = b[worst] - a @ beta
        beta += deficit / (a @ a) * a
    return beta, False
```

Update the caller at line 111:

```python
    beta, projection_feasible = _project_feasible(beta_unc, A, b)
```

Then thread `projection_feasible` into every `QPResult` constructed after that point — the two returns inside the loop (lines 147 and 163) and the exhaustion return (line 199):

```python
                return QPResult(
                    beta=beta,
                    active_set=active,
                    n_iter=it + 1,
                    converged=projection_feasible,
                )
```

and

```python
    return QPResult(beta=beta, active_set=active, n_iter=max_iter, converged=False)
```

The two early returns before the projection (lines 97 and 102) keep the default `converged=True`.

- [ ] **Step 5: Run the QP tests to verify they pass**

Run: `uv run pytest tests/test_constrained_qp.py -v`
Expected: all PASS.

- [ ] **Step 6: Surface the flag at the three call sites**

In `src/superglm/solvers/irls_direct.py`, after `prev_active_set = qp_result.active_set` (line 1634):

```python
                if not qp_result.converged:
                    logger.warning(
                        "fit_irls_direct: constrained QP did not converge at "
                        "iteration %d; monotone constraints may be only "
                        "approximately satisfied.",
                        it + 1,
                    )
```

In `src/superglm/solvers/scop.py`, add a module logger after the imports:

```python
import logging

logger = logging.getLogger(__name__)
```

After `result = solve_constrained_qp(H, g, A, b)` in `qp_initialize` (line 171) and in the solver-space `qp_initialize` (line 286), add:

```python
        if not result.converged:
            logger.warning(
                "SCOP QP initialization did not converge; falling back to an "
                "approximate shape-constrained starting point."
            )
```

- [ ] **Step 7: Test the call-site warnings**

Append to `tests/test_constrained_qp.py`:

```python
def test_scop_qp_initialize_warns_on_non_convergence(caplog, monkeypatch) -> None:
    import logging

    from superglm.solvers import constrained_qp, scop
    from superglm.solvers.constrained_qp import QPResult

    reparam = scop.build_scop_reparam(6, kind="increasing")

    def fake_solve(H, g, A, b, **kwargs):
        return QPResult(beta=np.ones(H.shape[0]), active_set=[], converged=False)

    monkeypatch.setattr(constrained_qp, "solve_constrained_qp", fake_solve)
    monkeypatch.setattr(scop, "solve_constrained_qp", fake_solve, raising=False)

    rng = np.random.default_rng(3)
    B = rng.normal(size=(40, 6))
    y = rng.normal(size=40)

    with caplog.at_level(logging.WARNING, logger="superglm.solvers.scop"):
        reparam.qp_initialize(B, y)

    assert "did not converge" in caplog.text
```

`qp_initialize` imports `solve_constrained_qp` inside the function body (`scop.py:151,266`), so the `monkeypatch.setattr` on the `constrained_qp` module is the one that takes effect; the `scop` one is a harmless belt-and-braces. If the local import defeats both, change `scop.py` to import the symbol at module scope and patch that instead — a module-scope import is the better shape anyway now that `scop.py` gains a logger.

Run: `uv run pytest tests/test_constrained_qp.py -v`
Expected: all PASS.

- [ ] **Step 8: Run the shape-constrained suites for regressions**

Run: `uv run pytest tests/ -q -k "scop or shape or monotone or constrained"`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
git add src/superglm/solvers/constrained_qp.py src/superglm/solvers/irls_direct.py src/superglm/solvers/scop.py tests/test_constrained_qp.py
git commit -m "Route pure-H QP solves through the rank policy; surface QPResult.converged (RFC-13, audit S16)

The three np.linalg.solve(H, g) calls were the only unguarded dense
solves in the solver subsystem; singular H raised LinAlgError instead
of being rank-truncated like everywhere else.  Decompose once and reuse,
which also drops a redundant O(p^3) per active-set iteration.

_project_feasible now reports whether its 100 sweeps reached
feasibility, so converged means what its name says, and all three call
sites log a warning instead of discarding the flag.

The indefinite KKT solve keeps its existing lstsq fallback."
```

---

### Task 5: Warn when the W(ρ) correction is dropped for want of `deriv2_inverse`

**Files:**
- Modify: `src/superglm/reml/w_derivatives.py` (add a helper; call it at the `dW_deta is None` branch, line 267)
- Create: `tests/test_w_correction_capability_warning.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks rely on.

**Background:** `compute_dW_deta` returns `None` when the link lacks `deriv2_inverse` or the distribution lacks `variance_derivative` (`w_derivatives.py:72`); `reml_w_correction` then returns `None` at line 267 and the REML gradient and Hessian silently lose the weight-derivative term. All eleven built-in links implement `deriv2_inverse`, so only user-supplied custom links reach this. The *observed* path already fails loudly for the same gap via `validate_observed_derivative_capability` (`observed_geometry.py:452-489`); the Fisher path has nothing.

Three placement constraints:
1. **Warn from `reml_w_correction`, not from `compute_dW_deta`.** `_compute_d2W_deta2_fd` calls `compute_dW_deta` three times internally (lines 158, 164, 168); warning inside it would fire from the finite-difference fallback.
2. **The structural-zero branch at line 270 (`not np.any(dW_deta)`) must stay silent.** That is Gamma/log, where the correction is genuinely zero rather than unavailable, and `tests/test_reml_fd.py:534-543` already asserts it returns `None`.
3. Per-iteration spam is handled by the stdlib default warning filter, which dedups on `(message, category, module, lineno)`. Including the class names in the message makes it one warning per unique class pair per process. `pytest.warns` still sees it because pytest resets filters inside its context.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_w_correction_capability_warning.py`:

```python
"""The Fisher-path W(rho) drop must warn, not vanish silently (RFC-13, audit J.4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import CubicRegressionSpline, SuperGLM
from superglm.reml import build_penalty_caches
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.irls_direct import fit_irls_direct


class _LinkWithoutDeriv2:
    """Delegates to a real link but hides deriv2_inverse."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in {"deriv2_inverse", "deriv3_inverse"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


class _DistributionWithoutVarianceDerivative:
    """Delegates to a real distribution but hides variance_derivative."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in {"variance_derivative", "variance_second_derivative"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def poisson_setup():
    """A fitted Poisson/log spline model plus the pieces reml_w_correction needs."""
    rng = np.random.default_rng(42)
    n = 400
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2)
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    model = SuperGLM(
        features={
            "x1": CubicRegressionSpline(n_knots=8),
            "x2": CubicRegressionSpline(n_knots=8),
        },
        family="poisson",
    )
    model.fit(df, y)

    sample_weight = np.ones(n)
    offset_arr = np.zeros(n)
    lambdas = {"x1": 10.0, "x2": 0.5}

    reml_groups = [
        (i, g)
        for i, (gm, g) in enumerate(zip(model._dm.group_matrices, model._groups))
        if g.penalized
    ]
    penalty_caches = build_penalty_caches(model._dm.group_matrices, reml_groups)

    pirls_result, XtWX_S_inv, _ = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset_arr,
        return_xtwx=True,
    )

    return {
        "dm": model._dm,
        "link": model._link,
        "distribution": model._distribution,
        "groups": model._groups,
        "pirls_result": pirls_result,
        "XtWX_S_inv": XtWX_S_inv,
        "lambdas": lambdas,
        "reml_groups": reml_groups,
        "penalty_caches": penalty_caches,
        "sample_weight": sample_weight,
        "offset_arr": offset_arr,
    }


def _call(setup, *, link=None, distribution=None):
    return reml_w_correction(
        dm=setup["dm"],
        link=link if link is not None else setup["link"],
        groups=setup["groups"],
        pirls_result=setup["pirls_result"],
        XtWX_S_inv=setup["XtWX_S_inv"],
        lambdas=setup["lambdas"],
        reml_groups=setup["reml_groups"],
        penalty_caches=setup["penalty_caches"],
        sample_weight=setup["sample_weight"],
        offset_arr=setup["offset_arr"],
        distribution=(
            distribution if distribution is not None else setup["distribution"]
        ),
    )


def test_link_without_deriv2_inverse_warns(poisson_setup) -> None:
    link = _LinkWithoutDeriv2(poisson_setup["link"])
    with pytest.warns(UserWarning, match="deriv2_inverse"):
        result = _call(poisson_setup, link=link)
    assert result is None


def test_distribution_without_variance_derivative_warns(poisson_setup) -> None:
    distribution = _DistributionWithoutVarianceDerivative(
        poisson_setup["distribution"]
    )
    with pytest.warns(UserWarning, match="variance_derivative"):
        result = _call(poisson_setup, distribution=distribution)
    assert result is None


def test_builtin_link_does_not_warn(poisson_setup) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _call(poisson_setup)


def test_gamma_log_structural_zero_does_not_warn() -> None:
    """Gamma/log has a genuinely zero correction; it must stay silent."""
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x1))
    y = np.maximum(rng.gamma(shape=5.0, scale=mu / 5.0), 1e-4)
    df = pd.DataFrame({"x1": x1})

    model = SuperGLM(
        features={"x1": CubicRegressionSpline(n_knots=8)}, family="gamma"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit_reml(df, y)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_w_correction_capability_warning.py -v`
Expected: the two `pytest.warns` tests FAIL with `DID NOT WARN`. The two negative tests should already PASS.

If the fixture errors while assembling `reml_w_correction`'s arguments, fix the fixture against the working pattern at `tests/test_reml_fd.py:23-100` (`_setup_model`) before touching source.

- [ ] **Step 3: Add the warning helper**

In `src/superglm/reml/w_derivatives.py`, add `import warnings` to the imports and this helper above `reml_w_correction`:

```python
def _warn_w_correction_unavailable(link: Any, distribution: Any) -> None:
    """Report the capability gap that silently drops the W(rho) correction.

    The message names the concrete classes so the stdlib default filter --
    which dedups on (message, category, module, lineno) -- emits one warning
    per unique link/distribution pair rather than one per REML iteration.
    """
    missing = []
    if not hasattr(link, "deriv2_inverse"):
        missing.append(f"{type(link).__name__}.deriv2_inverse")
    if not hasattr(distribution, "variance_derivative"):
        missing.append(f"{type(distribution).__name__}.variance_derivative")
    if not missing:
        return
    warnings.warn(
        f"REML W(rho) correction skipped: {' and '.join(missing)} "
        "is not implemented. Smoothing-parameter gradients omit the "
        "weight-derivative term, so REML may converge slowly or select "
        "slightly different smoothing parameters. Implement the missing "
        "method to restore the correction.",
        UserWarning,
        stacklevel=3,
    )
```

- [ ] **Step 4: Call it at the capability branch only**

In `reml_w_correction`, change lines 267-268 from:

```python
    if dW_deta is None:
        return None  # Custom link/distribution w/o 2nd-order
```

to:

```python
    if dW_deta is None:
        # Custom link/distribution w/o 2nd-order.  Distinct from the
        # structurally-zero branch below, which is silent by design.
        _warn_w_correction_unavailable(link, distribution)
        return None
```

Leave the `not np.any(dW_deta)` branch at line 270 exactly as it is.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_w_correction_capability_warning.py -v`
Expected: all 4 PASS.

- [ ] **Step 6: Run the REML suites for regressions**

Run: `uv run pytest tests/ -q -k "reml or w_correction"`
Expected: no new failures. In particular `tests/test_reml_fd.py::...::test_w_correction_zero` (the Gamma/log `assert result is None` at line 543) must still pass and must not have started warning.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/reml/w_derivatives.py tests/test_w_correction_capability_warning.py
git commit -m "Warn when the Fisher-path W(rho) correction is dropped (RFC-13, audit J.4)

A custom link without deriv2_inverse, or a distribution without
variance_derivative, silently cost the REML gradient and Hessian their
weight-derivative term.  The observed path already fails loudly for the
same gap via validate_observed_derivative_capability; the Fisher path
had nothing.

Warn from reml_w_correction rather than compute_dW_deta so the
finite-difference fallback in _compute_d2W_deta2_fd stays quiet, and
leave the structurally-zero Gamma/log branch silent.  Naming the
classes lets the stdlib default filter dedup to one warning per class
pair instead of one per REML iteration."
```

---

### Task 6: Bump to 0.16.2

**Files:**
- Modify: `pyproject.toml:3`, `src/superglm/__init__.py` (`__version__`), `uv.lock` — all via the script, never by hand
- Modify: the changelog, if `CHANGELOG.md` exists at the repo root

**Interfaces:**
- Consumes: Tasks 1-5 landed.
- Produces: the version the release manager will later tag.

**Background:** AGENTS.md and `.codex/agents/release_manager.toml` (authoritative) require that a patch-impact PR carry the exact next version in the same PR. Merging does **not** authorize a tag; that is a separate explicit instruction later.

- [ ] **Step 1: Apply the bump with the script**

Run: `uv run python scripts/bump_version.py 0.16.2 --impact patch`
Expected: it rewrites `pyproject.toml` and `__version__`. It refuses anything other than `0.16.2` for a patch from `0.16.1`.

- [ ] **Step 2: Refresh the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `superglm` version entry only.

- [ ] **Step 3: Add the changelog entry**

If `CHANGELOG.md` exists, add a `0.16.2` section following the format of the `0.16.1` entry above it, covering: `max_iter` validation; the pirls merit-delta port (**flagging that fitted coefficients can move for pirls fits with ill-conditioned smooth bases**); the QP rank-policy routing and `converged` surfacing (**flagging that singular `H` now returns a rank-truncated solve instead of raising**); and the new W(ρ) `UserWarning`.

- [ ] **Step 4: Verify the bump**

Run: `uv run python -c "import superglm; print(superglm.__version__)"`
Expected: `0.16.2`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/superglm/__init__.py uv.lock CHANGELOG.md
git commit -m "Bump to 0.16.2 for the RFC-13 behavioural batch"
```

---

### Task 7: Full-suite verification and PR

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: at least **4818 passed / 152 skipped** plus the new tests. Any failure blocks the PR — do not open it on a red tree.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: clean. If `ruff format` reports diffs in files you touched, run `uv run ruff format src tests` and amend.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fix/rfc13-behavioural-batch
```

Open the PR with a body following the #171/#172 conventions: what changed per item, the audit references (§E row 13, §J.4 item 6, S3/S13/S16), the two behaviour changes called out explicitly, and the release declaration:

```
release:patch (0.16.2)
```

with the rationale that item 2 can move fitted coefficients, item 3 changes singular-H QP from raising to solving, and item 4 adds a warning.

Note: `gh pr edit`/`gh pr merge` hit a Projects-classic GraphQL bug in this repo — use `gh api -X PATCH /repos/{owner}/{repo}/pulls/N` for body/title edits and `gh api -X PUT .../pulls/N/merge -f merge_method=rebase` to merge. Merge commits are blocked by ruleset; rebase-merge works.

- [ ] **Step 4: Request both reviews**

Comment `@claude please review` and `@codex please review` as two separate comments. Verify both comment URLs resolve after posting — `gh pr comment` with `--jq` fails silently.

Expect 3-4 rounds. **Verify every finding against the code before implementing it** — past rounds contained real catches alongside one wrong reachability claim that verification overturned. The claude bot occasionally dies on an infra error after ~3 minutes with no findings; rerun it or rely on its last completed verdict.

---

## Self-review notes

**Spec coverage.** Item 1 → Task 1. Item 2 → Tasks 2 (move/generalize) and 3 (wire into pirls); split so the pure refactor can be gated separately from the behaviour change. Item 3 → Task 4, including the `_project_feasible` extension the spec explicitly keeps. Item 4 → Task 5. Packaging → Tasks 6 and 7.

**Known plan risks, flagged rather than hidden.**
- Task 3 Step 1's `lambda2={"x": 1.0e6}` is a guess at an ill-conditioning threshold. The step says explicitly to raise it until the test fails first, rather than assuming it reproduces.
- Task 3 Step 1 assumes `IterationDiagnostics` exposes `step_rejected`; the step says to read `pirls.py:1126-1180` and use the real attribute name.
- Task 4 Step 7's monkeypatch has to defeat a function-local import (`scop.py:151,266`); the step gives the fallback of hoisting the import to module scope.
- Task 5's fixture assembles `reml_w_correction`'s arguments by hand; the step points at the working `tests/test_reml_fd.py:23-100` pattern if it needs repair.
