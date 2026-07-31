# Cold Certification-Retry Rung Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a final cold rung to the SCOP certification retry ladder, so a
candidate mode that fails certification gets one attempt from a fresh starting
point before the fit gives up.

**Architecture:** Both existing retry rungs tighten `pirls_tol` and warm-start
from the mode that just failed. When the inner fit has already converged tighter
than the bar, that reproduces the same mode bit-identically and cannot rescue it.
The change extends the ladder by one rung that keeps the tightest tolerance
already reached and drops the warm start — so it differs from its predecessor in
exactly one respect, the starting point. Six lines in one function, plus a test.

**Tech Stack:** Python, NumPy, pytest, `uv`, ruff.

**Spec:** `docs/superpowers/specs/2026-07-31-scop-cold-certification-retry-design.md`
**Issue:** #179

## Global Constraints

- Branch `fix/scop-cold-certification-retry` is checked out at `e21bf86`, rebased
  onto master `56c948e`, with the spec committed. Do not create another branch.
- **Version stays `0.17.0`. Do not bump.** `release:none`, folding into the
  existing unpublished candidate. **Do not tag, publish, or merge.**
- Do not change `_scop_mode_tolerance`, `_scop_mode_newton_relative`, or any of
  the four fatal call sites. The bar and what the solver accepts stay exactly as
  they are.
- Do not remove or reorder the two existing tolerance rungs — measured, they
  rescue 29 of 43 retries.
- `docs/superpowers/` is gitignored: `git add -f`, and confirm the file appears in
  `git log --stat`.
- Full suite (~3 min):
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q -p no:randomly`
- Baseline before this plan: **5111 passed / 84 skipped**.
- CI `quality` runs `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`.
- The project is now Python `>=3.12`. To run another interpreter or a pinned
  NumPy, always set `UV_PROJECT_ENVIRONMENT` to a scratch dir — never bare
  `uv run --python ...`, which repoints the project venv.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/superglm/reml/scop_efs.py` | The retry ladder in `_fit_scop_reml_mode` | 1 |
| `tests/test_scop_efs.py` | Regression test pinning the cold final rung | 1 |
| — | Cross-NumPy verification and issue update | 2 |

---

### Task 1: Add the cold rung

**Files:**
- Modify: `src/superglm/reml/scop_efs.py:934-941`
- Test: `tests/test_scop_efs.py` (new test in `TestSCOPNonConvergenceIsNotSpeciallyAccepted`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_fit_scop_reml_mode` keeps its exact signature, including
  `_certification_retry: int = 0`. Its ladder is now four evaluations deep
  (depths 0-3) instead of three (0-2).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scop_efs.py`, inside the existing class
`TestSCOPNonConvergenceIsNotSpeciallyAccepted`. It needs `Constraint`, `SuperGLM`,
`PSpline`, `np`, `pd` and `scop_efs_module`, all already imported at the top of
that file.

```python
    def test_a_failed_certification_gets_a_cold_final_attempt(self, monkeypatch):
        """The final retry rung drops the warm start, not just the tolerance.

        The two tolerance rungs re-fit from the mode that just failed. Once the
        inner fit has converged tighter than the bar, that reproduces the same
        mode bit-identically -- measured on the fit this exists for, three
        attempts all returned 1.3792e-06 against a bar of 7.1463e-08. Only a
        different starting point can move it, so the last rung starts cold.

        Certification is forced to reject the first three attempts, so the fit
        can only succeed if a fourth exists. The warm/cold pattern is asserted
        too: a fourth attempt that also warm-started would not be the fix.
        """
        warm_starts: list[bool] = []
        checks = {"n": 0}

        real_fit = scop_efs_module._fit_scop_reml_mode
        real_relative = scop_efs_module._scop_mode_newton_relative

        def recording_fit(context, lambdas, **kwargs):
            warm_starts.append(kwargs.get("beta_init") is not None)
            return real_fit(context, lambdas, **kwargs)

        def reject_first_three(mode):
            checks["n"] += 1
            if checks["n"] <= 3:
                return 1.0  # far above any achievable tolerance
            return real_relative(mode)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", recording_fit)
        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_first_three
        )

        rng = np.random.default_rng(0)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = np.round(np.exp(1.0 + 1.5 * x)).astype(float)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, penalty="ssp", constraint=Constraint.fit.increasing)
            },
        )
        model.fit_reml(frame, y, max_reml_iter=5)

        assert checks["n"] >= 4, "the ladder must reach a fourth attempt"
        assert warm_starts[1] is True, "rung 1 warm-starts"
        assert warm_starts[2] is True, "rung 2 warm-starts"
        assert warm_starts[3] is False, "the final rung must start cold"
```

- [ ] **Step 2: Run it to make sure it fails**

Run:
```bash
uv run pytest "tests/test_scop_efs.py::TestSCOPNonConvergenceIsNotSpeciallyAccepted::test_a_failed_certification_gets_a_cold_final_attempt" -q -p no:randomly
```
Expected: **FAIL** with
`RuntimeError: SCOP REML bootstrap did not converge to a coefficient mode`
raised from `src/superglm/reml/scop_efs.py` around line 1557. That is the correct
failure: with only three evaluations, all three are rejected by the injection and
the ladder is exhausted. If you see a different error, stop and report it.

- [ ] **Step 3: Add the cold rung**

In `src/superglm/reml/scop_efs.py`, replace exactly this block (currently at
`934-941`, inside `_fit_scop_reml_mode`, under `if mode_newton_relative > mode_tolerance:`):

```python
        if _certification_retry < 2:
            retry_tolerance = 10.0 ** (-10 - _certification_retry)
            retry_context = replace(
                context,
                pirls_tol=min(context.pirls_tol, retry_tolerance),
            )
            centered_scale = np.sqrt(np.maximum(np.diag(centered_xtwx), 0.0) / sum_w)
            warm_retry = _raw_centering_well_scaled(fisher_mean_x, centered_scale)
```

with:

```python
        if _certification_retry < 3:
            # Rungs 0->1 and 1->2 tighten the inner tolerance and re-fit from the
            # mode that just failed. That rescues a residual left by loose inner
            # convergence -- measured, 29 of 43 retries -- but it cannot move a
            # fit already converged tighter than the bar, which returns the same
            # mode bit-identically however hard the tolerance is squeezed.
            #
            # Rung 2->3 is therefore the cold one. It holds the tightest
            # tolerance the ladder reached rather than squeezing further, so it
            # differs from its predecessor in exactly one respect: the starting
            # point. The warm start it drops comes from a bootstrap fitted at
            # lambda=1e-4, a long way from these lambdas, which is the plausible
            # reason the mode landed off-stationary in the first place.
            cold_rung = _certification_retry == 2
            retry_tolerance = 10.0 ** (-10 - min(_certification_retry, 1))
            retry_context = replace(
                context,
                pirls_tol=min(context.pirls_tol, retry_tolerance),
            )
            centered_scale = np.sqrt(np.maximum(np.diag(centered_xtwx), 0.0) / sum_w)
            warm_retry = not cold_rung and _raw_centering_well_scaled(
                fisher_mean_x, centered_scale
            )
```

Change nothing else. The `return _fit_scop_reml_mode(...)` call below already
keys its `beta_init` / `intercept_init` / `scop_state_init` off `warm_retry`, so
setting `warm_retry` to `False` is what makes the rung cold.

Note `min(_certification_retry, 1)` in the tolerance: depth 0 gives `1e-10`,
depth 1 gives `1e-11`, and depth 2 gives `1e-11` again rather than `1e-12`. That
is deliberate — the cold rung must vary only the starting point.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
uv run pytest "tests/test_scop_efs.py::TestSCOPNonConvergenceIsNotSpeciallyAccepted::test_a_failed_certification_gets_a_cold_final_attempt" -q -p no:randomly
```
Expected: **PASS**.

- [ ] **Step 5: Run the file and the gates**

Run:
```bash
uv run pytest tests/test_scop_efs.py -q -p no:randomly
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```
Expected: all pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/reml/scop_efs.py tests/test_scop_efs.py
git commit -m "Give a failed certification one cold attempt before giving up

Both existing retry rungs tighten pirls_tol and re-fit from the mode that
just failed. That rescues a residual left by loose inner convergence -- 29
of 43 measured retries -- but cannot move a fit already converged tighter
than the bar, which reproduces the same mode bit-identically however hard
the tolerance is squeezed. Measured on the fit in #179: three attempts, one
value (1.3792e-06) against a fixed bar (7.1463e-08).

A cold start was never tried. The warm start comes from a bootstrap fitted
at lambda=1e-4, a long way from the candidate's lambdas, which is the
plausible reason the mode lands off-stationary. The new final rung holds the
tightest tolerance already reached and drops the warm start, so it differs
from its predecessor only in where it starts.

The bar, the two tolerance rungs and all four fatal call sites are
unchanged. Strictly more recoveries: the rung is reached only where the
previous code had already given up.

Refs #179

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Verify across the NumPy boundary and record the outcome

**Files:**
- No source changes. Verification plus an issue comment.

**Interfaces:**
- Consumes: Task 1 complete and committed.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Full suite on the default NumPy**

Run:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q -p no:randomly
```
Expected: **5112 passed / 84 skipped** — the 5111 baseline plus the one new test.
Any other number, or any failure, means the extra rung changed an outcome it
should not have; stop and report rather than adjusting a test.

- [ ] **Step 2: The regression, on the NumPy that fails today**

`test_stored_objective_reproduction_multi_scop` fails on NumPy ≤ 2.4.1 before
this change. Confirm it now passes:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/sg-np241 OMP_NUM_THREADS=1 \
  uv run --python 3.12 --extra dev --with numpy==2.4.1 \
  pytest "tests/test_scop_efs.py::TestMultiSCOPIntegration::test_stored_objective_reproduction_multi_scop" \
  -q -p no:randomly
```
Expected: **1 passed**. This is the load-bearing check — it is the reported
failure from #179.

- [ ] **Step 3: The whole SCOP suite on that NumPy**

One test passing is not enough; confirm the rung did not disturb the rest on the
older numerics.

```bash
UV_PROJECT_ENVIRONMENT=/tmp/sg-np241 OMP_NUM_THREADS=1 \
  uv run --python 3.12 --extra dev --with numpy==2.4.1 \
  pytest tests/test_scop_efs.py tests/test_scop_irls_state.py -q -p no:randomly
```
Expected: all pass. Record the counts in your report.

- [ ] **Step 4: Confirm the tolerance rungs still rescue**

The cold rung is additive. If the 29 tolerance-rung rescues dropped, the rungs
were reordered wrongly. Write this to `/tmp/retry_probe.py`:

```python
import superglm.reml.scop_efs as se

EVENTS = []
_rel = se._scop_mode_newton_relative
_tol = se._scop_mode_tolerance
_fit = se._fit_scop_reml_mode
DEPTH = {"n": 0}


def fit_probe(context, lambdas, **kw):
    DEPTH["n"] = kw.get("_certification_retry", 0)
    return _fit(context, lambdas, **kw)


def rel_probe(mode):
    v = _rel(mode)
    EVENTS.append({"rel": float(v), "retry": DEPTH["n"]})
    return v


def tol_probe(mode, pirls_tol):
    v = _tol(mode, pirls_tol)
    if EVENTS:
        EVENTS[-1]["tol"] = float(v)
        EVENTS[-1]["reject"] = EVENTS[-1]["rel"] > v
    return v


se._fit_scop_reml_mode = fit_probe
se._scop_mode_newton_relative = rel_probe
se._scop_mode_tolerance = tol_probe


def pytest_sessionfinish(session, exitstatus):
    checks = [e for e in EVENTS if "tol" in e]
    rescues = [e for e in checks if e["retry"] > 0 and not e.get("reject")]
    by_depth = {}
    for e in rescues:
        by_depth[e["retry"]] = by_depth.get(e["retry"], 0) + 1
    print(f"\nchecks={len(checks)} rescues={len(rescues)} by_depth={by_depth}")
```

Run:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=/tmp uv run pytest \
  tests/test_scop_efs.py tests/test_scop_irls_state.py \
  -q -p no:randomly -p retry_probe -s 2>&1 | grep "checks="
```
Expected: rescues at depths 1 and 2 still total **29 or more**. A count below 29
means the tolerance rungs were damaged — stop and report.

- [ ] **Step 5: Comment on issue #179**

Write the comment to `/tmp/issue179-comment.md`, filling in the real numbers you
measured in Steps 1-4:

```markdown
Partially addressed by the cold certification-retry rung on
`fix/scop-cold-certification-retry`.

**Cause.** Both existing retry rungs tighten `pirls_tol` and warm-start from the
mode that just failed (43 of 43 measured retries warm-start). Once the inner fit
has converged tighter than the bar, that reproduces the same mode
bit-identically — three attempts returned `1.3792e-06` against a fixed bar of
`7.1463e-08`. The starting point was never varied.

**Change.** A final rung holds the tightest tolerance already reached and refits
cold. The bar, the two tolerance rungs and the four fatal call sites are
untouched.

**Measured.** `test_stored_objective_reproduction_multi_scop` on NumPy 2.4.1:
FAIL before, PASS after. SCOP suites on 2.4.1: <counts from Step 3>. Tolerance-rung
rescues still <count from Step 4> (was 29) — the rung is additive.

**This issue stays open.** The asymmetry it was filed for is unchanged: a
certification rejection is still fatal at four call sites and survivable at the
fifth (the line search, which `continue`s). This makes reaching a fatal site
rarer; it does not make the five sites consistent.
```

Then:
```bash
gh issue comment 179 --body-file /tmp/issue179-comment.md
```

- [ ] **Step 6: Push and open a PR**

```bash
git push -u origin fix/scop-cold-certification-retry
```

PR body must state: `release:none`, version stays `0.17.0` folding into the
unpublished candidate, and that **this does not authorize a tag**; the measured
cause (both rungs vary tolerance, never the starting point; 43 of 43 retries
warm-start); the before/after on NumPy 2.4.1; that the bar and the four fatal
sites are untouched and the change is recovery-only; and the two limitations
from the spec — the asymmetry remains, and the cold rescue is one data point.

Do **not** use `gh pr edit --body-file`; it exits 1 here and silently discards
the body. Create with `gh pr create --body-file`, and for later edits use
`gh api -X PATCH repos/:owner/:repo/pulls/<N> -F body=@file`, verifying with a
string that exists only in the new body.

**Do not merge, tag, or publish.** Report back for review.
