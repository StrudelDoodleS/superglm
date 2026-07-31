# SCOP Candidate-Site Lambda-Step Backoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A candidate-site certification failure backs the lambda step off toward the certified mode it was taken from instead of aborting the fit (issue #179's remaining asymmetry).

**Architecture:** One new module-private helper `_backoff_scop_candidate_step` in `src/superglm/reml/scop_efs.py` applies the line search's damped-trial formula between a certified origin mode and the failed proposal; `optimize_scop_efs_reml` tracks the step's origin (`boot_mode`, then each accepted `retained_mode`) and calls the helper only where today's code raises. Bootstrap, fixed-lambda, and final-fit sites are untouched.

**Tech Stack:** Python 3.12+, numpy, pytest (+ `-p no:randomly` for deterministic runs), uv.

**Spec:** `docs/superpowers/specs/2026-07-31-scop-candidate-step-backoff-design.md`

## Global Constraints

- The success path must be bit-identical: the helper runs only after `_fit_scop_reml_mode` returned `None` at the candidate site, where the previous code raised.
- On exhaustion, raise exactly `"SCOP REML candidate did not converge to a coefficient mode"` — same message as today.
- No consulting or citing any external implementation (mgcv/scam source is forbidden); the mechanism derives from this repo's `_backtrack_scop_efs_candidate` and published literature already cited in the spec.
- `docs/superpowers/` is gitignored: every commit touching it needs `git add -f` and a `git log --stat` check.
- Branch: `fix/scop-candidate-step-backoff` (already created; design doc committed).
- PR will declare `release:none` (folds into unpublished 0.17.0). Never write a closing keyword next to `#179`, even negated — use "Refs #179".
- Repo check suite: `uv run pytest tests/ -q`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv lock --check`, `uv pip check`, `uv run python run_test.py`.

---

### Task 1: Baseline ladder instrumentation (before any code change)

The corpus-inertness check needs a before/after comparison. The branch currently
carries only the design doc, so measuring now measures master's behaviour.

**Files:**
- Create: `/home/max/.claude/jobs/f532cf83/tmp/ladder_instrumentation.py` (not committed — job-local tooling)
- Create (output): `/home/max/.claude/jobs/f532cf83/tmp/ladder_baseline.json`

**Interfaces:**
- Produces: `ladder_baseline.json` with keys `checks`, `rejections`, `retries`, `tolerance_rescues`, `cold_rescues`, `backoff_calls` — Task 5 compares against it byte-for-byte on the same test selection.

- [ ] **Step 1: Write the instrumentation plugin**

```python
"""Count certification-ladder activity across the SCOP suites.

A pytest plugin: patches the module attributes scop_efs call sites resolve at
runtime. Rescue classification happens at the check level: each certification
is one `_scop_mode_newton_relative` + `_scop_mode_tolerance` pair evaluated at
the current retry depth; a pass at depth 1-2 is a tolerance-rung rescue, a
pass at depth 3 a cold-rung rescue. Tests that monkeypatch the certification
call through these wrappers when they delegate to the real function, so the
same test selection yields identical counts run-to-run.

Usage (from the repo root):
    PYTHONPATH=/home/max/.claude/jobs/f532cf83/tmp uv run pytest \
        tests/test_scop_efs.py tests/test_scop_irls_state.py \
        -q -p no:randomly -p ladder_instrumentation \
        -k "not TestCandidateStepBackoff"
Writes ladder_counts.json to the current directory.
"""

import json

import superglm.reml.scop_efs as scop_efs

COUNTS = {
    "checks": 0,
    "rejections": 0,
    "retries": 0,
    "tolerance_rescues": 0,
    "cold_rescues": 0,
    "backoff_calls": 0,
}

_real_fit = scop_efs._fit_scop_reml_mode
_real_relative = scop_efs._scop_mode_newton_relative
_real_tolerance = scop_efs._scop_mode_tolerance

_DEPTH = {"value": 0}
_LAST = {"relative": None}


def _fit(context, lambdas, **kwargs):
    depth = kwargs.get("_certification_retry", 0)
    if depth > 0:
        COUNTS["retries"] += 1
    outer = _DEPTH["value"]
    _DEPTH["value"] = depth
    try:
        return _real_fit(context, lambdas, **kwargs)
    finally:
        _DEPTH["value"] = outer


def _relative(mode):
    value = _real_relative(mode)
    COUNTS["checks"] += 1
    _LAST["relative"] = value
    return value


def _tolerance(mode, pirls_tol):
    bar = _real_tolerance(mode, pirls_tol)
    if _LAST["relative"] is not None:
        depth = _DEPTH["value"]
        if _LAST["relative"] > bar:
            COUNTS["rejections"] += 1
        elif depth in (1, 2):
            COUNTS["tolerance_rescues"] += 1
        elif depth == 3:
            COUNTS["cold_rescues"] += 1
        _LAST["relative"] = None
    return bar


scop_efs._fit_scop_reml_mode = _fit
scop_efs._scop_mode_newton_relative = _relative
scop_efs._scop_mode_tolerance = _tolerance

if hasattr(scop_efs, "_backoff_scop_candidate_step"):
    _real_backoff = scop_efs._backoff_scop_candidate_step

    def _backoff(*args, **kwargs):
        COUNTS["backoff_calls"] += 1
        return _real_backoff(*args, **kwargs)

    scop_efs._backoff_scop_candidate_step = _backoff


def pytest_sessionfinish(session, exitstatus):
    with open("ladder_counts.json", "w") as handle:
        json.dump(COUNTS, handle, indent=2, sort_keys=True)
```

- [ ] **Step 2: Run it on the unchanged tree**

```bash
cd /home/max/projects/superglm && \
PYTHONPATH=/home/max/.claude/jobs/f532cf83/tmp OMP_NUM_THREADS=1 uv run pytest \
    tests/test_scop_efs.py tests/test_scop_irls_state.py \
    -q -p no:randomly -p ladder_instrumentation && \
mv ladder_counts.json /home/max/.claude/jobs/f532cf83/tmp/ladder_baseline.json && \
cat /home/max/.claude/jobs/f532cf83/tmp/ladder_baseline.json
```

Expected: all tests pass; JSON shows `backoff_calls: 0`, `tolerance_rescues` 29 and `cold_rescues` 8 (the issue's measured figures — if they differ, record the actual numbers as the baseline; the comparison in Task 5 is against this file, not against the issue).

---

### Task 2: The rescue path (TDD red → green)

**Files:**
- Modify: `src/superglm/reml/scop_efs.py:1140` area (helper, placed directly before `_backtrack_scop_efs_candidate`), `:1609-1612` (origin tracking), `:1644-1656` (call site), `:1837-1841` (Step 9 origin update)
- Test: `tests/test_scop_efs.py` (new class `TestCandidateStepBackoff`, placed after the class containing `test_a_failed_certification_gets_a_cold_final_attempt` — find it with `rg -n "class Test" tests/test_scop_efs.py`)

**Interfaces:**
- Consumes: `_fit_scop_reml_mode(context, lambdas, *, beta_init, intercept_init, scop_state_init, phase, reml_iteration, line_search_iteration=None, trial_alpha=None, require_converged, _certification_retry=0) -> _SCOPREMLMode | None`; `_SCOPREMLMode.lambdas: dict[str, float]`, `.result.beta`, `.result.intercept`, `.scop_states`; `_SCOP_EFS_MAX_BACKTRACK_ATTEMPTS = 8`.
- Produces: `_backoff_scop_candidate_step(context, origin, proposed_lambdas, *, reml_iteration) -> tuple[_SCOPREMLMode, dict[str, float]] | None` — Task 5's instrumentation wraps this exact name.

- [ ] **Step 1: Write the failing rescue test**

Fixture matches `test_a_failed_certification_gets_a_cold_final_attempt` (its
bootstrap is known to certify on the first check). Phase tracking targets the
candidate ladder without counting on the bootstrap's check count.

```python
class TestCandidateStepBackoff:
    """A candidate certification failure backs the lambda step off (#179).

    The iteration-1 candidate consumes the one EFS proposal that bypasses
    the line search, so it was the one lambda movement with no damping
    behind it: four call sites raised on a rejection the line search
    survives. The backoff applies the line search's own trial formula --
    damped geometric steps in log-lambda -- between the certified mode the
    step was taken from and the proposal that failed. Sites with no
    certified predecessor (bootstrap, fixed-lambda) keep raising.
    """

    @staticmethod
    def _model():
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
        return model, frame, y

    @staticmethod
    def _phase_tracking(monkeypatch, state):
        """Expose which top-level phase each certification check belongs to."""
        real_fit = scop_efs_module._fit_scop_reml_mode

        def tracking_fit(context, lambdas, **kwargs):
            previous = state["phase"]
            state["phase"] = kwargs.get("phase")
            try:
                return real_fit(context, lambdas, **kwargs)
            finally:
                state["phase"] = previous

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", tracking_fit)

    def test_a_failed_candidate_ladder_gets_a_damped_step(self, monkeypatch):
        """Candidate certification failure damps the lambda step, not the fit.

        The iteration-1 candidate's entire four-rung ladder is forced to
        reject; every other check is real. Before the backoff this raised
        ``SCOP REML candidate did not converge to a coefficient mode``; now
        a shorter step toward the certified bootstrap must be found and the
        fit must succeed. Asserted through the observable outcome -- the
        fit completes and the fitted curve respects the constraint -- plus
        the forced-rejection count, which pins that the whole ladder was
        exhausted rather than the rescue arriving early.
        """
        state = {"phase": None, "candidate_rejections": 0}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_the_first_candidate_ladder(mode):
            if state["phase"] == "candidate" and state["candidate_rejections"] < 4:
                state["candidate_rejections"] += 1
                return 1.0  # far above any achievable bar
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_the_first_candidate_ladder
        )

        model, frame, y = self._model()
        model.fit_reml(frame, y, max_reml_iter=5)

        assert state["candidate_rejections"] == 4, "the full ladder must be exhausted first"
        assert model.result.beta is not None
        fitted = model.predict(frame)
        assert np.all(np.diff(fitted) >= -1e-8), "the rescued fit still honours the constraint"
```

If `model.predict(frame)` is not the prediction API, mirror whatever the
nearest SCOP integration test in this file calls (`rg -n "\.predict\(" tests/test_scop_efs.py`);
keep a monotonicity assertion in some form.

- [ ] **Step 2: Run it to verify it fails for today's reason**

```bash
cd /home/max/projects/superglm && uv run pytest \
    "tests/test_scop_efs.py::TestCandidateStepBackoff::test_a_failed_candidate_ladder_gets_a_damped_step" \
    -q -p no:randomly
```

Expected: FAIL with `RuntimeError: SCOP REML candidate did not converge to a coefficient mode` (the pre-change behaviour, raised at the candidate site).

- [ ] **Step 3: Implement the helper**

Insert directly before `_backtrack_scop_efs_candidate` (currently line 1140):

```python
def _backoff_scop_candidate_step(
    context: _SCOPREMLFitContext,
    origin: _SCOPREMLMode,
    proposed_lambdas: dict[str, float],
    *,
    reml_iteration: int,
) -> tuple[_SCOPREMLMode, dict[str, float]] | None:
    """Retry a failed candidate at damped steps toward its certified origin.

    The candidate consumes the one EFS proposal that never went through the
    line search, so a certification failure there had no damping behind it
    and aborted the fit -- the same rejection the line search survives by
    trying the next alpha.  This applies the line search's trial formula,
    geometric interpolation in log-lambda at alpha = 0.5**attempt, between
    the mode the step was taken from and the proposal that failed, warm-
    starting each attempt from the origin.  The full step (alpha = 1.0) was
    the original candidate fit, so the attempts complete the same forward
    ladder the line search runs.

    Returns the first certified mode with its lambdas, or ``None`` when
    every damped attempt fails, in which case the caller keeps its raise.
    """
    changed_names = [
        name
        for name, proposed in proposed_lambdas.items()
        if name in origin.lambdas and proposed != origin.lambdas[name]
    ]
    if not changed_names:
        return None

    log_directions: dict[str, float] = {}
    for name in changed_names:
        old = float(origin.lambdas[name])
        proposed = float(proposed_lambdas[name])
        if old <= 0.0 or proposed <= 0.0 or not np.isfinite(old + proposed):
            raise ValueError("SCOP EFS lambda trials must be positive and finite")
        log_directions[name] = float(np.log(proposed) - np.log(old))

    for attempt in range(1, _SCOP_EFS_MAX_BACKTRACK_ATTEMPTS):
        alpha = 0.5**attempt
        trial_lambdas = origin.lambdas.copy()
        for name in changed_names:
            log_trial = np.log(origin.lambdas[name]) + alpha * log_directions[name]
            trial_lambdas[name] = float(np.clip(np.exp(log_trial), 1.0e-6, 1.0e10))
        mode = _fit_scop_reml_mode(
            context,
            trial_lambdas,
            beta_init=origin.result.beta,
            intercept_init=float(origin.result.intercept),
            scop_state_init=origin.scop_states if origin.scop_states else None,
            phase="candidate",
            reml_iteration=reml_iteration,
            trial_alpha=alpha,
            require_converged=True,
        )
        if mode is not None:
            return mode, trial_lambdas
    return None
```

- [ ] **Step 4: Track the step origin and rescue at the call site**

In `optimize_scop_efs_reml`, after the `warm_scop_states` initialisation
(currently line 1611, just above `retained_mode: _SCOPREMLMode | None = None`):

```python
    step_origin: _SCOPREMLMode = boot_mode
```

Replace the candidate site (currently lines 1644-1656):

```python
        if retained_mode is None:
            current_mode = _fit_scop_reml_mode(
                fit_context,
                lambdas,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                scop_state_init=warm_scop_states,
                phase="candidate",
                reml_iteration=n_reml_iter,
                require_converged=True,
            )
            if current_mode is None:
                # The one EFS proposal with no line search behind it; back the
                # step off toward the certified mode it was taken from.  The
                # bootstrap and fixed-lambda sites have no such mode and stay
                # fatal.
                rescue = _backoff_scop_candidate_step(
                    fit_context,
                    step_origin,
                    lambdas,
                    reml_iteration=n_reml_iter,
                )
                if rescue is None:
                    raise RuntimeError(
                        "SCOP REML candidate did not converge to a coefficient mode"
                    )
                current_mode, lambdas = rescue
        else:
            current_mode = retained_mode
            retained_mode = None
```

In Step 9 (after `warm_scop_states = ...`, currently line 1841):

```python
        step_origin = retained_mode
```

- [ ] **Step 5: Run the new test to verify it passes**

```bash
cd /home/max/projects/superglm && uv run pytest \
    "tests/test_scop_efs.py::TestCandidateStepBackoff::test_a_failed_candidate_ladder_gets_a_damped_step" \
    -q -p no:randomly
```

Expected: PASS.

- [ ] **Step 6: Run the neighbouring certification tests**

```bash
cd /home/max/projects/superglm && uv run pytest tests/test_scop_efs.py -q -p no:randomly
```

Expected: all pass, including `test_a_failed_certification_gets_a_cold_final_attempt` and `test_a_genuinely_non_converging_fit_still_raises`.

- [ ] **Step 7: Commit**

```bash
cd /home/max/projects/superglm && \
git add src/superglm/reml/scop_efs.py tests/test_scop_efs.py && \
git commit -m "Give a failed candidate certification a damped lambda step"
```

---

### Task 3: Pin the recoverability boundary

Three pins, all expected green on first run — they assert designed behaviour
that must not drift: the backoff is bounded (exhaustion stays loud), and sites
without a certified predecessor still raise.

**Files:**
- Test: `tests/test_scop_efs.py` (`TestCandidateStepBackoff`, from Task 2)

**Interfaces:**
- Consumes: `TestCandidateStepBackoff._model()` and `._phase_tracking(monkeypatch, state)` from Task 2, verbatim.

- [ ] **Step 1: Check what boundary coverage already exists**

```bash
cd /home/max/projects/superglm && \
rg -n "bootstrap did not converge|fixed-lambda SCOP fit did not converge" tests/ ; \
rg -n "fit_fixed_scop_reml|smoothing=" tests/test_scop_efs.py | head -20
```

If a test already forces the bootstrap or fixed-lambda raise via certification
rejection, skip the corresponding step below and note it in the plan narrative.
Use the fixed-lambda invocation the existing tests use (how a SCOP fit with all
lambdas held fixed is spelled at the `fit_reml` API — the `rg` above shows it).

- [ ] **Step 2: Add the exhaustion pin**

```python
    def test_an_unrecoverable_candidate_still_raises(self, monkeypatch):
        """When no damped step certifies either, the failure stays loud.

        Every candidate-phase certification is rejected, so the ladder and
        then every backoff attempt fail. The exact candidate error must
        surface: the backoff is bounded, and it must not convert a hard
        failure into a silent stall or an unbounded retry.
        """
        state = {"phase": None}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_every_candidate_check(mode):
            if state["phase"] == "candidate":
                return 1.0
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_every_candidate_check
        )

        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML candidate did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)
```

- [ ] **Step 3: Add the bootstrap pin (if Step 1 found no equivalent)**

```python
    def test_a_failed_bootstrap_has_nothing_to_back_off_to(self, monkeypatch):
        """The recoverability principle's boundary: no predecessor, no rescue.

        Rejecting every certification kills the bootstrap after its ladder.
        There is no earlier certified mode to damp toward, so the loud
        error is the designed outcome, unchanged by the candidate backoff.
        """
        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", lambda mode: 1.0)
        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML bootstrap did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)
```

- [ ] **Step 4: Add the fixed-lambda pin (if Step 1 found no equivalent)**

Same rejection monkeypatch as Step 3, invoking the fixed-lambda SCOP path the
way Step 1's `rg` showed existing tests do, expecting
`RuntimeError, match="fixed-lambda SCOP fit did not converge"`.

- [ ] **Step 5: Run the class and the file**

```bash
cd /home/max/projects/superglm && uv run pytest \
    "tests/test_scop_efs.py::TestCandidateStepBackoff" -q -p no:randomly && \
uv run pytest tests/test_scop_efs.py -q -p no:randomly
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/max/projects/superglm && git add tests/test_scop_efs.py && \
git commit -m "Pin the recoverability boundary around the candidate backoff"
```

---

### Task 4: Both sides of the NumPy boundary

**Files:** none (verification only)

- [ ] **Step 1: The issue's repro, on the failing-side NumPy**

```bash
cd /home/max/projects/superglm && \
UV_PROJECT_ENVIRONMENT=/home/max/.claude/jobs/f532cf83/tmp/venv-np241 OMP_NUM_THREADS=1 \
uv run --python 3.12 --extra dev --with numpy==2.4.1 pytest \
    "tests/test_scop_efs.py::TestMultiSCOPIntegration::test_stored_objective_reproduction_multi_scop" \
    -q -p no:randomly
```

Expected: PASS (rescued by the cold rung from #181, untouched by this change).

- [ ] **Step 2: The SCOP suites on NumPy 2.4.1**

```bash
cd /home/max/projects/superglm && \
UV_PROJECT_ENVIRONMENT=/home/max/.claude/jobs/f532cf83/tmp/venv-np241 OMP_NUM_THREADS=1 \
uv run --python 3.12 --extra dev --with numpy==2.4.1 pytest \
    tests/test_scop_efs.py tests/test_scop_irls_state.py -q -p no:randomly
```

Expected: all pass.

---

### Task 5: Corpus inertness

**Files:**
- Create (output): `/home/max/.claude/jobs/f532cf83/tmp/ladder_after.json`

**Interfaces:**
- Consumes: `ladder_baseline.json` from Task 1; the plugin wraps `_backoff_scop_candidate_step` by exact name now that it exists.

- [ ] **Step 1: Re-run the instrumentation on the branch, excluding the new tests**

```bash
cd /home/max/projects/superglm && \
PYTHONPATH=/home/max/.claude/jobs/f532cf83/tmp OMP_NUM_THREADS=1 uv run pytest \
    tests/test_scop_efs.py tests/test_scop_irls_state.py \
    -q -p no:randomly -p ladder_instrumentation \
    -k "not TestCandidateStepBackoff" && \
mv ladder_counts.json /home/max/.claude/jobs/f532cf83/tmp/ladder_after.json && \
diff <(python3 -m json.tool /home/max/.claude/jobs/f532cf83/tmp/ladder_baseline.json) \
     <(python3 -m json.tool /home/max/.claude/jobs/f532cf83/tmp/ladder_after.json)
```

Expected: `diff` is empty — identical counts on the identical test population
(`-k` deselects nothing on the baseline run and only the new class here), and
`backoff_calls` is 0: the change is invisible everywhere except injected
failure. If any count moved, stop and find out why before proceeding.

---

### Task 6: Full verification, plan narrative, and commit

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-scop-candidate-step-backoff.md` (append outcome narrative)

- [ ] **Step 1: The repo's ordinary checks**

```bash
cd /home/max/projects/superglm && uv run pytest tests/ -q && \
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && \
uv lock --check && uv pip check && uv run python run_test.py
```

Expected: suite passes (baseline 5112 passed / 84 skipped, plus the new tests), all checks clean.

- [ ] **Step 2: Append the outcome narrative to this plan**

Record: measured baseline vs after counts, both NumPy runs, full-suite totals,
and any deviation from the plan (e.g. boundary pins skipped as already
covered). The narrative is the authority over the checkboxes.

- [ ] **Step 3: Commit the plan (forced add — gitignored path)**

```bash
cd /home/max/projects/superglm && \
git add -f docs/superpowers/plans/2026-07-31-scop-candidate-step-backoff.md && \
git commit -m "Add the implementation plan for the candidate-site step backoff" && \
git log --stat -1
```

Expected: `git log --stat` shows the plan file in the commit.

---

## Outcome (2026-07-31, executed same-day; the narrative below is the authority over the checkboxes)

All six tasks ran in order on `fix/scop-candidate-step-backoff`; no deviations
of substance. The full record:

**Task 1 — baseline.** 146 tests, all passing, 4.03s. Counts:
`checks 565, rejections 51, retries 53, tolerance_rescues 29, cold_rescues 8,
backoff_calls 0`. The 29 and 8 match the figures measured on the issue
exactly; the check total differs from the issue's pre-#181 609 because the
cold rung added ladder traffic and the counting seam differs slightly. The
comparison basis is this file, as planned.

**Task 2 — rescue path.** The red run failed with today's exact error raised
from the candidate site (`scop_efs.py:1656`), confirming the injection
targets the right seam. Implementation landed as specified — helper before
`_backtrack_scop_efs_candidate`, `step_origin` beside the warm-start state,
rescue at the candidate site — and the test went green. File suite: 136
passed. Commit `c97e710`.

**Task 3 — boundary pins.** Step 1 found no existing coverage for either the
bootstrap or the fixed-lambda raise, so both pins were written (plus the
exhaustion pin). The fixed-lambda invocation is
`PSpline(..., lambda_policy=LambdaPolicy(mode="fixed", value=1.0))`, the
spelling `test_fixed_lambda_policy_still_works` uses. All three green on
first run, as expected for pins. File suite: 139 passed. Commit `a9589a5`
(plus `ruff format` line-joins in a follow-up commit).

**Task 4 — NumPy boundary.** The issue's repro passes on 2.4.1, and the SCOP
suites pass there too: 150 tests (146 + the new 4). Default env is
python 3.13.5 / numpy 2.4.2 / scipy 1.17.1, covered by Task 6.

**Task 5 — corpus inertness.** Re-run with `-k "not TestCandidateStepBackoff"`
(146 passed, 4 deselected): the JSON diff against the Task 1 baseline is
empty. Identical counts on the identical population, `backoff_calls` 0 —
the change is invisible everywhere except injected failure.

**Task 6 — full verification.** `uv run pytest tests/ -q`: **5116 passed /
84 skipped** in 3:26 (baseline 5112 + the 4 new tests). `ruff check` clean;
`ruff format --check` clean after formatting the two touched files;
`uv lock --check`, `uv pip check`, and `run_test.py` ("END-TO-END COMPLETE")
all clean.

## Review round 1 (2026-07-31, PR #183 — Codex + claude bot, same day)

CI on the PR: all checks green, including `Python 3.12 · complete
non-browser suite` (the successor of the job where #179 surfaced).

**Codex P2 (confirmed by a red test before fixing): rescue-then-stall.**
If the rescue certified but the very next line search accepted nothing, the
`line_search_stalled` branch would *return* the rescued mode
(`converged=False`) where pre-backoff code raised — publishing a vector no
acceptance gate ever endorsed. Fixed: a per-iteration `rescue_alpha` flag
makes that branch raise the exact candidate error; stalls from
accepted-progress states keep their existing semantics. Pinned by
`test_a_rescue_the_line_search_cannot_move_from_still_raises`. The claude
bot's finding 2 was the same boundary (it proposed documenting; the guard
is stronger and matches Codex's remedy). Spec updated.

**claude bot findings, dispositions:**
1. *Latent `KeyError`* — `trial_lambdas` seeded from the origin's key set
   could drop a proposal-only name. Not reachable today (upstream seeds
   both dicts identically) but real; fixed by seeding from
   `proposed_lambdas`, red-tested first
   (`test_the_backoff_preserves_the_proposal_key_set`, which also covers
   the no-movement `return None` branch).
2. Fixed via the stall guard above.
3. *No signal of a rescue outside the trace* — added a `verbose` line and
   `candidate_backoff_alpha` (null normally) to the level-2 reml payload;
   helper now returns its alpha.
4. *`lambda_history` recorded the never-fitted proposal* — on rescue the
   last entry (by construction the failed vector) is replaced with the
   adopted damped vector; history now contains fitted vectors only.
   Spec bullet rewritten.
5. *Dead Step-9 `step_origin` update* — kept, with the requested comment
   marking it defensive.
6. *`ValueError` vs `RuntimeError` on non-finite endpoints* — left as is,
   per the reviewer's own disposition (mirrors the line search, effectively
   unreachable).
7. *Mechanism not asserted* — the rescue test now pins the first backoff
   attempt at `alpha=0.5` and the adopted vector strictly between the
   bootstrap and the failed proposal (`_phase_tracking` records top-level
   fits with phase, `trial_alpha`, lambdas).

**Re-verification on the final tree:** backoff class 6 passed; file 141
passed; corpus inertness diff empty (146-test population, 6 deselected);
SCOP suites on numpy 2.4.1: 152 passed; full suite on the final tree:
**5118 passed / 84 skipped** (baseline 5112 + the 6 new tests).
