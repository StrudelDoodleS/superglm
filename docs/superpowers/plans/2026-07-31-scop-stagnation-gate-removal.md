# SCOP Stagnation Gate Removal Implementation Plan

> **Superseded in part — read the outcome, not this plan, for the position.**
> The removal shipped as planned, but three claims in the body below did not
> survive review and are retracted. **(1)** "Item 2b closes with it … a separate
> gradient-norm test would be **redundant**" (`:635-638`, restated at `:667`) —
> 2b is *unresolved*, not closed: its "keep deviance stagnation as the primary
> criterion" clause is ambiguous between the retired acceptance gate and the
> inner PIRLS convergence test, and on the second reading 2b is untouched and
> more attractive than before. What shipped is asymmetric against a gradient
> norm, not a superset of it. **(2)** "the sole acceptance path" (`:29`, `:342`)
> — the penalized-score certification is the sole *certification*, but
> `require_converged and not result.converged` still rejects a mode before it is
> ever computed. **(3)** "the gate was **reached** only by its own stub-driven
> tests" (`:34`, `:339`, `:663`) — reaching the gate and being accepted by it
> are different bars. Any non-converged inner fit reaches it, and real
> outer-loop fits do and are declined there. The measured finding is narrower:
> no real fit in the corpus is *accepted* by it.
>
> Authoritative: the **Status 2026-07-31** paragraph of
> `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md`, and
> `docs/superpowers/specs/2026-07-31-scop-stagnation-gate-removal-design.md`.
> The body below is left unedited as the historical record of what was planned.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the SCOP deviance-stagnation acceptance rule and the private
`stagnation_log` channel that feeds it, leaving the penalized-score mode
certification as the sole acceptance path.

**Architecture:** Pure removal in three files plus their tests. PR #176 fixed
the cause (rank truncation on the factor drops the unidentifiable
`exp(gamma) -> 0` direction), so the workaround built on top of the symptom has
no remaining consumer — measured: across the full suite the gate is reached only
by its own stub-driven tests. Task 1 removes the gate and lands the behaviour
change; Task 2 removes the now-unfed diagnostic channel; Task 3 verifies and
records.

**Tech Stack:** Python, NumPy, pytest, `uv`, ruff, `ty`.

**Spec:** `docs/superpowers/specs/2026-07-31-scop-stagnation-gate-removal-design.md`

## Global Constraints

- Branch `refactor/retire-scop-stagnation-gate` is already checked out, with the
  spec committed at `f3c3ac9`. Do not create another branch.
- **Version stays `0.17.0`. Do not bump.** This folds into the existing
  unpublished candidate; the PR body declares the behaviour change.
- **Do not tag or publish anything.** Publication requires a separate explicit
  instruction naming the tag.
- `docs/superpowers/` is in `.gitignore`. Any doc edit needs `git add -f <path>`,
  and must be confirmed present in `git log --stat`.
- Full suite (~3 min):
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q -p no:randomly`
- Baseline before this plan: **5133 passed / 84 skipped**.
- CI `quality` runs **both** `uv run ruff check` and `uv run ruff format --check`.
- `ty` baseline is **806**; it must not increase.
- **Never** run `uv run --python 3.10 ...` or `uv sync --extra dev` — both
  repoint or prune the project venv.
- Do not touch `bs`/`cr` QP-path behaviour, `iteration_log`/`record_diagnostics`,
  or `_scop_mode_newton_relative`/`_scop_mode_tolerance`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/superglm/reml/scop_efs.py` | Gate predicate, tuned constants, acceptance branch, channel request, publish-time scrub | 1, 2 |
| `src/superglm/solvers/pirls.py` | `StagnationRecord` type and the `stagnation_log` result field | 2 |
| `src/superglm/solvers/irls_direct.py` | `_record_stagnation` flag, accumulator, per-iteration append | 2 |
| `tests/test_scop_efs.py` | Gate tests; the replacement contract test | 1, 2 |
| `tests/test_scop_irls_state.py` | Channel/recorder agreement test | 2 |
| `tests/test_fit_transactions.py` | Freeze-semantics test that borrowed `StagnationRecord` | 2 |
| `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md` | Item 2c status | 3 |

---

### Task 1: Remove the gate and pin the new contract

**Files:**
- Modify: `src/superglm/reml/scop_efs.py:781-825`, `828-841`, `844-900`, `991-1014`
- Test: `tests/test_scop_efs.py:2363-2691`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_scop_deviance_stagnated`, `_scop_stagnation_window`,
  `_STAGNANT_DEVIANCE_TOLERANCE`, `_STAGNANT_DEVIANCE_WINDOW_MAX` and
  `_STAGNANT_DEVIANCE_WINDOW_MIN` no longer exist in
  `superglm.reml.scop_efs`. `_fit_scop_reml_mode` keeps its signature
  unchanged and returns `None` for any non-converged result when
  `require_converged=True`.

- [ ] **Step 1: Write the failing test**

Add this method inside the existing `TestSCOPBoundaryStagnationAcceptance`
class in `tests/test_scop_efs.py` (put it directly above
`test_a_rejected_candidate_is_not_marked_converged`, around line 2645). It uses
the class's existing `_stub` and `_run_gate` helpers.

```python
    def test_a_stagnant_candidate_is_no_longer_specially_accepted(self, monkeypatch):
        """A boundary-stagnant fit is a non-convergence like any other.

        Before item 2c the gate admitted this stub: it flipped ``converged``
        to True and the fit proceeded into geometry assembly, which a bare
        stub cannot satisfy, so the failure surfaced as ``retained centered
        fit geometry``. Rank truncation (PR #176) removes the cause, so the
        workaround is gone and the mode is rejected at ``require_converged``.
        """
        stub = self._stub(self.LONG_RUN)
        assert self._run_gate(monkeypatch, stub) is None
        # Nothing reclassifies how the iteration actually ended.
        assert stub.converged is False
        assert stub.termination_reason == "max_iter"
```

- [ ] **Step 2: Run it to make sure it fails**

Run:
```bash
uv run pytest "tests/test_scop_efs.py::TestSCOPBoundaryStagnationAcceptance::test_a_stagnant_candidate_is_no_longer_specially_accepted" -v
```
Expected: **FAIL** with `RuntimeError: SCOP REML requires retained centered fit
geometry` — the gate still accepts the stub and lets it run on into geometry
assembly. That failure is the proof the test bites.

- [ ] **Step 3: Remove the gate from `scop_efs.py`**

Delete four regions, top to bottom (delete from the bottom up so earlier line
numbers stay valid):

1. `991-1014` — collapse the acceptance branch. Replace:

```python
    if require_converged and not result.converged:
        if not _scop_deviance_stagnated(result, context.max_pirls_iter):
            return None
        # The predicate accepted, so the window it resolved from this same cap
        # was not None. Report that window; there is no fallback to guard.
        required = _scop_stagnation_window(context.max_pirls_iter)
        logger.info(
            "SCOP %s fit accepted as a boundary solution: deviance stagnant to "
            "%.3g relative across the last %d of %d iterations "
            "(coefficient criterion cannot terminate at a log-space boundary)",
            phase,
            _STAGNANT_DEVIANCE_TOLERANCE,
            required,
            result.n_iter,
        )
        # A boundary solution is a converged mode: the deviance has stopped
        # moving and the coefficient criterion is tracking a coordinate that no
        # longer measures progress. Record that decision on the result so every
        # downstream consumer -- the fixed-lambda ``REMLResult.converged`` flag
        # and the published terminal fit among them -- sees one coherent
        # convergence state instead of a mode that was accepted here but still
        # advertises a failure. ``termination_reason`` keeps its ``"max_iter"``
        # value as the durable record of how the iteration actually ended.
        result.converged = True
```

with:

```python
    if require_converged and not result.converged:
        return None
```

2. `844-900` — delete `_scop_deviance_stagnated` entirely.
3. `828-841` — delete `_scop_stagnation_window` entirely.
4. `781-825` — delete the `# Deviance-stagnation acceptance for boundary SCOP
   coefficient modes.` comment block and the three constants
   `_STAGNANT_DEVIANCE_TOLERANCE`, `_STAGNANT_DEVIANCE_WINDOW_MAX`,
   `_STAGNANT_DEVIANCE_WINDOW_MIN`.

Leave `_record_stagnation=True` (around line 981) and the scrub (around line
1293) alone — Task 2 owns those.

- [ ] **Step 4: Run the new test to verify it passes**

Run:
```bash
uv run pytest "tests/test_scop_efs.py::TestSCOPBoundaryStagnationAcceptance::test_a_stagnant_candidate_is_no_longer_specially_accepted" -v
```
Expected: **PASS**.

- [ ] **Step 5: Replace the obsolete gate tests**

The rest of `TestSCOPBoundaryStagnationAcceptance` now references functions
that no longer exist. Delete the whole class (`tests/test_scop_efs.py:2363-2691`)
and put this in its place. Three of its tests are worth keeping and are carried
over here; the roughly twenty that tested window arithmetic and tolerance
constants go.

```python
class TestSCOPNonConvergenceIsNotSpeciallyAccepted:
    """A non-converged SCOP inner fit is rejected, whatever its deviance did.

    Item 2c retired the deviance-stagnation acceptance rule. It existed
    because ``convergence="coefficients"`` cannot terminate when a SCOP
    coefficient drifts to its log-space boundary (``exp(gamma) -> 0``): the
    coefficient keeps moving while the fit stops. PR #176 fixed that at its
    cause by truncating the unidentifiable direction out of the Newton step,
    so the boundary fit converges normally and no fit in the corpus reaches
    this path any more.

    What certifies a mode is the penalized-score check in
    ``_fit_scop_reml_mode`` -- ``_scop_mode_newton_relative`` against
    ``_scop_mode_tolerance`` -- which every accepted mode always had to pass.
    """

    # Comfortably longer and shorter than any window the retired gate used, so
    # these pin behaviour rather than a constant's exact value.
    LONG_RUN = 256
    SHORT_RUN = 4

    @staticmethod
    def _context():
        return scop_efs_module._SCOPREMLFitContext(
            dm=SimpleNamespace(p=1, group_matrices=[]),
            distribution=SimpleNamespace(),
            link=SimpleNamespace(),
            groups=[],
            y=np.array([1.0]),
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            pirls_tol=1e-6,
            max_pirls_iter=200,
            reml_penalties=[],
            convergence="coefficients",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=1.0,
            gamma_scale_data=None,
        )

    def _run_gate(self, monkeypatch, solver_result, captured=None):
        def fake_solver(**kwargs):
            if captured is not None:
                captured.update(kwargs)
            return solver_result, None, np.array([[1.0]]), {}

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", fake_solver)
        return scop_efs_module._fit_scop_reml_mode(
            self._context(),
            {"x": 1.0},
            beta_init=None,
            intercept_init=None,
            scop_state_init=None,
            phase="candidate",
            reml_iteration=1,
            require_converged=True,
        )

    @staticmethod
    def _stub(n_iter):
        """A non-converged solver result that exhausted its budget."""
        return SimpleNamespace(
            converged=False,
            termination_reason="max_iter",
            beta=np.array([0.0]),
            intercept=0.0,
            rank_info=None,
            n_iter=n_iter,
        )

    def test_a_stagnant_candidate_is_no_longer_specially_accepted(self, monkeypatch):
        """A boundary-stagnant fit is a non-convergence like any other.

        Before item 2c the gate admitted this stub: it flipped ``converged``
        to True and the fit proceeded into geometry assembly, which a bare
        stub cannot satisfy, so the failure surfaced as ``retained centered
        fit geometry``. Rank truncation (PR #176) removes the cause, so the
        workaround is gone and the mode is rejected at ``require_converged``.
        """
        stub = self._stub(self.LONG_RUN)
        assert self._run_gate(monkeypatch, stub) is None
        # Nothing reclassifies how the iteration actually ended.
        assert stub.converged is False
        assert stub.termination_reason == "max_iter"

    def test_a_short_run_candidate_is_rejected(self, monkeypatch):
        """Run length never mattered to the outcome; now it cannot."""
        stub = self._stub(self.SHORT_RUN)
        assert self._run_gate(monkeypatch, stub) is None
        assert stub.converged is False

    def test_the_inner_fit_does_not_ask_for_the_full_recorder(self, monkeypatch):
        """``record_diagnostics`` builds a forty-field row per iteration.

        It also switches on the solver's per-iteration extrema capture --
        measured at 7-16% of SCOP REML wall time. Asking for it from the inner
        fits is a performance regression, so this pins that we do not.
        """
        captured = {}
        self._run_gate(monkeypatch, self._stub(self.SHORT_RUN), captured=captured)
        assert captured.get("record_diagnostics", False) is False

    def test_a_genuinely_non_converging_fit_still_raises(self):
        """A real SCOP fit that never settles is reported as a failure.

        This quasi-separated Poisson exhausts all its PIRLS iterations with
        zero halvings and zero rejections, its deviance still moving by ~1e-3
        relative per iteration.
        """
        x = np.linspace(0, 1, 200)
        frame = pd.DataFrame({"x": x})
        response = np.where(x > 0.8, 5000.0, 0.0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": PSpline(n_knots=12, penalty="ssp", constraint=Constraint.fit.increasing)
            },
        )
        with pytest.raises(RuntimeError, match="did not converge to a coefficient mode"):
            model.fit_reml(frame, response, max_reml_iter=20)
```

- [ ] **Step 6: Run the affected test files**

Run:
```bash
uv run pytest tests/test_scop_efs.py -q -p no:randomly
```
Expected: **PASS**, with the file's collected count down by roughly twenty
(the deleted window/tolerance tests).

- [ ] **Step 7: Commit**

```bash
git add src/superglm/reml/scop_efs.py tests/test_scop_efs.py
git commit -m "Retire the SCOP deviance-stagnation acceptance rule

The rule accepted a boundary fit whose deviance had stopped moving while
convergence=\"coefficients\" kept measuring a coordinate that no longer
tracked progress. PR #176 fixed that at its cause by truncating the
unidentifiable exp(gamma) -> 0 direction out of the Newton step.

Measured across the full suite, the gate was reached only by its own
stub-driven tests -- no real fit in the corpus can drive it. A fit that
still exhausts max_iter now reports non-convergence, and the penalized-score
mode certification is the sole acceptance path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Remove the stagnation_log channel

**Files:**
- Modify: `src/superglm/reml/scop_efs.py:973-981`, `1280-1293`
- Modify: `src/superglm/solvers/pirls.py:225-245`, `280-296`
- Modify: `src/superglm/solvers/irls_direct.py:71`, `391`, `436`, `488`, `564-568`, `1260-1264`, `1988-1996`, `2557`
- Test: `tests/test_scop_efs.py`, `tests/test_scop_irls_state.py`, `tests/test_fit_transactions.py`

**Interfaces:**
- Consumes: Task 1's removal of the gate — nothing reads `stagnation_log` any more.
- Produces: `StagnationRecord` no longer exists in `superglm.solvers.pirls`;
  `PIRLSResult` has no `stagnation_log` field; `fit_irls_direct` and
  `_fit_irls_direct_once` no longer accept `_record_stagnation`.

- [ ] **Step 1: Remove the channel request and the scrub from `scop_efs.py`**

At `973-981`, delete the `_record_stagnation=True` argument to
`fit_irls_direct` together with its comment block (the comment begins
`# Supplies the per-iteration deviance history that`). Keep every other
argument in that call untouched.

At `1280-1293`, delete the `result.stagnation_log = None` line and the comment
block above it that begins `# The inner fits set ``_record_stagnation=True```.
Keep the `return result` that follows, and keep the preceding non-finite check.

- [ ] **Step 2: Remove the field and record type from `pirls.py`**

Delete the `StagnationRecord` dataclass in full — the `@dataclass(frozen=True)`
decorator at line 225, the class, and its docstring.

Delete the `stagnation_log` field at line 296 together with the 16-line comment
above it that begins `# Narrow per-iteration channel, independent of`.

- [ ] **Step 3: Remove the flag and accumulator from `irls_direct.py`**

Eight edits:

1. Line 71 — drop `StagnationRecord` from the `from ... import (...)` block.
2. Line 391 — delete `_record_stagnation: bool = False,` from the
   `fit_irls_direct` signature.
3. Line 436 — delete `_record_stagnation=_record_stagnation,` from the
   `_fit_irls_direct_once` call inside `run_once`.
4. Line 488 — delete `_record_stagnation: bool = False,` from the
   `_fit_irls_direct_once` signature.
5. Lines 564-568 — delete the `_record_stagnation : bool` docstring entry and
   its four description lines.
6. Lines 1260-1264 — delete the `stagnation_log: list[StagnationRecord] = []`
   accumulator and the three-line comment above it beginning
   `# Same convention for the narrow channel.`
7. Lines 1988-1996 — delete the guarded append:

```python
        if _record_stagnation:
            stagnation_log.append(
                StagnationRecord(
                    iteration=it + 1,
                    deviance=dev,
                    step_rejected=step_rejected,
                    step_halvings=n_halvings,
                )
            )
```

   Delete the comment immediately above it as well — it describes only the
   narrow channel, so nothing in it survives:

```python
        # Narrow per-iteration record. Written from the same loop variables as
        # the diagnostics row below and at the same point in the iteration --
        # before the non-finite-deviance break -- so the two channels always
        # carry identical values over identical iteration ranges.
```
8. Line 2557 — delete
   `stagnation_log=stagnation_log if _record_stagnation else None,` from the
   `PIRLSResult(...)` construction.

- [ ] **Step 4: Update the tests that referenced the channel**

Four edits.

**4a.** `tests/test_scop_efs.py` — delete
`TestStagnationChannelMatchesTheDiagnosticsRecorder` in full
(lines `2692-2772` in the pre-Task-1 tree; locate it by name).

**4b.** `tests/test_scop_efs.py` — in `TestSCOPREMLDoesNotPublishDiagnostics`,
delete `test_published_result_carries_no_stagnation_log` and replace the class
docstring with:

```python
    """A REML fit must not turn on a public accessor its caller never requested.

    ``fit_reml`` exposes no diagnostics parameter and the non-SCOP REML path
    records nothing, so a REML caller has never been able to ask for a
    per-iteration log. Nothing may leak one onto the published result, or a
    SCOP REML fit would carry a field no other engine populates.
    """
```

**4c.** `tests/test_scop_irls_state.py` — the test at `341-388` compares the two
channels, so its subject is gone, but it is the only place that drives
`step_halvings` and `step_rejected` to non-trivial values. Re-point it at the
surviving recorder. Rename it and replace its docstring, its
`fit_irls_direct` call, and its assertions:

```python
@pytest.mark.parametrize(
    ("alpha", "halvings", "rejected"),
    [(0.5, 1, False), (0.25, 2, False), (1.0, 0, True)],
)
def test_recorder_captures_step_quality_fields(monkeypatch, alpha, halvings, rejected) -> None:
    """The recorder must carry the step-quality fields, not just the deviance.

    Every record the SCOP corpus produces carries ``step_rejected=False`` and
    ``step_halvings=0`` -- measured: 69 records, 2 inner fits, zero of each --
    so a fixture-driven test pins the deviance and nothing else. This drives
    both fields directly instead of hoping a fixture reaches them.
    """
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()

    def forced_decision(**kwargs):
        kwargs["evaluate_state"](alpha)
        return _IRLSStepDecision(alpha, halvings, rejected, trials_attempted=halvings + 1)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", forced_decision)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=1,
        record_diagnostics=True,
    )

    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == halvings
    assert result.iteration_log[0].step_rejected is rejected
```

**4d.** `tests/test_fit_transactions.py` — delete
`test_published_stagnation_records_keep_their_fields` (lines `368-397`) in full,
and drop `StagnationRecord` from the import at line 24, leaving
`from superglm.solvers.pirls import PIRLSResult`.

Do **not** write a replacement:
`test_published_result_deeply_freezes_diagnostics_and_rank_metadata` at
`348-365` already asserts `result.iteration_log[0].deviance = -1.0` raises
`AttributeError`, which pins the same freeze property on the surviving
dataclass.

**4e.** `tests/test_scop_efs.py` — `TestIterationDiagnosticsSmallSample`'s
docstring narrates the gate's history and goes stale with it. Keep every test
in the class untouched (the small-n recorder fix it pins is unrelated and still
live); replace only the docstring with:

```python
    """The diagnostics recorder must survive n <= 5.

    ``k = min(5, n)`` makes ``k == n`` for small samples, and numpy requires
    ``-n <= kth < n``, so the bottom-k partition needs ``k - 1``. The bug is
    latent while diagnostics are opt-in, and a caller that turns the recorder
    on unconditionally converts it into a crash on a default-argument REML
    fit. The opt-in test below is what keeps the fix pinned, and the REML test
    below keeps small-n SCOP fits themselves covered.
    """
```

- [ ] **Step 5: Confirm no reference survives**

Run:
```bash
grep -rn "stagnation_log\|StagnationRecord\|_record_stagnation" src/ tests/ --include=*.py
```
Expected: **no output**. Any hit is a missed edit.

- [ ] **Step 6: Run the affected test files**

Run:
```bash
uv run pytest tests/test_scop_efs.py tests/test_scop_irls_state.py tests/test_fit_transactions.py -q -p no:randomly
```
Expected: **PASS**.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/reml/scop_efs.py src/superglm/solvers/pirls.py \
        src/superglm/solvers/irls_direct.py tests/test_scop_efs.py \
        tests/test_scop_irls_state.py tests/test_fit_transactions.py
git commit -m "Remove the stagnation_log channel the retired gate fed on

StagnationRecord, PIRLSResult.stagnation_log and the _record_stagnation
flag existed solely to supply the deviance-stagnation gate; the field's own
note said so and scrubbed it before publication so it never became a
solver-dependent public surface. With the gate gone they have no consumer.

This also retires the hazard documented on the field -- that routing a new
caller through fit_pirls would silently downgrade to raising, because
fit_pirls populates no stagnation_log -- along with its cause. Each SCOP
inner fit stops paying for the per-iteration append.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Verify the removal and record the outcome

**Files:**
- Modify: `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md:208-213`
- Create: `/home/max/.claude/jobs/f532cf83/tmp/gate_probe_after.py` (throwaway, not committed)

**Interfaces:**
- Consumes: Tasks 1 and 2 complete.
- Produces: nothing consumed by later tasks. This task gates the PR.

- [ ] **Step 1: Run the full suite**

Run:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q -p no:randomly
```
Expected: **PASS**. The passed count drops from 5133 by exactly the number of
deleted tests and no more; skipped stays at 84. If any test fails, stop — a
failure here means a real fit did depend on the gate and the spec's central
evidence is wrong. Report it rather than working around it.

- [ ] **Step 2: Confirm the boundary regression specifically**

Run:
```bash
uv run pytest "tests/test_scop_efs.py::TestMultiSCOPIntegration::test_stored_objective_reproduction_multi_scop" -v
```
Expected: **PASS**. This is the genuine log-space-boundary fit that leaves the
identified Hessian near-singular — the load-bearing evidence that #176 carries
what the gate used to.

- [ ] **Step 3: Confirm no gate path survives**

The probe from the investigation patched `_scop_deviance_stagnated`, which no
longer exists, so this run just proves the symbol is gone everywhere:

```bash
uv run python -c "
import superglm.reml.scop_efs as se
import superglm.solvers.pirls as p
import superglm.solvers.irls_direct as d
import inspect
for name in ('_scop_deviance_stagnated', '_scop_stagnation_window',
             '_STAGNANT_DEVIANCE_TOLERANCE', '_STAGNANT_DEVIANCE_WINDOW_MAX',
             '_STAGNANT_DEVIANCE_WINDOW_MIN'):
    assert not hasattr(se, name), name
assert not hasattr(p, 'StagnationRecord')
assert 'stagnation_log' not in {f for f in p.PIRLSResult.__dataclass_fields__}
assert '_record_stagnation' not in inspect.signature(d.fit_irls_direct).parameters
print('gate fully removed')
"
```
Expected: `gate fully removed`.

- [ ] **Step 4: Run the CI gates**

Run:
```bash
uv run ruff check && uv run ruff format --check && uv run ty check 2>&1 | tail -3
```
Expected: ruff clean on both; `ty` at **806 or fewer** (the master baseline).
It must not increase.

- [ ] **Step 5: Record the outcome in the roadmap**

In `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md`, replace the
`**Status 2026-07-31:**` paragraph at lines `208-213` with:

```markdown
**Status 2026-07-31:** Item 2 is implemented (rank truncation moved to the
factor, resolved range carried through the determinant and curvature, factor
built only when the Gram cannot resolve the step). **Item 2c is done:** the
stagnation acceptance rule and the private `stagnation_log` channel behind it
are removed. The evidence was that across the full suite the gate was reached
only by its own stub-driven tests -- no real fit in the corpus could drive it.
**Item 2b closes with it:** the penalized-score certification it asked for
already exists (`_scop_mode_newton_relative` gated in `_fit_scop_reml_mode`)
and is now the sole acceptance path, so a separate gradient-norm test would be
redundant. Design: `docs/superpowers/specs/2026-07-31-scop-stagnation-gate-removal-design.md`.
```

- [ ] **Step 6: Commit the roadmap update**

`docs/superpowers/` is gitignored, so this needs `-f`:

```bash
git add -f docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md
git commit -m "Record item 2c done and item 2b closed with it

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git log --stat -1
```
Expected: the `git log --stat` output lists the roadmap file. If it does not,
the gitignore silently dropped it — re-add with `-f` and amend.

- [ ] **Step 7: Push and open the PR**

```bash
git push -u origin refactor/retire-scop-stagnation-gate
```

Then open a PR whose body states, in this order: that the version stays
`0.17.0` and folds into the existing unpublished candidate; the measured
evidence (gate reached only by its own tests, all stub-driven); the behaviour
change (a SCOP fit exhausting `max_iter` on a stagnant deviance now reports
non-convergence instead of being accepted, unreachable in the corpus but real);
that `stagnation_log`/`_record_stagnation` were private and scrubbed before
publication, so there is no public-contract impact; and that item 2b closes
with 2c.

Do **not** use `gh pr edit --body-file` — it exits 1 here and silently
discards the body. Create with `gh pr create --body-file`, and if the body
needs a later edit use:
```bash
gh api -X PATCH repos/:owner/:repo/pulls/<N> -F body=@body.md
```
verifying afterwards with a string that exists only in the new body.

Master requires all 9 status checks green (`type-check`, `quality`, `docs`,
`frontend`, `browser`, and four `Python 3.14 · non-browser regression suite ·
balanced A-D`). Approvals required: 0.

**Do not merge, tag, or publish.** Report back for review.
