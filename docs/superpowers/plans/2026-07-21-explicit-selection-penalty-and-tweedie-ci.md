# Explicit Selection Penalty and Tweedie CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give selection penalties one explicit meaning across every fitting path and let callers request an installed Tweedie profile confidence interval through `estimate_p(ci_alpha=...)`.

**Architecture:** Constructor intent remains immutable and may be `None`, `"auto"`, or a finite non-negative number. A shared boundary resolver converts that intent into an attempt-owned numeric penalty before any solver runs; REML has a separate guard that accepts only disabled selection. Tweedie CI computation remains opt-in and runs inside the existing profile-fit transaction immediately before the independently owned installed profile copy is prepared.

**Tech Stack:** Python 3.10–3.14, NumPy, pytest, Ruff, uv, existing transactional fit workspaces and Tweedie profile-likelihood engine.

---

## File responsibility map

- `src/superglm/model/base.py`: validate public selection configuration and resolve ordinary-fit attempt values.
- `src/superglm/model/api.py`: expose the updated types, defaults, and `ci_alpha` argument.
- `src/superglm/model/fit_ops.py`: invoke ordinary and REML resolution at their entry boundaries.
- `src/superglm/model/profile_ops.py`: validate and execute explicitly requested Tweedie CI work before publication.
- `src/superglm/profiling/tweedie.py`: reuse ordinary selection resolution during fixed-power fits and provide one alpha validator.
- `src/superglm/profiling/nb.py`: reuse ordinary selection resolution during theta profiling.
- `src/superglm/penalties/*.py`: permit the explicit `"auto"` constructor intent in built-in type annotations and documentation while keeping solver-time values numeric.
- `src/superglm/sklearn.py`: align wrapper resolution and annotations with the core API.
- `tests/test_api.py`, `tests/test_fit_ownership.py`: selection configuration and ordinary-fit resolution.
- `tests/test_reml_efs.py`: REML rejection and no-work-before-rejection evidence.
- `tests/test_tweedie_profile.py`, `tests/test_nb2.py`: profile-fit selection semantics, CI publication, rollback, and lazy-path invariants.
- `docs/guide/*.md` and API docstrings: user-facing examples and the distinction between configured and fitted values.

### Task 1: Define and validate selection-penalty intent

**Files:**
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/penalties/base.py`
- Modify: `src/superglm/penalties/group_lasso.py`
- Modify: `src/superglm/penalties/sparse_group_lasso.py`
- Modify: `src/superglm/penalties/group_elastic_net.py`
- Modify: `src/superglm/penalties/ridge.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing constructor and assignment tests**

Add tests demonstrating the complete public domain:

```python
@pytest.mark.parametrize("value", [None, 0, 0.0])
def test_disabled_selection_penalty_intent_is_preserved(value):
    model = SuperGLM(selection_penalty=value)
    assert model.selection_penalty == value


def test_explicit_auto_selection_penalty_intent_is_preserved():
    model = SuperGLM(selection_penalty="auto")
    assert model.selection_penalty == "auto"


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, np.nan, True, "automatic"])
def test_invalid_selection_penalty_is_rejected(value):
    with pytest.raises(ValueError, match="selection_penalty"):
        SuperGLM(selection_penalty=value)
```

Cover the property setter and a directly supplied built-in penalty object as well, proving that caller-owned objects remain unchanged.

- [ ] **Step 2: Run the new tests and confirm the intended failures**

Run:

```bash
rtk test uv run pytest tests/test_api.py -q
```

Expected: failures for `"auto"` and invalid-value validation because the current implementation accepts or mishandles them.

- [ ] **Step 3: Add one normalization function and update public types**

In `model/base.py`, add a side-effect-free normalizer with this contract:

```python
SelectionPenalty = float | Literal["auto"] | None


def normalize_selection_penalty(value: object) -> SelectionPenalty:
    if value is None or value == "auto":
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_SELECTION_PENALTY_ERROR)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_SELECTION_PENALTY_ERROR) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(_SELECTION_PENALTY_ERROR)
    return numeric
```

Avoid `value == "auto"` on arbitrary NumPy arrays by checking `isinstance(value, str)` first. Call the normalizer from `resolve_penalty()` for shorthand-created penalties and for a defensively copied direct penalty object. Call it from the `selection_penalty` setter before replacing configuration.

Update built-in `lambda1` and `SuperGLM.selection_penalty` annotations/docstrings to accept `Literal["auto"]`. Do not allow an unresolved string to reach `prox()`, `eval()`, `penalty_can_zero_groups()`, IRLS, or PIRLS.

- [ ] **Step 4: Run focused validation**

```bash
rtk test uv run pytest tests/test_api.py tests/test_fit_ownership.py -q
rtk ruff check src/superglm/model/base.py src/superglm/model/api.py src/superglm/penalties/ tests/test_api.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the configuration boundary**

```bash
rtk git add src/superglm/model/base.py src/superglm/model/api.py src/superglm/penalties tests/test_api.py tests/test_fit_ownership.py
rtk git commit -m "Make selection penalty intent explicit"
```

### Task 2: Resolve ordinary and profile fits consistently

**Files:**
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `src/superglm/profiling/nb.py`
- Test: `tests/test_api.py`
- Test: `tests/test_fit_ownership.py`
- Test: `tests/test_tweedie_profile.py`
- Test: `tests/test_nb2.py`

- [ ] **Step 1: Replace tests that encode `None = auto` and add explicit cases**

Change the existing ordinary-fit ownership tests to assert:

```python
model = SuperGLM(selection_penalty=None, features={"x": Numeric(), "z": Numeric()})
model.fit(X, y, sample_weight=weights)
assert model.selection_penalty is None
assert model.selection_penalty_ == pytest.approx(0.0)
```

Add the corresponding explicit automatic case:

```python
model = SuperGLM(selection_penalty="auto", features={"x": Numeric(), "z": Numeric()})
model.fit(X, y, sample_weight=weights)
assert model.selection_penalty == "auto"
assert np.isfinite(model.selection_penalty_)
assert model.selection_penalty_ > 0.0
```

Add a categorical-only regression asserting that a default ordinary fit has `selection_penalty_ == 0.0`, uses the direct solver route, and has EDF equal to `1 + rank_info.data.rank` within numerical roundoff.

For Tweedie and NB profile tests, spy on the coefficient solver and assert the numeric penalty received by each candidate is zero for `None`, zero for `0.0`, and positive for `"auto"`.

- [ ] **Step 2: Run the selection-resolution tests and verify they fail**

```bash
rtk test uv run pytest tests/test_api.py tests/test_fit_ownership.py tests/test_nb2.py tests/test_tweedie_profile.py -q -k "selection_penalty or auto_calibrate or categorical"
```

Expected: `None` cases still resolve positively in core ordinary and profile fits.

- [ ] **Step 3: Implement one attempt-local ordinary resolver**

Add a helper in `model/base.py` that mutates only the private attempt penalty:

```python
def resolve_selection_penalty_for_fit(model, penalty, y, weights) -> float:
    intent = normalize_selection_penalty(penalty.lambda1)
    if intent == "auto":
        resolved = float(compute_lambda_max(model, y, weights) * 0.1)
    elif intent is None:
        resolved = 0.0
    else:
        resolved = float(intent)
    penalty.lambda1 = resolved
    return resolved
```

Replace each duplicated `if penalty.lambda1 is None: ...` block in ordinary
`fit()`, Tweedie fixed-power preparation, and NB theta profiling with this helper.
Ensure resolution happens only after the design exists for `"auto"` but before
solver dispatch or `penalty_can_zero_groups()` runs.

Do not change `fit_path()`: it is explicitly controlled by its validated lambda
sequence and must continue to publish the terminal numeric path value.

- [ ] **Step 4: Run focused ordinary/profile tests**

```bash
rtk test uv run pytest tests/test_api.py tests/test_fit_ownership.py tests/test_nb2.py tests/test_tweedie_profile.py -q -k "selection_penalty or auto_calibrate or categorical"
rtk test uv run pytest tests/test_irls_direct.py tests/test_pirls_composite_optimizer.py -q
```

Expected: all tests pass; no unresolved `"auto"` reaches either solver.

- [ ] **Step 5: Commit unified fit resolution**

```bash
rtk git add src/superglm/model/base.py src/superglm/model/fit_ops.py src/superglm/profiling/tweedie.py src/superglm/profiling/nb.py tests/test_api.py tests/test_fit_ownership.py tests/test_tweedie_profile.py tests/test_nb2.py
rtk git commit -m "Resolve selection penalties consistently"
```

### Task 3: Make REML's no-selection contract unconditional

**Files:**
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/model/api.py`
- Test: `tests/test_reml_efs.py`
- Test: `tests/test_reml.py`
- Test: `tests/test_multi_penalty.py`

- [ ] **Step 1: Add REML acceptance and rejection tests**

Add parameterized tests proving `None` and `0.0` both fit successfully, while
`"auto"` and positive finite values fail before design construction:

```python
@pytest.mark.parametrize("selection_penalty", ["auto", 1e-8, 0.01])
def test_reml_rejects_selection_before_design_work(monkeypatch, selection_penalty):
    model = SuperGLM(selection_penalty=selection_penalty, features={"x": Numeric()})
    build_called = False

    def unexpected_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("design work must not start")

    monkeypatch.setattr(base, "model_build_design_matrix", unexpected_build)
    with pytest.raises(ValueError, match="does not support selection penalties"):
        model.fit_reml(X, y)
    assert build_called is False
```

Retain tests for `select=True`: this spline option is REML-managed shrinkage and
must continue to work when selection is disabled.

- [ ] **Step 2: Run the REML contract tests and verify `"auto"` currently breaks differently**

```bash
rtk test uv run pytest tests/test_reml_efs.py tests/test_reml.py -q -k "selection_penalty"
```

- [ ] **Step 3: Implement a pre-work REML guard**

Add a helper that normalizes configuration without needing data:

```python
def resolve_selection_penalty_for_reml(penalty) -> float:
    intent = normalize_selection_penalty(penalty.lambda1)
    if intent == "auto" or (intent is not None and intent > 0.0):
        raise ValueError(
            "fit_reml() does not support selection penalties; use None or 0.0, "
            "or use fit()/fit_path() for sparse selection."
        )
    penalty.lambda1 = 0.0
    return 0.0
```

Invoke it immediately after obtaining the private attempt penalty and before NB
theta profiling, feature auto-detection, design construction, or any REML work.
Remove the later contextual `None -> 0` mutation and redundant positive check.
Update the `fit_reml()` docstring to explain that `select=True` is distinct from a
selection penalty.

- [ ] **Step 4: Run REML-focused correctness tests**

```bash
rtk test uv run pytest tests/test_reml_efs.py tests/test_reml.py tests/test_multi_penalty.py tests/test_scop_efs.py -q -k "selection_penalty or select_true"
```

Expected: disabled configurations pass and all nonzero/automatic selection requests fail before work.

- [ ] **Step 5: Commit the REML boundary**

```bash
rtk git add src/superglm/model/base.py src/superglm/model/fit_ops.py src/superglm/model/api.py tests/test_reml_efs.py tests/test_reml.py tests/test_multi_penalty.py
rtk git commit -m "Reject selection penalties from REML"
```

### Task 4: Add explicit Tweedie CI orchestration

**Files:**
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/profile_ops.py`
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add failing public CI tests**

Extend `TestEstimatePFitMode` with deterministic monkeypatched profile results.
Cover all of these outcomes:

```python
returned = model.estimate_p(X, y, ci_alpha=0.05)
installed = model._tweedie_profile_result

assert returned._ci_cache[0.05] == pytest.approx(expected_interval)
assert installed._ci_cache[0.05] == pytest.approx(expected_interval)
assert returned._ci_cache is not installed._ci_cache
assert returned._ci_cache[0.05] is not installed._ci_cache[0.05]
assert model.summary(alpha=0.05)._info["tweedie_p_ci_status"] == "available"
assert model.summary(alpha=0.10)._info["tweedie_p_ci_status"] == "not computed"
```

Retain and strengthen the current lazy test: omitted `ci_alpha` must call no CI
method, add no fixed-power evaluations, and leave both caches empty.

Add a rollback test that starts from a fitted model, makes the requested CI raise,
and asserts the model's fit revision, family, result identities, profile result,
summary cache, and predictions are unchanged.

Add invalid alpha cases (`0`, `1`, non-finite, Boolean, nonscalar) and a Pearson
case, asserting rejection before profile or final-fit work begins.

- [ ] **Step 2: Run the new CI tests and confirm failure**

```bash
rtk test uv run pytest tests/test_tweedie_profile.py tests/test_metrics.py -q -k "ci_alpha or lazy_about_ci or profile_ci"
```

Expected: `ci_alpha` is currently forwarded through `**kwargs` to the lower-level profiler or absent from the public signature, and installed summary remains unavailable.

- [ ] **Step 3: Extract a reusable alpha validator**

In `profiling/tweedie.py`, extract the alpha-only portion of
`_validate_profile_ci_inputs()`:

```python
def _validate_profile_ci_alpha(alpha: object) -> float:
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    try:
        value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return value
```

Reject nonscalar arrays without emitting ambiguous truth-value errors. Reuse it
inside `_validate_profile_ci_inputs()` so CI validation has one source of truth.

- [ ] **Step 4: Implement transactional `ci_alpha`**

Add the explicit keyword to both public layers:

```python
def estimate_p(..., *, fit_mode="fit", phi_method="mle", method="auto",
               ci_alpha: float | None = None, progress_callback=None, **kwargs):
```

At entry, normalize `ci_alpha` when non-`None` and reject a requested CI with
`phi_method != "mle"` before input validation or profiling. Leave `None` untouched.

After `_synchronize_tweedie_profile_refit(final_model, y, result)` and compact-state
preparation, but before `_installed_tweedie_profile_copy(result)`, execute:

```python
if resolved_ci_alpha is not None:
    result.ci(alpha=resolved_ci_alpha)
```

Then prepare the installed copy normally. Do not share `_ci_cache` or
`_ci_details_cache`; the existing deep-copy publication must remain authoritative.
Do not make `summary()` evaluate a CI.

- [ ] **Step 5: Run CI transaction and summary tests**

```bash
rtk test uv run pytest tests/test_tweedie_profile.py tests/test_metrics.py -q -k "ci_alpha or lazy_about_ci or profile_ci or summary"
```

Expected: explicit intervals appear in matching summaries, lazy calls remain free, and failure publishes nothing.

- [ ] **Step 6: Commit explicit CI support**

```bash
rtk git add src/superglm/model/api.py src/superglm/model/profile_ops.py src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py tests/test_metrics.py
rtk git commit -m "Add explicit Tweedie profile CI requests"
```

### Task 5: Align sklearn wrappers and user documentation

**Files:**
- Modify: `src/superglm/sklearn.py`
- Modify: `README.md`
- Modify: `docs/guide/families.md`
- Modify: `docs/guide/fitting.md`
- Modify: `docs/notebooks/tweedie_profile_estimation.ipynb` through a focused JSON-aware prose edit; do not regenerate outputs or metadata
- Test: `tests/test_sklearn.py`
- Test: `tests/test_tweedie_profile_docs.py`

- [ ] **Step 1: Add wrapper contract tests**

Assert that wrapper defaults remain disabled, `"auto"` explicitly upgrades a
missing penalty to group lasso, positive numerics retain existing behavior, and
invalid settings fail through the same core validator.

- [ ] **Step 2: Update wrapper resolution without duplicating semantics**

Change `_resolve_wrapper_penalty()` so it performs only wrapper-specific penalty
type selection:

```python
if penalty is None and selection_penalty == "auto":
    penalty = "group_lasso"
elif penalty is None and isinstance(selection_penalty, Real) and selection_penalty > 0:
    penalty = "group_lasso"
```

Do not convert `None` to `0.0`; the core already defines both as disabled. Avoid
numeric comparison against strings and defer domain validation to the shared core
boundary.

- [ ] **Step 3: Correct user-facing examples**

Document these explicit examples:

```python
SuperGLM()                                  # no sparse selection
SuperGLM(selection_penalty="auto")         # explicit calibration
SuperGLM(selection_penalty=0.05)           # fixed strength
model.estimate_p(X, y, ci_alpha=0.05)      # CI cached for model.summary(alpha=0.05)
```

Remove claims that `result.ci()` populates the independently owned model summary
cache. State that `result.ci()` remains valid for the detached result, while
`ci_alpha` is the model-publication convenience.

- [ ] **Step 4: Run wrapper and docs tests**

```bash
rtk test uv run pytest tests/test_sklearn.py tests/test_tweedie_profile_docs.py -q
rtk uv sync --python /usr/bin/python3.14 --group docs --extra plotting
rtk test uv run mkdocs build --strict
rtk ruff check src/superglm/sklearn.py tests/test_sklearn.py
```

- [ ] **Step 5: Commit wrapper and documentation alignment**

```bash
rtk git add src/superglm/sklearn.py tests/test_sklearn.py docs README.md
rtk git commit -m "Document explicit penalty and profile CI behavior"
```

### Task 6: Verify correctness, performance invariants, and packaging

**Files:**
- Modify only if verification exposes a defect in files already in scope

- [ ] **Step 1: Prove the lazy CI route still performs no extra work**

Run the focused counter test and record that objective calls, profile evaluation
counts, coefficient-fit counts, and both CI caches remain unchanged when
`ci_alpha=None`:

```bash
rtk test uv run pytest tests/test_tweedie_profile.py::TestEstimatePFitMode::test_public_estimate_is_lazy_about_ci_and_profile_evaluations -q
```

- [ ] **Step 2: Run all focused affected suites**

```bash
rtk test uv run pytest tests/test_api.py tests/test_fit_ownership.py tests/test_fit_transactions.py tests/test_reml_efs.py tests/test_reml.py tests/test_multi_penalty.py tests/test_nb2.py tests/test_tweedie_profile.py tests/test_metrics.py tests/test_sklearn.py -q
```

- [ ] **Step 3: Run static and dependency checks**

```bash
rtk ruff check src/ tests/
rtk ruff format --check src/ tests/
rtk uv lock --check
rtk uv pip check
rtk test uv run ty check src/superglm
```

Treat unrelated pre-existing `ty` findings according to the repository policy;
all changed-file findings are blockers.

- [ ] **Step 4: Run the complete repository gates on normal Python 3.14**

```bash
rtk test uv run python run_test.py
rtk test uv run pytest tests/ -q -m "not slow"
rtk test uv run pytest tests/ -q
```

Record exact counts and duration. Do not weaken tests, tolerances, fixtures, or
markers to pass.

- [ ] **Step 5: Build and inspect artifacts**

```bash
rtk uv build
rtk uv run python scripts/verify_release_artifacts.py dist
```

Confirm wheel and sdist contain the changed package source, intentionally exclude
repository-only documentation, and contain no worktree-local files.

- [ ] **Step 6: Review the final diff and commit any verification-only corrections**

```bash
rtk git diff origin/master...HEAD --check
rtk git diff origin/master...HEAD --stat
rtk git status --short
```

If verification required a scoped correction, commit it separately with an
imperative message. Otherwise leave the already verified conceptual commits
unchanged.
