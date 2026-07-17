# Tweedie Consumer Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Tweedie repair with early validation, honest reporting and plots, bounded state retention, serialization-safe results, updated documentation, and requirement-level verification.

**Architecture:** Consume the numerical kernel, certified profile records, inference kinds, effective bounds, and staged public installation from the preceding plans. Replace three bound context callbacks with one detachable evaluator lifecycle, cache requested CI/curve data before release, and project one shared status model into summaries and plots. Validate all public controls before model construction or optimizer dispatch.

**Tech Stack:** Python dataclasses/pickle/gc, NumPy, pandas, Matplotlib, pytest, Ruff, mypy.

---

## File map

- Modify `src/superglm/_tweedie_numerics.py`: reusable numeric vector and integer/control validators.
- Modify `src/superglm/distributions.py`: consume strict scalar validation.
- Modify `src/superglm/profiling/tweedie.py`: public search/input validation, detachable evaluator, eager CI, pickle migration, and plot delegation.
- Create `src/superglm/profiling/_tweedie_plotting.py`: cache-only trace plot and serializable dense-curve construction.
- Modify `src/superglm/model/api.py` and `src/superglm/model/profile_ops.py`: `eager_ci_alpha` lifecycle.
- Modify `src/superglm/profiling/_reporting.py`, `src/superglm/model/report_ops.py`, and editor summary payloads: shared honest profile status.
- Create `tests/test_tweedie_validation.py`, `tests/test_tweedie_retention.py`, `tests/test_tweedie_serialization.py`, `tests/test_tweedie_reporting.py`, and `tests/test_tweedie_plotting.py`.
- Modify `tests/test_fit_state_retention.py`, `tests/test_profile_ci.py`, `tests/test_editor.py`, and `tests/test_tweedie_profile_docs.py`.
- Modify `docs/guide/families.md` and `docs/notebooks/tweedie_profile_estimation.ipynb`.

### Task 1: Validate public arrays and search controls before side effects

**Files:**
- Modify: `src/superglm/_tweedie_numerics.py`
- Modify: `src/superglm/profiling/tweedie.py:362-416,4174-4337,4357-4405`
- Modify: `src/superglm/model/api.py:668-715`
- Create: `tests/test_tweedie_validation.py`

- [ ] **Step 1: Write failing early-validation tests**

```python
@pytest.mark.parametrize(
    "offset",
    [
        np.zeros((12, 1)),
        np.zeros(11),
        np.r_[np.zeros(11), np.nan],
        np.ones(12, dtype=np.complex128),
        ["0"] * 12,
    ],
)
def test_bad_offset_fails_before_profile_context(monkeypatch, profile_problem, offset):
    model, X, y = profile_problem

    def unexpected(*args, **kwargs):
        raise AssertionError("profile context must not be built")

    monkeypatch.setattr(tweedie_module, "_build_profile_context", unexpected)
    with pytest.raises((TypeError, ValueError), match="offset"):
        model.estimate_p(X, y, offset=offset)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p_bounds": (1.9, 1.1)}, "p_bounds"),
        ({"p_bounds": (1.0, 1.9)}, "p_bounds"),
        ({"grid": np.array([[1.4, 1.5]])}, "grid"),
        ({"grid": np.array([1.4, np.nan])}, "grid"),
        ({"xatol": 0.0}, "xatol"),
        ({"maxiter": True}, "maxiter"),
        ({"n_grid": 1}, "n_grid"),
        ({"n_grid_coarse": 1.5}, "n_grid_coarse"),
        ({"trace_callback": 42}, "trace_callback"),
    ],
)
def test_search_controls_fail_before_context(monkeypatch, profile_problem, kwargs, match):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_tweedie_p(model, X, y, **kwargs)
```

- [ ] **Step 2: Demonstrate current indirect/raw failures**

Run: `rtk pytest tests/test_tweedie_validation.py`

Expected: malformed offsets reach solver/design construction and search controls expose raw
NumPy/SciPy exceptions or are silently coerced.

- [ ] **Step 3: Add reusable validators**

```python
def normalize_numeric_vector(value, *, name, length=None, positive=False, nonnegative=False):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a one-dimensional real numeric array") from error
    if raw.ndim != 1 or np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a one-dimensional real numeric array")
    result = np.array(raw, dtype=np.float64, copy=True)
    if length is not None and result.size != length:
        raise ValueError(f"{name} must have length {length}, got {result.size}")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def normalize_positive_int(value, *, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result
```

Add corresponding finite-positive real, boolean, optional-callable, bounds, and grid helpers.
`grid` is nonempty, one-dimensional, finite, strictly inside `(1, 2)`, and contains at least one
point; generated grid counts require at least two. Validate `X/y/weight/offset` row agreement before
feature auto-detection and validate all search controls before context construction.

- [ ] **Step 4: Run public validation suites and commit**

Run: `rtk pytest tests/test_tweedie_validation.py tests/test_tweedie_profile.py -k 'invalid or offset or bounds or grid or control'`

Expected: stable `TypeError`/`ValueError` messages and no downstream calls for invalid input.

```bash
rtk git add src/superglm/_tweedie_numerics.py src/superglm/profiling/tweedie.py src/superglm/model/api.py tests/test_tweedie_validation.py tests/test_tweedie_profile.py
rtk git commit -m "Validate Tweedie profile inputs early"
```

### Task 2: Replace bound context callbacks with one detachable evaluator

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:2568-3079,3237-3724,3780-3871`
- Create: `tests/test_tweedie_retention.py`

- [ ] **Step 1: Add failing callback and context reachability tests**

```python
def _reachable_objects(root):
    queue = [root]
    seen = set()
    while queue:
        value = queue.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        for child in gc.get_referents(value):
            if isinstance(child, (types.ModuleType, type, types.CodeType)):
                continue
            queue.append(child)


def test_released_profile_retains_no_context_or_training_rows(released_profile_model):
    model, X, y = released_profile_model
    result = model.estimate_p(X, y, method="grid", grid=np.array([1.45, 1.55]))
    reachable = tuple(_reachable_objects(model))
    assert result._evaluator is None
    assert not any(isinstance(value, (_ProfileContext, _ProfileContextREML)) for value in reachable)
    assert not any(isinstance(value, pd.DataFrame) and len(value) == len(X) for value in reachable)
    assert not any(
        isinstance(value, np.ndarray)
        and value.ndim > 0
        and value.shape[0] == len(X)
        and value is not result.search_trace
        for value in reachable
    )


def test_trace_callback_is_collectable_after_profile(retained_profile_problem):
    callback = RecordingCallback()
    reference = weakref.ref(callback)
    estimate_tweedie_p(*retained_profile_problem, trace_callback=callback, method="grid", grid=[1.5])
    del callback
    gc.collect()
    assert reference() is None
```

- [ ] **Step 2: Verify current `_objective.__self__` retains the context**

Run: `rtk pytest tests/test_tweedie_retention.py -k 'context or callback'`

Expected: a `_ProfileContext` and row-scale arrays remain reachable; callback weakref is alive.

- [ ] **Step 3: Add one evaluator object and frozen count**

```python
@dataclass
class _TweedieProfileEvaluator:
    context: _ProfileContext | _ProfileContextREML

    def evaluate(self, p, *, source=""):
        return self.context.evaluate(float(p), source=source)

    def count(self):
        return self.context.evaluation_count()

    def record(self, p):
        return self.context.evaluation_record(float(p))

    def evaluate_curve(self, points):
        temporary = copy.deepcopy(self.context)
        temporary.trace_callback = None
        temporary._evaluation_cache.clear()
        if hasattr(temporary, "warm_beta"):
            temporary.warm_beta = None
        if hasattr(temporary, "warm_intercept"):
            temporary.warm_intercept = None
        return tuple(
            (float(p), float(temporary.evaluate(float(p), source="profile_curve")))
            for p in points
        )

    def detach(self):
        self.context.trace_callback = None
```

Replace `_objective`, `_evaluation_count`, and `_evaluation_record` with `_evaluator` and
`_frozen_evaluation_count`. Result methods call evaluator methods only when it is present. Clear
`ctx.trace_callback` immediately after search finalization, before attaching the evaluator.

Add `TweedieProfileResult.detach_evaluator()` that freezes the count, clears context callback,
sets `_evaluator=None`, and is idempotent.

- [ ] **Step 4: Run and commit**

Run: `rtk pytest tests/test_tweedie_retention.py tests/test_tweedie_profile.py -k 'evaluation or callback or context or trace'`

Expected: context lifecycle tests pass while retained results still support lazy evaluation.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_retention.py tests/test_tweedie_profile.py
rtk git commit -m "Detach Tweedie profile evaluators"
```

### Task 3: Add eager CI support for released models

**Files:**
- Modify: `src/superglm/model/api.py:668-715`
- Modify: `src/superglm/model/profile_ops.py:16-72`
- Modify: `src/superglm/profiling/tweedie.py:2729-2846`
- Modify: `tests/test_tweedie_retention.py`
- Modify: `tests/test_fit_state_retention.py`

- [ ] **Step 1: Add failing eager-cache tests**

```python
def test_released_result_returns_eager_interval(released_profile_problem):
    model, X, y = released_profile_problem
    result = model.estimate_p(X, y, eager_ci_alpha=0.05)
    cached = result._ci_cache[0.05]
    assert result._evaluator is None
    assert result.ci(0.05) is cached
    with pytest.raises(RuntimeError, match="eager_ci_alpha"):
        result.ci(0.10)


def test_invalid_eager_alpha_fails_before_profile_context(profile_problem, monkeypatch):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **kwargs: pytest.fail("context built"),
    )
    with pytest.raises(ValueError, match="eager_ci_alpha"):
        model.estimate_p(X, y, eager_ci_alpha=1.0)
```

- [ ] **Step 2: Verify the option is currently unsupported**

Run: `rtk pytest tests/test_tweedie_retention.py -k eager`

Expected: `eager_ci_alpha` is rejected or forwarded into an unrelated low-level call.

- [ ] **Step 3: Implement the lifecycle**

Add keyword-only `eager_ci_alpha: float | None = None` to `SuperGLM.estimate_p` and
`profile_ops.estimate_p`. Validate `0 < alpha < 1`. After a certified profile is found but before
staged state release, call `result.ci(alpha)` and cache its details. After staged synchronization,
call `result.detach_evaluator()` when the caller's retention setting is false.

Change `ci()` ordering so a valid cached interval is returned before evaluator availability is
checked. If uncached and detached, raise:

```python
raise RuntimeError(
    "Tweedie profile fit state was released; pass eager_ci_alpha=<alpha> "
    "to estimate_p(...) or construct the model with retain_fit_state=True."
)
```

- [ ] **Step 4: Verify pickle-size and reachability scaling**

Fit released models at 100, 1,000, and 10,000 rows. Assert serialized size growth from 100 to 10,000
is below 2.0x and no row-scale array is reachable. Retained models may scale with rows.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_tweedie_retention.py tests/test_fit_state_retention.py`

Expected: eager CI, detachment, reachability, and size tests pass.

```bash
rtk git add src/superglm/model/api.py src/superglm/model/profile_ops.py src/superglm/profiling/tweedie.py tests/test_tweedie_retention.py tests/test_fit_state_retention.py
rtk git commit -m "Release Tweedie profiling state safely"
```

### Task 4: Make finalized results serialization-safe

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:2680-2727`
- Create: `tests/test_tweedie_serialization.py`
- Modify: `tests/test_tweedie_profile.py:5229-5307`

- [ ] **Step 1: Add an intentionally unpickleable callback regression**

```python
class UnpickleableCallback:
    def __call__(self, row):
        return None

    def __reduce__(self):
        raise TypeError("callback must not be serialized")


def test_callback_bearing_profile_model_round_trips(retained_profile_problem):
    model, X, y = retained_profile_problem
    result = model.estimate_p(
        X,
        y,
        method="grid",
        grid=np.array([1.4, 1.5]),
        trace_callback=UnpickleableCallback(),
        eager_ci_alpha=0.05,
    )
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    restored_result = restored._tweedie_profile_result
    np.testing.assert_allclose(restored.predict(X), model.predict(X))
    assert restored_result._evaluator is None
    assert restored_result.ci(0.05) == result.ci(0.05)
```

- [ ] **Step 2: Demonstrate current pickle follows the retained callback**

Run: `rtk pytest tests/test_tweedie_serialization.py -k callback_bearing`

Expected: `TypeError: callback must not be serialized`.

- [ ] **Step 3: Implement explicit result pickle state**

```python
def __getstate__(self):
    state = self.__dict__.copy()
    evaluator = state.get("_evaluator")
    if evaluator is not None:
        state["_frozen_evaluation_count"] = evaluator.count()
    state["_evaluator"] = None
    state["_objective"] = None
    state["_evaluation_count"] = None
    state["_evaluation_record"] = None
    return state
```

In `__setstate__`, initialize absent caches, migrate a valid legacy `_ci_p_range` into
`searched_bounds`, conservatively classify missing inference metadata as not LR-capable, freeze the
stored evaluation count, and never revive legacy callbacks/bound methods.

- [ ] **Step 4: Add complete round-trip matrix**

Cover result-only/model pickles for both retention settings, cached and uncached CI, cached profile
curves, callback through editor reprofiling, prediction equality, legacy cached tuples, and the
documented detached-state error for uncached post-unpickle probes.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_tweedie_serialization.py tests/test_tweedie_profile.py -k 'pickle or legacy'`

Expected: all modern and conservative legacy migrations pass.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_serialization.py tests/test_tweedie_profile.py
rtk git commit -m "Serialize Tweedie profile results safely"
```

### Task 5: Project one honest status model into all reporting consumers

**Files:**
- Modify: `src/superglm/profiling/_reporting.py`
- Modify: `src/superglm/model/report_ops.py:135-224`
- Modify: `src/superglm/inference/metrics.py`
- Modify: `src/superglm/inference/summary.py`
- Modify: `src/superglm/editor/summaries.py`
- Modify: `src/superglm/editor/widget.py`
- Modify: `src/superglm/editor/app/summary.js`
- Create: `tests/test_tweedie_reporting.py`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Add failing summary separation tests**

```python
def test_summary_separates_final_fit_and_profile_status(fitted_tweedie_model):
    model = fitted_tweedie_model
    model.result.converged = True
    model._tweedie_profile_result = fake_profile_result(
        inference_kind="reml_plugin",
        converged=False,
        density_exact=True,
        density_certified=True,
        phi_converged=False,
        outer_boundary="upper",
    )
    info = model.summary()._model_info
    assert info["converged"] is True
    assert info["tweedie_profile_converged"] is False
    assert info["tweedie_density_exact"] is True
    assert info["tweedie_density_certified"] is True
    assert info["tweedie_phi_converged"] is False
    assert info["tweedie_boundary"] == "upper"
    assert "REML plug-in" in info["tweedie_p_method"]
    assert "MLE" not in info["tweedie_p_method"]
```

- [ ] **Step 2: Verify current summary reports only final-fit convergence/MLE wording**

Run: `rtk pytest tests/test_tweedie_reporting.py -k separates`

Expected: profile status fields are absent and method label says profile MLE.

- [ ] **Step 3: Add a shared reporting projection**

```python
def tweedie_profile_report_fields(result, alpha=0.05):
    interval, status = cached_tweedie_profile_ci(result, alpha)
    return {
        "tweedie_p": result.p_hat,
        "tweedie_phi": result.phi_hat,
        "tweedie_p_ci": interval,
        "tweedie_p_ci_status": status,
        "tweedie_inference_kind": result.inference_kind,
        "tweedie_profile_converged": result.converged,
        "tweedie_density_exact": result.density_exact,
        "tweedie_density_certified": result.density_certified,
        "tweedie_phi_converged": result.phi_converged,
        "tweedie_boundary": result.outer_boundary or "interior",
        "tweedie_effective_bounds": result.searched_bounds,
        "tweedie_p_method": tweedie_profile_method_label(result),
        "tweedie_profile_nll": result.nll,
    }
```

Use it in model summary, metrics, editor compact summary, widget payload, and browser summary. Labels:
`Exact profile MLE`, `Constrained profile`, `Pearson plug-in profile`, `Penalized plug-in profile`,
`REML plug-in profile`, or `Approximate diagnostic profile`. Cached CI lookup gates on
`inference_kind == "exact_mle"` and the certified LR-support predicate, not merely `phi_method`.

- [ ] **Step 4: Test cache invalidation and all consumers**

Fingerprint inference kind, convergence, exactness, density certification, phi convergence,
boundary, bounds, and cached
interval in `tweedie_profile_report_identity`. Assert model summary, metrics summary, editor JSON,
and browser payload update when any field changes.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_tweedie_reporting.py tests/test_editor.py -k 'tweedie and (summary or profile)'`

Expected: all consumers use honest, consistent labels and fields.

```bash
rtk git add src/superglm/profiling/_reporting.py src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/inference/summary.py src/superglm/editor/summaries.py src/superglm/editor/widget.py src/superglm/editor/app/summary.js tests/test_tweedie_reporting.py tests/test_editor.py
rtk git commit -m "Report Tweedie profile status honestly"
```

### Task 6: Make trace and profile plots status- and range-aware

**Files:**
- Create: `src/superglm/profiling/_tweedie_plotting.py`
- Modify: `src/superglm/profiling/tweedie.py:2847-3079`
- Create: `tests/test_tweedie_plotting.py`
- Modify: `tests/test_profile_ci.py:203-253`

- [ ] **Step 1: Add failing range/provenance/side-effect tests**

```python
def test_dense_profile_uses_bounds_and_contains_estimate_exactly(profile_result_factory):
    seen = []
    result = profile_result_factory(
        p_hat=1.0004,
        searched_bounds=(1.0001, 1.08),
        evaluator=lambda p: seen.append(float(p)) or (p - 1.0004) ** 2,
    )
    before = result.n_total_evaluations
    result.profile_plot(n_points=8)
    assert min(seen) == 1.0001
    assert max(seen) == 1.08
    assert 1.0004 in seen
    assert result.n_total_evaluations == before


def test_trace_plot_marks_rejected_records_without_evaluation(mixed_trace_result):
    before = mixed_trace_result.search_trace.copy(deep=True)
    count = mixed_trace_result.n_total_evaluations
    fig = mixed_trace_result.trace_plot()
    labels = {artist.get_label() for artist in fig.axes[0].lines + fig.axes[0].collections}
    assert {"Selectable exact", "Rejected", "Approximate", "Nonconverged"} <= labels
    pd.testing.assert_frame_equal(mixed_trace_result.search_trace, before)
    assert mixed_trace_result.n_total_evaluations == count
```

- [ ] **Step 2: Demonstrate hard-coded ranges and unconditional deviance wording**

Run: `rtk pytest tests/test_tweedie_plotting.py`

Expected: near-boundary estimate is excluded and invalid/mixed trace is labelled profile deviance.

- [ ] **Step 3: Implement pure plotting projections**

Move plotting preparation to `_tweedie_plotting.py`. `trace_plot` reads only the immutable scalar
records/search trace and distinguishes record states. It uses likelihood-ratio wording only for a
certified interior `exact_mle`; all other profiles use objective-difference wording with explicit
kind/boundary status.

For `profile_plot`, build `np.linspace(*searched_bounds, n_points)`, replace the nearest element with
`p_hat`, and stable-sort. `_TweedieProfileEvaluator.evaluate_curve(points)` deep-copies its owned
context, clears the copied callback/cache/warm starts, evaluates only on that temporary context, and
returns scalar `(p, nll)` tuples. The authoritative context, trace, count, and warm state remain
unchanged. Store the resulting immutable tuples in a serializable `_curve_cache[n_points]`.

Add `show_interval: bool | None = None`: `None` displays only an already cached supported interval;
`True` raises on profiles without ordinary LR support; `False` never requests one.

- [ ] **Step 4: Run plotting tests and commit**

Run: `rtk pytest tests/test_tweedie_plotting.py tests/test_profile_ci.py -k 'plot or trace or range'`

Expected: plots include valid near-boundary estimates, expose invalid records, and remain side-effect
free relative to search/CI state.

```bash
rtk git add src/superglm/profiling/_tweedie_plotting.py src/superglm/profiling/tweedie.py tests/test_tweedie_plotting.py tests/test_profile_ci.py
rtk git commit -m "Make Tweedie profile plots status aware"
```

### Task 7: Update documentation and execute the completion audit

**Files:**
- Modify: `docs/guide/families.md`
- Modify: `docs/notebooks/tweedie_profile_estimation.ipynb`
- Modify: `tests/test_tweedie_profile_docs.py`
- Create: `docs/superpowers/plans/2026-07-17-tweedie-completion-checklist.md`

- [ ] **Step 1: Update tested public documentation**

Document exact density authority, inference kinds, LR restrictions, search bounds as the CI/plot
support, `retain_fit_state=False`, `eager_ci_alpha`, post-pickle evaluator detachment, cached
interval/curve behavior, transactional failure semantics, and explicit approximate diagnostics.
Remove examples that combine a nonzero penalty with MLE/LR language.

- [ ] **Step 2: Update documentation contract tests**

Assert the guide and notebook contain the new options/terms and no longer claim that penalized,
REML, Pearson, constrained, approximate, boundary, or nonconverged results support ordinary LR
inference.

- [ ] **Step 3: Run focused workstream validation**

Run: `rtk pytest tests/test_tweedie_validation.py tests/test_tweedie_retention.py tests/test_tweedie_serialization.py tests/test_tweedie_reporting.py tests/test_tweedie_plotting.py tests/test_fit_state_retention.py tests/test_editor.py tests/test_tweedie_profile_docs.py`

Expected: all consumer reliability tests pass.

- [ ] **Step 4: Run complete numerical/profile validation**

Run: `rtk pytest tests/test_tweedie_numerics.py tests/test_tweedie_density.py tests/test_tweedie_reference.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile_state.py tests/test_tweedie_profile.py tests/test_profile_ci.py tests/test_weighted_forwarding.py`

Expected: all Tweedie correctness tests pass.

- [ ] **Step 5: Run repository gates**

Run: `rtk pytest tests/ -q`

Run: `rtk ruff check src/ tests/`

Run: `rtk prettier --check docs/guide/families.md docs/notebooks/tweedie_profile_estimation.ipynb`

Run: `rtk proxy uv run mypy --follow-imports=skip src/superglm/_tweedie_numerics.py src/superglm/_tweedie_density.py src/superglm/profiling/tweedie.py src/superglm/profiling/_tweedie_plotting.py src/superglm/model/profile_ops.py src/superglm/model/base.py`

Run: `rtk git diff --check`

Expected: pytest and formatting/lint gates pass; touched-module typing has no errors introduced by
this work; diff check is clean.

- [ ] **Step 6: Complete the requirement evidence matrix**

In `2026-07-17-tweedie-completion-checklist.md`, copy every correctness invariant, scope item, audit
regression, tolerance, and completion gate from the approved design. For each, record the exact test
node or runtime command/output that proves it. Mark missing or indirect evidence as incomplete and
continue implementation until every row has direct evidence.

- [ ] **Step 7: Run the clean-room provenance scan and commit**

Run: `rtk proxy rg -n -f "$SUPERGLM_CLEAN_ROOM_DENYLIST" docs/superpowers/specs/2026-07-17-tweedie-correctness-design.md docs/superpowers/plans/2026-07-17-tweedie-*.md src/superglm/_tweedie_numerics.py src/superglm/_tweedie_density.py tests/test_tweedie_numerics.py tests/test_tweedie_density.py tests/test_tweedie_reference.py tests/fixtures/tweedie_reference_values.json`

Expected: no matches. Inspect new numerical code for copied external identifiers, comments, source
structure, or implementation descriptions; remove and independently rederive any questionable
material.

```bash
rtk git add docs/guide/families.md docs/notebooks/tweedie_profile_estimation.ipynb tests/test_tweedie_profile_docs.py docs/superpowers/plans/2026-07-17-tweedie-completion-checklist.md
rtk git commit -m "Document reliable Tweedie estimation"
```
