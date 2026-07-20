# Transactional Fit State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fit()`, `fit_path()`, `fit_reml()`, and post-fit shape repair strongly exception-safe, history-independent, and no slower or more memory-hungry in routine use.

**Architecture:** Constructor intent is stored as model-owned templates. Each fit runs on a model-like `FitWorkspace` built from those templates, validates before feature learning, and transfers newly created buffers into one frozen `FitState`; `_install_fit_state()` performs the only public commit. Legacy private attributes remain read-only projections during migration, while revision-keyed caches and fitted-value properties read from the authoritative state.

**Tech Stack:** Python 3.10–3.14, NumPy, pandas, SciPy, Tabmat 4.2.1, pytest, Ruff, mypy, `tracemalloc`, Git worktrees.

---

## File map

- Create `src/superglm/model/input_validation.py`: pure pre-build normalization and family-domain validation.
- Create `src/superglm/model/fit_state.py`: `ModelConfig`, `FitState`, state capture, consistency checks, fitted/configured penalty helpers, and the no-fail install boundary.
- Create `src/superglm/model/fit_workspace.py`: isolated model-like attempt state constructed from `ModelConfig`, never from a prior row-scale fit.
- Create `tests/_fit_state_oracles.py`: reusable behavioral snapshots and failure injection helpers.
- Create `tests/test_fit_input_validation.py`: the shared entry-point validation matrix.
- Create `tests/test_fit_transactions.py`: first-fit/refit exception guarantees and workspace isolation.
- Create `tests/test_fit_ownership.py`: caller ownership and history-independence regressions.
- Create `tests/test_shape_repair_transaction.py`: coherent revision tests for post-fit repair.
- Create `benchmarks/benchmark_fit_state_trace.py`: shared frozen wall-time, native peak-RSS, allocation, trace-overhead, and result-fidelity fixture matrix.
- Modify `src/superglm/model/base.py`: constructor template ownership and workspace-safe design setup.
- Modify `src/superglm/model/fit_ops.py`: split public transactional wrappers from private in-workspace fit implementations.
- Modify `src/superglm/model/api.py`: fitted-value properties and defensive fitted-feature view.
- Modify `src/superglm/model/reml_execute.py`: keep all temporary REML mutations inside the workspace.
- Modify `src/superglm/model/reml_finalize.py`: build one coherent candidate state before commit.
- Modify `src/superglm/model/runtime_canonicalize.py`: canonicalize only workspace-owned specs/results.
- Modify `src/superglm/model/shape_ops.py`: replace in-place coefficient mutation with a derived-state transaction.
- Modify `src/superglm/model/explain_ops.py`, `report_ops.py`, and `state_ops.py`: use revision and resolved fitted penalty rather than caller-object identity.
- Modify `src/superglm/distributions.py`: complete the built-in response-domain validator and custom hook.
- Modify `pyproject.toml` and `uv.lock`: correct the Tabmat/Python-3.14 API floor to `>=4.1.3` without changing the locked 4.2.1 runtime.

### Task 1: Freeze correctness, wall-time, and memory baselines

**Files:**
- Create: `benchmarks/benchmark_fit_state_trace.py`
- Create: `tests/test_fit_state_trace_benchmark.py`
- Create: `benchmarks/results/fit_state_transaction_baseline.json`

- [ ] **Step 1: Write the benchmark contract test**

```python
from benchmarks.benchmark_fit_state_trace import CASES, compare_runs


def test_transaction_benchmark_covers_required_paths():
    assert {
        "dense_fit",
        "categorical_fit",
        "spline_fit",
        "exact_reml",
        "discrete_reml",
        "compact_reml",
    } <= set(CASES)


def test_compare_runs_rejects_quality_drift():
    before = {"deviance": 10.0, "prediction_checksum": 4.0}
    after = {"deviance": 10.01, "prediction_checksum": 4.0}
    failures = compare_runs(before, after, numerical_rtol=1e-8)
    assert any("deviance" in failure for failure in failures)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `rtk uv run pytest tests/test_fit_state_trace_benchmark.py -q`

Expected: FAIL because `benchmarks.benchmark_fit_state_trace` does not exist.

- [ ] **Step 3: Implement the frozen harness**

Create deterministic case factories with fixed seeds and a subprocess-per-case runner. Each record must contain `case`, `repeat`, `order`, `wall_time_s`, `python_peak_bytes`, `rss_peak_bytes` when supported, `deviance`, `effective_df`, `prediction_checksum`, `n_iter`, and `converged`. Run one warmup and at least five counterbalanced repeats with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Implement comparison as:

```python
NUMERICAL_KEYS = ("deviance", "effective_df", "prediction_checksum")


def compare_runs(before, after, *, numerical_rtol=1e-10):
    failures = []
    for key in NUMERICAL_KEYS:
        if not np.isclose(before[key], after[key], rtol=numerical_rtol, atol=1e-12):
            failures.append(f"{key}: {before[key]} != {after[key]}")
    return failures
```

- [ ] **Step 4: Run and store the pre-refactor baseline**

Run: `rtk proxy env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --python 3.14 python benchmarks/benchmark_fit_state_trace.py --suite wall-time --warmups 2 --repeats 10 --output benchmarks/results/fit_state_transaction_baseline.json`

Expected: six cases, no nonfinite metrics, and JSON containing the current commit SHA and Python/NumPy/Tabmat versions.

- [ ] **Step 5: Commit**

```bash
rtk git add benchmarks/benchmark_fit_state_trace.py tests/test_fit_state_trace_benchmark.py benchmarks/results/fit_state_transaction_baseline.json
rtk git commit -m "Add fit transaction performance baselines"
```

### Task 2: Validate every fit input before feature learning

**Files:**
- Create: `src/superglm/model/input_validation.py`
- Create: `tests/test_fit_input_validation.py`
- Modify: `src/superglm/distributions.py:23-51,357-371`

- [ ] **Step 1: Write the failing shared validation matrix**

```python
@pytest.mark.parametrize("entrypoint", ["fit", "fit_path", "fit_reml"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("y", np.array([]), "non-empty"),
        ("y", np.array([[1.0]]), "one-dimensional"),
        ("y", np.array([0.0, np.nan]), "finite"),
        ("sample_weight", np.array([1.0, -1.0]), "nonnegative"),
        ("sample_weight", np.array([0.0, 0.0]), "not all zero"),
        ("offset", np.array([0.0, np.inf]), "finite"),
    ],
)
def test_fit_entrypoints_validate_before_feature_build(
    entrypoint, field, value, message, monkeypatch
):
    X = pd.DataFrame({"x": [0.0, 1.0]})
    kwargs = {"y": np.array([0.0, 1.0]), "sample_weight": None, "offset": None}
    kwargs[field] = value
    model = SuperGLM(features={"x": Numeric()}, selection_penalty=0.0)
    monkeypatch.setattr(Numeric, "build", lambda *args, **kwargs: pytest.fail("feature built"))
    with pytest.raises(ValueError, match=message):
        getattr(model, entrypoint)(X, **kwargs)
```

Also parameterize Binomial `{0,1}`, Poisson/NB nonnegative, Gamma positive, Tweedie nonnegative, Gaussian finite, row-count mismatch, duplicate/missing columns, complex arrays, strict-positive Tweedie weights, and a custom `validate_response()` hook.

- [ ] **Step 2: Run the test to verify current ordering and domain failures**

Run: `rtk uv run pytest tests/test_fit_input_validation.py -q`

Expected: FAIL for all three entry points, with `fit_reml()` reaching design construction or a numerical error for invalid responses.

- [ ] **Step 3: Implement one pure validator**

```python
@dataclass(frozen=True)
class ValidatedFitInput:
    X: pd.DataFrame
    y: NDArray[np.float64]
    sample_weight: NDArray[np.float64]
    offset: NDArray[np.float64] | None


def validate_fit_input(X, y, sample_weight, offset, family, required_columns):
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if not X.columns.is_unique:
        raise ValueError("X columns must be unique")
    missing = sorted(set(required_columns) - set(X.columns))
    if missing:
        raise ValueError(f"X is missing required columns: {missing}")
    y_arr = _finite_vector("y", y, len(X))
    w_arr = np.ones(len(X), dtype=np.float64) if sample_weight is None else _finite_vector(
        "sample_weight", sample_weight, len(X)
    )
    if np.any(w_arr < 0) or not np.any(w_arr > 0):
        raise ValueError("sample_weight must be nonnegative and not all zero")
    if isinstance(family, Tweedie) and np.any(w_arr <= 0):
        raise ValueError("Tweedie sample_weight must be strictly positive")
    offset_arr = None if offset is None else _finite_vector("offset", offset, len(X))
    validate_response(y_arr, family)
    return ValidatedFitInput(X, y_arr, w_arr, offset_arr)
```

Extend `validate_response()` with the documented built-in domains, and call a custom family hook only when it is not the module-level function itself.

- [ ] **Step 4: Run focused tests**

Run: `rtk uv run pytest tests/test_fit_input_validation.py tests/test_binomial.py tests/test_tweedie_profile.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/input_validation.py src/superglm/distributions.py tests/test_fit_input_validation.py
rtk git commit -m "Validate fit inputs before design construction"
```

### Task 3: Own constructor configuration and preserve automatic intent

**Files:**
- Create: `src/superglm/model/fit_state.py`
- Modify: `src/superglm/model/base.py:391-534`
- Modify: `src/superglm/model/api.py:175-193`
- Create: `tests/test_fit_ownership.py`

- [ ] **Step 1: Write caller-alias and sentinel tests**

```python
def test_constructor_defensively_owns_mutable_inputs():
    feature = Spline(n_knots=5)
    penalty = GroupLasso(lambda1=None)
    interactions = [("x", "z")]
    model = SuperGLM(
        penalty=penalty,
        features={"x": feature, "z": Numeric()},
        interactions=interactions,
    )
    assert model.penalty is not penalty
    assert model.features["x"] is not feature
    interactions.clear()
    penalty.lambda1 = 99.0
    assert model.penalty.lambda1 is None
    assert model._pending_interactions == (("x", "z"),)


def test_auto_intent_survives_successful_fit(sample_data):
    X, y, w = sample_data
    model = SuperGLM(selection_penalty=None)
    model.fit(X, y, sample_weight=w)
    assert model.penalty.lambda1 is None
    assert model.selection_penalty_ > 0.0


def test_supported_assignment_replaces_configuration_revision():
    model = SuperGLM(selection_penalty=None)
    before = model._config_revision
    model.selection_penalty = 0.25
    assert model._config_revision == before + 1
    assert model.penalty.lambda1 == pytest.approx(0.25)
```

- [ ] **Step 2: Run the tests to verify aliasing and sentinel mutation**

Run: `rtk uv run pytest tests/test_fit_ownership.py -q`

Expected: FAIL because penalty/features/interactions are currently retained or mutated by identity.

- [ ] **Step 3: Store configuration templates**

Add a frozen `ModelConfig` in `fit_state.py` with deep-copied `family`, `link`, `penalty`, feature templates, and tuple interactions. In `init_model()`, install model-owned copies and initialize `_config_revision = 0`, `_fit_revision = 0`, `_fit_state = None`, `_selection_penalty_fitted = None`, and `_distribution_fitted = None`. Back supported `family`, `link`, `penalty`, and `selection_penalty` assignments with `dataclasses.replace()` plus one configuration-revision increment. Return defensive copies from mutable configuration getters; nested mutation is intentionally not a configuration API. Make `features` return a defensive deep copy and add:

```python
@property
def selection_penalty_(self) -> float:
    if self._fit_state is None:
        raise RuntimeError("Model is not fitted")
    return self._fit_state.selection_penalty


@property
def distribution_(self) -> Distribution:
    if self._fit_state is None:
        raise RuntimeError("Model is not fitted")
    return self._fit_state.distribution
```

- [ ] **Step 4: Run ownership and API tests**

Run: `rtk uv run pytest tests/test_fit_ownership.py tests/test_api.py tests/test_interactions.py -q`

Expected: PASS after updating old identity assertions to the new ownership contract.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/base.py src/superglm/model/api.py src/superglm/model/fit_state.py tests/test_fit_ownership.py tests/test_api.py
rtk git commit -m "Separate fit configuration from learned state"
```

### Task 4: Add the workspace and atomic state install boundary

**Files:**
- Modify: `src/superglm/model/fit_state.py`
- Create: `src/superglm/model/fit_workspace.py`
- Create: `tests/_fit_state_oracles.py`
- Create: `tests/test_fit_transactions.py`
- Modify: `src/superglm/model/base.py:490-534`

- [ ] **Step 1: Write state transfer tests**

```python
def test_workspace_does_not_alias_previous_fitted_state(fitted_model):
    previous = fitted_model._fit_state
    workspace = FitWorkspace.start(fitted_model, mode="fit", validated_inputs=None)
    assert workspace.model._specs is not fitted_model._specs
    assert workspace.model._groups is not fitted_model._groups
    assert workspace.model._dm is None
    assert fitted_model._fit_state is previous


def test_install_is_one_revision_swap(fitted_model):
    workspace = FitWorkspace.start(fitted_model, mode="fit", validated_inputs=None)
    candidate = make_minimal_candidate(workspace, fitted_model._fit_revision + 1)
    old_dict = fitted_model.__dict__
    _install_fit_state(fitted_model, candidate)
    assert fitted_model._fit_state is candidate.state
    assert fitted_model._fit_revision == candidate.state.revision
    assert fitted_model.__dict__ is not old_dict
```

- [ ] **Step 2: Run to verify missing types**

Run: `rtk uv run pytest tests/test_fit_transactions.py -q`

Expected: FAIL importing `FitWorkspace`, `FitCandidate`, and `_install_fit_state`.

- [ ] **Step 3: Implement workspace/state primitives**

```python
@dataclass(frozen=True)
class FitState:
    revision: int
    selection_penalty: float
    distribution: Distribution
    projections: Mapping[str, object]
    retained: bool
    repair_revision: int = 0


@dataclass
class FitWorkspace:
    model: object
    mode: str
    validated_inputs: ValidatedFitInput | None
    previous_revision: int

    @classmethod
    def start(cls, public_model, *, mode, validated_inputs):
        work_model = public_model._config.materialize(type(public_model))
        return cls(work_model, mode, validated_inputs, public_model._fit_revision)


@dataclass(frozen=True)
class FitCandidate:
    state: FitState
    prepared_model_dict: dict[str, object]


def _install_fit_state(model, candidate):
    model.__dict__ = candidate.prepared_model_dict
```

`capture_fit_state()` must validate dimensions and finiteness, freeze owned result/covariance arrays without copying row-scale design buffers, create the complete replacement `prepared_model_dict` before installation, and include an empty repair set. `FitWorkspace.start()` must materialize a fresh working model from `ModelConfig`; it must not shallow-copy the prior public model or its fitted dictionary.

- [ ] **Step 4: Run primitive tests and an allocation spy**

Run: `rtk uv run pytest tests/test_fit_transactions.py::test_workspace_does_not_alias_previous_fitted_state tests/test_fit_transactions.py::test_install_is_one_revision_swap -q`

Expected: PASS; the allocation spy reports no copy of `_dm` or its group matrices during capture/install.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_state.py src/superglm/model/fit_workspace.py src/superglm/model/base.py tests/_fit_state_oracles.py tests/test_fit_transactions.py
rtk git commit -m "Add atomic fit state installation"
```

### Task 5: Route ordinary fit through the transaction

**Files:**
- Modify: `src/superglm/model/fit_ops.py:425-595`
- Modify: `src/superglm/model/runtime_canonicalize.py:338-427`
- Modify: `tests/test_fit_transactions.py`

- [ ] **Step 1: Write first-fit and refit failure tests**

```python
@pytest.mark.parametrize(
    "failure_target",
    ["fit_pirls", "_compute_fit_stats", "canonicalize_fitted_model", "_maybe_release_fit_state"],
)
def test_failed_refit_preserves_previous_revision(sample_data, failure_target, monkeypatch):
    X, y, w = sample_data
    model = SuperGLM(selection_penalty=0.0).fit(X, y, sample_weight=w)
    before = snapshot_model_behavior(model, X)
    monkeypatch_failure(failure_target, monkeypatch)
    with pytest.raises(InjectedFitFailure):
        model.fit(X, y, sample_weight=w)
    assert_model_behavior_unchanged(model, X, before)


def test_failed_first_fit_remains_unfitted(sample_data, monkeypatch):
    X, y, w = sample_data
    model = SuperGLM(selection_penalty=0.0)
    monkeypatch_failure("fit_irls_direct", monkeypatch)
    with pytest.raises(InjectedFitFailure):
        model.fit(X, y, sample_weight=w)
    assert model._fit_state is None
    with pytest.raises(RuntimeError):
        _ = model.result
```

- [ ] **Step 2: Run to demonstrate current hybrid state**

Run: `rtk uv run pytest tests/test_fit_transactions.py -k 'failed_first_fit or failed_refit' -q`

Expected: FAIL because `fit()` currently mutates the public model before injected failures.

- [ ] **Step 3: Split wrapper and implementation**

Rename the existing body to `_fit_in_workspace()`. The public wrapper must perform validation, create the workspace, run the entire existing build/solve/stats/canonicalization/release sequence there, capture a `FitState`, and install once:

```python
def fit(model, X, y, sample_weight=None, offset=None, **controls):
    validated = preflight_and_validate(model, X, y, sample_weight, offset, mode="fit")
    workspace = FitWorkspace.start(model, mode="fit", validated_inputs=validated)
    try:
        _fit_in_workspace(workspace.model, validated, **controls)
        candidate = capture_fit_state(workspace, revision=model._fit_revision + 1)
    except Exception:
        raise
    _install_fit_state(model, candidate)
    return model
```

Do not snapshot or restore the public model. Ensure canonicalization sees only workspace-owned specs and the compatibility projections are prepared before commit.

- [ ] **Step 4: Run focused fit suites and compare baseline numerics**

Run: `rtk uv run pytest tests/test_fit_transactions.py tests/test_api.py tests/test_fit_state_retention.py tests/test_nb2.py -q`

Expected: PASS with baseline coefficient, deviance, prediction, convergence, and evaluation-count parity.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/model/runtime_canonicalize.py tests/test_fit_transactions.py
rtk git commit -m "Make ordinary fits transactional"
```

### Task 6: Route fit_path through the same commit boundary

**Files:**
- Modify: `src/superglm/model/fit_ops.py:598-695`
- Modify: `src/superglm/model/path_ops.py:1-75`
- Modify: `tests/test_path.py:46-162`
- Modify: `tests/test_fit_transactions.py`

- [ ] **Step 1: Write path finalization failure tests**

```python
def test_failed_path_refit_preserves_previous_fit(sample_data, monkeypatch):
    X, y, w = sample_data
    model = SuperGLM(selection_penalty=0.01).fit(X, y, sample_weight=w)
    before = snapshot_model_behavior(model, X)
    monkeypatch.setattr(
        runtime_canonicalize,
        "canonicalize_intercept_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(InjectedFitFailure()),
    )
    with pytest.raises(InjectedFitFailure):
        model.fit_path(X, y, sample_weight=w, n_lambda=3)
    assert_model_behavior_unchanged(model, X, before)
```

- [ ] **Step 2: Run to verify current partial commit**

Run: `rtk uv run pytest tests/test_fit_transactions.py::test_failed_path_refit_preserves_previous_fit -q`

Expected: FAIL because result/stats are installed before intercept-path canonicalization.

- [ ] **Step 3: Add `_fit_path_in_workspace()`**

Return `(PathResult, FitState)` from local finalization, install only after the coefficient path and terminal canonical result agree, and keep path-wise penalty changes local to the workspace. The public model's automatic penalty sentinel must remain unchanged.

- [ ] **Step 4: Run path tests**

Run: `rtk uv run pytest tests/test_fit_transactions.py tests/test_path.py tests/test_api.py -k 'path or transaction' -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/model/path_ops.py tests/test_fit_transactions.py
rtk git commit -m "Make regularization paths transactional"
```

### Task 7: Route every REML branch and final refit through the workspace

**Files:**
- Modify: `src/superglm/model/fit_ops.py:697-1000`
- Modify: `src/superglm/model/reml_execute.py:98-375`
- Modify: `src/superglm/model/reml_finalize.py:24-195`
- Modify: `src/superglm/model/reml_state.py`
- Modify: `tests/test_fit_transactions.py`

- [ ] **Step 1: Write branch-specific failure tests**

```python
@pytest.mark.parametrize(
    "case,failure_target",
    [
        ("exact", "fit_irls_direct"),
        ("discrete", "canonicalize_fitted_model"),
        ("fixed_qp", "_compute_fit_stats"),
        ("scop", "freeze_prediction_plan"),
        ("compact", "fit_inference_info"),
    ],
)
def test_failed_reml_finalization_preserves_previous_state(
    reml_case, case, failure_target, monkeypatch
):
    model, X, y, w = reml_case(case)
    model.fit(X, y, sample_weight=w)
    before = snapshot_model_behavior(model, X)
    monkeypatch_failure(failure_target, monkeypatch)
    with pytest.raises(InjectedFitFailure):
        model.fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    assert_model_behavior_unchanged(model, X, before)
```

- [ ] **Step 2: Run to expose current partial REML commits**

Run: `rtk uv run pytest tests/test_fit_transactions.py -k reml_finalization -q`

Expected: FAIL for direct final refit, canonicalization, and compact-release injection points.

- [ ] **Step 3: Add `_fit_reml_in_workspace()`**

All exact/discrete/EFS/SCOP/fixed/QP branches must mutate only the workspace. `finalize_reml_fit()` must return a coherent candidate instead of publishing it. Restore stripped QP constraints on the workspace in `finally`; never touch the prior state's groups. Capture/install only after final refit, phi, statistics, canonicalization, prediction plan, compact distillation, and consistency checks pass.

- [ ] **Step 4: Run REML state suites**

Run: `rtk uv run pytest tests/test_fit_transactions.py tests/test_reml.py tests/test_reml_efs.py tests/test_shape_reml.py tests/test_fit_state_retention.py -q`

Expected: PASS with no changes to successful baseline numerics.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/model/reml_execute.py src/superglm/model/reml_finalize.py src/superglm/model/reml_state.py tests/test_fit_transactions.py
rtk git commit -m "Make REML fitting transactional"
```

### Task 8: Prove repeated fits are history-independent

**Files:**
- Modify: `src/superglm/model/fit_ops.py:184-195,425-433`
- Modify: `src/superglm/profiling/nb.py:294-330`
- Modify: `src/superglm/profiling/tweedie.py:3401-3536`
- Modify: `src/superglm/model/profile_ops.py:16-174`
- Modify: `src/superglm/features/categorical.py:96-121`
- Modify: `tests/test_fit_ownership.py`

- [ ] **Step 1: Write fresh-model equivalence tests**

```python
@pytest.mark.parametrize("mode", ["auto_lambda", "auto_theta", "most_exposed", "auto_schema"])
def test_second_fit_matches_fresh_model(mode, two_datasets):
    A, B = two_datasets[mode]
    sequential = make_model(mode)
    fit_case(sequential, A)
    fit_case(sequential, B)
    fresh = make_model(mode)
    fit_case(fresh, B)
    np.testing.assert_allclose(sequential.predict(B.X), fresh.predict(B.X), rtol=1e-10, atol=1e-10)
    assert sequential.selection_penalty_ == pytest.approx(fresh.selection_penalty_)
    assert getattr(sequential, "theta_", None) == pytest.approx(
        getattr(fresh, "theta_", None)
    )
```

- [ ] **Step 2: Run to reproduce history dependence**

Run: `rtk uv run pytest tests/test_fit_ownership.py -k second_fit -q`

Expected: FAIL for automatic lambda, NB theta, most-exposed base, and auto-detected schema.

- [ ] **Step 3: Resolve every learned choice from templates**

Auto-detect into empty workspace specs on every fit, profile NB theta on a nested workspace, resolve auto lambda into `_resolved_penalty`, and compute `Categorical(base="most_exposed")` from the current weights without overwriting the template sentinel. Expose fitted `theta_` and `selection_penalty_`; keep `family.theta == "auto"` and `penalty.lambda1 is None`.

High-level `estimate_theta()` and `estimate_p()` must profile and perform their final refit inside one outer workspace; they may not call a public fit that commits an intermediate state. Add injected final-refit failures to prove the previous fitted revision survives.

- [ ] **Step 4: Run ownership, NB, interaction, and categorical tests**

Run: `rtk uv run pytest tests/test_fit_ownership.py tests/test_nb2.py tests/test_weighted_forwarding.py tests/test_interactions.py tests/test_api.py -q`

Expected: PASS after updating fitted-value assertions to `theta_` and `selection_penalty_`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/profiling/nb.py src/superglm/profiling/tweedie.py src/superglm/model/profile_ops.py src/superglm/features/categorical.py src/superglm/model/api.py tests/test_fit_ownership.py tests/test_nb2.py tests/test_weighted_forwarding.py tests/test_api.py tests/test_tweedie_profile.py
rtk git commit -m "Make automatic fit choices history independent"
```

### Task 9: Revision-key caches and compact-state transfer

**Files:**
- Modify: `src/superglm/model/explain_ops.py:24-69`
- Modify: `src/superglm/model/fit_ops.py:358-423`
- Modify: `src/superglm/model/state_ops.py`
- Modify: `src/superglm/model/report_ops.py`
- Modify: `src/superglm/inference/metrics.py`
- Modify: `src/superglm/export/summary.py`
- Modify: `src/superglm/model_selection.py`
- Modify: `src/superglm/editor/apply.py`
- Modify: `tests/test_fit_state_retention.py`
- Modify: `tests/test_fit_transactions.py`

- [ ] **Step 1: Write cache and compact refit tests**

```python
def test_failed_refit_keeps_compact_state_compact(compact_fitted_model, sample_data, monkeypatch):
    X, y, w = sample_data
    before_revision = compact_fitted_model._fit_revision
    monkeypatch_failure("fit_irls_direct", monkeypatch)
    with pytest.raises(InjectedFitFailure):
        compact_fitted_model.fit(X, y, sample_weight=w)
    assert compact_fitted_model._fit_revision == before_revision
    assert compact_fitted_model._dm is None


def test_caches_are_keyed_by_revision(fitted_model, sample_data):
    X, y, w = sample_data
    fitted_model.metrics(X, y, sample_weight=w)
    assert fitted_model._fit_metrics_cache_revision == fitted_model._fit_revision
```

- [ ] **Step 2: Run to verify stale identity-based behavior**

Run: `rtk uv run pytest tests/test_fit_state_retention.py tests/test_fit_transactions.py -q`

Expected: FAIL because caches use object identity and compact release is destructive before a public commit boundary.

- [ ] **Step 3: Transfer compact state rather than clear public state**

Distill inference on the workspace, drop its row buffers, then capture it. Key mutable cache entries by `(fit_revision, repair_revision)` and remove result/input-object identity signatures. Keep a single fit-owned input snapshot only where an API cannot operate from compiled prediction/inference state; never retain caller objects as cache keys.

Use explicit `configured_penalty(model)` and `fitted_penalty(model)` helpers: new attempts and model-selection configuration use the former, while summaries, inference, exports, and fitted-basis editor operations use the latter. Direct editor changes must create derived revisions instead of mutating the installed `PIRLSResult`.

- [ ] **Step 4: Run retention, editor, metrics, and serialization tests**

Run: `rtk uv run pytest tests/test_fit_state_retention.py tests/test_fit_transactions.py tests/test_editor.py tests/test_training_telemetry.py -q`

Expected: PASS; compact pickle size remains below half the retained model and peak memory does not grow.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/explain_ops.py src/superglm/model/fit_ops.py src/superglm/model/state_ops.py src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/export/summary.py src/superglm/model_selection.py src/superglm/editor/apply.py tests/test_fit_state_retention.py tests/test_fit_transactions.py
rtk git commit -m "Key fit caches by state revision"
```

### Task 10: Make shape repair a coherent derived-state transaction

**Files:**
- Modify: `src/superglm/model/shape_ops.py`
- Create: `tests/test_shape_repair_transaction.py`
- Modify: `tests/test_shape_postfit.py`

- [ ] **Step 1: Write repair transaction tests**

```python
def test_shape_repair_commits_new_revision_without_mutating_old_state(shape_model, shape_data):
    X, y = shape_data
    model = shape_model.fit(X, y)
    old_state = model._fit_state
    old_beta = model.result.beta.copy()
    model.apply_shape_postfit(X)
    assert model._fit_state is not old_state
    np.testing.assert_array_equal(old_state.projections["_result"].beta, old_beta)
    np.testing.assert_allclose(model.predict(X), model._fit_mu, rtol=1e-10, atol=1e-10)
    assert model._fit_state.repair_revision == old_state.repair_revision + 1


def test_failed_repair_preserves_original_revision(shape_model, shape_data, monkeypatch):
    X, y = shape_data
    model = shape_model.fit(X, y)
    before = snapshot_model_behavior(model, X)
    monkeypatch_failure("_compute_fit_stats", monkeypatch)
    with pytest.raises(InjectedFitFailure):
        model.apply_shape_postfit(X)
    assert_model_behavior_unchanged(model, X, before)
```

- [ ] **Step 2: Run to expose in-place beta mutation**

Run: `rtk uv run pytest tests/test_shape_repair_transaction.py -q`

Expected: FAIL because current repair mutates `result.beta` and only clears four caches.

- [ ] **Step 3: Build and install a derived candidate**

Compute repaired beta locally, create replacement solver/public/REML results, recompute predictions and fit statistics, rebuild canonical/prediction state, transform covariance/EDF only when a repair Jacobian exists, otherwise attach a typed unavailable reason, clear revision caches, and install the candidate once. Ordinary successful fits must reset repair metadata.

- [ ] **Step 4: Run shape, inference, summary, and refit tests**

Run: `rtk uv run pytest tests/test_shape_repair_transaction.py tests/test_shape_postfit.py tests/test_shape_reml.py tests/test_refit.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/shape_ops.py tests/test_shape_repair_transaction.py tests/test_shape_postfit.py
rtk git commit -m "Make post-fit shape repair transactional"
```

### Task 11: Preserve legacy loading while rejecting incoherent hybrids

**Files:**
- Modify: `src/superglm/model/api.py`
- Create: `tests/test_fit_state_pickle.py`
- Modify: `tests/test_fit_state_retention.py`

- [ ] **Step 1: Write current and legacy pickle tests**

```python
def test_pickle_roundtrip_preserves_revision_and_predictions(fitted_model, sample_data):
    X, _, _ = sample_data
    restored = pickle.loads(pickle.dumps(fitted_model))
    assert restored._fit_revision == fitted_model._fit_revision
    np.testing.assert_allclose(restored.predict(X), fitted_model.predict(X))


def test_incoherent_legacy_state_is_rejected(fitted_model):
    payload = fitted_model.__getstate__()
    payload.pop("_fit_state", None)
    payload["_solver_result"] = None
    restored = SuperGLM.__new__(SuperGLM)
    with pytest.raises(ValueError, match="incoherent legacy fitted state"):
        restored.__setstate__(payload)
```

- [ ] **Step 2: Run to verify migration support is missing**

Run: `rtk uv run pytest tests/test_fit_state_pickle.py -q`

Expected: FAIL because there is no `FitState` migration path.

- [ ] **Step 3: Implement coherent legacy migration**

In `__setstate__`, accept unfitted legacy dictionaries, or validate that result/solver/design/groups/stats dimensions agree before constructing revision 1. Never guess through a hybrid. Preserve current-state pickle round trips without copying row-scale arrays.

- [ ] **Step 4: Run pickle/editor/export tests**

Run: `rtk uv run pytest tests/test_fit_state_pickle.py tests/test_fit_state_retention.py tests/test_editor.py tests/test_export.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/api.py tests/test_fit_state_pickle.py tests/test_fit_state_retention.py
rtk git commit -m "Migrate coherent legacy fit state"
```

### Task 12: Enforce performance, dependency, and Python 3.14 gates

**Files:**
- Modify: `benchmarks/benchmark_fit_state_trace.py`
- Modify: `tests/test_fit_state_trace_benchmark.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add comparison gates**

```python
def assert_acceptance(before, after):
    assert compare_runs(before, after) == []
    assert after["median_wall_time_s"] <= before["median_wall_time_s"] * 1.03
    assert after["peak_memory_bytes"] <= before["peak_memory_bytes"] * 1.02
    assert after["solver_calls"] == before["solver_calls"]
```

Use bootstrap confidence intervals for counterbalanced medians; apply the 3% wall gate only when the interval excludes zero and otherwise classify differences within 5% as noise.

- [ ] **Step 2: Correct the Tabmat floor and refresh lock metadata**

Change `tabmat>=4.0` to `tabmat>=4.1.3`, because `categories=` is used, 4.1 is the required API line, and 4.1.3 is the first practical ordinary-CPython-3.14 wheel floor. Keep the resolved version at 4.2.1.

Run: `rtk uv lock`

Expected: the lock retains Tabmat 4.2.1 and changes only the declared project requirement.

- [ ] **Step 3: Run counterbalanced post-refactor benchmarks**

Run: `rtk proxy env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --python 3.14 python benchmarks/benchmark_fit_state_trace.py --suite wall-time --compare benchmarks/results/fit_state_transaction_baseline.json --warmups 2 --repeats 10 --output benchmarks/results/fit_state_transaction_after.json`

Expected: numerical parity, no credible routine slowdown over 3%, and no peak-memory increase over 2%.

Measure native high-water RSS in fresh processes for retained and compact cases, five times each:

```bash
rtk proxy /usr/bin/time -v -o /tmp/superglm-retained.time uv run --python 3.14 python benchmarks/benchmark_fit_state_trace.py --worker --fixture large-retained --output /tmp/superglm-retained.json
rtk proxy /usr/bin/time -v -o /tmp/superglm-compact.time uv run --python 3.14 python benchmarks/benchmark_fit_state_trace.py --worker --fixture large-compact --output /tmp/superglm-compact.json
```

Use median maximum RSS for the 2% gate. Record `tracemalloc` only as a diagnostic because it does not include NumPy, BLAS, or Tabmat native allocations.

- [ ] **Step 4: Run the full verification matrix**

```bash
rtk uv run pytest tests/test_fit_input_validation.py tests/test_fit_ownership.py tests/test_fit_transactions.py tests/test_shape_repair_transaction.py -q
rtk uv run pytest tests/ -m "not slow" -q
rtk uv run pytest tests/ -q
rtk uv run ruff check src/ tests/ benchmarks/benchmark_fit_state_trace.py
rtk uv run mypy src/
rtk uv run python -VV
rtk uv run python run_test.py
```

Build and test the actual wheel under ordinary CPython 3.14, then repeat the focused Tabmat algebra oracle at the declared floor and locked version:

```bash
rtk proxy env UV_PYTHON=3.14 uv build --wheel --out-dir /tmp/superglm-wheel
rtk proxy uv venv --python 3.14 /tmp/superglm-py314
rtk proxy uv pip install --python /tmp/superglm-py314/bin/python /tmp/superglm-wheel/superglm-0.12.0-py3-none-any.whl pytest pytest-cov
rtk proxy env PYTHONNOUSERSITE=1 /tmp/superglm-py314/bin/python -m pytest tests/ -q -m "not browser"
rtk proxy uv venv --python 3.14 /tmp/superglm-tabmat-floor
rtk proxy uv pip install --python /tmp/superglm-tabmat-floor/bin/python /tmp/superglm-wheel/superglm-0.12.0-py3-none-any.whl 'tabmat==4.1.3' pytest
rtk proxy env PYTHONNOUSERSITE=1 /tmp/superglm-tabmat-floor/bin/python -m pytest tests/test_theory_invariants.py::TestBackendLinearAlgebraInvariants::test_high_cardinality_tabmat_subset_preserves_width tests/test_theory_invariants.py::TestBackendLinearAlgebraInvariants::test_tabmat_split_skips_sparse_ssp_groups tests/test_core.py::TestCategoricalDeterminism -q
rtk proxy uv pip install --python /tmp/superglm-tabmat-floor/bin/python 'tabmat==4.2.1'
rtk proxy env PYTHONNOUSERSITE=1 /tmp/superglm-tabmat-floor/bin/python -m pytest tests/test_theory_invariants.py::TestBackendLinearAlgebraInvariants::test_high_cardinality_tabmat_subset_preserves_width tests/test_theory_invariants.py::TestBackendLinearAlgebraInvariants::test_tabmat_split_skips_sparse_ssp_groups tests/test_core.py::TestCategoricalDeterminism -q
```

Expected: all tests pass under CPython 3.14.4 from the built wheel, and Tabmat 4.1.3/4.2.1 produce identical focused numerical results. Free-threaded 3.14t remains explicitly experimental and out of this support declaration.

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml uv.lock benchmarks/benchmark_fit_state_trace.py tests/test_fit_state_trace_benchmark.py benchmarks/results/fit_state_transaction_after.json
rtk git commit -m "Verify transactional fit performance"
```

## Completion review

Before declaring this plan complete, inspect every public fit entry point and assert that the only write to the original model is `_install_fit_state()`. Run `rtk proxy rg -n 'model\._[A-Za-z0-9_]+\s*=' src/superglm/model/fit_ops.py src/superglm/model/reml_execute.py src/superglm/model/reml_finalize.py` and classify each match as workspace-local or a violation. Compare the final benchmark artifact to the baseline and report both improvements and regressions, including confidence intervals and memory measurements.
