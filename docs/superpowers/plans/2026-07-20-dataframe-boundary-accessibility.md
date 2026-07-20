# Dataframe Boundary and Developer Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept eager pandas and Polars dataframes across SuperGLM's complete model-data API through one narrow boundary, preserve the existing compiled numerical designs and solver hot paths, and make the route from input data to IRLS/PIRLS/REML understandable from one permanent developer guide and one fitted-design summary.

**Architecture:** A private `EagerFrame` adapter normalizes only names, schema, selected-column extraction, positional row slicing, and retained-data fingerprints. Feature compilation consumes that adapter once and continues to produce the current `DesignMatrix`/`GroupMatrix` objects; no dataframe abstraction enters matrix kernels or solvers. Construction-time route reasons are exposed read-only through `design_summary()` without constructing Tabmat matrices or executing kernels.

**Tech Stack:** Python 3.10+, NumPy, pandas, Narwhals stable v2, optional Polars, SciPy, Tabmat, scikit-learn, pytest, Ruff, Astral ty, uv, MkDocs.

---

## Starting state and guardrails

- Worktree: `/home/mhick/python_projects/superglm/.worktrees/dataframe-boundary`
- Branch: `codex/dataframe-boundary-accessibility`
- Base: `origin/master` at `a73521f51903b62f02616dc36cf5923224444606`
- Approved design commit: `0c62773`
- Standard interpreter baseline: CPython 3.14.4, not the free-threaded 3.14 build.
- Baseline command already run:

  ```bash
  rtk .venv/bin/python -m pytest tests/ -q -m 'not slow and not browser'
  ```

  Result: 3740 passed, 103 skipped, 197 deselected.

The following rules apply to every task:

1. Keep the caller's native frame for callbacks and retained-state behavior; never publish the adapter.
2. Never call `to_pandas()` on a Polars input and never convert an unrelated column.
3. Keep all dataframe dispatch outside IRLS, PIRLS, REML, line search, Gram, matvec, rmatvec, sandwich, and compressed bin-space loops.
4. Preserve the concrete `GroupMatrix` classes, solver-column order, penalties, constraints, Tabmat decisions, discrete compression, terminal state, and numerical tolerances.
5. Preserve pandas outputs for summaries, diagnostics, plots, cross-validation records, and rating tables.
6. Preserve the current pandas retained-data digest byte-for-byte for the same pandas version and data.
7. Treat a repeatable pandas regression above 3% in stable microbenchmarks or 5% end to end, any meaningful memory increase, any lost accelerated call, or any observation-row materialization of a compressed design as a blocker.
8. Do not change CI sharding, pandas' dependency status, solver algorithms, or AFT support in this branch.

## Task 1: Make the dataframe dependency contract explicit

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_release_packaging.py`

**Invariant at risk:** Polars must remain optional at runtime, while Narwhals must be a direct dependency rather than an accidental Tabmat transitive dependency.

- [ ] Add a failing packaging test that parses the project and dev dependency sections and checks the exact ownership boundary:

  ```python
  def test_dataframe_boundary_dependencies_are_explicit() -> None:
      project = _toml_section("project")
      optional = _toml_section("project.optional-dependencies")

      assert '"narwhals>=2.17.0"' in project
      assert "polars" not in project
      assert '"polars>=1.42.1"' in optional
      assert "pyarrow" not in project
  ```

- [ ] Run the focused test and confirm it fails because Narwhals and Polars have not been declared directly:

  ```bash
  rtk uv run pytest tests/test_release_packaging.py::test_dataframe_boundary_dependencies_are_explicit -q
  ```

- [ ] Add `"narwhals>=2.17.0"` to `[project].dependencies` and `"polars>=1.42.1"` to the `dev` extra. Do not add Polars or PyArrow to runtime dependencies and do not change the locked Tabmat constraint.

- [ ] Refresh and validate the lock without broad upgrades:

  ```bash
  rtk uv lock
  rtk uv lock --check
  rtk uv sync --extra dev --python .venv/bin/python
  ```

- [ ] Re-run the packaging test and dependency check:

  ```bash
  rtk uv run pytest tests/test_release_packaging.py -q
  rtk uv pip check
  ```

- [ ] Inspect the lock diff and verify it adds the direct project edges without changing the resolved Tabmat or Narwhals versions:

  ```bash
  rtk git diff -- pyproject.toml uv.lock tests/test_release_packaging.py
  ```

- [ ] Commit this isolated dependency contract:

  ```bash
  rtk git add pyproject.toml uv.lock tests/test_release_packaging.py
  rtk git commit -m "Declare dataframe boundary dependencies"
  ```

## Task 2: Capture the pandas performance baseline before changing data access

**Files:**

- Create: `benchmarks/benchmark_dataframe_boundary.py`

**Invariant at risk:** A thin abstraction can still add repeated dispatch, conversion, or allocation. The before data must be captured while behavior still matches master.

- [ ] Add a deterministic subprocess-friendly benchmark harness with these named scenarios:

  - `ordinary_mixed_fit`: 6,000 rows, numeric columns plus one high-cardinality categorical.
  - `ordinary_scalar_fit`: 60,000 rows with many scalar numeric blocks.
  - `discrete_four_spline_fit`: 10,000 rows, four SSP splines, `discrete=True`.
  - `spline_reml`: a stable penalized spline fixture using bounded REML iterations.
  - `predict_exact`: 60,000 prediction rows after a warm fit.
  - `predict_fast_discrete`: 60,000 prediction rows through the fitted discrete predictor.

  The harness must accept `--backend pandas|polars`, `--scenario`, `--warmups`, `--repeats`, and `--output`. Each JSON record must contain the git SHA, Python/dependency versions, thread environment, scenario seed and dimensions, raw wall-time samples, median, median absolute deviation, traced Python peak, cold-process RSS delta, convergence, iteration count, deviance, EDF, coefficient checksum, prediction checksum, concrete group types, SplitMatrix build state, and actual counts for `sandwich`, `matvec`, `transpose_matvec`, stable grouped fallbacks, and compressed block kernels.

- [ ] Keep scenario construction separate from timed fitting. Collect plain wall-time samples without tracing or monkeypatch counters; collect `tracemalloc`, cold RSS, and actual kernel counters in separate fidelity runs so instrumentation cannot contaminate timing. Use `time.perf_counter_ns()`, `tracemalloc`, and `resource.getrusage`; do not add `psutil`.

- [ ] Add a `--smoke` mode that uses one warmup, one repeat, and reduced row counts while retaining every scenario's feature layout.

- [ ] Run the harness smoke check:

  ```bash
  rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --backend pandas --smoke --output /tmp/superglm-dataframe-boundary-smoke.json
  rtk json /tmp/superglm-dataframe-boundary-smoke.json
  ```

- [ ] Capture the authoritative pre-change pandas samples, keeping raw output outside the repository:

  ```bash
  rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --backend pandas --warmups 2 --repeats 7 --output /tmp/superglm-dataframe-boundary-before.json
  rtk json /tmp/superglm-dataframe-boundary-before.json
  ```

- [ ] Record the baseline path and SHA in the implementation session notes. Do not commit the generated JSON.

- [ ] Commit only the reproducible harness:

  ```bash
  rtk git add benchmarks/benchmark_dataframe_boundary.py
  rtk git commit -m "Benchmark dataframe boundary overhead"
  ```

## Task 3: Introduce the narrow eager-frame adapter

**Files:**

- Create: `src/superglm/_frame.py`
- Create: `tests/test_frame_adapter.py`

**Invariant at risk:** Backend support must remain a one-time boundary concern; dataframe objects and backend dispatch must not leak into feature implementations or numerical execution.

- [ ] Write failing tests covering:

  - pandas and eager Polars recognition;
  - idempotent `as_eager_frame(adapter)`;
  - clear rejection of NumPy and unrelated objects;
  - clear Polars `LazyFrame` rejection containing “eager” and “collect”;
  - ordered unique column names and missing-column reporting;
  - numeric, boolean, string, pandas categorical, Polars categorical, and Polars enum kinds;
  - logical labels rather than Polars physical categorical codes;
  - one raw extraction for repeated requests of the same column;
  - native pandas and Polars row slicing by positional integer indices;
  - backend-preserving native column selection;
  - an explicit error for empty native column selection, preventing silent Polars row-count loss;
  - no call to `polars.DataFrame.to_pandas`;
  - an intercept-only Polars fit using `features={}` and an otherwise unused native column;
  - the current pandas digest algorithm producing the same bytes through the adapter.

- [ ] Run the tests and confirm collection fails because `superglm._frame` does not exist:

  ```bash
  rtk uv run pytest tests/test_frame_adapter.py -q
  ```

- [ ] Implement a concrete adapter, not a registry or strategy hierarchy. The public shape should be:

  ```python
  from __future__ import annotations

  from dataclasses import dataclass, field
  from typing import TYPE_CHECKING, Literal, TypeAlias

  import narwhals.stable.v2 as nw
  import numpy as np
  import pandas as pd
  from numpy.typing import NDArray

  if TYPE_CHECKING:
      import polars as pl
      FrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
  else:
      FrameLike: TypeAlias = object

  FrameBackend = Literal["pandas", "polars"]
  ColumnKind = Literal["numeric", "boolean", "categorical", "unsupported"]

  @dataclass
  class EagerFrame:
      native: FrameLike
      backend: FrameBackend
      _frame: nw.DataFrame
      _arrays: dict[object, NDArray] = field(default_factory=dict, repr=False)
  ```

  The class must expose these exact contracts: `columns -> tuple[object, ...]`, `__len__() -> int`, `require_columns(names) -> None`, `column_kind(name) -> ColumnKind`, `column_array(name, *, dtype=None) -> NDArray`, `take_rows(indices) -> FrameLike`, `select_native(columns) -> FrameLike`, and `digest(columns=None) -> bytes`. The module must also expose `as_eager_frame(value) -> EagerFrame` and `is_supported_eager_frame(value) -> bool`.

- [ ] Implement `as_eager_frame()` with `nw.from_native(..., eager_only=True)`, then explicitly allow only pandas and Polars implementations. Translate Narwhals/backend exceptions into SuperGLM messages while chaining the original exception.

- [ ] Preserve the pandas extraction route with `Series.to_numpy(copy=False)`. Use Narwhals selected-series conversion for Polars. Cache only the raw array, and perform requested dtype coercion from that cached raw array.

- [ ] Keep pandas' existing digest algorithm unchanged: BLAKE2b personalization `superglm-fit-v1`, shape/dtype metadata, and `pd.util.hash_pandas_object(..., index=True, categorize=True)`. For Polars, hash ordered selected logical columns, schema strings, row count, and `hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3)`.

- [ ] Do not call Polars `select([])`: Polars cannot carry a positive row count in a native zero-column frame. Callers with no fitted feature columns must retain the original native frame and use an explicit empty feature configuration so all columns remain unused.

- [ ] Run focused tests, lint, and typing:

  ```bash
  rtk uv run pytest tests/test_frame_adapter.py -q
  rtk uv run ruff check src/superglm/_frame.py tests/test_frame_adapter.py
  rtk uv run ruff format --check src/superglm/_frame.py tests/test_frame_adapter.py
  rtk uv run ty check src/superglm/_frame.py
  ```

- [ ] Commit the boundary as an isolated primitive:

  ```bash
  rtk git add src/superglm/_frame.py tests/test_frame_adapter.py
  rtk git commit -m "Add eager dataframe boundary"
  ```

## Task 4: Route validation and feature compilation through the adapter

**Files:**

- Modify: `src/superglm/model/input_validation.py`
- Modify: `src/superglm/dm_builder.py`
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `tests/test_fit_input_validation.py`
- Create: `tests/test_dataframe_boundary.py`

**Invariant at risk:** Equivalent pandas and Polars data must compile to the same feature specs, `GroupInfo`, concrete `GroupMatrix` objects, penalties, constraints, and solver coordinates before any numerical algorithm runs.

- [ ] Add failing validation tests for all three transactional fit entry points using eager Polars, a Polars `LazyFrame`, missing columns, and row-count mismatch. Update the old ndarray error expectation from “pandas DataFrame” to “pandas or eager Polars DataFrame”.

- [ ] Add failing parity tests that build equivalent pandas and Polars fixtures containing numeric, boolean, string categorical, categorical/enum, spline, numeric-categorical interaction, spline-categorical interaction, and discrete tensor terms. Compare:

  ```python
  assert pandas_model._feature_order == polars_model._feature_order
  assert type(pandas_model._specs[name]) is type(polars_model._specs[name])
  assert [type(g) for g in pandas_model._dm.group_matrices] == [
      type(g) for g in polars_model._dm.group_matrices
  ]
  np.testing.assert_allclose(
      pandas_model._dm.toarray(),
      polars_model._dm.toarray(),
      rtol=0.0,
      atol=0.0,
  )
  ```

  Use exact equality for generated matrix structures where the same logical values and ordering imply the same operations. Retain existing nonzero tolerances only for fitted numerical results later.

- [ ] Add a spy test showing that a column used as both a main effect and interaction parent is extracted once from a Polars frame during one design build.

- [ ] Run the new tests and confirm they fail at the current pandas-only guard:

  ```bash
  rtk uv run pytest tests/test_fit_input_validation.py tests/test_dataframe_boundary.py -q -k 'polars or dataframe_boundary or extraction'
  ```

- [ ] Change `ValidatedFitInput.X` to `EagerFrame`. Normalize once at the beginning of `validate_fit_input()`, validate ordered uniqueness and required columns through the adapter, and preserve vector/domain validation unchanged.

- [ ] Preserve the existing rule that explicit-feature fits inspect only referenced columns while auto-detection inspects every candidate column. Keep the pandas mixed-object complex scan. Reject complex values universally and reject an unsupported logical dtype during auto-detection; an explicitly configured feature must still receive any logical array that its existing implementation already supports.

- [ ] Change `auto_detect_features()` and `build_design_matrix()` to accept `EagerFrame`. Replace only these three data reads:

  ```python
  x_col = X.column_array(name)
  x1 = X.column_array(p1)
  x2 = X.column_array(p2)
  ```

  Use `X.column_kind(col)` for auto-detection, mapping both `numeric` and `boolean` to the current `Numeric` behavior. Preserve the current `most_exposed` fallback, column order, feature construction calls, and interaction dispatch.

- [ ] Change the private `base.auto_detect()` and `base.model_build_design_matrix()` annotations and calls to carry `EagerFrame` only until `DesignMatrix` exists.

- [ ] Keep `X_ref = X` in `fit()`, `fit_path()`, and `fit_reml()` before validation. `_validate_entrypoint_input()` must return the adapter plus the normalized arrays, while `_prime_fit_caches()` must continue to receive the original native `X_ref`.

- [ ] Add an assertion that neither `DesignMatrix`, any `GroupMatrix`, nor `FitState` retains an `EagerFrame`, Narwhals frame, or native Polars frame.

- [ ] Run the focused compiler and transactional validation suites:

  ```bash
  rtk uv run pytest tests/test_frame_adapter.py tests/test_fit_input_validation.py tests/test_dataframe_boundary.py tests/test_fit_transactions.py -q
  rtk uv run pytest tests/test_interactions.py tests/test_discretize_fit.py tests/test_ordered_categorical.py -q -m 'not slow'
  rtk uv run ruff check src/superglm/_frame.py src/superglm/model/input_validation.py src/superglm/dm_builder.py src/superglm/model/base.py src/superglm/model/fit_ops.py tests/test_fit_input_validation.py tests/test_dataframe_boundary.py
  ```

- [ ] Inspect the diff and verify there are no changes below the `DesignMatrix` boundary:

  ```bash
  rtk git diff -- src/superglm/solvers src/superglm/reml src/superglm/_group_matrix
  ```

- [ ] Commit the compiler boundary:

  ```bash
  rtk git add src/superglm/model/input_validation.py src/superglm/dm_builder.py src/superglm/model/base.py src/superglm/model/fit_ops.py tests/test_fit_input_validation.py tests/test_dataframe_boundary.py
  rtk git commit -m "Compile pandas and Polars through one boundary"
  ```

## Task 5: Preserve prediction and retained-data safety for both backends

**Files:**

- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_data_guard.py`
- Modify: `src/superglm/model/runtime_canonicalize.py`
- Modify: `tests/test_dataframe_boundary.py`
- Modify: `tests/test_fit_data_guard.py`
- Modify: `tests/test_fit_ownership.py`
- Modify: `tests/test_runtime_canonicalization.py`

**Invariant at risk:** Exact and discrete prediction must use fitted feature state, while retained caller references must remain mutation-safe across refits, deepcopy, pickle, editor refresh, and compact-state release.

- [ ] Add failing pandas/Polars parity tests for `predict()`, `_predict_eta_exact()`, `_predict_fast_discrete()`, and exact versus fast-discrete prediction. Cover numeric, categorical, spline, spline-categorical, and tensor terms, including unseen categorical levels and a missing required column.

- [ ] Add a prediction extraction-count test: one interaction parent referenced by multiple fitted terms must be extracted once per prediction call.

- [ ] Add Polars guard tests for:

  - successful retained-data verification;
  - replacement by an equal independent Polars frame;
  - replacement by different values;
  - deepcopy and pickle round trips;
  - `retain_fit_state=False` releasing `_fit_X_ref` and `_fit_data_guard`;
  - a failed verification preserving the published fit revision;
  - the existing pandas mutation and writable-NumPy-alias tests remaining unchanged.

- [ ] Run the tests and confirm that prediction or guard code still performs direct pandas indexing:

  ```bash
  rtk uv run pytest tests/test_dataframe_boundary.py tests/test_fit_data_guard.py tests/test_fit_ownership.py tests/test_runtime_canonicalization.py -q -k 'polars or prediction or guard or canonical'
  ```

- [ ] Normalize once at `_predict_eta()` and pass the idempotent `EagerFrame` through exact and fast-discrete scorers. Replace every `np.asarray(X[name])` in prediction code with `X.column_array(name)` while preserving all fitted metadata, support discretization, scoring order, and offset handling.

- [ ] Normalize the retained native frame once in `_compile_fast_prediction_state()` and the two runtime-canonicalization paths. Do not replace `_fit_X_ref` with the adapter.

- [ ] Move digest ownership to `_frame.py`. Change `FitDataGuard` to store both `x_backend` and `x_digest`, and use this shape:

  ```python
  @dataclass(frozen=True)
  class FitDataGuard:
      x_backend: FrameBackend
      x_digest: bytes
      y_snapshot: NDArray[np.float64]
      x_columns: tuple[object, ...] | None = None

      @classmethod
      def capture(cls, X: FrameLike, y: NDArray, *, columns=None) -> FitDataGuard:
          frame = as_eager_frame(X)
          selected = None if columns is None else tuple(columns)
          return cls(
              x_backend=frame.backend,
              x_digest=frame.digest(selected),
              y_snapshot=_immutable_float64_copy(y),
              x_columns=selected,
          )
  ```

  `_matches_frame()` must reject a backend change and catch boundary/hash failures. Keep `matches()` and `matches_retained_values()` semantics distinct.

- [ ] Confirm `_prime_fit_caches()` still skips the O(n) digest for compact fits and captures only `model._feature_order`, so an unused unhashable pandas column remains irrelevant.

- [ ] Run focused prediction, ownership, canonicalization, and compact-state tests:

  ```bash
  rtk uv run pytest tests/test_dataframe_boundary.py tests/test_fit_data_guard.py tests/test_fit_ownership.py tests/test_fit_state_retention.py tests/test_runtime_canonicalization.py -q
  rtk uv run pytest tests/test_discretize_fit.py tests/test_interactions.py -q -k 'predict or fast or retained'
  rtk uv run ruff check src/superglm/_frame.py src/superglm/model/base.py src/superglm/model/fit_data_guard.py src/superglm/model/runtime_canonicalize.py tests/test_dataframe_boundary.py tests/test_fit_data_guard.py tests/test_fit_ownership.py tests/test_runtime_canonicalization.py
  ```

- [ ] Commit prediction and guard support:

  ```bash
  rtk git add src/superglm/_frame.py src/superglm/model/base.py src/superglm/model/fit_data_guard.py src/superglm/model/runtime_canonicalize.py tests/test_dataframe_boundary.py tests/test_fit_data_guard.py tests/test_fit_ownership.py tests/test_runtime_canonicalization.py
  rtk git commit -m "Preserve native frames through prediction and retention"
  ```

## Task 6: Prove all core fit and profile entry points are backend-neutral

**Files:**

- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/profile_ops.py`
- Modify: `tests/test_dataframe_boundary.py`
- Modify: `tests/test_fit_transactions.py`
- Modify: `tests/test_path.py`
- Modify: `tests/test_tweedie_profile.py`
- Modify: `tests/test_nb2.py`

**Invariant at risk:** `fit`, `fit_path`, `fit_reml`, Tweedie profiling, and NB profiling are transactional and publish native caller state only after the final authoritative fit succeeds.

- [ ] Add parameterized pandas/Polars parity tests for:

  - ordinary Gaussian and Poisson `fit()`;
  - mixed numeric/high-cardinality-categorical `fit()`;
  - four-spline `discrete=True` `fit()`;
  - spline `fit_reml()`;
  - `fit_path()` with an active group penalty;
  - `estimate_p()` on the committed stable interior Tweedie fixture;
  - `estimate_theta()` on a stable NB fixture.

  Compare coefficients, intercept, predictions, deviance, EDF, scale, lambda path, REML lambda/objective/rank, selected `p` or `theta`, convergence, iteration count, line-search trace decisions, and profile fallback reason using the tolerances already used by each subsystem's authority tests.

- [ ] Add a callback test asserting that a Tweedie trace callback receives the same public record type regardless of dataframe backend and cannot observe the private adapter.

- [ ] Add failed-refit tests that start from a successful pandas fit and fail on Polars input, and vice versa. Snapshot `model.__dict__`, `result`, `_fit_revision`, profile caches, and predictions before failure and assert atomic rollback.

- [ ] Run the tests and identify any nested profile call that incorrectly treats an `EagerFrame` as a public native frame:

  ```bash
  rtk uv run pytest tests/test_dataframe_boundary.py tests/test_fit_transactions.py tests/test_path.py tests/test_tweedie_profile.py tests/test_nb2.py -q -k 'polars or dataframe_backend or rollback'
  ```

- [ ] Make `as_eager_frame()` idempotence the only internal re-entry mechanism. Keep native `X_ref` captured at the outer public transaction and pass the adapter through profile workspaces and candidate fits without materializing a pandas frame.

- [ ] Change public `X` annotations in `model/api.py` to the private `FrameLike` alias and update docstrings to “pandas or eager Polars DataFrame”. Do not expose `EagerFrame` in the public API.

- [ ] Preserve Tweedie/NB profile candidate ordering, cache keys, coefficient-fit counts, exact-profile passes, final density validation, final refit ownership, and selected-family publication. No profiling kernel or solver file belongs in this commit.

- [ ] Run the full focused transactional and profile authority suites:

  ```bash
  rtk uv run pytest tests/test_dataframe_boundary.py tests/test_fit_transactions.py tests/test_path.py tests/test_profile_ci.py tests/test_tweedie_profile.py tests/test_tweedie_profile_reference.py tests/test_tweedie_reml_reference.py tests/test_nb2.py -q
  rtk uv run pytest tests/test_tweedie_profile_performance.py -q
  rtk uv run ruff check src/superglm/model/api.py src/superglm/model/profile_ops.py tests/test_dataframe_boundary.py tests/test_fit_transactions.py tests/test_path.py tests/test_tweedie_profile.py tests/test_nb2.py
  ```

- [ ] Commit core API parity:

  ```bash
  rtk git add src/superglm/model/api.py src/superglm/model/profile_ops.py tests/test_dataframe_boundary.py tests/test_fit_transactions.py tests/test_path.py tests/test_tweedie_profile.py tests/test_nb2.py
  rtk git commit -m "Support native frames across fit and profiling"
  ```

## Task 7: Preserve native frames through cross-validation

**Files:**

- Modify: `src/superglm/model_selection.py`
- Modify: `src/superglm/plotting/curve_similarity.py`
- Modify: `tests/test_cross_validate.py`
- Modify: `tests/test_curve_similarity.py`

**Invariant at risk:** Splitters and scorers are user callbacks. They must see the caller's native frame, while positional folds, OOF placement, model cloning, and output tables remain unchanged.

- [ ] Add a custom splitter that records `type(X)` and deterministic indices, plus a custom scorer that records `type(X_val)`. Parameterize over pandas and Polars and assert both callbacks see the native backend.

- [ ] Add parity tests for built-in deviance/NLL scoring, OOF predictions, returned estimators, fold indices, pooled scores, and `plot_terms_by_fold()` support selection.

- [ ] Add an error-path test proving a failed Polars fold obeys `error_score` without mutating the input model.

- [ ] Run the focused tests and confirm `.iloc` is the current failure point:

  ```bash
  rtk uv run pytest tests/test_cross_validate.py tests/test_curve_similarity.py -q -k 'polars or native_frame or fold'
  ```

- [ ] Normalize once in `cross_validate()` and once in `CrossValidationResult.plot_terms_by_fold()`. Keep `cv.split(X, y, groups)` on the original native object; replace fold slicing with:

  ```python
  frame = as_eager_frame(X)
  X_train = frame.take_rows(train_idx)
  X_test = frame.take_rows(test_idx)
  ```

- [ ] Use the same native slicing helper for fold-support data passed to curve comparison. Keep `fold_scores` and every aggregate output as pandas objects.

- [ ] Run the complete cross-validation suite:

  ```bash
  rtk uv run pytest tests/test_cross_validate.py tests/test_curve_similarity.py -q
  rtk uv run ruff check src/superglm/model_selection.py src/superglm/plotting/curve_similarity.py tests/test_cross_validate.py tests/test_curve_similarity.py
  ```

- [ ] Commit native fold behavior:

  ```bash
  rtk git add src/superglm/model_selection.py src/superglm/plotting/curve_similarity.py tests/test_cross_validate.py tests/test_curve_similarity.py
  rtk git commit -m "Keep dataframe backends native in cross-validation"
  ```

## Task 8: Route inference, diagnostics, and shape repair through the boundary

**Files:**

- Modify: `src/superglm/inference/_metrics_design.py`
- Modify: `src/superglm/inference/_term_model_ops.py`
- Modify: `src/superglm/diagnostics/discretize.py`
- Modify: `src/superglm/diagnostics/spline_checks.py`
- Modify: `src/superglm/diagnostics/term_diagnostics.py`
- Modify: `src/superglm/model/shape_ops.py`
- Modify: `src/superglm/debug_weights.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_model_tests.py`
- Modify: `tests/test_diagnostics.py`
- Modify: `tests/test_discretize.py`
- Modify: `tests/test_shape_postfit.py`
- Modify: `tests/test_debug_weights.py`

**Invariant at risk:** Secondary operations must evaluate the same frozen prediction/design contracts without building a second pandas-only API, allocating a full observation-by-coefficient matrix, or publishing an unsafe repaired fit.

- [ ] Add pandas/Polars parity tests for metrics, `drop1()`, `refit_unpenalised()`, selected-column chunked runtime design, model tests, `plot_diagnostics()`, term importance, term-drop refit and holdout modes, spline redundancy, discretization impact, debug-weight summaries, and successful shape post-fit repair.

- [ ] Add a failed Polars shape-repair test that verifies the prior coefficients, intercept, covariance/profile caches, fit revision, and predictions remain unchanged.

- [ ] Add a chunking test that monkeypatches `_exact_runtime_design_block` and proves `_ChunkedExactDesign.iter_dense_chunks()` receives backend-preserving slices and never calls whole-frame conversion.

- [ ] Run focused tests and confirm direct `X[name]`/`.iloc` sites fail:

  ```bash
  rtk uv run pytest tests/test_metrics.py tests/test_model_tests.py tests/test_diagnostics.py tests/test_discretize.py tests/test_shape_postfit.py tests/test_debug_weights.py -q -k 'polars or chunk or rollback'
  ```

- [ ] Store an `EagerFrame` in the operation-local `_ChunkedExactDesign`, use `take_rows(np.arange(start, stop))` for chunks, and use `column_array()` in `_exact_runtime_design_block()`. Do not materialize the complete runtime design.

- [ ] Normalize each public diagnostic operation once and pass the idempotent adapter to nested helpers. Replace pandas dtype checks with `column_kind()` and raw column reads with `column_array()`.

- [ ] Keep all diagnostic/report outputs pandas. Where existing code constructs a temporary pandas table from one selected column, build it from `column_array()` rather than converting the native frame.

- [ ] In `shape_ops.apply_shape_postfit()`, normalize only after retained-data verification and before certificate/proposal construction. Preserve exact polynomial-span certificates, objective ordering, canonical intercept synchronization, invalidation scope, and atomic publication.

- [ ] Run the complete affected suites:

  ```bash
  rtk uv run pytest tests/test_metrics.py tests/test_model_tests.py tests/test_diagnostics.py tests/test_discretize.py tests/test_shape_postfit.py tests/test_shape_fit.py tests/test_debug_weights.py -q
  rtk uv run ruff check src/superglm/inference/_metrics_design.py src/superglm/inference/_term_model_ops.py src/superglm/diagnostics src/superglm/model/shape_ops.py src/superglm/debug_weights.py tests/test_metrics.py tests/test_model_tests.py tests/test_diagnostics.py tests/test_discretize.py tests/test_shape_postfit.py tests/test_debug_weights.py
  ```

- [ ] Commit secondary numerical API support:

  ```bash
  rtk git add src/superglm/inference/_metrics_design.py src/superglm/inference/_term_model_ops.py src/superglm/diagnostics src/superglm/model/shape_ops.py src/superglm/debug_weights.py tests/test_metrics.py tests/test_model_tests.py tests/test_diagnostics.py tests/test_discretize.py tests/test_shape_postfit.py tests/test_debug_weights.py
  rtk git commit -m "Use the dataframe boundary in diagnostics and repair"
  ```

## Task 9: Keep plotting backend-neutral and outputs unchanged

**Files:**

- Modify: `src/superglm/model/plot_ops.py`
- Modify: `src/superglm/plotting/data.py`
- Modify: `src/superglm/plotting/comparison.py`
- Modify: `src/superglm/plotting/interactions.py`
- Modify: `src/superglm/plotting/group_display.py`
- Modify: `src/superglm/plotting/main_effects.py`
- Modify: `src/superglm/plotting/main_effects_plotly.py`
- Modify: `tests/test_plot_api.py`
- Modify: `tests/test_plot_comparison.py`
- Modify: `tests/test_interaction_plots.py`
- Modify: `tests/test_plot_diagnostics.py`

**Invariant at risk:** Plot support/density calculations must use native input semantics while preserving all existing plot payloads, exposure weights, grouped levels, surfaces, confidence intervals, and matplotlib/Plotly output types.

- [ ] Add pandas/Polars parity tests for `plot_data()` on numeric, categorical, spline, and tensor interactions. Compare every numeric array and ordered label list in the plain payload before testing renderers.

- [ ] Add smoke tests for matplotlib and Plotly main-effect, comparison, and interaction plots using Polars support data. Assert the same figure/axes types and trace counts as pandas.

- [ ] Add a spy that makes `polars.DataFrame.to_pandas()` fail and verifies plot support still succeeds.

- [ ] Run focused tests and confirm the current failures arise from `.to_numpy()`, `.astype()`, and pandas dtype checks on native Polars columns:

  ```bash
  rtk uv run pytest tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_plot_diagnostics.py -q -k 'polars or dataframe_backend'
  ```

- [ ] Normalize optional `X` once in `model/plot_ops.py`. Leaf plotting functions may accept `EagerFrame | None`; they must use `column_array()` and `column_kind()` rather than indexing the native object.

- [ ] Where a renderer needs a pandas `Series` for grouping or display, construct that series from one extracted logical array:

  ```python
  levels = pd.Series(frame.column_array(name), name=name).astype(str)
  ```

  Do not pass the complete input frame into a pandas constructor.

- [ ] Preserve all existing categorical ordering, density weighting, grid construction, knot placement, interaction orientation, CI calculation, and rendering code after the data-read boundary.

- [ ] Run all plotting suites:

  ```bash
  rtk uv run pytest tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_plot_diagnostics.py -q
  rtk uv run ruff check src/superglm/model/plot_ops.py src/superglm/plotting tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_plot_diagnostics.py
  ```

- [ ] Commit plotting support:

  ```bash
  rtk git add src/superglm/model/plot_ops.py src/superglm/plotting tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_plot_diagnostics.py
  rtk git commit -m "Support native frames in plotting"
  ```

## Task 10: Preserve pandas reporting while supporting Polars export and editor data

**Files:**

- Modify: `src/superglm/export/rating_tables.py`
- Modify: `src/superglm/editor/apply.py`
- Modify: `src/superglm/editor/collapse.py`
- Modify: `src/superglm/editor/evaluation.py`
- Modify: `src/superglm/editor/terms.py`
- Modify: `tests/test_rating_table_export.py`
- Modify: `tests/test_ordered_reference_export.py`
- Modify: `tests/test_editor.py`
- Modify: `tests/test_editor_evaluation_cache.py`

**Invariant at risk:** Rating-table and editor workflows consume retained fit data, but their external records and files are pandas/table-oriented. Backend support must not weaken retained-data checks or change deployment output.

- [ ] Add pandas/Polars parity tests for rating-table payloads and exported CSV/XLSX contents, including categorical exposure weights, numeric blocks, interactions, fitted offsets, explicit offset-source columns, and ordered categorical references.

- [ ] Add editor tests fitting from Polars and exercising term loading, category collapse proposals, cached evaluation, apply/refit, undo/redo state, and retained-data verification. Assert public snapshots and evidence tables match pandas fixtures.

- [ ] Run the new focused tests and confirm direct retained-frame indexing is the failure source:

  ```bash
  rtk uv run pytest tests/test_rating_table_export.py tests/test_ordered_reference_export.py tests/test_editor.py tests/test_editor_evaluation_cache.py -q -k 'polars or native_frame'
  ```

- [ ] Normalize rating-table input once in `build_rating_table_payload()`. Replace direct reads with `column_array()` and construct only the existing output pandas `Series`/`DataFrame` objects. Preserve block order, labels, multipliers, impact sweeps, offset identity checks, and writer behavior.

- [ ] In editor code, keep `_fit_X_ref` native. Use `as_eager_frame()` at the small number of retained-data read sites in `collapse.py` and `terms.py`; use native row count in evaluation. Do not serialize the adapter into editor session state.

- [ ] Preserve editor transaction, generation, evidence-cache, revision, and rollback behavior. This task changes data access only; no frontend files or browser tests should change.

- [ ] Run the complete affected suites:

  ```bash
  rtk uv run pytest tests/test_rating_table_export.py tests/test_ordered_reference_export.py tests/test_editor.py tests/test_editor_evaluation_cache.py -q
  rtk uv run ruff check src/superglm/export/rating_tables.py src/superglm/editor tests/test_rating_table_export.py tests/test_ordered_reference_export.py tests/test_editor.py tests/test_editor_evaluation_cache.py
  ```

- [ ] Commit export/editor support:

  ```bash
  rtk git add src/superglm/export/rating_tables.py src/superglm/editor tests/test_rating_table_export.py tests/test_ordered_reference_export.py tests/test_editor.py tests/test_editor_evaluation_cache.py
  rtk git commit -m "Keep reporting outputs stable across frame backends"
  ```

## Task 11: Extend the scikit-learn boundary without changing ndarray mode

**Files:**

- Modify: `src/superglm/sklearn.py`
- Modify: `tests/test_sklearn.py`
- Modify: `tests/test_sklearn_classifier.py`

**Invariant at risk:** The wrappers support named-frame and synthetic ndarray modes with different feature-reference rules. Polars must join named-frame mode without changing ndarray/sparse behavior or sklearn attributes.

- [ ] Add pandas/Polars parity tests for regressor and classifier fit/predict, auto-detection, explicit features, string/integer feature references, categorical and spline shorthands, one or multiple offset columns, `feature_names_in_`, `n_features_in_`, `coef_`, `intercept_`, `predict_proba()`, and `decision_function()`.

- [ ] Add regression tests proving NumPy and SciPy sparse inputs still convert to synthetic-name pandas frames and retain their existing validation errors.

- [ ] Run the tests and confirm `_normalize_X()` and dataframe-only slicing fail on Polars:

  ```bash
  rtk uv run pytest tests/test_sklearn.py tests/test_sklearn_classifier.py -q -k 'polars or dataframe or ndarray or sparse'
  ```

- [ ] Refactor `_normalize_X()` to return `FrameLike`, ordered names, and the existing synthetic-name flag. For supported native frames, use the adapter for names and return the same backend; for NumPy/SciPy inputs, keep the current pandas construction exactly.

- [ ] Refactor `_resolve_offset()` to extract selected offset columns through `column_array()`. Refactor non-empty fit/predict feature selection through `select_native(tuple(feature_cols))` so Polars remains Polars. When every column is an offset, retain the original native frame, force `features={}`, and let the explicit empty feature configuration ignore those columns; never call Polars `select([])`.

- [ ] Replace `input_is_dataframe` with an explicit `input_is_named_frame` boolean. Pass an adapter to `_build_features_or_splines()` for dtype-based shorthand detection; do not teach that helper about Polars directly.

- [ ] Preserve the existing defensive pandas copy on named-frame fitting only if an existing test proves it is part of wrapper ownership. For Polars, retain the immutable native object without copying.

- [ ] Run sklearn estimator checks already present in the suite, plus lint:

  ```bash
  rtk uv run pytest tests/test_sklearn.py tests/test_sklearn_classifier.py tests/test_penalties.py -q
  rtk uv run ruff check src/superglm/sklearn.py tests/test_sklearn.py tests/test_sklearn_classifier.py
  ```

- [ ] Commit wrapper support:

  ```bash
  rtk git add src/superglm/sklearn.py tests/test_sklearn.py tests/test_sklearn_classifier.py
  rtk git commit -m "Accept Polars in sklearn wrappers"
  ```

## Task 12: Add a no-work fitted-design summary with authoritative route reasons

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_execution.py`
- Create: `src/superglm/model/design_summary.py`
- Modify: `src/superglm/model/api.py`
- Modify: `tests/test_matrix_execution_plan.py`
- Create: `tests/test_design_summary.py`

**Invariant at risk:** Developers need inspectability, but merely asking for a summary must not build a SplitMatrix, execute a kernel, construct a compressed mixed-bin plan, or create a second dispatch policy.

- [ ] Add failing `MatrixExecutionPlan` tests for a read-only `ordinary_indices` property and stable `ordinary_partition_reason` values covering:

  - explicitly disabled policy;
  - explicitly forced candidates;
  - a compressed/mixed layout;
  - no native categorical;
  - more than one categorical;
  - categorical width at or below 100;
  - dense width below 3;
  - any sparse block;
  - row count below 50,000;
  - the currently certified automatic layout.

- [ ] Assert every case retains the exact current `_ordinary_indices` result. Assert reading the properties leaves `_ordinary_split_built` false.

- [ ] Add failing `design_summary()` tests for pre-fit rejection and fitted numeric, categorical, sparse SSP, spline-categorical, discretized SSP, discretized SCOP, discretized spline-categorical, and tensor groups. Require this exact output schema:

  ```python
  EXPECTED_COLUMNS = [
      "term",
      "feature",
      "solver_start",
      "solver_end",
      "n_columns",
      "representation",
      "compressed",
      "storage_rows",
      "ordinary_tabmat_partition",
      "specialised_discrete_route",
      "route_reason",
  ]
  ```

- [ ] Snapshot the lazy-plan flags before `design_summary()` and assert they are unchanged afterward:

  ```python
  before = (
      model._dm._tabmat_built,
      model._dm._mixed_bin_space_centering_plan_attempted,
      model._dm.raw_spline_tabmat_plan_built,
  )
  model.design_summary()
  after = (
      model._dm._tabmat_built,
      model._dm._mixed_bin_space_centering_plan_attempted,
      model._dm.raw_spline_tabmat_plan_built,
  )
  assert after == before
  ```

  Include one freshly constructed design whose three flags begin false, so the test also proves summary inspection does not trigger first construction.

- [ ] Run the tests and confirm the properties and method are absent:

  ```bash
  rtk uv run pytest tests/test_matrix_execution_plan.py tests/test_design_summary.py -q -k 'partition_reason or design_summary'
  ```

- [ ] Replace `_eligible_ordinary_indices()` internally with one construction-time decision record while preserving exact predicate and gate order:

  ```python
  @dataclass(frozen=True)
  class OrdinaryPartitionDecision:
      indices: tuple[int, ...]
      reason: str

  decision = self._ordinary_partition_decision()
  self._ordinary_indices = frozenset(decision.indices)
  self._ordinary_partition_reason = decision.reason
  ```

  Expose immutable properties only. Use stable reason tokens such as `policy-disabled`, `policy-forced`, `contains-compressed-group`, `categorical-layout`, `dense-width`, `contains-sparse-group`, `row-threshold`, and `auto-certified`. Do not change `_MIN_AUTO_TABMAT_MOMENT_ROWS` or any condition.

- [ ] Implement `build_design_summary(model) -> pd.DataFrame` by zipping `model._groups`, `model._dm.group_matrices`, and `model._dm.execution_plan.group_spans`. Use concrete class-to-representation metadata in one module-level immutable mapping.

- [ ] Define `compressed=True` only for `DiscretizedSSPGroupMatrix`, `DiscretizedSCOPGroupMatrix`, `DiscretizedSplineCategoricalGroupMatrix`, and `DiscretizedTensorGroupMatrix`. Derive `storage_rows` from `B_unique.shape[0]`, `B_scop_unique.shape[0]`, spline support, or observed tensor-pair support without calling `toarray()`.

- [ ] For ordinary groups, use the execution plan's stored decision reason. For compressed groups, report their specialised class route and why the ordinary partition excludes them. Do not claim that eligibility proves a kernel call; say this explicitly in the method docstring.

- [ ] Add `SuperGLM.design_summary()` as a read-only API method returning a newly constructed pandas table. Before first fit it must raise `RuntimeError("Model must be fitted before calling design_summary().")`. A fitted compact model that deliberately released `_dm` must raise a distinct error explaining that `retain_fit_state=False` discarded the fitted design; do not retain a second summary state solely to support that case.

- [ ] Run matrix execution, summary, pickle, and negative-control tests:

  ```bash
  rtk uv run pytest tests/test_matrix_execution_plan.py tests/test_design_summary.py tests/test_mixed_bin_space_centering.py tests/test_spline_tabmat_centering.py -q
  rtk uv run ruff check src/superglm/_group_matrix/_group_matrix_execution.py src/superglm/model/design_summary.py src/superglm/model/api.py tests/test_matrix_execution_plan.py tests/test_design_summary.py
  ```

- [ ] Commit inspectability without execution work:

  ```bash
  rtk git add src/superglm/_group_matrix/_group_matrix_execution.py src/superglm/model/design_summary.py src/superglm/model/api.py tests/test_matrix_execution_plan.py tests/test_design_summary.py
  rtk git commit -m "Explain fitted matrix routes without executing them"
  ```

## Task 13: Document the two-layer developer story and optional-backend contract

**Files:**

- Create: `docs/development/data-and-solver-boundaries.md`
- Modify: `docs/getting-started/installation.md`
- Modify: `docs/getting-started/quickstart.md`
- Modify: `docs/api/model.md`
- Modify: `mkdocs.yml`
- Modify: `tests/test_import_compat.py`
- Modify: `tests/test_release_packaging.py`

**Invariant at risk:** The architecture only lowers cognitive load if a maintainer can find the boundary and knows which layer owns a change. Optional Polars support must not become an import-time requirement.

- [ ] Add a subprocess import test that installs a `MetaPathFinder` rejecting `polars` and `polars.*`, then imports SuperGLM, fits a tiny pandas model, and predicts. The test must prove package import and pandas use do not import Polars.

- [ ] Add documentation tests asserting that installation docs name pandas and eager Polars, explain `LazyFrame.collect()`, state that outputs remain pandas, and link the permanent developer page from `mkdocs.yml`.

- [ ] Run the tests and confirm the docs/import contract is absent:

  ```bash
  rtk uv run pytest tests/test_import_compat.py tests/test_release_packaging.py -q -k 'polars or dataframe_boundary or installation'
  ```

- [ ] Write `docs/development/data-and-solver-boundaries.md` with exactly two visual layers:

  ```text
  User layer
  pandas.DataFrame | eager polars.DataFrame
                       -> fit / REML / profile / predict / metrics / CV

  Developer layer
  native frame
      -> EagerFrame boundary
      -> feature compiler
      -> DesignMatrix + GroupMatrix blocks
      -> construction-time execution plans
      -> IRLS / PIRLS / REML
  ```

- [ ] Add a “Where to make a change” table with these ownership rows:

  | Change | Owner |
  | --- | --- |
  | Accept/normalize a dataframe dtype | `superglm._frame` |
  | Change a basis or categorical encoding | feature spec and `dm_builder` |
  | Change block storage | `GroupMatrix` construction |
  | Change Gram/matvec/centering | group-matrix algebra/execution plan |
  | Change working responses or weights | working-row geometry |
  | Change coefficient iteration or line search | IRLS/PIRLS solver |
  | Change smoothing selection | REML objective/update/finalization |
  | Add a future AFT objective | response/objective/working-geometry contract, then only required kernels |

- [ ] Explain `design_summary()` versus actual traces: summary reports immutable storage and construction-time eligibility; fit/REML/profile traces and kernel call counters report execution. Include one pandas example and one Polars example.

- [ ] Document future AFT as a conceptual path only: validation of time/event/censoring data, objective value/gradient/curvature or working geometry, compatible coefficient solver, specialised kernels, terminal publication. Explicitly state that this branch adds no AFT API or generic objective framework.

- [ ] Update installation/quickstart docs with:

  ```python
  import polars as pl
  from superglm import SuperGLM

  X = pl.DataFrame({"age": [20.0, 30.0, 40.0], "region": ["N", "S", "N"]})
  model = SuperGLM(family="poisson", splines=["age"]).fit(X, y)
  predictions = model.predict(X)
  ```

  State that `pl.LazyFrame` must be collected by the caller and that Polars itself is installed separately.

- [ ] Add the page under the MkDocs `Development` navigation and ensure `design_summary()` appears in the model API page.

- [ ] Run documentation, import, and API checks:

  ```bash
  rtk uv run pytest tests/test_import_compat.py tests/test_release_packaging.py tests/test_design_summary.py -q
  rtk uv sync --group docs --extra plotting --extra dev --python .venv/bin/python
  rtk uv run mkdocs build --strict
  rtk uv run ruff check tests/test_import_compat.py tests/test_release_packaging.py
  ```

- [ ] Commit the public/developer documentation:

  ```bash
  rtk git add docs/development/data-and-solver-boundaries.md docs/getting-started/installation.md docs/getting-started/quickstart.md docs/api/model.md mkdocs.yml tests/test_import_compat.py tests/test_release_packaging.py
  rtk git commit -m "Document dataframe and solver boundaries"
  ```

## Task 14: Audit all remaining dataframe assumptions and type surfaces

**Files:**

- Modify: every production file from Tasks 3–13 that still has an incorrect `X: pd.DataFrame` input annotation
- Modify: affected tests only when the annotation or documented error is part of their assertion

**Invariant at risk:** A partial boundary leaves hidden pandas-only branches and misleads maintainers about where backend semantics end.

- [ ] Search for remaining direct model-input assumptions:

  ```bash
  rtk grep -n 'X: pd.DataFrame' src/superglm
  rtk grep -n 'X\[' src/superglm/model src/superglm/inference src/superglm/diagnostics src/superglm/plotting src/superglm/export src/superglm/editor src/superglm/debug_weights.py
  rtk grep -n '\.iloc' src/superglm/model_selection.py src/superglm/inference src/superglm/plotting src/superglm/editor
  rtk grep -n 'isinstance(X, pd.DataFrame)' src/superglm
  rtk grep -n 'to_pandas' src/superglm
  ```

- [ ] Classify every hit. Keep pandas annotations and operations only where pandas is deliberately the output or an internally constructed report table. Route every native model-input read through `EagerFrame`.

- [ ] Ensure the only runtime imports of Polars are in tests/benchmarks; production may reference Polars only under `TYPE_CHECKING`.

- [ ] Run Astral ty over the package. Fix diagnostics introduced in changed files without expanding the unrelated repository-wide backlog:

  ```bash
  rtk uv run ty check src/superglm
  ```

- [ ] Run Ruff and formatting over every changed Python file:

  ```bash
  rtk uv run ruff check src/ tests/ benchmarks/benchmark_dataframe_boundary.py
  rtk uv run ruff format --check src/ tests/ benchmarks/benchmark_dataframe_boundary.py
  ```

- [ ] Inspect the production diff specifically for forbidden solver leakage and whole-frame conversions:

  ```bash
  rtk git diff origin/master...HEAD -- src/superglm/solvers src/superglm/reml
  rtk git diff origin/master...HEAD -- src/superglm
  ```

- [ ] Commit only concrete annotation/boundary corrections found by the audit:

  ```bash
  rtk git add src tests
  rtk git commit -m "Complete dataframe boundary audit"
  ```

  If the audit produces no code change, do not create an empty commit.

## Task 15: Prove numerical, dispatch, memory, and end-to-end performance parity

**Files:**

- Modify: `benchmarks/benchmark_dataframe_boundary.py` only if the implemented Polars constructor or comparison report needs its post-boundary branch completed
- Do not commit generated benchmark JSON

**Invariant at risk:** Dataframe accessibility is not complete if pandas becomes slower, Tabmat/discrete dispatch disappears, compressed designs expand, numerical results diverge, or profiling performs extra fits/evaluations.

- [ ] Run the post-change pandas benchmark with the same interpreter, thread limits, warmups, repeats, seeds, and scenarios as the baseline:

  ```bash
  rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --backend pandas --warmups 2 --repeats 7 --output /tmp/superglm-dataframe-boundary-after-pandas.json
  ```

- [ ] Run the equivalent eager-Polars benchmark:

  ```bash
  rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --backend polars --warmups 2 --repeats 7 --output /tmp/superglm-dataframe-boundary-after-polars.json
  ```

- [ ] Use the harness comparison mode to print raw samples, medians, median absolute deviation, percentage change, memory change, group types, iteration counts, and numerical checksums. Make it exit nonzero for a stable microbenchmark regression above 3%, an end-to-end regression above 5%, a meaningful memory increase, or a structural/numerical mismatch:

  ```bash
  rtk .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --compare /tmp/superglm-dataframe-boundary-before.json /tmp/superglm-dataframe-boundary-after-pandas.json
  rtk .venv/bin/python benchmarks/benchmark_dataframe_boundary.py --compare-backends /tmp/superglm-dataframe-boundary-after-pandas.json /tmp/superglm-dataframe-boundary-after-polars.json
  ```

- [ ] If a threshold fails, rerun the affected scenario in counterbalanced order. Do not change a threshold, row count, fixture, or numerical tolerance to make it pass. Profile the boundary and remove repeated extraction/dispatch/allocation before proceeding.

- [ ] Run actual dispatch/call-count and negative-control suites:

  ```bash
  rtk uv run pytest tests/test_matrix_execution_plan.py tests/test_irls_direct.py tests/test_mixed_bin_space_centering.py tests/test_spline_tabmat_centering.py tests/test_fit_state_trace_benchmark.py -q
  ```

- [ ] Run focused correctness across every touched surface:

  ```bash
  rtk uv run pytest tests/test_frame_adapter.py tests/test_dataframe_boundary.py tests/test_fit_input_validation.py tests/test_fit_data_guard.py tests/test_fit_ownership.py tests/test_fit_transactions.py tests/test_runtime_canonicalization.py tests/test_path.py tests/test_cross_validate.py tests/test_curve_similarity.py tests/test_metrics.py tests/test_model_tests.py tests/test_diagnostics.py tests/test_discretize.py tests/test_shape_postfit.py tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_plot_diagnostics.py tests/test_rating_table_export.py tests/test_ordered_reference_export.py tests/test_editor.py tests/test_editor_evaluation_cache.py tests/test_sklearn.py tests/test_sklearn_classifier.py tests/test_design_summary.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py tests/test_nb2.py -q
  ```

- [ ] Run the established package gates:

  ```bash
  rtk uv run ruff check src/ tests/ benchmarks/benchmark_dataframe_boundary.py
  rtk uv run ruff format --check src/ tests/ benchmarks/benchmark_dataframe_boundary.py
  rtk uv lock --check
  rtk uv pip check
  rtk uv run ty check src/superglm
  rtk uv run python run_test.py
  rtk uv run pytest tests/ -q -m 'not slow and not browser'
  rtk uv run pytest tests/ -q -m 'not browser'
  rtk uv build
  rtk uv run mkdocs build --strict
  ```

- [ ] Inspect built artifacts and run a wheel smoke import:

  ```bash
  rtk ls dist
  rtk uv venv --clear --python /usr/bin/python3.14 /tmp/superglm-dataframe-wheel
  rtk uv pip install --python /tmp/superglm-dataframe-wheel/bin/python dist/*.whl
  rtk /tmp/superglm-dataframe-wheel/bin/python -c 'import pandas as pd; import superglm; print(superglm.__version__)'
  ```

- [ ] Review the final diff for accidental generated files, benchmark JSON, notebook churn, dependency upgrades, solver edits, tolerance changes, skips, or fixture reductions:

  ```bash
  rtk git status --short
  rtk git diff --check origin/master...HEAD
  rtk git diff --stat origin/master...HEAD
  rtk git diff origin/master...HEAD -- pyproject.toml uv.lock src tests benchmarks docs mkdocs.yml
  ```

- [ ] If the benchmark harness required a final Polars/comparison change, commit it after its smoke test:

  ```bash
  rtk git add benchmarks/benchmark_dataframe_boundary.py
  rtk git commit -m "Complete dataframe boundary benchmarks"
  ```

- [ ] Record in the final implementation report:

  - base and final SHAs;
  - Python, NumPy, SciPy, pandas, Polars, Narwhals, Tabmat, Numba, and BLAS/thread configuration;
  - exact test commands and results;
  - raw benchmark file paths and before/after tables;
  - maximum coefficient, prediction, deviance, EDF, lambda, objective, rank, scale, profile-parameter, density, and derivative differences;
  - matrix class and storage parity;
  - actual Tabmat/specialised-kernel call counts;
  - confirmation that no whole-frame conversion, compressed row materialization, added profile pass, added coefficient fit, tolerance weakening, skipped test, fallback reordering, or meaningful memory regression occurred.

## Definition of done

This plan is complete only when:

- pandas and eager Polars work through every named public model-data entry point covered above;
- `pl.LazyFrame` fails early with collection guidance;
- custom splitters/scorers receive native frames and all public table outputs remain pandas;
- dataframe dispatch is confined to `superglm._frame` and operation boundaries;
- feature specs and solver code continue to receive arrays and exact compiled matrix objects;
- equivalent inputs compile to the same concrete matrix structures and fitted numerical state;
- retained-data fingerprints, rollback, clone, pickle, and compact-state invariants pass for both backends;
- `design_summary()` explains storage and eligibility without executing or allocating an accelerated route;
- the permanent two-layer guide makes the future AFT change graph visible without adding AFT machinery;
- pandas timing, memory, Tabmat gains, discrete gains, and Tweedie profiling gains meet their gates;
- all focused, non-slow, full non-browser, lint, lock, packaging, build, and documentation checks complete with no new changed-file type diagnostics.
