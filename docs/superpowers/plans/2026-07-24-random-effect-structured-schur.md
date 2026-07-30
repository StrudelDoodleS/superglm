# RandomEffect and Scalar Structured Schur Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver `RandomEffect` with exact REML variance-component estimation, conditional and
population prediction, actuarial credibility reporting, and a tabmat-preserving scalar Schur
backend that is exact for both ordinary and `discrete=True` fits.

**Architecture:** Add an all-level categorical matrix and implicit identity penalty as the dense
oracle first. Then introduce a private Hessian-factor protocol and scalar Schur factor that
eliminates one dominant random-effect block while retaining every other term in a smaller dense
system. Exact REML, W corrections, discrete cached-W trials, inference, and reporting consume
factor operations instead of requiring a full inverse. `direct_solve="gram"` remains the oracle,
`"structured"` forces the new backend, and `"auto"` uses a benchmark-derived crossover.

**Tech Stack:** Python 3.13, NumPy, SciPy, pandas, numba, tabmat, pytest, Ruff, mypy, cProfile,
`superglm.profiling.harness`, and pinned the reference implementation reference fixtures.

---

## Scope and invariants

- This is Delivery A only. `FactorSmooth` is planned separately after this plan's review
  checkpoint.
- Do not modify LSS code, tests, documentation, branches, or worktrees.
- Do not change ordinary `Categorical` or `SplineCategorical` semantics.
- Do not allocate `np.eye(K)` for a random effect, an `n x K` one-hot matrix on the structured
  path, or a `K x K` random-effect Hessian/inverse on the structured path.
- `RandomEffect` is supported by `fit_reml()` only.
- `discrete=True` leaves random-effect factor codes exact while existing spline/tensor
  discretization remains unchanged.
- Every behavior change starts with a failing focused test and ends with a focused commit.

## File map

New production files:

- `src/superglm/features/random_effect.py`
  - public `RandomEffect` feature and level-policy logic.
- `src/superglm/solvers/hessian_factor.py`
  - dense adapter, factor protocol, penalty/operator queries.
- `src/superglm/solvers/structured.py`
  - scalar structured Gram cache, Schur factor, backend selection, and cache re-solves.
- `src/superglm/inference/random_effects.py`
  - `RandomEffectResult`, level table construction, credibility, and conditional unpooled update.

New focused tests:

- `tests/test_random_effect.py`
- `tests/test_random_effect_matrix.py`
- `tests/test_random_effect_penalty.py`
- `tests/test_structured_factor.py`
- `tests/test_structured_irls.py`
- `tests/test_random_effect_reml.py`
- `tests/test_random_effect_discrete.py`
- `tests/test_random_effect_inference.py`
- `tests/test_random_effect_the reference implementation_parity.py`
- `tests/test_structured_allocations.py`

New reference/profiling files:

- `tests/fixtures/random_effect_the reference implementation_reference.R`
- `tests/fixtures/random_effect_the reference implementation_reference.json`
- `benchmarks/profile_structured_credibility.py`
- `docs/guide/random-effects.md`

Existing production files expected to change:

- `src/superglm/types.py`
- `src/superglm/features/__init__.py`
- `src/superglm/__init__.py`
- `src/superglm/_group_matrix/_group_matrix_core.py`
- `src/superglm/_group_matrix/_group_matrix_algebra.py`
- `src/superglm/_group_matrix/_group_matrix_kernels.py`
- `src/superglm/_group_matrix/_group_matrix_tabmat.py`
- `src/superglm/group_matrix.py`
- `src/superglm/dm_builder.py`
- `src/superglm/model/api.py`
- `src/superglm/model/base.py`
- `src/superglm/model/fit_ops.py`
- `src/superglm/model/reml_setup.py`
- `src/superglm/model/reml_finalize.py`
- `src/superglm/model/report_ops.py`
- `src/superglm/model/state_ops.py`
- `src/superglm/reml/direct.py`
- `src/superglm/reml/discrete.py`
- `src/superglm/reml/gradient.py`
- `src/superglm/reml/objective.py`
- `src/superglm/reml/penalty_algebra.py`
- `src/superglm/reml/w_derivatives.py`
- `src/superglm/solvers/irls_direct.py`

## Task 1: Public feature contract and all-level compact matrix

### 1.1 Write the failing feature tests

**Files:**

- Create: `tests/test_random_effect.py`

- [ ] Add constructor validation tests for:

  ```python
  RandomEffect(unseen="population", missing="error")
  RandomEffect(unseen="error", missing="error")
  ```

  and assert invalid `unseen` or any `missing` value other than `"error"` raises `ValueError`.

- [ ] Add a build test using levels `["b", "a", "b", "c"]` and assert:

  ```python
  info.n_cols == 3
  info.cat_codes.tolist() == [1, 0, 1, 2]
  info.structured_kind == "random_effect"
  spec._levels == ["a", "b", "c"]
  ```

- [ ] Add missing-value tests for both build and score.

- [ ] Add score tests proving known levels select all-level coefficients and unknown levels return
  zero only under `unseen="population"`.

- [ ] Add reconstruct tests proving no base level is present and every fitted level is retained.

- [ ] Run the red test:

  ```bash
  rtk uv run pytest tests/test_random_effect.py -q
  ```

  Expected: import failure because `RandomEffect` does not exist.

### 1.2 Implement `RandomEffect`

**Files:**

- Create: `src/superglm/features/random_effect.py`
- Modify: `src/superglm/types.py`
- Modify: `src/superglm/features/__init__.py`
- Modify: `src/superglm/__init__.py`

- [ ] Extend `GroupInfo` with a defaulted marker after existing fields:

  ```python
  structured_kind: Literal["random_effect", "factor_smooth"] | None = None
  ```

- [ ] Implement this public constructor and learned state:

  ```python
  class RandomEffect:
      def __init__(
          self,
          *,
          unseen: Literal["population", "error"] = "population",
          missing: Literal["error"] = "error",
          lambda_policy: LambdaPolicy | None = None,
      ):
          self.unseen = unseen
          self.missing = missing
          self._lambda_policy = lambda_policy
          self._levels: list[Any] = []
          self._level_to_code: dict[Any, int] = {}
  ```

- [ ] In `build`, use `pandas.factorize(x, sort=True)`, reject `-1`, store the exact fitted levels,
  and return:

  ```python
  GroupInfo(
      columns=None,
      n_cols=len(self._levels),
      penalized=True,
      cat_codes=codes.astype(np.intp, copy=False),
      lambda_policies=(
          None if self._lambda_policy is None else {"_default": self._lambda_policy}
      ),
      structured_kind="random_effect",
  )
  ```

- [ ] Implement `score` with a pandas categorical code lookup. Unknown codes contribute zero for
  `"population"` and raise the same contextual `ValueError` for `"error"`. Reject missing values
  before applying either unseen policy.

- [ ] Implement a private `validate_prediction_values` helper that rejects missing values without
  applying the unseen-level policy. Population prediction uses it before zeroing the term.

- [ ] Implement `transform` only as a small-reference compatibility operation; it may materialize
  all-level one-hot output but must not be used by normal prediction because `score` exists.

- [ ] Implement `reconstruct` as:

  ```python
  {
      "levels": self._levels.copy(),
      "effects": dict(zip(self._levels, map(float, beta), strict=True)),
  }
  ```

- [ ] Export `RandomEffect` from both public export modules.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect.py -q
  rtk uv run ruff check src/superglm/features/random_effect.py src/superglm/types.py
  ```

### 1.3 Write compact-matrix red tests

**Files:**

- Create: `tests/test_random_effect_matrix.py`

- [ ] Compare `matvec`, `rmatvec`, `gram`, `toarray`, and `row_subset` against a small materialized
  all-level one-hot matrix.

- [ ] Assert `row_subset` returns `RandomEffectGroupMatrix` and preserves the original `n_levels`.

- [ ] Assert `_process_info` dispatches `structured_kind="random_effect"` to the new type.

- [ ] Assert a 101-level matrix produces a `tabmat.CategoricalMatrix` with 101 columns and
  `drop_first=False`; assert ordinary `CategoricalGroupMatrix` still has 100 non-base columns for
  the equivalent 101-level fixed factor.

- [ ] Run the red test:

  ```bash
  rtk uv run pytest tests/test_random_effect_matrix.py -q
  ```

### 1.4 Implement the compact matrix and tabmat dispatch

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_algebra.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_tabmat.py`
- Modify: `src/superglm/group_matrix.py`
- Modify: `src/superglm/dm_builder.py`

- [ ] Implement `RandomEffectGroupMatrix` as a `CategoricalGroupMatrix` subclass with no fitted
  sink code. Validate `0 <= code < K`; store `lambda_policies`; override `row_subset` so type,
  global level count, and policy metadata survive. Reuse inherited bincount/crosstab behavior.

- [ ] Add the class to runtime imports and `GroupMatrix`.

- [ ] In `_process_info`, test `info.structured_kind == "random_effect"` before the generic
  categorical branch and copy `info.lambda_policies` to the compact matrix.

- [ ] In tabmat construction, dispatch `RandomEffectGroupMatrix` before
  `CategoricalGroupMatrix`:

  ```python
  tabmat.CategoricalMatrix(
      gm.codes.astype(np.int32, copy=False),
      categories=np.arange(gm.n_levels),
      drop_first=False,
  )
  ```

- [ ] Preserve every current early-return condition for ordinary matrices.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect.py tests/test_random_effect_matrix.py -q
  rtk uv run pytest tests/test_categorical_ux.py tests/test_interactions.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm tests/test_random_effect.py tests/test_random_effect_matrix.py
  rtk git commit -m "Add all-level random effect feature"
  ```

## Task 2: Implicit identity penalty and dense REML oracle

### 2.1 Write implicit-penalty red tests

**Files:**

- Create: `tests/test_random_effect_penalty.py`

- [ ] Assert `collect_reml_groups` includes `RandomEffectGroupMatrix`.

- [ ] Assert its `PenaltyComponent` has:

  ```python
  component.penalty_kind == "identity"
  component.omega_raw is None
  component.omega_ssp is None
  component.rank == K
  component.log_det_omega_plus == 0.0
  component.eigvals_omega is None
  ```

- [ ] Monkeypatch `numpy.eye` and `numpy.linalg.eigvalsh` to fail when invoked with shape/size `K`
  while building random-effect penalty metadata.

- [ ] For a small dense oracle, assert `build_penalty_matrix` places lambda on only the
  random-effect diagonal and never calls `np.eye(K)`.

- [ ] Test helper results:

  ```python
  penalty_quadratic(component, beta) == beta @ beta
  inverse_penalty_trace(component, H_inv) == np.trace(H_inv[group_slice, group_slice])
  penalty_matvec(component, beta) == beta
  ```

- [ ] Run the red test:

  ```bash
  rtk uv run pytest tests/test_random_effect_penalty.py -q
  ```

### 2.2 Implement identity-aware penalty algebra

**Files:**

- Modify: `src/superglm/types.py`
- Modify: `src/superglm/model/reml_setup.py`
- Modify: `src/superglm/reml/penalty_algebra.py`
- Modify: `src/superglm/reml/result.py`

- [ ] Change `PenaltyComponent.omega_raw` to `NDArray | None` and add:

  ```python
  penalty_kind: Literal["dense", "identity", "repeated"] = "dense"
  ```

- [ ] Add small centralized helpers:

  ```python
  penalty_component_matrix(pc, gm) -> NDArray
  penalty_component_quadratic(pc, beta_g) -> float
  penalty_component_matvec(pc, beta_g) -> NDArray
  penalty_component_trace(pc, inverse_block_or_diagonal) -> float
  total_penalty_quadratic(beta, lambdas, penalties, group_matrices) -> float
  ```

  Identity branches use vectors/diagonals; dense branches preserve current calculations.

- [ ] Teach `collect_reml_groups` and `build_penalty_components` about
  `RandomEffectGroupMatrix`. Create one full-rank identity component named exactly after the
  feature group and copy `lambda_policy`.

- [ ] In `build_penalty_matrix`, fill the existing dense oracle's diagonal view:

  ```python
  diag = np.diagonal(S)
  diag[pc.group_sl] += lam
  ```

  If the diagonal view is read-only on the supported NumPy version, use explicit integer diagonal
  indices; do not allocate an identity matrix.

- [ ] Make `PenaltyCache.omega_ssp` and `eigvals_omega` optional so identity components can use
  existing cache plumbing without fake arrays.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_penalty.py tests/test_penalties.py \
    tests/test_multi_penalty.py tests/test_lambda_policy.py -q
  ```

### 2.3 Add the dense end-to-end oracle

**Files:**

- Create: `tests/test_random_effect_reml.py`
- Modify: `src/superglm/reml/direct.py`
- Modify: `src/superglm/reml/discrete.py`
- Modify: `src/superglm/reml/gradient.py`
- Modify: `src/superglm/reml/objective.py`
- Modify: `src/superglm/model/reml_finalize.py`

- [ ] Add a Gaussian random-intercept simulation with a fixed lambda and compare
  `direct_solve="gram"` coefficients to the analytic penalized least-squares solution.

- [ ] Add Poisson fixed-lambda tests with offset and nonuniform weights.

- [ ] Add estimated-lambda smoke tests for Gaussian and Poisson and assert finite lambda,
  objective, EDF, variance component, and convergence state.

- [ ] Replace direct `gm.R_inv.T @ gm.omega @ gm.R_inv`, `beta @ S @ beta`, and identity-block
  trace assumptions with the centralized helpers. Dense spline behavior must remain bitwise or
  tolerance equivalent.

- [ ] Update profiled-phi finalization to call `total_penalty_quadratic` rather than requiring a
  dense `S` solely for the quadratic.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_reml.py tests/test_reml.py \
    tests/test_reml_fd.py tests/test_reml_newton_fixes.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm tests/test_random_effect_penalty.py \
    tests/test_random_effect_reml.py
  rtk git commit -m "Add implicit random effect REML penalty"
  ```

## Task 3: Fit and prediction API contracts

### 3.1 Write red API tests

**Files:**

- Extend: `tests/test_random_effect.py`
- Modify: `tests/test_robust_solve.py`

- [ ] Assert `SuperGLM(direct_solve="structured")` is accepted and the error text for invalid
  values lists all four choices.

- [ ] Assert `fit()` and `fit_path()` reject any `RandomEffect` before solver work with a message
  directing users to `fit_reml()`.

- [ ] Assert:

  ```python
  model.predict(X_known)
  model.predict(X_known, random_effects="conditional")
  ```

  agree exactly.

- [ ] Assert `random_effects="population"` zeroes only `RandomEffect` contributions while retaining
  numeric, categorical, spline, intercept, and offset contributions.

- [ ] Assert conditional prediction of an unknown level gives the population prediction for
  `unseen="population"` and raises for `unseen="error"`.

- [ ] Assert invalid prediction mode raises `ValueError`.

- [ ] Run the red tests:

  ```bash
  rtk uv run pytest tests/test_random_effect.py tests/test_robust_solve.py -q
  ```

### 3.2 Implement fit guards and population prediction

**Files:**

- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_ops.py`

- [ ] Extend constructor validation and docs to
  `{"auto", "gram", "qr", "structured"}`.

- [ ] Add a shared `_contains_random_effect(model)` check and invoke it before ordinary `fit` or
  path design/solver work.

- [ ] Thread keyword-only `random_effects` through:

  ```python
  predict(X, offset=None, *, random_effects="conditional")
  _predict_eta(..., *, fast_discrete, random_effects)
  predict_eta_exact(..., *, random_effects="conditional")
  predict_eta_fast_discrete(..., *, random_effects="conditional")
  ```

- [ ] In the prediction loop, call `RandomEffect.validate_prediction_values` and then skip its
  contribution only for population mode. This preserves the rule that missing values fail under
  both modes while population prediction may accept unseen levels. Do not alter ordinary feature
  scoring.

- [ ] Keep sklearn wrappers unchanged; their existing `predict(X)` calls remain conditional.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect.py tests/test_api.py tests/test_sklearn.py \
    tests/test_fit_state_retention.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/model tests/test_random_effect.py tests/test_robust_solve.py
  rtk git commit -m "Add random effect fit and prediction contracts"
  ```

## Task 4: Hessian-factor protocol and scalar Schur algebra

### 4.1 Write exact linear-algebra red tests

**Files:**

- Create: `tests/test_structured_factor.py`

- [ ] Generate seeded SPD block systems with a noncontiguous small-index permutation:

  ```text
  H = [[A, C.T], [C, diag(d)]]
  ```

- [ ] Compare `solve` for vector and matrix RHS, `logdet`, selected small blocks, selected
  structured diagonal, and mixed blocks to `np.linalg.solve(H, ...)`.

- [ ] Compare the dominant identity penalty trace and self/cross Hessian traces to dense formulas.

- [ ] Compare `trace_inverse_operator` for arbitrary-sign block operators to
  `np.trace(np.linalg.inv(H) @ O)`.

- [ ] Cover a one-column small block, multiple small groups, and permuted dominant slice.

- [ ] Assert requesting a full dominant inverse block above the safety threshold raises a targeted
  error rather than allocating `K x K`.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_structured_factor.py -q
  ```

### 4.2 Implement the factor protocol

**Files:**

- Create: `src/superglm/solvers/hessian_factor.py`
- Create: `src/superglm/solvers/structured.py`

- [ ] Define the private protocol:

  ```python
  class HessianFactor(Protocol):
      shape: tuple[int, int]
      backend: str

      def solve(self, rhs: NDArray) -> NDArray: ...
      def logdet(self) -> float: ...
      def selected_inverse_block(self, indices: NDArray) -> NDArray: ...
      def selected_inverse_diagonal(self, indices: NDArray) -> NDArray: ...
      def trace_inverse_penalty(self, component: PenaltyComponent) -> float: ...
      def penalty_cross_trace(
          self,
          left: PenaltyComponent,
          right: PenaltyComponent,
          left_scale: float,
          right_scale: float,
      ) -> float: ...
      def trace_inverse_operator(self, operator: SymmetricBlockOperator) -> float: ...
  ```

- [ ] Implement `DenseHessianFactor` as an adapter around the existing inverse/logdet result. Its
  methods reproduce the current ndarray formulas.

- [ ] Implement immutable `SymmetricBlockOperator` with global small/structured index maps and
  arrays `A`, `C`, and `d`.

- [ ] Implement `ScalarSchurFactor`:

  - factor every positive `d`;
  - compute `F = C / d[:, None]`;
  - factor `Q = A - C.T @ F` with Cholesky first;
  - use the existing residual-checked robust fallback principles when Cholesky fails;
  - expose the exact block identities from the approved design;
  - compute dominant-block inverse diagonal without a `K x K` temporary;
  - compute the dominant identity self-trace using small-matrix identities.

- [ ] Preserve diagnostics: backend, dominant group name, minimum local diagonal, Schur condition
  estimate, fallback reason, and whether a dense fallback was used.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_structured_factor.py tests/test_robust_solve.py -q
  rtk uv run ruff check src/superglm/solvers/hessian_factor.py \
    src/superglm/solvers/structured.py
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/solvers tests/test_structured_factor.py
  rtk git commit -m "Add scalar Schur Hessian factor"
  ```

## Task 5: Structured sufficient statistics and tabmat-preserving Gram builder

### 5.1 Write red builder/kernel tests

**Files:**

- Extend: `tests/test_random_effect_matrix.py`
- Extend: `tests/test_structured_factor.py`
- Create: `tests/test_structured_allocations.py`

- [ ] For random effect plus numeric, fixed categorical, spline, discretized spline, tensor, and a
  second random effect, compare structured `A`, `C`, `d`, RHS, and intercept cross-products to
  slices of a small materialized weighted Gram.

- [ ] Repeat with arbitrary-sign weights for W-derivative operators.

- [ ] Verify the largest eligible random-effect block is selected and the other remains in `A`.

- [ ] Monkeypatch the dominant matrix's `toarray` and `gram` to raise, then build structured
  summaries successfully.

- [ ] Monkeypatch `_block_xtwx` and `np.zeros` guards to prove no full `p x p` array is requested
  for a large dominant `K`.

- [ ] Instrument tabmat and assert selected `sandwich(rows=..., cols=...)` operations are used when
  the full tabmat split is eligible; assert native aggregation is used when an SSP/discrete group
  disables tabmat.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_matrix.py tests/test_structured_factor.py \
    tests/test_structured_allocations.py -q
  ```

### 5.2 Implement summary kernels and builder

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_algebra.py`
- Modify: `src/superglm/solvers/structured.py`

- [ ] First compose existing `np.bincount`, `_agg_by_bin`, `_cross_gram`, and selected tabmat
  operations into a correct `build_scalar_structured_system`.

- [ ] Add one fused numba kernel only where the composed version makes multiple observation passes:

  ```python
  _random_effect_sufficient_stats(codes, W, Wz, n_levels)
      -> level_W, level_Wz
  ```

- [ ] Build the small-group Gram with reindexed `GroupSlice` objects. Build `C` by aggregating
  `W * X_small` by dominant level; choose tabmat selected sandwiches when available.

- [ ] Return both:

  - an unpenalized `SymmetricBlockOperator` for REML/EDF/W derivatives;
  - partitioned RHS and intercept statistics for the working solve.

- [ ] Add `select_structured_group` and eligibility diagnostics. Forced mode errors on constraints,
  SCOP, missing random effects, or unsupported penalty geometry; auto mode records a reason and
  returns dense.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_matrix.py tests/test_structured_factor.py \
    tests/test_structured_allocations.py tests/test_discretize_fit.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/_group_matrix src/superglm/solvers/structured.py \
    tests/test_random_effect_matrix.py tests/test_structured_factor.py \
    tests/test_structured_allocations.py
  rtk git commit -m "Add structured random effect sufficient statistics"
  ```

## Task 6: Exact IRLS integration

### 6.1 Write forced-backend parity tests

**Files:**

- Create: `tests/test_structured_irls.py`

- [ ] Compare `"gram"` and `"structured"` with fixed lambdas for Gaussian, Poisson, Gamma, NB2,
  and Tweedie.

- [ ] Cover offsets, nonuniform weights, one random effect, one dominant plus one smaller random
  effect, and random effect plus numeric/categorical/spline/tensor terms.

- [ ] Compare beta, intercept, eta, predictions, deviance, phi, EDF, log determinant, convergence,
  and iteration count with explicit tolerances.

- [ ] Assert `cache_out` stores a structured operator/factor and never a full `XtWX` in forced
  structured mode.

- [ ] Assert forced structured mode reports exact ineligibility errors for QP/SCOP constraints;
  auto mode records one dense fallback reason.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_structured_irls.py -q
  ```

### 6.2 Integrate Schur solves in direct IRLS

**Files:**

- Modify: `src/superglm/solvers/irls_direct.py`
- Modify: `src/superglm/solvers/pirls.py`
- Modify: `src/superglm/model/base.py`

- [ ] Resolve backend once at fit entry. Do not build dense `S` when structured was selected.

- [ ] At each IRLS iteration:

  1. build the unpenalized structured operator and RHS;
  2. add ordinary small-group penalties to `A`;
  3. add the dominant identity lambda to `d`;
  4. construct an augmented factor with intercept in the small partition;
  5. solve for intercept, small beta, and dominant beta;
  6. construct the coefficient-only factor for REML and EDF.

- [ ] Compute EDF as:

  ```text
  1 + p - sum_j lambda_j * trace(H^-1 S_j)
  ```

  using factor penalty traces. Compare it to the dense formula in tests.

- [ ] Put backend/fallback diagnostics into the profile dict and
  `model._last_fit_meta["direct_backend"]`.

- [ ] Return ndarray inverses on existing dense paths for private compatibility and a
  `HessianFactor` on the structured path. Normalize both through `as_hessian_factor` at new call
  sites.

- [ ] Cache both coefficient and augmented structured factors for final inference without adding
  them to public `PIRLSResult`.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_structured_irls.py tests/test_irls_direct.py \
    tests/test_robust_solve.py tests/test_tweedie_convergence.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/solvers src/superglm/model/base.py \
    tests/test_structured_irls.py
  rtk git commit -m "Use scalar Schur solves in exact IRLS"
  ```

## Task 7: Factor-aware exact REML and W corrections

### 7.1 Write REML derivative and trajectory red tests

**Files:**

- Extend: `tests/test_random_effect_reml.py`
- Extend: `tests/test_reml_fd.py`
- Extend: `tests/test_hessian_ift.py`

- [ ] At a fixed beta/weight state compare dense and structured:

  - objective;
  - log determinant;
  - gradient;
  - Hessian;
  - first-order W correction;
  - second-order W correction;
  - finite-difference gradient/Hessian.

- [ ] Compare full REML lambda history and accepted line-search objectives for one and two random
  effects.

- [ ] Cover one random-effect identity component crossed with a small spline penalty component in
  the REML Hessian.

- [ ] Run the red tests:

  ```bash
  rtk uv run pytest tests/test_random_effect_reml.py tests/test_reml_fd.py \
    tests/test_hessian_ift.py -q
  ```

### 7.2 Refactor objective/gradient/Hessian to the protocol

**Files:**

- Modify: `src/superglm/reml/objective.py`
- Modify: `src/superglm/reml/gradient.py`
- Modify: `src/superglm/reml/direct.py`
- Modify: `src/superglm/model/reml_ops.py`

- [ ] Accept `NDArray | HessianFactor` and normalize at entry.

- [ ] Replace inverse slicing and matrix products with:

  - `trace_inverse_penalty`;
  - `penalty_cross_trace`;
  - `solve`;
  - `selected_inverse_block` only for genuinely small dense components.

- [ ] Make the objective accept `SymmetricBlockOperator` as the Gram representation. If
  `log_det_H` is supplied, do not rebuild or materialize `H`.

- [ ] Stop constructing `S_cand` on a structured candidate. Pass lambdas and penalty components to
  IRLS; use `total_penalty_quadratic` for objective/phi calculations.

- [ ] Preserve dense ndarray call compatibility for existing tests and public diagnostic helpers.

### 7.3 Refactor W derivatives to block operators

**Files:**

- Modify: `src/superglm/reml/w_derivatives.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_algebra.py`

- [ ] Replace `H_inv @ rhs` with `factor.solve(rhs)`.

- [ ] When structured, build arbitrary-sign derivative Gram summaries as
  `SymmetricBlockOperator` and compute trace terms with `trace_inverse_operator`.

- [ ] Keep the current dense `_block_xtwx_signed` path untouched for dense factors.

- [ ] Ensure no W-correction branch requests a full dominant inverse or full derivative Gram.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_reml.py tests/test_reml_fd.py \
    tests/test_hessian_ift.py tests/test_reml_newton_fixes.py tests/test_reml.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/reml src/superglm/model/reml_ops.py \
    src/superglm/_group_matrix tests/test_random_effect_reml.py \
    tests/test_reml_fd.py tests/test_hessian_ift.py
  rtk git commit -m "Make exact REML use structured Hessian factors"
  ```

## Task 8: Structured `discrete=True` POI/fREML cache

### 8.1 Write discrete red tests

**Files:**

- Create: `tests/test_random_effect_discrete.py`

- [ ] Assert `should_discretize(RandomEffect(), True)` is false and fitted random-effect codes are
  identical between exact and discrete model builds.

- [ ] Compare dense and structured discrete fits for random effect plus discretized spline and
  discretized tensor terms.

- [ ] Compare exact versus discrete results at data where bin support is exact.

- [ ] Instrument `DesignMatrix.matvec`, group `rmatvec`, and all aggregation kernels after cache
  creation; assert a lambda trial solve performs no data pass.

- [ ] Assert cached lambda trials allocate `O(q^2 + Kq)` state, not `O(p^2)`.

- [ ] Compare cached structured solve beta/intercept/logdet to a small dense re-solve across at
  least five lambda vectors.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_discrete.py -q
  ```

### 8.2 Implement structured cached-W trials

**Files:**

- Modify: `src/superglm/reml/discrete.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `src/superglm/solvers/irls_direct.py`

- [ ] Store unpenalized structured `A`, `C`, `d`, `XtWz`, `XtW1`, `sum_W`, and `sum_Wz` in
  `cache_out`.

- [ ] Add `solve_cached_scalar_structured(cache, lambdas, penalties)` that updates only penalty
  blocks, factors Schur systems, and returns beta/intercept/coefficient factor/augmented factor.

- [ ] Dispatch `_solve_cached_augmented` and `_solve_cached_h_system` by cache representation.

- [ ] Refactor POI gradient, Hessian, objective, line search, and final refit to consume the factor
  and structured operator. Do not call `build_penalty_matrix` in a structured trial.

- [ ] Record data-pass count and cache-solve timing in `_reml_profile`.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_discrete.py tests/test_discretize.py \
    tests/test_discretize_fit.py tests/test_reml.py tests/test_reml_fd.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/reml/discrete.py src/superglm/solvers \
    tests/test_random_effect_discrete.py
  rtk git commit -m "Add structured cached-W random effect solves"
  ```

## Task 9: Compact fit state and selected covariance

### 9.1 Write state/inference red tests

**Files:**

- Create: `tests/test_random_effect_inference.py`
- Extend: `tests/test_fit_state_retention.py`

- [ ] Compare selected coefficient covariance blocks, diagonals, intercept covariance, term EDF,
  and random-effect posterior SE to dense augmented-inverse references.

- [ ] Cover `retain_fit_state=True` and `False`.

- [ ] Assert released state has no row-scale design/weights but retains levels, coefficients,
  lambdas, backend metadata, support totals, coefficient factor, and augmented factor.

- [ ] Monkeypatch dense covariance builders to fail and prove random-effect reporting and standard
  summaries for fixed terms still work.

- [ ] Assert no `K x K` covariance is cached for the dominant term.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_inference.py \
    tests/test_fit_state_retention.py -q
  ```

### 9.2 Add selected-covariance state

**Files:**

- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/model/reml_finalize.py`
- Modify: `src/superglm/model/state_ops.py`
- Modify: `src/superglm/inference/covariance.py`
- Modify: `src/superglm/inference/coef_tables.py`
- Modify: `src/superglm/inference/_term_covariance.py`
- Modify: `src/superglm/inference/metrics.py`

- [ ] Initialize and clear `model._linear_system_state` with other fit state.

- [ ] Carry the authoritative final coefficient and augmented factors out of exact/discrete REML
  and store them privately on the model.

- [ ] Add a compact covariance accessor whose supported operations are selected block, selected
  diagonal, intercept variance/cross-covariance, solve, and trace. Dense models continue returning
  ndarrays.

- [ ] Refactor coefficient tables and term inference to request only the current group block or
  diagonal. Do not silently materialize the full covariance when a compact accessor is present.

- [ ] Compute fixed-term group EDF through factor penalty traces/selected products. Compute the
  random-effect term EDF as its contribution to
  `trace(H^-1 XtWX)`, equivalently `K - lambda * trace(H^-1 I_RE)`.

- [ ] During state release, retain compact factors and precomputed support totals before clearing
  row-scale arrays.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_inference.py \
    tests/test_fit_state_retention.py tests/test_metrics.py tests/test_term_inference.py \
    tests/test_model_tests.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/model src/superglm/inference \
    tests/test_random_effect_inference.py tests/test_fit_state_retention.py
  rtk git commit -m "Retain compact structured inference state"
  ```

## Task 10: Actuarial credibility reporting

### 10.1 Write reporting red tests

**Files:**

- Extend: `tests/test_random_effect_inference.py`

- [ ] Assert `model.random_effects("broker")` returns:

  - name;
  - lambda;
  - phi;
  - `tau_squared == phi / lambda`;
  - `standard_deviation == sqrt(phi / lambda)`;
  - term EDF;
  - collapse/boundary diagnostics;
  - one table row per fitted level.

- [ ] Assert the table has exactly:

  ```text
  level, count, fit_weight, exposure, unpooled_effect, effect, relativity,
  posterior_se, credibility, shrinkage, finite, has_information, collapsed
  ```

- [ ] For Poisson/log-link, compare vectorized conditional unpooled effects to
  `log(actual / expected)` with all other fitted contributions held fixed.

- [ ] For Gaussian/logit/Gamma cases, compare the vectorized Fisher-scoring result to independent
  scalar reference optimizations on a small dataset.

- [ ] Assert `credibility == information / (information + lambda)` and
  `shrinkage == 1 - credibility`; do not clip material violations.

- [ ] Assert posterior SE uses the full augmented selected inverse and differs from the local
  conditional `sqrt(phi / (information + lambda))` when fixed-term correlation is present.

- [ ] Assert explicit exposure is length-validated and aggregated; absent exposure produces an
  all-missing column without interpreting offsets as exposure.

- [ ] Assert a released-state model returns compact quantities but asks for explicit training
  arrays when unpooled effects are requested and were not precomputed.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_inference.py -q
  ```

### 10.2 Implement reporting

**Files:**

- Create: `src/superglm/inference/random_effects.py`
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/report_ops.py`
- Modify: `src/superglm/__init__.py`

- [ ] Add frozen result dataclass fields matching the approved API.

- [ ] Implement a vectorized per-level Fisher scorer. Per iteration:

  1. subtract the fitted random-effect contribution from fitted eta;
  2. evaluate candidate per-level effects by code lookup;
  3. aggregate score and Fisher information with `np.bincount`;
  4. update every finite/informed level simultaneously;
  5. stop on maximum finite update tolerance.

- [ ] Store final local information, fit counts, and fit-weight totals in compact fit state so
  credibility remains available after row-state release.

- [ ] Set log-link relativity to `exp(effect)`; use `NaN` with a documented flag for non-log links.

- [ ] Flag lambda at the upper boundary as collapsed and emit one user warning per term.

- [ ] Add:

  ```python
  SuperGLM.random_effects(
      name: str,
      *,
      exposure: NDArray | None = None,
      X: pd.DataFrame | None = None,
      y: NDArray | None = None,
      sample_weight: NDArray | None = None,
      offset: NDArray | None = None,
  ) -> RandomEffectResult
  ```

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_inference.py tests/test_relativities.py \
    tests/test_model_tests.py -q
  rtk uv run ruff check src/superglm/inference/random_effects.py \
    src/superglm/model/api.py src/superglm/model/report_ops.py
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm/inference/random_effects.py src/superglm/model \
    src/superglm/__init__.py tests/test_random_effect_inference.py
  rtk git commit -m "Add random effect credibility reporting"
  ```

## Task 11: Full dense/structured parity and fallback matrix

### 11.1 Expand correctness coverage

**Files:**

- Extend: `tests/test_random_effect_reml.py`
- Extend: `tests/test_random_effect_discrete.py`
- Extend: `tests/test_structured_allocations.py`

- [ ] Parameterize dense versus structured REML over:

  - Gaussian, Poisson, Gamma, NB2, and Tweedie;
  - exact and discrete;
  - offset/no offset;
  - uniform/nonuniform weights;
  - one and two random effects;
  - numeric, fixed categorical, spline, tensor, and discrete smooth companions.

- [ ] Compare coefficient mapping, intercept, eta, prediction, deviance, phi, EDF, lambda path,
  variance component, objective, gradient, Hessian, posterior SE, credibility, and convergence.

- [ ] Add ordinary `Categorical` and `SplineCategorical` regression snapshots.

- [ ] Add explicit fallback reason tests for constraints, SCOP, and unsupported override geometry.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_reml.py \
    tests/test_random_effect_discrete.py tests/test_structured_allocations.py \
    tests/test_categorical_ux.py tests/test_interactions.py -q
  ```

### 11.2 Stabilize diagnostics and tolerances

**Files:**

- Modify only files identified by the failing parity cases.

- [ ] Use scale-aware `rtol`/`atol`; do not weaken tolerances globally.

- [ ] Ensure an SPD failure identifies term, level if applicable, lambda, minimum local diagonal,
  Schur condition estimate, and fallback action.

- [ ] Ensure auto fallback logs once and remains visible in `_reml_profile` and
  `_last_fit_meta`.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect*.py tests/test_structured*.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add src/superglm tests/test_random_effect_reml.py \
    tests/test_random_effect_discrete.py tests/test_structured_allocations.py
  rtk git commit -m "Harden structured random effect parity"
  ```

## Task 12: Pinned the reference implementation `bs="re"` parity

### 12.1 Generate and review reference fixtures

**Files:**

- Create: `tests/fixtures/random_effect_the reference implementation_reference.R`
- Create: `tests/fixtures/random_effect_the reference implementation_reference.json`

- [ ] In the R script, use fixed seeds and emit references for:

  - Gaussian `gam(y ~ x + s(f, bs="re"), method="REML")`;
  - Poisson with log exposure offset;
  - population prediction for an unseen level;
  - `bam(..., method="fREML", discrete=TRUE)` where supported.

- [ ] Record R version, the reference implementation version, seed, data-generation parameters, deviance, EDF, scale,
  variance component, fitted predictions, and population predictions.

- [ ] Run:

  ```bash
  rtk Rscript tests/fixtures/random_effect_the reference implementation_reference.R
  rtk git diff -- tests/fixtures/random_effect_the reference implementation_reference.json
  ```

- [ ] Manually verify the fixture has no machine-specific paths or timestamps.

### 12.2 Add parity tests

**Files:**

- Create: `tests/test_random_effect_the reference implementation_parity.py`

- [ ] Recreate the seeded datasets in Python and compare prediction, deviance, EDF, variance
  component, scale, and population behavior. Compare lambda only where penalty scaling is proven
  common.

- [ ] Use committed JSON values; ordinary CI must not require R.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_random_effect_the reference implementation_parity.py -q
  ```

- [ ] Commit:

  ```bash
  rtk git add tests/fixtures/random_effect_the reference implementation_reference.R \
    tests/fixtures/random_effect_the reference implementation_reference.json \
    tests/test_random_effect_the reference implementation_parity.py
  rtk git commit -m "Add the reference implementation random effect parity fixtures"
  ```

## Task 13: cProfile evidence, kernel decisions, and auto crossover

### 13.1 Build the profiling harness

**Files:**

- Create: `benchmarks/profile_structured_credibility.py`

- [ ] Follow `benchmarks/profile_superbooster_interactions.py` and
  `superglm.profiling.harness`.

- [ ] Support CLI parameters for `n`, `K`, family, exact/discrete, one/two random effects,
  dense/structured/auto backend, repetitions, warmups, and output directory.

- [ ] Emit:

  - raw `.pstats`;
  - cumulative and internal-time text reports;
  - wall repetitions;
  - RSS/USS and CPU telemetry;
  - tracemalloc summary;
  - model dimensions;
  - backend/fallback reason;
  - phase timings and data-pass count;
  - iterations, objective, and dense parity diagnostics.

- [ ] Include seeded matrix points:

  ```text
  K = 100, 300, 1_000, 3_000, 10_000
  small n and large n
  one dominant RE
  two REs with one dominant
  exact and discrete
  Gaussian, Poisson, Gamma
  ```

- [ ] Run a smoke profile:

  ```bash
  rtk uv run python benchmarks/profile_structured_credibility.py \
    --n 20000 --levels 1000 --family poisson --backend structured \
    --repetitions 2 --warmups 1
  ```

### 13.2 Decide fused kernels from evidence

**Files:**

- Modify only profiled hot paths in:
  - `src/superglm/_group_matrix/_group_matrix_kernels.py`
  - `src/superglm/solvers/structured.py`

- [ ] Inspect cumulative and internal-time call stacks.

- [ ] Add or fuse a kernel only if a composed aggregation is materially dominant and allocation
  evidence identifies the temporary.

- [ ] Re-run the same case and record before/after timings and allocation counts in a checked-in
  benchmark summary alongside the script.

- [ ] Verify current categorical benchmark behavior is not degraded.

### 13.3 Measure and encode `auto` crossover

**Files:**

- Modify: `src/superglm/solvers/structured.py`
- Extend: `tests/test_structured_irls.py`

- [ ] Run dense and structured repetitions for every dense-safe point.

- [ ] Choose a conservative cost rule based on measured `K`, small dimension `q`, and setup cost;
  encode it in one named function with documented coefficients.

- [ ] Test that representative small models stay dense and models beyond the measured crossover
  select structured.

- [ ] Do not put wall-clock assertions in pytest.

- [ ] Run:

  ```bash
  rtk uv run pytest tests/test_structured_irls.py tests/test_structured_allocations.py -q
  rtk uv run python benchmarks/profile_structured_credibility.py \
    --matrix core --repetitions 5 --warmups 2
  ```

- [ ] Commit:

  ```bash
  rtk git add benchmarks/profile_structured_credibility.py \
    src/superglm/_group_matrix/_group_matrix_kernels.py \
    src/superglm/solvers/structured.py tests/test_structured_irls.py
  rtk git commit -m "Profile and tune structured credibility fits"
  ```

## Task 14: Documentation and release verification

### 14.1 Document user behavior

**Files:**

- Create: `docs/guide/random-effects.md`
- Modify: `README.md`
- Modify the relevant MkDocs navigation file if this repository registers guide pages explicitly.

- [ ] Document:

  - `RandomEffect` versus fixed `Categorical`;
  - relationship to the reference implementation `s(f, bs="re")`;
  - REML-only fitting;
  - conditional versus population prediction;
  - unseen and missing policies;
  - `tau_squared = phi / lambda`;
  - credibility versus posterior uncertainty;
  - exposure being explicit and never inferred from offset;
  - exact/discrete behavior;
  - backend selection/fallback diagnostics;
  - current non-goals.

- [ ] Include runnable Gaussian and Poisson examples.

### 14.2 Run focused static checks

- [ ] Run:

  ```bash
  rtk uv run ruff format src/ tests/ benchmarks/profile_structured_credibility.py
  rtk uv run ruff check src/ tests/ benchmarks/profile_structured_credibility.py
  rtk uv run mypy src/
  ```

- [ ] Fix only issues caused by this branch; do not sweep unrelated code.

### 14.3 Run the full regression suite

- [ ] Run:

  ```bash
  rtk uv run pytest tests/ -q
  ```

- [ ] Record the exact pass/skip count and runtime.

- [ ] Re-run the core profile after formatting/static fixes and retain the final report.

### 14.4 Review scope and diff

- [ ] Run:

  ```bash
  rtk git status --short
  rtk git diff --stat
  rtk git diff --check
  rtk git diff 64af7c0 -- src tests benchmarks docs README.md
  ```

- [ ] Confirm no LSS path appears in the diff.

- [ ] Confirm no full dominant identity, one-hot, Hessian, inverse, or covariance allocation appears
  on the structured call stack.

- [ ] Confirm every public export, docstring, fit guard, prediction policy, fallback reason, and
  released-state behavior has a test.

- [ ] Commit:

  ```bash
  rtk git add README.md docs src tests benchmarks
  rtk git commit -m "Document and verify structured random effects"
  ```

## Delivery A review checkpoint

Before starting the separate FactorSmooth plan, present:

- exact full-suite result;
- dense/structured/reference parity summary;
- cProfile call-stack summary;
- memory/allocation evidence;
- measured auto crossover;
- remaining dominant runtime;
- known fallback configurations;
- commit range and changed-file summary;
- explicit confirmation that LSS was untouched.

Delivery B may reuse the factor protocol only after this checkpoint establishes that the scalar
backend is correct, materially faster beyond the crossover, and compact in memory.
