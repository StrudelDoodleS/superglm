# FactorSmooth and Block-Schur Delivery Plan

> Delivery B of the approved structured-credibility design. Execute with
> red-green-refactor checkpoints and keep LSS entirely out of scope.

**Goal:** Add an reference-style `FactorSmooth` interaction with a fully penalized
curve per factor level, exact and `discrete=True` REML, conditional/population
prediction, compact inference/reporting, and a profiled block-Schur backend.

**Base:** Delivery A on branch `feature/structured-credibility`, based on
`origin/master` `86b2a1c`.

**Architecture:** A factor smooth owns one P-spline marginal and one all-level
factor. Coefficients are level-major `(K, k)`. The compact matrix stores factor
codes plus a shared raw basis (or discrete support basis/index), and applies an
the reference implementation-compatible natural parameterization without constructing
`n x (Kk)`. Repeated penalty components store only `k x k` blocks. The existing
scalar Schur path remains intact; a parallel block-Schur factor eliminates one
dominant `K x k` term and exposes the same Hessian-factor protocol.

## Invariants

- Do not edit any LSS code or tests.
- `SplineCategorical` keeps its existing reference-level semantics.
- No structured path may allocate `n x K`, `n x (Kk)`, or `(Kk) x (Kk)`.
- Dense Gram remains the small-model numerical oracle.
- The initial public basis is `kind="ps"` only.
- Every level is represented; no reference level or per-curve centering is
  imposed.
- Wiggle and each marginal null-space direction have separate lambdas shared
  across all levels.
- `fit()` and `fit_path()` reject `FactorSmooth`; `fit_reml()` is supported.
- A `RandomEffect` on the same grouping column is rejected because it
  duplicates the factor smooth's constant null-space geometry.

## Task 1: Public interaction and constructor ownership

**Files**

- Create `src/superglm/features/factor_smooth.py`
- Modify `src/superglm/features/__init__.py`
- Modify `src/superglm/__init__.py`
- Modify `src/superglm/model/api.py`
- Modify `src/superglm/model/base.py`
- Modify `src/superglm/model/fit_state.py`
- Modify `src/superglm/model/fit_ops.py`
- Modify `src/superglm/dm_builder.py`
- Create `tests/test_factor_smooth_feature.py`

**Red tests**

- constructor validation for `kind`, `k`, `m`, unseen/missing policies, and
  lambda-policy component names;
- `interactions=[FactorSmooth("x", group="broker")]` is accepted even when
  `x` and `broker` are not both main features;
- clone/config publication owns an independent deep copy;
- duplicate names and duplicate `RandomEffect(group)` geometry fail clearly;
- `fit()` and `fit_path()` reject with a `fit_reml()` message;
- required-column validation names both source columns.

**Implementation**

- Give `FactorSmooth` stable `parent_names`, default name
  `"{variable}:{group}:fs"`, and fit-owned levels/marginal state.
- Normalize constructor interactions into tuple interactions versus explicit
  spec objects. Keep tuple interactions in `ModelConfig.interactions`; capture
  explicit objects through the existing interaction-template fields.
- Extend prediction-plan and fit guards by structured-term capability, not by
  changing ordinary interactions.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_feature.py -q
uv run pytest tests/test_model_config.py tests/test_fit_state.py -q
```

**Commit:** `Add FactorSmooth public interaction contract`

## Task 2: Compact exact and discrete matrix

**Files**

- Modify `src/superglm/types.py`
- Modify `src/superglm/group_matrix.py`
- Modify `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify `src/superglm/_group_matrix/_group_matrix_discretized.py`
- Modify `src/superglm/_group_matrix/_group_matrix_execution.py`
- Modify `src/superglm/_group_matrix/_group_matrix_tabmat.py`
- Modify `src/superglm/_group_matrix/_group_matrix_algebra.py`
- Modify `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify `src/superglm/dm_builder.py`
- Create `tests/test_factor_smooth_matrix.py`

**Red tests**

- exact/discrete `matvec`, `rmatvec`, Gram, weighted RHS, cross-Gram, row
  subset, and `toarray()` match a small materialized level-major design;
- fitted levels include all levels and remain fixed after row subsetting;
- exact and discrete matrices agree at shared support points;
- monkeypatch guards prove ordinary operations never call `toarray()`;
- allocation guards reject `n x (Kk)` construction;
- tabmat planning excludes the compact factor-smooth block while retaining
  tabmat for eligible small blocks.

**Implementation**

- Add factor-smooth fields to `GroupInfo` without placing repeated penalties in
  `penalty_matrix`.
- Implement `FactorSmoothGroupMatrix` with exact raw CSR basis or
  `B_unique/bin_idx`, factor codes, `K`, `k`, natural-basis map, levels, and
  repeated component metadata.
- Add compiled row kernels for factor-smooth sufficient statistics and
  factor-smooth-by-small cross-products only where the tests/profile require
  them.
- Build the raw P-spline marginal once, then reproduce the reference implementation
  `nat.param(..., type=1)` so the wiggle block and null coordinates have stable
  scaling.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_matrix.py -q
uv run pytest tests/test_matrix_execution_plan.py tests/test_group_matrix_algebra.py -q
```

**Commit:** `Add compact factor smooth matrices`

## Task 3: Repeated penalty algebra and dense oracle

**Files**

- Modify `src/superglm/types.py`
- Modify `src/superglm/model/reml_setup.py`
- Modify `src/superglm/reml/penalty_algebra.py`
- Modify `src/superglm/reml/gradient.py`
- Modify `src/superglm/solvers/hessian_factor.py`
- Modify `src/superglm/solvers/irls_direct.py`
- Create `tests/test_factor_smooth_penalties.py`
- Create `tests/test_factor_smooth_reml.py`

**Red tests**

- the wiggle component rank is `K * rank(S_base)` and each null component rank
  is `K`;
- quadratic, matvec, trace, joint log determinant, gradient, Hessian, and
  active penalty rank match explicit Kronecker references;
- no repeated component contains a `(Kk) x (Kk)` array;
- fixed-lambda Gaussian dense fits match an explicitly materialized penalized
  least-squares oracle;
- estimated-lambda Gaussian and Poisson dense fits converge with finite
  objectives and fully penalized curves.

**Implementation**

- Extend `PenaltyComponent` with repeat count and local block width.
- Keep repeated component matrices at `k x k`; implement closed-form repeated
  quadratic/matvec/rank/logdet derivatives.
- Materialize repeated blocks only inside forced dense-reference assembly and
  dense Hessian-factor methods.
- Make factor-smooth groups REML eligible and preserve their fixed natural
  basis across lambda trials.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_penalties.py tests/test_factor_smooth_reml.py -q
```

**Commit:** `Add repeated factor smooth REML penalties`

## Task 4: Exact block-Schur factor

**Files**

- Modify `src/superglm/solvers/structured.py`
- Modify `src/superglm/solvers/hessian_factor.py`
- Create `tests/test_block_schur_factor.py`

**Red tests**

- solve, multi-RHS solve, log determinant, rank, selected inverse blocks, and
  selected inverse diagonals match dense SPD references;
- repeated-penalty traces and pairwise cross traces match dense references;
- dense-small penalty cross traces match;
- block-diagonal-plus-low-rank operator traces, diagonals, squared diagonals,
  and cross traces match materialized references;
- singular local blocks and Schur complements expose useful term/block
  diagnostics;
- large structured blocks refuse full inverse materialization.

**Implementation**

- Add a block operator with `A(q,q)`, `C(K,k,q)`, and `D(K,k,k)`.
- Add `BlockSchurFactor` using batched local Cholesky/inverse operations and
  the existing residual-checked Schur policy.
- Add profiled-intercept adapter implementing the full `HessianFactor`
  protocol.
- Add block-diagonal-plus-low-rank internal algebra for W-correction and EDF
  operations. Keep scalar DLR code unchanged.

**Verify**

```bash
uv run pytest tests/test_block_schur_factor.py -q
uv run pytest tests/test_structured_irls.py -q
```

**Commit:** `Add block Schur Hessian factor`

## Task 5: Factor-smooth structured sufficient statistics

**Files**

- Modify `src/superglm/solvers/structured.py`
- Modify `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify `src/superglm/_group_matrix/_group_matrix_algebra.py`
- Create `tests/test_factor_smooth_structured_system.py`
- Modify `tests/test_structured_allocations.py`

**Red tests**

- `A`, `C`, per-level `D`, `X'W`, and `X'Wz` match dense references for
  exact and discrete matrices;
- nonuniform and signed derivative weights are supported;
- a second smaller random effect and ordinary numeric/categorical/spline
  terms remain in the exact small block;
- cached layout reuse avoids rebuilding the small execution plan;
- monkeypatch allocation guards forbid dominant materialization and full
  tabmat sandwiching.

**Implementation**

- Add block layout/system dataclasses parallel to scalar layout/system.
- Dispatch generic structured builders from the selected dominant matrix
  type.
- Reuse the fused dense-small moment path and pruned tabmat execution plan.
- Add only profiled factor-smooth aggregation kernels.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_structured_system.py \
  tests/test_structured_allocations.py -q
```

**Commit:** `Add factor smooth structured moments`

## Task 6: Exact IRLS and REML integration

**Files**

- Modify `src/superglm/solvers/irls_direct.py`
- Modify `src/superglm/reml/gradient.py`
- Modify `src/superglm/reml/w_derivatives.py`
- Modify `src/superglm/reml/observed_geometry.py`
- Modify `src/superglm/model/reml_finalize.py`
- Create `tests/test_factor_smooth_structured_parity.py`

**Red tests**

- forced Gram versus structured parity covers coefficients, intercept,
  predictor, deviance, EDF, lambdas, objective, log determinant, gradient,
  Hessian, and convergence;
- Gaussian, Poisson, and one estimated-scale family are covered;
- parity covers offsets, nonuniform weights, a global main smooth, and a
  smaller secondary random effect;
- finite differences validate all factor-smooth lambda derivatives;
- retained compact factors support covariance/EDF without a dense inverse.

**Implementation**

- Replace scalar-only calls at IRLS/observed/W-correction boundaries with
  structured dispatch while retaining scalar fast paths.
- Distill scalar or block terminal state through one structured-state
  contract.
- Ensure auto selection compares actual `K`, block width `k`, and small width
  `q`; forced structured errors clearly when ineligible.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_structured_parity.py \
  tests/test_random_effect_reml.py tests/test_structured_irls.py -q
```

**Commit:** `Use block Schur solves for factor smooth REML`

## Task 7: Structured discrete POI/fREML cache

**Files**

- Modify `src/superglm/reml/discrete.py`
- Modify `src/superglm/solvers/structured.py`
- Create `tests/test_factor_smooth_discrete.py`

**Red tests**

- exact versus discrete predictions/lambdas/objective agree at adequate bin
  resolution;
- forced Gram versus structured discrete fits agree;
- lambda-only trials update repeated local penalties and perform zero data
  passes;
- telemetry reports block cache solves/data passes;
- dense-factor and row-materialization guards remain armed during cached
  trials.

**Implementation**

- Generalize the cached structured solution dispatch to scalar or block
  systems.
- Reuse fixed `B_unique/bin_idx` data summaries across all lambda trials.
- Keep terminal exact prediction state independent of the discrete fit cache.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_discrete.py \
  tests/test_random_effect_discrete.py -q
```

**Commit:** `Add cached factor smooth discrete REML`

## Task 8: Prediction, compact state, and reporting

**Files**

- Modify `src/superglm/model/base.py`
- Modify `src/superglm/model/api.py`
- Modify `src/superglm/model/reml_finalize.py`
- Modify `src/superglm/model/state_ops.py`
- Modify `src/superglm/inference/covariance.py`
- Create `src/superglm/inference/factor_smooths.py`
- Modify `src/superglm/inference/__init__.py`
- Modify `src/superglm/__init__.py`
- Create `tests/test_factor_smooth_inference.py`

**Red tests**

- conditional prediction scores known levels; population prediction zeros the
  whole factor-smooth deviation;
- unseen population/error and missing policies work in exact and discrete
  models;
- compact prediction/reporting survives `retain_fit_state=False` and pickle
  round trips;
- `factor_smooth()` returns shared lambdas/variance views, term EDF, level
  support, level EDF, normalized credibility, common-grid curves, and
  pointwise posterior SE;
- local credibility equals
  `trace(I - D_j^{-1} P_j) / k`;
- pointwise SE and level EDF match dense covariance references;
- collapsed and insufficient-support diagnostics are explicit.

**Implementation**

- Treat `FactorSmooth` as a structured contribution in population prediction.
- Retain per-level local information blocks and support totals in the
  structured state.
- Use selected per-level inverse blocks; never request the full dominant
  covariance.

**Verify**

```bash
uv run pytest tests/test_factor_smooth_inference.py \
  tests/test_random_effect_inference.py tests/test_prediction.py -q
```

**Commit:** `Add factor smooth prediction and reporting`

## Task 9: Pinned the reference implementation `bs="fs"` parity

**Files**

- Create `tests/fixtures/factor_smooth_the reference implementation_reference.R`
- Create `tests/fixtures/factor_smooth_the reference implementation_reference.json`
- Create `tests/test_factor_smooth_the reference implementation_parity.py`

**Cases**

- Gaussian `gam(..., method="REML")`;
- Poisson `gam(..., method="REML")`;
- Poisson `bam(..., method="fREML", discrete=TRUE)`;
- with and without a global `s(x)`;
- population and unseen-level predictions.

Use `s(x, f, bs="fs", k=..., xt=list(bs="ps"), m=2)` and pin R 4.5.3 /
the reference implementation 1.9-4. Compare predictions, deviance, EDF, fitted penalty/variance views,
curve shapes, and population behavior. Compare lambdas only after confirming
the natural-parameter penalty scaling.

**Verify**

```bash
Rscript tests/fixtures/factor_smooth_the reference implementation_reference.R
uv run pytest tests/test_factor_smooth_the reference implementation_parity.py -q
```

**Commit:** `Pin factor smooth parity against the reference implementation`

## Task 10: cProfile, crossover, docs, and release verification

**Files**

- Extend `benchmarks/profile_structured_credibility.py`
- Extend `benchmarks/structured_credibility_profile_summary.md`
- Modify user/API documentation and changelog files selected from the current
  repository structure
- Add allocation/crossover regression tests where profiling demonstrates a
  stable decision boundary

**Profile matrix**

- `K` in tens through the supported profiled range;
- `k` in 5 and 10;
- exact and discrete Poisson;
- Gaussian/estimated-scale spot checks;
- with and without a global spline;
- dense comparison only where resource-safe.

Record clean wall repetitions separately from cProfile, tracemalloc, and
system telemetry. Explain dominant call stacks and any added kernels. Do not
encode wall-clock assertions in CI.

**Final verification**

```bash
uv run ruff format src/ tests/ benchmarks/
uv run ruff check src/ tests/ benchmarks/
uv run mypy src/
uv run pytest tests/ -q
uv run python run_test.py
git diff --check
```

Confirm no LSS paths changed:

```bash
git diff --name-only origin/master...HEAD | grep -i lss
```

Expected: no output.

**Commit:** `Document and profile factor smooths`
