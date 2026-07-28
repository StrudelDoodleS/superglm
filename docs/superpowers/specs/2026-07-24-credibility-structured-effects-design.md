# Credibility, Random Effects, Factor Smooths, and Structured REML Design

## Goal

Add actuarial credibility to SuperGLM through genuine REML-estimated random
effects, then extend the same machinery to mgcv-style factor smooths. Both
features must preserve SuperGLM's compact categorical and discretized
representations and must remain practical when the structured term has far
more coefficients than the rest of the model.

The work has two user-visible outcomes:

1. `RandomEffect` supplies partially pooled level effects, population
   prediction for unseen levels, variance-component reporting, and an
   actuarial credibility table.
2. `FactorSmooth` supplies a fully penalized smooth curve per factor level,
   with smoothing parameters shared across levels and population prediction
   for unseen levels.

A new exact block-Schur linear algebra backend supports both outcomes. The
existing dense direct solver remains the numerical oracle and fallback.

LSS is explicitly outside this design and must not be modified.

## Approved delivery split

### Delivery A: scalar credibility

- `RandomEffect` public feature.
- Compact all-level categorical matrix representation.
- Implicit identity penalty.
- Dense reference implementation.
- Exact largest-block Schur backend for ordinary and `discrete=True` REML.
- Conditional and population prediction.
- Variance-component and level credibility reporting.
- Dense-versus-structured, mgcv, and performance evidence.

### Delivery B: random smooth curves

- `FactorSmooth` public interaction term.
- Compact exact and discretized factor-by-spline representations.
- Fully penalized repeated spline basis with smoothing parameters shared
  across levels.
- Block-diagonal extension of the Schur backend.
- Conditional and population prediction.
- Per-level curve, uncertainty, support, and effective-credibility reporting.
- Dense-versus-structured, mgcv `bs="fs"`, and performance evidence.

Delivery B starts only after Delivery A has a correct dense oracle, structured
parity, and stable profiling results. The structured backend is designed for
block size greater than one from the outset so Delivery B extends it rather
than creating a second solver.

The two deliveries receive separate implementation plans and review
checkpoints. Delivery A must be complete and verified before the Delivery B
plan is executed. This keeps the solver foundation and scalar statistical
contract independently reviewable while retaining one coherent architecture.

## Relationship to mgcv

### Random effects

The first `RandomEffect` contract matches the factor random-intercept case
`s(f, bs="re")`:

- all fitted levels receive a coefficient;
- the design is equivalent to `~f - 1`;
- coefficients receive a full-rank identity penalty;
- no centering side condition is required;
- REML estimates the penalty/variance component;
- an unseen prediction level contributes zero.

General mgcv `bs="re"` terms may also contain numeric predictors and their
parametric interactions. Random slopes and general random-coefficient terms
are later extensions, not part of Delivery A.

### Factor smooths

`FactorSmooth(x, group=f, ...)` matches the statistical structure of
`s(x, f, bs="fs")`:

- every fitted factor level receives a smooth curve;
- every curve uses the same basis definition;
- wiggliness smoothing parameters are shared across levels;
- the base spline null space is also penalized, so an entire unsupported
  curve can shrink to zero;
- no per-curve centering constraint is imposed;
- an unseen level contributes the population curve deviation of zero.

The existing `SplineCategorical` term is not changed and is not relabelled as
an `fs` equivalent. It remains a reference-level varying-coefficient
interaction with its current grouping and penalty semantics.

Delivery B initially supports a singly penalized P-spline marginal. Additional
singly penalized bases and sum-to-zero (`bs="sz"`) factor interactions are
separate work.

## Public API

### RandomEffect

```python
from superglm import RandomEffect, SuperGLM

model = SuperGLM(
    family="poisson",
    features={
        "driver_age": Spline(kind="ps", k=14),
        "vehicle_model": RandomEffect(),
        "broker": RandomEffect(),
    },
    selection_penalty=0,
    direct_solve="auto",
    discrete=True,
)
model.fit_reml(X, y, offset=np.log(exposure))
```

Initial constructor:

```python
RandomEffect(
    *,
    unseen="population",
    missing="error",
    lambda_policy=None,
)
```

- `unseen` accepts `"population"` and `"error"`. `"population"` is the
  default and produces a zero contribution for an unknown level.
- `missing` accepts only `"error"` in this release. The parameter is explicit
  so a future governed missing-level policy does not require an incompatible
  API.
- `lambda_policy` follows the existing `LambdaPolicy` contract. The default
  estimates lambda; a fixed policy is useful for parity tests and controlled
  deployment.

`RandomEffect` is officially supported through `fit_reml()`. `fit()` and
`fit_path()` reject it with a message directing users to REML rather than
silently applying selection-penalty semantics to a variance component.

### FactorSmooth

```python
from superglm import FactorSmooth, RandomEffect, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={
        "driver_age": Spline(kind="ps", k=14),
        "vehicle_model": RandomEffect(),
    },
    interactions=[
        FactorSmooth(
            "driver_age",
            group="broker",
            kind="ps",
            k=6,
        ),
    ],
    selection_penalty=0,
    direct_solve="auto",
    discrete=True,
)
model.fit_reml(X, y, offset=np.log(exposure))
```

Initial constructor:

```python
FactorSmooth(
    variable,
    *,
    group,
    kind="ps",
    k=6,
    m=2,
    unseen="population",
    missing="error",
    lambda_policy=None,
    name=None,
)
```

`interactions` is extended to accept explicit interaction-spec objects in
addition to `(left, right)` tuples. An explicit `FactorSmooth` reads its two
named columns directly and owns its spline basis. Its columns need not both be
registered as main features. This allows an optional global `Spline(variable)`
without forcing a fixed group main effect.

A global `Spline(variable)` is recommended when the factor smooth represents
level-specific deviations from a portfolio curve, but it is not required.
The factor-group column should not simultaneously be included as a
`RandomEffect` unless the factor smooth excludes an equivalent constant
component; that exclusion is not available in this release. The model rejects
that duplicate random-intercept geometry with a clear error.

### Prediction

```python
model.predict(X_new)
model.predict(X_new, random_effects="conditional")
model.predict(X_new, random_effects="population")
```

The signature becomes:

```python
predict(X, offset=None, *, random_effects="conditional")
```

- `"conditional"` is the backward-compatible default and uses fitted
  structured effects for known levels.
- `"population"` sets every `RandomEffect` and `FactorSmooth` contribution to
  zero while retaining fixed effects and ordinary smooths.
- Under conditional prediction, a term configured with
  `unseen="population"` also contributes zero for an unseen level.
- Under conditional prediction, `unseen="error"` preserves strict validation.
- Missing values fail under both modes in this release.

The sklearn wrappers retain conditional prediction through their conventional
`predict(X)` signature. An explicit sklearn population-mode API is deferred.

### Reporting

```python
result = model.random_effects("broker", exposure=exposure)
result.variance_component
result.standard_deviation
result.table

curves = model.factor_smooth("driver_age:broker:fs", exposure=exposure)
curves.level_summary
curves.curves
```

`RandomEffectResult` contains:

- term name;
- estimated lambda;
- dispersion phi used for the conversion;
- variance component `tau_squared = phi / lambda`;
- standard deviation `sqrt(tau_squared)`;
- term EDF;
- boundary/collapse diagnostics;
- a level table.

The level table contains:

- level;
- row count;
- summed fit weight;
- summed exposure when explicitly supplied;
- vectorized conditional unpooled effect;
- shrunk conditional effect;
- response-scale relativity for a log-link model;
- posterior standard error;
- effective credibility;
- shrinkage `1 - credibility`;
- finite/support diagnostic flags.

The unpooled effect is the unpenalized scalar level update with the fitted
intercept and every other fitted term held fixed. It is computed by vectorized
per-level Fisher scoring, not by fitting one model per level. For a Poisson
log-link it reduces to the familiar aggregated actual-versus-expected log
ratio. It is explicitly described as a conditional diagnostic, not as a
separate full unpooled model.

If row-scale fit state has been released, reporting quantities that require
training rows accept explicitly supplied training arrays or raise a targeted
error. Compact fitted effects, lambdas, variance components, standard errors,
and support totals remain available after state release.

For an identity-penalized random effect, actuarial credibility uses the local
conditional information for the level. If `I_j` is its fitted working
information before the random-effect penalty, then

```text
Z_j = I_j / (I_j + lambda)
    = 1 - lambda * [D_j^{-1}]
```

where `D_j = I_j + lambda` is the level's conditional penalized block. This is
the familiar exposure-versus-process-variance shrinkage view. Posterior
standard errors are separate: they use the selected inverse block from the
full fitted system and therefore include uncertainty and correlation from the
other fitted terms. Credibility values are permitted a small numerical
tolerance around `[0, 1]`; material violations produce a diagnostic rather
than silent clipping.

`FactorSmoothResult` contains:

- shared penalty lambdas and corresponding variance-component views;
- term EDF;
- per-level row count, fit weight, optional exposure, level EDF, and
  normalized effective credibility;
- per-level evaluated curves on a common grid;
- pointwise posterior standard errors;
- collapse and insufficient-support diagnostics.

For a factor-smooth block of size `k`, let `G_j` be the level's unpenalized
working-information block, `P_j` its fitted total penalty block, and
`D_j = G_j + P_j`. Its local conditional influence is
`I_k - D_j^{-1} P_j`; normalized effective credibility is that matrix's trace
divided by `k`. Full-model level EDF and posterior curve uncertainty are
reported separately. Delivery B does not attempt an unpenalized raw curve for
thin levels.

## Statistical representation

### RandomEffect

For level code `j`, the contribution is `b_j`, with

```text
b ~ N(0, tau_squared I)
S(lambda) = lambda I
lambda = phi / tau_squared
```

All `K` levels are represented. A reference level is not dropped. The
full-rank penalty identifies the term jointly with the unpenalized intercept,
matching random-effect rather than fixed-categorical semantics.

The identity penalty is implicit. Code must not allocate `np.eye(K)`,
eigendecompose an identity matrix, or route the term through SSP
reparameterization. Penalty rank is `K`, the unscaled identity log determinant
is zero, the quadratic is `b.T @ b`, and penalty matvec/trace operations use
closed forms.

### FactorSmooth

Let `B(x)` be a `k`-column spline basis and `b_j` the coefficient vector for
factor level `j`. The contribution is

```text
g_j(x) = B(x) b_j
```

The base spline penalty is decomposed into its penalized range and null-space
components. Across `K` levels, each component is repeated:

```text
S_wiggle = I_K kron S_base
S_null_r = I_K kron (u_r u_r.T)
```

Each repeated component has one smoothing parameter shared across all levels.
This fully penalizes every curve without imposing a sum-to-zero constraint.
The repeated Kronecker penalties are implicit and are never materialized as
`(Kk) x (Kk)` arrays.

## Compact matrix types

### RandomEffectGroupMatrix

`RandomEffectGroupMatrix` shares the integer-code representation and algebraic
dispatch of `CategoricalGroupMatrix`, but:

- codes cover all `K` columns;
- there is no base/sink observation among fitted levels;
- metadata marks a structured scalar block and an implicit identity penalty;
- row subsetting preserves level geometry and penalty metadata;
- tabmat construction uses all categories explicitly (`drop_first=False`).

Ordinary `CategoricalGroupMatrix` remains unchanged. Shared behavior should be
factored into private helpers or inheritance only where doing so preserves its
current storage and public behavior.

### FactorSmoothGroupMatrix

`FactorSmoothGroupMatrix` stores:

- factor codes `(n,)`;
- either an exact shared sparse basis `B` or discretized `B_unique` plus
  `bin_idx`;
- `K`, `k`, knots, basis metadata, and repeated penalty metadata;
- level-major coefficient layout `(K, k)`.

Its operations never materialize the masked `n x (Kk)` design:

- `matvec` selects each row's level coefficient vector and evaluates one basis
  row;
- `rmatvec` aggregates weighted basis rows into `(K, k)`;
- diagonal Gram construction aggregates `(k, k)` blocks per level;
- row subsetting preserves global fitted levels;
- prediction evaluates known levels directly and returns zero/error for
  unknown levels according to policy.

The existing `SplineCategoricalGroupMatrix` remains available for current
reference-level interactions and is not rewritten into this representation.

## Kernel plan

Existing categorical, crosstab, sparse-basis, and discrete histogram kernels
are reused whenever their layouts match. New compiled kernels are limited to
operations demonstrated by profiling to be material:

1. scalar structured sufficient statistics:
   per-level `sum(W)`, `sum(Wz)`, and intercept cross-products;
2. scalar structured cross-products:
   fused aggregation of `W * X_small` by level;
3. categorical-by-discrete cross-products:
   level-by-bin weighted histograms followed by multiplication by
   `B_unique`, with a direct level-by-basis aggregation alternative;
4. factor-smooth sufficient statistics:
   per-level `B.T W B` blocks and `B.T Wz` vectors;
5. factor-smooth cross-products:
   per-level basis-by-small-block aggregation;
6. batched structured algebra:
   small SPD block factorization, solve, log determinant, inverse diagonal,
   and trace primitives.

For categorical-by-discrete work, the dispatcher selects between a dense
`K x n_bins` histogram and direct `K x k` basis aggregation using an explicit
cell/memory cap. No path creates an `n x K`, `n x (Kk)`, or `(Kk) x (Kk)`
temporary.

The initial correct implementation may compose existing kernels. cProfile,
phase timings, and allocation evidence decide which composed operations merit
a fused kernel. New kernels are not added solely on intuition.

## Structured linear algebra backend

### Block system

The backend selects one eligible structured term for elimination. Its
coefficients `b` have dimension `Kr`; every other coefficient, and the
intercept when solving the augmented IRLS system, is placed in the smaller
vector `a` of dimension `q`.

```text
H = [ A   C.T ]
    [ C    D  ]
```

- For `RandomEffect`, `r = 1` and `D` is diagonal.
- For `FactorSmooth`, `r = k` and `D` is block diagonal with `K` small SPD
  blocks.
- `A` is dense and small.
- `C` is stored as `K x r x q` or an equivalent level-major two-dimensional
  view.

With `F = D^{-1} C`, the exact Schur system is

```text
Q = A - C.T D^{-1} C
a = Q^{-1} (rhs_a - C.T D^{-1} rhs_b)
b = D^{-1} (rhs_b - C a)
```

and

```text
log|H| = log|D| + log|Q|.
```

Selected inverse blocks follow the standard block inverse:

```text
Hinv_aa = Q^{-1}
Hinv_bb = D^{-1} + F Q^{-1} F.T
Hinv_ba = -F Q^{-1}.
```

Only requested diagonal blocks, diagonals, solves, and traces are computed.
The backend never materializes `Hinv_bb` for a large structured term.

### Internal factor protocol

REML and inference stop requiring an unconditional dense `H^{-1}`. Both dense
and structured backends implement an internal factor protocol with:

- `solve(rhs)`;
- `logdet()`;
- `quadratic_penalty(beta, component)`;
- `trace_inverse_penalty(component)`;
- `selected_inverse_block(group_slice)`;
- `selected_inverse_diagonal(group_slice)`;
- the small products needed by REML gradient, Hessian, and W-correction terms.

The dense implementation wraps the current decomposition and is the reference
behavior. The structured implementation uses Schur identities. Public
`PIRLSResult` remains coefficient-based; factor objects and caches remain
private fit state.

### Backend selection

`direct_solve` accepts:

- `"auto"`: select structured algebra only when an eligible block exists and
  the estimated dense cost exceeds the structured cost;
- `"structured"`: force structured algebra and fail clearly when the model is
  ineligible;
- `"gram"` and `"qr"`: retain their current meanings and force existing dense
  behavior.

When more than one structured term is present, the eligible term with the
largest coefficient count is eliminated. Other random effects or factor
smooths remain in `A`. This is exact; it merely limits the speedup when several
terms are equally large.

The automatic crossover is selected from benchmark evidence and encoded
conservatively. Small models stay on the dense path to avoid Schur setup
overhead.

### Initial fallback conditions

The structured backend falls back under `"auto"` and errors under
`"structured"` for:

- fitted coefficient constraints or SCOP geometry that cannot yet be
  expressed in the small Schur system;
- an unsupported structured-term interaction;
- a numerically non-SPD local block or Schur complement after the same
  stabilization attempts allowed to the dense solver;
- any configuration for which required REML derivatives cannot be evaluated
  exactly by the factor protocol.

Fallback is reported once in fit diagnostics with a reason. It is never
silent in profiling output.

## Exact and discrete fitting

### Exact path

The ordinary direct IRLS loop continues to compute working responses and
weights. Instead of assembling a full Gram matrix, a structured Gram builder
produces `A`, `C`, `D`, and partitioned right-hand sides. The Schur backend
solves the same penalized working model and supplies exact log determinants
and selected inverse operations to direct REML.

W-correction code is refactored to consume factor solves and selected products
rather than a mandatory full inverse. Dense and structured gradients,
Hessians, objectives, accepted steps, and final fits must agree on models
small enough for both.

### `discrete=True` path

`RandomEffect` is never discretized; its factor codes remain exact.
`FactorSmooth` discretizes only its numeric spline marginal. Other existing
discrete smooths retain their current bases and binning.

The POI/fREML cache stores structured blocks and right-hand sides instead of a
full `XtWX` when the Schur backend is active. A lambda trial updates only the
appropriate local penalty blocks and reuses cached data summaries. Trial
solves and log determinants therefore require no data pass and avoid the
current dense `O(p^3)` cached solve.

The same factor protocol is used by exact and discrete REML. Discrete support
is a release requirement, not a later optimization.

## tabmat contract

tabmat remains the preferred matrix engine on eligible non-discrete paths.
The design does not replace or vendor it.

- Ordinary categorical and numeric models retain the current
  `tabmat.SplitMatrix` behavior.
- `RandomEffectGroupMatrix` has an explicit all-level
  `tabmat.CategoricalMatrix` representation.
- Selected tabmat sandwich operations may supply `A` and `C` blocks when that
  is faster than the native group kernels.
- Existing code still disables the whole-matrix tabmat path in the presence
  of SSP/discretized groups; structured fitting then uses the specialized
  group kernels rather than materializing a full sandwich.
- Dense fallback paths remain free to use tabmat exactly as before.

Performance changes to tabmat integration require call-stack and allocation
evidence. This work must not degrade current categorical benchmarks.

## Identifiability and overlapping terms

- A full-rank random-effect penalty makes all-level coding identifiable with
  the model intercept; no reference level or explicit centering projection is
  added.
- A fully penalized factor smooth is identifiable without per-level
  centering.
- A global main smooth may coexist with a factor smooth. The result follows
  `fs` semantics: the penalty identifies the decomposition but does not force
  the global curve to explain the maximum possible signal.
- A separate random intercept for the same grouping factor duplicates the
  constant null-space geometry of the initial factor smooth and is rejected.
- Existing fixed categorical and `SplineCategorical` identifiability behavior
  is unchanged.

## Numerical behavior and diagnostics

- Local `D` blocks and the Schur complement use Cholesky first and the same
  residual-checked robust fallback principles as the dense solver.
- Factorization failures identify the structured term, level/block when
  applicable, lambda state, minimum eigenvalue estimate, and whether dense
  fallback was used.
- REML lambda bounds retain the current global policy.
- A variance component at the effective zero boundary produces a collapsed
  component diagnostic and a user warning in the structured result, not a
  fit failure.
- Levels with no response information, separation, or non-finite conditional
  raw estimates remain fitted through pooling and receive explicit table
  flags.
- Prediction and reporting never silently reinterpret offsets as exposure.
  Exposure is aggregated only when the user explicitly supplies it.

## Correctness tests

Implementation follows red-green-refactor development. Tests are grouped as
follows.

### Feature and matrix invariants

- all `K` random-effect levels are represented with no base level;
- identity and repeated factor-smooth penalties remain implicit;
- compact matvec, rmatvec, Gram, row-subset, and prediction operations match
  materialized small references;
- unseen and missing policies behave exactly as configured;
- exact and discretized factor-smooth bases agree at shared support points;
- no compact operation calls `toarray()` on a prohibited large block.

### Dense versus structured parity

Forced `"gram"` and `"structured"` fits are compared on:

- Gaussian analytic random-intercept cases;
- Poisson, Gamma, NB2, and Tweedie fits;
- offsets and nonuniform fit weights;
- one random effect;
- one dominant plus one smaller random effect;
- random effects combined with numeric, categorical, spline, tensor, and
  discrete smooth terms;
- factor smooths with and without a global main smooth;
- exact and `discrete=True` fitting.

Parity covers coefficients after canonical mapping, intercept, linear
predictor, predictions, deviance, phi, EDF, lambda path, variance components,
REML objective, log determinant, gradient, Hessian, standard errors,
credibility, and convergence state.

Finite-difference tests cover REML derivatives for identity and repeated
factor-smooth penalty components.

### mgcv parity

Pinned R scripts and committed reference summaries cover:

- Gaussian and Poisson `s(f, bs="re")`;
- unseen-level population prediction;
- `s(x, f, bs="fs")` with a matched singly penalized spline basis;
- factor smooths with and without a global `s(x)`;
- `gam(..., method="REML")` and `bam(..., discrete=TRUE)` where applicable.

Because basis scaling can differ, parity prioritizes predictions, deviance,
EDF, variance components, curve shape, and population behavior. Lambda values
are compared only after a demonstrated common penalty scaling.

### Regression and fallback tests

- every pre-existing categorical, interaction, REML, discrete, prediction,
  summary, and state-retention test remains unchanged and passing;
- `"auto"` selects and reports the intended backend;
- `"structured"` rejects unsupported constraints explicitly;
- `"auto"` uses dense fallback with a recorded reason;
- ordinary `Categorical` and `SplineCategorical` outputs do not change;
- released fit state preserves compact structured prediction and reporting.

## Performance and profiling gates

A dedicated benchmark/profiling entry point builds on
`superglm.profiling.harness` and produces:

- raw `cprofile.pstats`;
- cumulative-time and total-time call-stack reports;
- phase-level SuperGLM REML timings;
- wall-time repetitions after warmup;
- RSS/USS and CPU telemetry;
- tracemalloc summaries;
- model dimensions, backend choice, fallback reason, iteration counts,
  objective, and parity diagnostics.

The benchmark matrix includes:

- random effect: `K` from 100 through 10,000, small and large `n`, and small
  blocks of ordinary model terms;
- two random effects with one dominant block;
- factor smooth: level count from tens through the supported profiled range,
  `k` values 5 and 10;
- exact and `discrete=True`;
- Gaussian and Poisson core cases plus an estimated-scale case;
- dense comparisons wherever the dense reference is resource-safe.

Acceptance requires:

- no allocation asymptotic in `n*K`, `n*K*k`, or `(K*k)^2` on the structured
  path;
- no Python loop over observations or separate model fit per level;
- exact data aggregation dominated by compiled kernels/tabmat;
- cached discrete lambda trials perform no data pass;
- structured results remain within the agreed numerical parity tolerances;
- structured fitting is materially faster and lower-memory beyond the
  measured crossover;
- `"auto"` does not regress representative small-model performance by choosing
  structured algebra prematurely;
- the profile report explains remaining dominant time rather than merely
  reporting a headline wall time.

Wall-clock assertions are not placed in ordinary CI. CI enforces path,
allocation, and parity invariants; the committed benchmark report establishes
the release performance claim.

## Compatibility

- Existing `Categorical`, grouping, `SplineCategorical`, prediction, and
  rating-table behavior remains unchanged.
- Existing model persistence continues to own fitted levels, knots, compact
  matrices, coefficients, and prediction policy.
- Existing models default to the same dense behavior unless an eligible
  structured term is present and `"auto"` selects it.
- `predict(X, offset)` remains source compatible because
  `random_effects` is keyword-only.
- Explicit interaction-spec objects extend rather than replace tuple
  interactions.
- Public exports are added through `superglm.features` and `superglm`.

## Non-goals

- LSS changes of any kind.
- A general sparse GLMM or arbitrary mixed-model formula engine.
- Correlated random-effect covariance matrices.
- Nested/crossed multi-block sparse elimination beyond selecting one largest
  exact structured block.
- Random slopes or general multi-predictor `bs="re"` terms.
- `bs="sz"` factor smooth interactions.
- Changing existing `SplineCategorical` semantics.
- Making every spline basis available inside `FactorSmooth` in the first
  release.
- Supporting random effects through selection-penalty `fit()`/`fit_path()`.
- Claiming unlimited level counts without benchmark evidence.
