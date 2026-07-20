# Dataframe Boundary and Developer Accessibility Design

## Goal

Make SuperGLM accept eager pandas and Polars dataframes through one clear data boundary while
leaving its numerical core on the existing exact `DesignMatrix`, `GroupMatrix`, Tabmat, and
compressed discrete representations.

The same boundary must reduce the effort needed to understand and change the library. A developer
should be able to locate dataframe handling, feature compilation, matrix execution, and solver
logic without tracing backend-specific operations through IRLS, PIRLS, or REML.

This project copies the useful architectural lesson from GLUM—normalize dataframe access at the
edge and compile once into an algorithm-specific matrix representation. It does not import GLUM's
solver architecture or replace SuperGLM's spline, interaction, penalty, REML, constraint, state,
or discrete execution systems.

CI sharding and the broader dependency-compatibility audit are separate projects. This design adds
only the dependency declarations and tests directly required by the dataframe boundary.

## Selected approach

Three approaches were considered:

1. **A narrow dataframe adapter followed by the existing feature compiler (selected).** Wrap an
   eager native frame once per public operation, expose a small set of named column/schema/slicing
   operations, and compile referenced columns into the current numerical representations. This
   isolates backend semantics without moving abstraction or dispatch into hot numerical paths.
2. **Convert Polars to pandas at entry.** This is initially small but creates an avoidable
   whole-frame conversion, can lose categorical semantics, increases peak memory, and makes
   “Polars support” mostly cosmetic.
3. **Rewrite all feature and matrix code against a generic dataframe API.** This offers the widest
   theoretical backend support but spreads dataframe abstraction through a mature numerical
   system, increases cognitive load, and risks changing established memory and dispatch behavior.

The selected approach uses Narwhals only at the public dataframe boundary. It does not pass
Narwhals objects into feature implementations, matrix kernels, IRLS/PIRLS, REML, or published
model state. It does not call `to_pandas()` or convert the whole input frame.

## Two-layer mental model

The user-facing layer is deliberately short:

```text
pandas.DataFrame or eager polars.DataFrame
                    |
                    v
        SuperGLM public model-data API
 fit / fit_reml / profile / predict / metrics / CV
```

The developer-facing layer makes the compilation boundary explicit:

```text
native eager frame
        |
        v
private frame adapter
  names, schema, selected column arrays, row slicing, fingerprint
        |
        v
feature compiler
  Numeric / Categorical / Spline / Interaction / constraints / penalties
        |
        v
exact numerical design
  DesignMatrix + GroupMatrix blocks
  dense / sparse / native categorical / compressed support-space
        |
        v
construction-time execution plans
  specialised kernels / Tabmat / discrete bin-space / stable fallbacks
        |
        v
IRLS / PIRLS / REML
```

Dataframe choice ends at the frame-adapter-to-feature-compiler boundary. Solver code receives the
same arrays, matrix classes, group slices, penalties, and execution plans for equivalent pandas
and Polars inputs.

## Scope

### Supported inputs

The first supported native frame types are:

- `pandas.DataFrame`;
- eager `polars.DataFrame`.

Polars `LazyFrame` is intentionally unsupported. The error must say that SuperGLM requires an
eager frame and that the caller should collect it first. Other Narwhals-compatible backends are
not advertised or accepted until they have explicit semantic and regression coverage.

SuperGLM continues to use positional row alignment between `X`, `y`, `sample_weight`, and
`offset`. A pandas index is metadata for mutation fingerprinting, not a join key. No automatic
index alignment is introduced.

### Public entry points

All public model-data entry points that currently accept `X` must share the same boundary. This
includes fitting, REML, path fitting, family-parameter profiling, exact and fast-discrete
prediction, metrics, model tests and diagnostics that consume data, shape post-fit operations,
rating-table construction, and cross-validation.

Supporting only `fit()` while leaving prediction, profiling, cross-validation, or diagnostics
pandas-only would create two effective APIs and is not acceptable. Output tables remain pandas
objects in this project.

Custom user callbacks, cross-validation splitters, and scorers continue to receive the native
frame type supplied by the caller. The private adapter is never part of a public callback
contract.

### Non-goals

This project does not:

- remove pandas as a core dependency;
- add a selectable output-dataframe backend;
- support lazy query execution;
- support arbitrary dataframe-interchange objects;
- add NumPy matrix input to named-feature model APIs;
- rewrite feature mathematics or categorical estimation rules;
- route raw dataframes directly into `tabmat.from_df`;
- change ordinary, discrete, Tabmat, IRLS, PIRLS, REML, SCOP, or profiling algorithms;
- replace specialised matrix classes with one generic matrix abstraction;
- move runtime backend selection into solver iterations;
- add Polars or PyArrow as runtime dependencies;
- change numerical tolerances, candidate ordering, fallback ordering, or transactional
  publication.

## Boundary component

Add one focused private module for dataframe access. It owns a concrete eager-frame adapter rather
than a registry, plugin framework, or hierarchy of backend strategies.

The adapter has five responsibilities:

1. validate that the native object is a supported eager frame;
2. expose row count, ordered unique column names, and normalized column kinds;
3. return a selected column as a NumPy array without converting unrelated columns;
4. slice rows for cross-validation while preserving the caller's native backend;
5. calculate the fit-data fingerprint used by retained-state mutation checks.

The adapter lazily caches a referenced column's extracted array for the duration of one public
operation. Main effects and interactions must not independently convert the same Polars column.
The cache is operation-local, bounded by referenced columns, and discarded before fitted state is
published. It must not become another long-lived model cache.

The adapter exposes backend-neutral facts needed by auto-detection, such as numeric, boolean,
string, or categorical kind. Feature implementations continue to consume NumPy arrays and their
existing explicit configuration. They do not import Polars or inspect Narwhals objects.

Narwhals is declared directly as `narwhals>=2.17.0`, the version already present in the current
lock through Tabmat. The implementation does not upgrade Narwhals merely to make this dependency
direct. Relying on Tabmat's transitive dependency would make an implementation detail of Tabmat
part of SuperGLM's public input contract. Polars remains optional and appears only in the
development/test environment used to verify the integration. PyArrow is not required by this
path.

## Type and categorical semantics

Equivalent logical pandas and Polars columns must compile to equivalent SuperGLM features.

- Numeric integer and floating columns remain numeric.
- Boolean columns retain the current pandas behavior.
- pandas string/object/categorical columns and Polars string/categorical/enum columns are
  categorical candidates during auto-detection.
- Backend category codes are not solver inputs. Polars categorical and enum values are decoded to
  logical values before the existing SuperGLM categorical normalization runs. This avoids relying
  on backend-specific or process-local category codes.
- Existing `Categorical` base-level, sorting, grouping, unseen-level, and missing-value rules stay
  authoritative.
- Existing `OrderedCategorical` configuration, rather than backend category metadata, remains the
  authority for order.
- Existing feature-level finite-value and null rejection remains in force. This project does not
  introduce imputation.
- Complex columns retain the current rejection. Temporal, nested, list, struct, and mixed-object
  columns gain no implicit new interpretation: an already-supported explicit pandas feature case
  remains supported, while an unrepresentable pandas or Polars case fails before feature state is
  learned and names the offending column.

The adapter must preserve current pandas behavior for already-supported inputs. Any unavoidable
cross-backend difference must be rejected explicitly rather than silently coerced.

## Data flow and ownership

### Fit and REML

1. The public entry point retains the caller's native references for existing callback and fitted
   state behavior.
2. Input validation constructs one operation-local frame adapter before a `FitWorkspace` is
   started or feature state is learned.
3. Validation checks frame type, eager execution, row count, column uniqueness, required columns,
   and unsupported dtypes. Response, weight, offset, and family-domain validation remain
   unchanged.
4. The private fit workspace receives the validated adapter and normalized vectors.
5. Auto-detection and design construction request only referenced column arrays from the adapter.
6. Feature specs compile those arrays into the same `GroupInfo`, `GroupMatrix`, `GroupSlice`,
   penalty, and constraint objects used today.
7. The adapter is no longer involved after `DesignMatrix` construction. Every solver iteration
   operates on the existing numerical design and execution plans.
8. Successful publication retains the native caller frame where current retained-fit behavior
   requires it, plus a backend-aware content fingerprint. The adapter itself is not published.
9. Failure before publication leaves the previous fitted revision unchanged, as it does today.

### Prediction and secondary operations

Prediction creates a short-lived adapter, validates required feature and interaction columns, and
extracts each referenced column at most once. Exact and fast-discrete scoring then use the same
NumPy arrays and fitted feature metadata as the pandas path.

Secondary model-data operations reuse the same boundary. They must not create private pandas
copies merely to call another SuperGLM method. Reporting code can still construct pandas output
tables after numerical work is complete.

### Cross-validation

The splitter receives the original native frame. Fold indices are normalized as they are today.
The adapter performs backend-correct row selection: pandas uses positional selection and Polars
uses positional row gathering. Fold estimators and custom scorers receive native pandas folds for
pandas input and native Polars folds for Polars input.

No fold converts a complete Polars frame to pandas. Response, weight, offset, group, and
out-of-fold arrays remain NumPy arrays.

## Retained fit data and mutation safety

The existing `FitDataGuard` contract remains:

- retained caller data can be used only if its fit-time content is still verified;
- failed verification does not silently refresh or reuse stale fitted caches;
- cloned and unpickled models can verify an equal independent retained frame;
- compact `retain_fit_state=False` models do not retain row-scale input state.

The current pandas digest remains the pandas implementation so pandas mutation behavior does not
change. The Polars implementation hashes selected logical columns, schema metadata, row order, and
row count deterministically. The guard records the backend kind; cross-backend equality is not a
retained-state contract. This is distinct from prediction, where an equivalent supported backend
is accepted normally.

Because Polars frames are immutable at the API level, the fingerprint primarily protects clone,
pickle, and retained-reference assumptions. It must still detect replacement with different
values.

## Developer accessibility and inspectability

Add one permanent developer-facing architecture page linked from the contributor documentation.
It contains the two-layer graph above and a “where to make a change” map:

| Change | Primary boundary |
| --- | --- |
| Accept or normalize a dataframe dtype | frame adapter |
| Change feature basis or categorical encoding | feature implementation/compiler |
| Change storage for a feature block | `GroupMatrix` construction |
| Change Gram, matvec, or centered-moment execution | matrix execution plan/algebra |
| Change working responses or weights | working-row geometry |
| Change coefficient iteration or line search | IRLS/PIRLS solver |
| Change smoothness selection | REML objective/update/finalization |
| Add a non-GLM objective such as future AFT | objective/working-geometry contract, then only the specialised kernels it requires |

Add `SuperGLM.design_summary()`, a read-only fitted-design summary built from already-available
construction metadata. Calling it before a successful fit raises `RuntimeError`. It returns a
pandas table with one row per fitted term and these columns:

- `term` and `feature` identity;
- `solver_start`, `solver_end`, and `n_columns`;
- concrete `representation`;
- `compressed` and `storage_rows`;
- `ordinary_tabmat_partition`;
- `specialised_discrete_route`;
- the construction-time `route_reason` for accepting or rejecting those eligible routes.

The summary reports static storage and construction-time eligibility, not dynamic execution. It
must not claim that constructing a SplitMatrix proves a kernel was called. Existing fit/REML trace
and profile records remain authoritative for actual kernel calls, numerical certification
fallbacks, and line-search behavior; the developer page points maintainers to those records. No
unconditional per-iteration instrumentation, counters, scans, or allocations are added solely for
the summary. Producing the summary may inspect or construct the existing immutable
`MatrixExecutionPlan`, but it must not construct a Tabmat `SplitMatrix`, execute a matrix kernel,
or change a dispatch decision.

The summary is computed on demand and returned as a pandas table. It is not another authoritative
state inventory and does not participate in fit publication.

## Performance and memory invariants

The boundary is outside repeated numerical operations. In particular:

- no Narwhals, pandas, or Polars dispatch occurs inside IRLS, PIRLS, REML, line search, Gram,
  matvec, rmatvec, sandwich, or bin-space kernels;
- no whole-frame conversion is allowed;
- each referenced input column is extracted at most once per public operation unless an existing
  algorithm deliberately requests a distinct dtype conversion;
- no observation-by-full-design materialization is introduced;
- compressed discrete spline/tensor representations stay compressed;
- the existing lazy Tabmat construction and dispatch gates remain authoritative;
- unsupported Tabmat layouts continue to use their current specialised or stable fallback paths;
- pandas numeric columns retain zero-copy views where the current path safely does so;
- required contiguous/float64 conversion remains at the same numerical ownership boundaries;
- no adapter or native dataframe object is captured by a compiled kernel or execution plan.

Before/after measurements use current master as the pandas baseline and compare equivalent pandas
and Polars inputs. Representative cases include ordinary numeric, high-cardinality categorical,
mixed numeric/categorical, multiple splines, interactions, and `discrete=True` spline/tensor
designs for `fit`, `fit_reml`, and prediction.

A repeatable pandas median regression above 3% in stable matrix/design microbenchmarks or above 5%
end to end is a blocker unless raw samples demonstrate measurement noise. Any meaningful peak
memory increase, loss of actual Tabmat calls, extra full-column conversions, or discrete row
materialization is a blocker. Polars timings are reported separately; Polars support is not
allowed to slow the existing pandas route.

## Error handling

Boundary failures occur before feature learning and before a fit workspace can publish state.
Messages identify:

- unsupported native frame type;
- lazy rather than eager input;
- duplicate or missing columns;
- unsupported column and logical dtype;
- row-count mismatch;
- backend conversion failure;
- null, non-finite, complex, or unseen categorical values under the existing feature policy.

Narwhals or backend-specific exception text is not exposed as the primary public message. The
original exception remains as the chained cause for debugging. A failed fit, profile, or
post-fit operation preserves the previously published model revision.

## Testing

### Boundary tests

- supported pandas and eager Polars detection;
- clear rejection of Polars `LazyFrame` and unrelated objects;
- unique/missing column checks and positional row counts;
- selected-column extraction without whole-frame conversion;
- operation-local caching of repeated interaction columns;
- positional fold slicing with native backend preservation;
- package import and pandas use when Polars is not installed.

### Semantic parity

For equivalent pandas and Polars fixtures, compare:

- auto-detected feature kinds;
- categorical levels, base levels, grouping, unseen-level errors, and missing-value errors;
- ordered categorical behavior;
- numeric, boolean, string, categorical, and enum inputs;
- spline knots, bases, penalties, and discretization metadata;
- interaction designs;
- `GroupMatrix` concrete types, shapes, and support storage;
- coefficients, intercept, predictions, deviance, EDF, fitted scale, REML lambdas/objective/rank,
  and profile parameters within existing tolerances;
- convergence, iteration counts, line-search decisions, and fallback decisions.

### State and API coverage

- transactional rollback for invalid pandas and Polars calls;
- retained-state fingerprints, caller mutation behavior, compact state, deepcopy, pickle, and
  `clone_unfitted`;
- `fit`, `fit_path`, `fit_reml`, Tweedie/NB profiling, prediction variants, metrics, drop tests,
  diagnostics, shape operations, rating tables, and cross-validation;
- native frame types observed by custom splitters and scorers;
- pandas output types remain unchanged.

### Dispatch and performance coverage

- actual Tabmat sandwich/matvec/transpose calls for eligible ordinary designs;
- negative controls continue to reject Tabmat;
- specialised discrete and mixed bin-space calls remain active;
- zero calls to whole-frame pandas conversion;
- one extraction per referenced column per public operation;
- representative cold, warm, end-to-end, and peak-memory comparisons.

Run the established lint, formatting, type, lock, package, focused numerical, non-slow, and full
test gates. Polars tests run in CI with an explicit test dependency, while a separate import test
uses an environment without Polars.

## Implementation boundaries

Implementation proceeds in behavior-preserving layers:

1. Introduce the adapter and prove existing pandas validation, extraction, slicing, and digest
   behavior through focused tests.
2. Route fit validation, auto-detection, and design construction through the adapter while keeping
   generated numerical structures identical.
3. Route prediction and all secondary public model-data entry points through the same boundary.
4. Add eager Polars semantics and parity coverage without adding a Polars runtime dependency.
5. Add the on-demand design summary and the permanent two-layer developer guide.
6. Run numerical, dispatch, memory, and timing comparisons before publication.

Each layer remains reviewable. No layer combines dataframe work with solver or matrix-kernel
refactoring.

## Future extension story

This boundary deliberately makes later work easier without designing that work now.

For example, a future accelerated-failure-time objective would not add pandas or Polars branches.
It would reuse the frame adapter and compiled numerical design, define its response/censoring
contract at validation, add the objective's value/gradient/curvature or working-geometry contract,
and introduce only the specialised kernels its mathematics requires. Gram construction changes
would remain in matrix algebra; coefficient iteration changes would remain in the solver.

The dataframe project does not add AFT support or a generic objective framework. It merely makes
the ownership boundary clear enough that such a project has a visible starting point and call
graph.

## Acceptance criteria

The work is complete when:

- eager pandas and Polars frames work across the complete public model-data API;
- dataframe backend handling is confined to one private boundary;
- feature implementations receive arrays rather than native dataframe objects;
- solver and matrix hot paths contain no dataframe dispatch;
- equivalent inputs produce equivalent numerical structures and fitted results;
- categorical and retained-state semantics are explicit and tested;
- no whole-frame Polars-to-pandas conversion occurs;
- Polars and PyArrow are not runtime dependencies;
- pandas output behavior remains unchanged;
- the existing Tabmat and discrete dispatch decisions and gains remain;
- existing pandas fit time and memory remain within the stated gates;
- the fitted-design summary exposes storage and eligibility without adding hot-path profiling;
- the two-layer guide lets a maintainer identify the correct subsystem for dataframe, feature,
  matrix, working-geometry, solver, and REML changes;
- all established correctness, packaging, and performance gates pass.
