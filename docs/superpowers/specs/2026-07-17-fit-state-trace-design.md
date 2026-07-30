# Transactional Fit State and Authoritative Trace Design

Date: 2026-07-17

Status: approved for implementation. The user explicitly authorized the design gates to be
treated as approved while work proceeds autonomously.

## Scope

This is the first implementation unit in the broader fit and fit_reml remediation. It addresses:

- strong exception safety for fit(), fit_path(), fit_reml(), profiling, finalization, and
  post-fit shape repair;
- ownership of mutable feature, penalty, family, interaction, and retained-data state;
- history-independent resolution of automatic lambda, negative-binomial theta, categorical
  bases, auto-detected schemas, and fitted feature metadata;
- common response, weight, offset, and shape validation;
- truthful fit, IRLS, REML, SCOP, and finalization traces;
- convergence claims tied to the exact state on which they were evaluated;
- performance and memory gates for the refactor.

It does not change the REML criterion, Hessian, penalty-rank convention, scale likelihood, or
curvature policy. Those changes belong to subsequent Wood-correct numerical-core specs. This
unit creates the state and trace boundaries those changes will consume.

## Numerical and licensing provenance

Published work by Simon Wood is the authoritative numerical reference for this development
cycle, in particular:

- Wood (2011), “Fast stable restricted maximum likelihood and marginal likelihood estimation
  of semiparametric generalized linear models,” Journal of the Royal Statistical Society B.
- Wood, Pya, and Säfken (2016), “Smoothing parameter and model selection for general smooth
  models,” Journal of the American Statistical Association.
- Wood and Fasiolo (2017), “A generalized Fellner-Schall method for smoothing parameter
  optimization with application to Tweedie location, scale and shape models,” Biometrics.

No the reference implementation source code, implementation detail, or copied algorithm text may be used as a design or
implementation source. No the reference implementation code may be vendored. Existing black-box parity fixtures may be
retained as secondary regression evidence, but formulas, invariants, and implementation decisions
must be independently derived from the published papers and documented dense oracles.

QuantCo glum and Tabmat may inform general solver and matrix-engineering patterns under their
respective licenses. Code must still be independently implemented unless an explicit compatible
reuse decision is documented.

## Current defects established by audit

The current model has no commit boundary. A fit attempt mutates model configuration, fitted
features, family, penalty, design, groups, retained arrays, caches, result, statistics, and
canonical prediction state in stages. A failure can therefore retain an old coefficient result
alongside a new design or partially cleared caches.

Specific established failures include:

- fit() validates the response only after building and installing the design.
- fit_reml() does not call the central response validator.
- the validator implements only the Binomial domain despite claiming broader checks.
- caller-owned FeatureSpec, Penalty, and interaction objects are stored and mutated by identity.
- selection_penalty=None is replaced by the first resolved lambda and is no longer automatic.
- NegativeBinomial(theta="auto") is replaced by a fixed family and is no longer automatic.
- categorical most-exposed bases and auto-detected schemas can be inherited from earlier data.
- final REML refits and runtime canonicalization mutate the model before all failure points pass.
- post-fit shape repair changes public coefficients while leaving solver, REML, statistics,
  prediction, covariance, EDF, and cache state inconsistent.
- non-SCOP REML debug rows are synthesized after the fact.
- enabling detailed non-SCOP tracing runs a discarded one-iteration replay.
- actual exact, discrete, EFS, and final-refit states are not connected to one authoritative
  terminal identity.
- current iteration rows mix pre-step weights and factorization metrics with post-step retained
  predictions.

## Design principles

1. No externally visible model mutation occurs until a complete fitted artifact is valid.
2. Configuration intent and learned fit state are separate.
3. Every learned scalar, array, factorization, trace value, and convergence claim belongs to an
   explicit fitted-state identity.
4. A failed first fit leaves an unfitted model. A failed refit preserves the previous successful
   fit behaviorally and structurally.
5. Enabling diagnostics cannot add solver evaluations or change fitted results.
6. Safety must not be implemented by copying an existing row-scale fitted model.
7. Routine wall time may not regress beyond benchmark noise, and peak memory may not increase.
8. All temporary compatibility mirrors are projections of one authoritative FitState and may
   only be installed through the commit function.

## Components

### ModelConfig

ModelConfig records constructor and explicit post-construction intent:

- family and link specifications, including theta="auto";
- a model-owned Penalty template, including lambda1=None;
- model-owned feature templates and feature ordering;
- immutable pending interaction specifications;
- solver, discretization, convergence, retention, and penalty settings.

Constructor inputs are defensively copied. Non-empty interaction lists become tuples. Feature
dictionaries, FeatureSpec instances, family objects, link objects, and Penalty instances cannot
remain aliased to caller-owned mutable objects.

Configuration sentinels are permanent. A successful fit does not replace automatic intent with a
resolved value.

Supported public configuration assignments are properties backed by ModelConfig and create a new
configuration revision. Mutating a nested fitted Penalty or FeatureSpec object in place is not a
supported way to reconfigure a model; explicit setters replace the relevant model-owned template.

### FitWorkspace

FitWorkspace is private, mutable, and local to one attempt. It contains:

- cloned feature and interaction templates;
- resolved distribution and link;
- normalized response, weights, and offset;
- resolved fit-specific penalty values and profiling results;
- design matrix and groups;
- solver outputs and temporary factorization state;
- finalization and canonicalization intermediates;
- a trace sink and monotonically increasing state identifiers.

Existing build and solver functions may initially operate on this model-like workspace to reduce
refactor risk. They must not receive the public model when they can mutate it. The workspace is
discarded on failure.

The workspace borrows input data read-only during computation. It does not deep-copy the full
DataFrame or row-scale design merely to provide rollback. Only small configuration objects are
copied eagerly.

A successful FitState does not retain caller objects as identity-based cache keys. It takes
ownership of the normalized arrays and compiled design already produced by the attempt. APIs that
need original feature values use a single minimal model-owned columnar snapshot or require
explicit evaluation data; they do not keep both a raw full-frame copy and an equivalent duplicate
row-scale representation.

### SolverState

SolverState is an immutable evaluated coefficient state. At minimum it owns:

- state_id, coordinate-space name, and basis_id;
- beta, intercept, eta, mu, and deviance;
- resolved lambda vector and dispersion where applicable;
- convergence measures and the reason for termination;
- factorization/rank metadata needed by downstream numerical code;
- the trace evaluation identifier that created it.

Working quantities such as W, z, and Hessian diagnostics either belong to the base state from
which they were computed or to a separately identified evaluation. They cannot be attached to a
retained post-step row without their source-state identity.

### FitState

FitState is the authoritative complete successful artifact. It contains:

- a monotonically increasing revision;
- fitted feature and interaction specifications;
- resolved distribution, link, selection penalty, smoothing parameters, and profile results;
- groups and, when retained, the fit-time design and row arrays;
- authoritative private solver result and public canonical result;
- REML result, penalties, lambdas, optimizer profile, and convergence decomposition;
- fit statistics, fitted/null means, prediction plan, and runtime canonical mapping;
- compact inference state or retained row-scale inference state;
- shape-repair revision and repair metadata;
- the terminal trace state and run identifiers.

FitState is a frozen value object. Owned NumPy arrays are read-only after construction, and mutable
cache storage is revision-keyed outside the value object.

Public properties and reports read learned values from FitState. Configuration remains available
separately. Fitted attributes follow the usual trailing-underscore convention where a public name
is needed, including selection_penalty_, distribution_, and theta_.

During migration, legacy private attributes such as _result, _solver_result, _dm, and
_reml_result may remain as compatibility projections. _install_fit_state() must install all of
them from one artifact in a no-fail section. No other code may mutate a projected fitted
attribute directly.

### FitTrace

FitTrace is one globally ordered append-only event stream. JSONL remains an appropriate storage
format, but separate suffix files must share a run-wide sequence and state identities.

Every event has:

- schema_version, run_id, sequence, timestamp, and event_kind;
- solver, phase, purpose, and authoritative flag;
- outer and inner iteration where applicable;
- state_space and basis_id;
- base_state_id, proposal_state_id, and committed_state_id as applicable;
- resolved lambdas and dispersion for the identified state;
- convergence criterion, value, tolerance, and reason;
- only the numerical payload actually evaluated for that state.

Supported event kinds are:

- evaluation: an actual objective/deviance/derivative evaluation, never an implicit commit;
- step_decision: acceptance, half-step, or rejection linked to evaluated state IDs;
- state_commit: an actually evaluated retained state;
- terminal: the authoritative private solver state and, after canonicalization, public state;
- run_failed: the last committed state and exception class for an unsuccessful attempt.

An optional diagnostic replay must be marked purpose="diagnostic_replay" and
authoritative=false. This design removes the existing non-SCOP replay because diagnostics must
not add evaluations.

## Fit lifecycle

### 1. Preflight

Preflight resolves call controls and validates all non-data configuration without modifying the
model:

- fit mode and interaction mode;
- convergence and iteration controls;
- selection/shape-constraint compatibility;
- lambda policies;
- required columns and feature declarations;
- input dimensionality and common row count.

The trace run begins with metadata that identifies the previous successful revision, if any.

### 2. Input validation

Input validation runs before feature learning or design construction.

Common requirements:

- X is a non-empty DataFrame with unique required columns.
- y is one-dimensional, non-empty, row-aligned, numeric, and finite.
- weights are one-dimensional, row-aligned, numeric, finite, and not all zero.
- offsets are one-dimensional, row-aligned, numeric, and finite.
- ordinary non-Tweedie case weights may be zero when the current likelihood semantics permit
  zero influence; negative weights are rejected.
- Tweedie EDM prior weights remain strictly positive.

Response domains:

- Binomial: exactly 0 or 1.
- Poisson and negative binomial: nonnegative finite values. Integrality is not required because
  weighted actuarial frequency responses are supported.
- Gamma: strictly positive finite values.
- Tweedie with 1 < p < 2: nonnegative finite values.
- Gaussian: any finite values.
- custom distributions may implement validate_response; absence means common finite checks only.

fit(), fit_path(), and fit_reml() call the same validator.

### 3. Local preparation

The attempt clones configuration templates, resolves auto-detection and interactions locally,
estimates automatic theta locally, builds the design locally, and resolves automatic lambda
locally. Caller objects and the public model remain unchanged.

Every successful fit starts from configuration intent. Automatic theta, lambda, most-exposed
categorical bases, and auto-detected schemas are recomputed for the new data. No implicit
coefficient or optimizer warm start is permitted.

### 4. Solve

Solvers return immutable SolverState values and emit events at the computation site. The
workspace may retain mutable scratch buffers, but no result becomes public during optimization.

A finite best iterate that reaches max_iter may continue to finalization with converged=false. A
nonfinite coefficient, prediction, criterion, or structurally inconsistent result aborts the
attempt.

### 5. Local finalization

Before commit, the workspace completes:

- final REML refit, if mathematically required;
- dispersion and covariance finalization;
- fit and null statistics;
- runtime canonicalization;
- prediction-plan compilation;
- inference distillation;
- optional release of row-scale state;
- trace terminal consistency checks.

Canonicalization operates on workspace-owned fitted feature copies. A failure cannot alter the
previous model.

### 6. Consistency validation

The candidate FitState must pass:

- dimensional agreement among beta, groups, design, penalties, and covariance;
- finite predictions and family-domain means;
- solver/public prediction parity under the declared coordinate transform;
- result, REML result, dispersion, lambdas, and terminal trace identity agreement;
- cache revision agreement;
- compact-state completeness when retain_fit_state=false.

### 7. Commit

_install_fit_state() assigns the new authoritative state and compatibility projections. It must
allocate nothing, call no numerical routines, perform no I/O, and contain no expected failure
point.

If any earlier stage raises, the previous FitState, predictions, profiles, repairs, caches,
retention mode, and reports remain unchanged. A failed first fit remains unfitted.

## Convergence and trace semantics

Fit convergence is decomposed rather than overloaded:

- inner_converged: the final IRLS/PIRLS solve satisfied its declared criterion;
- outer_converged: the REML optimizer satisfied a criterion on the exact accepted lambda state;
- final_refit_converged: the authoritative final coefficient fit converged;
- overall_converged: the conjunction of every required phase.

A state cannot be called converged after changing lambdas without evaluating the convergence
criterion at the changed state. Analytical trial objects cannot set fit_converged=true merely
because they were constructible.

Step invariants:

- every state_commit references an actual evaluation;
- rejection commits the base state, accepts alpha zero, records all trials attempted, and cannot
  establish convergence;
- a half-step commit references the exact evaluated fixed-endpoint alpha;
- trials_attempted and accepted-step halvings are distinct;
- QP passthrough labels the unconstrained REML state and constrained final state separately;
- SCOP proposal events link to the enclosing IRLS decision and are authoritative only if that
  enclosing state is committed;
- terminal private state matches _solver_result in coefficients, intercept, deviance, lambdas,
  dispersion, state space, and basis;
- terminal public state matches result after canonicalization;
- REMLResult objective, coefficients, lambdas, and dispersion come from one terminal state;
- summaries read the terminal event, never the last row of a phase-specific file.

The public iteration_log contains committed coefficient states only. Proposal and rejected-trial
details live in FitTrace. Coordinate space is explicit.

Tracing level controls payload only. With tracing disabled, event construction uses a null sink.
No trace level may add solver calls, replay fits, objective evaluations, or different branching.

## Shape-repair transaction

Post-fit repair creates a derived workspace from the current FitState and commits a new revision.
It never mutates result.beta in place.

The repair path:

1. computes repaired coefficients locally;
2. recomputes public predictions, fitted means, fit statistics, canonical state, and prediction
   plan;
3. synchronizes the private/public/REML relationship explicitly;
4. transforms covariance and EDF only when a valid repair Jacobian is available;
5. otherwise marks affected inference unavailable with a typed reason;
6. invalidates all revision-keyed caches;
7. commits new repair metadata.

Every ordinary successful fit starts with an empty repair set.

## Performance and memory contract

Correctness work must not create a general slowdown.

Implementation constraints:

- do not deep-copy an existing FitState or row-scale design for rollback;
- copy only small configuration objects during preparation;
- borrow X, y, weights, and offset read-only during solving;
- avoid duplicate normalized response/weight/offset arrays;
- transfer ownership of newly built design buffers into FitState rather than copying them;
- distill retain_fit_state=false candidates before commit and release temporaries promptly;
- stream detailed traces to the sink instead of retaining coefficient vectors per event;
- retain only compact committed diagnostics in result.iteration_log;
- use integer state/revision IDs rather than coefficient hashes in the hot path;
- a null trace sink must be branch-predictable and allocation-free per iteration.

Benchmark protocol:

- freeze routine dense, categorical, sparse, spline, discrete, constrained, and REML fixtures
  before implementation;
- use warmups, counterbalanced execution order, at least five timed repeats, medians, and a
  controlled thread count;
- record total fit wall time, preparation, design, solver, finalization, trace, and
  canonicalization time;
- record peak RSS or tracemalloc peak where representative;
- compare trace level zero and level two solver-call counts exactly.

Acceptance gates:

- no routine median wall-time regression greater than 3% when the counterbalanced confidence
  interval excludes zero; otherwise classify differences within 5% as benchmark noise;
- no peak-memory regression greater than 2% on large retained and compact fits;
- trace-disabled overhead no greater than 1% in focused solver microbenchmarks;
- detailed tracing may incur I/O cost but may not change solver evaluation counts or results;
- any statistically credible regression requires redesign or explicit user approval;
- wall-time or memory reductions are desirable and are reported even when not required by this
  first unit.

## API and compatibility

Correct state semantics take precedence over preserving history-dependent behavior.

- Constructor configuration remains stable after fit.
- Learned values are exposed through fitted attributes and existing summaries.
- model.features no longer exposes caller-owned objects. A fitted-feature view is read-only or a
  defensive copy.
- Direct mutation of fitted private attributes is unsupported and replaced internally with
  revisioned transformations.
- Pickle migration accepts legacy models by constructing one FitState from their coherent
  retained attributes. Incoherent legacy hybrids fail with a descriptive migration error rather
  than silently loading.
- Numerical outputs should remain unchanged in this unit except where old behavior depended on
  stale or history-dependent state.

## Test design

### Ownership and repeated-fit tests

- caller FeatureSpec, feature dict, Penalty, family, link, and interactions remain unchanged
  after successful and failed fits;
- two models built from the same caller objects do not share mutable state;
- sequential fit A then fit B equals a fresh model fit on B for automatic lambda, automatic
  theta, most-exposed base, auto-detection, splines, and interactions;
- explicit configuration remains explicit across refits;
- caller mutation after construction does not alter model configuration or predictions.

### Strong exception tests

Inject failures at:

- common validation;
- family-specific validation;
- theta profiling;
- feature build and interaction resolution;
- solver bootstrap and main solve;
- REML line search and final refit;
- statistics;
- canonicalization;
- inference distillation;
- compact-state release preparation;
- trace finalization.

For a failed first fit, result remains unavailable and no fitted revision exists. For a failed
refit, compare the entire prior public behavior, fitted revision, design retention, profiles,
REML metadata, repairs, caches, reports, and predictions.

### Validation matrix

Parameterize fit(), fit_path(), and fit_reml() across:

- empty, multidimensional, mismatched, NaN, and infinite response;
- invalid family domains;
- negative, nonfinite, all-zero, and mismatched weights;
- nonfinite and mismatched offsets;
- missing and duplicate feature columns;
- custom-distribution validation hooks.

Assert validation occurs before any FeatureSpec build method.

### Trace lineage tests

- force full acceptance, half-step acceptance, and total rejection;
- verify state-ID lineage, alpha, trials attempted, and committed identity;
- compare trace-level-zero and trace-level-two solver call counts and fitted outputs;
- parameterize exact, discrete, EFS, SCOP, fixed-lambda monotone, and QP passthrough paths;
- independently recompute terminal eta, mu, deviance, and current production objective;
- assert terminal equality with solver result, public result, REML result, and telemetry;
- force exact and discrete line-search fallbacks and ensure convergence applies to the terminal
  lambda state;
- ensure SCOP proposals rejected by enclosing IRLS are non-authoritative;
- ensure summaries derive only from terminal events;
- ensure failed runs contain run_failed and cannot replace model state.

### Shape-repair tests

- shape repair creates a new revision without changing the old FitState;
- predictions and statistics match the repaired coefficients;
- covariance/EDF are correctly transformed or explicitly unavailable;
- all metrics, summaries, and prediction caches use the repaired revision;
- a later ordinary fit clears repair metadata;
- failed repair preserves the original fitted revision.

### Performance tests

- benchmark the frozen fixture matrix before and after the refactor;
- assert no extra row-scale copies with allocation spies;
- assert compact fits release row-scale data before commit;
- measure null-sink trace overhead;
- assert enabling diagnostics never changes numerical or call-count results.

## Acceptance criteria

This design is complete when:

1. all fit entry points use common preflight and input validation;
2. no fit attempt mutates the public model before _install_fit_state();
3. automatic configuration is re-resolved per fit;
4. caller-owned objects are not mutated or aliased;
5. failed first fits and refits satisfy the strong exception guarantee;
6. shape repair produces a coherent revision;
7. traces contain only actual evaluations with complete lineage;
8. terminal trace, solver, public, REML, and telemetry states agree;
9. diagnostics add no evaluations;
10. the focused and full regression suites pass on Python 3.10 through 3.14;
11. ordinary CPython 3.14 passes from the built wheel;
12. the wall-time and memory gates pass.

## Follow-on boundaries

After this unit is implemented and verified, the remaining remediation proceeds in separate
specification cycles:

1. likelihood score/observed-curvature and penalized globalization;
2. Wood-correct REML identifiable space, augmented Hessian, M_p, scale, and derivatives;
3. direct/discrete/EFS/SCOP integration on the shared criterion state;
4. Tabmat, derivative-memory, canonicalization, and duplicate-work performance remediation;
5. dependency metadata, wheel validation, and Python 3.14 support declaration.
