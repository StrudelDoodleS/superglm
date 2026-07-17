# Tweedie Estimation Correctness and Reliability Design

**Status:** Approved design, amended for clean-room implementation
**Date:** 2026-07-17
**Branch:** `codex/tweedie-correctness`

## Purpose

Make Tweedie estimation correct, numerically stable, honest about its inferential status,
transactional with respect to model state, and reliable throughout the supported open power range
`1 < p < 2`. The work closes every defect found in the post-merge audit while preserving valid
public behavior where that does not conflict with correctness.

The implementation will be independently derived from published Tweedie and exponential-
dispersion-model mathematics. External implementations may be used only as black-box numerical
oracles for input/output comparison. No external implementation source may be inspected, copied,
translated, or used to shape the internal code structure.

## Scope

This design covers:

1. exact positive-density and zero-mass evaluation;
2. stable unit deviance and Pearson dispersion;
3. fixed-power fitting and power/dispersion profiling;
4. candidate certification, search boundaries, and inference policy;
5. atomic final refitting and model-state installation;
6. faithful model and editor cloning;
7. training-state retention and serialization;
8. profile reporting and plotting;
9. public input validation and error quality; and
10. independent numerical-reference, adversarial, regression, and full-suite validation.

This work does not add new Tweedie power regimes outside `1 < p < 2`, introduce an R runtime
dependency, add compiled extension code, or redesign unrelated model families.

## Clean-room boundary

The project will enforce the following provenance rules:

- Implementation decisions come from published formulas, independently derived identities, and
  the existing superglm public contract.
- A reference-oracle process may evaluate selected numeric inputs using an external executable.
  It may record only inputs, numeric outputs, convergence status, and version-neutral tolerance
  metadata.
- Committed reference fixtures use neutral names such as
  `tests/fixtures/tweedie_reference_values.json`. They contain no external source fragments,
  identifiers, comments, symbol names, or implementation descriptions.
- Implementation workers consume the mathematical specification and numeric fixtures, not the
  external implementation source.
- Review includes a provenance scan for copied names, comments, control-flow descriptions, or
  distinctive source structure. Any questionable material is removed and independently rederived.
- New branch, worktree, commit, document, test, fixture, class, function, and reporting titles use
  neutral Tweedie terminology.

## Correctness invariants

The repaired system must maintain these invariants:

1. A result described as an MLE uses an exact, finite log likelihood and converged nuisance fits.
2. An approximate density evaluation can never silently compete for or replace an exact MLE.
3. A standard likelihood-ratio interval is available only for an exact, unpenalized, regular,
   interior profile.
4. `Tweedie.deviance_unit(y, mu)` is nonnegative for valid finite inputs, is finite whenever the
   mathematical result is representable in `float64`, and is exactly zero when `y == mu`.
5. A public profiling failure leaves the caller model observationally unchanged.
6. A successful installed profile agrees across `model.family`, the resolved distribution, public
   fit result, solver result, fit statistics, covariance state, predictions, reporting, and cached
   profile metadata.
7. `retain_fit_state=False` leaves no row-scale response, weight, offset, design-matrix, data-frame,
   scratch-model, callback, or evaluator closure reachable from the fitted model or result.
8. Search, CI, and plot ranges derive from the effective configured support and always include the
   reported estimate.
9. Finalized results and models without inherently unserializable user objects round-trip through
   pickle without retaining callbacks or hidden training data.
10. Invalid scalar, shape, finiteness, or domain inputs fail early with stable public exceptions.

## Architecture

### 1. Independent numerical kernel

Two private modules will separate numerical mathematics from profile orchestration:

- `src/superglm/_tweedie_numerics.py` owns scalar validation, stable unit deviance, zero-mass,
  compound-Poisson parameter identities, and Pearson-dispersion primitives.
- `src/superglm/_tweedie_density.py` owns exact positive log-density evaluation, optional analytic
  log-dispersion scores, density diagnostics, and an explicitly approximate diagnostic evaluator.

The exact evaluator will use the positive compound-Poisson/Gamma mixture series in log space. It
will locate the dominant term analytically or by a bounded local search, accumulate outward with
`logsumexp`, and certify both tails using monotone term-ratio bounds. The log-density and its
log-dispersion score will come from the same normalized series terms so the objective and score
cannot disagree because of a numerical branch switch.

An overlapping special-function route may remain as an acceleration only where both routes have
been independently cross-validated. Overflow or loss of certification falls back to the controlled
log-series route, never to an asymptotic approximation. If the exact series cannot certify the
requested tolerance within its safety limits, it raises a typed density-evaluation error.

The saddlepoint evaluator remains available only behind an explicit approximate mode. Its result
is permanently labelled approximate and is excluded from MLE selection and likelihood-ratio
inference.

`Tweedie.deviance_unit` will call the shared stable deviance primitive. Near `y / mu == 1`, the
implementation will use `log1p`/`expm1` or a local series to avoid cancellation. Zero responses use
their analytic limit. Negative round-off smaller than a documented floating-point tolerance is
clamped to zero; larger negative values raise an internal numerical error during development and
tests.

Pearson dispersion will evaluate the documented formula without replacing valid positive means by
`1e-10`. Computation will use scaled or log-domain operations when direct powers would overflow or
underflow.

### 2. Certified profile records

Every fixed-power evaluation will produce an immutable record containing:

- finite power, dispersion, and objective values;
- mean-fit and solver convergence;
- dispersion-profile convergence, score, and boundary state;
- exact/approximate density provenance and numerical certification;
- effective search bounds and endpoint identity;
- penalty/REML status; and
- a machine-readable rejection reason when the record is not selectable.

Selection uses a strict predicate. An exact MLE candidate must have a finite objective, positive
finite dispersion, converged mean fit, converged dispersion fit, and certified exact density. The
best record is chosen only from candidates satisfying the predicate. Rejected records remain in the
trace for diagnosis but cannot determine `p_hat`.

If no certified candidate exists, the low-level profiler raises `TweedieProfileError`. The
exception carries the immutable search trace and machine-readable rejected-record summary. The
public `SuperGLM.estimate_p()` path propagates that failure and does not refit or mutate the caller.

The profile result will expose an explicit inference kind:

- `exact_mle` for regular unpenalized likelihood profiling;
- `constrained_profile` when shape or coefficient constraints make ordinary regular likelihood
  asymptotics unavailable;
- `pearson_plugin` when dispersion uses the Pearson estimate;
- `penalized_plugin` when nuisance means use a selection or ridge penalty;
- `reml_plugin` for the REML-based nuisance path; and
- `approximate` only for explicitly requested diagnostic evaluation.

Only `exact_mle` may use MLE wording or ordinary likelihood-ratio confidence intervals. Plug-in
results may still be useful point estimates, but reporting and plotting must call them plug-in
profiles and explain that bootstrap or separately calibrated inference is required. Approximate and
failed results receive no inferential interval.

### 3. Search and boundary behavior

All search methods will explicitly evaluate both configured endpoints in addition to their
interior probes. Endpoint classification uses a tolerance derived from optimizer tolerance and
floating-point scale rather than exact equality alone.

The effective search bounds are stored on the result and become the single source of truth for:

- boundary diagnostics;
- CI bracketing;
- dense profile evaluation;
- trace/profile plot limits; and
- summary wording.

An interior optimizer result does not suppress a better certified endpoint. A winning endpoint is
reported as a boundary estimate even when the optimizer returns success. Dense evaluation grids
always contain `p_hat` exactly.

### 4. Transactional public profiling

The public operation has three phases:

1. **Prepare:** validate and own inputs, then build a faithful isolated profiling model.
2. **Certify:** profile power/dispersion and perform the final fit on isolated state. Check exactness,
   all convergence flags, fitted power, fitted dispersion, and synchronized fit statistics.
3. **Commit:** transfer the already validated final state to the caller in one guarded operation and
   install the immutable profile result last.

No caller field changes during prepare or certify. A callback exception, density failure, search
failure, nonconverged final fit, synchronization failure, or commit precondition failure leaves the
original family, resolved distribution, result, caches, predictions, and prior profile result
unchanged.

The progress callback receives copied, serialization-safe payloads. It is never retained by the
final model or profile result.

### 5. Faithful cloning and editor integration

`clone_without_features` will deep-copy complete surviving interaction specifications instead of
reducing them to parent-name pairs. It preserves custom interaction name, kind, knots, degree,
decomposition, constraint metadata, resolved/pending state, ordering, and model solver/retention
configuration. Interactions whose parents are removed remain excluded.

The editor and profiler will share this faithful cloning primitive. Tests compare design-matrix
group names, group widths, interaction configuration, and predictions before permitting editor
reprofiling to replace a model.

### 6. Retention and serialization

Profile results store immutable scalar metadata, tabular trace values, cached intervals, and
serializable diagnostics. They do not directly store bound methods or user callbacks.

When `retain_fit_state=True`, a private evaluator may remain reachable for lazy CI/profile probes,
but it is removed during pickling. After unpickling, only already cached intervals and curve values
remain available. When `retain_fit_state=False`, finalization detaches the evaluator and releases
every row-scale object. Callers that need an interval with released state may request an eager
interval during `estimate_p`; `ci()` then returns the cached interval. Without retained state or an
eager cache, `ci()` raises a precise error explaining how to request it.

Serialization tests inspect the complete reachable object graph and serialized-size growth, not
only named model cache fields. Callback-bearing estimation must serialize after callbacks are
discarded at finalization.

### 7. Validation and public error behavior

Shared validators will require:

- `p` to be a finite, non-boolean real scalar strictly inside `(1, 2)`;
- `phi` and residual degrees of freedom to be finite, non-boolean positive real scalars;
- response, mean, weight, and offset arrays to be one-dimensional, numeric, finite, owned where
  needed, and of exactly matching length;
- means and weights to be strictly positive where required;
- search grids to be finite one-dimensional numeric arrays inside support;
- search bounds to be ordered finite scalar pairs inside support; and
- optimizer counts/tolerances to have explicit integer/real domains.

The distribution constructor stores a built-in `float`, not a one-element array or object scalar.
Malformed offsets fail before solver construction. Public exceptions use `TypeError` for wrong
types and `ValueError` for invalid values/shapes; internal numerical-certification failure uses a
dedicated runtime exception.

### 8. Reporting and plotting

Summary output includes profile convergence, density exactness, dispersion convergence, boundary
status, and inference kind. It must never report final-refit convergence as if it proved profile
convergence.

`trace_plot()` remains cache-only and side-effect free. It visually distinguishes selectable exact
records from rejected, approximate, or nonconverged records and uses neutral objective-difference
wording unless the result is a certified `exact_mle`. The winning marker includes boundary or
failure state where applicable.

`profile_plot()` uses the stored effective bounds, includes `p_hat`, and refuses inferential
overlays for profiles that do not support ordinary likelihood-ratio inference. Both plotting methods
must work with caller-supplied axes without mutating evaluation counts or caches.

## Compatibility policy

Correct existing behavior is preserved:

- prior weights continue to represent exponential-dispersion weights;
- offsets/exposures and solver controls are forwarded unchanged;
- generator RNG and vector-dispersion semantics remain stable;
- exact regular profiles retain current result fields where their meaning remains correct;
- legacy pickle fields are migrated where unambiguous; and
- successful ordinary fits continue to clear stale profile results.

Compatibility does not preserve behavior that can install an invalid estimate, call a penalized
plug-in result an MLE, compute an unjustified interval, retain hidden training data against the
documented contract, or return mathematically impossible deviance/dispersion values.

## Test strategy

### Mathematical unit tests

- Compound-Poisson parameter identities and zero mass.
- Positive log-series terms, mode location, two-sided tail certification, and score identity.
- Exact log-density against hand-computable finite/truncated cases and high-precision independent
  calculations.
- Stable deviance at `y == mu`, near equality, zero response, extreme scales, and powers close to
  both open boundaries.
- Pearson dispersion with means below `1e-10`, extreme powers, and invalid residual DF.

### Neutral reference tests

`tests/fixtures/tweedie_reference_values.json` will cover:

- powers whose distance from either boundary ranges from `1e-6` to ordinary interior values;
- zero and positive responses spanning multiple orders of magnitude;
- small, ordinary, and large means and dispersions;
- scalar and prior-weighted observations; and
- deterministic end-to-end generated datasets.

The fixture test target is maximum absolute log-density error `<= 1e-8` where the reference is
finite. Deterministic profile estimates must agree with the independent reference to absolute
power error `<= 2e-4` and relative dispersion error `<= 5e-4`. Tighter tolerances are used where
conditioning permits them.

An optional developer-only black-box differential harness regenerates and compares fixtures. It is
not a runtime or package dependency and does not expose or consume external source code.

### Audit regressions

Tests will reproduce and close every audited defect:

- the seed-101 low-power false-boundary winner;
- a single-observation saddlepoint/exact discrepancy;
- negative and nonzero equal-value deviance;
- penalized/REML inference mislabelling and invalid CI access;
- public installation of a nonconverged profile;
- callback and final-refit failure atomicity;
- custom decomposed interaction loss in editor reprofiling;
- row-scale retention with `retain_fit_state=False`;
- endpoint omission in transformed optimization;
- hard-coded CI and plot ranges near both power boundaries;
- invalid `df_resid` and tiny-mean Pearson dispersion;
- unsupported power scalar representations;
- malformed offsets and search controls; and
- callback-bearing pickle round trips.

Every repair follows red/green testing: demonstrate the regression test fails against the original
implementation, make the smallest coherent correction, and then run the focused neighboring suite.

### Completion gates

Completion requires all of the following evidence from the final branch state:

1. neutral reference fixture suite passes at its stated tolerances;
2. live black-box differential validation passes when the oracle environment is available;
3. all audit regression tests pass;
4. focused Tweedie generator, density, profile, CI, weight, offset, editor, reporting, plotting,
   retention, and serialization suites pass;
5. the full repository pytest suite passes with no new skips or expected failures;
6. full Ruff check and format check pass;
7. touched modules pass focused static typing, with no increase in repository baseline typing errors;
8. worktree diff check and provenance scan are clean; and
9. a requirement-by-requirement completion audit maps every invariant and audit defect to direct
   test or runtime evidence.

## Implementation decomposition

The work will be executed as four dependent, reviewable plans:

1. **Numerical kernel and reference parity:** private numerical modules, density/deviance/dispersion
   tests, and neutral reference fixtures.
2. **Certified profiling and inference:** candidate validity, search endpoints, inference kinds,
   CI restrictions, and honest reporting.
3. **Transactional state and faithful cloning:** isolated final fit, atomic commit, rollback tests,
   interaction preservation, and editor integration.
4. **Retention, serialization, plotting, validation, and final audit:** evaluator detachment, eager CI
   caching, pickle behavior, range-aware plots, public validators, documentation, and complete
   verification.

Each plan must leave the repository in a tested, reviewable state. Later plans may depend on public
or private interfaces established by earlier plans, but no plan may weaken an invariant to make a
later phase easier.

## Decision record

- Use a correctness-layer refactor rather than scattered patches in the existing profiler.
- Keep the implementation pure Python/SciPy and avoid new runtime or compiled dependencies.
- Make exact density authoritative; retain approximation only as an explicit diagnostic.
- Separate MLE, penalized plug-in, REML plug-in, and approximate result semantics.
- Stage and certify final state before mutating the caller.
- Prefer honest compatibility breaks over preserving silently incorrect statistical behavior.
- Use neutral artifact naming and a documented clean-room provenance boundary.
