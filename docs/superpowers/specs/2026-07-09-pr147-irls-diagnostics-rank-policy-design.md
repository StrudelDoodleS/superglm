# PR 147 IRLS State, Shared Rank, and Benchmark Design

## Goal

Address every actionable Codex review finding on PR 147 and replace the interim
QR-only rank change with one coherent fitted-subspace policy across coefficient
solving, effective degrees of freedom, covariance, group inference, and REML
bookkeeping.

The implementation must not restore the removed working-weight floor. It must
expose quasi-separation rather than hiding it, make IRLS state transitions
atomic, and quantify every material change in numerical output, stability, fit
time, and memory. Performance thresholds are review flags rather than automatic
rejection criteria: measured regressions will be reported with their correctness
benefit, and the user will decide the final trade-off.

## Decision summary

The approved design has three parts:

1. Profile the intercept, equilibrate the coefficient system, and apply one
   rank policy throughout the fit and its downstream inference.
2. Evaluate IRLS and SCOP step-halving trials without mutating the committed
   state, then commit exactly one accepted state and all of its derived caches.
3. Compare the original PR head, the rejected interim rank policy, and the final
   implementation with a deterministic numerical and performance harness.

The interim rule

```text
QR rcond = sqrt(1e-10) = 1e-5
```

is rejected. It is an undocumented condition regularizer that can delete a
strongly supported contrast and materially worsen deviance.

## Confirmed problems

### Diagnostic-state problems

- Legitimate zero sample weights force the old ratio calculation to infinity
  even though those rows do not participate in the weighted least-squares
  system.
- Direct IRLS previously combined pre-solve eta and mu with a post-solve
  intercept and deviance.
- The current two-phase diagnostic recomputation exposed a pre-existing
  line-search bug: both direct IRLS and PIRLS can retain a rejected trial.
- Direct IRLS performs clipping min/max scans even when diagnostics and debug
  recording are disabled.

### Step-halving and SCOP problems

- Both halving loops mutate live coefficients before validating a trial. If an
  accepted half-step is followed by a rejected quarter-step, the rejected
  quarter-step is returned.
- A first rejected trial changes the returned coefficients while reporting zero
  accepted halvings.
- SCOP trials are evaluated by interpolating mapped coefficients, then committed
  by interpolating latent `beta_eff`. The exponential map means those are
  different models.
- `gamma_eff` and `H_scop_penalized` can describe a different state from the
  returned coefficients. EFS preferentially consumes the stale gamma cache.

A controlled sequence with committed/full/half/quarter deviances
`2 / 10 / 5 / 6` returned the rejected quarter state at deviance 6 while
reporting one accepted halving. A real Tweedie SCOP fit reproduced a maximum
gamma-cache mismatch of approximately `8.08` and a retained-state Hessian
mismatch of approximately 7%.

### Rank and inference problems

- Fit-time QR, fit-time Gram, final EDF, dense covariance, Gram covariance, and
  group EDF use different cutoffs and coordinate systems.
- Ordinary and pivoted Cholesky may accept a direction without applying the
  nominal Gram cutoff. A small residual proves solve accuracy for the supplied
  right-hand side; it does not prove statistical identifiability.
- Raw Gram rank decisions depend on feature units. In a deterministic example,
  rescaling one independent column changed the reported rank.
- QR currently detects rank jointly with the intercept, while final EDF
  decomposes a coefficient-only matrix and unconditionally adds one.
- A constant numeric feature can be split differently between the intercept and
  feature by QR and Gram, while both report EDF 2 for an augmented rank of 1.
- Post-fit inference re-selects groups from `norm(beta[group]) > 1e-12`, which
  confuses a fitted zero with an unselected group.
- Truncated solve directions receive zero inverse during fitting but an
  artificial large finite inverse during covariance calculation.
- Rank metadata would currently be lost by explicit `PIRLSResult` copies and
  by a final REML refit that does not preserve the requested direct solver.

### Evidence against the interim cutoff

For an equilibrated Gaussian design `[1, x, x + delta*z]` with signal in the
weak contrast:

- at `delta=1e-6`, the design condition is about `2e6`;
- machine-scale QR recovers the contrast to approximately `2e-11`;
- `rcond=1e-5` drops the contrast;
- with response noise `1e-9`, the discarded contrast has a directional
  t-statistic of roughly `2.2e4`.

In an actual SuperGLM fit at this scale, Gram achieved deviance near
`2.6e-16`, while the interim QR rule dropped the signal and produced deviance
near `5e-10`. Both nevertheless reported EDF 2, demonstrating that coefficient
solving and inference described different fitted subspaces.

## Statistical and numerical principles

Rank, conditioning, and statistical uncertainty are related but distinct:

- A direction is data-null when the centered weighted design maps it to zero.
- A penalty can make an estimator unique without making that coefficient
  data-identified.
- A small singular value does not by itself imply an insignificant contrast.
  Significance also depends on response noise, dispersion, sample size, and the
  target contrast.
- Normal equations square the design condition:
  `cond(A.T @ A) = cond(A)**2`.
- Feature scaling must not decide rank.

Accordingly, the implementation will distinguish:

- data rank;
- penalty-augmented solver rank;
- pre-truncation condition;
- retained numerical subspace;
- penalty-supported versus data-supported directions;
- non-estimable individual coefficient functionals.

## Shared centered system

The intercept will be profiled before solving or deciding feature rank. For
working weights `W`, working response with offset removed `z_off`, and
feature matrix `X`, define:

```text
sum_w = 1.T @ W
xtw1 = X.T @ W
xtwz = X.T @ (W * z_off)
mean_x = xtw1 / sum_w
mean_z = W.T @ z_off / sum_w
G = X.T @ (W[:, None] * X) - outer(xtw1, xtw1) / sum_w
h = xtwz - xtw1 * mean_z
H = G + S
```

The feature solve is `H @ beta = h`, and the original-coordinate intercept is:

```text
intercept = mean_z - mean_x @ beta
```

This gives the intercept deterministic priority. An unpenalized constant
feature becomes a zero centered direction, receives coefficient zero, and
contributes zero feature EDF.

The same geometry can be represented by the centered augmented factor:

```text
A = [sqrt(W) * (X - mean_x); sqrt(S)]
```

QR works on `A`; Gram, Cholesky, and pivoted Cholesky work on `A.T @ A`.

## Rank policy

All rank decisions occur after column equilibration. The initial shared
stability rule is:

```text
factor_rcond = sqrt(machine_epsilon)
gram_rcond = machine_epsilon
```

These are equivalent on an exactly formed equilibrated normal equation. The
factor cutoff is deliberately much smaller than the rejected `1e-5` rule but
more conservative than statsmodels' machine-scale weighted least squares. It
sets a common resolution boundary that the Gram path can support.

This boundary is a numerical stability policy, not a test of statistical
significance. Pre-truncation condition and objective loss will be reported
separately. Comparisons with machine-scale QR and statsmodels will quantify the
directions sacrificed for cross-solver consistency.

The policy will also define:

- treatment of exact zero columns;
- the condition level that triggers a warning;
- the severe-condition level that triggers a stronger warning;
- whether a decomposition was resolution-limited by normal equations;
- versioned semantics so stored rank metadata remains interpretable.

Changing the cutoff after baseline capture requires benchmark evidence and a
design update, not an unreviewed constant edit.

## Decomposition ladder

Cholesky remains the ordinary fast path:

1. Equilibrate the centered system.
2. Attempt ordinary Cholesky.
3. Estimate reciprocal condition from the factor.
4. Accept the factor only when it is safely full rank under the shared policy
   and its solve-quality check passes.
5. Otherwise use pivoted Cholesky with the explicit shared tolerance.
6. If the rank-revealing Gram solve cannot produce a trustworthy retained
   subspace, use QR/SVD when feasible.

The fallback method and any resolution limitation are recorded. A successful
Cholesky call is no longer treated as proof that every coefficient direction is
identified.

Explicit QR uses the same equilibration and retained-subspace cutoff. This
deliberately gives up QR's last factor-resolvable directions in exchange for
consistent default behavior with Gram. The benchmark report will expose the
cost of that choice.

## Rank metadata and downstream reuse

An internal, versioned `RankInfo` object will be attached to the solver result.
It will contain a compact method-specific factorization plus enough information
to reproduce the fitted subspace:

- solver method and policy version;
- weighted feature means and scaling;
- selected groups and columns;
- data and penalty-augmented ranks;
- retained pivots or basis;
- aliases and non-estimable functionals;
- pre-truncation condition and cutoff;
- fallback and resolution-limited flags.

Full-rank Cholesky results need not store a dense singular-vector basis.
Rank-deficient results retain the reduced basis required by covariance and EDF.
The object exposes operations such as solve, pseudo-inverse, EDF, null-basis,
and estimability checks without downstream code choosing a new cutoff.

The metadata will survive:

- runtime canonicalization;
- known-scale and REML result correction;
- final REML refits;
- retained-fit-state release;
- serialization;
- legacy `PIRLSResult` construction through a default `None` value.

Result reconstruction should use a metadata-preserving helper or
`dataclasses.replace` rather than manually copying selected fields.

## EDF, covariance, and estimability

In centered feature coordinates:

```text
F = H_pinv @ G
feature_edf = diag(F)
total_edf = intercept_edf + trace(F)
```

The intercept EDF is one when positive total working weight identifies it.
Per-group EDF is the sum over the group's feature slices, so the implementation
must satisfy:

```text
total_edf == intercept_edf + sum(group_edf)
```

The augmented covariance is built in centered coordinates from the intercept
variance and the retained feature covariance, then transformed back through
`intercept = centered_intercept - mean_x @ beta`.

Discarded directions will not receive an invented large finite inverse.
Estimable contrasts retain covariance; non-estimable individual coefficients
are marked explicitly and their inferential statistics are suppressed. A
sparse-tail indicator that is geometrically independent remains retained after
equilibration and naturally receives a large variance from its tiny original
weighted norm.

Group selection must come from solver selection state, not coefficient
magnitude. Direct L2 fits select all modeled groups; L1/PIRLS fits preserve
their actual selected-group state.

## REML boundary

REML-specific determinant and penalty algebra will not be silently replaced by
the centered inference Hessian. The existing REML derivation uses
coefficient-space quantities that need separate validation.

REML will nevertheless use the same equilibration, cutoff policy, and retained
subspace operations for each matrix it decomposes. Its inverse,
pseudo-log-determinant, objective, and gradient must refer to the same REML
subspace. The final REML refit must preserve `direct_solve` and rank metadata.

Tests will separately verify:

- coefficient-fit and inference subspace consistency;
- REML objective/gradient finite-difference consistency;
- smoothing-parameter and EDF behavior under exact and near aliases.

## Atomic IRLS state transitions

Every direct-IRLS and PIRLS iteration will act as a transaction:

1. Preserve one immutable committed state.
2. Produce a full proposal without overwriting it.
3. Build trials from fixed endpoints with
   `alpha = 1, 1/2, 1/4, ...`.
4. Evaluate trial eta, mu, deviance, and derived constrained state without
   committing.
5. Commit exactly one accepted trial and all its caches.
6. If no trial is accepted, restore the committed state and record rejection.

The full proposal retains current behavior when no safeguard is triggered. If
a proposal is non-finite, has a family-specific invalid sign state, or exceeds
the existing catastrophic-deterioration threshold, the line search chooses the
largest trial fraction that clears the same centralized safety predicate. If no
trial clears it, the previous state is retained.

This preserves the intended safeguard rather than introducing an unreviewed
monotonic-deviance requirement. Any change in accepted steps, iterations, or
convergence will be measured.

`step_halvings` records the selected trial depth. A separate rejected-step
signal distinguishes a restored state from an accepted full step.

## SCOP state

SCOP trials interpolate latent `beta_eff`, not mapped `gamma_eff`. Each
trial then derives:

```text
gamma_eff = forward(beta_eff)
```

and inserts that exact mapped value into the full coefficient vector before
eta and deviance evaluation.

After acceptance, one pure state evaluator regenerates:

- `beta_eff`;
- `gamma_eff`;
- full-model beta slices;
- retained eta, mu, and deviance;
- actual retained step norm;
- `H_scop_penalized` at the retained state.

EFS and warm starts therefore receive the same model that was evaluated and
returned. Hessian recomputation is required at final export and after an outer
halving changes the retained state; unnecessary row-scale recomputation should
be avoided.

## Diagnostic semantics

An iteration exposes two phases:

- `working_*` fields describe the eta and mu that generated `W` and `z`;
- unprefixed eta, mu, intercept, and deviance describe the retained updated
  state.

A `working_intercept` will not be added. It is parameterization-dependent and
cannot reconstruct the working model without the complete coefficients, design,
and offset. Eta and mu are the invariant quantities relevant to working weights.

Working-weight warning ratios use strictly positive weights. Raw minima and
maxima still expose zeros. The ratio divides by the actual positive minimum
under overflow suppression; it does not use an absolute `1e-300` floor, so it
remains scale-invariant for subnormal weights.

Clipping and extrema reductions run only when iteration diagnostics or the
corresponding debug level are enabled. Fallback diagnostics distinguish
`rank_truncated` from `used_svd_fallback`.

## Statsmodels interpretation

For an unpenalized Tweedie GLM with fixed variance power, matched design,
offset, and observation weights, SuperGLM and statsmodels solve the same
coefficient estimating equations. A common scalar dispersion does not enter
the coefficient update.

Controlled experiments found:

- maximum finite coefficient difference below `5e-15` after matching
  parameter convergence;
- materially different reported dispersion under fractional weights while
  coefficients still matched;
- ordinary default coefficient differences driven primarily by stopping rules;
- coefficient shifts up to roughly `0.67` when offset was omitted;
- naive reference-level coefficient differences around `1.31`, despite
  matching reparameterized predictions;
- a separated all-zero category with no finite MLE, where finite tail
  coefficients depended strongly on stopping and clipping while
  well-identified coefficients matched.

Statsmodels 0.14.6 uses machine-scale weighted least squares and a final
pseudoinverse cutoff near `1e-15`. It can therefore retain directions that the
approved cross-solver `sqrt(epsilon)` stability policy drops. Near that
boundary, predictions, estimable contrasts, score norm, and objective loss are
the primary comparisons; equality of non-identifiable or deliberately
truncated coefficients is not expected.

Material causes of coefficient differences remain:

- convergence criterion and tolerance;
- design coding and reference levels;
- row alignment and missing-value handling;
- offset/exposure handling;
- weight interpretation;
- eta/mu safeguards in sparse or separated tails;
- penalties, smoothing-parameter estimation, Tweedie-power estimation, or
  NB2-theta estimation;
- explicit rank policy in ill-conditioned designs.

## Benchmark design

The deterministic harness compares:

1. original PR review head `29cca23`;
2. interim branch `4ad6673`, including the rejected `1e-5` policy;
3. final shared-rank implementation;
4. statsmodels 0.14.6 where estimating equations can be matched.

Scenarios include:

- well-conditioned Gaussian, Poisson, Gamma, and Tweedie fits;
- weighted Tweedie with offset;
- sparse all-zero categorical tails;
- legitimate zero sample weights;
- constant, zero, and duplicate columns;
- a near-collinear sweep with and without response signal;
- feature rescaling from `1e-12` through `1e12`;
- penalized splines, multi-penalty REML, and final REML refits;
- large discretized fits;
- SCOP fits that stop on a halved iteration;
- PIRLS and group-selection fits.

Every scenario records:

- coefficients and estimable contrasts;
- predictions, deviance, log likelihood, score norm, and dispersion;
- convergence, iteration count, halvings, rejected steps, and fallback method;
- data rank, augmented rank, aliases, total EDF, group EDF, and covariance
  rank;
- total wall time and existing solver-phase timings;
- peak process memory.

Timing uses fixed seeds, fixed BLAS thread counts, warm-up runs, and repeated
in-process fits. Reports include medians and spread rather than a single run.

## Soft performance review thresholds

The following thresholds trigger investigation and explicit reporting:

- more than 5% median fit-time regression on ordinary well-conditioned dense
  or discretized cases;
- more than 10% unexplained peak-memory growth;
- a new decomposition fallback on an ordinary case;
- a material increase in iteration count.

They are not automatic rejection gates. A regression may be accepted when the
stability or correctness gain justifies it, but the report must identify the
affected workload, absolute and relative cost, cause, and available alternative.
The user retains the final judgment.

Ill-conditioned fallback cases may be slower. Their report must pair the cost
with the corrected rank, objective, convergence, or inference behavior.

## Test matrix

Implementation is regression-first and includes:

1. Positive-weight ratio tests for direct IRLS, PIRLS, all-zero weights, and
   subnormal positive weights.
2. Two-phase diagnostic tests for direct IRLS and PIRLS.
3. Rejected-first-trial and accepted-then-rejected halving tests.
4. Non-finite proposal and exhausted-halving restoration tests.
5. Mixed SCOP/non-SCOP latent-space trial tests.
6. SCOP gamma, Hessian, warm-start, and returned-prediction invariants.
7. Constant, zero, duplicate, and rescaled-column rank tests.
8. Near-collinear tests on both sides of the shared cutoff, including signal in
   the weak direction.
9. Sparse-tail Tweedie preservation after equilibration.
10. Exact agreement among fitted rank, total EDF, group EDF, and covariance
    subspace.
11. Dense, Gram, QR, Cholesky, and pivoted-Cholesky parity where the shared
    policy says the problem is resolved.
12. Penalty-only identification and data-rank distinction.
13. REML objective/gradient, smoothing-parameter, and final-refit metadata
    tests.
14. Rank-metadata preservation through canonicalization, state release, and
    serialization.
15. Matched statsmodels convergence on an identifiable weighted-offset
    Tweedie dataset.

Targeted tests run after each behavioral change. Final verification includes
the full suite, Ruff, formatting, diff checks, baseline-aware type checking,
and the benchmark report.

## Delivery checkpoints

The work remains one design but will be delivered in independently verifiable
checkpoints:

1. Commit the benchmark harness and capture immutable baseline results before
   changing solver behavior.
2. Correct weight diagnostics, diagnostic-state cost, and atomic direct/PIRLS
   transitions.
3. Correct SCOP trial coordinates and retained-state caches.
4. Introduce the centered rank kernel and metadata without switching consumers.
5. Integrate Gram, Cholesky, pivoted Cholesky, and QR solves.
6. Switch EDF, covariance, group inference, and active-selection state to the
   retained subspace.
7. Integrate and validate REML decomposition and result propagation.
8. Run the full numerical, stability, memory, and timing comparison.

Each checkpoint receives focused regression tests before the next starts. If a
soft performance threshold is crossed, the implementation will not silently
weaken the approved rank semantics; the measured alternatives and trade-off
will be brought back to the user.

## Compatibility

- `PIRLSResult.rank_info` defaults to `None` so existing constructors and
  older serialized results remain readable.
- Public coefficient ordering and prediction APIs remain unchanged.
- Exact aliases may change coefficient representation and will now expose
  non-estimability rather than misleading finite inference.
- Identifiable ordinary fits should retain coefficient and prediction parity
  within numerical tolerance.
- The benchmark harness and final Markdown comparison report will be committed
  so the performance and stability conclusions are reproducible.

## Non-goals

- Restoring the removed relative working-weight floor.
- Treating a condition number as a statistical significance test.
- Forcing separated tail coefficients to equal statsmodels when no finite MLE
  exists.
- Changing Tweedie dispersion or residual-degrees-of-freedom conventions in
  this work.
- Hiding performance regressions because a correctness test passes.
- Replying to or resolving GitHub review threads without separate user
  authorization.
