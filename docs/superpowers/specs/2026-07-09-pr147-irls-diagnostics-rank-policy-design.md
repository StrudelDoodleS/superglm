# PR 147 IRLS Diagnostics and Rank Policy Design

## Goal

Address the three actionable Codex review findings on PR 147 without restoring the removed
relative working-weight floor or broadening the pull request into a general solver rewrite.
The resulting diagnostics must distinguish the state that generated an IRLS update from the
state produced by that update, and QR fitting must preserve weak but identifiable sparse-tail
directions without retaining numerically null collinear contrasts.

## Confirmed problems

1. Legitimate zero-frequency observations currently force the reported working-weight ratio
   to approximately `1e300`, producing a false extreme-ratio warning even though those rows do
   not participate in the weighted least-squares system.
2. Direct-IRLS diagnostic rows combine pre-solve eta, mu, and weights with the post-solve
   intercept and deviance. The fields are individually accurate but do not describe one solver
   state, and their names do not make the phase boundary explicit.
3. The QR path's machine-epsilon cutoff preserves near-collinear contrasts that downstream
   Gram-based rank and covariance handling treats as null. This can yield enormous cancelling
   coefficients while effective degrees of freedom report a lower rank.

## Diagnostic state model

An IRLS iteration has two relevant states:

- The **working state** is the parameter state at the beginning of iteration `k`. Its eta and mu
  produce the working weights `W` and working response `z` used by the solve.
- The **updated state** is the parameter state after the solve and any step-halving. Its eta,
  mu, intercept, and deviance describe the model passed to the next iteration or returned to
  the caller.

The public diagnostic row will expose both phases explicitly:

- `working_eta_min`, `working_eta_max`, `working_eta_min_unclipped`,
  `working_eta_max_unclipped`, `working_eta_clipped`, `working_mu_min`, and
  `working_mu_max` describe the state that generated `W`.
- `eta_min`, `eta_max`, `eta_min_unclipped`, `eta_max_unclipped`, `eta_clipped`, `mu_min`,
  `mu_max`, `intercept`, and `deviance` describe the updated state.
- Weight, condition, fallback, and step-halving fields retain their natural iteration meaning:
  weights are solve inputs; condition/fallback describe the solve; step-halving describes the
  transition to the updated state.

A standalone `working_intercept` will not be added. It is parameterization-dependent and does
not reconstruct the working state without the entire coefficient vector, design matrix, and
offset. Explicit eta/mu phase names provide the needed provenance without presenting the
working intercept as a statistically meaningful quantity.

The newly added PR fields have not shipped, so their semantics can be corrected before merge.
Direct IRLS's established unprefixed eta/mu fields retain their historical post-update meaning.
PIRLS currently records those fields from the working state; it will adopt the same explicit
two-phase schema so diagnostic meaning is consistent across solver paths. The model-level
diagnostic DataFrame and debug JSONL output will use the same phase terminology.

## Working-weight ratio semantics

The ratio used for warnings and `W_ratio`/`raw_W_ratio` will be calculated over strictly
positive working weights. Raw minimum and maximum fields will continue to report the actual
array extrema, including a raw minimum of zero. Consequently, a zero-frequency row is visible
but does not masquerade as quasi-separation. A positive infinite weight still produces an
infinite ratio, while a NaN remains visible in the raw extrema and existing non-finite-fit
handling.

The same helper and semantics will be used by direct IRLS, PIRLS, and statsmodels comparison
diagnostics. If no positive finite working weights exist, the ratio is infinite; fitting is
already invalid in that state and the diagnostic should remain conspicuous.

## QR rank policy

The augmented weighted design used by QR will be column-equilibrated before rank detection.
Rank will then be determined in the normalized coefficient geometry using a singular-value
cutoff derived from the downstream normal-equation relative cutoff. Because singular values
are squared by normal equations, the QR cutoff is the square root of the shared Gram cutoff.
Coefficients will be transformed back to the original parameterization after solving.

Column equilibration distinguishes the two cases that a raw global singular-value cutoff
confounds:

- A sparse-tail indicator may have a tiny norm because its positive working weights are tiny,
  while remaining geometrically independent of other columns. Normalization preserves it.
- Nearly duplicate columns remain nearly duplicate after normalization. Their contrast is
  truncated consistently with downstream rank and covariance handling.

Zero-norm augmented columns will receive a zero coefficient and be treated as rank deficient.
The solver will continue reporting whether any singular direction was truncated.

This is intentionally narrower than replacing Gram, QR, REML, and covariance decomposition
with a new shared factorization. Such a refactor can be considered separately after PR 147.

## Statsmodels interpretation

For an unpenalized Tweedie GLM with fixed variance power, matched design, offset, and prior or
frequency weights, SuperGLM and statsmodels use the same coefficient estimating equations.
Dispersion is not part of either IRLS coefficient update. Controlled experiments reached a
maximum coefficient difference below `5e-15` when parameter convergence rules were matched,
despite materially different reported dispersion under fractional frequency weights.

Ordinary small coefficient differences therefore come primarily from stopping rules:
SuperGLM defaults to relative deviance convergence at `1e-6`, while statsmodels defaults to an
absolute deviance criterion at `1e-8`. Initialization affects early iterations but disappears
at a shared finite optimum. Design coding, reference levels, offset, row alignment, and weight
interpretation can produce material differences and must be matched before solver comparison.

All-zero or completely separated positive-family levels have no finite coefficient MLE. Their
reported finite coefficients depend on stopping rules, eta/mu safeguards, and rank handling;
the meaningful parity targets are finite coefficients, fitted means/deviance, and clear
separation diagnostics rather than equality of the diverging tail coefficient.

## Tests

Implementation will be test-driven and add focused regressions for:

1. Direct IRLS with a legitimate zero sample weight: raw minimum remains zero, the positive
   weight ratio is finite, and no false extreme-ratio warning is logged.
2. PIRLS with the same zero-weight condition, proving path parity.
3. A one-iteration direct fit showing that `working_*` eta/mu fields describe the pre-solve
   state while unprefixed eta/mu/intercept fields describe the updated state.
4. A nearly collinear Gaussian design showing bounded QR coefficients, fitted-value/deviance
   parity with the rank-reduced solution, and effective-rank consistency.
5. The existing sparse-tail Tweedie QR regression, proving column equilibration does not delete
   the weak but independent rare-level direction.
6. A deterministic statsmodels comparison using matched parameter convergence and a common
   design, weights, and offset, proving finite coefficients agree to numerical precision.

Targeted solver, quasi-separation, diagnostics, statsmodels-consistency, lint, and type checks
will run before the full test suite.

## Non-goals

- Changing the removed relative working-weight floor.
- Making separated tail coefficients numerically equal to statsmodels.
- Changing Tweedie dispersion or residual-degrees-of-freedom conventions in this PR.
- Redesigning all Gram, REML, and covariance decompositions.
- Replying to or resolving GitHub review threads without separate user authorization.
