# RFC-12b Disposition: Retired per Audit §J.6 — Retained-Factor Seam Kept

**Status: CLOSED (retired on measurement).** This document opened the
RFC-12b arc (cached-factorization REML line-search trials) and was rewritten
as a disposition note when review surfaced audit §J.6, which had already
retired the RFC on instrumentation. The retained-factor seam shipped with
this document stays: it is the prerequisite RFC-2 and RFC-7 list.

## Why RFC-12b is retired

Audit 2026-07-28 §J.6 ("RFC-12b retired on measurement", commit `3d41fe3`)
supersedes the §J.2/§J.4 entries that promoted the RFC — and the
2026-07-29 next-session brief's objective 3, which was written against
§J.2/§J.4 and missed §J.6:

- The premise "every Armijo trial runs a full PIRLS to convergence" is
  literally true, but instrumentation shows those trials ARE the one
  converged inner fit per outer iteration: b_L800-class 20 outer / 13
  trials / 13 accepts (7 iterations skip the search at bounds), a_m15
  11/10/10, two-tensor flagship 8/8 with 7 carry-forward reuses.
- **Every measured line search accepted its first trial; zero
  step-halvings anywhere.** §E row 12's "8 trials measured" was ~1
  trial/iteration, not 8 halvings in one search.
- With carry-forward already eliminating the duplicate candidate fit, the
  loop runs exactly one full fit per outer iteration — the structural
  minimum under the exact contract. Because any sound surrogate design
  keeps a mandatory exact fit at acceptance (audit §E item 12:
  pure-surrogate trials are not safe), a surrogate can only cheapen
  REJECTED trials — of which the measured workloads have none. **A
  surrogate would skip nothing.**

### Reopening condition

Only reopen on telemetry showing material step-halving counts on a real
workload — and note the published profile can only prove halvings in ONE
direction: `reml_n_linesearch_fits > reml_n_outer_iter` implies halvings,
but the converse fails, because iterations that skip the search at bounds
contribute zero fits (7 of 20 on b_L800 itself), and that slack can absorb
real halvings undetected. The accepted-trial count the audit's triple
(20/13/13) relied on is NOT published — `optimize_direct_reml` has no
accepts counter on the Fisher path
(`reml_observed_mode_rejected_trial_count` increments only in
observed-geometry mode). Reviving therefore starts by adding an accepts
counter next to `accepted = True` in the step-halving loop so
`halvings = fits − accepts` falls out directly.

If that ever fires, the archived surrogate design — sixteen round-1 and
eighteen round-2 review findings (6 + 12) on its algebra — is preserved in
the PR #173 review threads, which are AUTHORITATIVE over this file's git
history: the committed text at `d96230d` predates the round-2 findings and
is known-wrong exactly where they apply (surrogate logdet missing the
equilibration term, "every Ω_j carries eigenstructure", unscaled update
vectors, unsigned Woodbury determinants, per-component log|S|₊), while
`154746e` is the round-0 sketch. The durable index of what the findings
cover:
equilibrated-coordinate updates with the `2·Σ log(column_scale[active])`
logdet term; Δλ-scaled update vectors; per-`penalty_kind` `U_j`
construction (identity/repeated/sum-to-zero components carry no full
coefficient-space eigenstructure, and `PenaltyComponent` stores no
eigenvectors, so `U_j` needs an `eigh` per component even in the dense
case); the wide-identity whole-fit scope exclusion; Woodbury determinant
signs for indefinite capacitance; shared-block log|S|₊ joint evaluation;
estimated-scale LAML families; rank-truncation AND base-fit-stationarity
eligibility gates; the full `decompose_gram` certificate re-application
after updates (an rcond band alone is not the policy); the two-sided
exact-trial fallback plus restart-from-first-feasible-step after a failed
exact re-check; the `trace_run` state-id contract on every evaluated
trial; and complete-fit memory/dispatch validation before any default
flip. Any revival must re-derive from those findings — not from the
retired §J.2 sketch, and not from `d96230d`'s hardened-looking but
pre-correction text.

## What ships instead: the retained-factor seam (Task 1)

`fit_irls_direct` computes the centered slope-system decomposition
(`reml_slope_rank = decompose_gram(centered_final.hessian)` in the dense
`_compute_reml_geometry` branch of `_fit_irls_direct_once`) and previously
kept only its pseudo-inverse. The seam retains the `RankDecomposition`
itself on `PIRLSResult` behind an opt-in internal kwarg — default off,
zero behavior and memory change for every existing caller.

**Why it stays despite the retirement:** the strongest current grounds are
MEMORY, not the retired speed headlines. §E row 7's memory figure is
untouched by §J: the per-candidate p×p pseudo-inverse materialisation it
names is exactly this seam's site (the `pseudo_inverse()` call on the
retained decomposition), and routing consumers through a retained factor
~halves exact-REML peak memory ((2q+3)p² → ~q·p² + p²). Retention alone
does NOT deliver that: with the flag on, a fit holds the factor AND the
still-unconditionally-materialized inverse — strictly more memory. The
win's second half is making the `pseudo_inverse()` materialization
conditional once consumers route solves through the retained factor; that
is RFC-2/RFC-7 work, not this seam's. On speed, cite the audit's own
supersessions honestly: §J.2 DEMOTES RFC-2 for compressible designs
(W-correction measured 0.36 s of the 4.7 s tensor fit post-compression),
§J.5 re-promotes it narrowly for the truly-continuous-covariate regime
inside `ti()`, and §J.6's close drops RFC-7's urgency (the explicit-inverse
cost shrank 35× under the BLAS cap). Factor-L retention remains the listed
prerequisite for that narrowed RFC-2 and for RFC-7's memory win — this
seam is that retention, shipped and contract-tested.

**Interface** (for the RFC-2/RFC-7 consumers):
- `PIRLSResult.reml_slope_decomposition: RankDecomposition | None`
  (default `None`); `fit_irls_direct(..., _retain_reml_decomposition=False)`
  (underscore-prefixed: internal-only, matching `_compute_reml_geometry`
  et al.).
- Consumers get `.cholesky_factor`, `.solve()`, `.log_pdet`, `.method`,
  `.rank_truncated`, and the decomposition's OWN `column_scale`. Beware:
  `REMLGeometrySummary.column_scale` is `sqrt(diag(data_gram))` — a
  DIFFERENT matrix's diagonal than the decomposition's
  `sqrt(diag(data_gram + S))`; any equilibrated-basis work must use the
  retained decomposition's own scale. Note also that the factor
  decomposes the equilibrated active system `D⁻¹ H_c D⁻¹`, its
  `cholesky_factor` is frozen read-only, and a present `cholesky_factor`
  does NOT imply full rank (truncated `pivoted_cholesky`/certified
  `qr_svd` decompositions carry a representative-submatrix factor) — gate
  on `method`/`rank_truncated`, never on factor presence.
- `log_det_H` measure contract (contract comment on
  `PIRLSResult.log_det_H`): at full rank log|H_aug|; under rank truncation
  `log(sum(W)) + log|H_c|_+` where `log_pdet` already includes the
  `2·Σ log(column_scale[active])` equilibration term.
- Retention entry decision for any hot-path consumer: `PIRLSResult`
  retention means `_publish`/`_freeze_result_arrays` clones the
  decomposition (O(p²) per publication copy); the `cache_out` idiom in
  `fit_irls_direct` retains dense per-fit state WITHOUT entering the
  published result graph and is the right home for per-iteration reuse.

**Task 1 record (shipped in PR #173):**

- [x] Failing tests first: retention `None` by default and without REML
  geometry; opt-in retention matches an independent oracle
  (`slogdet`/`solve` of the centered Hessian assembled from
  `S_override`); the certified rank-truncated path retains the certified
  decomposition (`method == "qr_svd"`) with the measure identity intact;
  the structured path retains `None` by design.
- [x] Implementation: `PIRLSResult` field + kwarg, populated under
  `_compute_reml_geometry and _retain_reml_decomposition`; structured
  branch hardcodes `None` (structured Schur factors have their own
  retained-factor protocol).
- [x] Full suite green; carry-forward invariant test untouched.
