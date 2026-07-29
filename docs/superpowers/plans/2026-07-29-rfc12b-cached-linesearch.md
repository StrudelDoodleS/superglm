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

Only reopen if line-search telemetry (`_n_linesearch_fits` vs outer
iterations in `optimize_direct_reml`) shows a workload class with material
step-halving counts. If that ever happens, the archived surrogate design —
including sixteen round-1 and a dozen round-2 review findings on its
algebra (equilibrated-coordinate updates, Δλ-scaled update vectors,
Woodbury determinant signs for indefinite capacitance, shared-block
log|S|₊ joint evaluation, estimated-scale LAML families, rank-truncation
and stationarity eligibility gates, the two-sided exact-trial fallback) —
is preserved in the PR #173 review threads and this file's git history
(`154746e`, `d96230d`). Any revival must re-derive from those findings,
not from the retired §J.2 sketch.

## What ships instead: the retained-factor seam (Task 1)

`fit_irls_direct` computes the centered slope-system decomposition
(`reml_slope_rank = decompose_gram(centered_final.hessian)` in the dense
`_compute_reml_geometry` branch of `_fit_irls_direct_once`) and previously
kept only its pseudo-inverse. The seam retains the `RankDecomposition`
itself on `PIRLSResult` behind an opt-in internal kwarg — default off,
zero behavior and memory change for every existing caller.

**Why it stays despite the retirement:** factor retention is the listed
prerequisite of two live audit items — RFC-2 (batched whitened
W-correction, "needs factor-L retention (RFC-7)", 39-85% of exact-fit
runtime) and RFC-7 (factorization-backed `HessianFactor`, which also
~halves exact-REML peak memory). This seam is that retention, shipped and
contract-tested.

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
  branch hardcodes `None` (Schur factors are not dense-updatable).
- [x] Full suite green; carry-forward invariant test untouched.
