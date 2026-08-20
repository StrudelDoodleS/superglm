# Issue #339 on v0.29.0 — re-measured, not inferred

**Question.** #339 reported that REML settles a crossed-categorical
`RandomEffect` on a variance component roughly 17× larger than an established
reference reaches on the same specification and data (τ 0.344 vs 0.0207;
effects spanning [−0.28, +0.23] against [−0.009, +0.005]). v0.29.0 shipped the
exact Tweedie scale profile whose absence the distribution-estimation audit
(this directory's README, §2) identified as the criterion defect behind that
failure class. Is the 17× gone?

**Answer: the 17×-class divergence is gone; a residual factor ≈ 2 in τ
remains.** On a fingerprint-faithful reconstruction of the reporting
configuration, v0.28.0 sits 6.2× above the reference in τ (37.9× in τ²) and
far outside the reference's own 95% interval; v0.29.0 lands 2.0× above in τ
(3.9× in τ²), **inside** the reference's 95% interval, with the criterion
near-flat between the two optima (measured below). Improved, not fully
closed.

## 1. What was run

The probe that produced the issue's numbers was not preserved, so the
configuration was reconstructed from the issue text and the reporter's field
notes, then validated two ways: (i) its design fingerprint reproduces the
issue's exactly — 84,138 training rows, a 228-cell crossed factor (23-level
vehicle attribute × 11-level regional band), 34 claim-free cells, 73 cells
under 30 rows, 83.0% zero responses; (ii) the pre-fix release reproduces the
reported failure class (large τ, wide effect span) on it. The private
dataset's identifying details do not appear here or in any file behind this
document; features are described by role only.

Specification (identical in all arms): Tweedie(p = 1.5), log link, EDM prior
weights, log-scale offset; five ordered axes as cubic regression splines on
their band positions (k = 10, 8, 5, 4, 8; two axes carry one structural level
each as a free, unpenalised effect); eight unpenalised categoricals
(reference = most-exposed level); the crossed 228-cell factor as a
random-effect term. Monotone shape constraints off in all arms (the reference
cannot express them). fit_reml / `method="REML"`, max 30 outer iterations.

Arms, run sequentially under the bench lock, BLAS/OpenMP pools pinned to 1:

- **superglm v0.28.0** (`.worktrees/release-0.28.0`, tag build) — the
  Gaussian-shaped scale substitution, i.e. the criterion #339 was filed
  against. Reconstruction-validation arm.
- **superglm v0.29.0** (`244c2f87`, the release under verification).
- **mgcv 1.9.3** on R 4.5.0, `gam(..., method="REML")` with the same formula
  content, `s(cell, bs="re")` for the crossed factor and fixed-p Tweedie
  (Wood 2011; Wood, Pya & Säfken 2016; Wood 2017). Used strictly as a
  black-box oracle: run, outputs pinned below, no source consulted. Its
  deductible-axis basis is capped at k = 3 because that axis has three
  distinct positions after the source specification's banding (mgcv refuses
  k > distinct values; both systems fit a saturated 3-point curve there, so
  the cap cannot move the RE variance).

τ² is φ/λ_cell under each system's own published dispersion, the same
convention as the audit's §2.5 probe. mgcv's τ comes from `gam.vcomp`
(std.dev row of the RE smooth), its span from the fitted RE coefficients.

## 2. Results

| arm | τ | τ² | λ_cell (sp) | operative φ | RE effect span | total edf | deviance |
|---|---|---|---|---|---|---|---|
| superglm v0.28.0 | 0.4405 | 0.1940 | 1,015 | 197.0 | [−0.584, +0.633] | 189.7 | 5,676,015 |
| superglm v0.29.0 | **0.1408** | **0.0198** | 10,019 | 198.5 | **[−0.244, +0.155]** | 127.0 | 5,693,241 |
| mgcv 1.9.3 REML | 0.0715 | 0.0051 | 38,500 | 196.9 | [−0.109, +0.058] | 102.8 | 5,702,048 |

mgcv's 95% interval for τ on this fit: **[0.0288, 0.1776]** (`gam.vcomp`).
Operative φ for mgcv is the value its variance-component report is built on
(τ² × sp = 196.9); dispersion therefore agrees across all three arms to 0.8%,
matching the issue's observation that the divergence is specifically in the
smoothing-parameter search.

Residual ratios against the reference:

| | τ ratio | τ² ratio | span width ratio | inside reference 95% CI? |
|---|---|---|---|---|
| v0.28.0 | **6.2×** | 37.9× | 5.4× | no — 2.5× above the upper bound |
| v0.29.0 | **2.0×** | 3.9× | 2.2× | **yes** |

Both fits converged (13 and 14 outer iterations); mgcv reported full
convergence in 7. All smoothing parameters, not only the RE, moved toward the
reference on v0.29.0 (total edf 189.7 → 127.0 against mgcv's 102.8), and the
fitted deviance moved from the over-fit side (5,676,015) to within 0.15% of
the reference (5,693,241 vs 5,702,048).

### Criterion flatness at the residual gap

Refitting on v0.29.0 with λ_cell pinned at the reference's value (38,500 — a
3.84× move, the full residual gap; other smoothing parameters re-estimated)
raises the profiled REML criterion by **2.50** (166,638.27 free vs 166,640.76
pinned), about the size of a 1-df 95% profile-likelihood threshold (1.92),
and lands the fitted deviance on the reference's to five significant figures
(5,702,021 vs 5,702,048). Both fits converged. The residual ≈2× in τ is
therefore a near-flat-basin placement difference the criterion itself can
barely distinguish, not a wrong optimum: the v0.28.0 criterion preferred its
answer to the reference's by a qualitatively different margin (λ 38× apart,
deviance 0.46% apart, τ outside the reference CI).

## 3. Reading

1. **The headline failure is fixed in kind.** The pre-fix criterion put the
   crossed RE multiples beyond anything the reference's own sampling
   uncertainty could explain. The v0.29.0 criterion lands inside the
   reference's 95% interval on the same data. A user fitting this
   specification on v0.29.0 no longer gets the [−0.58, +0.63] separation-led
   effect surface; they get [−0.24, +0.16] against the reference's
   [−0.11, +0.06].
2. **Not numerically closed.** A factor ≈ 2.0 in τ (3.9 in τ²) remains, on
   the same side (under-shrinkage). This is the audit's §2.5 "flat direction"
   regime — the marginal likelihood is weakly informative about this τ (the
   reference's own CI spans a factor of 6.2) and remaining criterion
   differences (outer-optimiser stopping in the flat basin, LAML vs mgcv's
   exact-REML determinant details, W-correction order) move the optimum by
   factors where curvature is this low.
3. **Magnitude caveat.** The original probe was lost, so the issue's exact
   17× is not bit-reproducible; this reconstruction shows the same failure
   class at 6.2× pre-fix. The verdict (gone in kind, ≈2× residual) is
   measured on one fixed reconstruction with all three arms on identical
   data, so the comparison itself is exact.

## 4. Disposition

Recommended: keep #339 open with these numbers, re-scoped to the residual
≈2× flat-basin gap, or close it as "fixed to within criterion flatness" if
the maintainer weighs the CI containment as closing evidence. The verifying
comment posted to the issue carries this table; the close decision is left
with the maintainer. Refs #339.

Reproduction artifacts (neutral CSV, arm JSONs, pinned mgcv output) are
session-local and deliberately uncommitted; the design fingerprint above is
sufficient to rebuild them from the reporter's material.
