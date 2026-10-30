# Issue 544: truncated and censored observations for LSS families — assessment

2026-10-30, against master `40309929`. Stage 1 (section 9) is built on this note: the
log-normal, Gaussian and generalized Pareto families, the observation plumbing and the
splice-recipe docs.

## Verdict

The issue is right and reproduces exactly on master. It is worth doing, and smaller than the
issue states: families do not need analytic third and fourth derivatives to take part in
REML, because superglm already differences the analytic Hessian when a family lacks them.
Cost depends on the family. Families whose CDF is a normal CDF (Gaussian, log-normal, the
two-piece families) and the generalized Pareto are nearly free; the gamma and generalized
gamma need shape derivatives of the incomplete gamma function and are the expensive part.

## 1. Reproduction on master

The issue's script, unchanged apart from timing prints (200,000 gamma claims, mean 5,000,
CV 1.2, `u` = 12,574 at the 90% point):

| fit | fitted mean / CV | mean below `u` | `P(Y > u)` |
|---|---|---|---|
| truth | 5,000 / 1.20 | 3,428 | 10% |
| rows above `u` dropped, `GammaLS` | 3,435 / 1.11 | 2,961 | 3.5% |
| same rows, truncated likelihood (scipy) | 5,014 / 1.20 | 3,435 | 10.0% |
| claims capped at `u`, `GammaLS` | 4,338 / 1.13 | 3,346 | 6.8% |
| same rows, censored likelihood (scipy) | 4,984 / 1.20 | 3,429 | 9.9% |

## 2. What the problem is

Each row gains one term in its natural parameters `theta_i`:

- truncated to `(a_i, b_i]`: `log f(y_i) - log(F(b_i) - F(a_i))`;
- right-censored at `c_i`: `log S(c_i)`; left- and interval-censoring are `log F(c_i)` and
  `log(F(r_i) - F(l_i))`, the same machinery as two-sided truncation.

The log-likelihood stays row-separable with a few natural parameters per row, so it sits
inside the assumptions of Wood, Pya and Säfken (2016, JASA 111, sections 3.1-3.2, appendix
A). What is new is special-function evaluation (`log Phi`, `log P(a, x)`, `log Q(a, x)` with
accurate tails and derivatives) and the plumbing that carries per-row bounds.

## 3. Literature and other software

Sweep of 2026-10-30: alphaXiv, package documentation and manuals, and LICENSE files. No GPL
source was opened.

**How others do it.**

- mgcv (documentation only): `cnorm`, `clog`, `cpois` and `bcg` censor through a
  two-column response, as extended families with one linear predictor and a shared scale.
  mgcv documents no truncation family, and none of its multi-predictor families (`gaulss`,
  `gammals`, ...) is censored. Its documentation warns that censored-only data "contain
  very little information".
- gamlss.cens and gamlss.tr (manuals): turn any family into a censored or truncated one,
  with per-row bounds (`varying = TRUE`). The manuals say the wrapper switches the first
  derivatives to numerical ones and that the default step "could be inappropriate". Both
  are labelled experimental, and the fitting is RS/CG penalised likelihood, not LAML.
- brms `cens()` and `trunc()`: per-row bounds, fitted by MCMC.
- Fung, Tzougas and Wüthrich (2021, arXiv:2103.07200), eq. 2.1 and section 5: the body is
  right-truncated `f(y)/F(tau)`, and an ordinary splice's likelihood factorises into
  exceedance, body and tail. Both claims in the issue are confirmed.
- Concavity: the Tobit likelihood is concave only after Olsen's reparameterisation (Olsen
  1978, Econometrica 46(5)), and Pratt (1981, JASA 76) covers location parameters only. So
  per-row Hessians in `log sigma` or the gamma shape can be indefinite.

**Where we are new.** No documented software combines per-row truncation or censoring,
several linear predictors, and LAML with exact derivatives. This is assembly of known
pieces, not new mathematics.

**The normal CDF.**

- Values: `scipy.special.log_ndtr` (Cody-type branches) is the standard.
- Derivatives: they come from the inverse Mills ratio `r`, which superglm already uses in
  `two_piece_left_mills` / `two_piece_right_mills`. That recurrence cancels in the far
  lower tail. The sweep measured relative error on the fourth derivative of 1e-11 at
  `z = -3`, 4e-8 at `z = -8` and 1e-4 at `z = -20`.
- The fix: the Laplace continued fraction for `r`, evaluated on order-4 Taylor jets, is
  accurate to about 1e-15 at every order. It runs only on tail rows (`z < -3`).

**Incomplete gamma, shape derivatives.**

- Every partial that involves `x` is closed form, because `dP/dx` is the gamma density.
  Only the pure shape derivatives `d^k P / da^k` need an algorithm.
- Published methods:
  - Moore (1982), Algorithm AS 187, Applied Statistics 31(3): orders 1 and 2, by series
    and continued fraction.
  - Gautschi (1979), ACM TOMS 5(4), the basis of Stan Math's routine: order 1.
  - Figurnov, Mohamed and Mnih (2018, arXiv:1805.08498, appendices B-C): forward-mode
    differentiation through the series and continued fraction, at about twice the CDF's
    cost and about 5e-13 absolute error.
  - No tested routine for orders 3 and 4 was found.
- Recipe the sweep tested against mpmath:
  - the series with Bell-polynomial terms for `x < a`, and the Legendre continued fraction
    for `Q` on order-4 Taylor jets in `a` for `x >= a`;
  - every order is within 1e-13 for `a` in 0.3-200, and the two branches agree at `x = a`;
  - untested for `a < 0.05` (CV above about 4.5) and `a > 1e3` (CV below about 0.03).
  - I spot-checked orders 1 and 2 against differences of `scipy.special.gammaincc` at four
    points, and they agreed within the differencing error (at most 5e-6).
- Differentiating under the integral sign (Gauss-Legendre on truncated log-moments) needs
  128 nodes for 1e-11 and fails for `x >> a`. Use it as a test oracle, not in production.

**Generalized Pareto.** Closed form, but it needs a Taylor branch in `xi` near zero: the
fourth derivative's error is 7e-5 at `xi = 1e-3` without one.

**Expected information.** There is no closed form for the truncated gamma. mgcv/WPS use
observed information throughout, which is superglm's default. Truncated or censored plans
should not claim expected information.

## 4. What superglm already has

- `ObservationContract(kind="complete")` in `distributional/family.py`, which refuses any
  other kind, and an unused `FamilyCapabilities.censored_response` flag.
- Third and fourth derivatives: when a family does not implement the analytic hooks, they
  come from Richardson-extrapolated finite differences of the analytic packed Hessian
  (`distributional/smoothing/endpoint_direction.py`, an `eps^(4/5)` floor). A new
  observation kind needs only value, score and Hessian to take part in the whole REML
  ladder. Analytic orders 3 and 4 are a speed-up for the LAML endgame.
- The default `fit_reml` on the timing model below ran EFS only (status
  `converged_uncertified`, no LAML endgame), so orders 3 and 4 were not evaluated at all.
- Curvature policy (`distributional/solver/curvature.py`): observed curvature by default,
  with a nearest-semidefinite fallback after Wood and Fasiolo (2017). This handles
  indefinite truncated rows.
- `tools/generate_lss_fourth_derivatives.py` already emits the Mills-ratio derivative
  blocks.

## 5. Plumbing a build must cover

- **API.** Per-row bounds passed to `fit` / `fit_reml` beside `sample_weight` and
  `offsets`. A scalar broadcasts, because a splice uses one `u`.
- **Plan data.** The contract is a schema descriptor; the bounds become plan data, owned
  like weights, and `take()` must slice them for chunks and cross-validation.
- **Likelihood reuse cache.** The bounds must enter the plan identifier and the reuse
  digests (`prepared_array_fields`). Otherwise a cached evaluation can be reused across
  different bounds.
- **Weight semantics.**
  - Under prior (EDM) weights, the row law has shape `w * alpha`, and the normaliser uses
    that row's own law.
  - Under literal replication, the weight multiplies the whole row term, normaliser
    included.
- **Consumers that bind `COMPLETE_OBSERVATION` today** (14 files) need the row's own
  contract:
  - residuals: randomised PIT on `(F(c), 1)` for censored rows;
  - held-out scores in `checks/scores.py`;
  - the posterior;
  - the null model;
  - serialization.
- **`coefficient_curvature="fisher"`.** Refused, or not offered, for truncated or
  censored fits.
- **Meaning of the fitted law.** The bounds describe how rows were observed, so the
  fitted law is the parent (untruncated) law. After a truncated body fit, `predict()`
  returns the parent mean (5,014 in the example), not the mean below `u`. This is
  user-visible and needs documenting next to the splice recipe.

## 6. Performance

**Measured after stage 1 (PR #545).** All fits are 50,000 rows, `fit_reml` defaults, one
thread, on the dense backend (`distributional-dense-v1`).

- **Complete fits on all nine families:** master and branch take the same wall time
  (16.77 s and 16.72 s in total), give bitwise-equal predictions and use the same peak
  RSS (440-465 MiB).
- **Log-normal body truncated at its 90% point:** 1.20 s, against 0.77 s for the same rows
  read as complete.
  - Smoothing selection takes 12 outer iterations instead of 6.
  - The likelihood costs 231 ns per row per call instead of 110.
  - Peak RSS is 442 MiB, against 441 MiB complete.
- **Censored generalized Pareto:** 1.49 s, against 5.59 s for the complete reading of the
  capped rows.

The gamma figures below are estimates written before the build; stage 2 measures them.

Measured on master, one thread, 10^6 rows, best of 5:

| per-row operation | ns/row |
|---|---|
| current `GammaLS` row kernel, order 2 (value, score, Hessian) | 739 |
| gamma log-density alone | 23 |
| `scipy.special.gammainc` (P) | 257 |
| `scipy.special.gammaincc` (Q) | 437 |
| digamma + trigamma | 242 |
| `scipy.special.log_ndtr` | 29 |

**Fit attribution.** 200,000 rows, `GammaLS` with `mean ~ s(x1) + s(x2)` and
`scale ~ s(x3)`, `fit_reml` defaults, 4 cores: 8.9 s in all.

- Likelihood evaluation: 2.82 s (32%) over 31 calls.
- Penalty assembly: 2.31 s.
- Predictor compilation: 1.68 s.
- The search took 6 EFS iterations.

**Estimates (not measured in a fit).**

- **Gaussian / log-normal truncation:** about 50 ns per truncated row per call, or roughly
  0.3 s on the 8.9 s fit.
- **Generalized Pareto censoring:** negligible.
- **Gamma truncation:**
  - The sweep's vectorised numpy prototype took 13.5 µs/row; a numba kernel is estimated
    at 1-3 µs/row.
  - On a fully truncated 200,000-row body that adds about 6-19 s to the 8.9 s fit, before
    any optimisation.
  - Censoring costs only on the censored rows.

**Optimisation, after the method is in.**

- Evaluate only the derivative order a call asks for: value-only calls need just `P`, at
  257 ns.
- Skip rows whose `F(b)` rounds to one.
- When the scale predictor takes few distinct values (intercept-only, a factor), build one
  Chebyshev interpolant in `log x` per distinct shape, in place of a series or continued
  fraction per row. This is untested.
- Analytic orders 3 and 4 remove the differenced Hessian calls from the LAML endgame.

**Side finding, outside this issue.** The existing `GammaLS` row kernel spends about
240 ns/row on digamma and trigamma, recomputed per row even when the shape is constant
across rows. It is a follow-up candidate.

## 7. Export to `aggregate`

**The package.**

- `aggregate` 1.11.0, BSD-3-Clause (LICENSE file; PyPI `license_expression`), Python 3.12
  or later.
- Its dependencies include matplotlib, lark, IPython and Pygments, so it belongs behind an
  optional import, not in core dependencies.

**Severities in DecL** (its specification language):

- Severities are scipy names, and `gamma` takes mean and CV natively, which is exactly
  `GammaLS`'s parameterisation.
- `splice [lb ub]` applies conditional truncation, so the body in a splice is precisely
  the truncated law this issue fits.
- Today's complete-observation body under `splice [0 u]` is the 14%-low case in the table
  above.
- A segment's spliced severity is a two-component mixture:
  - weights `[1 - p, p]`;
  - `gamma(mean, cv)` spliced to `[0, u]`;
  - the generalized Pareto shifted to start at `u`.
- The exact mixed-family syntax and the open/closed convention at `u` are unverified.

**Discretisation and cost.**

- The default grid is 2^16 buckets, with bucket probabilities from `cdf` / `sf`
  differences. An export must provide `cdf`, `sf` and moments.
- Per segment that is about 65,000 CDF evaluations: around 20-30 ms for a gamma, about
  2 ms for a log-normal, plus the FFT. A rating table of 1,000 cells takes tens of
  seconds.
- A per-policy portfolio mixture costs rows × 65,000 evaluations, which is not feasible on
  a book. Export per segment or rating cell, not per row.

**Roadmap.** This is the established engine for C11 (deterministic portfolio aggregation).
An export avoids hand-rolling FFT or Panjer aggregation.

## 8. Risks

- **A low `u` weakly identifies the shape.** Fits may take more iterations or use the
  semidefinite fallback more often. Disclose this; do not refuse the fit.
- **Generalized gamma near `Q = 0`.** The truncated law must reach the log-normal limit
  smoothly, as the existing kernel's series does for the density.
- **Shape range.** The incomplete-gamma engine is untested for CV above about 4.5 or below
  about 0.03.

## 9. Staging

1. Plumbing (API, plan data, cache digests, residuals, scores, posterior), Gaussian and
   log-normal truncation and censoring, generalized Pareto censoring at policy limits, and
   the docs paragraph. Acceptance: the issue's test on a log-normal analogue, plus
   regression tests that fail on today's complete-observation path.
2. `GammaLS`: the series and continued-fraction engine at order 2, with orders 3-4
   differenced. Acceptance: the issue's own criteria (5,000 / 1.2 within sampling error).
3. The two-piece families (normal-CDF pieces), then `GeneralizedGammaLSS`.
4. The optimisation pass, the splice wrapper (issue item 4) and the `aggregate` export.

## Runs behind this note

These scratch scripts are not committed:

- the issue's reproduction on master;
- the per-row cost table;
- the 200,000-row phase attribution through `SuperLSS.diagnose()`;
- a finite-difference spot check of the incomplete-gamma jets;
- the sweep's own mpmath experiments for the normal CDF, the incomplete gamma, the
  quadrature route and the generalized Pareto.

Everything heavy ran through benchboard.
