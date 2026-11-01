# Large-loss (peaks-over-threshold) modelling: literature sweep and staged plan

2026-10-31, against master `72c60b21`. Section 8, on threshold choice for the
layer method, was added the same day. PR #545 (per-row `truncation=` and
`censoring=`, head `90b0eb46`, unmerged and on hold) is read as the next state
of the code. This note is research input for the roadmap (C11, C12, C13 and
the generalized Pareto shape note). It changes no code.

**Status, 2026-11-01.** PR #545 has merged (`9d0b2de3`). It takes a per-row flag, `censored=`, in place of the `censoring=(lower, upper)` pairs described here; the GPD shape-wall disclosure merged as #558 (`5d89809b`).

## Verdict

- **The core method is already here.** superglm fits the generalized Pareto
  distribution (GPD) with smooth scale and shape. It chooses smoothness by
  Laplace-approximate marginal likelihood (LAML) with exact derivatives. That
  is the method evgam uses (Youngman 2022), the reference GAM-for-extremes
  package. Inside its shape walls the fitted shape is within 2e-4 of scipy's
  maximum-likelihood estimate.
- **The three-fit splice is the maximum-likelihood splice at a fixed
  threshold**, provided no row's censoring interval crosses `u`. It also needs
  the body fitted as truncated, which PR #545 supplies. Reynkens et al. (2017)
  use the same construction with the threshold fixed by expert extreme-value
  analysis, and say why they do not estimate it by likelihood.
- **The biggest gap is the shape walls, not threshold modelling.** The link
  keeps `0 < shape < 1`. Published insurance and operational-risk tails sit
  on both sides of that range:
  - catastrophe, cyber and operational-risk losses, and some marine
    liability, have tail index ξ ≥ 1;
  - Weibull-type and bounded property excesses fit ξ slightly below 0 at
    practical thresholds.
- **At the walls the fit is pinned, and today it says nothing.** A simulated
  ξ = 1.2 tail fits as ξ = 1 − 1.1e-16, and `predict()` returns 1.03e16 with
  no warning. That is a silently wrong number in something a user reads,
  which AGENTS.md says must never happen (see "Measured: what the walls do").
- **The next user-visible steps are cheap and need no new solver.** In
  order:
  - disclose when the shape is at a wall;
  - add threshold-stability diagnostics;
  - build a splice object that combines the three fits;
  - add layer functionals.

  Covariate thresholds and a jointly estimated threshold (C13) come later,
  behind gates.
- **For the layer method Max has chosen, `u` is a joint body–tail choice**
  (section 8).
  - The held-out log score of the whole layer density is a proper way to
    compare thresholds, but the body dominates it.
  - The censored log score at a high validation threshold (Northrop,
    Attalides and Jonathan 2017) targets the tail, but separates thresholds
    only weakly.
  - In a simulation neither score picked the threshold that priced an
    excess layer best. The practical check is a stability plot of each
    priced quantity against `u`.

## 0. What the problem is, in standard vocabulary

- **Peaks over threshold.** The excess `Y − u | Y > u` tends to a GPD as `u`
  rises (Pickands–Balkema–de Haan). The GPD fitted to excesses above a
  finite `u` is an approximation whose quality depends on `u`. Belzile and
  Davison (2026, section 2) call the GPD that fits best at a finite threshold
  the *penultimate* model.
- **GPD regression with smooth effects** is a vector GAM, or GAMLSS, with a
  non-exponential-family likelihood. That is a "general smooth model" in the
  sense of Wood, Pya and Säfken (2016): row-separable, a few natural
  parameters per row, with smoothing parameters chosen by LAML.
- **A covariate-dependent threshold** is a conditional-quantile M-estimation
  problem: check loss, or a working asymmetric-Laplace likelihood. It is not a
  likelihood for the data.
- **A splice** (also called a composite model) is a two-component mixture
  whose components have disjoint supports `[t_l, u]` and `(u, T]`. Both
  components are truncated densities. The likelihood factorises into body,
  tail and splicing weight whenever every observation lies wholly on one side
  of `u` (Reynkens et al. 2017, section 4.1).
- **Layers, limits, deductibles and open claims** are coarsened observations
  of the ground-up loss:
  - a deductible is left truncation;
  - a policy limit is right censoring;
  - a claim known only to lie between two values (between non-contiguous
    layers, or between paid and incurred) is interval censoring.
- **Layer cost** is the limited expected value
  `E[min((X − M)+, L)] = ∫_M^{M+L} S(z) dz` (Albrecher and Beirlant 2025,
  section 2).
- **Portfolio aggregation** is a compound distribution, computed by
  simulation, FFT or Panjer recursion.

## 1. Threshold choice

### The standard method and what it assumes

**A fixed threshold chosen by threshold stability.** If the excesses over
`u` follow a GPD(σ_u, ξ), the excesses over any higher `v` follow
GPD(σ_u + ξ(v − u), ξ) (Murphy, Tawn and Varty 2025, section 2). So:

- **Mean residual life (mean excess) plot.** It is linear in `v` above a
  valid `u`. It is unusable when ξ ≥ 1, because the mean excess is then
  infinite (Chen and Wang 2024, section 4). Reynkens et al. (2017, section
  6.2) use a version with the Turnbull estimator under censoring.
- **Parameter-stability plots.** The shape, and the modified scale
  `σ* = σ_u − ξu`, should be constant above a valid `u`. The usual automatic
  rule picks the lowest threshold whose shape estimate lies inside the
  profile-likelihood intervals at every higher threshold (Belzile and Davison
  2026, procedure P1).

**Automated selection and threshold uncertainty.** Belzile and Davison
(2026) review more than 40 procedures and run a common simulation study.

- No procedure is uniformly best.
- An oracle that targets the 0.999 quantile picks thresholds between the
  0.87 and 0.9 quantiles. Most automatic methods pick the lowest candidate
  instead.
- Two methods perform reasonably well:
  - Northrop, Attalides and Jonathan (2017), cross-validation with Bayesian
    model averaging across thresholds;
  - Collings et al. (2025, TAILS).

  The expected-quantile-discrepancy method of Murphy, Tawn and Varty (2025)
  does well relative to the oracle.
- Ignoring threshold uncertainty under-covers. Murphy, Tawn and Varty
  propagate it with a double bootstrap: resample, reselect `u`, refit. On
  their Gaussian case (Table 6), 95% intervals that carry only parameter
  uncertainty cover 0.51–0.75. Adding threshold uncertainty raises that to
  0.76–0.87, at 1.5–1.7 times the width.
- Maximum-likelihood estimation degrades when ξ ≥ 0.5. The authors say such
  tails "arise rarely in applications"; insurance is the exception (section
  5 below).

**The penultimate shape is what a fit sees.** Belzile and Davison (Table 3)
fit maximum-likelihood GPDs to a million exceedances of known laws:

| law | limit ξ | fitted ξ at the 0.8–0.99 quantile |
|---|---|---|
| standard lognormal | 0 | 0.28 to 0.22 |
| gamma, shape 2 | 0 | −0.04 to −0.01 |
| Weibull, shape 1.25 | 0 | −0.06 to −0.03 |
| Weibull, shape 0.75 | 0 | 0.11 to 0.05 |

The lognormal row matters for this book: lognormal-like severities fit
ξ ≈ 0.25 at working thresholds, inside the current walls. Gamma- and
Weibull-like excesses want a negative shape.

**A covariate-dependent threshold `u(x)`.**

- **Methods.**
  - `u(x)` is a conditional τ-quantile from quantile regression (Northrop
    and Jonathan 2011, via Youngman 2022).
  - Youngman (2019) estimates it as a GAM-form quantile and fits GAM-form
    GPD parameters to its excesses, with REML smoothing (abstract).
  - evgam does this with an asymmetric-Laplace "family" that uses the
    modified check function of Oh, Lee and Nychka (2011) (Youngman 2022,
    section 2.1).
  - qgam (Fasiolo et al. 2021) is the calibrated alternative.
  - In insurance, Fissler, Merz and Wüthrich (2021) splice at a conditional
    90% quantile, not a constant. They prove joint elicitability of the
    quantile with the lower and upper expected shortfall, and fit all three
    by one strictly consistent score.
- **What it buys.**
  - By construction `P(Y > u(x) | x) = 1 − τ` for every row, so the
    exceedance GLM of the splice becomes a constant.
  - A fixed `u` sits at very different conditional quantiles for small and
    large risks; `u(x)` adapts to heteroskedasticity (Fissler et al.).
- **What it assumes.**
  - The quantile fit is correct.
  - τ is low enough to leave data in the tail: Belzile and Davison note
    such methods set it at 0.8 or 0.9.
  - The asymmetric-Laplace likelihood is a working likelihood, so its
    intervals need calibration (the point of Fasiolo et al.).
- **Threshold uncertainty is mostly ignored.** evgam's return-level example
  simulates the quantile and GPD parameters together from their separate fits
  (Youngman 2022, section 4.3). It does not refit the excess set at each
  draw.

**What is special to insurance.** Albrecher and Beirlant (2025, section 2)
make two points:

- **Commercial attachments are not statistical thresholds.** The commercial
  retention `M` and limit `L` need not be statistically good thresholds,
  which is why actuaries need whole-range (spliced) models.
- **Catastrophe claims are aggregated per event.** Claims from one event are
  summed into one claim before a retention applies. That is the insurance
  counterpart of declustering.

A fixed monetary `u` (the large-loss or reinsurance threshold) is therefore
the natural default for pricing. `u(x)` suits lines where size scales with
an exposure measure, such as sum insured.

### Where superglm stands

- **Fixed `u`: inside the method, without its diagnostics.** The fit is
  there. Mean-residual-life plots, parameter-stability plots and threshold
  uncertainty are not.
  - A stability plot is a loop of intercept-only GPD fits, each 0.1–0.2 s at
    4,000 rows (measured below).
  - With the lower wall at 0, a stability plot cannot show the negative
    shapes that gamma- or Weibull-like excesses produce. Its shape track would
    sit on the wall (measured below).
- **Covariate `u(x)`: outside.**
  - superglm has no quantile-regression family; C6/C7 are under Research.
  - The rest of the pipeline already accepts one threshold per row. PR #545
    takes per-row truncation bounds, so a body truncated at `u_i` works, and
    the GPD response `y_i − u_i` works today.
  - A threshold read off a fitted body's quantile is circular, because the
    body fitted below `u` depends on `u`. It should not be offered as the
    method.

## 2. GPD regression with smooth scale and shape

### The standard methods

| | Chavez-Demoulin and Davison (2005) | evgam (Youngman 2019, 2022) | EGPD in gamlss (Le Carrer and Gaetan 2022) |
|---|---|---|---|
| Parameters | shape κ, additive, identity link; ν = σ(1 + κ), log link | log scale; shape on identity link, any sign, exponential limit for \|ξ\| < 1e-6 | shape, scale and one or two lower-tail parameters |
| Why that parameterisation | ν is orthogonal to κ in Fisher information (Cox and Reid 1987); the authors say orthogonality is crucial for convergence | not stated | gamlss defaults |
| Fitting | penalised likelihood with natural cubic splines, by backfitting: a componentwise Newton step, then a weighted GAM fit per parameter | penalised likelihood; inner Newton for coefficients | gamlss RS/CG with symbolic first and second derivatives |
| Smoothness | chosen by AIC; likelihood ratio or AIC to compare models | LAML (Wood 2011; Wood, Pya and Säfken 2016); outer BFGS by default, or Newton | gamlss local criteria (`pb`, `ga(..., method="REML")`) |
| Uncertainty | deviance differences; "post-blackend" bootstrap: residuals from undersmoothed fits added to oversmoothed curves | Gaussian posterior simulation of the coefficients | standard errors only for fitted rows |
| Censoring | none | interval censoring via `cens(left, right)` | none |

Sources:

- Chavez-Demoulin and Davison. The method summary is from the published
  abstract and Chavez-Demoulin's 2007 Tokyo slides, which state AIC for
  smoothing. The backfitting steps are from Hofert's 2014 slides on
  Chavez-Demoulin, Embrechts and Hofert (2016), whose `gamGPDfit` (QRM)
  implements them.
- evgam: Youngman (2022), sections 2.4–3.1.
- EGPD: Le Carrer and Gaetan (2022), sections 3–4.

**A common modelling choice: keep the shape simple.**

- The evgam GEV example fixes the shape constant.
- Chavez-Demoulin and Davison let it depend linearly on altitude only.
- Le Carrer and Gaetan find that making the shape constant, or only
  time-varying, generalises better out of sample. They agree with Evin et
  al. (2018) that the shape is the hardest parameter to fit.
- Chavez-Demoulin, Embrechts and Hofert (2016) select a shape that varies by
  business line, a categorical effect.

### Where superglm stands

**Inside the method, and its strongest form.**

- `GeneralizedParetoLSS` uses a log link on the scale and a bounded logit on
  the shape.
- Smoothing is chosen by REML/LAML, with EFS and an optional Newton endgame.
- It uses observed curvature, with analytic derivatives to fourth order.
- Series branches keep the derivatives accurate as ξ → 0.
- This is the evgam approach, with smoothing chosen by marginal likelihood
  rather than AIC or a local criterion.
- Measured agreement with scipy's unrestricted maximum likelihood is
  2e-4 or better on the shape inside the walls. The superglm fit carries a
  smooth of a noise covariate on the scale, so the two models are not
  identical.

**Parameterisation.** superglm does not use the orthogonal ν. Backfitting
needed orthogonality to converge. superglm takes joint Newton steps on the
coupled Hessian, so it does not obviously need it. The original GPD EFS
failures are recorded as resolved without a reparameterisation (ROADMAP,
C3 follow-through). Measure outer-iteration counts before changing anything.

**Shape domain.** Here superglm is outside the standard method, which
leaves ξ unrestricted. See section 5.

## 3. Splicing with censored and truncated data, and the joint fit (C13)

### Reynkens, Verbelen, Beirlant and Antonio (2017)

- **The model** (their section 2.1).
  - The spliced density is `π f₁(x)` on `[t_l, t]` and `(1 − π) f₂(x)` on
    `(t, T]`.
  - `f₁` is the body density truncated to `[t_l, t]`, a mixed Erlang in
    their case.
  - `f₂` is the tail density truncated to `[t, T]`, a Pareto, with the GPD
    as an alternative.
  - The two parts share no parameters.
- **The splicing point is fixed, not estimated** (section 4.2).
  - They do not estimate `t` by likelihood with continuity or
    differentiability at `t` (Cooray–Ananda style), because that "results in
    less flexibility".
  - They choose `t` from the mean-excess plot. Under censoring they use the
    Turnbull-based version.
  - They note that adaptive EVT threshold rules can pick thresholds unsuited
    to splicing, as on the Danish fire data.
- **Censoring** (section 3).
  - Each row is an interval `[l_i, u_i]`, so that:
    - an uncensored row has `l_i = u_i`;
    - right censoring sets `u_i = T`;
    - left censoring sets `l_i = t_l`.
  - Open claims become interval-censored, between the payments to date and
    the incurred amount (the case estimate).
  - Five cases arise. In four of them every row lies on one side of `t`, and
    the likelihood separates.
  - Case v, an interval that straddles `t`, contributes
    `log(π + (1 − π)F₂(u_i) − πF₁(l_i))`. That term couples π, the body and
    the tail, which is why they use EM.
  - Without censoring, π̂ is the share of rows at or below `t`, and the tail
    is the Hill estimator.
- **Assumption.** The censoring is independent of the loss. Albrecher and
  Beirlant (2025, section 4) warn that this fails for long-tailed lines,
  where large claims stay open longer (Stupfler 2019).

### Where superglm stands

**The three-fit recipe covers Reynkens' cases i and ii, plus one-sided
censoring inside either part, with a GAM on every component.**
Interval-censored rows inside one part (cases iii and iv) wait for Stage D;
case v waits for Stage G. The body is truncated at `u` (PR #545). The exceedance is a
binomial GLM. The tail is a GPD on `y − u`, censored at `limit − u`. Each
fit is maximum likelihood for its own factor, because the likelihood
factorises (also confirmed by Fung, Tzougas and Wüthrich 2021, cited in the
issue-544 note).

**What is missing.**

1. **Rows that straddle `u`** (case v), for example an open claim paid below
   `u` with an incurred amount above it.
   - The three fits can no longer be separate.
   - A policy limit above `u` is not case v: the capped claim is censored
     inside the tail. A deductible below `u` is not either: it is the body's
     lower truncation bound.
2. **A splice object.**
   - Users combine the three fits by hand today.
   - The three parameter sets are disjoint and the likelihood factorises, so
     the Laplace posterior is block-diagonal. Independent draws from the
     three fits are therefore the joint posterior under the approximation
     the fits already use.
   - That makes spliced quantiles, survival and layer costs with intervals a
     composition of existing pieces.
3. **Gamma and generalized-gamma bodies.** They need issue #544 stage 2,
   the incomplete-gamma shape derivatives.

### What a joint body–tail fit (C13) would need beyond a fixed `u`

**Requirements.**

- **One data set.** Every candidate threshold must be judged on the same
  rows. Belzile and Davison (section 8.1.1) note that GPD fits above
  different `u` use different samples, so likelihood ratios and information
  criteria cannot compare them.
- **A normalised body.** A whole-range splice fixes this: at each `u` it is
  a proper density for every row, which makes `u` selectable by information
  criteria or held-out score (same section). The truncated body from PR #545
  is the piece that makes the splice normalised.
- **A selection rule.** The candidates, from cheapest to most work:
  1. **Profile over a grid of `u`.** Refit the three fits at each `u` and
     compare the held-out full-data log score. The spliced likelihood jumps
     wherever a row crosses `u`, so it is profiled on a grid, not optimised
     by Newton. This needs no new solver.
  2. **Average over thresholds**, with weights from cross-validated
     predictive performance (Northrop, Attalides and Jonathan 2017; one of
     the better performers in Belzile and Davison's study).
  3. **A threshold-free family.** The extended GPD (Naveau et al. 2016) has
     a GPD upper tail and a flexible lower tail. Le Carrer and Gaetan (2022)
     give it additive predictors. It would be a new LSS family with no
     threshold at all.
  4. **The threshold as a parameter** of a smooth mixture (MacDonald et al.
     2011, evmix; Behrens et al. 2004, Bayesian). These are hard to fit and
     sensitive to their priors (Murphy et al. 2025, section 3).
- **Uncertainty propagation.** A double bootstrap or averaging over
  thresholds. Murphy et al. show that ignoring threshold uncertainty
  under-covers.
- **Shape regularisation.** The dossier's Coles–Dixon penalty is a modelling
  choice, as the roadmap already says, not a fix for the walls.

## 4. Reinsurance layers

### Gaps between layers make interval-censored rows

Suppose an insurer sees ground-up losses up to the top `b₁` of one layer,
and sees them again only from the attachment `a₂ > b₁` of the next. A claim
that exhausts the first layer but does not reach the second is known only to
lie in `[b₁, a₂]`. Its likelihood term is `log(F(a₂) − F(b₁))`. The same
term covers three other cases:

- an open claim between paid and incurred (Reynkens et al., section 6.2);
- values recorded only to a rounding grid (Belzile and Davison, section
  8.1.4; Varty et al. 2021);
- evgam's `cens(left, right)` (Youngman 2022, section 3.1).

### What supporting it takes in superglm

PR #545 already computes the needed quantity: tail-safe window log
probabilities `log(F(b) − F(a))` with their score and Hessian, and quadrature
for very narrow windows.

1. **The explicit per-row censoring flag pending on PR #545.**
   - The current rule decides censoring by exact equality of `y` with its
     bound. That cannot express an interval-censored row, whose response is
     neither end.
   - Equality on derived floats is also fragile. A capped amount that took a
     different route from its bound (rounded to pennies, stored as decimal,
     converted from another currency) need not equal it bitwise, and the row
     then reads as complete without a warning.
   - The flag should be a per-row kind (exact, left, right or interval),
     with the bounds as data.
2. **A row term.** An interval-censored row contributes
   `log P(l < Y ≤ r)` with no density term and a zero carrier. Under
   truncation, that is divided by the truncation window's probability.
3. **Diagnostics.**
   - PIT drawn uniformly on `(F(l), F(r))`, the same randomisation PR #545
     uses for censored rows.
   - The log score.
   - CRPS refused, as for bounded rows today.
4. **Separation.**
   - An interval with two finite ends anchors a location level, because
     `F(r) − F(l)` peaks at a finite location. Only one-sided rows can
     escape.
   - Prove this per family and predictor before relying on it, as PR #545
     did for its scan.
5. **Splice interaction.** An interval that crosses `u` is Reynkens' case v
   (section 3).

### Layer functionals (C12)

**Every bounded layer is finite.** It is an integral of the survival
function over a bounded range, so it is finite for every ξ, including ξ ≥ 1
(the roadmap's existing note). The layer cost is
`Π(M) − Π(M + L)` with `Π(M) = E(X − M)+` (Albrecher and Beirlant 2025,
section 2).

**Wanted per row and per segment:**

- expected layer cost;
- layer variance (finite for a bounded layer at any ξ);
- the probability of reaching the layer;
- expected layer loss count (frequency × exceedance);
- increased-limit factors.

**For the GPD excess law the layer cost is closed form.** For
`0 ≤ a < b` and ξ ∉ {0, 1}:

```
E[min((Y − a)+, b − a)] = σ / (1 − ξ) · [ (1 + ξa/σ)^(1 − 1/ξ) − (1 + ξb/σ)^(1 − 1/ξ) ]
```

with limits `σ log((σ + b)/(σ + a))` at ξ = 1 and
`σ (exp(−a/σ) − exp(−b/σ))` at ξ = 0. Its gradient in (σ, ξ) is analytic. The ξ = 1 branch needs
the same series treatment the kernel already gives ξ → 0.

**Uncertainty is available today for tail-only layers.**
`posterior_bounds(X, callable)` pushes coefficient draws through any
function of the parameter matrix. A GPD layer cost with intervals is
therefore a one-line callable. A layer that straddles `u` also needs the
body's partial integral and joint draws from the three fits (section 3).
The log-normal limited expected value is closed form; the gamma one goes
through the incomplete gamma.

### Export to `aggregate` (C11)

**Licence and packaging.**

- `aggregate` 1.11.0 is BSD-3-Clause: the GitHub `LICENSE` file and the PyPI
  `license_expression`.
- It needs Python ≥ 3.12.
- Its core dependencies include matplotlib, lark, IPython, Pygments and
  greater_tables, so it belongs behind an optional import.

**What it can express.**

- Severities are any scipy continuous distribution with 0, 1 or 2 shape
  parameters, in scipy's shape–scale–location form (Mildenhall 2024). So
  `genpareto`, `lognorm` and `gamma` are all expressible.
- DecL also offers:
  - mixtures with `wts`;
  - shifts (`+ 20`);
  - occurrence layers (`100 xs 10`) and aggregate terms (`aad`,
    `agglimit`);
  - `splice [lb ub]` (DecL reference).
- The release notes say severities can be made "conditional on being in a
  layer (see splice)". The reference page does not define the ends'
  open/closed convention, so verify it by running it before an export
  relies on it.

**Numerics** (Mildenhall 2024).

- The FFT result is checked against the first three analytic moments.
- Aliasing is controlled by padding.
- "When severity is thick-tailed, truncation error is unavoidable", and
  such a severity "should not be normalized".

**Implications for an export.**

- **Moment checks run out as the shape rises.** For a GPD the third moment
  is infinite at ξ ≥ 1/3, the second at ξ ≥ 1/2 and the mean at ξ ≥ 1, so
  the moment checks become unavailable in that order.
- **Export limited or layered severities for heavy tails.** That is also
  what is priced.
- **Export per segment or rating cell, not per row.** The issue-544 note
  estimates tens of seconds for 1,000 cells at the default 2¹⁶ buckets.
- **The FFT complements simulation.** `portfolio()` already simulates book
  totals with parameter uncertainty. Monte Carlo tail error shrinks slowly
  when the variance is infinite, and that is where the FFT pays.

## 5. Heavier tails

### What real books show

Tail index ξ = 1/α, where α is the regular-variation index.

| book | ξ | source (as read) |
|---|---|---|
| marine liability | 0.8–1.1 | Albrecher and Beirlant 2025, Figure 1 (Guevara-Alarcón et al. 2020) |
| general liability (Loss–ALAE, 34 censored) | 0.5–1; splice fit 0.67 | Albrecher and Beirlant 2025, section 4 |
| Danish fire (ME–Pareto splice) | 0.53 | Reynkens et al. 2017, Table 1 |
| motor third-party liability, censored (ME–Pareto) | 0.44 | Reynkens et al. 2017, Table 4 |
| French marine | ≈ 1.1 (α ≈ 0.9) | Chen and Wang 2024, section 3 |
| Lloyd's commercial property | above 1 (α "considerably less than 1") | Chen and Wang 2024, citing Biffis and Chavez 2014 |
| Swiss Solvency Test default large-loss tails | 0.5–1 (α 1–2); 1 for some aircraft | Chen and Wang 2024, citing FINMA 2021 |
| earthquake losses | 0.67–1.67 (α 0.6–1.5) | Chen and Wang 2024, citing Ibragimov et al. 2009 |
| wind catastrophe | ≈ 1.4 (α ≈ 0.7) | Chen and Wang 2024, citing Rizzo 2009 |
| operational risk, Basel data | 0.83–1.43 (α 0.7–1.2) | Chen and Wang 2024, citing Moscadelli 2004 |
| operational risk, public losses by business line | above 1 in "80% of the cases" | Hofert 2014 slides (Chavez-Demoulin, Embrechts and Hofert 2016) |
| cyber | 1.43–1.67 (α 0.6–0.7) | Chen and Wang 2024, citing Eling and Wirfs 2019 |

At the light end, gamma- and Weibull-like excesses fit ξ between −0.06 and
−0.01 at working thresholds (section 1). An upper-truncated tail, such as one
bounded by sum insured, has ξ = −1 near its endpoint (Albrecher and Beirlant
2025, section 3.1).

**So the current walls exclude real books at both ends.** The upper wall
excludes catastrophe, cyber and operational risk, and parts of marine and
liability. The lower wall excludes light-tailed and bounded property excess.

### What the shape changes, and what it does not

- **Estimation is regular above −1/2.** Maximum-likelihood GPD estimation is
  regular for ξ > −1/2, super-efficient but non-regular for −1 < ξ < −1/2,
  and generally has no maximum below −1. This is Smith (1985) as summarised
  by the VGAM `gpd` documentation. Nothing changes for estimation at
  ξ = 1, so the upper wall exists for the prediction, not for the fit (the
  family's own docstring says so).
- **ξ ≥ 1/2: infinite variance.**
  - `variance()` already returns `inf`.
  - Finite-sample maximum-likelihood estimation degrades (Belzile and
    Davison 2026).
  - Monte Carlo totals converge slowly.
- **ξ ≥ 1: infinite mean.**
  - The mean, the expected shortfall and any unbounded excess-of-loss layer
    are infinite.
  - The mean-excess plot is unusable (Chen and Wang 2024, section 4). Klar
    (2023) proposes a moment-free alternative.
  - Diversification can increase risk: VaR becomes asymptotically
    superadditive for iid regularly varying losses with ξ > 1 (Chen and Wang
    2024, section 6.1). That changes how `portfolio()` output should be read.
- **What stays finite:**
  - quantiles, the CDF and the log score;
  - every bounded-layer functional;
  - CRPS while ξ < 2. Its integral `∫ (F(z) − 1{y ≤ z})² dz` has upper tail
    `∫ S(z)² dz`, which converges while 2/ξ > 1. The repository's
    generalized-gamma CRPS rule (finite exactly when σ|Q| < 2) is the same
    statement. Propriety results for CRPS are usually stated on the
    finite-mean class; check that before using CRPS to compare
    infinite-mean fits.

### What a larger shape domain would need

**An upper wall above 1** is mostly functional work, not solver work:

1. **A new default prediction.** The default cannot be the mean when it is
   infinite. Either `predict()` refuses with a message that points to
   quantiles and layer costs, or the default becomes a named quantile. Raw
   `inf` in a price column is not acceptable.
2. **Functionals already in shape.** `generalized_pareto_mean` and the
   expected shortfall already return `inf` for ξ ≥ 1, and named posterior
   bounds already refuse non-finite plug-ins.
3. **CRPS.** Extend its tail corrections to 1 ≤ ξ < 2.
4. **Portfolio wording.** `portfolio()` should report quantiles of totals,
   not means, for rows with infinite mean.

**Negative shapes** are the genuinely hard part.

- The support `y < −σ/ξ` depends on the parameters, so every row imposes a
  nonlinear constraint `1 + ξ y_i / σ_i > 0` on the coefficients. The
  family's docstring already names this as the reason for the lower wall.
- Options:
  - a wall at a small negative value with feasibility-preserving steps;
  - a reparameterisation through the endpoint;
  - an upper-truncated Pareto for physically bounded losses (Beirlant,
    Fraga Alves and Gomes 2016, via Albrecher and Beirlant).

  Each needs its own regularity and inference story below ξ = −1/2.

### Measured: what the walls do

**Set-up.**

- Six simulated excess samples of 4,000 rows each (seed 20261031).
- The model is `SuperLSS(GeneralizedParetoLSS(), scale(s("x", kind="cr",
  k=6)), shape())` with `fit_reml` defaults, where `x` is noise.
- The reference is `scipy.stats.genpareto.fit(y, floc=0)`, which leaves the
  shape unrestricted.
- Run on this branch's base `72c60b21`, one thread, through
  `benchboard heavy pot_sweep-gpd_walls --cores 1`; peak RSS 349 MiB.

| excess sample | scipy ξ̂ | superglm ξ̂ | stop, certified | inner iterations | wall s |
|---|---|---|---|---|---|
| exponential (ξ = 0) | −0.0165 | 2.2e-14 (lower wall) | `lambda_change`, yes | 24 | 3.12 (first fit, includes compilation) |
| gamma(2) above its 90% point | −0.0257 | 0.0000 (lower wall) | `lambda_change`, yes | 35 | 0.17 |
| lognormal above its 90% point | 0.2593 | 0.2593 | `practical_plateau`, no | 13 | 0.09 |
| GPD ξ = 0.3 | 0.2789 | 0.2789 | `practical_plateau`, no | 18 | 0.15 |
| GPD ξ = 0.8 | 0.7511 | 0.7509 | `practical_plateau`, no | 16 | 0.11 |
| GPD ξ = 1.2 | 1.2189 | 1 − 1.1e-16 (upper wall) | `practical_plateau`, no | 42 | 0.21 |

**Inside the walls, superglm agrees with scipy to 2e-4 or better.**

**At the walls the fit is pinned, and nothing tells the user.**

- **Upper wall (ξ = 1.2).**
  - `predict()` returns about 1.03e16 per row, which is `σ/(1 − ξ)` with
    `1 − ξ` equal to the unit roundoff, 1.1e-16. No warning is raised.
  - The 0.995-quantile posterior band, [216, 240], looks ordinary and hides
    the wall.
  - This is a silently wrong number in something a user reads. AGENTS.md
    treats that as always required to prevent, so it is listed below as a
    follow-up to fix, independent of any new capability. A fix is on branch
    `claude/gpd-wall-disclosure` (`fa5daa3d`), unmerged.
- **Lower wall (exponential).**
  - The only signal is `GeneralizedParetoInitializationWarning`, which says
    the method-of-moments start is unusable and the fit starts from 0.25.
  - `smoothing_certified_` is `True`, and `exact_face_components_` is empty:
    the wall is not treated as a face.
  - The posterior band on the 0.995 quantile is [5.30, 215.8] around an
    estimate of 5.54 (truth 5.30). On the logit scale the wall has
    near-zero curvature, so coefficient draws spread across the whole shape
    range.
- **The earlier PR #545 measurement fits the same picture.** A Pareto excess
  capped at a limit and read as complete fitted a shape of 0.0, not 0.30.
  The pinned fit took 167 inner iterations against 29 for the censored fit,
  and 5.59 s against 1.49 s.

## 6. What superglm already does well for large losses

- **GPD regression with smooth scale and shape.** Smoothness is chosen by
  LAML with analytic derivatives to fourth order: the evgam method, in
  Python, with no GPL dependency. Inside the walls it matches scipy to 2e-4
  in 0.1–0.2 s at 4,000 rows.
- **Per-row truncation and censoring in a multi-predictor LAML fit**
  (PR #545). The issue-544 sweep found no other documented software with
  that combination.
- **Exact splice likelihood at a fixed threshold.** With the truncated body
  from PR #545, the three-fit recipe is the exact maximum-likelihood splice,
  and the docs say why a complete-row body is wrong.
- **Posterior draws for any derived quantity.** `posterior_bounds` accepts
  any callable of the fitted parameters. Layer costs with intervals for the
  tail model take one line today.
- **Tail functionals and scores.** Closed-form `predict_cdf` and
  `predict_quantile`, certified expected shortfall, and threshold-weighted
  CRPS with an explicit GPD tail correction. `calibration.tails` compares
  expected with realised exceedance counts. Censored rows get a randomised
  PIT (PR #545).
- **Censoring separation scan.** It names levels whose every claim hit a
  limit, before the fit (PR #545).
- **Book totals.** `portfolio()` simulates totals with parameter
  uncertainty, by segment.
- **Weight honesty.** Frequency weights replicate excess rows. Prior weights
  are refused because the GPD has no averaging law.

## 7. Staged proposal

Each stage names its user-visible payoff, what it depends on and the gate
that promotes it. The stages run in order unless marked independent.

**Stage 0, in flight: PR #545 and the explicit per-row censoring flag.**

- **Payoff:** truncated bodies, censored tails and correct PITs.
- **Depends on:** the owner's decision on the hold.
- **Gate:** already defined on the PR.

**Stage A: disclose the walls, and add threshold diagnostics.**
Independent of Stage 0.

- **Payoff.**
  - A fit whose shape sits at a wall says so, in a warning and in the fit
    record. `predict()` never returns a 1e16 mean silently.
  - Users can choose `u` from a mean-residual-life plot and
    parameter-stability plots (shape and modified scale `σ − ξu` with
    profile intervals), drawn from a loop of intercept-only fits over a
    quantile grid.
  - For the layer method (section 8), two more checks at each candidate `u`:
    - a stability plot of each priced quantity, with intervals;
    - the body's predicted exceedance count `Σ S_B(u | x_i)` against the
      observed count.
- **Depends on:** nothing new. The stability plot's shape track is honest
  only down to 0 until Stage F. It should mark the wall, not hide it.
- **Gate.**
  - A forced-wall regression test on both walls that fails on the unfixed
    code.
  - Stability-plot point estimates and intervals agree with a black-box
    oracle (`threshr` or `ismev`, both GPL-2+, run as oracles with pinned
    outputs) on a public data set.
  - Complete-fit wall time and predictions inside the walls are unchanged.

**Stage B: a splice object.**

- **Payoff.** One fitted whole-range severity built from the three fits,
  with:
  - `predict_cdf`, `predict_quantile`, survival and simulation;
  - the spliced mean where finite;
  - posterior intervals from block-diagonal joint draws;
  - a held-out log score and PIT over all rows, so a spliced model can be
    compared with a single-family fit;
  - both constructions: the layer method (body censored at `u`) and the
    split method (body truncated at `u`). The exceedance probability comes
    from the body's survival or from a separate model; in the second case
    the body is renormalised below `u`, so the density stays proper
    (section 8);
  - the whole-density and censored log scores used in section 8.
- **Depends on:** Stage 0. Log-normal and Gaussian bodies work now; gamma
  bodies wait for issue #544 stage 2.
- **Gate.**
  - Agrees with a direct scipy maximum-likelihood fit of the spliced
    likelihood on simulated data.
  - On freMTPL2sev and on Max's book, the held-out log score and
    threshold-weighted CRPS above `u` are no worse than the best single
    family.
  - Actual against expected exceedance counts are reported by segment.

**Stage C: layer functionals (C12).**

- **Payoff.** Expected layer cost, layer variance, probability of reaching
  the layer and increased-limit curves, per row or segment, with intervals,
  for tail-only and straddling layers.
- **Depends on:** Stage B for layers that straddle `u`. Tail-only layers can
  ship first.
- **Gate.**
  - Closed forms agree with quadrature.
  - Interval coverage is checked by simulation from a known law, against a
    stated tolerance.
  - Bounded layers are finite for every shape; unbounded layers refuse at
    ξ ≥ 1 with a message.
  - The roadmap's C12 gate: a concrete quantity not served today, its
    existence checked, its calibration checked.

**Stage D: interval censoring.**

- **Payoff.** Claims seen only between non-contiguous layers, or bounded by
  an incurred amount, or rounded, fit as intervals instead of being dropped
  or imputed.
- **Depends on:** Stage 0's explicit flag.
- **Gate.**
  - Agrees with scipy maximum likelihood.
  - PIT is uniform on simulated gapped data.
  - The separation rule is proven per family.
  - Rows that straddle `u` are refused by the three-fit splice, with a
    message pointing to Stage G.

**Stage E: export to `aggregate` (C11).**

- **Payoff.** Deterministic FFT aggregate distributions per segment, for
  pricing and capital on layered terms.
- **Depends on:** Stages B and C, and an optional import.
- **Gate.**
  - `aggregate` quantiles agree with `portfolio()` simulation within Monte
    Carlo error.
  - Layered exports at ξ ≥ 1/2 match the analytic layer moments.
  - The open/closed convention of `splice` is verified by running it.

**Stage F: widen the shape domain.**

- **Payoff.** Catastrophe, cyber, operational-risk and heavy liability books
  fit their actual tail.
- **Plan.** First an upper wall above 1, with a non-mean default prediction.
  Then negative shapes down to −1/2.
- **Depends on:** Stage C, because layer costs are the finite quantities
  those books price, and Stage A's wall disclosure.
- **Gate.**
  - A real or public book whose shape estimate sits at a wall, where moving
    the wall changes a layer price or a held-out tail score.
  - The existence of every functional is checked at the new shapes.
  - Negative shapes also need feasible steps and an inference contract
    below −1/2, scoped separately.

**Stage G: joint threshold (C13).**

- **Payoff.** The threshold is chosen, and its uncertainty carried, by a
  rule that compares thresholds on the same rows.
  - First, a K-fold scan over the candidate thresholds. It reports the
    whole-density log score and the censored log score at the highest
    candidate, each with paired standard errors, beside the
    decision-quantity stability plot (section 8). It reuses existing fits:
    about candidates × folds × 2 fits, roughly a minute for 10 × 5 at
    50,000 rows.
  - Then averaging over thresholds with weights proportional to the
    exponentiated held-out score (Northrop, Attalides and Jonathan 2017).
  - An extended-GPD family follows only if thresholds prove unstable.
- **Depends on:** Stage B.
- **Gate:** the roadmap's existing C13 gate. Threshold uncertainty must
  change a held-out decision (a layer price interval or a tail score) on at
  least two books.

**Stage H: covariate threshold `u(x)`.**

- **Payoff.** A threshold at a fixed conditional quantile, with a constant
  exceedance probability, for lines where size scales with exposure.
- **Depends on:** a calibrated quantile family (C7 research, qgam-style)
  and Stage B.
- **Gate:** the held-out tail score beats a fixed monetary `u` on a real
  book.

### Roadmap changes this note suggests

- **The GPD shape note.** Add that published catastrophe, cyber,
  operational-risk and some marine books have ξ ≥ 1, and that the current
  walls pin such fits without a warning (section 5).
- **C12.** Name Stage C's quantities as the concrete targets.
- **C13.** Prefer a threshold profile or average over a grid to estimating
  `u` as a parameter. Name the extended GPD as the threshold-free
  alternative.
- **C11.** Record `aggregate` as the established engine, behind an optional
  import, exported per segment.

## 8. Threshold choice for the layer method

Max has chosen the layer method (2026-10-31). It differs from the split
recipe of section 3 in how the body is fitted and where `P(Y > u | x)` comes
from:

| | layer method | split method (section 3) |
|---|---|---|
| body | every claim, as `min(y, u)`, right-censored at `u` | claims at or below `u`, truncated at `u` |
| tail | GPD on `y − u` above `u` | the same |
| `P(Y > u \| x)` | the body's own survival `S_B(u \| x)`, or a separate exceedance model | a separate exceedance model `p(x)` |
| whole-range density | `f_B(y)` below `u`, `S_B(u) g(y − u)` above | `(1 − p) f_B(y) / F_B(u)` below, `p g(y − u)` above |

**The density is proper only in some combinations.** With the body's own
survival, the layer density integrates to `F_B(u) + S_B(u) = 1` for every
`u`. With a separate exceedance model, `f_B(y)` below and `p(x) g(y − u)`
above integrate to `F_B(u) + p(x)`, which is not one. The body must then be
renormalised below `u` to `(1 − p(x)) f_B(y) / F_B(u)` before any score is
compared across thresholds.

**superglm with PR #545 fits the layer method directly:**

- the body is `LogNormalLS` or `GaussianLS` on `np.minimum(y, u)` with
  `censoring=(None, u)`;
- the tail is `GeneralizedParetoLSS` on `y − u` above `u`;
- the exceedance probability is `1 − body.predict_cdf(X, u)`.

**The whole-density log score needs nothing new.** It is the body's censored
log score (`log f_B(y)` below `u`, `log S_B(u)` for a censored row), which
PR #545's `scores(..., which=("log",))` returns per row, plus the GPD's log
score on the exceedances.

**Why `u` is a joint choice.** It sets how much of the range the body has to
describe and how much data the tail gets, so it cannot be chosen from the
tail alone. Each part of Max's concern is covered below.

### 8.1 Cross-validated comparison and averaging over thresholds

Northrop, Attalides and Jonathan (2017), read in full (arXiv:1504.06653).

**What they do** (sections 2.1–2.2):

- At each candidate `u` they fit a binomial–GPD model: the exceedance
  probability, the GPD scale and the shape. Below `u` the model is just
  `1 − p_u`.
- Each held-out observation is scored by the model's implied law above a
  fixed validation threshold `v`: `1 − p_v` for an observation at or below
  `v`, and `p_v g(x − v)` above, with `σ_v = σ_u + ξ(v − u)` (their equation
  3).
- The sum `T_v(u)` of leave-one-out log predictive densities estimates, up
  to a constant, the negated Kullback–Leibler divergence at `v`. The best
  `u` maximises it.
- They set `v` to the highest candidate: a higher `v` loses validation
  information and gains nothing.

**Averaging** (section 3):

- Each threshold gets the weight `exp(T_v(u_i))`, normalised over the
  candidates. These are pseudo-Bayes factors (Geisser and Eddy 1979), with
  a uniform prior on candidates equally spaced in probability.
- Predictions are averaged with those weights (their equations 15–17).
- On their Gulf of Mexico data, averaging protected against the implausible
  extrapolations of single high thresholds.

**What it assumes:**

- Independent, identically distributed data. Covariates are named as
  ongoing work.
- A Bayesian fit, with leave-one-out by importance sampling.
- A reference prior, or a weakly informative one: a Cauchy prior on the
  shape truncated to ξ > −1 (their section 3.3).

The results are sensitive to `v` and to the candidate set (their section 2.5;
Murphy, Tawn and Varty 2025 report the same).

**What it judges: the tail only.** The body never enters. Their criterion is
the censored likelihood score at `v` (section 8.2). `threshr`, which
implements it, is GPL (≥ 2), so it is usable only as a black-box oracle; its
source was not opened.

**Where superglm stands.** It is outside the method as written, which is
Bayesian and iid. The frequentist analogue is a loop of existing fits: K-fold
refits at each `u`, then held-out scores from `predict_cdf` and the log
density.

### 8.2 Is the held-out log score of the whole layer density a sound criterion?

**It is valid.**

- The logarithmic score is a proper scoring rule (Allen 2023, section 2.1).
- Every candidate's layer density, built on the body's survival, is a
  proper density on the same outcome space. Held-out log scores on the same
  rows therefore compare the candidates on equal terms.
- This is the "extended model" route that Belzile and Davison (2026, section
  8.1.1) say makes a threshold selectable by likelihood-based criteria. Fits
  above different `u` alone cannot be compared, because their samples
  differ.
- **Covariates change nothing in principle.** Each held-out row is scored
  under its own conditional density, and the mean over rows compares the
  conditional models. Everything, including smoothing selection, is refit
  inside each fold. The threshold may be a constant or `u(x)`.

**It answers a different question from the tail's.**

- It asks which `u` gives the best whole-range law. Most rows lie below `u`,
  so the body dominates the score.
- In the simulation below it picks the lowest candidate, at 3–5 standard
  errors. The cause is that the log-normal body misfits the Burr body, so
  handing more of the range to the GPD improves the body's share of the
  score. The tail is not better there.

**The tail-focused alternative is the censored likelihood score at `v`.**

- It is `log f(y)` for `y > v` and `log F(v)` for `y ≤ v` (Diks, Panchenko
  and van Dijk 2011, as defined in Allen 2023, section 2.2).
- It is proper, and it scores both the tail and the probability of reaching
  it.
- It is exactly Northrop et al.'s criterion.
- Its weakness is power: only the rows above `v` discriminate. In the
  simulation the thresholds differ by 1–2 standard errors on it.

**Recommendation.**

- Report both scores, paired, with standard errors of the differences.
  superglm's `compare` already reports paired standard errors for its
  scores.
- Choose by the score that matches the decision: the whole-density score
  for a whole-range severity, and the censored score at a `v` near the
  attachment or limit for layer pricing.
- Never compare thresholds by scoring only the claims above each `u` with
  an unweighted score: the samples differ across `u`.

### 8.3 Checking the body below `u` when it is censored at `u`

**What the censored body claims.** It describes its law below `u` and the
mass `S_B(u)` above. It says nothing about the shape above `u`, which is the
GPD's job, so the checks target those two parts.

**Below `u`:**

- **Randomised quantile residuals.** Each censored row's PIT is drawn
  uniformly on `(F_B(u), 1)` (Dunn and Smyth 1996, as implemented by PR
  #545's `residual_set` and diagnostic plots). Read the Q-Q and worm plots
  of the rows below `u`, and the residuals against each covariate.
- **Without covariates.** Censoring at a single fixed `u` makes the
  Kaplan–Meier estimate below `u` equal to the empirical distribution
  function. A plain P-P or Q-Q plot of the rows below `u` against the
  fitted body is then the nonparametric check. Reynkens et al. (2017) and
  Albrecher and Beirlant (2025) use Kaplan–Meier and Turnbull versions for
  general censoring.

**At `u`.** The layer method uses `S_B(u | x)` as the exceedance probability,
so a body misfit near `u` moves tail mass.

- Compare `Σ S_B(u | x_i)` with the observed count above `u`, overall and
  by segment. `calibration.tails` does exactly this for any threshold.
- In the simulation, the body's own survival overstated the exceedance
  share by 2–11%, depending on `u`.
- A separate exceedance model removes that dependence, at the cost of
  renormalising the body.

### 8.4 How sensitive the decision quantities are to `u`

**The practitioner's check is a decision-quantity stability plot.** Plot
each priced quantity against `u`, with intervals:

- the capped mean `E[min(Y, L)]`;
- the excess cost per claim `E[(Y − M)+]`;
- a high quantile.

It is the decision version of the parameter-stability plot. Northrop et al.
(2017, Figure 7) plot return levels against the training threshold in the
same way. Belzile and Davison (2026, section 1) note that the best threshold
can depend on the quantity being estimated.

**What moved in the simulation:**

- **Quantities the body sets barely moved.** A capped mean with its limit at
  the 95% point stayed within 2% of the truth (0.984–1.014) across every
  candidate and both variants.
- **Quantities the tail sets moved most.** The excess cost per claim above
  the 99% point ranged from 0.83 to 1.24 of the truth with the body's
  survival, and from 0.81 to 1.19 with the empirical exceedance share.
- **The 99.5% quantile moved less:** 0.98–1.07.
- **The fitted shape never settled.** It rose steadily from 0.30 to 0.49
  across the candidates (the Burr's limit is 0.4). A parameter-stability
  plot would not have settled the choice either.

**Neither score picked the threshold that priced the excess best.**

- The excess cost was closest to the truth with `u` between the 70% and 85%
  points.
- The whole-density score preferred the 50% point; the censored score
  preferred the 95–97% points.
- So price from a stable stretch of the decision plot, or average over
  thresholds, rather than from the single best-scoring `u`.

### Measured: a layer-method threshold scan

**Set-up.**

- **Data.** 20,000 Burr XII losses (`c = 2`, `d = 1.25`, scale 1,000), so
  the tail limit is ξ = 0.4. The body is deliberately misspecified as
  log-normal.
- **Fits.** At each candidate `u`, both by scipy maximum likelihood,
  independent of superglm:
  - the log-normal body on `min(y, u)`, right-censored at `u`;
  - a GPD on `y − u` above `u`.
- **Scores.** Five-fold cross-validated mean held-out log score per row
  (higher is better), shown as the difference from the best candidate with
  its paired standard error. The censored score is taken at `v` = the 98%
  point.
- **Decision quantities.** Fitted on all rows, as ratios to the Burr truth:
  - the capped mean at `L` = the 95% point (truth 1,085.4);
  - the excess cost per claim over `M` = the 99% point (truth 42.29);
  - the 99.5% quantile (truth 8,265).
- **Run** through `benchboard heavy pot_sweep-layer_threshold --cores 1`.

Exceedance from the body's own survival `S_B(u)`:

| `u` (point) | whole-density log score | censored score at `v` | ξ̂ | `S_B(u)` / observed share | capped mean | excess cost | q99.5 |
|---|---|---|---|---|---|---|---|
| 50% | best | −0.00029 ± 0.00018 | 0.302 | 1.024 | 1.008 | 0.825 | 0.986 |
| 60% | −0.00223 ± 0.00060 | −0.00026 ± 0.00016 | 0.316 | 1.040 | 1.009 | 0.892 | 1.005 |
| 70% | −0.00337 ± 0.00073 | −0.00023 ± 0.00014 | 0.334 | 1.060 | 1.010 | 0.972 | 1.027 |
| 80% | −0.00413 ± 0.00078 | −0.00025 ± 0.00013 | 0.351 | 1.086 | 1.011 | 1.050 | 1.047 |
| 85% | −0.00413 ± 0.00079 | −0.00024 ± 0.00012 | 0.364 | 1.097 | 1.011 | 1.095 | 1.056 |
| 90% | −0.00406 ± 0.00079 | −0.00020 ± 0.00011 | 0.385 | 1.109 | 1.012 | 1.160 | 1.067 |
| 92.5% | −0.00405 ± 0.00079 | −0.00016 ± 0.00009 | 0.400 | 1.107 | 1.012 | 1.186 | 1.067 |
| 95% | −0.00406 ± 0.00080 | −0.00006 ± 0.00006 | 0.434 | 1.091 | 1.013 | 1.225 | 1.059 |
| 97% | −0.00411 ± 0.00080 | best | 0.475 | 1.052 | 1.013 | 1.236 | 1.035 |
| 98% | −0.00415 ± 0.00080 | −0.000004 ± 0.00004 | 0.488 | 1.015 | 1.014 | 1.209 | 1.017 |

Exceedance from the observed share, with the body renormalised below `u` (the
fitted shapes are the same as above):

| `u` (point) | whole-density log score | censored score at `v` | capped mean | excess cost | q99.5 |
|---|---|---|---|---|---|
| 50% | best | −0.00029 ± 0.00017 | 0.995 | 0.806 | 0.978 |
| 60% | −0.00198 ± 0.00049 | −0.00022 ± 0.00015 | 0.989 | 0.857 | 0.991 |
| 70% | −0.00292 ± 0.00060 | −0.00015 ± 0.00011 | 0.986 | 0.917 | 1.004 |
| 80% | −0.00353 ± 0.00065 | −0.00011 ± 0.00009 | 0.984 | 0.967 | 1.014 |
| 85% | −0.00364 ± 0.00068 | −0.00009 ± 0.00007 | 0.986 | 0.998 | 1.018 |
| 90% | −0.00376 ± 0.00071 | −0.00004 ± 0.00005 | 0.991 | 1.046 | 1.023 |
| 92.5% | −0.00392 ± 0.00072 | −0.00003 ± 0.00003 | 0.996 | 1.071 | 1.023 |
| 95% | −0.00415 ± 0.00073 | best | 1.003 | 1.123 | 1.020 |
| 97% | −0.00436 ± 0.00074 | −0.000001 ± 0.00003 | 1.010 | 1.175 | 1.013 |
| 98% | −0.00442 ± 0.00074 | −0.00003 ± 0.00005 | 1.013 | 1.191 | 1.010 |

This is one law, one seed and one sample size: an illustration of how the
criteria behave, not a simulation study. A real-book version needs PR #545
for the censored body. The script is in the second appendix.

## Licences checked (2026-10-31)

| package | licence (source) | use here |
|---|---|---|
| aggregate 1.11.0 | BSD-3-Clause (GitHub `LICENSE`; PyPI `license_expression`) | read docs; may be adapted and depended on |
| scipy | BSD-3-Clause | oracle and reference |
| pyextremes 2.5.0 | MIT (PyPI classifier) | may be read for diagnostics plots |
| evgam | GPL-3 (CRAN `DESCRIPTION`) | papers and docs only; black-box oracle |
| ReIns | GPL (≥ 2) | papers only; black-box oracle |
| threshr, ismev, QRM | GPL (≥ 2) | papers only; black-box oracle |
| evmix, mev | GPL-3 | papers only |
| qgam | GPL-2 \| GPL-3 | papers only |
| gamlss.cens, gamlss.tr | GPL-2 \| GPL-3 | documentation only (issue-544 note) |

No GPL source was opened.

## Sources read

Papers and documentation read in this sweep. Works marked "via" were not
read directly; their claims come from the named source.

- Allen, S. (2023). Weighted scoringRules: emphasising particular outcomes
  when evaluating probabilistic forecasts. arXiv:2305.07312. Via it: Diks,
  Panchenko and van Dijk (2011), *J. Econometrics* 163(2), 215–230 (the
  censored and conditional likelihood scores); Holzmann and Klar (2017).
- Albrecher, H. and Beirlant, J. (2025). Statistics of extremes for the
  insurance industry. arXiv:2511.22272 (to appear in *Handbook of Statistics
  of Extremes*). Via it: Guevara-Alarcón et al. (2020); Stupfler (2019);
  Beirlant, Fraga Alves and Gomes (2016).
- Belzile, L. R. and Davison, A. C. (2026). Choosing the threshold in
  extreme value analysis. arXiv:2606.28540. Via it: Northrop and Jonathan
  (2011); Northrop, Attalides and Jonathan (2017); Northrop and Coleman
  (2014); Wadsworth (2016); Collings et al. (2025); Varty et al. (2021);
  Fasiolo et al. (2021); MacDonald et al. (2011); Naveau et al. (2016).
- Chavez-Demoulin, V. and Davison, A. C. (2005). Generalized additive
  modelling of sample extremes. *JRSS C* 54(1), 207–222,
  doi:10.1111/j.1467-9876.2005.00479.x: abstract, plus Chavez-Demoulin's
  slides "Generalized Additive Modelling for Sample Extremes: An
  Environmental Example" (Tokyo, March 2007).
- Chavez-Demoulin, V., Embrechts, P. and Hofert, M. (2016). An extreme value
  approach for modeling operational risk losses depending on covariates.
  *J. Risk Insur.* 83(3), 735–776, doi:10.1111/jori.12059: through Hofert's
  slides (WU Vienna research seminar, 2014-03-20). Via them: Cox and Reid
  (1987).
- Chen, Y. and Wang, R. (2024). Infinite-mean models in risk management:
  discussions and recent advances. arXiv:2408.08678. Via it: Moscadelli
  (2004); Eling and Wirfs (2019); Ibragimov et al. (2009); Rizzo (2009);
  Biffis and Chavez (2014); FINMA (2021); Klar (2023).
- Fissler, T., Merz, M. and Wüthrich, M. V. (2021). Deep quantile and deep
  composite model regression. arXiv:2112.03075.
- Le Carrer, N. and Gaetan, C. (2022). Distributional regression models for
  extended generalized Pareto distributions. arXiv:2209.04660. Via it: Evin
  et al. (2018).
- Mildenhall, S. (2024). Aggregate: fast, accurate, and flexible
  approximation of compound probability distributions. *Annals of Actuarial
  Science* 19(2), doi:10.1017/S1748499524000216. Also the `aggregate` DecL
  reference (aggregate.readthedocs.io) and its PyPI release notes.
- Murphy, C., Tawn, J. A. and Varty, Z. (2025). Automated threshold
  selection and associated inference uncertainty for univariate extremes.
  *Technometrics* 67, 215–224; arXiv:2310.17999.
- Northrop, P. J., Attalides, N. and Jonathan, P. (2017). Cross-validatory
  extreme value threshold selection and uncertainty with application to
  ocean storm severity. *JRSS C* 66(1), 93–120, doi:10.1111/rssc.12159;
  arXiv:1504.06653. Via it: Geisser and Eddy (1979).
- Reynkens, T., Verbelen, R., Beirlant, J. and Antonio, K. (2017).
  Modelling censored losses using splicing: a global fit strategy with mixed
  Erlang and extreme value distributions. *IME* 77, 65–77;
  arXiv:1608.01566.
- Youngman, B. D. (2019). Generalized additive models for exceedances of
  high thresholds with an application to return level estimation for U.S.
  wind gusts. *JASA* 114(528), 1865–1879,
  doi:10.1080/01621459.2018.1529596: abstract only.
- Youngman, B. D. (2022). evgam: an R package for generalized additive
  extreme value models. *JSS* 103(3), doi:10.18637/jss.v103.i03;
  arXiv:2003.04067.
- VGAM `gpd` help page, for its summary of Smith (1985), *Biometrika* 72,
  67–90.
- The issue-544 assessment on PR #545
  (`notes/research/2026-10-issue-544-truncated-censored-assessment.md` on
  that branch). Via it: Fung, Tzougas and Wüthrich (2021), arXiv:2103.07200.
  Via PR #545's description: Dunn and Smyth (1996), randomised quantile
  residuals.

## Appendix: the wall experiment

The script is shown as run, from a scratch directory, through
`benchboard heavy pot_sweep-gpd_walls --cores 1 -- timeout 300 uv run python
<script>`. A second run printed the fitted shape to full precision, the first
three `predict()` values, the 0.995-quantile `posterior_bounds` with 200
draws, and the warning text, for the exponential and ξ = 1.2 cases.

```python
import numpy as np, pandas as pd
from scipy import stats
from superglm import GeneralizedParetoLSS, SuperLSS, s

rng = np.random.default_rng(20261031)
n = 4000

def excess_above_90(sample):
    u = np.quantile(sample, 0.9)
    return sample[sample > u] - u

cases = {
    "exponential (xi=0)": rng.exponential(1.0, n),
    "gamma(2) above its 90%": excess_above_90(stats.gamma(2.0).rvs(10 * n, random_state=rng)),
    "lognormal above its 90%": excess_above_90(stats.lognorm(1.0).rvs(10 * n, random_state=rng)),
    "GPD xi=0.3": stats.genpareto(0.3).rvs(n, random_state=rng),
    "GPD xi=0.8": stats.genpareto(0.8).rvs(n, random_state=rng),
    "GPD xi=1.2": stats.genpareto(1.2).rvs(n, random_state=rng),
}
for name, y in cases.items():
    X = pd.DataFrame({"x": rng.uniform(size=len(y))})
    xi_scipy, _, _ = stats.genpareto.fit(y, floc=0.0)
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(s("x", kind="cr", k=6)), family.shape())
    model.fit_reml(X, y)
    xi_fit = np.asarray(model.predict_parameters(X.iloc[:1]))[0, 1]
    print(name, xi_scipy, xi_fit, model.smoothing_convergence_reason_,
          model.smoothing_certified_, model.result_.n_inner_iter)
```

## Appendix: the layer-method threshold scan

Run from a scratch directory through `benchboard heavy pot_sweep-layer_threshold --cores 1 --
timeout 600 uv run python <script>`; it uses numpy and scipy only.

```python
"""Threshold choice for the layer method, by scipy maximum likelihood (independent of superglm).

Truth: Burr XII ground-up losses, tail shape 1/(c d) = 0.4. Layer method at threshold u:
log-normal body fitted to min(y, u) right-censored at u; GPD fitted to y - u above u;
P(Y > u) from the body's own survival. The whole-range density is f_B below u and
S_B(u) g(y - u) above, proper for every u. Five-fold CV compares thresholds by
(a) the whole-density log score and (b) the censored likelihood score at v, the highest
candidate (Diks, Panchenko and van Dijk 2011; Northrop, Attalides and Jonathan 2017).
"""
import numpy as np
from scipy import integrate, optimize, stats

truth = stats.burr12(c=2.0, d=1.25, scale=1000.0)
rng = np.random.default_rng(20261031)
n = 20_000
y_all = truth.rvs(n, random_state=rng)
levels = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98)
limit, retention, p_high = truth.ppf(0.95), truth.ppf(0.99), 0.995
true_capped = integrate.quad(truth.sf, 0, limit, limit=200)[0]
true_excess = integrate.quad(truth.sf, retention, np.inf, limit=200)[0]
true_q = truth.ppf(p_high)


def fit_layer(y, u, separate_frequency=False):
    below, above = y[y <= u], y[y > u]
    def nll(par):
        mu, log_s = par
        s = np.exp(log_s)
        return -(stats.norm.logpdf(np.log(below), mu, s).sum() - np.log(below).sum()
                 + len(above) * stats.norm.logsf(np.log(u), mu, s))
    start = (np.log(below).mean(), np.log(np.log(below).std()))
    mu, log_s = optimize.minimize(nll, start, method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000}).x
    body = stats.lognorm(np.exp(log_s), scale=np.exp(mu))
    xi, _, sigma = stats.genpareto.fit(above - u, floc=0.0)
    p_u = len(above) / len(y) if separate_frequency else float(body.sf(u))
    return u, body, stats.genpareto(xi, scale=sigma), p_u


def logpdf(model, y):
    # below u the body is renormalised to carry 1 - p_u; with p_u = S_B(u) that is the body itself
    u, body, tail, p_u = model
    out = body.logpdf(y) + np.log1p(-p_u) - body.logcdf(u)
    hi = y > u
    out[hi] = np.log(p_u) + tail.logpdf(y[hi] - u)
    return out


def logcdf_at(model, v):
    u, body, tail, p_u = model
    if v > u:
        return np.log1p(-p_u * tail.sf(v - u))
    return np.log1p(-p_u) + body.logcdf(v) - body.logcdf(u)


def sf(model, z):
    u, body, tail, p_u = model
    z = np.asarray(z, dtype=float)
    below = 1.0 - (1.0 - p_u) * body.cdf(z) / body.cdf(u)
    return np.where(z <= u, below, p_u * tail.sf(z - u))


def decisions(model):
    # every retention and quantile below sits above every candidate u
    u, body, tail, p_u = model
    capped = integrate.quad(lambda z: float(sf(model, z)), 0, limit, points=[u] if u < limit else None, limit=200)[0]
    excess = p_u * integrate.quad(tail.sf, retention - u, np.inf, limit=200)[0]
    q = u + tail.isf((1.0 - p_high) / p_u)
    return capped / true_capped, excess / true_excess, q / true_q


folds = rng.permutation(n) % 5
thresholds = {q: np.quantile(y_all, q) for q in levels}
v = thresholds[levels[-1]]
print(f"truth: capped mean at L=q95 {true_capped:.1f}, excess per claim over M=q99 {true_excess:.2f}, q99.5 {true_q:.0f}")
for separate in (False, True):
    print("P(Y>u) from", "the empirical exceedance share, body renormalised below u" if separate else "the body's own survival")
    rows, per_row = [], {}
    for q, u in thresholds.items():
        logs_rows, cels_rows = np.empty(n), np.empty(n)
        for k in range(5):
            test_idx = np.flatnonzero(folds == k)
            model = fit_layer(y_all[folds != k], u, separate)
            test = y_all[test_idx]
            logs_rows[test_idx] = logpdf(model, test)
            hi = test > v
            c = np.full(len(test), logcdf_at(model, v))
            c[hi] = logpdf(model, test[hi])
            cels_rows[test_idx] = c
        per_row[q] = (logs_rows, cels_rows)
        model = fit_layer(y_all, u, separate)
        rows.append((q, u, model[2].args[0], model[3] / np.mean(y_all > u), *decisions(model)))
    best = {j: max(per_row, key=lambda q: per_row[q][j].mean()) for j in (0, 1)}
    print(f"{'u level':>8} {'u':>6} {'dlogS':>8} {'se':>7} {'dCeLS':>9} {'se':>8} {'xi':>6} {'p_u/emp':>7} {'capped':>7} {'excess':>7} {'q99.5':>7}")
    for q, u, xi, ratio, c, e, qq in rows:
        out = []
        for j in (0, 1):
            d = per_row[q][j] - per_row[best[j]][j]
            out += [d.mean(), d.std(ddof=1) / np.sqrt(n)]
        print(f"{q:8.3f} {u:6.0f} {out[0]:8.5f} {out[1]:7.5f} {out[2]:9.6f} {out[3]:8.6f} {xi:6.3f} {ratio:7.3f} {c:7.3f} {e:7.3f} {qq:7.3f}")
```
