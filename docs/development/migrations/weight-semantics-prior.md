# Migration: `sample_weight` is now read as an EDM prior weight

*Refs issue #349. Ships with the first release containing `weight_semantics`.*

## What changed

What `sample_weight` says about a row used to be decided by the family:
Tweedie read it as an EDM prior weight, every other family read it as a
frequency weight, and there was no way to say otherwise. It is now a declared
parameter, and the default for every family is the prior reading:

```python
SuperGLM(..., weight_semantics="prior")      # new default, every family
SuperGLM(..., weight_semantics="frequency")  # pre-change behaviour, non-Tweedie
```

- **`"prior"`** — `w_i` is a precision: `Var(Y_i) = phi V(mu_i) / w_i`, and row
  `i` contributes `log f(y_i; mu_i, phi / w_i)`.
- **`"frequency"`** — `w_i` is a replication count: row `i` contributes
  `w_i log f(y_i; mu_i, phi)` and an integer weight is exactly a repeated row.

`"frequency"` reproduces the previous non-Tweedie behaviour and `"prior"`
reproduces the previous Tweedie behaviour, so neither contract is new — only
which one you get by default, and the fact that you can now choose.

The contract reaches six seams, all reading one resolved value:

1. `solvers/dispersion.py` — the likelihood size and the residual degrees of
   freedom, and through them Pearson dispersion, Wald standard errors and
   intervals, the effective `n` in AIC/BIC/AICc, and screening.
2. `reml/scale.py` — the Gaussian, Gamma and Tweedie REML scale profilers,
   and through them `lambda`, `edf` and the fitted surface.
3. `dm_builder.py` and the two diagnostics geometry helpers — whether learned
   spline knots and discretized bins follow weight mass or physical rows.
4. `distributions.py` — the reported log-likelihood's normalizer, and hence
   the absolute AIC and BIC.
5. `inference/metrics.py` — the randomized quantile residuals' reference
   distribution.
6. `profiling/nb.py` — the negative-binomial theta profile, whose docstring
   said outright "Must be frequency weights, not variance weights". Under the
   prior contract `w Y ~ NB2(w mu, w theta)`, which changes the profile score's
   digamma pair to `psi(w(y+theta)) - psi(w theta)` and nothing else.

## Why

The two readings are different likelihoods, not different scalings of one:

- Prior (also *variance*, *analytic*) weights are McCullagh & Nelder's
  (*Generalized Linear Models*, 2nd ed., 1989, §2.2.2) prior weights. R's
  `glm` documents its only weight argument as "prior weights", with the values
  "inversely proportional to the dispersions", and glum documents
  `sample_weight` as "weights or exposure to which the variance is inversely
  proportional".
- Frequency (*case*) weights are statsmodels' `freq_weights`, Stata's
  `fweight`, and SAS `GENMOD`'s `FREQ` statement — which SAS documents as
  integer-valued and as changing the sample size used in downstream formulas,
  while its `WEIGHT` statement divides the dispersion parameter and does not.
- Where both exist they are separate declared inputs (statsmodels
  `var_weights` / `freq_weights`; Stata `aweight` / `fweight`), and where only
  one exists it is the prior reading (R, glum).

The two coincide only at `w == 1` — integer weights do **not** make them
agree, and at `w == 2` the residual degrees of freedom, `phi`, every Wald
standard error and BIC already differ. At fractional `w` only the prior reading
is a likelihood at all: a row cannot be replicated 0.4 times, and `sum(w) - edf` stops
counting anything. Exposure is continuous, and in the setting this library
targets the aggregated `y = incurred / exposure` with `sample_weight = exposure`
is the ordinary case — so the previous default was wrong for the primary use
case, silently, in every weighted non-Tweedie fit.

## Who is affected, and by how much

All magnitudes below are **measured**, not estimated.

### Nothing moves unless the fit is weighted

Unweighted fits and fits with `w == 1` are identical under both contracts. This
is pinned per family in `tests/test_weight_semantics.py`.

### The `"frequency"` arm is master, bit for bit

On the fixture below, every published quantity under `weight_semantics=
"frequency"` is **exactly** the value the previous release produced — `phi`,
deviance, `effective_df`, log-likelihood, AIC, BIC, `beta`, predictions,
`lambda`, and the resolved knot vector, on both `fit` and `fit_reml`. The
change therefore moves nothing that was already shipping; every difference
below is the new default, not a side effect.

### What the new default moves

Fixture: Gamma, log link, 1,500 rows, an 8-knot `quantile_rows` spline plus a
5-level categorical, continuous weights with `sum(w) = 1850.70` and
`mean(w) = 1.234` (min 0.056, max 5.801).

| quantity | `"frequency"` (= previous release) | `"prior"` (new default) | change |
|---|---|---|---|
| **`fit`** | | | |
| residual d.f. | 1835.891 | 1485.183 | ÷1.2361 |
| `phi` | 0.18397248 | 0.22734645 | ×1.2358 |
| every Wald standard error | — | — | **×1.1117** |
| `effective_df` | 14.80783 | 14.81727 | +0.0009 |
| deviance | 338.8411 | 338.8453 | +1.2e-5 rel |
| log-likelihood | −3201.756 | −2833.832 | +367.92 |
| AIC | 6433.128 | 5697.298 | −735.83 |
| BIC | 6514.916 | 5776.026 | −738.89 |
| **`fit_reml`** | | | |
| `lambda` (spline) | 143.563 | 191.589 | ×1.3345 |
| `effective_df` | 8.25719 | 8.05957 | −0.198 |
| `phi` | 0.17893483 | 0.21680067 | ×1.2116 |
| log-likelihood | −3203.120 | −2834.403 | +368.72 |
| learned knots | weight mass | physical rows | moved |

The `phi` shift is the ratio of the two likelihood sizes: `1835.891 / 1485.183
= 1.23614` against a measured `phi` ratio of `1.23576`, the remainder being the
slightly different design. Standard errors scale as its square root:
`sqrt(1.23576) = 1.11165` against a measured per-coefficient ratio of 1.11166
to 1.11190 across the five coefficients.

**AIC and BIC are not comparable across the two contracts.** Their fall of
about 736 points is entirely the likelihood's `(y, w)`-only normalizer —
`-2 (l_prior - l_frequency) = -735.85` against a measured AIC difference of
−735.83. Within one contract they compare models as before; across contracts
they compare nothing.

**Nor will they match R's `AIC()` on a weighted Gamma, Poisson or negative
binomial fit**, and this document leans on R elsewhere, so it is worth saying
plainly. R applies the weight *outside* a common-shape density —
`sum(w * dgamma(y, 1/disp, scale = mu * disp, log = TRUE))` with
`disp = deviance / sum(w)`, and `sum(w * dpois(y, mu, log = TRUE))` — whereas
superglm evaluates the EDM prior form, which scales each row's own shape by
`w`. Only R's Gaussian arm carries the `0.5 sum(log w)` term that superglm's
reproduces. Measured with R 4.5.0 as an oracle, R's weighted Gamma `logLik` is
−1725.469 at `deviance / sum(w)` and −1901.508 at the dispersion its own
`summary()` prints, so R does not agree with itself here either. superglm's
form is the exact prior-weight likelihood; it is simply not R's convention,
and AIC differences within either system remain the comparable quantity.

`beta` is unchanged in substance: the two contracts share a score equation.
The small movement in predictions above (mean 3.9161137 to 3.9161069) comes
from the knots, not the likelihood — a fixed or preconstructed knot vector
removes it entirely.

### What it costs

The prior contract is slower to fit for Gamma, and only for Gamma. Its
saturated log-likelihood is `sum_i G(w_i k)` — one special-function evaluation
per *distinct* weight, re-evaluated at each step of the dispersion root-find —
where the frequency arm's is `sum(w) G(k)`, a single scalar. Some of that is a
difference in the likelihood rather than in the implementation: there is no way
to learn `sum_i G(w_i k)` without touching every distinct weight, and no
sufficient statistic stands in for it under continuous weights.

Measured on a quiet machine with all thread pools pinned, interleaved,
six runs per arm, after the accelerations described below:

| fixture | `"frequency"` | `"prior"` | ratio |
|---|---|---|---|
| Gamma, 50k rows, every weight distinct | 0.221 s | 0.283 s | **1.28x** |
| Gamma, 50k rows, 12 distinct weights | 0.236 s | 0.261 s | 1.11x |
| Gaussian, 50k rows, every weight distinct | 0.112 s | 0.118 s | — |

The Gaussian arms overlap, so that row is noise rather than an effect; its
prior term is a single `0.5 sum(log w)` constant.

**Repeated weights are nearly free**: the profiler reduces over distinct
weights with multiplicities, so rounding exposure to a few dozen bands recovers
most of the difference. The cost is per *distinct* weight, not per row.

#### What the accelerations were, and what the first analysis got wrong

The prior arm was first measured at 1.67x, and that was read as irreducible on
the strength of a micro-benchmark of the score expression in isolation, where
`digamma` is 917 us of 1004 us. Profiling the real fit contradicted that on
three counts, none of which the isolated benchmark could show:

- **A third of the profile calls are exact repeats.** An accepted line-search
  trial's lambdas are re-evaluated identically at the top of the next outer
  iteration — 21 calls carrying 14 distinct `(D_p, M_p)` pairs. Whole terms are
  now memoized on that key.
- **`polygamma(1, x)` discards a full digamma pass.** SciPy forms
  `(-1)^(n+1) * gamma(n+1) * zeta(n+1, x)` and then `where`-selects it against
  `psi(x)`, so requesting the trigamma computes a digamma and a `where` that
  are thrown away. `zeta(2.0, x)` is the same value by construction, and is
  verified bitwise identical over 1.2M points spanning the branch's range.
- **About a quarter of the subsystem was numpy bookkeeping** — boolean masks,
  fancy-index gather/scatter copies, and a multiplicity multiply that is a
  no-op when every weight is distinct. Branch dispatch is now by slice, which
  `np.unique`'s ascending output makes valid.

Two further levers: the root-find is warm-started from a secant predictor over
the last two roots (falling back to the shipped ±30 window), and the profile
curvature is deferred until the derivative is read — which rejected trials, the
boot evaluation and the post-fit `phi` recompute never do.

A safeguarded Newton solve was **refuted analytically** rather than tried:
`S'(u) = S(u) + C(u)`, so each derivative costs a trigamma pass at 11.6x a
digamma pass. Newton breaks even only at two iterations or fewer, and
evaluating the curvature exactly at the final root erases even that.
Derivative-free is correct at this cost ratio.

None of this touches the answer. The `"frequency"` arm is bitwise identical and
runs the shipped solver body unchanged, which a test enforces by forbidding it
from reaching the warm solver or the caches. The prior arm is bitwise identical
on the all-distinct fixture; the twelve-distinct one moves 3.3e-13 in `edf` and
8.2e-12 in a `lambda` of 5.1e5 — a flat near-boundary optimum, and inside the
root-finder's own `xtol=1e-12`. Warm and cold solves answer the same equation
to the same tolerance and differ within it.

### Zero weights

Admissible under both contracts. Under `"frequency"` a zero weight drops out of
`sum(w)` on its own; under `"prior"` it is excluded from the row count, which
is R's rule — a twelve-row Gamma fit with four zero weights and rank 2 returns
`df.residual = 6`, not `10`, with the explicit note that "observations with
zero weight [are] not used for calculating dispersion".

The one exception: a **Tweedie** fit under `"prior"` still requires strictly
positive weights, because its compound-Poisson normalizer carries `log w`.
Under `"frequency"` the weight never enters that normalizer, so Tweedie admits
zero weights there like every other family.

### Counting families have a lattice

The prior construction for Poisson and the negative binomial is
`w Y ~ Poisson(w mu)` and `w Y ~ NB2(w mu, w theta)`, both supported on the
non-negative integers. The canonical weighting is on that lattice by
construction — `y = count / exposure` with `sample_weight = exposure` recovers
the count — and that is the case this change exists to serve.

Prior-weighted Poisson and negative-binomial rates are accepted **without an
integrality warning**, including deliberately adjusted rates (for example, a
20% uplift) and round-off in `w * y`. Fitting and likelihood evaluation do not
round the supplied response.

Where `w * y` is not integral, `gammaln` gives a smooth continuation of the
count likelihood, not an exact count probability. Interpret the reported
log-likelihood, AIC and BIC accordingly. For Poisson and NB with fixed `theta`,
the gamma-function terms do not depend on the mean. When estimating `theta`,
the negative-binomial continuation is `theta`-dependent and can affect the
estimated `theta`, its profile interval and, through refitting, the fitted
means. Randomized quantile residuals still use neighbouring integer counts.

Use `weight_semantics="frequency"` only for replication weights. That contract
still warns about fractional replication weights or fractional counting
responses. The prior-weighted binomial warning is also unchanged.

### One declared limitation

`estimate_p` profiles the Tweedie power against the compound-Poisson density
with the weight inside its normalizer, which is the prior contract and only
that one. Under `weight_semantics="frequency"` with non-unit weights it raises
rather than answering under the wrong likelihood. The combination was
unreachable before this release — Tweedie always read prior weights — so
nothing that worked has stopped working. `estimate_theta` has no such
limitation: the negative-binomial profile carries both contracts.

## What to do

- **Unweighted pipelines**: nothing.
- **Tweedie**: nothing. Its default is unchanged — it already read prior
  weights, and now says so.
- **You aggregate and weight by exposure or claim count** (the case this
  change exists for): nothing, and the new numbers are the calibrated ones.
  Expect `phi`, every standard error and interval, and REML's `lambda` to
  move; re-check any threshold tuned against the old dispersion.
- **You genuinely have compressed duplicate rows**: pass
  `weight_semantics="frequency"` and you get the previous fit exactly.
- **You must reproduce a previous fit bit-for-bit** (regulatory refits,
  frozen-model operations): pass `weight_semantics="frequency"`.
- **Pickled models**: a model or `ModelConfig` pickled before this release
  restores under the contract its family carried at the time — `"frequency"`
  for non-Tweedie, `"prior"` for Tweedie — and keeps reproducing what it
  recorded. Only newly constructed models adopt the new default.
- **Pipelines comparing AIC or BIC across releases**: recompute both sides
  under one contract. The level is not comparable; differences within a
  contract are.

## Verification trail

- Established practice checked against R's `glm` documentation, statsmodels'
  GLM parameter documentation and weighted-GLM guide, SAS/STAT's `GENMOD`
  `WEIGHT` and `FREQ` statements, Stata's weight taxonomy, and glum's
  documentation.
- The zero-weight rule was measured by running R 4.5.0 as a black-box oracle,
  not read from its source.
- Both contracts' definitions are pinned in `tests/test_weight_semantics.py`:
  `"frequency"` against literal row replication for all six families, and
  `"prior"` against `scipy.stats` densities for Gaussian, Gamma, Poisson,
  negative binomial and binomial, plus the existing Tweedie path.
- The before/after table is the two arms of a single fixture, with the
  `"frequency"` arm checked against a detached `origin/master` worktree.
