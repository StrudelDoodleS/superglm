# Interaction Screening (PSST)

`SuperGLM.screen_interactions` ranks every candidate pair of fitted spline
features by how much of the model's leftover signal a `ti()` tensor smooth
on that pair could absorb. It is a Penalized Smooth Score Test (PSST): one
O(n) pass per pair against the fitted mains model, no refits, so screening
ten pairs costs a few seconds where fitting ten tensor terms would cost
minutes.

```python
model = SuperGLM(family="poisson", features=feats).fit_reml(X, y, sample_weight=exposure)
table = model.screen_interactions(X, y, sample_weight=exposure)
print(table.head())
#  feature_a feature_b  statistic      z  edf0  lambda0  n_cells
#       long       lat     98.252 14.540  16.0    0.673   339889
#         bm      agec     23.592 10.796   2.0  110.156     1127
```

The test asks, for each pair: *after profiling out what the pair's own main
effects already explain, does the working residual carry curvature shaped
like this pair's actual tensor basis?* The probe is evaluated at a ladder
of complexity budgets (`edf0`, default `(2, 4, 8, 16)` effective degrees of
freedom) and each pair is ranked by its best noise-normalized score
`z = (T - edf0) / sqrt(2 * edf0)`, so smooth surfaces and high-frequency
surfaces are both visible.

## Reading the output

- **Rank by `z`, and only `z`.** `statistic`, `edf0`, and `lambda0`
  describe the pair's *winning rung*, so they are not comparable between
  rows. `statistic` is the dispersion-scaled score statistic at that rung.
- **`z` is noise-floor-normalized, not a p-value.** Under the null each
  rung has mean zero and unit-order scale, and the statistic is scaled by
  the Pearson dispersion of the mains fit — so the same reading applies to
  Poisson, Gamma, Gaussian, binomial, and overdispersed data. Measured
  across a battery of null datasets (families from Bernoulli to
  sigma=10 Gaussian, correlated factors, 300x220 grids, heavy exposure
  spread), the best null `z` never exceeded ~4.5. Treat `z` below 4-5 as
  noise-level; treat the release-gate bound of 10 as generous.
- **The winning `edf0` is a shape diagnostic** — for pairs with material
  `z`. A win at rung 2 means tilt-level evidence (a simple in-in surface);
  wins at 8-16 mean genuinely curved or high-frequency structure. Under
  the pure null the winning rung is meaningless.
- **Confirm by refitting, always.** The screen is a ranking device.
  Refit the top three pairs as `ti()` terms and judge by deviance gain —
  near-tied `z` values are common and the refit, not the screen, is the
  gate. Evidence *density* is not payoff: a strong tilt (rung-2 win) can
  out-`z` a curved surface that buys three times the deviance.
- **NaN rows are skipped pairs**, not failures: the pair's joint support
  exceeded `max_cells` even after the quantile-binning fallback (or the
  statistic degenerated). They sort last. Rows computed on binned support
  carry `approx=True`; exact rows carry `approx=False`.

## What it inherits from the fit

Screening linearizes at the fitted model: the offset used at fit time is
applied automatically (pass `offset=` only to override), `sample_weight`
keeps its exposure meaning, and the mains model's own smoothness choices
define what "leftover" means. A badly specified mains model screens
against the wrong baseline — screening quality is downstream of fit
quality.

Parent smooths must be single-penalty: mains fitted with `select=True`
raise up front, because `ti()` terms cannot be built on such parents
either.

## Measured limits

- **Corner-localized effects.** An interaction confined to a thin corner
  of the joint support (young driver x high power, with little data
  there) screens weakly — there is simply little signal-carrying data,
  and a full `ti()` refit faces the same limit. Expect low `z`, not a
  false positive.
- **Heavily correlated pairs.** At rho ~ 0.85 the joint support is a
  ridge; the off-ridge tensor directions are thinly identified and a real
  signal is demoted, not lost. The `ti()` refit faces the same
  identifiability.
- **Continuous x continuous cardinality.** Pairs whose unique-value grid
  exceeds `max_cells` fall back to quantile binning (`screen_bins`
  empirical-quantile support points per margin, basis evaluated at
  within-bin means) and are flagged `approx=True` in the output. Pairs
  within budget are always computed exactly — the fallback never touches
  them. Screening-only: a confirmatory `ti()` refit of a flagged pair
  uses the full data.

## Provenance

The scan-over-budgets design follows the adaptive score-testing family
(Eubank & Hart order selection; Fan's adaptive Neyman test; multiscale
testing), applied to penalized tensor smooths with cell-collapsed
assembly. Validated end to end on freMTPL2 frequency and severity, the
Belgian beMTPL97 book (where it independently surfaces the long:lat
spatial interaction the literature models on that data), the 2015
Pricing Game book, and a null/power gauntlet across families,
dispersions, and support geometries. It is a screening tool: it orders
the refit queue, it does not certify significance.
