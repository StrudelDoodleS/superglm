# Interaction Screening (PSST)

`SuperGLM.screen_interactions` ranks every candidate pair of fitted features
by how much of the model's leftover signal the interaction *that pair would
actually refit as* could absorb. Five pair kinds screen today, each probed as
the block its own refit would build:

- **`ti`** — spline x spline: a `ti()`-style tensor deviation surface.
- **`spline_cat`** — spline x factor: a reference-coded deviation curve for
  each non-base level.
- **`numeric_cat`** — numeric x factor: a slope on the numeric for each
  non-base level.
- **`cat_cat`** — factor x factor: the cross-level cells of the two-way table.
- **`numeric_numeric`** — numeric x numeric: a single product tilt.

A spline-mode `OrderedCategorical` margin screens as a spline **on its mapped
level scores** — the axis its own refit builds — so an OC x spline pair is a
`ti` row and an OC x factor pair a `spline_cat` row.

It is a Penalized Smooth Score Test (PSST): one O(n) pass per pair against the
fitted mains model, no refits, so screening ten pairs costs a few seconds
where fitting ten interaction terms would cost minutes.

```python
model = SuperGLM(family="poisson", features=feats).fit_reml(X, y, sample_weight=exposure)
table = model.screen_interactions(X, y, sample_weight=exposure)
print(table.head())
#  feature_a feature_b kind  statistic      z  edf0  lambda0  n_cells  approx
#       long       lat   ti     98.252 14.540  16.0    0.673   339889   False
#         bm      agec   ti     23.592 10.796   2.0  110.156     1127   False
```

The test asks, for each pair: *after profiling out what the pair's own main
effects already explain, does the working residual carry structure shaped like
the block this pair would refit as?* For the penalized kinds the probe is
evaluated at a ladder of complexity budgets (`edf0`, default `(2, 4, 8, 16)`
effective degrees of freedom) and each pair is ranked by its best
noise-normalized score `z = (T - edf0) / sqrt(2 * edf0)`, so smooth surfaces
and high-frequency surfaces are both visible. The unpenalized kinds have no
penalty to scan: they are evaluated once, at the block's own dimension.

## Pair kinds

| kind | probe block | penalty | probe df | refit target |
|---|---|---|---|---|
| `ti` | centered spline ⊗ centered spline | kron-sum | `edf0` ladder | `TensorInteraction` |
| `spline_cat` | centered spline menu ⊗ level contrasts | `kron(S_spline, I)` | `edf0` ladder, rank-clamped | `SplineCategorical` |
| `numeric_cat` | the numeric's column on each non-base level | none | `L - 1` | `NumericCategorical` |
| `cat_cat` | level contrasts ⊗ level contrasts | none | `(L1 - 1) * (L2 - 1)` | `CategoricalInteraction` |
| `numeric_numeric` | the two numerics' product | none | 1 | `NumericInteraction` |

For the three unpenalized kinds the probe columns *are* the refit term's
columns: the kron of two contrast menus on a level-pair cell is that cell's
indicator, and the per-level slope columns are exactly what a
`NumericCategorical` builds. For `ti` and `spline_cat` the probe spans the
refit term's identifiable deviation space — the part the pair's own mains
cannot absorb. `probe df` is the unpenalized block's dimension before the data
reduces it: for those kinds the `edf0` column reports the rank actually
achieved, which is lower when cells are empty or their columns collinear (an
11 x 22 `cat_cat` table can report 208 rather than 210).

A `spline_cat` row can also be confirmed as a `FactorSmooth` when pooling
across levels is wanted rather than reference-coded deviations — same parents,
penalized level curves; see [Interactions](interactions.md).

**What gets swept.** `candidates=None` pairs every eligible fitted feature:
splines, spline-mode `OrderedCategorical`, `Categorical` with at least two
levels after any grouping, and `Numeric`. `Polynomial`, `RandomEffect`,
step-mode `OrderedCategorical` and single-level factors have no screenable
margin; spline x numeric has both margins but no refit target yet, and is
deferred until a varying-coefficient term exists (respec the `Numeric` as a
`Spline` to screen the pair as `ti`). Both cases drop out of the default sweep
rather than reporting a null result, and naming one in `candidates=` raises
that deferral specifically, not a generic "unknown feature" error. Pairs the
model already fits as an interaction — of any class, `FactorSmooth`
included — are excluded too: the screen profiles only the parent mains, so it
cannot re-screen a term already in the model.

A mixed sweep comes back as one sorted table — here an 80,000-row sample of
the freMTPL2 frequency book, fitted with two splines, a numeric and two
factors, on the house exposure contract (`y` the claim *rate*
`ClaimNb / Exposure`, `sample_weight` the exposure, `phi` estimated at 2.52):

```python
table = model.screen_interactions(df, y, sample_weight=exposure)
print(table.to_string(index=False))
#  feature_a feature_b        kind  statistic         z       edf0      lambda0  n_cells  approx
#     VehAge  VehBrand  spline_cat  43.579430  4.875400  16.000000 6.820539e-01      627   False
# BonusMalus  VehBrand numeric_cat  18.139171  1.819974  10.000000 0.000000e+00       11   False
#    DrivAge    VehAge          ti   0.733057 -0.633471   2.000000 1.611914e+01     4560   False
#    DrivAge    Region  spline_cat  13.267251 -1.193182  20.999944 6.861393e+09     1760   False
# BonusMalus    Region numeric_cat  11.890018 -1.405701  21.000000 0.000000e+00       22   False
#    DrivAge  VehBrand  spline_cat   3.632010 -1.423923   9.999981 1.300382e+10      880   False
#     VehAge    Region  spline_cat   3.889631 -2.640197  21.000112 2.053060e+09     1254   False
#   VehBrand    Region     cat_cat  76.234205 -6.460350 208.000000 0.000000e+00      242   False
```

Every eligible pair, four kinds, one ranking by `z` alone — the sweep's only
spline x spline pair (`DrivAge x VehAge`, the `ti` row) places third of eight,
on a rung-2 win worth less than its own noise floor. Eight rows, not ten: the
two spline x numeric pairs are deferred, so they never enter the sweep. The
queue this produces is one pair long: `VehAge x VehBrand` is three noise units
clear of the next row, and refitting it as a `SplineCategorical` — the gate,
not the score — buys 327.5 deviance on this sample.

`statistic` is not comparable down that column: the `cat_cat` row's 76 is a
208-dimensional block and the `numeric_cat` row's 18 a 10-dimensional one. The
large `lambda0` values are bracket edges at clamped rungs, not fitted smoothing
parameters. Six of the eight rows carry a negative `z`, which is ordinary: a
statistic can land below its rung's noise floor, and on this book most pairs
do. The `cat_cat` row's -6.46 is the extreme case — a 208-df block whose
statistic, scaled by the *whole model's* dispersion, lands at 76, so the global
`phi` is conservative for that block. A large negative `z` only pushes a pair
down the queue; it never promotes one. Nothing here was binned or refused
(`approx` is False throughout, no NaN rows).

Read the top row against its own kind's measured noise maximum below (5.53 for
`spline_cat`): 4.88 does not clear it — and the refit bought 327.5 deviance
anyway. That is the screen working as described rather than a contradiction:
the floor is the largest value a wide null battery produced, not a threshold a
real pair must beat, and the confirmatory refit is what settles the question.

## Reading the output

- **`kind` names the block that was probed**, and with it the term you would
  refit to confirm the row. Kinds share one sorted table because `z`
  normalizes each block against its own scale — but that normalization is not
  equal across probe df, so read the floors below before ranking a 1-df row
  against a 16-df one.
- **Rank by `z`, and only `z`.** For the penalized kinds (`ti`, `spline_cat`)
  `statistic`, `edf0` and `lambda0` describe the pair's *winning rung*, so
  they are not comparable between rows; at a clamped rung `edf0` holds the
  achieved value and `lambda0` is a bracket edge rather than an interpretable
  smoothing parameter. For the unpenalized kinds (`numeric_cat`, `cat_cat`,
  `numeric_numeric`) there is a single rung: `edf0` reports the block's
  achieved rank and `lambda0` is `0`, and the `edf0=` argument does not apply
  to them. `statistic` is the dispersion-scaled score statistic.
- **`z` is noise-floor-normalized, not a p-value.** Under the null each rung
  has mean zero and unit-order scale, and the statistic is scaled by the
  Pearson dispersion of the mains fit — so the same reading applies to
  Poisson, Gamma, Gaussian, binomial, and overdispersed data. What the ladder
  then reports is the best of four rungs, which is why the penalized kinds sit
  a little above zero on a pure null. The measured floors are below.
- **`n_cells` is the grid the probe assembled**: the product of the two
  margins' grid sizes, where a spline or OC margin contributes its support
  size, a factor contributes its level count `L`, and a numeric contributes 1
  (so a `numeric_cat` row reports the factor's levels and a `numeric_numeric`
  row reports 1).
- **The winning `edf0` is a shape diagnostic** for penalized rows with
  material `z`. A win at rung 2 means tilt-level evidence (a simple in-in
  surface); wins at 8-16 mean genuinely curved or high-frequency structure.
  Under the pure null the winning rung is meaningless.
- **Confirm by refitting, always.** The screen is a ranking device. Refit the
  top three pairs as their `kind`'s refit target and judge by deviance gain —
  near-tied `z` values are common and the refit, not the screen, is the gate.
  Evidence *density* is not payoff: a strong tilt (rung-2 win) can out-`z` a
  curved surface that buys three times the deviance.
- **NaN rows are skipped or refused pairs**, not failures. A gridded pair
  (`ti`, `spline_cat`) is skipped when it exceeds the cell or intermediate
  budgets even after the quantile-binning fallback, when its tensor curvature
  block alone is too large for the budget (binning cannot shrink basis
  dimensions, so those skip with no binning attempted), or when the statistic
  degenerates. A `numeric_cat` pair has no grid to shrink, so a factor too
  wide for the pair's blocks is *refused* rather than approximated: the gate
  holds the largest of those blocks, the `(L + 1)`-wide overlap curvature, to
  the budget — `(L + 1)^2 <= max_cells`, which admits factors up to 2235
  levels at the default — and raising `max_cells` lifts the refusal and
  computes the pair exactly.
  `numeric_numeric` contracts to 3x3 blocks whatever the supports and is never
  refused. All such rows sort last.
- **`approx=True` means the row's probe basis differs from what a confirmatory
  refit would build** — either a spline margin was quantile-binned for the
  screen, or that pair's refit would discretize *lossily*. Only rows with a
  spline or OC margin can carry it: `cat_cat`, `numeric_cat` and
  `numeric_numeric` rows are always `approx=False`, and refusal is not
  approximation.

### Measured null floors

These are **measured maxima over a null battery, not calibrated quantiles** —
no probability statement is attached to any number here.
`benchmarks/screening_null_floors.py --seeds 40` fits 160 mains models (four
families x 40 seeds, n=8000 rows each) with no interaction anywhere in the
truth, and screens 3520 pairs:

| kind | rows | mean `z` | p90 `z` | max `z` | probe df |
|---|---|---|---|---|---|
| `ti` | 480 | 0.42 | 1.56 | 7.31 | 2-16 |
| `spline_cat` | 1440 | 0.42 | 1.82 | 5.53 | 2-16 |
| `numeric_cat` | 960 | 0.00 | 1.19 | 7.53 | 1-3 |
| `cat_cat` | 480 | -0.01 | 1.20 | 3.98 | 2-6 |
| `numeric_numeric` | 160 | -0.07 | 1.07 | 4.91 | 1 |

Read the last three columns together: nine of every ten rows in the battery
fell below 1.1-1.8 depending on kind, and yet the largest single row of each
kind reached 3.98-7.53, two of the five kinds above 6. **A `z` of 5 is
therefore not evidence by itself.** This supersedes the earlier reading of
this guide ("the best null `z` never exceeded ~4.5; treat `z` below 4-5 as
noise-level"), which came from a smaller, splines-only battery — a maximum
grows with the number of draws, so that is a sample-size correction rather
than a regression.

The heaviest tails sit at low probe df. The two largest rows in the battery
are a 1-df `numeric_cat` (a slope on a two-level factor) and a rung-2 `ti`,
and all six of the largest sit at `edf0 <= 3`.

| kind | probe df | rows | max `z` |
|---|---|---|---|
| `numeric_cat` | 1 | 320 | 7.53 |
| `numeric_cat` | 2 | 320 | 4.45 |
| `numeric_cat` | 3 | 320 | 4.78 |
| `cat_cat` | 2 | 160 | 3.18 |
| `cat_cat` | 3 | 160 | 3.98 |
| `cat_cat` | 6 | 160 | 2.90 |
| `numeric_numeric` | 1 | 160 | 4.91 |

That is a statement about the *probe's* df, not about the kind, and it is a
tendency at the extreme rather than a law: `numeric_cat`'s 1-df row tops every
other configuration in the table by 2.6, but neither kind is monotone in df
(`numeric_cat` 7.53 / 4.45 / 4.78 at df 1 / 2 / 3; `cat_cat` 3.18 / 3.98 /
2.90 at df 2 / 3 / 6). Rank a 1-df `numeric_cat` on a two-level factor against
a 16-df `ti` and the low-df row can win on noise alone, so compare like with
like before spending a refit. None of this makes `numeric_numeric` the heavy
kind: at 4.91 it sits below the `numeric_cat` maximum on the same 1 df, and on
the fewest draws — one pair per sweep, 160 rows.

The families do not agree on the floor, and the dispersed Gaussian carries it:
Gaussian tops three of the five kinds and holds both maxima above 6 (7.53 on
`numeric_cat`, 7.31 on `ti`), where no other family reached beyond 5.53
anywhere in the battery (Poisson at most 4.08, gamma 5.34, binomial 5.53).
Read the headline maxima as Gaussian-driven rather than as something every
family reproduces — the quoted floor is the maximum over all four, so for any
one family it is the conservative reading.

Ordered-categorical margins do not move it. Against the plain spline margins
of the same kind the four bulk gaps sit within 0.11 on the means and 0.09 on
the p90s (`spline_cat` mean 0.11 and p90 0.05; `ti` mean 0.10 and p90 0.09),
and the direction is not consistent: OC is *lower* on the `spline_cat` mean
and *higher* on the `ti` one, while the maxima disagree with the means on both
kinds (`spline_cat` OC 5.53 against plain 5.34; `ti` OC 4.25 against plain
7.31). The p90 is the one statistic where OC sits above plain on both kinds,
by 0.05 and 0.09 against p90s of 1.5-1.9. Differences that small, with no
direction that survives across kinds, are sampling noise rather than a short
score grid inflating anything.

One thing follows for the release-gate bound of `z < 10`: it holds with
headroom, but it is a bound, not a floor measurement. The suite's null gates
cover every kind and every probe df this battery measures — `spline_cat`,
`cat_cat` at df 6 and `numeric_cat` at df 2 and 3 in both gates, `ti` in the
Poisson gate only, and `numeric_cat` at df 1, `cat_cat` at df 2 and 3 and
`numeric_numeric` in the Gaussian one — against a battery whose widest single
row anywhere was 7.53
over 3520 rows. But a floor is a maximum, so it grows with the width of the
sweep: a wide book screened in one pass draws more null rows than this whole
battery did. Treat 10 as generous for a handful of pairs and thin for
hundreds.

## What it inherits from the fit

Screening linearizes at the fitted model: **both the offset and the
`sample_weight` used at fit time are applied automatically** (weights
only when the fit's were non-unit; pass either only to override), and
the mains model's own smoothness choices define what "leftover" means.
Inherited arrays are in training row order, so inheriting requires
`X`/`y` to be the retained training data — to screen a holdout,
subsample, or reordered frame, pass `sample_weight` (and `offset`)
explicitly. A badly specified mains model screens against the
wrong baseline — screening quality is downstream of fit quality. The
Pearson dispersion that scales the statistic is attached to the result as
`table.attrs["phi"]` and can be overridden with `phi=` (the estimate uses
positive-weight-row count − edf residual degrees of freedom per the
exposure weight contract,
which keeps rankings invariant to the units exposure is measured in; a
frequency-weight user can supply their own `phi`).

Factor margins are read through the fitted spec: levels are indexed in the
fit's own order, any `LevelGrouping` collapse is applied exactly as the fit
applies it, and a level the fit never saw raises through the spec's own
validator rather than screening quietly.

Parent smooths must be single-penalty: mains fitted with `select=True` raise
up front, because `ti()` terms cannot be built on such parents either. For an
`OrderedCategorical` margin the same check applies to its inner spline.

Screening always probes the exact-basis tensor, including for parents
that discretize (whose confirmatory refit uses binned marginal
supports). That is the same support-discretization gap as the quantile
fallback — measured at ~3.5% relative z on signal pairs — and it never
affects which basis the confirmatory refit itself uses. To make the gap
visible in the output rather than doc-only, **any pair whose refit would
discretize lossily carries `approx=True`**, applying the gate that refit
itself uses: a `ti()` refit bins its marginal supports only when BOTH parents
resolve to fit-time discretization (per-spec `discrete` overriding the model
flag), while a `SplineCategorical` refit bins whenever its ONE spline parent
does. Either way the row is flagged only when some margin that refit would bin
has a cardinality exceeding its resolved bin count — lossless binning returns
the exact unique support, so low-cardinality rating factors stay
`approx=False`. An OC margin lives on at most one score point per level: it
does not reach the binning fallback in practice, and its refits do not
discretize at all, so OC pairs stay exact on both sides.

## Measured limits

- **Corner-localized effects.** An interaction confined to a thin corner
  of the joint support (young driver x high power, with little data
  there) screens weakly — there is simply little signal-carrying data,
  and a full refit faces the same limit. Expect low `z`, not a
  false positive.
- **Rare cells and rare levels.** A `cat_cat` cell or a `spline_cat` level
  with few rows contributes little to the block, so it screens honestly weak
  for the same reason — the confirmatory refit is equally starved there. Rare
  levels cost ranking power, not correctness.
- **Heavily correlated pairs.** At rho ~ 0.85 the joint support is a
  ridge; the off-ridge tensor directions are thinly identified and a real
  signal is demoted, not lost. The refit faces the same
  identifiability.
- **Continuous x continuous cardinality.** Pairs whose unique-value grid
  (or curvature-intermediate allocation, bounded at a small multiple of
  the same budget) exceeds `max_cells` fall back to quantile binning (`screen_bins`
  empirical-quantile support points per margin, basis evaluated at
  within-bin means) and are flagged `approx=True` in the output. Pairs
  within budget are always computed exactly — the fallback never touches
  them. Screening-only: a confirmatory refit of a flagged pair
  uses the full data.
- **Factor and numeric margins have no such cardinality limit.** A factor
  margin never bins — its support *is* the fitted level set — and a numeric
  margin never grids at all: it enters its probe linearly, so moments of the
  numeric accumulated over the other margin's cells are its exact sufficient
  statistics, at any cardinality. Neither margin can raise `approx`. The one
  degradation available to a numeric-margin pair is refusal (`numeric_cat`
  with a factor wider than its blocks), and a refused row is a NaN row, never
  an approximated one.

## Provenance

The scan-over-budgets design follows the adaptive score-testing family
(Eubank & Hart order selection; Fan's adaptive Neyman test; multiscale
testing), applied to penalized tensor smooths with cell-collapsed
assembly. The unpenalized kinds are classical Rao score tests (Rao 1948) on
the refit term's own columns, profiled against the pair's mains and scaled by
the fit's dispersion. The penalized kinds follow the score-test line for
penalized smooths and their variance-component representation (Lin 1997;
Zhang & Lin 2003), which is where normalizing `T` against `edf0` comes from.
Screening an ordered factor on its level scores is the standard scoring device
for ordered categorical predictors (Graubard & Korn 1987; Gertheiss & Tutz
2009; Azzalini 2023/2024). The ranking-first stance — spend interaction
complexity only where the main effects cannot explain the signal, and let the
refit be the gate — has a complementary literature in reluctant interaction
modeling and inference (Yu et al. 2019; Huang et al. 2025). Validated end to
end on freMTPL2 frequency and severity, the Belgian beMTPL97 book (where it
independently surfaces the long:lat spatial interaction the literature models
on that data), the 2015 Pricing Game book, and a null/power gauntlet across
families, dispersions, support geometries, and every pair kind. It is a
screening tool: it orders the refit queue, it does not certify significance.
