# Choose smoothing by held-out prediction

A spline term needs a smoothing parameter to balance following the signal
against chasing noise. {py:meth}`~superglm.SuperGLM.fit_reml` chooses that
smoothing from a likelihood criterion (REML) that **trusts the declared count
law and that rows are independent** once you account for the covariates.
{py:meth}`~superglm.SuperGLM.fit_ncv` **trusts neither**: it chooses the
smoothing parameters by predicting rows it did not fit (neighbourhood
cross-validation; Wood, 2024).

When to reach for `fit_ncv` and when not to bother:

- **Reach for `fit_ncv`** when counts are far more variable than the declared
  law allows (so Poisson REML reads overdispersion as real wiggliness), or when
  rows share unobserved risk — repeated policies or fleets, calendar periods,
  or nearby spatial cells.
- **Do not bother** when the declared law is reasonable and rows within a group
  barely move together. On the public Spanish motor book below, `fit_reml` and
  `fit_ncv` produce the same curves and agree to within a few thousandths of a
  percent on held-out error, while `fit_ncv` takes about three to four times as
  long to fit.

You tell `fit_ncv` which rows leave the fit together, and it picks the smoothing
that minimises the summed held-out deviance:

```python
from sklearn.model_selection import GroupKFold

model.fit_ncv(X, y, sample_weight=w, offset=o, groups=df["policy_id"])   # each group held out in turn
model.fit_ncv(X, y, cv=GroupKFold(5), groups=df["policy_id"])            # the folds cross_validate would use
model.fit_ncv(X, y, neighbours=near)                                     # per row, the rows removed with it
model.fit_ncv(X, y)                                                      # one row at a time
```

`fit_ncv` is opt-in and leaves `fit` and `fit_reml` unchanged. It currently
supports `SuperGLM` with a **Gaussian family and identity link** or a **Poisson
family and log link**, without covariate binning (`discrete=False`). See
[When to use it and when not](#when-to-use-it-and-when-not) for the
full list of limits.

## A real example: the Spanish health portfolio

Consider a public Spanish health insurance portfolio: 210,998 insured-year rows
across 41,256 policies (families and collectives), recording 3,820,391 medical
services over 204,987.7 insured-years (2017 to 2019). The model is a Poisson
frequency model with log exposure as offset, a fine age curve for women
(`k=48`, roughly a knot every two years of age) plus a male age-difference
interaction, a spline for years with the insurer (`seniority`, `k=20`), and six
underwriting factors:

```python
import numpy as np
from superglm import Categorical, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={
        "age": Spline(kind="ps", k=48),
        "seniority": Spline(kind="ps", k=20),
        "gender": Categorical(),
        "product": Categorical(),
        "policy_type": Categorical(),
        "reimbursement": Categorical(),
        "channel": Categorical(),
        "habitat": Categorical(),
        "climate": Categorical(),
    },
    interactions=[("age", "gender")],
    discrete=False,
)
model.fit_ncv(X, y, offset=np.log(exposure), groups=df["ID_policy"])
```

`ID_policy` is not a term of the model and never enters a prediction. It only
tells `fit_ncv` which rows leave the fit together: every insured-year of a
policy is held out at once.

Fitting the same model with `fit_reml` and with `fit_ncv(..., groups=df["ID_policy"])`
produces very different age curves:

```{figure} ../images/ncv/health_age_curves.png
:name: fig-ncv-health-age-curves
:alt: Expected medical services per insured-year by age for women and men under fit_reml and fit_ncv with policies held out, with percentage ratio panels underneath showing fit_reml oscillating up to 13 percent around fit_ncv.

Expected medical services per insured-year by age (reference member) on the Spanish health portfolio. `fit_reml` produces jagged single-age wiggles (20 turning points for women, 27 for men); `fit_ncv` with whole policies held out produces graduated curves (4 turning points each).
```

Because medical service counts are heavily overdispersed relative to a Poisson
law, `fit_reml` treats single-age fluctuations in 3,820,391 services as real
signal and spends 133.5 total effective degrees of freedom (46.4 on women's
age, 44.1 on men's age difference, 19.0 on seniority). Its curves turn 20 times
for women and 27 times for men, sitting between 9.9% below (at age 12) and
13.0% above (at age 0) `fit_ncv`'s curve for women, and between 8.2% below (at
age 89) and 8.8% above (at age 40) for men. Holding out each policy with
`fit_ncv` converges in 13 evaluations at 74.1 total degrees of freedom (20.4 on
women's age, 11.1 on men's difference, 18.6 on seniority) and leaves 4 turning
points on each age curve.

### Comparing declared variance laws against `fit_ncv`

A natural question is whether declaring a more flexible count or variance law
inside `fit_reml` solves the same problem. To test that, the committed script
`docs/scripts/ncv_health_seven_methods.py` splits the portfolio into five folds
of whole policies and scores seven ways of choosing the smoothing on policies
each fit never saw — all evaluated on the same held-out Poisson deviance per
row:

```{figure} ../images/ncv/health_seven_methods.png
:name: fig-ncv-health-seven-methods
:alt: Dot and mean-diamond plot comparing six smoothing methods against Poisson fit_reml across five folds of unseen policies on the Spanish health book, with mean EDF annotated on the right.

Held-out Poisson deviance per row less Poisson `fit_reml`'s (24.282348, mean EDF 133.6) across five folds of unseen policies. Dots show individual folds, diamonds show the five-fold mean, and values to the left of the red line are better.
```

| Method | Mean EDF | Deviance less Poisson `fit_reml` | Fold SD | Folds better |
|---|---:|---:|---:|---:|
| `fit_reml`, Poisson (baseline: 24.282348) | 133.6 | 0 | — | — |
| `fit_reml`, Tweedie $p = 1.01$ (fold-mean scale 34.4) | 129.1 | −0.001383 (−0.006%) | 0.000796 | 5 of 5 |
| `fit_reml`, Tweedie $p = 1.2$ (fold-mean scale 20.2) | 91.5 | −0.012483 (−0.051%) | 0.008656 | 5 of 5 |
| `fit_reml`, Tweedie $p = 1.5$ (fold-mean scale 9.1) | 81.0 | −0.011221 (−0.046%) | 0.019818 | 3 of 5 |
| `fit_reml`, negative binomial ($\theta$ 0.504 to 0.507) | 76.3 | +0.009688 (+0.040%) | 0.035896 | 1 of 5 |
| `fit_ncv`, single rows held out | 78.1 | −0.014494 (−0.060%) | 0.007814 | 5 of 5 |
| `fit_ncv`, policies held out | 73.2 | −0.015310 (−0.063%) | 0.008494 | 5 of 5 |

Three things stand out:

- **`fit_reml`'s smoothing moves with the variance law declared.** Across the
  five laws, mean EDF runs from 133.6 down to 76.3, and held-out deviance moves
  both ways: Tweedie $p = 1.01$ estimates a fold-mean scale of 34.4 yet still
  keeps 129.1 EDF, Tweedie $p = 1.2$ improves all 5 folds (−0.012483), Tweedie
  $p = 1.5$ is mixed (3 of 5 folds), and negative binomial is worse on 4 of 5
  folds (+0.009688).
- **`fit_ncv` declares no overdispersion law and predicts unseen policies as
  well as the best declared law, without your having to find it.** With
  policies held out it lands at 73.2 EDF and lowers held-out deviance by
  0.015310 (0.063%, SD 0.008494) against Poisson `fit_reml`, on all 5 of 5
  folds, while removing the single-age wiggles. Its margin over the best
  declared law, Tweedie $p = 1.2$, is smaller than the spread between folds.
- **On this book, single rows already get most of the gain.** Holding out
  single rows alone reaches −0.014494 (−0.060%, 5 of 5 folds, 78.1 EDF).
  Holding out whole policies instead of single rows changes deviance by
  −0.000816 (SD 0.001543, better on 3 of 5 folds). Here the improvement comes
  chiefly from not trusting the Poisson count law, not from grouping rows by
  policy.

### Data, licence and changes

The real-data figures are generated from public datasets that are fetched, never
redistributed. To fetch and verify both Spanish books into `data/`:

```bash
uv run python scripts/fetch_spanish_books.py --dest data/
```

The health book is Lledó, J., Espinosa Adamez, P. and Perez Gimenez, V. (2025),
*Dataset of health insurance portfolio*, version 4, Mendeley Data
([doi:10.17632/386vmj2tbk.4](https://doi.org/10.17632/386vmj2tbk.4)), licensed
under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0). Changes made:
the fetch script converts the first sheet of the spreadsheet (228,711 rows by
42 columns) to parquet without altering rows or columns; before fitting, the
figure scripts drop 8,645 rows with zero exposure (which carry 24,663 services
and cannot be placed on a rate) and 9,068 dental-only rows, cap age at 90 and
years with the insurer at 40, and omit post-underwriting columns (`premium`,
`cost_claims_year`, `lapse`).

## Where it changes nothing: the Spanish motor book

`fit_ncv` is not always different from `fit_reml`. On the public Spanish motor
book by coverage — 353,576 rows across 185,407 insureds, 37,695 liability
claims over 249,980.1 exposure-years (2022 to 2024) — fitting the same Poisson
liability frequency model with `fit_reml` and with `fit_ncv(..., groups=insured)`
produces curves that lie on top of each other:

```{figure} ../images/ncv/motor_curves_agree.png
:name: fig-ncv-motor-curves-agree
:alt: Driver-age and vehicle-age liability claim frequency relativities on the Spanish motor book under fit_reml and fit_ncv with insureds held out, lying nearly on top of each other.

Driver-age and vehicle-age liability frequency relativities on the Spanish motor book. `fit_reml` and `fit_ncv` with insureds held out agree closely across the bulk of the exposure.
```

Across the whole book, total EDF is 64.3 under `fit_reml` and 61.0 under
`fit_ncv` (driver age 7.2 against 6.0; vehicle age 7.8 against 7.2). Over the
central 90% of exposure, the curves differ by at most 2.0% on driver age (ages
31 to 69) and 1.3% on vehicle age (ages 8 to 46), reaching 6.5% only at driver
age 85 (the end of the range) and 2.6% at vehicle age 0. On five folds of
unseen insureds, held-out Poisson deviance per row is 0.507529 for `fit_reml`
(mean EDF 63.2) and 0.507556 for `fit_ncv` (mean EDF 63.5): a difference of
+0.000028 (+0.0055%, SD 0.000026; `fit_ncv` better on 1 of 5 folds). Because
the held-out criterion is flat there, `fit_ncv`'s fold EDF moves slightly more
from fold to fold (59.1 to 66.8 against 61.5 to 65.5) with no effect on the
score.

Why they agree: a policyholder's liability claim counts barely move together
from one year to the next (correlation 0.054 across 168,169 consecutive
insured-year pairs), so holding out an insured's years together leaves almost
the same information as treating rows as independent.

The motor book is Espinosa, P., Lledó, J. and Atance, D. (2026), *A detailed
dataset of motor insurance policies with coverage-specific financial
information*, version 1, Mendeley Data
([doi:10.17632/sw4jmdb2sm.1](https://doi.org/10.17632/sw4jmdb2sm.1)), licensed
under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0). Changes made:
the fetch script converts the `;`-separated CSV (354,140 rows by 47 columns) to
parquet; before fitting, the figure script drops 49 rows with
zero liability exposure and 515 rows missing vehicle value, vehicle age or
licence age, lumps vehicle brands with fewer than 2,000 rows into `"OTHER"`,
and caps continuous covariates at fixed limits.

## What is held out

A neighbourhood has two parts: the rows **removed from the fit** so they cannot
influence the leave-out coefficients, and the subset of those rows that are
**scored** in the held-out deviance. Separating the two lets a split exclude
buffer rows or future rows from the fit without scoring them twice.

```{figure} ../images/ncv/holdout_forms.png
:name: fig-ncv-holdout-forms
:alt: Four panels showing which rows stay in the fit, which leave the fit and are scored, and which leave the fit without being scored for single rows, whole groups, a spatial ring of neighbours, and a forward-in-time split.

The four hold-out forms of `fit_ncv`. Grey dots stay in the fit, red dots leave the fit and are scored, and amber dots leave the fit to prevent leakage without being scored in that neighbourhood.
```

Each call pattern sets the neighbourhoods directly:

- **Single rows** (`fit_ncv(X, y)`): every row is removed and scored on its own.
- **Whole groups** (`groups=df["policy_id"]`): every distinct label is removed
  and scored as one block.
- **Splitter folds** (`cv=GroupKFold(5), groups=...` or `cv=TimeSeriesSplit(5)`):
  any splitter accepted by `cross_validate`. For each `(train, test)` pair, the
  rows outside `train` are removed and the `test` rows are scored. For
  `GroupKFold` and `KFold`, removed and scored rows match; for
  `TimeSeriesSplit`, earlier periods stay in the fit, the next period is scored,
  and still-later periods are excluded from the fit. `cv.split` is called once
  and frozen for the whole search, so a shuffling splitter cannot shift the
  criterion between steps.
- **Row-wise neighbours** (`neighbours=near`): for each row, the positions of
  the rows removed with it (as a list of index arrays or a sparse $n \times n$
  matrix); only the focal row itself is scored, and the row is included in its
  own removed set automatically if omitted.

Pass at most one hold-out form: `groups` alone, `cv` (optionally with `groups`
forwarded to `cv.split`), `neighbours` alone, or none of them for single rows.
Combining `neighbours` with `cv` or `groups` is refused.

`sample_weight` and `offset` apply to every held-out prediction as well as to
the full fit. A row with zero weight stays out of every fit and adds nothing to
the criterion, though it is still predicted when scored. Under
`weight_semantics="frequency"`, a row's replicates leave the fit together.

### Textbook case: rows in a group that share noise

When rows within a group genuinely share a large unobserved shock, holding out
whole groups is what stops the curve chasing it. In a simulated Gaussian book
of 600 rows across 30 groups of 20 rows sitting in local stretches of $x$
(group shock SD 0.45 on top of row noise SD 0.12 around
$f(x) = \sin(2\pi x) + 0.5x$):

```python
from superglm import Spline, SuperGLM

spec = {"x": Spline(kind="ps", k=25)}

reml_model = SuperGLM(family="gaussian", features=spec).fit_reml(X, y)
ncv_model = SuperGLM(family="gaussian", features=spec).fit_ncv(X, y, groups=groups)
```

```{figure} ../images/ncv/grouped_curve_comparison.png
:name: fig-ncv-grouped-curve
:alt: Scatter of 600 simulated grouped observations and 30 group means with the true smooth curve, an overly wiggly fit_reml curve, and a smooth fit_ncv curve tracking the truth.

Simulated grouped data (30 groups of 20 rows). `fit_reml` bends toward each displaced cluster; holding out whole groups with `fit_ncv` recovers the smooth curve.
```

`fit_reml` reads the 20 agreeing rows in each cluster as signal
($\lambda = 0.25$, EDF 17.3, RMSE 0.205 against the truth). Removing all 20
rows of each group before scoring moves `fit_ncv` to $\lambda = 343.2$
(EDF 4.0) and cuts RMSE by 63.2%, to 0.075. Sweeping $\lambda$ across seven
decades shows the leave-group-out criterion `fit_ncv` walks down in 5 outer
steps (from mean held-out error 0.571 at `fit_reml`'s choice to 0.421, a 26.4%
drop):

```{figure} ../images/ncv/criterion_vs_smoothing.png
:name: fig-ncv-criterion-vs-smoothing
:alt: Leave-group-out mean squared error plotted against smoothing parameter lambda on a log axis, with fit_reml marked high on the overfitted left side and fit_ncv marked at the minimum.

Leave-group-out mean squared error per row on the grouped dataset above, with `fit_reml` on the overfitted left side and `fit_ncv` at the minimum.
```

### Spatial grids with ring neighbourhoods

On a regular spatial mesh such as an H3 hexagonal grid, adjacent cells share
local weather, road layout or neighbourhood risk, so holding out one cell alone
still leaves its neighbours in the fit. With `neighbours=`, each cell is scored
after removing both the cell and its surrounding ring (either as a list of
row-position arrays or a sparse $n \times n$ adjacency matrix):

```python
import numpy as np
from superglm import Spline, SuperGLM

# Hexagonal grid in odd-r offset coordinates converted to axial (q, r)
q = col - (row // 2)
r = row
dq = np.abs(q[:, None] - q[None, :])
dr = np.abs(r[:, None] - r[None, :])
ds = np.abs((q[:, None] + r[:, None]) - (q[None, :] + r[None, :]))
hex_dist = (dq + dr + ds) // 2

# For each cell, remove the cell itself and its 2-ring neighbours (up to 19 cells)
neighbours = [np.flatnonzero(hex_dist[i] <= 2) for i in range(len(q))]

model = SuperGLM(
    family="gaussian",
    features={"east": Spline(kind="cr", k=8), "north": Spline(kind="cr", k=8)},
    interactions=[("east", "north")],
)
model.fit_ncv(X, y, neighbours=neighbours)
```

On a simulated $16 \times 16$ hexagonal grid (256 cells) with a local 1-ring
disturbance, removing each cell's 2-ring neighbourhood (up to 19 cells; corner
and edge cells remove 7 to 18) drops EDF from 52.2 (`fit_reml`, RMSE 0.374) to
30.8 (`fit_ncv`, RMSE 0.302, a 19.3% reduction), though some pockets still
remain:

```{figure} ../images/ncv/spatial_ring_grid.png
:name: fig-ncv-spatial-ring-grid
:alt: Three hexagonal grid heat maps comparing the true spatial risk surface with an example 2-ring highlighted, a blotchy fit_reml surface, and a smoother fit_ncv surface using 2-ring neighbours.

Simulated $16 \times 16$ hexagonal grid (256 cells) with locally correlated spatial disturbances. Left: true spatial risk surface with one interior cell (red) and its 18 ring neighbours (amber) outlined. Centre: `fit_reml`. Right: `fit_ncv` with 2-ring neighbourhoods.
```

## Reading held-out results by group

Once `fit_ncv` finishes, the held-out prediction for every scored row is
already in hand — no extra refits are needed. Two accessors return copies of
those results as pandas DataFrames:

- {py:meth}`~superglm.SuperGLM.ncv_heldout` returns one row per scored
  `(neighbourhood, row)` pair with columns `neighbourhood`, `label`, `row`,
  `weight`, `actual`, `heldout`, `fitted` and `heldout_deviance`.
- {py:meth}`~superglm.SuperGLM.ncv_heldout_groups` aggregates those pairs by
  group into `rows`, `weight`, `actual`, `heldout_expected`,
  `actual_over_expected` and `heldout_deviance`.

For a `groups=` or `cv=` fit, `ncv_heldout_groups()` groups by the fit's own
groups or splits automatically. After any fit (including single-row or
`neighbours=`), pass `by=` with one label per row of `X` to aggregate by
another key such as calendar year or product. When a model is constructed with
`retain_fit_state=False`, the per-group table of a `groups=` or `cv=` fit is
still kept, while row-level arrays are released so `ncv_heldout()` and
`ncv_heldout_groups(by=...)` are unavailable. Where a group's held-out expected
total is zero or negative, `actual_over_expected` is reported as missing
(`NaN`).

On the Spanish health fit above (`groups=df["ID_policy"]`), calling
`ncv_heldout_groups()` gives every policy's experience against a fit that never
saw it — where experience rating of a collective starts, at no extra fit:

```python
by_policy = model.ncv_heldout_groups()
large_policies = by_policy[by_policy["rows"] >= 100].sort_values("rows", ascending=False)
by_year = model.ncv_heldout_groups(by=df["period"])
```

```{figure} ../images/ncv/health_heldout_policies.png
:name: fig-ncv-health-heldout-policies
:alt: Log-log scatter plot of actual against held-out expected medical services for the 23 Spanish health policies with 100 or more rows, with a plus or minus 25 percent band and standout policies 4728, 28288, 32614 and 7303 labelled.

Actual against held-out expected services for the 23 Spanish health policies with 100 or more insured-year rows (7,832 rows in all), read directly from `model.ncv_heldout_groups()`. Fourteen of the 23 stand more than 25% away from what the rest of the portfolio predicts for them.
```

Out of 41,256 policies, 23 have 100 or more insured-year rows (7,832 rows in
all), and 14 of them sit more than 25% away from their held-out expected
services. The table below shows the largest policies alongside the four
standout collectives labelled on the chart:

| Policy | `rows` | `actual` | `heldout_expected` | `actual_over_expected` |
|---|---:|---:|---:|---:|
| `31307` | 2,507 | 39,086 | 39,598.0 | 0.99 |
| `31306` | 925 | 16,835 | 14,621.0 | 1.15 |
| `31133` | 777 | 7,687 | 7,798.9 | 0.99 |
| `28288` | 519 | 0 | 1,508.3 | 0.00 |
| `36163` | 463 | 4,269 | 5,747.3 | 0.74 |
| `4728` | 155 | 3,245 | 590.0 | 5.50 |
| `7303` | 116 | 274 | 815.3 | 0.34 |
| `32614` | 103 | 334 | 1,632.0 | 0.20 |

Policy `28288` has 519 insured-year rows and not a single recorded medical
service against 1,508.3 held-out expected — a question about the cover or the
data, surfaced without another fit. Policy `4728` runs at 5.50 times expected
(3,245 services against 590.0 held-out expected), while `32614` (0.20) and
`7303` (0.34) sit far below. Because each policy's expected total comes from
the fit without that policy, a large collective cannot pull the curves toward
its own history and mask its gap.

**Know where the table adds information and where it does not.** Calling
`ncv_heldout_groups(by=year)` gives actual-over-expected ratios of 0.893 (2017,
70,917 rows), 0.931 (2018, 68,910 rows) and 1.177 (2019, 71,171 rows) — showing
that utilisation jumped in 2019, which no choice of smoothing answers. Those
ratios match the ordinary in-sample fitted actual-over-expected (0.893, 0.932
and 1.177) to three decimals. The same holds for large product segments:
product `S` (169,634 rows) reads 1.000 held out against 1.000 fitted, and
product `P` (38,468 rows) reads 1.000 against 1.000. For any broad segment, one
policy is a tiny share of the total, so held-out and fitted ratios coincide.
The table earns its place on the held-out groups themselves, and on small
factor levels (such as product `I`, 2,896 rows: 0.989 held out against 1.000
fitted).

## When to use it and when not

`fit_ncv` is not a drop-in replacement for `fit_reml`. On real books, the two
Spanish portfolios above mark out the boundary:

- Reach for `fit_ncv` when the declared count law is badly misspecified (as in
  the Spanish health book, where Tweedie $p = 1.01$ estimates a scale of 34.4
  and Poisson `fit_reml` under-smooths the age curves) or when rows inside a
  neighbourhood genuinely share unobserved risk, provided there are enough
  neighbourhoods to judge held-out error reliably.
- Stay with `fit_reml` on motor-like frequency books where counts are close to
  Poisson and one insured's claims barely correlate across years (on the
  Spanish motor book the two methods agree to within 0.0055% while `fit_ncv`
  does several times the work), or when you have only a handful of
  neighbourhoods.

To make the small-neighbourhood and independent-row trade-offs concrete, the
committed study script repeats the simulated grouped curve above over 50 random
seeds across three settings:

1. **Correlated groups (30 groups of 20 rows)**: rows in a group share a shock
   and are held out together.
2. **Few groups (5 groups of 120 rows)**: rows are correlated within groups,
   but only 5 groups exist — below the 10-neighbourhood warning threshold.
3. **Independent rows (600 rows, no group shock)**: single-row `fit_ncv(X, y)`
   against `fit_reml(X, y)`.

```{figure} ../images/ncv/when_to_use_comparison.png
:name: fig-ncv-when-to-use
:alt: Strip and summary plot of the ratio of fit_ncv RMSE to fit_reml RMSE across 50 seeds in three regimes: 30 correlated groups, 5 groups, and independent rows.

RMSE ratio (`fit_ncv` / `fit_reml`) against the true curve over 50 random seeds in each regime. Values below 1 (green side) mean `fit_ncv` is closer to the truth; values above 1 (red side) mean `fit_reml` is closer. Diamonds mark medians and thick bars span the interquartile range.
```

| Setting (50 seeds each) | `fit_ncv` closer | Median RMSE (`fit_reml`) | Median RMSE (`fit_ncv`) | Median ratio | 90th pct ratio | Worst ratio | Median work (`fit_reml` iter / `fit_ncv` evals) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correlated groups (30 groups × 20 rows) | 48 / 50 | 0.246 | 0.172 | 0.71× | 0.88× | 1.02× | 5 / 10 |
| Few groups (5 groups × 120 rows) | 21 / 50 | 0.348 | 0.367 | 1.00× | 1.12× | 1.94× | 6 / 8 |
| Independent rows (600 rows, single-row) | 29 / 50 | 0.052 | 0.052 | 0.98× | 1.17× | 1.76× | 6 / 9 |

Read the three rows plainly:

- **Where `fit_ncv` wins**: with 30 correlated groups, it beats `fit_reml` on
  48 of 50 seeds (median ratio 0.71×, worst 1.02×).
- **Where `fit_ncv` loses — few groups**: with 5 groups, every fit warns that
  the criterion is noisy (50 of 50 seeds), `fit_ncv` loses on 29 of 50 seeds,
  and the worst seed reaches 1.94× `fit_reml`'s error.
- **Where `fit_ncv` does not pay — independent rows**: with no shared group
  noise and a well-specified law, medians tie (0.0515 against 0.0520), mean
  RMSE is slightly higher for `fit_ncv` (0.0548 against 0.0542), occasional
  seeds under-smooth up to 1.76×, and it still does more fitting work (median
  10 inner solves against 6 REML iterations; on the 30-group model, also a
  median of 300 leave-out factorisations).

### Practical limits to keep in mind

- **Fewer than 10 neighbourhoods**: `fit_ncv` still fits, including for a
  single train/test split, but emits a `UserWarning` that the held-out
  criterion is noisy with fewer than 10 neighbourhoods (a rule of thumb, not a
  measured threshold). With only a handful of folds, the chosen smoothing can
  move noticeably between splits. Use more groups or folds where you can, or
  stick with `fit_reml`.
- **Forward-in-time splits (`TimeSeriesSplit`)**: forward-in-time hold-out is
  described under [What is held out](#what-is-held-out) rather than
  demonstrated on real data here, because public insurance books typically span
  only three to seven calendar years — giving just one to four yearly splits.
  With that few splits, `fit_ncv` emits the fewer-than-10-neighbourhoods
  warning and its smoothing choice is erratic (shrinking a curve to a straight
  line in one fit or under-smoothing in another). Reserve forward-in-time
  splits for books with enough periods (such as monthly or quarterly series) to
  form at least ~10 neighbourhoods.
- **Poisson one-step approximation**: for a Gaussian model with an identity
  link, every leave-out fit inside `fit_ncv` is exact. For a Poisson model with
  a log link, each leave-out fit is one Newton step from the full fit. Against
  exact brute-force leave-group-out refits in the study script, the relative
  error in the summed held-out deviance is 0.004% at 120 rows per coefficient
  (1,200 rows, 30 groups, 10 coefficients) and 0.036% at 21.4 rows per
  coefficient (600 rows, 15 groups, 28 coefficients), rising to 1.762% when rows
  per coefficient drop to 5.0 with large quarter-book folds (140 rows, 4
  groups, 28 coefficients). Treat Poisson `fit_ncv` carefully on very small
  books or very large folds.
- **Fitting cost**: `fit_ncv` typically takes about three to four times as long
  as `fit_reml`. On the public French motor book (freMTPL2), measured on one
  machine with default threads: `fit_reml` takes 3.9 to 4.0 s, `fit_ncv` with
  five folds takes 10.6 to 11.0 s, and `fit_ncv` with each policy held out
  takes 14.1 to 16.0 s. When the two methods choose the same curves, that extra
  work buys nothing.
- **Standard errors and scale**: `fit_ncv` chooses the smoothing parameters and
  then publishes the ordinary fit at those smoothing parameters. Standard
  errors, effective degrees of freedom and the Pearson scale are the same ones
  `fit()` gives at that fixed smoothing; standard errors are **not** corrected
  for residual dependence between rows.

### What is not supported yet

`fit_ncv` refuses unsupported configurations before running any fit, with an
error naming what is unsupported and what to do instead:

- **Other families and links**: only Gaussian with an identity link and Poisson
  with a log link are supported. Gamma, Tweedie, NegativeBinomial and Binomial
  models must choose smoothing with `fit_reml`.
- **Binned fits (`discrete=True`)**: `fit_ncv` runs on the exact route only
  (`discrete=False`). Use `fit_reml` for binned fits on very large books.
- **Selection penalties**: `selection_penalty` must be `None` or `0.0`. Use
  `select=True` on spline terms to let `fit_ncv` shrink a smooth's unpenalised
  part, or use `fit` / `fit_path` for sparse selection.
- **Shape-constrained terms**: monotone, convex, concave, capped or step-limited
  terms (`Constraint.fit.*`) are not supported.
- **Credibility terms**: `RandomEffect` and `FactorSmooth` are not supported
  (and `direct_solve="structured"` is refused before fitting).
- **Distributional models**: `SuperLSS` does not offer `fit_ncv`.
- **Factor levels isolated in one neighbourhood**: if every positive-weight row
  of a categorical level (including the reference level) lies inside a single
  neighbourhood, holding that neighbourhood out leaves nothing to identify that
  level. `fit_ncv` refuses before fitting and names the term, level and
  neighbourhood; merge the level or choose a split that divides its rows. A
  neighbourhood whose remaining rows fail to identify the model for another
  reason (such as two columns coinciding once that neighbourhood leaves) is
  refused before the search starts.

## How `fit_ncv` relates to `cross_validate`

{py:func}`~superglm.cross_validate` and {py:meth}`~superglm.SuperGLM.fit_ncv`
play complementary roles:

- **`cross_validate()` measures** a fitted model on held-out folds.
- **`fit_ncv()` chooses** the smoothing parameters inside a fit using the same
  kind of hold-out.

When you pass the same `cv` splitter and `groups` to both, on a model whose
knots and factor levels do not depend on the fold,
`model.ncv_diagnostics()["criterion_mean"]` equals
`cross_validate(..., fit_mode="fit").pooled_scores["deviance"]` at the chosen
smoothing (exact for Gaussian, within the one-step approximation for Poisson;
if knots or factor levels are rebuilt per fold, the two differ by that rebuild).

To check honestly on your own book whether choosing smoothing by `fit_ncv`
improves out-of-sample predictions over `fit_reml`, run an outer
`cross_validate` over your groups with `fit_mode="fit_ncv"` beside
`fit_mode="fit_reml"`:

```python
from sklearn.model_selection import GroupKFold
from superglm import Spline, SuperGLM, cross_validate

model = SuperGLM(family="gaussian", features={"x": Spline(kind="ps", k=25)})
outer = GroupKFold(n_splits=5)

reml_cv = cross_validate(model, X, y, cv=outer, groups=groups, fit_mode="fit_reml")
ncv_cv = cross_validate(model, X, y, cv=outer, groups=groups, fit_mode="fit_ncv")
```

How `fit_mode="fit_ncv"` runs inside each outer fold:

- **Inner hold-out**: when `groups` is passed to `cross_validate`, it is
  forwarded to the outer splitter and sliced to each fold's training rows for
  `fit_ncv`, so by default every training group in that fold is held out in
  turn. Pass an inner splitter as `fit_kwargs={"cv": inner}` to hold out inner
  folds instead (the same splitter object is reused across folds and receives
  the sliced `groups`); omit both `groups` and an inner `cv` for single-row
  hold-out. `fit_kwargs` cannot set `groups` (pass it to `cross_validate` so it
  slices per fold) or `neighbours` (whose row positions index the whole frame).
- **Cost and `warm_start`**: every outer fold runs a whole `fit_ncv` smoothing
  search on its training rows (several `fit_reml` fits' worth each). Unlike
  `fit_mode="fit_reml"`, `warm_start` defaults to `False` for `"fit_ncv"`,
  because the held-out criterion can have more than one local minimum and
  starting later folds from the first converged fold's smoothing can move them
  to a different stop than their own rows would choose.
- **Fold table (`result.fold_scores`)**: adds `n_ncv_iter` (accepted smoothing
  search steps) and `warm_started`. A fold's `converged` is `True` only when
  both its smoothing search and its final coefficient fit converged; a fold
  whose search stops short is still fitted and scored, with `converged=False`
  and a `ConvergenceWarning`.
- **Per-fold refusals**: model-level refusals (an unsupported family,
  `discrete=True`, or a `SuperLSS`) raise before any fold runs. A refusal that
  depends on a fold's rows — for instance, only one group left in a training
  fold, or a factor level whose weighted rows all sit in one group of that fold
  — fails that fold only: under the default `error_score=np.nan`, `fold_scores`
  records `NaN` for that fold's scores and logs a warning naming the fold while
  the remaining folds are scored (under `error_score="raise"`, the exception
  carries a note naming the fold). Choose outer and inner groupings that leave
  several groups in every training fold.

Passing `groups=` to `cross_validate(..., fit_mode="fit_ncv")` uses those
groups both to split the outer folds and to form the inner hold-out inside each
fold. To score single-row `fit_ncv` on outer folds of whole policies or groups
(as in the seven-methods comparison on the health book above), precompute the
outer `(train, test)` splits once and pass them through a small fixed-fold
splitter without `groups=`:

```python
from sklearn.model_selection import GroupKFold
from superglm import cross_validate


class FixedFolds:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y=None, groups=None):
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
ncv_rows_cv = cross_validate(
    model, X, y, cv=FixedFolds(folds), fit_mode="fit_ncv"
)
```

Running the grouped call above on the 30-group simulated Gaussian book from earlier
(5 outer folds, each training on 24 groups of 20 rows and holding out those 24
groups in turn inside `fit_ncv`) gives:

| Outer fold | `fit_reml` deviance | `fit_ncv` deviance | Delta (`ncv − reml`) | `fit_reml` EDF | `fit_ncv` EDF | `n_ncv_iter` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.784 | 0.480 | −0.304 | 18.94 | 5.22 | 5 |
| 1 | 0.662 | 0.512 | −0.150 | 16.09 | 5.33 | 7 |
| 2 | 0.393 | 0.301 | −0.092 | 19.18 | 4.85 | 7 |
| 3 | 0.496 | 0.418 | −0.078 | 17.83 | 4.60 | 6 |
| 4 | 0.563 | 0.523 | −0.040 | 19.34 | 4.73 | 6 |
| **Mean / pooled** | **0.579** | **0.447** | **−0.133 (−22.9%)** | — | — | — |

All five folds converged in both modes with zero warnings (`effective_df` in
`fold_scores` includes the intercept, and `warm_started` was `False` on every
`fit_ncv` fold). `fit_ncv` lowered held-out deviance on all 5 folds, cutting
pooled deviance by 22.9%. Compare the fold-level deltas (not just one average)
before switching a production tariff from `fit_reml` to `fit_ncv`.

## Reading diagnostics, stop reasons and warnings

Call {py:meth}`~superglm.SuperGLM.ncv_diagnostics` after any `fit_ncv` fit to
inspect how the search went as plain Python values:

```python
diag = model.ncv_diagnostics()
diag["converged"]          # True when both the search and final coefficient fit passed
diag["termination_reason"] # why the smoothing search stopped
diag["criterion_mean"]     # held-out deviance per unit of likelihood size
diag["lambdas"]            # chosen smoothing parameter per penalty component
diag["neighbourhoods"]     # form, count, min/median/max size, scored pairs
diag["at_bound"]           # components sitting on the floor or ceiling of the search range
```

`converged` is `True` only when **both** `search_converged` and
`coefficients_converged` are `True`. A search or final coefficient fit that
stops early still returns a fitted model and discloses the stop with a
`ConvergenceWarning`; it is never silently treated as converged and never
refused. `model.summary()` shows `Method: NCV` and repeats any non-convergence
or bound note, while `model.reml_diagnostics()` answers with `enabled=False`
and `criterion="ncv"` (no REML score) and points to `ncv_diagnostics()`.

### Stop reasons (`termination_reason`)

| `termination_reason` | Search converged? | What it means | What to do |
|---|---|---|---|
| `"gradient"` | Yes | Every estimated smoothing parameter's slope of held-out deviance in $\log \lambda$ met `ncv_tol` relative to the criterion (a first-order stop of a local search). | Nothing required; inspect `at_bound` if any term sits on a limit. |
| `"no_smoothing_parameters"` | Yes | The model had no estimated smoothing parameter (for example, all penalties were fixed with `LambdaPolicy.fixed(...)`). | Nothing required; held-out tables still report the fit's held-out predictions. |
| `"max_iter"` | No | The search hit `max_ncv_iter` (default 100) before slopes met `ncv_tol`. Returned at the lowest held-out deviance reached. | Refit with a larger `max_ncv_iter` and `lambda2_init=model.ncv_diagnostics()["lambdas"]` to continue from here. |
| `"line_search"` | No | No trial step lowered the held-out deviance further from the returned smoothing; the criterion is flat or uneven nearby. | Refit with `lambda2_init=model.ncv_diagnostics()["lambdas"]` to search again from here, or raise `ncv_tol` if the curve is already stable. |
| `"criterion_settled"` | No | Steps lowered the held-out deviance by less than `ncv_tol**2` relative to it before the slopes met `ncv_tol`. | Same as `"line_search"`: check whether the curve moves on a warm-started refit, or widen `ncv_tol`. |
| `"no_finite_criterion"` | No | No smoothing setting tried yielded a finite held-out deviance. Model is returned as a coefficient fit at the starting smoothing, with no held-out tables. | Read `diag["infeasible_reasons"]` and the warning message: raise `max_pirls_iter` if inner fits failed, check extreme responses/offsets, drop redundant terms, or adjust neighbourhoods. |

### Warnings you may see

- **Fewer than 10 neighbourhoods (`UserWarning`)**: see
  [Practical limits to keep in mind](#practical-limits-to-keep-in-mind).
- **Unconverged search or final coefficient fit (`ConvergenceWarning`)**:
  names which stage stopped and why. If `search_converged` is `True` but
  `coefficients_converged` is `False`, the smoothing search succeeded while the
  final coefficient fit at that smoothing stopped early (inspect
  `diag["coefficient_termination"]` and raise `max_pirls_iter`); held-out
  tables are then omitted (`heldout_available` is `False` and `heldout_failure`
  records why).
- **Coefficient fits at the limit of floating-point accuracy
  (`ConvergenceWarning`)**: when the response, less its offset, lies very far
  from zero compared with its spread, the coefficient fits cannot meet their
  convergence test (`diag["coefficient_termination"]` is `"score_stagnated"`).
  The model is still returned with the smoothing the search chose and its
  held-out tables, and reads as not converged. Subtract a constant from the
  response before fitting and add it back to the predictions.
- **Smoothing parameter on a bound (`ConvergenceWarning`)**: names every
  estimated component sitting at the floor or ceiling of the search box
  (`diag["at_bound"]`), and notes separately if the held-out deviance was still
  falling at that limit (`diag["bound_pressure"]`). A term at the floor is
  nearly unpenalised and can chase noise (use fewer basis functions `k` or fix
  its smoothing parameter). A term at the ceiling is shrunk close to its
  unpenalised shape (a straight line for an ordinary spline; replace with a
  linear term if expected, or check the covariate).
- **Unknown key in `lambda2_init` (`UserWarning`)**: a name in `lambda2_init`
  matched no smoothing component or term and was ignored. Use the keys from
  `model.ncv_diagnostics()["lambdas"]` or term names.

## Comparing the three fit methods

| Method | What chooses the smoothing | What it assumes | When to use it |
|---|---|---|---|
| `fit()` | Fixed by you (`spline_penalty`) | Nothing about automatic smoothness; you set the penalties | Sparse screening (`selection_penalty > 0`), fixed-penalty challengers, or reproducing known penalties |
| `fit_reml()` | Marginal / restricted likelihood (REML / Laplace approximation) | The declared variance/count law holds and rows are conditionally independent given the covariates | Default for pricing models, motor-like books where the declared law fits reasonably well, large binned books (`discrete=True`), Gamma/Tweedie/NB/Binomial, monotone terms, `RandomEffect` and `FactorSmooth` |
| `fit_ncv()` | Held-out deviance over neighbourhoods you define | Observations outside a neighbourhood predict it without sharing its noise (does not trust the count law or row independence) | Gaussian/identity or Poisson/log exact fits when counts are far more variable than the declared law allows, or when rows share group, period or spatial risk and you have enough neighbourhoods (at least ~10) |

## How `fit_ncv` works

For a trial vector of log smoothing parameters $\rho_j = \log \lambda_j$,
`fit_ncv` solves the penalised GLM once on all rows to get the full-fit
coefficients $\hat{\boldsymbol{\beta}}$ and penalised Hessian $\mathbf{H}$.
For each neighbourhood $k$ that removes rows $a(k)$ and scores rows
$d(k) \subseteq a(k)$, it forms the downdated Hessian
$\mathbf{H}_{-a} = \mathbf{H} - \mathbf{X}_a^\top \mathbf{W}_a \mathbf{X}_a$
and takes one Newton step of the leave-out problem from $\hat{\boldsymbol{\beta}}$
to obtain the leave-out predictor on $d(k)$ (Wood, 2024).

- For a Gaussian family with an identity link, the working weights do not move
  with $\boldsymbol{\beta}$, so that single Newton step is the **exact**
  leave-out fit.
- For a Poisson family with a log link, it is a one-step approximation to
  refitting each neighbourhood to convergence, avoiding a full iterative solve
  per neighbourhood.
- The criterion is the summed weighted deviance over all scored pairs, and its
  gradient in $\log \lambda_j$ is computed analytically via one adjoint solve
  per neighbourhood regardless of how many smoothing parameters the model has.
- A bound-constrained quasi-Newton search (L-BFGS-B) minimises the criterion
  over the estimated smoothing parameters, and the model is then fitted and
  published at the chosen smoothing using the same steps as `fit()`.

**For readers familiar with `mgcv`:** R's `mgcv` package offers the same method
via `method="NCV"`. Both engines optimise the same score — the summed held-out
deviance with prior weights, not averaged and not scaled — and choose the same
fit on the same design and penalties. `fit_ncv`'s standard errors are the
ordinary ones at the chosen smoothing, whereas `mgcv`'s default on single-row
and row-with-neighbours structures adjusts them for the neighbourhood
structure, so the two differ there unless `mgcv` is told
`nei$jackknife = FALSE`. When comparing the two engines directly, compare
without an offset.

## References

- Wood, S. N. (2024). On Neighbourhood Cross Validation.
  [arXiv:2404.16490](https://arxiv.org/abs/2404.16490).
- Lledó, J., Espinosa Adamez, P. and Perez Gimenez, V. (2025). *Dataset of
  health insurance portfolio*, version 4. Mendeley Data,
  [doi:10.17632/386vmj2tbk.4](https://doi.org/10.17632/386vmj2tbk.4),
  [Mendeley record](https://data.mendeley.com/datasets/386vmj2tbk/4).
  Licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0).
- Espinosa, P., Lledó, J. and Atance, D. (2026). *A detailed dataset of motor
  insurance policies with coverage-specific financial information*, version 1.
  Mendeley Data,
  [doi:10.17632/sw4jmdb2sm.1](https://doi.org/10.17632/sw4jmdb2sm.1),
  [Mendeley record](https://data.mendeley.com/datasets/sw4jmdb2sm/1).
  Licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0).

## Next steps

- [Choose a fitting path](choose-a-fitting-path.md)
- [Validation and model comparison](compare-models-on-holdout.md)
- [How REML chooses smoothness](../explanation/how-reml-chooses-smoothness.md)
- {py:meth}`superglm.SuperGLM.fit_ncv`,
  {py:meth}`superglm.SuperGLM.ncv_diagnostics`,
  {py:meth}`superglm.SuperGLM.ncv_heldout`,
  {py:meth}`superglm.SuperGLM.ncv_heldout_groups`
