# Choose smoothing by held-out prediction

A spline term needs a smoothing parameter to balance following the signal
against chasing noise. {py:meth}`~superglm.SuperGLM.fit_reml` chooses that
smoothing from a likelihood criterion (REML) that assumes the rows are
independent once you account for the covariates.

On many insurance books, rows are not independent:

- **Repeated policies or fleets**: several policy years or vehicles from the
  same insured share unobserved risk habits.
- **Calendar periods**: claims in the same month or accident year share weather
  events, court decisions or claims-inflation shifts.
- **Spatial grids**: nearby cells or postcodes share local weather, road layout
  or neighbourhood effects.

When rows that belong together move together, an independence criterion reads
their agreement as a real feature of the curve and chooses a smoothing
parameter that is too small. The fitted curve bends to chase those group or
local shocks.

{py:meth}`~superglm.SuperGLM.fit_ncv` chooses the smoothing parameters by
**held-out prediction** instead (neighbourhood cross-validation; Wood, 2024).
You say which rows belong together, and `fit_ncv` removes each neighbourhood
from the fit together, predicts the held-out rows, and picks the smoothing that
minimises the summed held-out deviance:

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

## When rows in a group share noise

Consider a simulated Gaussian book of 600 rows split into 30 groups of 20 rows
each (for example, 30 fleets or multi-year policies sitting in different parts
of a continuous covariate $x$). Every row in a group shares a group shock
(standard deviation 0.45) on top of smaller row-level noise (standard deviation
0.12) around a smooth true curve $f(x) = \sin(2\pi x) + 0.5x$. Fitting the same
25-basis P-spline with `fit_reml` and with `fit_ncv(..., groups=groups)` gives
two very different curves:

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

Because 20 rows in the same group all sit above or below the curve together,
`fit_reml` picks a small smoothing parameter ($\lambda = 0.25$, EDF 17.3) and
chases every group's shock (RMSE 0.205 against the truth). `fit_ncv` removes
all 20 rows of each group before predicting them; a bump fitted to one group
hurts predictions for neighbouring groups, so the search moves to
$\lambda = 343.2$ (EDF 4.0) and cuts RMSE by 63.2%, to 0.075.

### Held-out error against smoothing

Sweeping $\lambda$ by hand across seven decades on the same grouped dataset
shows what `fit_ncv` optimises:

```{figure} ../images/ncv/criterion_vs_smoothing.png
:name: fig-ncv-criterion-vs-smoothing
:alt: Leave-group-out mean squared error plotted against smoothing parameter lambda on a log axis, with fit_reml marked high on the overfitted left side and fit_ncv marked at the minimum.

Leave-group-out mean squared error per row on the grouped dataset above, with `fit_reml` on the overfitted left side and `fit_ncv` at the minimum.
```

At small $\lambda$ the spline follows individual group shocks (`fit_reml`'s
$\lambda = 0.25$ gives mean held-out error 0.571); at very large $\lambda$ it
collapses toward a straight line and misses the true curvature. `fit_ncv` uses
the analytic gradient to walk to the bottom at $\lambda = 343.2$ (mean held-out
error 0.421, a 26.4% drop) in 5 outer steps, without a grid search.

## Spatial grids with ring neighbourhoods

Spatial pricing models often aggregate policies onto a regular mesh such as an
H3 hexagonal grid. Adjacent hex cells share local weather, road conditions and
unobserved neighbourhood risk, so holding out a single cell still leaves its
six immediate neighbours in the fit to copy the same local disturbance.

With `neighbours=`, each cell is scored after removing both the cell and its
surrounding ring. You can pass either a list of row-position arrays (one per
row) or a sparse $n \times n$ adjacency matrix whose stored entries mark the
removed rows:

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

On a simulated $16 \times 16$ hexagonal grid (256 cells) where a smooth spatial
risk surface is overlaid with a local 1-ring disturbance, two cells two steps
apart still overlap through a shared neighbour. Removing a 2-ring neighbourhood
blocks that spillover: each interior cell leaves the fit with its 18 nearest
neighbours (19 cells in total; corner and edge cells remove 7 to 18).

```{figure} ../images/ncv/spatial_ring_grid.png
:name: fig-ncv-spatial-ring-grid
:alt: Three hexagonal grid heat maps comparing the true spatial risk surface with an example 2-ring highlighted, a blotchy fit_reml surface, and a smoother fit_ncv surface using 2-ring neighbours.

Simulated $16 \times 16$ hexagonal grid (256 cells) with locally correlated spatial disturbances. Left: true spatial risk surface with one interior cell (red) and its 18 ring neighbours (amber) outlined. Centre: `fit_reml`. Right: `fit_ncv` with 2-ring neighbourhoods.
```

Because `fit_reml` assumes independent cells, it spends 52.2 effective degrees
of freedom fitting isolated pockets across the grid (RMSE 0.374 against the
true surface). Holding out each cell's 2-ring neighbourhood with `fit_ncv`
lowers the effective degrees of freedom to 30.8 and cuts RMSE by 19.3%, to
0.302. Even so, `fit_ncv`'s map still shows pockets the true surface does not
have: it cuts the error by 19.3%, rather than removing the disturbance.

## Inspecting held-out predictions by group

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
groups or splits automatically. After a single-row or `neighbours=` fit, pass
`by=` with one label per row of `X` (for example, region or postcode sector).
When a model is constructed with `retain_fit_state=False`, the per-group table
of a `groups=` or `cv=` fit is still kept, while row-level arrays are released
so `ncv_heldout()` and `ncv_heldout_groups(by=...)` are unavailable. Where a
group's held-out expected total is zero or negative, `actual_over_expected` is
reported as missing (`NaN`).

Below is a simulated Poisson claim-frequency book of 700 rows across 20 fleets
(35 rows per fleet, 1,760 claims in total), with exposure carried as a log
offset and `groups=fleets`:

```python
import numpy as np
from superglm import Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={"veh_age": Spline(kind="cr", k=14)},
).fit_ncv(X, claims, offset=np.log(exposure), groups=fleets)

by_fleet = model.ncv_heldout_groups()
```

```{figure} ../images/ncv/heldout_groups_ae.png
:name: fig-ncv-heldout-groups-ae
:alt: Scatter plot of actual claims against held-out expected claims for 20 simulated fleets, with a diagonal balance line, a plus or minus 25 percent reference band, and standout fleets labelled with their A/E ratios.

Actual claims against held-out expected claims for the 20 simulated fleets, read directly from `model.ncv_heldout_groups()`. Overall actual-to-expected is 0.998 (1,760 actual against 1,762.9 held-out expected), while five fleets stand more than 25% away from what the other 19 fleets predict for them.
```

Sorting `by_fleet` by `actual_over_expected` highlights the fleets whose
realised experience stands furthest from what the rest of the book predicts for
them (showing the three highest, two near balance, and three lowest fleets):

| `group` | `rows` | `weight` | `actual` | `heldout_expected` | `actual_over_expected` | `heldout_deviance` |
|---|---:|---:|---:|---:|---:|---:|
| `Fleet_20` | 35 | 35 | 160 | 86.7 | 1.85 | 92.3 |
| `Fleet_04` | 35 | 35 | 138 | 91.4 | 1.51 | 48.2 |
| `Fleet_12` | 35 | 35 | 90 | 72.8 | 1.24 | 33.1 |
| `Fleet_18` | 35 | 35 | 86 | 88.1 | 0.98 | 44.3 |
| `Fleet_15` | 35 | 35 | 73 | 75.6 | 0.97 | 36.0 |
| `Fleet_01` | 35 | 35 | 95 | 134.2 | 0.71 | 59.9 |
| `Fleet_16` | 35 | 35 | 66 | 100.0 | 0.66 | 37.9 |
| `Fleet_10` | 35 | 35 | 51 | 77.6 | 0.66 | 45.7 |

Because each fleet's expected claims come from the fit with that fleet removed,
a large fleet cannot pull the smooth curve toward its own claims history and
hide its own A/E gap.

## When to use it and when not

`fit_ncv` is not a drop-in replacement for `fit_reml`. It pays off when rows
belong to correlated groups or spatial neighbourhoods and there are enough
neighbourhoods to judge held-out error reliably. It loses when neighbourhoods
are few, and it adds extra fitting work for no practical gain when rows are
independent.

To make that trade-off concrete, the committed study script repeats the
simulated grouped curve above over 50 random seeds across three settings:

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
  noise, medians tie (0.0515 against 0.0520), mean RMSE is slightly higher for
  `fit_ncv` (0.0548 against 0.0542), occasional seeds under-smooth up to 1.76×,
  and it still does more fitting work (median 10 inner solves against 6 REML
  iterations; on the 30-group model, also a median of 300 leave-out
  factorisations).

### Practical limits to keep in mind

- **Fewer than 10 neighbourhoods**: `fit_ncv` still fits, including for a
  single train/test split, but emits a `UserWarning` that the held-out
  criterion is noisy with fewer than 10 neighbourhoods (a rule of thumb, not a
  measured threshold). With only a handful of folds, the chosen smoothing can
  move noticeably between splits. Use more groups or folds where you can, or
  stick with `fit_reml`.
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
- **Fitting cost**: `fit_ncv` typically takes several times as many outer
  evaluations and factorisations as `fit_reml`. When rows are independent, that
  extra work does not buy better curves.
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

Running the call above on the 30-group simulated Gaussian book from earlier
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
| `fit_reml()` | Marginal / restricted likelihood (REML / Laplace approximation) | Rows are conditionally independent given the covariates | Default for pricing models, large binned books (`discrete=True`), Gamma/Tweedie/NB/Binomial, monotone terms, `RandomEffect` and `FactorSmooth` |
| `fit_ncv()` | Held-out deviance over neighbourhoods you define | Observations outside a neighbourhood predict it without sharing its noise | Gaussian/identity or Poisson/log exact fits when rows share group, period or spatial disturbances and you have enough neighbourhoods (at least ~10) |

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
deviance with prior weights, not averaged and not scaled — and both use one
Newton step per neighbourhood for non-Gaussian families, choosing the same fit
on the same design and penalties. `fit_ncv`'s standard errors are the ordinary
ones at the chosen smoothing, whereas `mgcv`'s default on single-row and
row-with-neighbours structures adjusts them for the neighbourhood structure, so
the two differ there unless `mgcv` is told `nei$jackknife = FALSE`. When
comparing the two engines directly, compare without an offset.

## Reference

- Wood, S. N. (2024). On Neighbourhood Cross Validation.
  [arXiv:2404.16490](https://arxiv.org/abs/2404.16490).

## Next steps

- [Choose a fitting path](choose-a-fitting-path.md)
- [Validation and model comparison](compare-models-on-holdout.md)
- [How REML chooses smoothness](../explanation/how-reml-chooses-smoothness.md)
- {py:meth}`superglm.SuperGLM.fit_ncv`,
  {py:meth}`superglm.SuperGLM.ncv_diagnostics`,
  {py:meth}`superglm.SuperGLM.ncv_heldout`,
  {py:meth}`superglm.SuperGLM.ncv_heldout_groups`
