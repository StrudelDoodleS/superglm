# SuperGLM

Penalised GLMs for insurance pricing. SuperGLM supports standard penalised fits, exact REML, large-*n* discrete/fREML-style REML, spline double-penalty shrinkage, group penalties, interactions, and statsmodels-style summaries for Poisson, Gamma, NB2, and Tweedie models.

## Features

- **Group penalties** — group lasso, sparse group lasso, group elastic net, ridge
- **Spline bases** — B-splines, natural splines, cubic regression splines with SSP reparametrisation
- **REML smoothness selection** — exact and discrete/fREML paths
- **Double-penalty shrinkage** — mgcv-style `select=True` for automatic term selection
- **Interactions** — spline × categorical, polynomial × categorical, and more
- **Inference** — statsmodels-style summary tables, Wood (2013) smooth tests, pointwise and simultaneous confidence bands
- **Families** — Poisson, Gamma, Negative Binomial (NB2), Tweedie (with profile estimation)

## Example plots

Poisson frequency model on French MTPL2 (678k policies), REML smoothness selection.
95% pointwise confidence bands with exposure-weighted density strip and interior knot positions.

| Vehicle Age (`quantile_rows` knots) | Bonus-Malus (`quantile_tempered`, α=0.2) |
|:---:|:---:|
| ![VehAge](images/readme_vehage.png) | ![BonusMalus](images/readme_bonusmalus.png) |
