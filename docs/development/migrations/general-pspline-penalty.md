# Migration: P-splines on uneven knots take the general difference penalty

*Ships in 0.39.*

## What changed

A P-spline (`Spline(kind="ps")`) penalises differences between neighbouring
coefficients. That penalty measures wiggliness only when the knots are evenly
spaced. Knots placed by the `"uniform"` rule are unchanged. Knots stated with
`knots=[...]`, or placed by a quantile rule (`"quantile"`, `"quantile_rows"` or
`"quantile_tempered"`), are unevenly spaced, and for those the penalty now is
the general difference penalty:

```python
Spline(kind="ps", knots=[18, 25, 35, 50, 70])     # general penalty
Spline(kind="ps", n_knots=8, knot_strategy="uniform")  # standard penalty, unchanged
```

The general penalty of Li and Cao, "General P-splines for non-uniform
B-splines" (2022, arXiv:2201.06808), smooths towards a straight line, that is
towards polynomials of degree below the penalty order `m`, wherever the knots
sit. On evenly spaced knots it equals the standard penalty.

## Why

With uneven knots, the standard penalty no longer measures wiggliness. As the
smoothing parameter grows, the fit is pulled towards a shape set by where the
knots fall. The general penalty removes that
dependence on knot placement, so a heavily smoothed term is a straight line
whatever its knots.

## Who is affected

- **Uniform knots**: nothing changes.
- **Stated knots or a quantile rule**: the fit can change. The smoothing
  parameter, the effective degrees of freedom and the curve can all move.
- **A penalty order `m` above the degree**: the standard penalty is kept, so
  nothing changes.
- **`bs` and `cr` splines**: nothing changes. Their penalties already handle
  uneven knots.
- **`ns` splines**: nothing changes. They keep the standard penalty.

## What to do

- **Saved models**: a model fitted and saved under 0.38 predicts the same under 0.39, and its
  summary is unchanged. Only a refit uses the new penalty.

- **Uniform knots**: nothing.
- **Stated or quantile-placed knots**: refit under 0.39 and compare the
  validation deviance and the curve with the 0.38 fit on the same data. A change
  in the curve is expected. Re-check any threshold tuned on the old fit.
- **Editor knot changes**: an editor session can record its knots in a structure
  file. Upgrade to 0.39 before you apply such a file: a 0.38 or older superglm
  refuses a structure file that has a `knots` entry.

## Verification trail

- The general penalty is taken from Li and Cao (2022) as cited above.
- The behaviour is pinned in `tests/test_pspline_general_penalty.py`: a straight
  line is unpenalised on stated and quantile-placed knots, the most-smoothed fit
  is a line, and the uniform rule, evenly spaced knots and an order above the
  degree keep the standard penalty.
