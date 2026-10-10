# Migration: P-splines on uneven knots take the general difference penalty

*Ships in 0.40.*

## What changed

A P-spline (`Spline(kind="ps")`) penalises differences between neighbouring
coefficients. That penalty measures wiggliness only when the knots are evenly
spaced. Evenly spaced knots are unchanged: those the `"uniform"` rule places,
and stated or quantile-placed knots that fall at the same even spacing, such as
`fitted_knots` passed back with `fitted_boundary`. Other knots stated with
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
sit. It is scaled to the size of the standard penalty for the same number of
coefficients, so a fixed `spline_penalty` smooths about as strongly as it did.

Sometimes the general penalty cannot be computed reliably. Its largest and
smallest stiffnesses then differ by more than a factor of `1/sqrt(eps)`, about
7e7. This happens on heavily skewed columns such as freMTPL2's `Density`, and
on knots crowded into a small part of the axis. With cubic splines and `m = 3`,
it also happens from about 60 knots, even when the knots are only a little
uneven. For those the penalty is the standard one with the polynomials
of degree below `m` taken out of it. A heavily smoothed term is still a
straight line.
The switch between the two is a step, not a gradual change: a knot more or
fewer, or a refit on new data, can move a term across the limit and change its
fit by more than the knots alone explain; `diagnostics()` and `knot_summary()`
name the penalty each term took as `difference_penalty`.

## Why

With uneven knots, the standard penalty no longer measures wiggliness. As the
smoothing parameter grows, the fit is pulled towards a shape set by where the
knots fall. The general penalty removes that
dependence on knot placement, so a heavily smoothed term is a straight line
whatever its knots.

## Who is affected

- **Uniform knots, or stated knots at the same even spacing**: nothing changes.
- **Stated knots or a quantile rule**: the fit can change. The smoothing
  parameter, the effective degrees of freedom and the curve can all move.
- **A penalty order `m` above the degree**: the standard penalty is kept, so
  nothing changes.
- **`bs` and `cr` splines**: the general penalty does not change them, because
  their penalties already handle uneven knots. On strongly skewed knots, see the
  next three bullets.
- **Explicit `cr` or `bs` on strongly skewed knots** (quantile knots on a
  long-tailed column, for example): under `fit_reml`, REML counts one more
  penalised direction, so the smoothing parameter, the effective degrees of
  freedom and the curve can all move.
- **Explicit `cr` or `bs` with `select=True` and `discrete=True` on strongly
  skewed knots**: under `fit_reml`, a fit that 0.39 refused now runs.
- **Explicit `cr` or `bs` with `select=True` on an extremely long tail**
  (quantile knots on, for example, a sum insured or a mileage): when double
  precision cannot hold the penalty on the widest knot interval beside the
  narrowest, a fit that 0.39 fitted now stops with an error that names
  `kind="ps"`. On lognormal data with 20 quantile knots this starts near a
  log-scale spread of 2.5; shorter tails fit.
- **`ns` splines**: nothing changes. They keep the standard penalty.

## What to do

- **Saved models**: a model fitted and saved under 0.39 predicts the same under 0.40, and its
  summary is unchanged. Only a refit uses the new penalty.

- **Uniform knots**: nothing.
- **Stated or quantile-placed knots**: refit under 0.40 and compare the
  validation deviance and the curve with the 0.39 fit on the same data. A change
  in the curve is expected. Re-check any threshold tuned on the old fit.
- **Editor knot and kind changes**: an editor session can record a spline's knots and
  its kind in a structure file. Upgrade to 0.40 before you apply such a file: a 0.39
  or older superglm refuses a structure file that has a `knots` or `basis` entry.

## Verification trail

- The general penalty is taken from Li and Cao (2022) as cited above.
- The behaviour is pinned in `tests/test_pspline_general_penalty.py`: a straight
  line is unpenalised on stated and quantile-placed knots, the most-smoothed fit
  is a line, and the uniform rule, evenly spaced knots and an order above the
  degree keep the standard penalty.
- The same file pins the knots too uneven for the general penalty. These are a
  crowded cluster of stated knots, knots 1e-160 apart, `m=3` on lognormal(0, 2)
  quantile knots, and a tensor interaction with a skewed quantile margin. Each
  fit completes. On the crowded and the 1e-160 knots the line stays
  unpenalised. On `quantile_rows` knots of lognormal(0, 1.5) data REML counts
  the penalty's full rank. `tests/test_realdata_parity.py` fits `select=True`
  on freMTPL2's `VehAge` and `Density` with `quantile_rows` knots.

## Performance

Measured on the full freMTPL2 frequency book (678,013 rows): a Poisson
`fit_reml` with a log-exposure offset, eight threads in every pool, runs in
A-B-B-A order. Columns are clipped as in `tests/test_realdata_parity.py`. The
backend dispatched was `gram` in every run.

**Like for like.** `BonusMalus` (`quantile`, 12 knots), `DrivAge`
(`quantile_tempered`, 12) and `VehPower` (`quantile_tempered`, 8) take the
general penalty both before and after the rescaling, so only its size differs.

| Build | Wall time (s) | Peak RSS (MiB) | Deviance | EDF | REML iterations |
|---|---|---|---|---|---|
| Unscaled general penalty | 4.07, 4.13 | 1223, 1331 | 217456.43271 | 31.14590 | 7 |
| Scaled, with the fallback | 4.57, 3.92 | 1275, 1273 | 217456.43274 | 31.14588 | 7 |

The smoothing parameters differ by the scale factor: `BonusMalus` 2.117 and
2.924, `DrivAge` 21.67 and 41.48, `VehPower` 2.071 and 6.423.

**With a strongly skewed column.** `VehAge` and `BonusMalus` (`quantile`,
12 knots), `Density` (`quantile`, 10) and `DrivAge` (uniform, 10). `Density`'s
general penalty has a condition of 1.7e9, so it now takes the fallback, and
`VehAge`'s quantile knots fall at the even spacing and take the standard
penalty.

| Build | Wall time (s) | Peak RSS (MiB) | Deviance | EDF | REML iterations | Density lambda |
|---|---|---|---|---|---|---|
| 0.39.0 (standard penalty) | 36.55, 32.68 | 841, 821 | 211895.68 | 37.68 | 8 | 13550 |
| Scaled, with the fallback | 39.30, 38.34 | 824, 809 | 211888.12 | 40.38 | 10 | 907.5 |
| Unscaled general penalty (earlier session) | 35.86, 32.62 | 835, 823 | 211899.95 | 40.37 | 8 | 6.088 |

The first two rows are one A-B-B-A session. Against 0.39 the fit is about
12% slower on this set: REML takes two more iterations to settle
`Density`'s penalty, and each iteration is about 10% cheaper. The deviance
is 7.6 lower, and peak memory is unchanged.
