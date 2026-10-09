# Migration: a spline with no kind is a cubic regression spline

*Ships in 0.40.*

## What changed

`Spline(...)` and `s(...)` with no `kind` now give a cubic regression spline,
`kind="cr"`. Before 0.40 they gave a P-spline, `kind="ps"`.

```python
Spline(n_knots=10)               # cubic regression spline from 0.40
Spline(kind="ps", n_knots=10)    # the P-spline that 0.39 fitted
```

Four changes follow from it:

- **A model that names no kind refits as `cr`.** The curve, the smoothing
  parameter, the effective degrees of freedom and the deviance can all move.
  A `cr` term with `n_knots` interior knots has two fewer coefficients than a
  `ps` term with the same `n_knots`.
- **A fit-time shape constraint with no kind moves from SCOP to the QP.**
  `Spline(constraint=Constraint.fit.increasing)` is now a `cr` term, which the
  QP constrains. The QP chooses the smoothing parameter without the constraint
  and then refits with it. SCOP, which a `ps` term uses, chooses it with the
  constraint in place.
- **`cr` refuses a degree other than 3.** A cubic regression spline is always
  cubic. In 0.39, `Spline(kind="cr", degree=2)` fitted a cubic without saying
  so. It now raises an error that names `kind="ps"` and `kind="bs"`. So does
  `Spline(degree=2)` with no kind, because its kind is now `cr`.
- **An ordered term's default basis is `cr`.** `OrderedCategorical(...)` with
  no `basis` now uses `Spline(kind="cr", n_knots=5)`. In 0.39 it used
  `Spline(kind="ps", n_knots=5)`. The knot count is still limited to the
  number of levels minus one.

## Who is affected

- **Code that names a kind**: nothing changes.
- **`Spline(...)` or `s(...)` with no kind**: a refit can change.
- **`OrderedCategorical` with no `basis`**: a refit can change.
- **Interactions of splines with no kind**: a refit can change. A tensor
  interaction or a spline-by-factor interaction takes its margins from its
  parents' kind.
- **A spline with no kind on a column with one distinct value**: the fit now
  stops with an error naming the feature. A P-spline fitted it.
- **An ordered term with no `basis` whose levels are all grouped into one
  band**: the fit now stops with an error. A P-spline fitted it.
- **`FactorSmooth` curves**: nothing changes. They stay P-splines whatever the
  main effect's kind.
- **`splines=` auto-detection and `SuperGLMRegressor(spline_features=...)`**:
  nothing changes. They build P-splines.

## What to do

- **To keep the 0.39 fit**: add `kind="ps"` to every `Spline(...)` and `s(...)`
  that names no kind. Add `basis=Spline(kind="ps", n_knots=5)` to every
  `OrderedCategorical` that omits `basis`.
- **A degree other than 3**: pass `kind="ps"` or `kind="bs"` with it.
- **To move to `cr`**: refit under 0.40 and compare the validation deviance and
  the curves with the 0.39 fit on the same data.
- **Saved models**: a model fitted and saved under 0.39 keeps its P-splines.
  It predicts the same under 0.40, and refitting it keeps its P-splines.
- **Structure files**: a structure file records a spline's kind only when the
  kind was chosen in the editor. Applied to a saved 0.39 model, it gives the
  same terms as in 0.39. Applied to a model declared again in code with no
  kind, it gives `cr` terms. Name the kind in code to keep a P-spline.

## Measured

The freMTPL2 frequency book (678,013 rows), Poisson with exposure weights,
fitted with `fit_reml()`. The splines are `DrivAge` (`n_knots=8`), `VehAge`
(`n_knots=12`) and `BonusMalus` (`n_knots=12`, `knot_strategy="quantile_tempered"`),
all with no kind, beside `LogDensity`, `VehBrand` and `Region`. Each row is the
same code run on 0.39 and on 0.40, alternated 0.39, 0.40, 0.40, 0.39, with
every thread pool pinned to one thread. Times are for the fit alone; peak
memory is the whole process's, data loading included.

| Model | Version | Kind | Fit time (s) | Peak memory (MiB) | Deviance | Total EDF | Solver |
|---|---|---|---|---|---|---|---|
| No constraint | 0.39 | `ps` | 11.16, 11.11 | 778 | 213,390.6 | 65.14 | direct, Gram |
| No constraint | 0.40 | `cr` | 12.58, 13.23 | 772 | 215,411.8 | 61.31 | direct, Gram |
| `BonusMalus` increasing | 0.39 | `ps` | 85.04, 83.81 | 980 | 213,471.5 | 59.24 | SCOP |
| `BonusMalus` increasing | 0.40 | `cr` | 13.15, 12.71 | 1,020 | 215,491.1 | 61.31 | QP, Gram |

On this book both 0.40 fits also emit a `SeparationWarning`. It comes from one
inner fit during the smoothing-parameter search, and names `VehAge`. The final
fit converged.

## Verification trail

- `tests/test_spline_factory.py::TestDefaultKindIsCr` pins the default kind for
  `Spline()` and `s()` and the degree refusal for `cr` and `cr_cardinal`.
- `tests/test_ordered_categorical_api.py` pins the ordered default basis, its
  clamp warning, and a bit-identical fit against `Spline(kind="cr", n_knots=5)`.
