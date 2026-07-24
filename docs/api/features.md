# Features

Feature specs define how raw columns become model terms. This page keeps the
public `Spline(...)` factory separate from the concrete feature classes so the
API reads in the same order users encounter it.

## Factory

`Spline(...)` is the public entry point for spline specs. Use `kind="ps"` for a
difference-penalized P-spline and `kind="bs"` for an integrated-derivative
B-spline smooth.

Use `constraint=` to request monotone or curvature-constrained fits:

- `Constraint.fit.increasing`
- `Constraint.fit.decreasing`
- `Constraint.fit.convex`
- `Constraint.fit.concave`
- `Constraint.postfit.increasing`
- `Constraint.postfit.decreasing`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

These constraints are interpreted on the spline term's contribution to the
linear predictor. For non-identity links, that means the constrained object is
the term on the `eta` scale, not the back-transformed response mean.

Fit-time engine selection depends on the concrete spline class:

| Feature spec | Fit-time constraints | Engine | `fit_reml()` automatic lambda behavior |
|---|---|---|---|
| `PSpline(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | SCOP | integrated constrained REML / EFS |
| `BSplineSmooth(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | QP | unconstrained REML followed by constrained refit at those lambdas |
| `CubicRegressionSpline(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | QP | unconstrained REML followed by constrained refit at those lambdas |

`Constraint.postfit.*` keeps the unconstrained fit and applies a repair later
with `model.apply_shape_postfit(...)`.

::: superglm.Spline

## Spline Classes

::: superglm.PSpline

::: superglm.BSplineSmooth

::: superglm.NaturalSpline

::: superglm.CubicRegressionSpline

## Other Feature Classes

::: superglm.Categorical

::: superglm.RandomEffect

::: superglm.Numeric

::: superglm.Polynomial

## Interaction Specs

::: superglm.FactorSmooth
