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

`FactorSmooth(variable, group=..., basis=..., kind=...)` separates two choices:

- `basis="fs"` builds fully penalized random curves with shared wiggle and
  null-space REML components. A global spline is optional.
- `basis="sz"` builds pointwise sum-to-zero deviation curves with one shared
  wiggle component. A matching global `Spline` is required.
- `kind="ps"` selects the marginal P-spline basis for either construction.

The grouping column must not also be configured as `Categorical` or
`RandomEffect`; that would duplicate lower-order group geometry.

::: superglm.FactorSmooth
