# Features

Feature specs define how raw columns become model terms. This page keeps the
public `Spline(...)` factory separate from the concrete feature classes so the
API reads in the same order users encounter it.

## Factory

`Spline(...)` is the public entry point for spline specs. Use `kind="ps"` for a
difference-penalized P-spline and `kind="bs"` for an integrated-derivative
B-spline smooth. Use `constraint=Constraint.fit.increasing` or
`constraint=Constraint.postfit.decreasing` for the supported public shape
constraint API. Reserved tokens such as `Constraint.fit.convex` and
`Constraint.postfit.concave` currently raise `NotImplementedError`.

::: superglm.Spline

## Spline Classes

::: superglm.PSpline

::: superglm.BSplineSmooth

::: superglm.NaturalSpline

::: superglm.CubicRegressionSpline

## Other Feature Classes

::: superglm.Categorical

::: superglm.Numeric

::: superglm.Polynomial
