# Monotone And Curvature Constraints

If the business rule is monotone, convex, or concave, prefer fitting it inside
the model rather than repairing the spline afterward.

## Constraint Settings

The public spline API now uses a single `constraint=` argument:

- `constraint=Constraint.fit.increasing`
- `constraint=Constraint.fit.decreasing`
- `constraint=Constraint.fit.convex`
- `constraint=Constraint.fit.concave`
- `constraint=Constraint.postfit.increasing`
- `constraint=Constraint.postfit.decreasing`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

Use `Constraint.fit.*` when the shape constraint should live inside the solver.
Use `Constraint.postfit.*` when you want the fitted spline repaired after
estimation instead.

## Linear Predictor Semantics

Solver-backed shape constraints are enforced on the spline term's contribution
to the linear predictor `eta = X beta`.

That matters when the model uses a non-identity link:

- monotone direction is specified on the linear predictor scale
- convex/concave are also specified on the linear predictor scale
- do not assume response-scale curvature matches unless the inverse link
  preserves that shape

For common log-link pricing models, monotone direction carries through to the
mean because the inverse link is increasing, but curvature is still best
interpreted on the linear predictor / relativity scale.

## Engine Selection

`Constraint.fit.*` selects a different constrained engine depending on the
feature class:

| Feature spec | Fit-time kinds | Engine | Notes |
|---|---|---|---|
| `PSpline(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | SCOP | exact and `discrete=True` paths, integrated constrained `fit_reml()` |
| `BSplineSmooth(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | QP | constrained solve on the B-spline smooth basis |
| `CubicRegressionSpline(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)` | monotone + curvature | QP | constrained solve on the cubic regression spline basis |

Specifically:

- `PSpline(..., constraint=Constraint.fit.*)` uses SCOP
- `BSplineSmooth(..., constraint=Constraint.fit.*)` uses QP
- `CubicRegressionSpline(..., constraint=Constraint.fit.*)` uses QP

## QP-Backed Shape Fits

Use QP-backed fitting when the constrained term is a `BSplineSmooth` or
`CubicRegressionSpline`.

```python
from superglm import BSplineSmooth, Constraint, CubicRegressionSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x1": BSplineSmooth(
            n_knots=8,
            constraint=Constraint.fit.convex,
        ),
        "x2": CubicRegressionSpline(
            n_knots=8,
            constraint=Constraint.fit.concave,
        ),
    },
)
model.fit(df, y)
```

This keeps the monotone / curvature constraint in the actual optimization
problem rather than applying an after-the-fact repair.

## SCOP-Backed Shape Fits

Use `PSpline(..., constraint=Constraint.fit.*)` when you want the SCOP path.

```python
from superglm import Constraint, PSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": PSpline(
            n_knots=10,
            constraint=Constraint.fit.convex,
        ),
    },
)
model.fit_reml(df, y)
```

This works with both exact and `discrete=True` fitting paths and is the
preferred solver-backed shape story for P-splines.

## REML Semantics

Solver-backed shape splines can be used with `fit_reml()`, but the REML
semantics are different for SCOP and QP:

| Path | `fit_reml()` with fixed lambdas | `fit_reml()` with automatic lambda estimation |
|---|---|---|
| SCOP (`PSpline(..., constraint=Constraint.fit.increasing/decreasing/convex/concave)`) | supported | integrated constrained REML / EFS path |
| QP (`BSplineSmooth(..., constraint=Constraint.fit.*)`, `CubicRegressionSpline(..., constraint=Constraint.fit.*)`) | supported | passthrough heuristic: unconstrained REML followed by constrained refit |

The important nuance is that "SCOP works with REML but QP does not" is too
strong. QP-constrained terms do work with `fit_reml()`. The difference is that
automatic lambda estimation on the QP path is not exact joint constrained REML;
it estimates lambdas from an unconstrained REML pass and then refits with the
shape constraints at those lambdas.

For large data, you can also combine the SCOP path with `discrete=True`:

```python
from superglm import Constraint, PSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    discrete=True,
    features={
        "x": PSpline(
            n_knots=10,
            constraint=Constraint.fit.concave,
        ),
    },
)
model.fit_reml(df, y)
```

Fixed-lambda shape-constrained REML works for both SCOP and QP paths.

## Current Guard Rails

These combinations are intentionally guarded:

- fit-time shape constraints with `selection_penalty > 0`
- fit-time shape constraints with `select=True`
- mixed SCOP and QP constrained engines in the same model
- `kind="ns"` fit-time shape constraints

If you need one of these combinations, treat it as unsupported rather than
assuming it is a valid workflow.

## Post-Fit Repair

Post-fit repair still exists for all `Constraint.postfit.*` tokens:

```python
model.apply_shape_postfit(df)
```

Use it when you already have a fitted model and need a manual monotone,
convex, or concave repair. Do not treat it as the preferred modeling path when
a solver-backed
fit is available.

## Practical Advice

- choose solver-backed monotone / curvature constraints when the business rule
  is part of the actual tariff design
- use QP for constrained B-spline smooths and cubic regression splines
- use SCOP for constrained P-splines, especially when you want integrated
  automatic lambda estimation in `fit_reml()`
- keep `selection_penalty=0` for these workflows
- validate the fitted shape on a prediction grid before signing off
- interpret the constraint on the linear predictor scale
