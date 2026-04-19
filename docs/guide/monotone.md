# Monotone Splines

If monotonicity is part of the model specification, prefer fitting it inside
the model rather than repairing it afterward.

## Constraint Settings

The public spline API now uses a single `constraint=` argument:

- `constraint=Constraint.fit.increasing`
- `constraint=Constraint.fit.decreasing`
- `constraint=Constraint.postfit.increasing`
- `constraint=Constraint.postfit.decreasing`

Use `Constraint.fit.*` when the shape constraint should live inside the solver.
Use `Constraint.postfit.*` when you want the fitted spline repaired after
estimation instead.

Reserved tokens already exist for future work:

- `Constraint.fit.convex`
- `Constraint.fit.concave`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

Those names are public, but they currently raise `NotImplementedError`.

## Engine Selection

`Constraint.fit.*` selects a different constrained engine depending on the
feature class:

| Feature spec | Engine | Best for |
|---|---|---|
| `BSplineSmooth(..., constraint=Constraint.fit.increasing)` | QP | actual B-spline smooth with monotone constraint |
| `CubicRegressionSpline(..., constraint=Constraint.fit.decreasing)` | QP | cubic regression spline with monotone constraint |
| `PSpline(..., constraint=Constraint.fit.increasing)` | SCOP | monotone P-spline path |

Specifically:

- `PSpline(..., constraint=Constraint.fit.increasing)` uses SCOP
- `BSplineSmooth(..., constraint=Constraint.fit.increasing)` uses QP
- `CubicRegressionSpline(..., constraint=Constraint.fit.decreasing)` uses QP

## QP-Backed Monotone Fits

Use QP-backed monotone fitting when the term is a constrained B-spline smooth
or cubic regression spline.

```python
from superglm import BSplineSmooth, Constraint, CubicRegressionSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x1": BSplineSmooth(
            n_knots=8,
            constraint=Constraint.fit.increasing,
        ),
        "x2": CubicRegressionSpline(
            n_knots=8,
            constraint=Constraint.fit.decreasing,
        ),
    },
)
model.fit(df, y)
```

This keeps the monotone constraint in the actual optimization problem rather
than applying an after-the-fact repair.

## SCOP-Backed Monotone Fits

Use `PSpline(..., constraint=Constraint.fit.increasing)` when you want the SCOP path.

```python
from superglm import Constraint, PSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": PSpline(
            n_knots=10,
            constraint=Constraint.fit.increasing,
        ),
    },
)
model.fit_reml(df, y)
```

This works with both standard and discrete fitting paths and is the preferred
shape-constrained story for P-splines.

## REML Semantics

Monotone solver-backed splines can be used with `fit_reml()`, but the REML
semantics are different for SCOP and QP:

| Path | `fit_reml()` with fixed lambdas | `fit_reml()` with automatic lambda estimation |
|---|---|---|
| SCOP (`PSpline(..., constraint=Constraint.fit.increasing)`) | supported | dedicated monotone-aware REML / EFS path |
| QP (`BSplineSmooth(..., constraint=Constraint.fit.increasing)`, `CubicRegressionSpline(..., constraint=Constraint.fit.decreasing)`) | supported | passthrough heuristic: unconstrained REML followed by constrained refit |

The important nuance is that "SCOP works with REML but QP does not" is too
strong. QP monotone terms do work with `fit_reml()`. The difference is that
automatic lambda estimation on the QP path is not exact joint constrained REML;
it estimates lambdas from an unconstrained REML pass and then refits with the
monotone constraints at those lambdas.

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
            constraint=Constraint.fit.increasing,
        ),
    },
)
model.fit_reml(df, y)
```

Fixed-lambda monotone REML works for both SCOP and QP paths.

## Current Guard Rails

These combinations are intentionally guarded:

- monotone fit-time constraints with `selection_penalty > 0`
- monotone fit-time constraints with `select=True`
- mixed SCOP and QP monotone engines in the same model
- `kind="ns"` monotone fitting

If you need one of these combinations, treat it as unsupported rather than
assuming it is a valid workflow.

## Post-Fit Repair

Post-fit isotonic repair still exists:

```python
model.apply_monotone_postfit(df)
```

Use it when you already have a fitted model and need a manual monotone repair.
Do not treat it as the preferred monotone modeling path when a solver-backed
fit is available.

## Practical Advice

- choose solver-backed monotonicity when the business constraint is part of the
  actual tariff design
- use QP for constrained B-spline smooths and cubic regression splines
- use SCOP for monotone P-splines
- keep `selection_penalty=0` for these workflows
- validate the fitted shape on a prediction grid before signing off
