# Monotone Splines

If monotonicity is part of the model specification, prefer fitting it inside
the model rather than repairing it afterward.

## Monotone Settings

These are the accepted monotonicity arguments on the public spline specs:

- `monotone=None`: default, no monotonicity constraint
- `monotone="increasing"`: enforce a nondecreasing spline
- `monotone="decreasing"`: enforce a nonincreasing spline
- `monotone_mode="postfit"`: default, fit the spline first and optionally apply
  isotonic repair afterward
- `monotone_mode="fit"`: keep monotonicity inside the solver during `fit()` or
  `fit_reml()`

## Engine Selection

`monotone_mode="fit"` selects a different constrained engine depending on the
feature class:

| Feature spec | Engine | Best for |
|---|---|---|
| `BSplineSmooth(..., monotone_mode="fit")` | QP | actual B-spline smooth with monotone constraint |
| `CubicRegressionSpline(..., monotone_mode="fit")` | QP | cubic regression spline with monotone constraint |
| `PSpline(..., monotone_mode="fit")` | SCOP | monotone P-spline path |

Specifically:

- `PSpline(..., monotone_mode="fit")` uses SCOP
- `BSplineSmooth(..., monotone_mode="fit")` uses QP
- `CubicRegressionSpline(..., monotone_mode="fit")` uses QP

## QP-Backed Monotone Fits

Use QP-backed monotone fitting when the term is a constrained B-spline smooth
or cubic regression spline.

```python
from superglm import BSplineSmooth, CubicRegressionSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x1": BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
        ),
        "x2": CubicRegressionSpline(
            n_knots=8,
            monotone="decreasing",
            monotone_mode="fit",
        ),
    },
)
model.fit(df, y)
```

This keeps the monotone constraint in the actual optimization problem rather
than applying an after-the-fact repair.

## SCOP-Backed Monotone Fits

Use `PSpline(..., monotone_mode="fit")` when you want the SCOP path.

```python
from superglm import PSpline, SuperGLM

model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": PSpline(
            n_knots=10,
            monotone="increasing",
            monotone_mode="fit",
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
| SCOP (`PSpline(..., monotone_mode="fit")`) | supported | dedicated monotone-aware REML / EFS path |
| QP (`BSplineSmooth(..., monotone_mode="fit")`, `CubicRegressionSpline(..., monotone_mode="fit")`) | supported | passthrough heuristic: unconstrained REML followed by constrained refit |

The important nuance is that "SCOP works with REML but QP does not" is too
strong. QP monotone terms do work with `fit_reml()`. The difference is that
automatic lambda estimation on the QP path is not exact joint constrained REML;
it estimates lambdas from an unconstrained REML pass and then refits with the
monotone constraints at those lambdas.

For large data, you can also combine the SCOP path with `discrete=True`:

```python
model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    discrete=True,
    features={
        "x": PSpline(
            n_knots=10,
            monotone="increasing",
            monotone_mode="fit",
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
