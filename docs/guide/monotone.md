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
- `constraint=Constraint.postfit.convex`
- `constraint=Constraint.postfit.concave`

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
| `PSpline(...)` | increasing/decreasing at supported degrees; convex/concave only when `degree <= 3` | SCOP | exact and `discrete=True` paths, integrated constrained `fit_reml()` |
| `BSplineSmooth(...)` | increasing/decreasing at supported degrees; convex/concave only when `degree <= 3` | QP | constrained solve on the B-spline smooth basis |
| `CubicRegressionSpline(...)` | increasing/decreasing/convex/concave | QP | inherently cubic; knot-spacing-aware constraints on the natural cubic regression spline basis |

Specifically:

- `PSpline(..., constraint=Constraint.fit.*)` uses SCOP
- `BSplineSmooth(..., constraint=Constraint.fit.*)` uses QP
- `CubicRegressionSpline(..., constraint=Constraint.fit.*)` uses QP

Fit-time convexity and concavity use an exact coefficient-space curvature
characterization only for splines of degree three or lower. Consequently,
`PSpline` and `BSplineSmooth` with `degree > 3` still support fit-time
increasing/decreasing constraints, but not `Constraint.fit.convex` or
`Constraint.fit.concave`. `CubicRegressionSpline` is always degree three.
This restriction does not apply to the separate `Constraint.postfit.*` repair
workflow.

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
            degree=3,
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
            degree=3,
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
| SCOP (`PSpline(..., constraint=Constraint.fit.*)`) | supported within the degree limits above | integrated constrained REML / EFS path |
| QP (`BSplineSmooth(..., constraint=Constraint.fit.*)`, `CubicRegressionSpline(..., constraint=Constraint.fit.*)`) | supported within the degree limits above | passthrough heuristic: unconstrained REML followed by constrained refit |

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
- `Constraint.fit.convex` or `Constraint.fit.concave` on a `PSpline` or
  `BSplineSmooth` with `degree > 3`

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

### Inference after a repair is withheld, not reported

A repair replaces the published coefficients with a **projection onto the
shape cone**. That projection is a constrained estimator, and its reference
distribution is not the unconstrained fit's, so `summary()` withholds the
repaired term's chi-square test, its p-value, its `ref_df` and its curve SE
band and prints `repaired (inference withheld)` in their place. The point
estimate, `edf` and the fitted level or curve values are still reported.

The reason is that the effective dimension of a shape-restricted fit depends
on how many cone edges are active, which is a random variable, not a quantity
readable off the penalty (Meyer and Woodroofe, "On the degrees of freedom in
shape-restricted regression", *Annals of Statistics* 28(4):1083–1104, 2000),
and the null distribution of the corresponding test is a mixture rather than a
fixed-df chi-square (Meyer, "Inference using shape-restricted regression
splines", *Annals of Applied Statistics* 2(3):1013–1033, 2008, §3). A repair
that binds is by definition on the boundary of the constrained parameter
space, which is exactly where the constrained and unconstrained estimators
stop agreeing. The point estimate itself needs no such caveat: projecting onto
the cone is a weak improvement in every `L_p` norm regardless of how the
original estimate was produced (Chernozhukov, Fernández-Val and Galichon,
"Improving point and interval estimators of monotone functions by
rearrangement", *Biometrika* 96(3):559–575, 2009, Proposition 1).

If you need a test or an interval for a shape-constrained term, fit the
constraint — `Constraint.fit.*` — rather than repairing after the fact.

### Constraints on an `OrderedCategorical` bind on the whole level axis

An `OrderedCategorical` maps its `L` levels to `L` positions and fits a spline
through them. Nothing is ever predicted between those positions, but **both
shape engines constrain the continuous curve over the whole interval**, not
only the `L` fitted level values. This is deliberately conservative and is a
stated contract, not an accident — but it is stronger than what the ordinal
literature prescribes, and the difference is measurable: against a level-only
projection of the same fitted values, the interval constraint costs roughly
1–8% of weighted SSE at `L` between 6 and 12, in the direction theory
predicts.

The published methods for a monotone effect of an ordinal predictor constrain
the `L` level effects directly, and none constrains a curve between category
positions:

- Rufibach, "An active set algorithm to estimate parameters in generalized
  linear models with ordered predictors", *Computational Statistics and Data
  Analysis* 54(6):1442–1456, 2010 — inequalities on the level dummies; §5,
  Lemma 5.1 reduces the Gaussian one-factor case exactly to weighted isotonic
  regression on the level means, solved by PAVA.
- Barlow, Bartholomew, Bremner and Brunk, *Statistical Inference under Order
  Restrictions*, Wiley, 1972; Robertson, Wright and Dykstra, *Order Restricted
  Statistical Inference*, Wiley, 1988 — the isotonic-cone projection and PAVA.
- Bürkner and Charpentier, "Modelling monotonic effects of ordinal predictors
  in Bayesian regression models", *British Journal of Mathematical and
  Statistical Psychology* 73(3):420–451, 2020 — a simplex over the `L`
  categories, monotone at the levels by construction.
- Gertheiss, Scheipl, Lauer and Ehrhardt, "Statistical inference for ordinal
  predictors in generalized additive models with application to Bronchopulmonary
  Dysplasia", *BMC Research Notes* 15:112, 2022 — a level-dummy basis with a
  difference penalty, which cannot express an interval constraint at all.
- Helwig, "Regression with ordered predictors via ordinal smoothing splines",
  *Frontiers in Applied Mathematics and Statistics* 3:15, 2017 — the ordinal
  reproducing kernel is defined only on the `L` category values.

The reason the interval version is strictly stronger is standard: positive
differences of adjacent spline coefficients are "sufficient but not necessary
for monotonically increasing effects" (Hofner, Kneib and Hothorn, "A unified
framework of constrained regression", *Statistics and Computing* 26:1–14, 2016,
§3.3), and for cubic bases the gap is unavoidable — "a linear combination of
cubic I-splines might be nondecreasing while one or more of the coefficients is
negative", and the necessary and sufficient conditions for a cubic to be
monotone on an interval "can not be written as a set of linear inequality
constraints" (Meyer 2008, §2). Meyer's own prescription, §1, is to constrain at
the design points and interpolate monotonically afterwards only if a curve is
actually wanted. For an ordered factor no curve between levels is ever wanted.

So the current behaviour is safe — every level-only-feasible shape our engine
accepts is also level-wise feasible — but it can refuse fits the literature
would accept. A level-only projection is not offered as a separate engine
today.

## Practical Advice

- choose solver-backed monotone / curvature constraints when the business rule
  is part of the actual tariff design
- use QP for constrained B-spline smooths and cubic regression splines
- use SCOP for constrained P-splines, especially when you want integrated
  automatic lambda estimation in `fit_reml()`
- keep `selection_penalty=0` for these workflows
- validate the fitted shape on a prediction grid before signing off
- interpret the constraint on the linear predictor scale
