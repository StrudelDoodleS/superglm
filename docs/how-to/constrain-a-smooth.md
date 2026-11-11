# Monotone And Curvature Constraints

If the business rule is monotone, convex, or concave, prefer fitting it inside
the model rather than repairing the spline afterward.

## Constraint Settings

The public spline API uses a single `constraint=` argument. It takes one rule
or several rules joined with `&`:

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

`Constraint.fit` also offers rules that `Constraint.postfit` does not:

- `Constraint.fit.flat`, a flat stretch of the curve
- `Constraint.fit.cap(lo, hi)`, a floor and a ceiling on the printed relativity
- `Constraint.fit.max_step(up, down)`, a limit on the step between adjacent
  bands of an ordered categorical

Join rules with `&`, and limit a shape rule to part of the feature with
`.below(x)`, `.above(x)` or `.within(a, b)`. The next section shows each.

## Combining Rules, Stretches And Caps

A term can carry several fit-time rules at once. The rules sit inside the fit,
so the other terms and the intercept adjust to them. Hand edits made after the
fit leave the book off balance.

The examples on this page share one small data set. Run this block once. The
later blocks reuse `df` and `y`.

```python
import numpy as np
import pandas as pd

from superglm import Categorical, Constraint, OrderedCategorical, Spline, SuperGLM

C = Constraint.fit

rng = np.random.default_rng(7)
n = 3_000
bands = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-85", "86+"]
band_effect = dict(zip(bands, np.linspace(-0.9, 0.9, len(bands))))
region_effect = {"North": 0.0, "South": 0.25, "East": -0.3, "West": 0.6}

df = pd.DataFrame(
    {
        "age": rng.integers(10, 86, n).astype(float),
        "power": rng.uniform(0.0, 10.0, n),
        "band": rng.choice(bands, n),
        "region": rng.choice(list(region_effect), n),
    }
)
age_effect = np.where(
    df["age"] < 25, 0.08 * (25 - df["age"]), np.where(df["age"] > 40, 0.02 * (df["age"] - 40), 0.0)
)
eta = (
    -1.0
    + age_effect
    + 1.5 * (1.0 - np.exp(-df["power"] / 2.0))
    + df["band"].map(band_effect)
    + df["region"].map(region_effect)
)
y = rng.poisson(np.exp(eta.to_numpy())).astype(float)
```

### Two shapes on one curve

Any direction can be combined with any curvature:

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={"power": Spline(kind="ps", k=10, constraint=C.increasing & C.concave)},
)
model.fit(df, y)
```

This curve rises and never speeds up. The curvature limits on
[Engine Selection](#engine-selection) still apply to the `concave` half.

### A rule on a stretch

Limit a rule to a stretch of the feature with `.below(x)`, `.above(x)` or
`.within(a, b)`. This rule makes the price fall to age 25, stay level from 25
to 40, and rise after 40:

```python
rules = C.decreasing.below(25) & C.flat.within(25, 40) & C.increasing.above(40)

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={"age": Spline(kind="ps", n_knots=9, constraint=rules)},
)
model.fit(df, y)
```

A rule with no stretch covers the whole range. Outside every stretch the curve
is free.

- Two direction rules, two curvature rules or two flat rules may not overlap.
  The fit is refused and names the overlap.
- A stretch rule works on `ps`, `bs` and `cr` splines, and on an
  `OrderedCategorical` with a spline basis.
- Stretch ends are values of the feature. On an `OrderedCategorical` they are
  band labels, as described under
  [Ordered categoricals](#ordered-categoricals).

### Flat stretches

`C.flat` holds the curve level on a stretch.

- `flat` needs a stretch, such as `C.flat.within(25, 40)`. A flat rule over
  the whole range is refused, and so is a stretch that widens to every knot
  span: the term would have no effect, so remove it.
- `flat` needs a spline of degree 1 or more. On a degree-0 spline, put knots at
  the stretch edges or band the feature with `OrderedCategorical`.
- A flat stretch covers every whole knot span it touches. When a stretch
  edge is not a knot, the fit warns and names the interval it actually
  flattens.
- On a `cr` spline, the natural end condition can also flatten the end span
  beside a flat stretch. The warning names that case too.
- Two flat stretches fewer than `degree` knot spans apart (three for a cubic)
  also flatten the gap between them, since the curve has too few knots there
  to bend. The warning names the whole interval; leave more knot spans between
  the stretches to keep the gap free.

For exact edges, use one of these:

- `PolynomialRange(degree=0)` on a `bs` or `cr` spline. This pins the curve
  flat on the range and changes the basis. See
  [Polynomial ranges](specify-features.md#polynomial-ranges-polynomial_ranges).
  A term with polynomial ranges cannot also carry a shape rule.
- Knots at the stretch edges, set with `knots=[...]` or by choosing
  `n_knots`. This keeps the shape rules. In the example above the ages run
  from 10 to 85. `n_knots=9` puts evenly spaced knots 7.5 apart, so two of
  them land on 25 and 40 and the fit does not warn.

### Caps

`C.cap(lo, hi)` keeps the printed relativity between `lo` and `hi`. Joined
with a shape rule, it caps the shaped curve:

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={
        "power": Spline(kind="ps", k=10, constraint=C.increasing & C.concave & C.cap(0.6, 1.8)),
    },
)
model.fit(df, y)

curve = model.term_inference("power", with_se=False, n_points=200).relativity
curve.min(), curve.max()  # inside 0.6 and 1.8, to solver tolerance
```

- A cap is measured on the printed curve, not on the raw coefficients.
- For a spline, the printed curve is the native one. Its log relativity
  averages to 0 over the training rows, so its geometric mean is 1.
- For a categorical or an ordered categorical, the cap is measured against the
  base level, whose relativity is 1.
- A cap needs a log link and is refused under any other link, because only
  then is the printed relativity a factor.
- A cap must contain 1: `0 < lo <= 1 <= hi` and `lo < hi`.
- A term takes at most one cap.
- On a spline whose rules make it monotone over the whole range, the cap is
  exact at the ends.
- On a curve that is not monotone, the cap holds everywhere, but the curve may
  stop slightly short of it.
- A cap holds on the training range. With `extrapolation="extend"` the fit
  warns that the tails beyond it are not capped. Use `extrapolation="clip"` to
  keep predictions inside the cap.

### Ordered categoricals

Put the rules on the spline basis of an `OrderedCategorical`. Add a cap to
limit the relativity of every band against the base band, and `max_step` to
limit the step between neighbours:

```python
steps = C.increasing & C.cap(0.8, 1.25) & C.max_step(0.15)

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={"band": OrderedCategorical(order=bands, basis=Spline(k=6, constraint=steps))},
)
model.fit(df, y)

ti = model.term_inference("band", with_se=False)
pd.Series(ti.relativity, index=ti.levels)
```

`C.max_step(up, down)` keeps every ratio between adjacent bands inside
`[1 - down, 1 + up]`. `C.max_step(0.15)` allows 15% up and 15% down. A
separate limit down, as in `C.max_step(0.15, 0.10)`, allows 15% up and 10%
down. Both are fractions, `up > 0` and `0 < down < 1`.

- `down` defaults to `up`. When `up` is 1 or more, give `down` explicitly:
  `C.max_step(1.0)` is refused, while `C.max_step(1.0, 0.5)` fits.
- A term takes at most one `max_step`.
- `max_step` applies only to ordered categoricals, where it limits the step
  between the levels.
- On a spline term `max_step` is refused. To limit steps on a numeric
  feature, band it with `OrderedCategorical` first.
- `max_step` needs a log link, like a cap.
- Shape rules and `flat` bind on the whole level axis, as stated under
  [Constraints on an `OrderedCategorical`](#constraints-on-an-orderedcategorical-bind-on-the-whole-level-axis).
- Caps and `max_step` bind at the levels.
- Stretch ends are level labels, inclusive, in level order. When the term uses
  a grouping, a group label names its group. For example,
  `C.increasing.above("36-45")` applies from the 36-45 band to the last.
- A cap or `max_step` together with special levels is refused. A special
  level is a free effect off the curve, so no level row can hold it.

### Caps on plain categoricals

A `Categorical` takes a cap and nothing else. Every level is held between
`lo` and `hi` against the base level:

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={"region": Categorical(constraint=C.cap(0.8, 1.25))},
)
model.fit(df, y)

ti = model.term_inference("region", with_se=False)
pd.Series(ti.relativity, index=ti.levels)
```

`cat("region", constraint=C.cap(0.8, 1.25))` is the same declaration in the
`terms=[...]` form:

```python
from superglm import cat

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    terms=[cat("region", constraint=C.cap(0.8, 1.25))],
)
model.fit(df, y)
```

- A shape rule on a plain categorical is refused. Use an `OrderedCategorical`
  for a shape.
- A capped categorical needs `selection_penalty=0`, as every fit-time shape,
  cap and step limit does. A flat stretch alone is part of the basis and fits
  under a selection penalty.

### What the rules do not do

- Rules bind the main-effect curve only. In an interaction, the main effect
  times the interaction effect can go past a cap.
- Smoothing is chosen without the rules, then the model is refit with them.
  See [REML Semantics](#reml-semantics).
- A rule does not narrow the intervals. A term whose rule binds keeps the
  standard errors of the fit without the rule. See
  [QP-Backed Shape Fits](#qp-backed-shape-fits).
- `Constraint.postfit` takes only a single shape over the whole range.
  Combinations, stretches and caps are fit-time only.
- `Constraint.postfit` is not applied by `SuperLSS`, which has no post-fit
  repair: it warns and fits the term unconstrained. `Constraint.fit` rules are
  enforced there; see [Rules On A Distributional Model](#rules-on-a-distributional-model).
- `fit_path()` does not enforce shapes, caps or step limits either. It warns
  once, naming every term whose rules it ignores, and fits those terms
  unconstrained; refit with `fit()` or `fit_reml()` to enforce the rules. The
  summary and `term_inference()` mark those rule lines `not enforced: fit_path`.
  A flat stretch is part of the term's basis, so it still holds on the path and
  is not named.

## Rules On A Distributional Model

`SuperLSS` enforces `Constraint.fit` rules on a term of any parameter, so a
monotone rating factor can sit in the mean of a Tweedie, negative binomial or
gamma model:

```python
from superglm import Constraint, TweedieLSS, SuperLSS, s

family = TweedieLSS()
model = SuperLSS(
    family,
    family.mu(s("BonusMalus", kind="cr", k=8, constraint=Constraint.fit.increasing)),
    family.phi(),
    family.p(),
)
model.fit_reml(X, y)
```

The fit works in two steps, as `SuperGLM` does:

1. The smoothing search runs without the rules.
2. The coefficients are refitted at that smoothing with the rules, together
   across every parameter, from a start that meets them. If the fit without
   the rules already obeys every rule, nothing is refitted and the fit is that
   fit.

What you can rely on:

- The curve meets every rule at every point, not only at the rows of the data.
- Standard errors, bands and posterior draws use the covariance of the fit
  without the rules, at the same smoothing, around the ruled curve. Where a
  rule binds is itself estimated from the data, so a covariance that treats the
  binding checkpoints as known gives intervals that are too narrow. See
  [What the intervals of a ruled term mean](#what-the-intervals-of-a-ruled-term-mean).
- Effective degrees of freedom treat the rules that bind as fixed: a term held
  flat counts as no free parameters.
- `covariance="corrected"` needs a fit free of its rules, so on a fit with a
  binding rule it returns the fixed covariance and says so.
- `diagnose()` says what each rule did, one sentence per constrained term: how
  many checkpoints along the curve bind, and the largest change from the fit
  without the rule.
- A refit that does not converge is returned and disclosed, never refused.
- A rule the term cannot state as rows, or one that cannot be applied as
  written, warns with `ShapeConstraintIgnoredWarning` and the term is fitted
  unconstrained.

What it does not do yet: choose the smoothing under the rules. The smoothing
is the unconstrained one, so a rule that binds hard leaves a curve a little
smoother or rougher than the best one under that rule. The editor does not
show which rules bind.

### Editing a constrained SuperLSS term

The model editor names each term's rules in the term bar, such as
**rule increasing**, on every parameter. Every edit of a SuperLSS term with
`Constraint.fit` rules keeps to them: after a drag, a handle move, a menu
action or a reset of part of the curve, the whole curve moves to the nearest
curve the rules allow. Nearest is measured in the exposure-weighted least
squares the editor writes every edit with, among the curves the term's basis
draws with coefficients that meet the fit's own rows. That curve is the one
the exported model takes, and Undo puts back the curve from before the edit.
An edit the rules already allow, such as raising the whole curve, is kept as
drawn. An edit the rules take back entirely, such as raising a level that
already sits at its cap, records nothing and says why. On a term with levels,
the base stays where it is unless the edit moves it. On a categorical term the
other levels an edit does not touch stay too, so a level pushed past its cap
stops at the cap; an edit of the base moves only the levels whose cap it
pushes them past, as little as the rule needs. An ordered term draws a smooth
curve through its levels, which cannot move one level alone, so there the
levels an edit does not touch follow the nearest curve, as they do on an
ordered term without a rule. A theta term draws the overdispersion 1/θ, so
its rules are named on 1/θ: `decreasing` on θ reads **rule increasing**, and
`cap(0.7, 1.5)` reads **rule cap(0.667, 1.43)**. A rule holds on the log
scale, where log(1/θ) is −log θ, so a curvature swaps as a direction does:
`concave` on θ reads **rule convex**. A term with a rule
takes no shaped range.

A `SuperGLM` term names its rules the same way, but its hand edits are not
held to them, so check the curve after an edit.

### Post-fit rules on a distributional model

A `Constraint.postfit` rule means the same on `SuperLSS` as on `SuperGLM`
(see [Post-Fit Repair](#post-fit-repair)), for a term of any parameter. The
term is fitted without the rule, and `apply_shape_postfit()` repairs it:

```python
from superglm import Constraint, GammaLS, SuperLSS, s

family = GammaLS()
model = SuperLSS(
    family,
    family.mean(s("DrivAge", kind="bs", k=8, constraint=Constraint.postfit.increasing)),
    family.scale(),
)
model.fit_reml(X, y)
model.apply_shape_postfit()
```

The repair is SuperGLM's: the fitted curve is projected onto the shape in the
term's own basis, on its parameter's link scale, weighted by where the
training rows sit along the axis. The basis keeps the fit's centring, so the
term's mean over the training rows does not move. The parameter's intercept
is then re-estimated on the joint likelihood, with every term and every other
parameter held. Two repairs are refused, and nothing changes: one whose
penalized objective (the negative log-likelihood plus the fit's smoothing
penalty, at the fit's smoothing parameters) is worse than with the term
dropped, and one that leaves a training row without valid parameters. A
repaired curve can fit the data better than no term and still lose on the
penalty, when it needs large coefficients; then fit the rule instead
(`Constraint.fit`), which chooses the smoothing under the rule.

What you get afterwards:

- Predictions, term curves, `coef_` and rating tables follow the repaired
  curve. `result_` and the covariance still describe the fit.
- A repaired term keeps its effective degrees of freedom. Its standard
  errors, bands and test are withheld, as on `SuperGLM`, and say so with
  `ShapeRepairedWarning`. `summary()` shows NaN for them and a note, and so
  does the intercept row of the term's parameter, which the repair
  re-estimated. In the editor the term has no band. An editor session open
  on the model sees the repair: its cross-validation and final-fit results
  are marked as made on an earlier version.
- Standard errors that read the whole covariance (`predict_link(se_fit=True)`,
  posterior draws and bounds) warn that the covariance is the fit's from before
  the repair.
- `diagnose()` reports each repair, its size and the intercept's shift. It also
  names each post-fit rule that is still to apply, or that the fitted curve
  already obeys, and warns when a repaired curve has been edited so that it
  breaks its rule again. The rating table payload carries the same notes.
- A term that already obeys its rule is left as it is, so a second call changes
  nothing. A term edited after its repair so that it breaks the rule again is
  repaired again, and the new repair replaces the old one. The repair needs the
  fit's rows (`retain_rows=True`, the default); a restored model takes the
  frame it was fitted on as `apply_shape_postfit(X)`.

Fit-time and post-fit rules combine on `SuperLSS` as they do on `SuperGLM`.
On one term they cannot: `Constraint.postfit` refuses a combination. On
different terms of one model, the fit-time rules are enforced inside the fit
and the post-fit rules are repaired afterwards.

The two models differ in what they recompute after a repair. `SuperGLM`
recomputes its working weights, effective degrees of freedom, covariance and
scale at the repaired coefficients, so the other terms' standard errors move.
`SuperLSS` keeps the fit's covariance and degrees of freedom, as it does after
a hand edit: the other terms' standard errors are those of the fit before the
repair, and `ShapeRepairedWarning` says so wherever they are quoted.

Monotone repairs can fail to converge on some curves, on both models. When a
term's derivative has its lowest point inside a knot span, the repair adds
checkpoints until it runs out. It then raises `RuntimeError`, and the model is
unchanged. Prefer `Constraint.fit` where you need the rule to hold.

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

`Constraint.fit.*` fits on the constrained QP for every spline kind:

| Feature spec | Fit-time kinds | Engine | Notes |
|---|---|---|---|
| `PSpline(...)` | increasing/decreasing at supported degrees; convex/concave only when `degree <= 3` | QP | the same rows as `BSplineSmooth`, with the P-spline difference penalty |
| `BSplineSmooth(...)` | increasing/decreasing at supported degrees; convex/concave only when `degree <= 3` | QP | constrained solve on the B-spline smooth basis |
| `CubicRegressionSpline(...)` | increasing/decreasing/convex/concave | QP | inherently cubic; knot-spacing-aware constraints on the natural cubic regression spline basis |

Specifically:

- `PSpline(..., constraint=Constraint.fit.*)` uses QP
- `BSplineSmooth(..., constraint=Constraint.fit.*)` uses QP
- `CubicRegressionSpline(..., constraint=Constraint.fit.*)` uses QP
- `Spline(..., constraint=Constraint.fit.*)` with no `kind` is a cubic
  regression spline, so it uses QP.

On the QP a monotone P-spline holds the first differences of its B-spline
coefficients to one sign. By de Boor's derivative formula the curve's slope is
a non-negative combination of those differences, so the curve is monotone
everywhere on the knot range, not only at the data.

Fit-time convexity and concavity use an exact coefficient-space curvature
characterization only for splines of degree three or lower. Consequently,
`PSpline` and `BSplineSmooth` with `degree > 3` still support fit-time
increasing/decreasing constraints, but not `Constraint.fit.convex` or
`Constraint.fit.concave`. `CubicRegressionSpline` is always degree three.
This restriction does not apply to the separate `Constraint.postfit.*` repair
workflow.

### Models fitted on SCOP

Releases up to 0.41.0 fitted a shape-constrained `PSpline` on SCOP, the shape
constrained P-spline reparameterization of Pya and Wood (2015). That engine is
retired:

- A `PSpline`, or a `Spline(kind="ps")`, declared with a `Constraint.fit.*`
  and no `shape_engine` (the code 0.41.0 ran) now fits on the QP with no
  warning. Its curve and smoothing parameter differ from those 0.41.0 gave.
- `shape_engine="scop"` is still accepted. It warns that SCOP is retired and
  fits the constraint on the QP, so the fitted curve and its smoothing
  parameter differ from those 0.41.0 gave. Drop the argument to silence the
  warning.
- A model saved with a SCOP term still loads, predicts, summarizes and exports
  as it was saved.
- Refitting such a model fits the term on the QP and warns once; a model
  refitted and saved again no longer warns. A spline declared with no kind
  beside a SCOP term (0.41.0 fitted it as `ps` on SCOP) refits as the `cr` it
  declares, with the same warning.

## QP-Backed Shape Fits

QP-backed fitting is what every constrained spline gets: `PSpline`,
`BSplineSmooth` and `CubicRegressionSpline`.

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

A QP-backed fit counts its parameters on its *active face*: the coefficient
directions left free by the constraint rows that bind at the fit. Effective
degrees of freedom, leverage, the Pearson scale and AIC/BIC come from the fit
on that face (Meyer and Woodroofe 2000; Tibshirani and Taylor, "Degrees of
freedom in lasso problems", *Annals of Statistics* 40(2), 2012, Theorem 3).

- A term the constraint holds flat counts as no free parameters.
- Under `fit_reml()`, the published scale is still the REML estimate, which
  does not read the face.
- The editor's **Free levels** comparison moves the curve along the face when
  it asks how a level's data pull on it, so a level the rule holds flat does
  not move with its data.

The coefficient covariance does not come from the face.

### What the intervals of a ruled term mean

Standard errors, pointwise intervals, simultaneous bands and posterior draws
of a fit whose rule binds use the covariance the fit has with no constraint
row held, at the same smoothing parameters. They are drawn around the ruled
curve. `SuperGLM` and `SuperLSS` do the same.

- **Why not the face.** Which rows bind is estimated from the data. A
  covariance that treats them as known is conditional on that estimate, and
  its intervals cover the true curve too rarely (Wood, "Shape constrained
  additive smooth models", *Statistical Modelling* 26(4), 2026, section 5.1).
  In simulation, where a rule that held for the true curve bound in the fit,
  nominal 95% pointwise intervals from the face covered 92% of the curve, and
  39% to 75% of it at the points the rule held when the true curve was
  strictly increasing. Intervals from the covariance with no row held covered
  96% and 71% to 96%.
- **What you get.** Intervals that cover at least as often as those of the
  fit without the rule. Where a rule binds they are wider than they need to
  be: the rule moves the curve but is not credited with any precision.
- **A term held flat, or held on a cap,** keeps the interval width of the fit
  without the rule. Its interval is not zero-width.
- **The limits need not obey the rule.** The lower limit of an increasing
  curve can dip. The fitted curve always obeys the rule.
- **A rule the data contradict.** The curve is the best one that obeys the
  rule, and the interval is centred on it, so it can miss a true curve that
  breaks the rule.
- **Term tests are conservative.** The chi-square test of a ruled term in
  `summary()` (`SuperGLM`) and in `term_test()` and `summary()` (`SuperLSS`)
  reads this covariance with the reference degrees of freedom of the face, so
  where a rule binds it rejects a term with no effect less often than its
  level says. In simulation with a flat true curve, among fits where the rule
  bound, the 5% test rejected 1.6% of `SuperGLM` Poisson fits and 0.9% of
  `SuperLSS` Gamma fits, against 2.6% and 2.4% with the face's covariance.

Releases up to 0.43.0 reported the covariance on the active face, with
zero-width intervals along a term held flat. A model saved by one of them
keeps that covariance in every standard error, interval, band, prediction
error and rating table it reports, and a `SuperLSS` model's `diagnose()`
says so; refit it to get the intervals described here.

## REML Semantics

Shape-constrained splines work with `fit_reml()`:

| `fit_reml()` with fixed lambdas | `fit_reml()` with automatic lambda estimation |
|---|---|
| supported within the degree limits above | unconstrained REML followed by a constrained refit |

Automatic lambda estimation is not exact joint constrained REML: it estimates
the lambdas from an unconstrained REML pass and then refits with the shape
constraints at those lambdas. Combined rules, stretches, caps and step limits
take the same route. REML chooses the smoothing without the shapes, caps and step
limits, then the model is refit with them at those smoothing parameters. A flat
stretch is part of the term's basis, so REML already holds it.

### With a random effect

A shape constraint and a `RandomEffect` can share a model:

```python
from superglm import Constraint, PSpline, RandomEffect, SuperGLM

model = SuperGLM(
    family="gaussian",
    features={
        "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing),
        "region": RandomEffect(),
    },
)
model.fit_reml(df, y)
```

Nothing special is being done here. To the extended Fellner–Schall update a
variance component and a smoothing parameter are the same object — Fellner's
original 1986 case was a penalty assembled from identity blocks, which is what
a random effect contributes — so the credibility shrinkage is estimated by the
same iteration that picks the smoothing parameters. This is a shape-constrained
additive *mixed* model in the sense of Pya Arnqvist (2024), §3.

Two properties worth knowing. On a truth that already obeys the constraint the
constrained fit reproduces the unconstrained one, variance component included:
a shape constraint that does not bind does not perturb the credibility
estimate. And when it does bind, the constraint is honoured on the term itself,
not on predictions — `predict()` adds the level effect, so predicted values are
not monotone in `x` and are not supposed to be. Read the term's own curve via
`plot_data()` to check the shape.

## Current Guard Rails

These combinations are intentionally guarded:

- fit-time shapes, caps and step limits, including caps on a `Categorical`,
  with `selection_penalty > 0` (a term whose only rules are flat stretches fits
  under a selection penalty)
- fit-time shape constraints with `select=True`
- `kind="ns"` fit-time shape constraints
- `Constraint.fit.convex` or `Constraint.fit.concave` on a `PSpline` or
  `BSplineSmooth` with `degree > 3`
- a direction rule on a stretch, such as `C.increasing.below(5)`, on a spline
  of degree 1
- a curvature rule on a stretch, such as `C.convex.above(5)`, on a spline of
  degree 2 or less
- `Constraint.fit.cap` or `Constraint.fit.max_step` under any link but log
- `Constraint.fit.cap` or `Constraint.fit.max_step` on an `OrderedCategorical`
  with special levels
- `Constraint.fit.max_step` on a spline term
- `Constraint.fit.flat` with no stretch, or with a stretch that holds every knot
  span of the term
- `Constraint.fit.flat` on a spline of degree 0
- `polynomial_ranges=` on a term that also has a shape constraint
- a shape rule other than a cap on a `Categorical`

If you need one of these combinations, treat it as unsupported rather than
assuming it is a valid workflow.

## Post-Fit Repair

Post-fit repair still exists for all `Constraint.postfit.*` tokens:

```python
model.apply_shape_postfit(df)
```

`Constraint.postfit` takes only a single shape over the whole range. A
combination, a stretch, `flat`, a cap or `max_step` is fit-time only, and
`Constraint.postfit` refuses it.

Use it when you already have a fitted model and need a manual monotone,
convex, or concave repair. Do not treat it as the preferred modeling path when
a solver-backed
fit is available.

### Inference after a repair is withheld, not reported

A repair replaces the published coefficients with a **projection onto the
shape cone**. That projection is a constrained estimator, and its reference
distribution is not the unconstrained fit's, so the uncertainty attached to a
repaired term is withheld rather than reported:

| surface | behaviour for a repaired term |
|---|---|
| `summary()` / `metrics().summary()` | chi-square, p-value, `ref_df`, curve SE band, per-level SE and CI all withheld; the row prints `repaired (inference withheld)`; `detail="full"` prints no per-coefficient rows |
| `term_inference(name)` | warns and returns without SE/CI; `monotone_repaired` is `True` |
| `relativities(with_se=True)` | warns and drops `se_log_relativity` for the repaired feature's frame only |
| `simultaneous_bands(name)` | refuses |
| `plot()` / `plot_data()` | no interval band, via `term_inference` |

The point estimate, `edf`, and the fitted level or curve values are still
reported everywhere. Only a term that was **actually projected** is affected —
a constrained term that was already shape-feasible when another term's repair
ran keeps its inference in full.

`metrics().feature_se(name)` refuses for a repaired term, for the same reason
as `simultaneous_bands`.

One surface is deliberately **not** qualified:
`summary()["standard_errors"]["coefficient_se"]` (and its `_raw` sibling). It
is a single dict over every group rather than a statement about one term, and
it already spends the all-zeros array on "not selected", so blanking a repaired
group there would collide with that sentinel and change what a consumer reads
as a live value. It still returns the unconstrained fit's values for a repaired
term; read it with the caveat this section states.

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
constraint — `Constraint.fit.*` — rather than repairing after the fact. On the
QP engine that interval uses the covariance with no row held (see
[What the intervals of a ruled term mean](#what-the-intervals-of-a-ruled-term-mean)).

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
- state a relativity cap or a limit on the step between bands inside the fit,
  rather than editing the table afterwards
- the QP fits constrained splines of every kind, P-splines included
- keep `selection_penalty=0` for these workflows
- validate the fitted shape on a prediction grid before signing off
- interpret the constraint on the linear predictor scale
