# Feature Types

Choose the simplest feature type that matches the shape you want in the final
pricing model.

## Splines

`Spline(kind, k)` is the main public spline API. `k` is the public basis size
in the mgcv sense; the fitted smooth then absorbs the identifiability
constraint.

```python
Spline(kind="ps", k=14)                   # default P-spline choice
Spline(kind="bs", k=14)                   # integrated-derivative B-spline smooth
Spline(kind="cr", k=10)                   # cubic regression spline
Spline(kind="ns", k=10)                   # natural spline
Spline(kind="ps", k=14, select=True)      # REML + double-penalty shrinkage
Spline(kind="cr", k=12, m=(1, 2))         # multi-order penalty
```

### Which spline kind to choose

| Kind | Use when | Notes |
|------|----------|-------|
| `"ps"` | default pricing spline | P-spline with difference penalty |
| `"bs"` | you want a proper B-spline smooth / mgcv-style `bs` basis | integrated-derivative penalty on the same raw B-spline geometry |
| `"cr"` | you want a cubic regression spline / mgcv-style `cr` basis | natural boundary constraints plus identifiability |
| `"ns"` | you want a natural spline with fixed natural boundaries | does not support monotone fitting |

### Knot strategies

| Strategy | Description |
|----------|-------------|
| `"uniform"` | evenly spaced interior knots |
| `"quantile_rows"` | more knots where the training data is dense |
| `"quantile_tempered"` | blend between uniform and quantile placement |

`quantile_tempered` with a small `knot_alpha` is often a good pricing default
for skewed variables like Bonus-Malus.

### `select=True`

`select=True` adds mgcv-style double-penalty shrinkage to the spline term. This
is the REML-native way to let a smooth shrink toward linear or zero while
staying in the `fit_reml()` workflow.

### Multi-order penalties with `m=`

`m` can be a single integer or a tuple. With a tuple, the spline emits
multiple penalty components on the same coefficient block, each with its own
REML smoothing parameter.

```python
Spline(kind="cr", k=12, m=2)
Spline(kind="cr", k=12, m=(1, 2))
Spline(kind="ps", k=14, m=(2, 3))
```

Current limitations:

- `select=True + m=(...)` is supported for tuples allowed by the selected
  spline class (for example, `m=(1, 2)` for P-splines and cubic regression
  splines). It produces separate null-space and derivative-order penalty
  components, each with its own REML lambda. Per-class order limits still
  apply.
- tensor interactions with multi-order spline parents are not yet supported
- `kind="cr_cardinal"` currently supports only `m=2`

## Monotone Splines

If monotonicity is part of the model specification, prefer solver-backed
monotone fitting rather than post-fit repair.

```python
from superglm import BSplineSmooth, Constraint, CubicRegressionSpline, PSpline

BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)   # QP
CubicRegressionSpline(n_knots=8, constraint=Constraint.fit.decreasing)  # QP
PSpline(n_knots=10, constraint=Constraint.fit.increasing)        # SCOP
```

See [Monotone Splines](monotone.md) for the full decision guide.

## Polynomial

Orthogonal polynomials are a good option when the shape is simple and stable.
The basis is orthonormalized against the training `sample_weight`, so the
per-power coefficient estimates are uncorrelated (exactly under fixed weights,
approximately at the IRLS working weights) and the classic "is the cubic
needed?" drop test reads cleanly from the per-power z-statistics. When exposure
enters through an offset (`offset=np.log(exposure)`, the documented count
workflow) `sample_weight` stays at ones, so the basis is orthonormalized
against the row-count measure.

```python
Polynomial(degree=2)            # common quadratic pricing curve
Polynomial(degree=3)            # cubic
Polynomial(powers=[1, 2, 4])    # keep the linear, quadratic and quartic components
```

`powers=` selects orthogonal *components*: the basis is built up to
`max(powers)` and the stated components are kept, so under fixed weights
dropping a middle power leaves the retained coefficients unchanged. Excluding
power 3 excludes the degree-3 orthogonal component, not the raw `x**3`
monomial. Dropping powers by their z-statistics is response-driven selection —
validate out-of-fold or state the powers from the plan. The standardization is
a main-effect property: interaction blocks built from polynomial margins are
products of components and are not themselves weight-orthonormal.

## Piecewise

`Piecewise(breaks=[...])` fits a continuous kinked line on stated breakpoints:
one coefficient per knot, each the log relativity of that knot against the
base knot. Free joins are already a binned `Categorical` and smooth joins are
already a `Spline`; this is the remaining cell — continuous, deliberately not
smooth, with every kink a stated, testable input.

```python
Piecewise(breaks=[3000, 6000, 12000])   # stated kinks — the filed form
Piecewise(breaks=4)                     # exploratory: weighted-quantile placement
```

Because the breaks are *inputs*, the design stays linear in every parameter
and each per-knot Wald row is ordinary regression inference — none of the
estimated-breakpoint machinery (Muggeo 2003, *Stat. Med.* 22:3055–3071) is
needed, and none of its corrections apply.

On the numeric axis a `Piecewise` is deliberately degree-1 only: the exported
knot rows *are* the model under linear interpolation, so the workbook alone
reproduces every prediction exactly. Any higher degree would break that
workbook-alone contract, so a numeric-axis `degrees=` other than all-1 refuses
loudly. Higher-degree shapes on a numeric axis belong to `Spline`, or to the
composition below.

### The hinge composition (curvature plus a stated corner)

To combine a smooth body with a stated corner at `c` — the cap-then-flat and
curve-then-straight shapes banded factors keep wanting — compose two terms on
transformed columns instead of reaching for a segmented numeric basis:

```python
X = X.assign(
    age_body=X["age"].clip(upper=60),
    age_tail=X["age"].clip(lower=60),
)
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={
        "age_body": Spline(kind="cr", n_knots=6),   # smooth below the corner
        "age_tail": Piecewise(breaks=[70]),         # kinked line above it
    },
)
```

Value continuity at `c` is structural (both terms are flat past their own
range), the kink at `c` is an estimated, testable contrast, and the penalties
stay block-separate. The device is the additive hinge basis of MARS —
reflected pairs of one-sided truncated power functions (Friedman 1991,
"Multivariate Adaptive Regression Splines", *Ann. Statist.* 19(1):1–67) —
with the knot stated rather than searched. A C¹ seam between the blocks would
need a cross-block constraint and is out of scope; if the join must be smooth,
use one `Spline` with stated knots instead.

## Categorical

Categoricals are one-hot encoded with a reference level. The entire factor is
treated as one group for selection and inference.

```python
Categorical(base="most_exposed")
Categorical(base="first")
Categorical(base="B")
```

### Collapsing Sparse Levels

`collapse_levels(...)` lets you merge sparse levels for fitting while keeping
the mapping back to original levels for inference and plotting.

```python
from superglm import Categorical, collapse_levels

grouping = collapse_levels(df["Area"], groups={"Rural": ["E", "F"]})
area = Categorical(base="most_exposed", grouping=grouping)
```

This is useful when a tariff factor has many thin levels but you still want a
single grouped factor inside the model. Interaction terms and interaction
screening use the same mapping: pass original labels at fit and predict time,
and each grouped interaction is built in the collapsed level geometry.

## RandomEffect

`RandomEffect()` gives every observed factor level a penalized intercept. REML
estimates the shared variance component, so thin levels shrink more strongly
toward the portfolio prediction than thick levels.

```python
RandomEffect()                    # unseen levels use the population prediction
RandomEffect(unseen="error")      # fail on an unseen level
```

It is the SuperGLM analogue of mgcv's `s(group, bs="re")`. Unlike
`Categorical`, it retains every level rather than choosing a reference level,
and it requires `fit_reml()`. See [Credibility terms](credibility.md) for
reporting, prediction, and a real insurance example.

## OrderedCategorical

Use `OrderedCategorical(...)` when a factor has a real order and you want a
smooth effect across levels. Level positions are equally spaced unless you
provide an explicit `values={level: position}` mapping.

```python
OrderedCategorical(
    order=["A", "B", "C", "D"],
    basis=Spline(kind="ps", k=6),
)
```

`basis=` is the only configuration channel and takes the shape itself — a
`Spline(...)`, a `Piecewise(...)`, or a `Polynomial(...)` object; omitting
`basis` keeps the default P-spline (`kind="ps"`, `n_knots=5`). The legacy
`basis="spline"` string, the spline shortcut arguments (`kind=`, `n_knots=`,
`degree=`, `select=`, `penalty=`), and step smoothing with `basis="step"` were
removed in 0.24.0 — configure the shape on `basis=`, or use `Categorical(...)`
for independent level effects.

With `basis=Spline(...)`, inference follows the spline model, not a saturated
categorical model. The summary reports one Wood-style whole-smooth p-value for
the ordered term; its null hypothesis is that the smooth contributes no
variation after centering. Per-level rows are base-relative effect estimates
with standard errors and confidence intervals, but deliberately have no
p-values or significance codes. Changing the reporting base therefore changes
the displayed level contrasts, not the whole-smooth p-value.

### Choosing the shape on a band axis

Every option states structure in band vocabulary; pick by what you are
prepared to defend:

- **Smooth, with knots at stated bands** — `Spline(knots=["Mi060", "Mi066"])`.
  A spline *is* the C¹ piecewise polynomial, so smooth-at-stated-breaks needs
  no new device: each knot name resolves to the named level's *value* on the
  smooth's axis, so names and numeric entries live on one scale.
- **A corner you can test** — `Piecewise(breaks=["Mi060", "Mi066"])`. Stated
  kinks, no smoothing penalty, one summary row per break answering "do I need
  this kink?". Integer positions are the escape hatch for unnamed axes.
- **No breaks, one global shape** — `Polynomial(powers=[1, 2])`: the classical
  orthogonal ordinal contrasts (`contr.poly`'s device) on the level positions,
  orthonormalized against the training exposure, one clean-z row per stated
  power. Classical trend practice keeps lower-order contrasts under a
  significant higher one (the hierarchical convention); `powers=` deliberately
  allows non-contiguous subsets, each component individually in or out.
- **Segmented curves** — `Piecewise(breaks=[...], degrees=[...])`, one degree
  per segment: the classical grafted/segmented polynomial (Gallant & Fuller
  1973, *JASA* 68:144–147) with value-continuous seams.
- **A grouped tail** — degree `0`: the plateau model (Anderson & Nelson 1975,
  *Biometrics* 31:303–318). `degrees=[2, 1, 0]` reads "curved, then straight,
  then flat".

```python
mileage_bands = ["Mi006", "Mi030", "Mi060", "Mi090", "Mi120", "Mi180"]
OrderedCategorical(
    order=mileage_bands,
    basis=Piecewise(breaks=["Mi060", "Mi120"], degrees=[2, 1, 0]),
    specials=["MISSING"],
)
```

The inner basis evaluates on level *positions* `0..L-1` (with `values=`, the
values still set the order but not the spacing — band structure is
positional). Rating-table export stays one row per band whatever the basis, so
the workbook is exact at any degree — which is why per-segment `degrees=`
exist here and are refused on the numeric axis. `Piecewise`'s `extrapolation`
parameter is inert on a level axis (every level is in range by construction).

For a segmented term the summary reports **structural contrasts**: one
slope-change Wald row per stated break and one curvature row per segment of
degree ≥ 2 — ordinary fixed-knot inference in the truncated-power
parameterization (Smith 1979, *Am. Statist.* 33(2):57–62). There are
deliberately no per-segment per-power z rows: with value-continuous seams the
segments share their joint values, so within-segment orthogonal components
are not free parameters and that clean-z geometry does not exist. The clean
per-power z is exactly what `basis=Polynomial(...)` gives, because there the
whole term is one orthogonal family.

An editor collapse (or a `grouping=`) that merges a stated break level with a
neighbour, or spans levels on both sides of one, refuses loudly naming the
break — a break is a stated kink, and regrouping it is a spec change, not an
edit. Grouping entirely within a segment stays allowed, and the named break
follows its level to the new position. On a grouped term's plot, the
between-band shape is display interpolation only: the rated values are the
band markers themselves, and the stated kink lives at the break even where
the drawn curve smooths it. The same guard covers `Spline` knots
given by name. Terms with a `Piecewise` or `Polynomial` basis cannot parent
interactions and are deferred by interaction screening (the interaction
machinery crosses a penalized marginal smooth).

### Free levels (`specials=`)

Some levels do not belong on the ordering at all — a `MISSING` band, a
structural zero. Listing them in `specials=` holds them out of the smooth and
fits each one as a free, unpenalized level effect:

```python
OrderedCategorical(
    order=["1", "2", "3", "4", "5", "6"],
    specials=["MISSING"],
    basis=Spline(kind="ps", k=6),
)
```

The smooth then spans the ordered levels only, and the special reports its own
base-relative relativity beside them. Use this for levels that are
*structurally* different, never for merely sparse ones: the penalty already
handles a sparse band better than a free level does.

The summary marks each level in a `Fit` column reading `smooth` or `free`, and
the exported workbook records the term as `smooth+free` with the special's own
row as a `free level`. Plots draw the fitted curve across the ordered levels
and place free levels as detached points past its end.

**The reported intercept changes when you add or remove a special.** The
smooth's identifiability constraint is taken over the rows it is built on, so
with `specials=` the intercept is the baseline of the *ordered* rows alone;
without it, the special's rows are inside that baseline. Reporting relativities
against the base level removes that constraint shift exactly, so the two models'
level relativities are directly comparable; their intercepts are not.

That is a statement about *reporting*, not a claim that the fitted curve is
identical. Adding a special leaves the ordered levels' fitted values unchanged
to machine precision only when the term is the model's sole predictor and the
smoothing parameter is held fixed. With another correlated, imbalanced predictor
the shared IRLS weights change and the ordered curve moves a little — measured
at around 1e-3 in log relativity with one imbalanced factor at a 5% special
share — and it moves again if `fit_reml()` re-selects the smoothing parameter,
which the default path does. Expect small differences rather than none.

A special must be present in the training data and carry positive weight (an
all-zero indicator column, or one whose rows all have zero weight, has no
identifiable coefficient), may not be the reporting `base=`, and may not be
merged into a level group. `specials=` works with every `basis=` — the main
block comes first and the unpenalized special block second, always;
interactions and PSST screening on a term with specials are not supported yet
and are reported as deferred rather than silently skipped.

This interpretation depends on the numeric positions assigned to the levels.
`order=[...]` uses equal spacing on `[0, 1]`; use `values={...}` when the real
distances are unequal. If spacing and smoothness are not defensible assumptions,
fit `Categorical(...)` and use a whole-term comparison instead.

## Numeric

`Numeric()` is a simple passthrough for continuous variables that should enter
linearly.

```python
Numeric()
```

## Interactions

Interactions are declared separately via `interactions=[(...)]`, and the type
is inferred from the parent specs.

```python
model = SuperGLM(
    features={"age": Spline(kind="ps", k=14), "region": Categorical()},
    interactions=[("age", "region")],
    selection_penalty=0.01,
)
```

See [Interactions](interactions.md) for the full interaction map.
