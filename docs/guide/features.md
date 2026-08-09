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
The basis is orthonormalized against the training exposure weights, so the
per-power coefficient estimates are uncorrelated (exactly under fixed weights,
approximately at the IRLS working weights) and the classic "is the cubic
needed?" drop test reads cleanly from the per-power z-statistics.

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
validate out-of-fold or state the powers from the plan.

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

The legacy `basis="spline"` and spline shortcut arguments such as `kind=` and
`n_knots=` are deprecated; configure them on `basis=Spline(...)`. Step smoothing
with `basis="step"` is also deprecated and will be removed. Use `Spline(...)`
for smoothing or `Categorical(...)` for independent level effects.

Inference follows the spline model, not a saturated categorical model. The
summary reports one Wood-style whole-smooth p-value for the ordered term; its
null hypothesis is that the smooth contributes no variation after centering.
Per-level rows are base-relative effect estimates with standard errors and
confidence intervals, but deliberately have no p-values or significance codes.
Changing the reporting base therefore changes the displayed level contrasts,
not the whole-smooth p-value.

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
merged into a level group. `specials=` requires `basis=Spline(...)`;
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
