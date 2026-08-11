# Interactions

Tuple interactions are dispatched from their parent feature specs. They add
an interaction alongside the configured main effects; for two spline parents,
the result is a functional-ANOVA `ti()`-style interaction rather than a full
`te()` or thin-plate `tp()` surface.

```python
from superglm import Categorical, Spline, SuperGLM

model = SuperGLM(
    features={"age": Spline(k=14), "region": Categorical()},
    interactions=[("age", "region")],
    selection_penalty=0.01,
)
model.fit(df, y, sample_weight=exposure)
```

## Auto-detected interaction types

| Parent types | Interaction class | Geometry |
|---|---|---|
| Spline + Spline | `TensorInteraction` | `ti()`-style interaction, excluding parent main effects |
| Spline + Categorical | `SplineCategorical` | Reference-coded, unpooled spline deviations |
| Polynomial + Categorical | `PolynomialCategorical` | Reference-coded polynomial deviations |
| Numeric + Categorical | `NumericCategorical` | Per-level slopes |
| Categorical + Categorical | `CategoricalInteraction` | Cross-level indicators |
| Numeric + Numeric | `NumericInteraction` | Product term |
| Polynomial + Polynomial | `PolynomialInteraction` | Tensor-product polynomial term |

The dispatcher is intentionally type-based. With one spline it selects the
appropriate spline interaction; with no spline it selects the matching
numeric, polynomial, or categorical geometry. Full tensor-product `te()` and
thin-plate `tp()` APIs are not currently exposed.

An `OrderedCategorical` parent participates as a spline on its
mapped level scores — the same axis its own main effect uses — so
`OrderedCategorical` + `Categorical` builds a `SplineCategorical` and
`OrderedCategorical` + `Spline` builds a `TensorInteraction`. OC-parented tensors always use the exact
prediction path: fit-time discretization is skipped for them, because the
margin already lives on at most one score point per level and there is nothing
for binning to compress. Such pairs also screen — as `ti` and `spline_cat`
rows on the mapped scores; see [Interaction Screening](screening.md).

## Choosing a factor-varying curve

`SplineCategorical` and `FactorSmooth` answer related but different questions:

| Need | API | Main spline | Sparse-level behavior |
|---|---|---|---|
| Reference-coded fixed interaction | `SplineCategorical` | Model-dependent | No pooling |
| Fully penalized random curves | `FactorSmooth(..., basis="fs")` | Optional | Wiggle and null-space directions shrink |
| Centered deviation curves | `FactorSmooth(..., basis="sz")` | Required | Wiggle shrinks; polynomial null space remains |

Here `basis=` chooses the factor-smooth construction. `kind=` chooses the
continuous marginal spline family; this release supports `kind="ps"` for both
factor bases. SZ means “sum to zero”: it is a continuous rating curve for each
level, not a finite two-way rating table.

### Fully penalized FS curves

```python
from superglm import FactorSmooth, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={"DrivAge": Spline(kind="ps", k=8)},
    interactions=[
        FactorSmooth(
            "DrivAge",
            group="Region",
            basis="fs",
            kind="ps",
            k=6,
        )
    ],
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=log_exposure)
```

FS is analogous to mgcv's `bs="fs"`: every level has a complete curve and
shared wiggle/null-space smoothing components. It can be fitted without the
global `Spline`; when both are present, identifiability is supplied by the
full FS penalty, but the construction does not force the global curve to do as
much work as possible.

### Sum-to-zero SZ deviations

```python
from superglm import FactorSmooth, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={"age": Spline(kind="ps", k=7, m=2)},
    interactions=[
        FactorSmooth(
            "age",
            group="region",
            basis="sz",
            kind="ps",
            k=6,
            m=2,
        )
    ],
    selection_penalty=0.0,
).fit_reml(X, y)
```

SZ is analogous to mgcv's `bs="sz"` with one shared smoothing parameter. At
every value of `age`, the fitted regional deviations sum exactly to zero, so
the required global `Spline` is the portfolio curve and the SZ term describes
departures from it. Adding the global spline is therefore intended, not a
source of duplicate main-effect geometry.

`group=` names the factor whose levels receive curves. It corresponds to the
factor argument in mgcv's `s(x, factor, bs=...)`; it is not a generic `by=`
multiplier. SuperGLM rejects a `Categorical` or `RandomEffect` main effect on
that same grouping column because FS already contains its constant direction
and SZ's uncentered level curves contain lower-order factor geometry. A random
effect for a different column is supported.

## Level universes are inherited, never redeclared

A `Categorical`-parented interaction — `SplineCategorical`,
`CategoricalInteraction`, `NumericCategorical`, `PolynomialCategorical` — takes
its level universe from the parent term at build time. Whatever bound the
parent (a `levels=` list, the column dtype, a `cross_validate` full-frame bind,
or plain inference) binds the interaction too, along with the parent's pinned
levels and its `unseen=` policy. There is no `levels=` on the interaction
itself: one declaration on the main effect is the whole story, and a second one
would be a second thing to keep in step.

A level pinned on the parent gets no block in the interaction either — no
all-zero columns for a level with no data — and predicts through the parent's
base, so the interaction contributes nothing for those rows. See
[Feature types](features.md#the-level-universe) for the sources and the pin
semantics.

`FactorSmooth` is the one exception to "every source works everywhere": it is
not `Categorical`-parented, and in this release only the explicit
`FactorSmooth(levels=...)` channel reaches it — the column-dtype and
`cross_validate`/`bind_levels` full-frame channels bind main-loop features
only. Declare its universe explicitly when folds may drop a level. With
`basis="fs"` an empty declared level is absorbed by the penalty (its curve
shrinks to the population); `basis="sz"` rejects empty declared levels
outright, because a level with no rows makes the centered system numerically
singular (measured: minimum penalized eigenvalue collapses from ~0.6 to
~4e-10).

## Prediction behavior

Known levels receive their fitted FS curve or SZ deviation. With the default
`unseen="population"`, an unseen level receives zero deviation; set
`unseen="error"` to reject it with its label. Missing values always fail.
`FactorSmooth(levels=...)` binds the grouping factor's universe the same way
`Categorical` does, so folds and refreshes share one set of curves; an empty
declared level shrinks to the population smooth under `basis="fs"`, and is
refused under `basis="sz"`, whose sum-to-zero contrast needs every level to
carry rows.

```python
conditional = model.predict(test)
population = model.predict(test, random_effects="population")
```

`random_effects="population"` removes `RandomEffect` and `FactorSmooth`
contributions while retaining fixed effects and the global spline. For SZ,
that is exactly the global curve.

## Hierarchical make/model/trim data

Sparse nested vehicle levels can be represented with explicit composite IDs,
for example `make_model` and `make_model_trim`, and separate FS terms. Their
penalties are estimated independently: this is partial pooling of each curve,
not a correlated nested random-effect covariance.

Use one SZ term at a chosen hierarchy level when centered deviations are the
scientific target. Stacking make/model/trim SZ terms is not advertised as a
hierarchical decomposition in this release, because each constraint is global
rather than sum-to-zero within its parent. FS is usually the clearer first
choice for sparse nested levels.

See [Credibility terms](credibility.md) for reporting, prediction intervals,
and structured-solver behavior.
