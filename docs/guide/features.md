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

- `select=True + m=(...)` is not yet supported
- tensor interactions with multi-order spline parents are not yet supported
- `kind="cr_cardinal"` currently supports only `m=2`

## Monotone Splines

If monotonicity is part of the model specification, prefer solver-backed
monotone fitting rather than post-fit repair.

```python
from superglm import BSplineSmooth, CubicRegressionSpline, PSpline

BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit")   # QP
CubicRegressionSpline(n_knots=8, monotone="decreasing", monotone_mode="fit")  # QP
PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")        # SCOP
```

See [Monotone Splines](monotone.md) for the full decision guide.

## Polynomial

Orthogonal polynomials are a good option when the shape is simple and stable.

```python
Polynomial(degree=2)            # common quadratic pricing curve
Polynomial(degree=3)            # cubic
```

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
single grouped factor inside the model.

## OrderedCategorical

Use `OrderedCategorical(...)` when a factor has a real order and you want a
smooth or stepped effect across levels.

```python
OrderedCategorical(order=["A", "B", "C", "D"], basis="spline")
OrderedCategorical(order=["1", "2", "3", "4"], basis="step")
```

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
