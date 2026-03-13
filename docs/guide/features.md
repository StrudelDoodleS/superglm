# Feature Types

## Splines

`Spline(kind, k)` is the recommended API for creating spline features. `kind` selects the basis type, `k` is the basis dimension matching mgcv's `k`. You can also use `n_knots` (interior knot count) instead of `k`.

```python
Spline(kind="bs", k=14)                   # 14-column P-spline (default kind)
Spline(kind="ns", k=10)                   # 10-column natural spline (linear tails)
Spline(kind="cr", k=10)                   # 9-column cubic regression spline (k-1 after identifiability)
Spline(kind="bs", k=14, split_linear=True) # mgcv double penalty: spline-vs-linear selection
```

| Kind | Basis | Penalty | Constraints | Built cols |
|------|-------|---------|-------------|-----------|
| `"bs"` | B-spline | Second-difference | None | `k` |
| `"ns"` | B-spline | Second-difference | f''=0 at boundaries | `k` |
| `"cr"` | B-spline | Integrated f'' squared | Natural + identifiability | `k - 1` |

`k` matches mgcv's `k` for all kinds. For `"cr"`, the built column count is `k - 1` because the identifiability direction is physically removed (mgcv absorbs it via a side constraint instead).

### Knot strategies

Control where interior knots are placed:

| Strategy | Description |
|----------|-------------|
| `"uniform"` | Evenly spaced across the feature range (default) |
| `"quantile_rows"` | At quantiles of the raw feature values — more knots where data is dense |
| `"quantile_tempered"` | Blend between uniform and quantile, controlled by `knot_alpha` |

`knot_alpha` (0–1) controls the tempering: 0 = fully uniform, 1 = fully quantile. Values like 0.2 pull knots slightly toward data-dense regions without fully collapsing them there. This is useful for heavy-tailed features like Bonus-Malus where pure quantile placement wastes knots in the long tail.

### Double-penalty shrinkage

`select=True` (or `split_linear=True` for BS) decomposes the penalty eigenspace into a linear subgroup and a wiggly subgroup, both penalised (mgcv-style double penalty). With `fit_reml()`, REML estimates separate lambdas for each subgroup — driving a lambda to infinity effectively zeros that component. Three-way selection: nonlinear, linear, or dropped.

## Polynomial

Orthogonal polynomial (Legendre basis). Very stable across refits — ideal for features with simple monotone or quadratic shapes.

```python
Polynomial(degree=2)            # quadratic (common insurance choice)
Polynomial(degree=3)            # cubic (default)
```

## Categorical

One-hot encoded with a reference level. The entire factor is selected or removed as a group.

```python
Categorical(base="most_exposed")  # base = highest-exposure level (default)
Categorical(base="first")         # base = alphabetically first level
Categorical(base="B")             # explicit base level
```

## Numeric

Single continuous feature, standardised by default. Group size 1, so group lasso reduces to standard L1.

```python
Numeric()                       # standardised (default)
Numeric(standardize=False)      # raw scale
```
