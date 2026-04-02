# Feature Types

## Splines

`Spline(kind, k)` is the recommended API for creating spline features. `kind` selects the basis type, `k` is the basis dimension matching mgcv's `k`. You can also use `n_knots` (interior knot count) instead of `k`.

```python
Spline(kind="bs", k=14)                   # 13-column P-spline (k-1 after identifiability)
Spline(kind="ns", k=10)                   # 9-column natural spline (k-1 after identifiability)
Spline(kind="cr", k=10)                   # 9-column cubic regression spline (k-1 after identifiability)
Spline(kind="bs", k=14, select=True)       # mgcv double penalty: spline-vs-linear selection
Spline(kind="cr", k=12, select=True)       # CR with double penalty selection
Spline(kind="cr", k=12, m=(1, 2))         # separate 1st- and 2nd-order penalties
```

| Kind | Basis | Penalty | Constraints | Built cols |
|------|-------|---------|-------------|-----------|
| `"bs"` | B-spline | Second-difference | Identifiability | `k - 1` |
| `"ns"` | B-spline | Second-difference | f''=0 at boundaries + identifiability | `k - 1` |
| `"cr"` | B-spline | Integrated f'' squared | Natural + identifiability | `k - 1` |

`k` matches mgcv's `k` for all kinds. The built column count is always `k - 1` because the identifiability constraint (unweighted sum-to-zero) removes one direction. mgcv absorbs this via a side constraint instead of physically removing the column.

### Multi-order penalties with `m=`

`m` can be either a single integer or a tuple of integers. With a tuple, the
spline emits multiple penalty components on the same coefficient block, each
with its own REML smoothing parameter.

- `kind="cr"` uses integrated derivative penalties.
- `kind="bs"` and `kind="ns"` use difference penalties.

Examples:

```python
Spline(kind="cr", k=12, m=2)        # default cubic-regression-spline penalty
Spline(kind="cr", k=12, m=(1, 2))   # first + second derivative penalties
Spline(kind="bs", k=14, m=(2, 3))   # second + third difference penalties
```

Current limitations:

- `select=True + m=(...)` is not yet supported.
- tensor interactions with multi-order spline parents are not yet supported.
- `kind="cr_cardinal"` currently supports only `m=2`.

### What `ssp` means here

In this repo, `penalty="ssp"` means the spline is stored in a reparameterized
form that separates:

- the sparse basis matrix `B`
- a small dense transform `R_inv`
- the penalty matrix `omega`

So internally the effective spline design is `B @ R_inv`, but the code tries
not to materialize that full product unless it has to. This makes the penalized
system better conditioned, keeps the basis-side operations sparse, and makes it
cheap to rebuild the spline representation when REML changes the smoothing
parameter.

That is why internal names like `SparseSSPGroupMatrix` and
`DiscretizedSSPGroupMatrix` show up in the optimization code: they are just the
factored spline representation used for penalized spline groups.

### Knot strategies

Control where interior knots are placed:

| Strategy | Description |
|----------|-------------|
| `"uniform"` | Evenly spaced across the feature range (default) |
| `"quantile_rows"` | At quantiles of the raw feature values — more knots where data is dense |
| `"quantile_tempered"` | Blend between uniform and quantile, controlled by `knot_alpha` |

`knot_alpha` (0–1) controls the tempering: 0 = fully uniform, 1 = fully quantile. Values like 0.2 pull knots slightly toward data-dense regions without fully collapsing them there. This is useful for heavy-tailed features like Bonus-Malus where pure quantile placement wastes knots in the long tail.

### Double-penalty shrinkage

`select=True` decomposes the penalty eigenspace into a linear subgroup and a wiggly subgroup, both penalised (mgcv-style double penalty). Works for BS, CR, and CR cardinal splines. With `fit_reml()`, REML estimates separate lambdas for each subgroup — driving a lambda to infinity effectively zeros that component. Three-way selection: nonlinear, linear, or dropped. Not supported for NS (its constrained penalty has only 1 null eigenvalue).

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

Single continuous feature. Group size 1, so group lasso reduces to standard L1.

```python
Numeric()
```
