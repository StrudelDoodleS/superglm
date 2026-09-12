# Migration: declare predictors through the family

`SuperLSS` now takes the family as its first positional argument, followed by
one declaration for every parameter. Terms include their input-column names.
This replaces the `family=` and `predictors=` keyword constructor.

## Replace predictor templates with family helpers

Before:

```python
from superglm import GaussianLS, Numeric, Predictor, Spline, SuperLSS

family = GaussianLS()
model = SuperLSS(
    family=family,
    predictors=(
        Predictor("location", {"age": Spline(kind="cr", k=8), "value": Numeric()}),
        Predictor("scale", {}),
    ),
)
```

After:

```python
from superglm import GaussianLS, SuperLSS, s

family = GaussianLS()
model = SuperLSS(
    family,
    family.location(s("age", kind="cr", k=8), "value"),
    family.scale(),
)
```

An empty helper replaces the empty feature mapping for an intercept-only
predictor. A bare column name replaces `Numeric()`. Use `cat("region")` for a
categorical effect, `re("broker")` for a random effect, or
`term("column", existing_spec)` for another feature specification.

The old constructor keywords raise `TypeError`. Lower-level `Predictor`
templates remain available for inspecting model configuration and for
numerical interfaces; the public constructor takes family-bound declarations.

## Unpack a sequence when building declarations programmatically

```python
family = GaussianLS()
declarations = (
    family.location("age"),
    family.scale(),
)
model = SuperLSS(family, *declarations)
```

Passing `declarations` without `*` supplies one tuple where a predictor is
expected. Declare each parameter exactly once. The family determines result
order, so reordering the declarations does not reassign their meanings.

## Move predictor controls onto the helper

```python
family = GaussianLS()
model = SuperLSS(
    family,
    family.location("age", intercept=False, link="identity"),
    family.scale(),
)
```

Each helper accepts `intercept` and `link`. For interactions, declare the
parent terms in the same predictor and include `ti("left", "right")` for a
two-spline tensor, or `interaction(existing_spec)` for another supported
interaction. `ti` supplies the interaction only; the parent smooths must be
declared separately.

## Keep parameter names in results and offsets

Tweedie's construction helpers are `mu`, `phi` and `p`. Their names in
prediction tables, offsets, penalty keys and saved models remain `mean`,
`dispersion` and `power`. For example, a known addition to the mean's log
predictor belongs under `offsets={"mean": values}`.

Existing supported model artifacts still load through `SuperLSS.from_bytes`.
The constructor migration does not change their parameter ordering or schema.

For a complete runnable example, follow
[Your first distributional model](../../getting-started/distributional.md).
