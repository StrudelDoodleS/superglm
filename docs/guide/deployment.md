# Deployment

The fitted estimator is the deployment artifact.

That matters more here than in a plain linear model because a fitted
`SuperGLM` contains:

- registered feature specs
- learned knot geometry and constraints
- fitted coefficients and intercept
- REML smoothing parameters
- enough state for summaries, plots, and term reconstruction

This is model state, not generic preprocessing.

## Native API Round-Trip

```python
import pickle
from pathlib import Path

import numpy as np
from superglm import Categorical, Numeric, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    discrete=True,
    features={
        "age": Spline(kind="ps", k=12, knot_strategy="quantile_tempered", knot_alpha=0.2),
        "density": Numeric(),
        "region": Categorical(base="most_exposed"),
    },
)

model.fit_reml(train_df, claim_count, offset=np.log(exposure), max_reml_iter=20)

with Path("pricing_model.pkl").open("wb") as f:
    pickle.dump(model, f)

with Path("pricing_model.pkl").open("rb") as f:
    loaded = pickle.load(f)

pred = loaded.predict(score_df, offset=np.log(score_exposure))
age_term = loaded.term_inference("age", with_se=False)

print(age_term.spline.interior_knots)
```

The loaded model can still:

- score new rows with `predict()`
- rebuild term-level curves with `term_inference()`
- produce summaries and relativity views

without refitting.

## Rating Table Export With Term Offsets

For rating-table deployment, offsets are exported as an applied multiplier
when the model was fitted with an offset. This is useful for policy-term
adjustments such as a 36-month policy costing three times a 12-month policy.

```python
import numpy as np
import pandas as pd

from superglm import Categorical, SuperGLM

train_df = pd.DataFrame(
    {
        "region": ["A", "B", "A", "B"] * 40,
        "term_months": [12.0, 12.0, 36.0, 36.0] * 40,
    }
)

y = np.array([0.3, 0.5, 1.1, 1.5] * 40)
w = np.array([1.0, 2.0, 1.0, 2.0] * 40)  # Gamma case/frequency weights
offset = np.log(train_df["term_months"].to_numpy() / 12.0)

model = SuperGLM(
    family="gamma",
    link="log",
    selection_penalty=0.0,
    features={"region": Categorical(base="first")},
)
model.fit(train_df[["region"]], y, sample_weight=w, offset=offset)

term = train_df["term_months"].to_numpy()
payload = model.rating_table_payload(
    train_df[["region"]],
    y,
    sample_weight=w,
    offset=offset,
    offset_source=term,
    offset_name="Term",
)
offset_table = next(block.table for block in payload.main_effects if block.name == "Term")
print(offset_table)

model.export_rating_tables(
    "rating_tables.xlsx",
    train_df[["region"]],
    y,
    sample_weight=w,
    offset=offset,
    offset_source=term,
    offset_name="Term",
)
```

The source-aware offset table is keyed by the raw deployment value:

```text
Term    Relativity    Weight
12      1.0           ...
36      3.0           ...
```

The fitted model still receives the link-scale offset:

```text
Raw source:        Term = 36
Link-scale offset: log(36 / 12)
Response factor:   3
```

When `offset_source` is supplied, the exporter validates that each raw source
level maps to one offset multiplier. A high-cardinality source is exported as a
single `per_unit` row carrying the derived scale — **`Relativity` *is* the scale,
and the factor is `Term × Relativity`** — rather than being binned, and the
proportionality is verified on every row before the block is written. Read that
equation carefully: taking `Relativity` to be `scale × Term` and then applying
the documented multiply rule computes `scale × Term²`.

If no `offset_source` is supplied, the `Offset Multiplier` block follows the same
rule, keyed on the multiplier itself because no column was named: exact levels up
to `offset_max_exact_levels`, and a single `per_unit` row above that. Neither is
an approximation. Binning is opt-in — `offset_kind="binned"` writes the
sample-weighted average multiplier per bin into the selected rating-table bin
count, as a summary of the fitted exposure rather than something to rate from.

### Exact Smooth Terms: `continuous_kind="ppform"`

A binned smooth term is the one part of the workbook that is an approximation by
construction — a spline has no keys, so the exporter invents them by binning.
`continuous_kind="ppform"` exports the fitted curve as the exact piecewise cubic
it already is. On a real motor book the worst row of a binned `DrivAge` block was
out by a factor of **0.600**; the same term as a ppform block is out by
**2.4e-15**, in 13 rows rather than 61.

The block is a **superset** of the ordinary three-column shape, so a loader that
has not been upgraded still finds it by the same header signature, slices the
same three columns, and scores it as the step function it scores today:

```
DrivAge                     Relativity   Weight     a          b          c          d
[-inf, 18.0)                  3.312618      0.00    1.197739   0          0          0
[18.0, 25.363636363636363)    3.312618   2896.43    1.197739  -0.401915  -1.095004   0.570373
...
[99.0, inf)                   2.783720      8.65    1.023788   0          0          0
```

Two rules a loader must implement, and one it must not assume:

1. **Bounded rows are evaluated.** Read both bounds out of the interval key —
   they round-trip exactly, which is why no `from`/`to` columns exist — and
   compute `factor = exp(a + u*(b + u*(c + u*d)))` with
   `u = (x - lower) / (upper - lower)`. `u` is normalised onto `[0, 1]`; a raw
   `x - lower` on a covariate ranging to 1e5 loses enough precision in a
   fixed-scale `DECIMAL` column to produce a 3.3× relativity error.
2. **Unbounded rows are read, not evaluated.** The leading and trailing rows are
   constant pieces and their factor is `Relativity`. Do **not** put them through
   the formula: `u` on an infinite width is `inf/inf`, which is `NaN`, and the
   zero higher coefficients do not absorb it because `0 * NaN` is `NaN`. You
   would get `NaN` on exactly the rows that price the extremes of the book. The
   branch costs nothing — `-inf` does not cast to `DECIMAL`, so an unbounded key
   has to be recognised before it can be parsed at all.
3. **Read the closing bracket.** Rows are `[lower, upper)` — except the last row
   under `extrapolation="error"`, which is `[lower, upper]` so that the boundary
   knot, which the model rates, has a row. Under `"clip"` the trailing unbounded
   row covers it and every key is right-open.

If you fingerprint staged rows with a content digest over an allow-list of
columns, **include the coefficients**. Two models differing only in `a b c d`
otherwise fingerprint identically, and the second is silently deduplicated into
the first.

## Production Framing

For deployment, the key question is usually not "how do I rebuild the design
matrix manually?" but "what exactly do I need to persist?" The answer is: the
fitted estimator.

That keeps:

- knot placement consistent with training
- monotone and boundary constraints consistent with training
- scoring behavior aligned with the fitted model
- inference and diagnostics reproducible after reload

## sklearn Pipeline Round-Trip

If you need upstream preprocessing, keep it explicit in the pipeline and let
`SuperGLMRegressor` consume the transformed DataFrame.

The main rule is:

```python
column_transformer.set_output(transform="pandas")
```

That preserves column names so the final estimator can refer to them.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from superglm import SuperGLMRegressor

pre = ColumnTransformer(
    [
        ("spline", "passthrough", ["age"]),
        ("num", StandardScaler(), ["density"]),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["region"]),
        ("meta", "passthrough", ["log_exposure"]),
    ]
).set_output(transform="pandas")

pipe = Pipeline(
    [
        ("pre", pre),
        (
            "model",
            SuperGLMRegressor(
                family="poisson",
                selection_penalty=0.0,
                spline_features=["spline__age"],
                offset="meta__log_exposure",
                n_knots=10,
            ),
        ),
    ]
)

pipe.fit(train_df, y)
pred = pipe.predict(score_df)
```

### Pipeline With Native `features=`

If you want full control over spline kinds and feature specs inside a pipeline,
pass `features=` directly:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from superglm import Numeric, Spline, SuperGLMRegressor

pre = ColumnTransformer(
    [
        ("keep_age", "passthrough", ["age"]),
        ("scale_density", StandardScaler(), ["density"]),
        ("meta", "passthrough", ["log_exposure"]),
    ]
).set_output(transform="pandas")

pipe = Pipeline(
    [
        ("pre", pre),
        (
            "model",
            SuperGLMRegressor(
                features={
                    "keep_age__age": Spline(kind="ps", k=12, knot_strategy="quantile_tempered"),
                    "scale_density__density": Numeric(),
                },
                offset="meta__log_exposure",
                selection_penalty=0.0,
            ),
        ),
    ]
)

pipe.fit(train_df, y)
pred = pipe.predict(score_df)
```

## Why This Is Not Just A Spline Transformer

`SuperGLM` does more than expand columns into basis functions:

- it owns the fitted spline specification
- it fits the penalized model
- it estimates smoothness via REML when requested
- it keeps enough state for post-fit inference and plotting

That is why the fitted estimator, not a detached transformer, is the thing you
deploy.

## Executable Round-Trip Checks

```bash
uv run pytest tests/test_core.py -q -k pickle_preserves_knots
uv run pytest tests/test_sklearn.py -q -k pickle_roundtrip
```

Those repository checks:

1. fit a spline-based Poisson model
2. serialize it with `pickle`
3. reload it
4. verify predictions and spline metadata are unchanged
