# Deployment and Serialization

`SuperGLM` deployment is straightforward: fit once, serialize the fitted estimator, load it later, and call `predict()` on new rows.

This is especially important for the native explicit-spec API. A fitted `SuperGLM` is not just a matrix of coefficients. It also carries:

- the registered feature specs (`Spline`, `Categorical`, `Numeric`, ...)
- learned spline geometry such as knot locations and boundaries
- fitted coefficients and intercept
- REML smoothing parameters when you used `fit_reml()`

That is model state, not generic preprocessing.

## Native API round-trip

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
        "age": Spline(kind="bs", k=12, knot_strategy="quantile_tempered", knot_alpha=0.2),
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
- reconstruct spline terms with `term_inference()`
- produce summaries and relativities

without refitting.

## sklearn Pipeline round-trip

If you want explicit preprocessing upstream, keep it in the pipeline and let `SuperGLMRegressor` consume the transformed DataFrame.

The key trick is to preserve column names with:

```python
column_transformer.set_output(transform="pandas")
```

That way the final estimator can still refer to transformed columns by name, for example:

- `spline__age` for the spline feature passed through untouched
- `num__density` for a scaled numeric
- `cat__region_A`, `cat__region_B`, ... for one-hot columns

Example:

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

This keeps preprocessing explicit while still letting the final estimator own spline fitting and REML smoothing.

### Pipeline with native `features=`

When you need heterogeneous spline configs or want the full power of the native API inside a pipeline, pass `features=` directly:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from superglm import SuperGLMRegressor, Spline, Numeric

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
                    "keep_age__age": Spline(kind="bs", k=12, knot_strategy="quantile_tempered"),
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

`features=` is mutually exclusive with the shorthand wrapper arguments (`spline_features`, `categorical_features`, `numeric_features`, non-default `n_knots`/`degree`/`categorical_base`). The wrapper validates this at fit time.

## Why this is different from a plain spline transformer

A standalone spline basis transformer only expands columns into basis functions.

`SuperGLM` does more than that:

- stores the spline spec itself
- fits the penalized model
- estimates smoothness via REML when requested
- keeps enough state for post-fit inference and plotting

So the fitted estimator is the deployment artifact.

## Runnable example

Run:

```bash
uv run python scratch/examples/deployment_roundtrip.py
uv run python scratch/examples/sklearn_pipeline_roundtrip.py
```

Those scripts do complete round-trips for:

- native `SuperGLM`
- sklearn `Pipeline` + `SuperGLMRegressor`

1. fits a spline-based Poisson model
2. serializes it with `pickle`
3. reloads it
4. verifies predictions and spline metadata are unchanged
