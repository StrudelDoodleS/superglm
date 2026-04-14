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

## Runnable Examples

```bash
uv run python scratch/examples/deployment_roundtrip.py
uv run python scratch/examples/sklearn_pipeline_roundtrip.py
```

Those scripts:

1. fit a spline-based Poisson model
2. serialize it with `pickle`
3. reload it
4. verify predictions and spline metadata are unchanged
