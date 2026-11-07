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
w = np.array([1.0, 2.0, 1.0, 2.0] * 40)  # prior weights, the default reading
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
   row covers it and every key is right-open. A binned block's last row can also
   be closed, `[x, x]`, when it holds only the largest value: `bin_strategy="exact"`
   gives that value a band of its own when its factor differs from its
   neighbours', and any strategy does on a feature with a single value.

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

## Running In A Cloud Function

A function platform (Google Cloud Run functions, AWS Lambda, a Databricks
serving endpoint) starts and stops the processes that run your code. It starts
a new one after a deploy, when traffic rises past what the running ones can
take, and after it has shut idle ones down. A request that reaches a process
that is still starting waits for it: that wait is the cold start. Once the
process is up it serves many requests, so work done once at start costs
nothing per request.

### Load the model once per instance, outside the handler

Cloud Run functions run code at global scope once per instance
([Cloud Run functions: tips](https://docs.cloud.google.com/functions/docs/samples/functions-tips-scopes)),
and Lambda runs initialization code outside the handler once per execution
environment
([Lambda execution environments](https://docs.aws.amazon.com/lambda/latest/operatorguide/execution-environments.html)).
Put `import superglm` and the model load there, and keep the handler to
building the rows and predicting:

```py
import joblib
import pandas as pd

model = joblib.load("severity.joblib")  # runs once, when the instance starts


def handler(request):
    rows = pd.DataFrame(request.get_json()["rows"])
    link, se = model.predict_link(rows, se_fit=True)
    return {"link": link.to_dict("list"), "se": se.to_dict("list")}
```

A server that answers on several threads should also make one prediction in
that start-up code, before it serves. superglm loads each of its own modules
the first time a name needs it, and loads them one thread at a time. It cannot
make Python's own import locks resolve two threads that import its modules by
name at once (`import superglm.model` on one thread while another thread first
reads a name) in a process that has not loaded them yet.

### Keep a warm floor with the platform's own setting

A minimum number of instances that the platform keeps running removes the cold
start for requests that reach them. It does not remove it when traffic rises
past the minimum, because the platform then starts new instances, and idle
instances kept for the minimum are billed.

- Cloud Run functions: set a minimum number of instances
  ([Configure minimum instances](https://cloud.google.com/functions/1stgendocs/configuring/min-instances)).
- AWS Lambda: configure provisioned concurrency, which keeps execution
  environments initialized ahead of requests
  ([Lambda execution environments](https://docs.aws.amazon.com/lambda/latest/operatorguide/execution-environments.html)).
- Databricks model serving: set a minimum provisioned concurrency and leave
  scale to zero off. Databricks does not recommend scale to zero for
  production endpoints, and scaling up from zero usually takes 10 to 20
  seconds
  ([Create custom model serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)).

### What a cold start costs

Measured on one machine (AMD EPYC 9645, Python 3.13, Numba 0.68, one process
on a shared 16-core host), for a fitted `SuperLSS` with three predictors and
229 coefficients, whose file is 151 MiB:

| Step | Time | Resident memory |
|---|---|---|
| `import superglm` | 0.004 s | 14 MiB |
| `joblib.load` of the model, which imports numpy, pandas, SciPy and Numba | 0.8 to 0.9 s | 370 MiB |
| first `predict_link` of one row | 3 ms | |
| later `predict_link` of one row (median of 1,500 calls) | 0.9 ms | |
| later `predict_link(..., se_fit=True)` of one row | 1.8 ms | |

`import superglm` loads nothing until you use a name: plotting, scikit-learn,
the diagnostics and the fitting code are imported when something asks for
them. Loading the model is what imports the libraries a prediction needs, so
that is where the time goes, and the figures do not change with the Numba
cache.

A prediction of fewer than 8,192 rows compiles nothing. From 8,192 rows up,
with covariate values that differ from row to row, the first call compiles two
small kernels: 0.14 s when Numba's cache holds them and 0.65 s when it does
not. A function that scores single quotes or small batches never reaches them,
and needs nothing from the next section.

### Ship Numba's cache for large batches

Run one `predict_link` on a batch of 8,192 rows or more, whose numeric values
differ from row to row, while the image is built, with `NUMBA_CACHE_DIR` set,
and ship the directory in the image. The cache then holds the two kernels,
about 100 KiB. Numba keys it
to the CPU model and features, to the file's contents and to the path superglm
is installed at; see the
[Numba notes on caching](https://numba.readthedocs.io/en/stable/developer/caching.html)
and the how-to *Starting fast on ephemeral clusters* for the same keys at
cluster scale.

Numba needs a directory it can write to, even to read a cache that is already
full. AWS Lambda container images run on a read-only file system with a
writable `/tmp`
([Lambda container images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)),
and on Cloud Run functions the directory holding your source is read-only while
the temporary directory is an in-memory file system
([Cloud Run functions execution environment](https://cloud.google.com/functions/docs/concepts/execution-environment),
[best practices](https://docs.cloud.google.com/run/docs/tips/functions-best-practices)).

When no cache location is set (no `NUMBA_CACHE_DIR`, no
`NUMBA_CACHE_LOCATOR_CLASSES`, and none in `numba.config` or
`.numba_config.yaml`) and neither the directory superglm is installed in nor
your user cache directory can be written to, `import superglm` makes a new
directory for the process inside a directory of its own in the temporary
directory (`superglm-numba-cache-u<your user id>`, made for you alone, mode
0700, on Linux and macOS), points Numba at it, and removes the new directory
when the process exits. Loading the model works with no setting from you.
Nothing Numba compiles outlives the process, so the first prediction of 8,192
rows or more compiles its two kernels (0.65 s), and a function that scores
single quotes or small batches is not affected. superglm never changes a cache
location you set, and a directory you name has to be one you can write to.

Child processes inherit the setting and share the cache while the process that
made it runs. A child that imports superglm after that process has ended finds
the directory gone, does not trust a directory that is not private to its user,
and makes its own. A child that never imports superglm and outlives the
process can make Numba create the directory again, inside the directory that
only your user can enter; set `NUMBA_CACHE_DIR` yourself for such children.

To keep the benefit of a shipped cache on such a platform, copy it into a new
private directory before anything imports superglm, and point Numba at the
copy:

```py
import os
import shutil
import tempfile

cache = tempfile.mkdtemp(prefix="numba-cache-")  # new, and readable by you alone
shutil.copytree("/opt/numba-cache", cache, dirs_exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = cache

import superglm  # noqa: E402
```

Numba reads this cache with `pickle`, so use a directory that no other user
can create or write to: not a fixed name such as `/tmp/numba-cache`.

Measured on a read-only file system with the cache copied to the temporary
directory this way, the first prediction of 8,192 rows took 0.14 s, against
0.61 s when no cache had been shipped.

A cache built on one CPU model is not read on another, and a platform may start
your function on either. The Numba documentation names `NUMBA_CPU_NAME` for a
cache that has to be read on other CPUs
([environment variables](https://numba.readthedocs.io/en/stable/reference/envvars.html)).
Setting `NUMBA_CPU_NAME=generic` when the cache is built and again when the
function runs worked here: a process that set it read the cache in 0.18 s
instead of compiling in 0.49 s, a process that did not set it ignored the cache,
and the 638,007-row prediction took the same 0.54 s with the generic cache as
with the default one. We have not tried a second CPU model.

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
