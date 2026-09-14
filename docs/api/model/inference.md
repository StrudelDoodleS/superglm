---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Inference

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for candidate in (
    Path("../../_static/superglm.mplstyle"),
    Path("docs/_static/superglm.mplstyle"),
):
    if candidate.exists():
        plt.style.use(str(candidate))
        break
```

Start with {py:meth}`~superglm.SuperGLM.summary`, the statsmodels-style
coefficient table. {py:meth}`~superglm.SuperGLM.term_inference` is the object
behind every curve and band, the per-term curve, uncertainty and metadata in
one place; {py:meth}`~superglm.SuperGLM.simultaneous_bands` returns
simultaneous confidence bands for a spline feature, which hold jointly across
the curve. {py:meth}`~superglm.SuperGLM.metrics` computes the fit statistics
and {py:meth}`~superglm.SuperGLM.drop1` the drop-one deviance per feature;
{py:meth}`~superglm.SuperGLM.term_importance` scores each term by the weighted
variance of its contribution to the linear predictor, and
{py:meth}`~superglm.SuperGLM.term_drop_diagnostics` by what dropping it costs
in AIC, BIC or holdout loss. {py:meth}`~superglm.SuperGLM.random_effects` and
{py:meth}`~superglm.SuperGLM.factor_smooth` report variance components, level
diagnostics and smooth curves for random-effect and factor-smooth terms, while
{py:meth}`~superglm.SuperGLM.knot_summary`,
{py:meth}`~superglm.SuperGLM.design_summary` and the
{py:attr}`~superglm.SuperGLM.result` property expose what was actually built
and fitted.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.summary
   ~superglm.SuperGLM.term_inference
   ~superglm.SuperGLM.simultaneous_bands
   ~superglm.SuperGLM.random_effects
   ~superglm.SuperGLM.factor_smooth
   ~superglm.SuperGLM.metrics
   ~superglm.SuperGLM.drop1
   ~superglm.SuperGLM.term_importance
   ~superglm.SuperGLM.term_drop_diagnostics
   ~superglm.SuperGLM.knot_summary
   ~superglm.SuperGLM.design_summary
   ~superglm.SuperGLM.result
```

## Example

A simulated motor book: 4,000 policies, a non-linear age effect, four regions
with different base rates, a mild vehicle-power slope, and Poisson claim counts
whose rate is multiplied by exposure. The exposure enters as a log offset, so
the model estimates a claim rate per unit of exposure.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from superglm import Categorical, Numeric, Spline, SuperGLM

rng = np.random.default_rng(0)
n = 4000
region = rng.choice(["North", "South", "East", "West"], n, p=[0.35, 0.3, 0.2, 0.15])
age = rng.uniform(18, 80, n)
veh_power = rng.uniform(4, 12, n)
exposure = rng.uniform(0.1, 1.0, n)
region_effect = pd.Series(region).map(
    {"North": 0.0, "South": 0.25, "East": -0.2, "West": 0.4}
).to_numpy()
log_rate = (
    -1.2
    + 1.1 * np.exp(-((age - 24) ** 2) / 90.0)
    + 0.004 * (age - 50) ** 2 / 10.0
    + region_effect
    + 0.09 * (veh_power - 8)
)
claims = rng.poisson(np.exp(log_rate) * exposure)
X = pd.DataFrame({"age": age, "region": region, "veh_power": veh_power})
offset = np.log(exposure)

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={
        "age": Spline(kind="cr", k=10),
        "region": Categorical(),
        "veh_power": Numeric(),
    },
).fit_reml(X, claims, offset=offset)
model.reml_diagnostics()["converged"]
```

`summary` is the whole fit in one table: the header block carries the family,
the effective degrees of freedom and the fit statistics, then each feature gets
its own section. A spline is reported as one block — its basis size, its
effective degrees of freedom, the smoothing parameter REML chose and a Wood
(2013) test — rather than as nine uninterpretable coefficients. The
categorical levels are shown against the reference level.

```{code-cell} ipython3
print(model.summary())
```

`term_inference` is the object behind every curve and band. Its `x` grid,
`relativity` and pointwise `ci_lower` / `ci_upper` are plain arrays, so a
DataFrame is one call away. The age curve starts high for the youngest drivers
and falls; `edf` says the fit spent about 4.6 of the nine available spline
parameters on that shape.

```{code-cell} ipython3
ti = model.term_inference("age")
curve = pd.DataFrame(
    {
        "age": ti.x,
        "relativity": ti.relativity,
        "ci_lower": ti.ci_lower,
        "ci_upper": ti.ci_upper,
    }
)
print(f"edf {ti.edf:.2f}, lambda {ti.smoothing_lambda:,.0f}")
curve.head().round(3)
```

`simultaneous_bands` returns the same curve with both band types side by side.
Pointwise bands hold at each age separately; simultaneous bands hold jointly
across the whole curve, which is what you need before claiming the curve is
non-flat. They are wider for it — about half again as wide here. Drawn
together, the difference is the outer shaded band: even the simultaneous band
stays clear of the flat line at 1.0 below age 35 and again between the early
forties and the low seventies, so the shape survives the joint statement, not
only the pointwise one.

```{code-cell} ipython3
import matplotlib.pyplot as plt

bands = model.simultaneous_bands("age")
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(
    bands["x"],
    bands["ci_lower_simultaneous"],
    bands["ci_upper_simultaneous"],
    alpha=0.25,
    label="simultaneous 95%",
)
ax.fill_between(
    bands["x"],
    bands["ci_lower_pointwise"],
    bands["ci_upper_pointwise"],
    alpha=0.45,
    label="pointwise 95%",
)
ax.plot(bands["x"], bands["relativity"], color="black", linewidth=1.6, label="fitted")
ax.axhline(1.0, color="0.5", linewidth=0.8)
ax.set_xlabel("age")
ax.set_ylabel("relativity")
ax.set_title("Age relativity with pointwise and simultaneous bands")
ax.legend(loc="upper right")
fig.tight_layout()
```

`drop1` refits without each feature and reports the deviance it cost;
`term_importance` scores the same features by the weighted variance of their
contribution to the linear predictor. They answer different questions —
significance against spread — and here they agree on the order: age first,
then region, then vehicle power.

```{code-cell} ipython3
effects = model.drop1(X, claims, offset=offset)[
    ["feature", "delta_deviance", "delta_df", "p_value"]
].merge(model.term_importance(X)[["feature", "sd_eta", "edf"]], on="feature")
effects["p_value"] = effects["p_value"].map("{:.1e}".format)
effects.round(3)
```

`metrics` recomputes the fit statistics on any frame, so the same call gives
training numbers here and holdout numbers on a validation split. The fields are
plain attributes.

```{code-cell} ipython3
fit_metrics = model.metrics(X, claims, offset=offset)
pd.DataFrame(
    {
        "value": [
            fit_metrics.n_obs,
            fit_metrics.effective_df,
            fit_metrics.deviance,
            fit_metrics.explained_deviance,
            fit_metrics.log_likelihood,
            fit_metrics.aic,
            fit_metrics.bic,
        ]
    },
    index=[
        "n_obs",
        "effective_df",
        "deviance",
        "explained_deviance",
        "log_likelihood",
        "aic",
        "bic",
    ],
).round(3)
```
