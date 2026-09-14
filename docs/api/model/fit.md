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

# Fit

```{code-cell} ipython3
:tags: [remove-cell]

import logging
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
else:
    raise FileNotFoundError("superglm.mplstyle: run this page from its own directory")
```

{py:meth}`~superglm.SuperGLM.fit_reml` is the normal path: it estimates a
smoothing parameter for every penalised term by optimising a Laplace
approximate REML objective, except terms whose
{py:class}`~superglm.LambdaPolicy` pins the value, and it does not accept a
selection penalty. {py:meth}`~superglm.SuperGLM.fit` holds the smoothing
parameters fixed at the configured `spline_penalty` and is the path for sparse
and group selection; {py:meth}`~superglm.SuperGLM.fit_path` walks a
regularisation path from `lambda_max` down to `lambda_min`, warm-starting each
step from the last, and {py:meth}`~superglm.SuperGLM.refit_unpenalised` refits
on the active features alone with no selection penalty.
{py:meth}`~superglm.SuperGLM.estimate_p` and
{py:meth}`~superglm.SuperGLM.estimate_theta` profile the Tweedie power and the
NB2 dispersion respectively, then refit at the estimate. Before any of these,
{py:meth}`~superglm.SuperGLM.bind_levels` fixes every categorical level
universe from the full frame so that later fits on slices share it, and
{py:meth}`~superglm.SuperGLM.clone_unfitted` returns an independent copy of
the configuration without the fit. The [how-to on choosing a fitting
path](../../how-to/choose-a-fitting-path.md) says which to use when.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.fit
   ~superglm.SuperGLM.fit_reml
   ~superglm.SuperGLM.fit_path
   ~superglm.SuperGLM.refit_unpenalised
   ~superglm.SuperGLM.estimate_p
   ~superglm.SuperGLM.estimate_theta
   ~superglm.SuperGLM.bind_levels
   ~superglm.SuperGLM.clone_unfitted
```

## Example

The same simulated motor book as the rest of this section: 4,000 policies, a
non-linear age effect, four regions and a mild vehicle-power slope, with
exposure carried as a log offset. `fit_reml` estimates the smoothing parameter
for the age spline; nothing else has to be chosen.

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
    features={
        "age": Spline(kind="cr", k=10),
        "region": Categorical(),
        "veh_power": Numeric(),
    },
).fit_reml(X, claims, offset=offset)
model.reml_diagnostics()["converged"]
```

`reml_diagnostics` is the record of that outer loop: the smoothing parameter
after each step and the REML objective it reached. The first entry of
`lambda_history` is the starting value before any step, so the table below
pairs the steps taken with the objective they produced; not every optimiser
path records an objective, so the cell reads the history as optional rather
than assuming it. The objective falls by
about 2.4 at the first step and by a tenth at the second; the last two steps
move it by less than a thousandth, which is why the loop stops on its objective
tolerance.

```{code-cell} ipython3
diag = model.reml_diagnostics()
objective = diag.get("objective_history") or []
steps = diag["lambda_history"][1 : len(objective) + 1]
print(
    f"converged={diag['converged']} in {diag['n_reml_iter']} iterations "
    f"({diag['termination_reason']}), starting lambda "
    f"{diag['lambda_history'][0]['age']:g}"
)
pd.DataFrame(
    {
        "lambda_age": [step["age"] for step in steps],
        "reml_objective": objective[: len(steps)],
    },
    index=pd.RangeIndex(1, len(steps) + 1, name="iteration"),
).round(4)
```

`fit` is the same PIRLS solve with the smoothing parameters held where the
configuration put them, which is what the selection paths need. The model above
never set `spline_penalty`, so `clone_unfitted` plus `fit` refits the identical
specification at the default penalty. That penalty is far below the value REML
chose, so the age curve keeps all nine of its spline parameters (9.000 to three
decimals) instead of the 4.6 REML spent.

```{code-cell} ipython3
fixed = model.clone_unfitted().fit(X, claims, offset=offset)
pd.DataFrame(
    {
        "age edf": [model.term_inference("age").edf, fixed.term_inference("age").edf],
        "lambda": [
            model.term_inference("age").smoothing_lambda,
            fixed.term_inference("age").smoothing_lambda,
        ],
        "deviance": [
            model.metrics(X, claims, offset=offset).deviance,
            fixed.metrics(X, claims, offset=offset).deviance,
        ],
    },
    index=["fit_reml", "fit (fixed penalty)"],
).round(3)
```

Set `spline_penalty` yourself and the same call sweeps the trade-off by hand.
Deviance falls as the penalty falls, because a wigglier curve always fits the
training rows better; REML picks the point where the extra wiggle stops paying
for itself, which no row of this table can tell you.

```{code-cell} ipython3
rows = []
for penalty in (1.0, 100.0, 10_000.0):
    swept = SuperGLM(
        family="poisson",
        spline_penalty=penalty,
        features={
            "age": Spline(kind="cr", k=10),
            "region": Categorical(),
            "veh_power": Numeric(),
        },
    ).fit(X, claims, offset=offset)
    rows.append(
        {
            "spline_penalty": penalty,
            "age edf": swept.term_inference("age").edf,
            "deviance": swept.metrics(X, claims, offset=offset).deviance,
        }
    )
pd.DataFrame(rows).round(3)
```
