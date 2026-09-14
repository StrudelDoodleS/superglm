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

# Predict

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

{py:meth}`~superglm.SuperGLM.predict` returns the mean on the response scale
for new rows, with an optional offset and a choice of conditional or
population random effects. {py:meth}`~superglm.SuperGLM.relativities` returns
plot-ready relativity tables for every feature, the multiplicative form a
rating engine wants from a log-link model (under other links they are
exponentiated link-scale contributions, not factors of the mean);
{py:meth}`~superglm.SuperGLM.reconstruct_feature` returns one feature's fitted
curve or effect on its original scale.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.predict
   ~superglm.SuperGLM.relativities
   ~superglm.SuperGLM.reconstruct_feature
```

## Example

The same simulated motor book as the rest of this section: 4,000 policies, a
non-linear age effect, four regions and a mild vehicle-power slope, with
exposure carried as a log offset.

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

Three hand-written policies are enough to price. The model was fitted with
exposure in the offset, so `predict` with no offset returns the expected claim
count at one unit of exposure — a rate. Pass the offset for the exposure you
are actually rating and the mean scales with it: at half a year every number
below halves, because the offset enters the log link with a coefficient of one.

```{code-cell} ipython3
new_policies = pd.DataFrame(
    {
        "age": [22.0, 45.0, 70.0],
        "region": ["West", "North", "East"],
        "veh_power": [10.0, 7.0, 6.0],
    }
)
new_policies.assign(
    rate=model.predict(new_policies),
    half_year=model.predict(new_policies, offset=np.log(np.full(3, 0.5))),
).round(4)
```

`relativities` is the rating-engine view of the same fit: one table per
feature, multiplicative because the link is log. The region table is keyed by
level against the reference level, and the ordering it recovers — East cheapest,
then North, South and West — is the ordering the data was simulated with.

```{code-cell} ipython3
model.relativities()["region"].round(3)
```

`reconstruct_feature` returns one feature's fitted curve on its original scale,
with the interior knots and the coefficients in the original basis alongside,
which is what an external renderer or a rating engine needs to redraw the
smooth without SuperGLM in the loop.

```{code-cell} ipython3
age_curve = model.reconstruct_feature("age")
print("interior knots:", np.round(age_curve["knots_interior"], 1))
pd.DataFrame(
    {key: age_curve[key] for key in ("x", "log_relativity", "relativity")}
).head().round(3)
```
