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

# Plot

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

{py:meth}`~superglm.SuperGLM.plot` is the single entry point for drawing
terms: all main effects, one, a subset, or an interaction, with pointwise or
simultaneous bands on the main effects.
{py:meth}`~superglm.SuperGLM.plot_data` returns the plain DataFrames, arrays
and metadata behind those figures so you can rebuild them in matplotlib,
plotly, Excel or a reporting system.
{py:meth}`~superglm.SuperGLM.plot_diagnostics` is the four-panel residual
figure on quantile residuals, with a simulation-based Q-Q envelope.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.plot
   ~superglm.SuperGLM.plot_data
   ~superglm.SuperGLM.plot_diagnostics
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
    features={
        "age": Spline(kind="cr", k=10),
        "region": Categorical(),
        "veh_power": Numeric(),
    },
).fit_reml(X, claims, offset=offset)
model.reml_diagnostics()["converged"]
```

One named term draws one figure: the fitted age curve with its pointwise band
and, because `X` was passed, the density of the fitting rows underneath, so a
thin stretch of data cannot masquerade as a confident part of the curve. Drop
the term name to draw every main effect, pass a list for a subset, and pass
`ci="simultaneous"` for bands that hold jointly across the curve.

```{code-cell} ipython3
fig = model.plot("age", X=X, engine="matplotlib")
```

`plot_data` returns the numbers behind that figure instead of drawing it: one
entry per term, each with the effect frame, the density, and the metadata
describing how the curve was centred and how much of the basis the fit used.
Nothing in it needs SuperGLM to render — this is the handover to plotly, Excel
or a reporting system.

```{code-cell} ipython3
payload = model.plot_data("age", X=X)
age_term = payload["terms"][0]
print(
    f"kind={payload['kind']}, centering={age_term['metadata']['centering_mode']}, "
    f"edf={age_term['metadata']['edf']:.2f}"
)
age_term["effect"].head().round(3)
```

`plot_diagnostics` is the residual view rather than the effect view: four
panels on quantile residuals, with the Q-Q panel carrying an envelope
simulated from the fitted model, so the question "is this tail worse than this
model would produce anyway" has an answer on the page. It needs the frame and
the response back, because residuals are not stored on the model.

```{code-cell} ipython3
diagnostics = model.plot_diagnostics(X, claims, offset=offset)
```
