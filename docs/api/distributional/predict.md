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

{py:meth}`~superglm.SuperLSS.predict` returns the conditional mean per row on
the response scale the model was fitted to, for a built-in family; a custom
family defines its own default prediction quantity.
{py:meth}`~superglm.SuperLSS.predict_parameters` returns every fitted
parameter on its natural scale, one column per name in
{py:attr}`~superglm.SuperLSS.parameter_names_`, and
{py:meth}`~superglm.SuperLSS.predict_link` the same columns on the link scale,
offsets included. {py:meth}`~superglm.SuperLSS.predict_cdf` and
{py:meth}`~superglm.SuperLSS.predict_quantile` evaluate the fitted law per
row. For uncertainty by simulation,
{py:meth}`~superglm.SuperLSS.posterior_draws` draws coefficients from the
fit's Bayesian posterior, {py:meth}`~superglm.SuperLSS.posterior_bounds`
pushes them through a parameter, a predictive quantile, an exceedance
probability or an expected shortfall to give per-row intervals, and
{py:meth}`~superglm.SuperLSS.posterior_predictive` simulates responses.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.predict
   ~superglm.SuperLSS.predict_parameters
   ~superglm.SuperLSS.predict_link
   ~superglm.SuperLSS.predict_cdf
   ~superglm.SuperLSS.predict_quantile
   ~superglm.SuperLSS.posterior_predictive
   ~superglm.SuperLSS.posterior_draws
   ~superglm.SuperLSS.posterior_bounds
```

## Example

The same simulated severity book as the [inference page](inference.md): 3,000
claim amounts whose log-normal location peaks near age 30 and shifts by
region, and whose scale widens with age.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from superglm import LogNormalLS, SuperLSS, cat, s

rng = np.random.default_rng(1)
n = 3000
book = pd.DataFrame(
    {
        "age": rng.uniform(18, 80, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
    }
)
age = book["age"].to_numpy()
by_region = book["region"].map(
    {"North": 0.0, "South": 0.25, "East": -0.15, "West": 0.1}
).to_numpy()
location = 7.0 + 0.9 * np.exp(-(((age - 30) / 14) ** 2)) + by_region
scale = 0.35 + 0.006 * (age - 18)
amount = np.exp(rng.normal(location, scale))

family = LogNormalLS(parametrisation="location")
model = SuperLSS(
    family,
    family.location(s("age", kind="cr", k=8), cat("region")),
    family.scale(s("age", kind="cr", k=6)),
).fit_reml(book, amount, outer="efs+newton")
model.smoothing_certified_
```

`predict_parameters` gives the fitted law row by row, one column per name in
`parameter_names_`; `predict_link` gives the same two columns on their link
scales.

```{code-cell} ipython3
parameters = model.predict_parameters(book)
links = model.predict_link(book)
parameters.head(3).round(3).join(links.head(3).round(3), rsuffix="_link")
```

Under this parametrisation `location` is the mean of log amount and `scale`
its standard deviation. The location link is the identity, so `location_link`
repeats it; the scale link is `log(scale - 0.01)`, which is why 0.533 appears
as -0.648. Both columns move from row to row, which is the point of the
distributional fit: rows differ in spread as well as in level.

`predict`, `predict_quantile` and `predict_cdf` on the first five rows: the
conditional mean, the 90th percentile of the claim, and the observed claim
read back through its own fitted law.

```{code-cell} ipython3
first = book.head(5)
pd.DataFrame(
    {
        "age": first["age"].round(1),
        "region": first["region"],
        "mean": model.predict(first),
        "q90": model.predict_quantile(first, 0.9),
        "observed": amount[:5],
        "pit": model.predict_cdf(first, amount[:5]),
    }
).round(3)
```

`predict` is the conditional mean of the claim amount, not of its logarithm,
so it sits above the median of a right-skewed law. `predict_quantile` answers
the price question instead: the 90th percentile of what this policy would
claim, between 1.55 and 1.9 times the mean on these rows. `predict_cdf` reads
the observed claim back through its own fitted law — the five values are spread
across the unit interval, as they should be for rows that are neither
systematically over- nor under-predicted.

`posterior_predictive` simulates responses for those same rows, drawing
coefficients from the fit's posterior and then a response from each drawn
law, so the interval carries both parameter uncertainty and the claim's own
randomness.

```{code-cell} ipython3
draws = model.posterior_predictive(first, n_draws=500)
pd.DataFrame(
    np.quantile(draws, [0.05, 0.5, 0.95], axis=0).T,
    columns=["p05", "p50", "p95"],
).round(0)
```

Each row's 5% to 95% span covers a factor of roughly four to ten
— the spread of a single claim dwarfs the uncertainty in where its law sits.
