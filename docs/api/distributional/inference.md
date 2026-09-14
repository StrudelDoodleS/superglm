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

{py:meth}`~superglm.SuperLSS.summary` gives one row per intercept and per term
of every parameter: effective degrees of freedom, smoothing parameter, the
Wood (2013) statistic with its p-value and, for single-coefficient terms, the
estimate and standard error. {py:meth}`~superglm.SuperLSS.term_inference`
sweeps one term of one parameter over its training range with pointwise and
simultaneous bands, and {py:meth}`~superglm.SuperLSS.term_test` is the test
that the term is flat; all three read the training frame, so a model restored
from bytes needs `X_train=`. The fitted attributes are the fit's own state:
{py:attr}`~superglm.SuperLSS.coef_` and
{py:attr}`~superglm.SuperLSS.coef_by_predictor_`,
{py:attr}`~superglm.SuperLSS.covariance_`,
{py:attr}`~superglm.SuperLSS.smoothing_parameters_`,
{py:attr}`~superglm.SuperLSS.result_`, and the
{py:attr}`~superglm.SuperLSS.family_`,
{py:attr}`~superglm.SuperLSS.predictors_` and
{py:attr}`~superglm.SuperLSS.parameter_names_` the fit resolved.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.summary
   ~superglm.SuperLSS.term_inference
   ~superglm.SuperLSS.term_test
   ~superglm.SuperLSS.parameter_names_
   ~superglm.SuperLSS.family_
   ~superglm.SuperLSS.predictors_
   ~superglm.SuperLSS.coef_
   ~superglm.SuperLSS.coef_by_predictor_
   ~superglm.SuperLSS.covariance_
   ~superglm.SuperLSS.result_
   ~superglm.SuperLSS.smoothing_parameters_
```

## Example

A simulated severity book of 3,000 claims. The location of the log-normal law
rises to a peak near age 30 and shifts by region; its scale widens steadily
with age. Both parameters carry real structure, so both predictors have
something to find.

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

The fit certified its smoothing parameters, so the summary below reads as a
converged fit rather than a stopping point.

```{code-cell} ipython3
model.summary().round(3)
```

One row per intercept and per term of each parameter. The age smooth in
`location` spends 6.3 effective degrees of freedom of the eight basis
functions it was given; the age smooth in `scale` spends 2.1 of six, so the
spread moves with age far more gently than the location does. Every term's
Wood statistic is large enough that its p-value rounds to zero at three
decimal places.

```{code-cell} ipython3
age_effect = model.term_inference("location", "age")
band = pd.DataFrame(
    {
        "age": age_effect.x,
        "effect": age_effect.effect,
        "lower": age_effect.lower,
        "upper": age_effect.upper,
        "lower_simultaneous": age_effect.lower_simultaneous,
        "upper_simultaneous": age_effect.upper_simultaneous,
    }
)
print(f"edf {age_effect.edf:.2f}, critical value {age_effect.critical_value:.2f}")
band.head().round(3)
```

`term_inference` returns the term swept over its training range: the centred
effect on the link scale — identity here, so log-amount units — with pointwise
bounds and the simultaneous ones. The simultaneous bounds use a critical value
of 2.97 rather than the pointwise 1.96, which is why they sit further out at
every age.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.fill_between(
    band["age"],
    band["lower_simultaneous"],
    band["upper_simultaneous"],
    alpha=0.25,
    label="simultaneous",
)
ax.fill_between(band["age"], band["lower"], band["upper"], alpha=0.45, label="pointwise")
ax.plot(band["age"], band["effect"], color="black", linewidth=1.5, label="effect")
ax.set_xlabel("age")
ax.set_ylabel("location effect (log amount)")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
```

The curve peaks near age 30 and falls away on both sides, which is the shape
the data were built with. The pointwise band answers "is the effect at this
age different from the average?"; the simultaneous band, about 1.5 times as
wide here, answers the question a pricing review actually asks — "could this
whole curve have been flat?" — and a flat line does not fit inside it.

```{code-cell} ipython3
model.term_test("scale", "age")
```

The same question for the scale predictor, as a test rather than a picture.
The statistic is 177 on 2.6 ranks and the p-value is far below any usual
threshold, so the width of the claim distribution genuinely varies with age:
a location-only model would misstate the tail at both ends of the age range.
