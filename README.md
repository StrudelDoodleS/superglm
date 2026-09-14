<p align="center">
  <img src="https://raw.githubusercontent.com/StrudelDoodleS/superglm/master/docs/images/logo.png" alt="SuperGLM" width="300">
</p>

[![CI](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml/badge.svg)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/StrudelDoodleS/superglm/graph/badge.svg?token=2HO71TA2ZY)](https://codecov.io/github/StrudelDoodleS/superglm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)

Penalised GLMs and GAM-style pricing models for insurance. SuperGLM combines
explicit feature specs, exact REML, large-`n` discrete REML, solver-backed
monotone splines, actuarial validation tooling, and deployable fitted
estimators for Poisson, Gamma, NB2, Tweedie, Binomial, Gaussian, and Gaussian
or Gamma location–scale models.

## Install

```bash
pip install superglm
```

Interactive Plotly charts are optional: `pip install "superglm[plotting]"`.
The browser model editor is included.

## Fit a pricing model

```python
from superglm import Categorical, Numeric, Spline, SuperGLM

features = {
    "DrivAge": Spline(kind="ps", k=14, knot_strategy="quantile_rows"),
    "VehAge": Spline(kind="cr", k=10, knot_strategy="quantile_rows"),
    "BonusMalus": Spline(kind="cr", k=12, knot_strategy="quantile_tempered"),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}
model = SuperGLM(family="poisson", features=features)
model.fit_reml(train_df, y_train, sample_weight=exposure_train)
print(model.summary())
```

REML chooses the smoothness of every spline. Monotone and curvature
constraints are enforced inside the fit. `SuperLSS` fits location, scale and
shape parameters together for severity and distributional work.

## Documentation

- [Get started](https://strudeldoodles.github.io/superglm/get-started/index.html)
- [Tutorials](https://strudeldoodles.github.io/superglm/tutorials/index.html)
- [How-to guides](https://strudeldoodles.github.io/superglm/how-to/index.html)
- [Explanation](https://strudeldoodles.github.io/superglm/explanation/index.html)
- [API reference](https://strudeldoodles.github.io/superglm/api/index.html)
- [Governance](https://strudeldoodles.github.io/superglm/governance/index.html)

## Licence

MIT. Free for everyone, commercial use included.
