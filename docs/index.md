---
html_theme.sidebar_secondary.remove: true
---

```{rst-class} sg-visually-hidden
```

# superglm

```{raw} html
<div class="sg-hero">
  <div class="sg-hero__logo"><img src="_static/logo.png" alt="superglm"></div>
  <div class="sg-hero__copy">
    <div class="sg-hero__title">Super GLM</div>
    <p>Penalised GLMs and GAM pricing models for insurance, with the smoothness chosen by REML and the constraints you would otherwise enforce by hand.</p>
    <a class="sg-btn" href="get-started/index.html">Get started</a>
    <a class="sg-btn sg-btn--ghost" href="tutorials/index.html">Tutorials</a>
    <span class="sg-hero__meta">MIT licensed · free for everyone · built on NumPy, SciPy and pandas</span>
  </div>
</div>
```

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Get started
:link: get-started/index
:link-type: doc

Install, then the quick start: which fit to call, and how to read the summary it prints.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Executed notebooks you can open in Colab. First, a distributional model; pricing tutorials on French motor data follow.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

One goal per page: choosing a fit path, features and levels, constraints, screening, validation, deployment.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

Why REML, what weights mean, how credibility becomes smoothing, what screening can and cannot detect.
:::

::::

```{rst-class} sg-section-title
```

## Twelve lines to a fitted model

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
model.fit_reml(df, y, sample_weight=exposure)
print(model.summary())
```

Why the fitted curves look the way they do: [How REML chooses smoothness](explanation/how-reml-chooses-smoothness.md).

The [API reference](api/index.md) documents every public name. The
[governance section](governance/index.md) is for model-risk reviewers.

```{toctree}
:hidden:

get-started/index
tutorials/index
how-to/index
explanation/index
api/index
governance/index
development/index
```
