# SuperLSS

`SuperLSS` fits several parameters of a response distribution together, one
predictor per parameter. Pass a family first, then one predictor declaration
per parameter using the family's helper methods; build the terms inside each
declaration with {py:func}`~superglm.s`, {py:func}`~superglm.cat`,
{py:func}`~superglm.re`, {py:func}`~superglm.ti`, {py:func}`~superglm.term`
and {py:func}`~superglm.interaction`. Start with the
[tutorial](../tutorials/distributional-model.md). The strip below follows a
model through its life. Each page under it opens with the members you reach
for first, and its table lists every member in the group, each with its own
page.

```{eval-rst}
.. autosummary::
   :nosignatures:

   superglm.SuperLSS
```

::::{grid} 2 3 5 5
:gutter: 2

:::{grid-item-card} 1 · Declare
:link: distributional/declarations-and-families
:link-type: doc
A family, then one predictor per parameter.
:::
:::{grid-item-card} 2 · Fit
:link: distributional/fit
:link-type: doc
`fit_reml` estimates smoothing jointly; `fit` holds it fixed.
:::
:::{grid-item-card} 3 · Predict
:link: distributional/predict
:link-type: doc
Means, every parameter, quantiles, simulated draws.
:::
:::{grid-item-card} 4 · Check
:link: distributional/check-the-fit
:link-type: doc
Residuals, binned moments, calibration, scores.
:::
:::{grid-item-card} 5 · Price
:link: distributional/price
:link-type: doc
Risk curves, density fans, the book total.
:::
::::

All the groups, in the order you meet them:

```{toctree}
:maxdepth: 1

distributional/declarations-and-families
distributional/fit
distributional/inference
distributional/predict
distributional/check-the-fit
distributional/price
distributional/plot
distributional/certification
distributional/save-and-load
distributional/configuration
```
