# SuperGLM

`SuperGLM` is the estimator for penalised GLMs and GAM-style pricing models.
Construct it with a family and a feature specification, fit it with
{py:meth}`~superglm.SuperGLM.fit_reml` for REML smoothness selection or
{py:meth}`~superglm.SuperGLM.fit` for fixed penalties, then read the fit
through {py:meth}`~superglm.SuperGLM.summary`,
{py:meth}`~superglm.SuperGLM.term_inference` and the plotting methods. The
strip below follows a model through its life. Each page under it opens with
the members you reach for first, and its tables list every member in the
group, each with its own page.

```{eval-rst}
.. autosummary::
   :nosignatures:

   superglm.SuperGLM
```

::::{grid} 2 3 5 5
:gutter: 2

:::{grid-item-card} 1 · Build
:link: model/build
:link-type: doc
A family, a feature spec, a penalty policy.
:::
:::{grid-item-card} 2 · Fit
:link: model/fit
:link-type: doc
`fit_reml` for REML; `fit` for a fixed penalty.
:::
:::{grid-item-card} 3 · Inference
:link: model/inference
:link-type: doc
Summary, per-term curves, diagnostics.
:::
:::{grid-item-card} 4 · Predict
:link: model/predict
:link-type: doc
Means, relativities, reconstructed effects.
:::
:::{grid-item-card} 5 · Deploy
:link: model/deploy
:link-type: doc
Rating tables and the payload behind them.
:::
::::

All the groups, in the order you meet them:

```{toctree}
:maxdepth: 1

model/build
model/fit
model/inference
model/predict
model/plot
model/diagnose
model/constrain-shapes
model/screen-interactions
model/deploy
model/results-and-records
```
