# Plot

{py:meth}`~superglm.SuperLSS.plot` draws a grid of term panels per parameter
with pointwise and simultaneous bands, one figure per parameter, which is the
view that sets what drives the location against what drives the scale.
{py:meth}`~superglm.SuperLSS.plot_data` returns the JSON-clean payload behind
any figure without drawing it, keyed by `kind`, so a front end can draw it
without the model. {py:meth}`~superglm.SuperLSS.plot_diagnostics` is the
six-panel distributional diagnostic: the first three panels ask whether the
family is right, the last three where it is wrong.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.plot
   ~superglm.SuperLSS.plot_data
   ~superglm.SuperLSS.plot_diagnostics
```
