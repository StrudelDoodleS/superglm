# Plot

{py:meth}`~superglm.SuperLSS.plot` draws a grid of term panels per parameter
with pointwise and simultaneous bands, one figure per parameter, which is the
view that sets what drives the location against what drives the scale.
{py:meth}`~superglm.SuperLSS.plot_data` returns the JSON-clean payload behind
any figure without drawing it, keyed by `kind`, so a front end can draw it
without the model. {py:meth}`~superglm.SuperLSS.plot_diagnostics` is the
six-panel distributional diagnostic: the first three panels ask whether the
family is right, the last three where it is wrong.

To inspect one row's fitted distribution directly,
{py:meth}`~superglm.SuperLSS.distribution_curve` computes the density,
tail-safe survival probability or mean excess over a grid of amounts (also
reachable as JSON through `plot_data("distribution", row=..., curve=...)`),
and {py:meth}`~superglm.SuperLSS.plot_distribution` draws it with Matplotlib
or Plotly. {py:class}`~superglm.SplitSplice` provides the same two methods
with body and tail marked across the threshold. Both methods are experimental
(their names, arguments and return layouts may change in a minor release
without a deprecation period, emitting one
{py:class}`~superglm.ExperimentalWarning` per session; their numbers are
tested like the rest of `superglm`). See
[Seeing the fitted law of one row](../../how-to/fit-a-distributional-model.md#seeing-the-fitted-law-of-one-row).

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.plot
   ~superglm.SuperLSS.plot_data
   ~superglm.SuperLSS.plot_diagnostics
   ~superglm.SuperLSS.distribution_curve
   ~superglm.SuperLSS.plot_distribution
```
