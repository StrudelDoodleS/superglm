# Plot

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
