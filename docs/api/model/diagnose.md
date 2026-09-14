# Diagnose

{py:meth}`~superglm.SuperGLM.diagnostics` is the per-group dictionary for
programmatic and audit access. {py:meth}`~superglm.SuperGLM.spline_redundancy`
reports knot spacing, basis correlation and effective rank, and
{py:meth}`~superglm.SuperGLM.discretization_impact` measures what binning the
smooth terms into rating-table bins and grids does to the predictions.
{py:meth}`~superglm.SuperGLM.iteration_diagnostics` is the per-iteration IRLS
table, a DataFrame available after `fit(record_diagnostics=True)`;
{py:meth}`~superglm.SuperGLM.reml_diagnostics` and
{py:meth}`~superglm.SuperGLM.training_telemetry` are the solver's own records
as plain JSON-serialisable dictionaries with no tracking dependency, ready for
MLflow, files, logs or a governance system.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.diagnostics
   ~superglm.SuperGLM.spline_redundancy
   ~superglm.SuperGLM.discretization_impact
   ~superglm.SuperGLM.iteration_diagnostics
   ~superglm.SuperGLM.reml_diagnostics
   ~superglm.SuperGLM.training_telemetry
```
