# Smoothing certification and telemetry

After {py:meth}`~superglm.SuperLSS.fit_reml`,
{py:attr}`~superglm.SuperLSS.smoothing_certified_` says whether the fit met
strict matched certification and
{py:attr}`~superglm.SuperLSS.smoothing_convergence_reason_` how automatic
smoothing stopped, and
{py:attr}`~superglm.SuperLSS.smoothing_unresolved_upper_bound_` lists the
smoothing components with unresolved pressure at the finite cap; all three are
`None` for a fixed fit. {py:attr}`~superglm.SuperLSS.exact_face_components_`
always returns a tuple, the components accepted at the exact infinity face.
{py:attr}`~superglm.SuperLSS.coefficient_curvature` reports which curvature
the coefficient solve was asked to use, observed or Fisher, and
{py:meth}`~superglm.SuperLSS.training_telemetry` returns the immutable audit
metadata for the accepted fit.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.smoothing_certified_
   ~superglm.SuperLSS.smoothing_convergence_reason_
   ~superglm.SuperLSS.smoothing_unresolved_upper_bound_
   ~superglm.SuperLSS.exact_face_components_
   ~superglm.SuperLSS.coefficient_curvature
   ~superglm.SuperLSS.training_telemetry
```
