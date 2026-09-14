# Smoothing certification and telemetry

After {py:meth}`~superglm.SuperLSS.fit_reml`, three attributes say how
automatic smoothing ended. {py:attr}`~superglm.SuperLSS.smoothing_convergence_reason_`
names why it stopped. {py:attr}`~superglm.SuperLSS.smoothing_certified_` is
`True` only when every part of that stop was checked analytically: the
gradient conditions held, no curvature fallback was needed, and any term
whose penalty ran to a boundary carries an analytic certificate. Families
whose boundary check is numerical (generalized gamma, generalized Pareto, the
two-piece families and log-normal) report `False` even on a converged fit.
{py:attr}`~superglm.SuperLSS.smoothing_unresolved_upper_bound_` lists the
terms whose penalty was still pushing against the finite cap when the fit
stopped. All three are `None` for a fit with fixed penalties.
{py:attr}`~superglm.SuperLSS.exact_face_components_` is always a tuple: the
terms whose penalty was accepted as infinite, leaving only what the penalty
cannot charge for (a straight line, for a spline).
{py:attr}`~superglm.SuperLSS.coefficient_curvature` reports which curvature
the coefficient solve used, observed or Fisher, and
{py:meth}`~superglm.SuperLSS.training_telemetry` returns the immutable audit
record of the accepted fit.

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
