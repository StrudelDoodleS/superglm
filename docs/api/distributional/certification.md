# Smoothing certification and telemetry

After {py:meth}`~superglm.SuperLSS.fit_reml`, three attributes say how
automatic smoothing ended; after a plain {py:meth}`~superglm.SuperLSS.fit`,
which selects no smoothing, all three are `None`.
{py:attr}`~superglm.SuperLSS.smoothing_convergence_reason_` names why it
stopped (`fixed_only` when every penalty was fixed by policy).
{py:attr}`~superglm.SuperLSS.smoothing_certified_` is `True` only for a
strict interior stop with every check passed: the fit converged at a
stationary point rather than a `practical_plateau`, the coefficient solve
converged on the curvature it was asked for with no fallback, no term sat at
the finite cap, and no term ended on an exact face. Any one of those voids
it, whatever the family, so read the other attributes before treating
`False` as a defect: a term selected away or shrunk to its null space is
listed in {py:attr}`~superglm.SuperLSS.exact_face_components_`, a term still
pushing against the cap in
{py:attr}`~superglm.SuperLSS.smoothing_unresolved_upper_bound_`, and a
plateau stop, which the default `practical_reml=True` allows, in the
convergence reason. {py:attr}`~superglm.SuperLSS.exact_face_components_` is
always a tuple: the terms whose penalty was accepted as infinite. What such
a term keeps depends on its penalty and is named in the fit diagnostics:
nothing at all, a straight line, or the rest of the penalty's null space.
{py:attr}`~superglm.SuperLSS.coefficient_curvature` reports the curvature the
coefficient solve was asked to use, observed or Fisher; the curvature used
at the end, after any fallback, is `training_telemetry().curvature.actual_source`.
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
