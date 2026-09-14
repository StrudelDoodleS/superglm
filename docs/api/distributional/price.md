# Price and portfolio views

{py:meth}`~superglm.SuperLSS.risk_curves` sweeps one covariate and returns
predicted response quantiles with posterior bands drawn from one shared draw
set, so the curves are coherent with one another;
{py:meth}`~superglm.SuperLSS.density_fan` is the same sweep but returns the
whole conditional density at each point, the picture that shows a shape
change; it supports continuous families only, and families with atoms refuse.
{py:meth}`~superglm.SuperLSS.parameter_spread` shows how far the fitted
parameters spread across rows and, among identically priced rows, how far the
tail probability does. {py:meth}`~superglm.SuperLSS.portfolio` simulates the
total over a book of rows, optionally by segment, carrying the dependence the
shared coefficient draws induce.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.risk_curves
   ~superglm.SuperLSS.density_fan
   ~superglm.SuperLSS.parameter_spread
   ~superglm.SuperLSS.portfolio
```
