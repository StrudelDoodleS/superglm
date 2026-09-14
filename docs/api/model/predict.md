# Predict

{py:meth}`~superglm.SuperGLM.predict` returns the mean on the response scale
for new rows, with an optional offset and a choice of conditional or
population random effects. {py:meth}`~superglm.SuperGLM.relativities` returns
plot-ready relativity tables for every feature, the multiplicative form a
rating engine wants from a log-link model (under other links they are
exponentiated link-scale contributions, not factors of the mean);
{py:meth}`~superglm.SuperGLM.reconstruct_feature` returns one feature's fitted
curve or effect on its original scale.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.predict
   ~superglm.SuperGLM.relativities
   ~superglm.SuperGLM.reconstruct_feature
```
