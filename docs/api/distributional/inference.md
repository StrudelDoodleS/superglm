# Inference

{py:meth}`~superglm.SuperLSS.summary` gives one row per intercept and per term
of every parameter: effective degrees of freedom, smoothing parameter, the
Wood (2013) statistic with its p-value and, for single-coefficient terms, the
estimate and standard error. {py:meth}`~superglm.SuperLSS.term_inference`
sweeps one term of one parameter over its training range with pointwise and
simultaneous bands, and {py:meth}`~superglm.SuperLSS.term_test` is the test
that the term is flat; all three read the training frame, so a model restored
from bytes needs `X_train=`. The fitted attributes are the fit's own state:
{py:attr}`~superglm.SuperLSS.coef_` and
{py:attr}`~superglm.SuperLSS.coef_by_predictor_`,
{py:attr}`~superglm.SuperLSS.covariance_`,
{py:attr}`~superglm.SuperLSS.smoothing_parameters_`,
{py:attr}`~superglm.SuperLSS.result_`, and the
{py:attr}`~superglm.SuperLSS.family_`,
{py:attr}`~superglm.SuperLSS.predictors_` and
{py:attr}`~superglm.SuperLSS.parameter_names_` the fit resolved.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.summary
   ~superglm.SuperLSS.term_inference
   ~superglm.SuperLSS.term_test
   ~superglm.SuperLSS.parameter_names_
   ~superglm.SuperLSS.family_
   ~superglm.SuperLSS.predictors_
   ~superglm.SuperLSS.coef_
   ~superglm.SuperLSS.coef_by_predictor_
   ~superglm.SuperLSS.covariance_
   ~superglm.SuperLSS.result_
   ~superglm.SuperLSS.smoothing_parameters_
```
