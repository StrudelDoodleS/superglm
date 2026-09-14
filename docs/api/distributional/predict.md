# Predict

{py:meth}`~superglm.SuperLSS.predict` returns the conditional mean per row on
the response scale the model was fitted to, for a built-in family; a custom
family defines its own default prediction quantity.
{py:meth}`~superglm.SuperLSS.predict_parameters` returns every fitted
parameter on its natural scale, one column per name in
{py:attr}`~superglm.SuperLSS.parameter_names_`, and
{py:meth}`~superglm.SuperLSS.predict_link` the same columns on the link scale,
offsets included. {py:meth}`~superglm.SuperLSS.predict_cdf` and
{py:meth}`~superglm.SuperLSS.predict_quantile` evaluate the fitted law per
row. For uncertainty by simulation,
{py:meth}`~superglm.SuperLSS.posterior_draws` draws coefficients from the
fit's Bayesian posterior, {py:meth}`~superglm.SuperLSS.posterior_bounds`
pushes them through a parameter, a predictive quantile, an exceedance
probability or an expected shortfall to give per-row intervals, and
{py:meth}`~superglm.SuperLSS.posterior_predictive` simulates responses.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.predict
   ~superglm.SuperLSS.predict_parameters
   ~superglm.SuperLSS.predict_link
   ~superglm.SuperLSS.predict_cdf
   ~superglm.SuperLSS.predict_quantile
   ~superglm.SuperLSS.posterior_predictive
   ~superglm.SuperLSS.posterior_draws
   ~superglm.SuperLSS.posterior_bounds
```
