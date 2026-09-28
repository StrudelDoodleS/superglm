# Families and links

Response families define the variance function and the weight semantics;
links map the linear predictor to the mean.
{py:meth}`~superglm.SuperGLM.estimate_theta` and
{py:meth}`~superglm.SuperGLM.estimate_p` estimate the extra parameter those
families carry and return the profile results listed here.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.families
   superglm.Poisson
   superglm.Gaussian
   superglm.Gamma
   superglm.Binomial
   superglm.NegativeBinomial
   superglm.Tweedie
   superglm.LogLink
   superglm.LogitLink
   superglm.IdentityLink
   superglm.ProbitLink
   superglm.CloglogLink
   superglm.CauchitLink
   superglm.InverseLink
   superglm.InverseSquaredLink
   superglm.SqrtLink
   superglm.PowerLink
   superglm.NegativeBinomialLink
   superglm.NBProfileResult
   superglm.TweedieProfileResult
   superglm.tweedie_logpdf
   superglm.generate_tweedie_cpg
```
