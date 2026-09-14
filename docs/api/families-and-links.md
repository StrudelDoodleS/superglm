# Families and links

Response families define the variance function and the weight semantics;
links map the linear predictor to the mean. The negative-binomial and Tweedie
profilers estimate the extra parameters those families carry.

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
   superglm.estimate_nb_theta
   superglm.NBProfileResult
   superglm.estimate_tweedie_p
   superglm.estimate_phi
   superglm.TweedieProfileResult
   superglm.tweedie_logpdf
   superglm.generate_tweedie_cpg
```
