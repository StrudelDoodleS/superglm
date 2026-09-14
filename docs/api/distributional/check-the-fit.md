# Check the fit

The residual checks start from {py:meth}`~superglm.SuperLSS.residuals`: the
probability-integral transform of each row under its fitted law, or its normal
inverse, which a correct family makes uniform or standard normal;
{py:meth}`~superglm.SuperLSS.residual_set` is the full payload the residual
comes from. {py:meth}`~superglm.SuperLSS.check` bins those residuals along a
covariate and reports mean, standard deviation and skewness per bin with
bootstrap bands, so it says where the fit is wrong and in which moment;
{py:meth}`~superglm.SuperLSS.check_2d` reports the mean on a grid of two
covariates. {py:meth}`~superglm.SuperLSS.actual_expected` reports realised
against predicted totals per bin as a ratio of weighted sums, and
{py:meth}`~superglm.SuperLSS.calibration` answers the coverage, tail, quantile
and reliability questions in one payload. {py:meth}`~superglm.SuperLSS.scores`
gives proper scores per row, the log score and the CRPS, and
{py:meth}`~superglm.SuperLSS.compare` the paired score difference against
another fitted candidate. The [how-to on checking a distributional
fit](../../how-to/check-a-distributional-fit.md) reads each of these in turn.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.residuals
   ~superglm.SuperLSS.residual_set
   ~superglm.SuperLSS.check
   ~superglm.SuperLSS.check_2d
   ~superglm.SuperLSS.actual_expected
   ~superglm.SuperLSS.calibration
   ~superglm.SuperLSS.scores
   ~superglm.SuperLSS.compare
```
