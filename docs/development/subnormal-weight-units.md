# Objective units for very small fitting weights

An ordinary direct IRLS fit lifts its weights and smoothing parameters by a
common, even power of two before forming weighted products when the largest
prior weight is below `2**-512`. This fixes loss of information in a subnormal
Gram matrix; equilibrating that already rounded matrix cannot recover its
entries. The objective and its constraints have the same minimizer after the
lift. Every constrained step still uses `solvers/constrained_qp.py`.

This follows the safe-scaling principle in [LAPACK DLASCL](https://netlib.org/lapack/lapack-3.1.1/html/dlascl.f.html)
and the gradual-underflow analysis of [Demmel (1984)](https://doi.org/10.1137/0905062).
It uses portable binary64 arithmetic throughout. An even exponent makes the
change of unit after a square root exact as well. The trigger is an exponent
range criterion, not a rank or convergence tolerance.

The chosen exponent puts the largest weight in `[1, 4)` unless an active
penalty limits it. The range check reads the existing penalty components and
leaves their sum below `2**960`; it also keeps each active lambda representable
before penalty assembly. Configured penalties absent from the objective do
not limit the lift. If no positive lift fits, the fit keeps its original
unit and its existing behavior. No penalty target, rank tolerance, solver,
or convergence refusal is added.

## Ownership and publication

`PIRLSResult.weight_exponent = K` identifies the unit of all retained rank and
factor geometry: `H_solver = 2**K H_physical`. Estimated dispersion is retained
as `solver_phi` in the same unit. Known-scale families keep `phi = 1` and
`solver_phi = None`. Published scalar dispersion, deviance, log determinant and
iteration weight/deviance diagnostics use physical units. Likelihood sizes
and frequency counts always use physical weights. Old saved results have
class defaults `K = 0`, `solver_phi = None`.

Legacy `fit_path` results keep their Pearson dispersion for quasi-likelihood
diagnostics. Known-scale coefficient tables continue to use unit dispersion
for those results as well.

Inference pairs dispersion with its inverse before changing units. When a
covariance entry exceeds binary64's range, a covariance view carries its
base and even exponent until a standard error or test statistic is formed.
A physical variance can overflow while its standard error remains finite.
A view may also carry an even part of dispersion's exponent. Recovering a
response influence from its base must divide by the matching dispersion
mantissa, preserving both parts of the covariance's unit.
Evaluation geometry may use a different unit from the fit; its dispersion
and inverse are paired accordingly. Post-fit repair uses the same unit for
its weights and smoothing penalties, and refreshes both dispersion readings
after a coefficient change.

No new numerical cache is retained. The physical weight reference is private
to one solve, and scaled weights and lambdas live only for that solve. A
covariance view shares the existing retained factor's lifetime and is
invalidated with the existing inference state. At `K = 0` there is no weight
or penalty copy and no rescaling.

## Likelihoods and limits

For estimated-scale prior densities, `W_solver / phi_solver` preserves the
row's variance without first rounding physical dispersion to zero. Frequency
densities retain their physical replication counts. Their saturated
normalizer uses `log(phi_solver) - K log(2)`, while the deviance ratio uses
solver units. The Tweedie calculation reuses its existing saturated series
and corrected saddlepoint; it accepts the logarithmic dispersion on this
range path.

Evaluation weights can be much larger than fitting weights. Row standard
deviations and frequency deviance ratios combine normal mantissas and their
net exponents before restoring units; they never require an overflowing
common lift of the evaluation weights.

Gamma and Tweedie quantile residuals at relative dispersion below `2**-512`
use their normal limit. The standardized skewness is at most
`2 sqrt(relative_dispersion) < 2**-255`, far below unit roundoff
`u = 2**-53`, including at the quantile residuals' existing probability
clipping endpoints. See [DLMF 8.12.3 and 8.12.7](https://dlmf.nist.gov/8.12)
for Gamma and [Jorgensen (1987)](https://doi.org/10.1111/j.2517-6161.1987.tb01685.x)
for exponential dispersion models. Independent density checks use rectangular
least squares, SciPy and high-precision Gamma and Bessel references in tests.

REML/fREML, moment-reuse callers and the selection-penalty PIRLS route keep
their existing units. Extending the lift to REML requires matching units for
log smoothing parameters, cached moments and criterion derivatives. This
change also leaves penalty dominance beyond binary64's range and working
weights that underflow because of the fitted mean to their existing routes.

Two additional limits remain follow-ups. At subnormal prior weights,
Tweedie `estimate_p` reaches the existing overflow in its dispersion-profile
search, on both the original and revised code. Its eventual repair will need
to carry dispersion units through profile publication as well. Some
discretization diagnostics still materialize physical covariance entries;
they need the same root-before-rescaling treatment to preserve finite
standard errors when variance entries overflow.
