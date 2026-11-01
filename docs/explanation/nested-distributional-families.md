# Why Nested Families Have Their Own Class

Three `SuperLSS` families are special cases of a richer one: the gamma and the
log-normal are generalized gammas, and the Gaussian is a two-piece normal with
no skew. Each still has its own class. This page says when a special case earns
a class of its own and gives the evidence for each one that has.

The numbers come from
[`benchmarks/benchmark_nested_families.py`](https://github.com/StrudelDoodleS/superglm/blob/master/benchmarks/benchmark_nested_families.py),
which fits each special case beside its general family on the same simulated
rows. Its output is
[`benchmarks/nested_families_receipt.json`](https://github.com/StrudelDoodleS/superglm/blob/master/benchmarks/nested_families_receipt.json).
Wall times are single-threaded. Held-out log scores are per row, and lower is
better. A difference is reported with its paired standard error, as
`model.compare` gives it (see
[Checking a distributional fit](../how-to/check-a-distributional-fit.md)).

## Which families nest in which

| Family | Special case of | Where | Pinned by |
|---|---|---|---|
| `GammaLS` | `GeneralizedGammaLSS` | `shape = scale` (Prentice's `Q = σ`; the CV is `σ`) | `test_generalized_gamma_at_shape_equal_scale_is_the_gamma_family` |
| `LogNormalLS` | `GeneralizedGammaLSS` | `shape = 0`, a limit of the generalized gamma's formulas rather than a value you substitute | `test_log_normal_equals_the_generalized_gamma_at_shape_zero` |
| `LogNormalLS` | `TwoPieceLogNormalLSS` | `skew = 0` | `test_log_normal_equals_the_two_piece_log_normal_at_zero_skew` |
| `GaussianLS` | `TwoPieceNormalLSS` | `skew = 0` | `test_two_piece_normal_at_zero_skew_is_the_gaussian_family` |

Each test checks that the two families give the same row log-likelihood and
score at that point, to within rounding, and the log-normal and Gaussian tests
check the curvature too. So a special case is never a second, slightly
different definition of its law.

Three relationships are left out of the table on purpose:

- **Weibull** is the generalized gamma at `shape = 1`
  (`test_generalized_gamma_reduces_to_lognormal_and_weibull_densities`). It has
  no class: fit `GeneralizedGammaLSS` and read its `shape` against 1. A
  Weibull class would need the same evidence as the families below.
- **`TweedieLSS` does not contain the gamma.** Its power is fitted on
  `1 < p < 2`, so the gamma is the limit as `p` approaches 2, not a member.
- **`SuperGLM`'s `poisson()` and `gamma()`** are the limits as `p` approaches 1
  and 2 of `tweedie(p)`, which also accepts only `1 < p < 2`. A Tweedie fit
  cannot stand in for either.

## When a special case gets its own class

A special case keeps a class only when at least one of these holds, and the
claim is measured:

1. **It supports something the general family refuses.**
2. **Its regression model cannot be declared in the general family.**
3. **Where its restriction holds, it fits faster, or predicts better on
   held-out rows at the sample sizes people fit.**

It must also equal the general family at its special point, which the tests
above pin.

A special case that meets none of the three has no reason for a class. It is
just a value of the general family's parameter.

| Family | 1. Supports what the general family refuses | 2. Cannot be declared in the general family | 3. Faster, or better on held-out rows |
|---|---|---|---|
| `GammaLS` | yes: prior weights for averaged claims | yes, whenever the CV has a model | better at small samples |
| `LogNormalLS` | no | yes | faster; better at small samples |
| `GaussianLS` | no | yes | faster |

## 1. Averaged claims: only the gamma has a law for them

The gamma is an exponential dispersion model, and a reproductive one. The
average of `w` independent gamma claims with the same mean and CV is again a
gamma, with the same mean and CV `/ √w` (Jørgensen 1997, chapter 3). That is
the law `GammaLS` gives a row with prior weight `w`, so one row can carry a
policy's average claim, with the claim count as its weight.

The generalized gamma and the log-normal have no such law: an average of
log-normal claims is not log-normal. Both refuse non-unit prior weights, and
the error tells you to fit claim-level rows instead.

Measured on 25,155 gamma claims (CV 1.2) from 10,000 policies:

| Fit | Rows | Mean table against the claim-level fit | CV | Wall s |
|---|---|---|---|---|
| `GammaLS`, claim-level rows | 25,155 | (reference) | 1.197 | 0.52 |
| `GammaLS`, policy averages, claim counts as prior weights | 10,000 | within 5.6e-5 relative | 1.204 | 0.25 |
| `GeneralizedGammaLSS`, policy averages, prior weights | 10,000 | refused | | |
| `LogNormalLS`, policy averages, prior weights | 10,000 | refused | | |

The gamma's prior-weighted law is pinned against scipy by
`test_gamma_prior_weighted_functions_match_scipy`. The two refusals are pinned
by `test_nonunit_prior_weights_are_refused_and_unit_prior_bridges_to_frequency`
in the generalized-gamma and log-normal family tests.

## 2. The restricted model cannot be declared in the general family

A generalized gamma is a gamma only when `shape` equals `scale` on every row.
When the CV varies with risk factors, the shape has to vary with it, through
the same predictor. The generalized gamma's shape is a separate predictor on
an identity link, and nothing ties it to the scale. So a gamma with a CV model
is not a generalized gamma model you can write down.

Measured on 20,000 training and 20,000 held-out gamma claims whose CV runs
from 0.6 to 1.6 with a covariate `x2`:

| Fit | Held-out log score, minus `GammaLS` | Wall s |
|---|---|---|
| `GammaLS`, CV on `x2` | (reference) | 0.39 |
| `GeneralizedGammaLSS`, scale on `x2`, constant shape | +0.0074 ± 0.0009 (worse) | 0.74 |
| `GeneralizedGammaLSS`, scale and shape both on `x2` | +0.0002 ± 0.0001 | 0.78 |

Giving the shape its own model recovers the fit only approximately, and at
twice the time. It also spends coefficients estimating a relationship
the gamma fixes exactly.

The other restrictions cannot be declared either. A `SuperLSS` predictor
cannot be held at a fixed value: a shape or skew predictor with no terms and
no intercept is refused. So the only way to fit the generalized gamma at
`shape = 0`, or the two-piece normal at `skew = 0`, is to fit `LogNormalLS` or
`GaussianLS`.

## 3. Where the restriction holds, the special case is cheaper and steadier

On 20,000 training and 20,000 held-out rows from each special case's own law:

| Data | Special case, wall s | General family, wall s | Held-out log score, general minus special |
|---|---|---|---|
| gamma, CV 1.2 | `GammaLS` 0.44 | `GeneralizedGammaLSS` 0.45 | +0.0001 ± 0.0001 |
| log-normal, scale on `x2` | `LogNormalLS` 0.15 | `GeneralizedGammaLSS` 0.41 | +0.00005 ± 0.00003 |
| Gaussian, scale on `x2` | `GaussianLS` 0.13 | `TwoPieceNormalLSS` 0.20 | −0.000001 ± 0.000002 |

The fitted shapes sat where the special case puts them: generalized-gamma shape
1.22 against a fitted scale of 1.18, shape −0.011 against 0, and skew −0.0003
against 0.

At this size the extra parameter costs nothing in prediction, but it can cost
nearly three times the fitting time.

Censoring the same kind of claims at a policy limit, the 85th percentile of the
training claims, widens the log-normal's lead. Its censored rows need only the
normal distribution function, while the generalized gamma's need the incomplete
gamma or its uniform expansion:

| Data, 15% censored | Special case, wall s | General family, wall s | Held-out log score, general minus special |
|---|---|---|---|
| log-normal, scale on `x2` | `LogNormalLS` 0.14 | `GeneralizedGammaLSS` 0.55 | −0.00002 ± 0.00002 |
| gamma, CV 1.2 | `GammaLS` 0.46 | `GeneralizedGammaLSS` 0.55 | +0.00004 ± 0.00003 |

At 300 training rows the extra parameter costs prediction as well. These are
40 replicates, each scored on the same 5,000 held-out rows:

| Data | Held-out log score, `GeneralizedGammaLSS` minus special case | Replicates where `GeneralizedGammaLSS` is worse | Fitted shape, mean ± sd | Mean table, RMS relative error |
|---|---|---|---|---|
| gamma, CV 1.2 | +0.0042 ± 0.0011 | 82% | 1.28 ± 0.22 (gamma: 1.2) | 0.164 (`GammaLS` 0.165) |
| log-normal, scale 0.9 | +0.0023 ± 0.0005 | 88% | 0.02 ± 0.16 (log-normal: 0) | 0.125 (`LogNormalLS` 0.124) |

The mean table is no less accurate; what suffers is the shape of the predicted
distribution, which is what prices layers and large losses. This matches the
long-standing finding that generalized-gamma estimates are unsteady in small
samples, and that below a few hundred observations little is gained over the
simpler Weibull (Hager, Bain and Antle 1971).

Two further differences do not show up in a table:

- **The log-likelihood is smooth.** The Gaussian and log-normal log-likelihoods
  are smooth to every order. The two-piece families' curvature jumps at the
  mode, so REML's smoothing-parameter search has to fall back to finite
  differences for rows near it. See [Derivative orders, and which families
  supply them](../how-to/fit-a-distributional-model.md#derivative-orders-and-which-families-supply-them).
- **The mean always exists.** A gamma or log-normal mean is always finite. A
  generalized gamma with negative shape has no mean once `scale · |shape| ≥ 1`,
  and a mean-form fit has to back away from that region.

## Choosing between a special case and its general family

Fit both and compare them on held-out rows with `model.compare`. The
comparison is paired row by row, so its standard error is that of the
difference.

The special case is enough when two things hold:
- the general family's estimated shape (or skew) sits near the special value:
  the scale for the gamma, 0 for the log-normal, 1 for the Weibull;
- the held-out difference is inside about two standard errors.

In Prentice's parametrisation every one of those special values lies inside
the shape's range, not on its edge (Prentice 1974; Cox et al. 2007). So the
general family can settle on a special case from either side, and a shape
estimate near it means what it appears to.

## References

- Cox, C., Chu, H., Schneider, M. F. and Muñoz, A. (2007). Parametric survival
  analysis and taxonomy of hazard functions for the generalized gamma
  distribution. *Statistics in Medicine* 26(23), 4352–4374.
- Hager, H. W., Bain, L. J. and Antle, C. E. (1971). Reliability estimation for
  the generalized gamma distribution and robustness of the Weibull model.
  *Technometrics* 13(3), 547–557.
- Jørgensen, B. (1997). *The Theory of Dispersion Models*. Chapman & Hall.
- Prentice, R. L. (1974). A log gamma model and its maximum likelihood
  estimation. *Biometrika* 61(3), 539–544.
