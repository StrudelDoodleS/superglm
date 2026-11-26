"""Model adequacy tests for count and overdispersion diagnostics.

Provides zero-inflation detection, score tests for zero-inflation
(van den Broek 1995), Cameron-Trivedi overdispersion test (1990),
and Vuong's non-nested model comparison test (1989).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from scipy.special import gammaln

from superglm._roundoff import UNIT_ROUNDOFF
from superglm.solvers.dispersion import (
    PRIOR_WEIGHTS,
    dispersion_likelihood_size,
    model_weight_semantics,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.model import SuperGLM


@dataclass(frozen=True)
class ZeroInflationResult:
    """Result from :func:`superglm.zero_inflation_index`."""

    observed_zeros: float
    expected_zeros: float
    zero_inflation_index: float
    ratio: float


@dataclass(frozen=True)
class ScoreTestZIResult:
    """Result from :func:`superglm.score_test_zi`."""

    statistic: float
    p_value: float
    direction: str


@dataclass(frozen=True)
class DispersionTestResult:
    """Result from :func:`superglm.dispersion_test`."""

    statistic: float
    p_value: float
    alpha_hat: float
    alternative: str


@dataclass(frozen=True)
class VuongTestResult:
    """Result from :func:`superglm.vuong_test`."""

    statistic: float
    p_value: float
    preferred: str
    correction: str
    mean_diff: float
    omega: float


# ── Private helpers ──────────────────────────────────────────────

_VUONG_A = "the Vuong test's log-likelihood of model_a"
_VUONG_B = "the Vuong test's log-likelihood of model_b"


def _get_mu(model: SuperGLM, X, y, offset=None, *, sample_weight=None, what: str) -> NDArray:
    """``predict``'s means for ``what``, refused where a weighted row has none.

    ``base.predict_priced``: a test statistic read from a row with no mean
    would be NaN, or a number from the clipped one.
    """
    from superglm.model.base import predict_priced

    return np.asarray(predict_priced(model, X, offset, sample_weight, what), dtype=float)


def _check_family(model: SuperGLM, allowed: set[str], func_name: str) -> str:
    """Check that the model's family is in the allowed set."""
    from superglm.distributions import NegativeBinomial, Poisson

    family = getattr(model, "_distribution")
    if isinstance(family, Poisson):
        name = "Poisson"
    elif isinstance(family, NegativeBinomial):
        name = "NB2"
    else:
        name = type(family).__name__

    if name not in allowed:
        raise ValueError(f"{func_name} requires family in {sorted(allowed)}, got {name}.")
    return name


def _frequency_weights(sample_weight, n: int) -> tuple[NDArray, float]:
    """Validate replication weights and return their likelihood size."""
    from superglm._utils import contract_weights

    weights = contract_weights(sample_weight, n)
    likelihood_size = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(likelihood_size):
        raise ValueError("sample_weight must have a finite sum")
    return weights, likelihood_size


def _log_likelihood_spread(ll_a: NDArray, ll_b: NDArray, measure: NDArray) -> float:
    """Each model's mean absolute deviation of its per-row log-likelihood, summed.

    ``measure`` weights the rows as the Vuong moments do; a row it gives no
    weight does not enter.
    """
    carried = measure > 0.0
    mass = measure[carried]
    spread = 0.0
    for ll in (ll_a[carried], ll_b[carried]):
        centre = float(np.sum(mass * ll)) / float(np.sum(mass))
        spread += float(np.sum(mass * np.abs(ll - centre))) / float(np.sum(mass))
    return spread


def _per_obs_ll_poisson(y: NDArray, mu: NDArray) -> NDArray:
    """Per-observation Poisson log-likelihood."""
    return y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)


def _per_obs_ll_nb2(y: NDArray, mu: NDArray, theta: float) -> NDArray:
    """Per-observation NB2 log-likelihood, the family's own stable density."""
    from superglm.distributions import _negative_binomial_log_density

    return _negative_binomial_log_density(y, mu, theta)


def _per_obs_ll_gaussian(y: NDArray, mu: NDArray, phi: float) -> NDArray:
    """Per-observation Gaussian log-likelihood."""
    phi_safe = max(phi, 1e-300)
    return -0.5 * (np.log(2 * np.pi * phi_safe) + (y - mu) ** 2 / phi_safe)


def _per_obs_ll_gamma(y: NDArray, mu: NDArray, phi: float) -> NDArray:
    """Per-observation Gamma log-likelihood."""
    k = 1.0 / phi
    return k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)


def _per_obs_ll_binomial(y: NDArray, mu: NDArray) -> NDArray:
    """Per-observation Bernoulli log-likelihood."""
    mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
    return y * np.log(mu_safe) + (1 - y) * np.log(1 - mu_safe)


def _per_obs_ll_tweedie(
    y: NDArray,
    mu: NDArray,
    phi: float,
    p: float,
    sample_weight=None,
) -> NDArray:
    """Per-observation Tweedie log-density at dispersion ``phi / w_i``."""
    from superglm._tweedie import tweedie_logpdf

    return tweedie_logpdf(y, mu, phi, p, weights=sample_weight)


def _per_obs_log_likelihood(
    model: SuperGLM,
    X,
    y: NDArray,
    offset=None,
    *,
    sample_weight=None,
    measure=None,
    what: str = "the log-likelihood",
) -> NDArray:
    """Compute per-observation log-likelihood for any supported family.

    With ``sample_weight`` omitted the result is the unweighted row log
    density, which is what the replication contract wants: the caller weights
    the rows itself.  With a prior weight supplied, the weight is part of each
    row's density -- through ``phi / w_i``, or through the equivalent scaling
    of the family's own parameters -- and the rows come back already carrying
    it.  ``measure`` weighs the rows as the caller reads them (``sample_weight``
    when omitted): a row it gives weight has to have a mean, and a row it gives
    none is not evaluated and reads 0, whatever its mean or response, so it
    cannot reach the caller's sums as an overflow or a NaN.
    """
    from superglm.model.base import score_rows

    measure = sample_weight if measure is None else measure
    mu = _get_mu(model, X, y, offset, sample_weight=measure, what=what)
    y = np.asarray(y, dtype=float)
    # a binomial row past the solver's clip is read from the likelihood at its own predictor
    mu, held = score_rows(model, X, offset, y, measure, mu)
    carried = None if measure is None else np.asarray(measure, dtype=np.float64) > 0.0
    if carried is not None and not np.all(carried):
        # Each row's density is its own, so the carried rows' are the same evaluated alone.
        density = np.zeros(y.shape, dtype=np.float64)
        density[carried] = _row_log_densities(
            model,
            y[carried],
            np.asarray(mu)[carried],
            None if sample_weight is None else np.asarray(sample_weight)[carried],
        )
    else:
        density = _row_log_densities(model, y, mu, sample_weight)
    if held is None:
        return density
    # Read the weighted log density directly before deviance can overflow.
    mass = np.ones(y.shape) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    exact = held.weighted_log_likelihood(mass)
    if carried is not None:
        exact = np.where(carried[held.index], exact, 0.0)
    return held.patched(density, exact)


def _row_log_densities(model: SuperGLM, y: NDArray, mu: NDArray, sample_weight) -> NDArray:
    """``_per_obs_log_likelihood``'s per-row log densities of ``y`` at the means ``mu``."""
    from superglm._weight_units import likelihood_scale
    from superglm.distributions import (
        Binomial,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
        prior_weight_log_density,
    )

    family = getattr(model, "_distribution")
    # Per-row log-likelihoods are densities: they read the likelihood scale.
    result = likelihood_scale(model.result)
    if getattr(result, "weight_exponent", 0) and getattr(result, "solver_phi", None) is not None:
        from superglm._weight_units import unit_prior_log_density

        weights = (
            np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        density = unit_prior_log_density(family, y, mu, weights, result)
        if density is not None:
            return density

    if sample_weight is not None and not isinstance(family, Tweedie):
        density = prior_weight_log_density(
            family,
            y,
            mu,
            np.asarray(sample_weight, dtype=np.float64),
            float(result.phi),
        )
        if density is None:
            raise ValueError(
                f"Unsupported family for prior-weight densities: {type(family).__name__}"
            )
        return density

    if isinstance(family, Poisson):
        return _per_obs_ll_poisson(y, mu)
    elif isinstance(family, NegativeBinomial):
        return _per_obs_ll_nb2(y, mu, family.theta)
    elif isinstance(family, Gaussian):
        return _per_obs_ll_gaussian(y, mu, result.phi)
    elif isinstance(family, Gamma):
        return _per_obs_ll_gamma(y, mu, result.phi)
    elif isinstance(family, Binomial):
        return _per_obs_ll_binomial(y, mu)
    elif isinstance(family, Tweedie):
        return _per_obs_ll_tweedie(
            y,
            mu,
            result.phi,
            family.p,
            sample_weight=sample_weight,
        )
    else:
        raise ValueError(f"Unsupported family: {type(family).__name__}")


# ── Public functions ─────────────────────────────────────────────


def zero_inflation_index(
    y,
    mu,
    sample_weight=None,
    *,
    family: str = "poisson",
    theta: float | None = None,
) -> ZeroInflationResult:
    """Descriptive zero-inflation index.

    Computes the ratio of observed to expected zeros under the assumed family.
    ``sample_weight`` is a replication weight here: both counts and the index
    denominator use the corresponding replicated-row measure.

    Parameters
    ----------
    y : array-like
        Observed response values.
    mu : array-like
        Fitted mean values.
    sample_weight : array-like or None
        Optional nonnegative replication weights. Integer weights are
        equivalent to literal row replication.
    family : str
        Family assumption: ``"poisson"`` or ``"nb2"``.
    theta : float or None
        NB2 theta parameter (required if ``family="nb2"``).

    Returns
    -------
    ZeroInflationResult
    """
    from superglm.model.input_validation import _finite_vector

    y_raw = np.asarray(y)
    if y_raw.ndim != 1 or y_raw.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    n = len(y_raw)
    y = _finite_vector("y", y_raw, n)
    mu = _finite_vector("mu", mu, n)
    if np.any(mu < 0.0):
        raise ValueError("mu must be nonnegative")
    weights, likelihood_size = _frequency_weights(sample_weight, n)

    family_lower = family.lower()
    if family_lower not in {"poisson", "nb2"}:
        raise ValueError(
            f"zero_inflation_index requires family 'poisson' or 'nb2', got {family!r}."
        )

    observed_zeros = float(np.sum(weights * (y == 0)))

    if family_lower == "poisson":
        p_zero = np.exp(-mu)
    else:  # nb2
        if theta is None:
            raise ValueError("theta must be provided for NB2 family.")
        theta = float(theta)
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError("theta must be finite and strictly positive for NB2 family.")
        p_zero = (theta / (theta + mu)) ** theta

    expected_zeros = float(np.sum(weights * p_zero))
    zi_index = (observed_zeros - expected_zeros) / likelihood_size
    ratio = observed_zeros / expected_zeros if expected_zeros > 0 else float("inf")

    return ZeroInflationResult(
        observed_zeros=observed_zeros,
        expected_zeros=expected_zeros,
        zero_inflation_index=zi_index,
        ratio=ratio,
    )


#: Below this mean ``_log_zi_information`` sums a series instead of subtracting.
_ZI_SERIES_BELOW = (25200.0 * UNIT_ROUNDOFF) ** (1.0 / 6.0)


def _log_zi_information(mu: NDArray) -> NDArray:
    """Compute ``log(exp(mu) - 1 - mu)`` for ``mu >= 0`` (``-inf`` at zero).

    For ``mu >= _ZI_SERIES_BELOW`` (``(25200 u)**(1/6)``, about 0.012),
    ``mu + log(-expm1(-mu) - mu * exp(-mu))`` never overflows; at small ``mu`` it subtracts two
    terms near ``mu`` differing by ``mu**2 / 2``, with relative error about ``10 u / mu``.  Below
    that threshold, ``2 * log(mu)`` plus the log of the series through ``mu**4`` takes over
    (avoiding ``mu**2`` underflow); its first omitted term is ``mu**5 / 2520`` of the value.  The
    two errors meet near ``1e-13``, the most either adds to the log.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        series = 2.0 * np.log(mu) + np.log(
            0.5 + mu * (1.0 / 6.0 + mu * (1.0 / 24.0 + mu * (1.0 / 120.0 + mu / 720.0)))
        )
        closed = mu + np.log(-np.expm1(-mu) - mu * np.exp(-mu))
    return np.where(mu < _ZI_SERIES_BELOW, series, closed)


def score_test_zi(
    model: SuperGLM,
    X,
    y,
    sample_weight=None,
    offset=None,
) -> ScoreTestZIResult:
    """Score test for zero inflation in a fitted Poisson model.

    Tests H0, that every count is Poisson with its fitted mean ``mu``, against a zero-inflated
    Poisson where a constant share ``p`` of rows is zero whatever its mean.  Because ``p`` may be
    negative, zero is not on its boundary and the test is two-sided: too few zeros reject as well
    as too many.  ``p_value`` is the upper tail of ``statistic`` under chi-square with one degree
    of freedom; ``direction`` is the sign of the score ``S`` (``"inflated"`` when ``S > 0``,
    ``"deflated"`` otherwise).

    A rejection says the fitted Poisson model does not account for the zeros, not that a
    zero-inflated model should replace it; a wrong mean or other overdispersion makes it reject.

    **When the result can be trusted**

    - **Many coefficients or a flexible model.**  Errors in ``mu`` shift the expected score without
      entering ``V``, so a model with many coefficients for its rows (such as a factor with many
      levels) rejects too often: toward ``"deflated"`` on its own fitted rows, and more often
      toward ``"inflated"`` on other rows.
    - **Rows the model was not fitted to.**  The rows should be the ones the model was fitted to.
      On unseen rows, errors in ``mu`` push the expected score upward whichever way they go and
      the excess rejection rate grows with the number of rows tested; a rejection on held-out rows
      from a flexible model is therefore not evidence of zero inflation.
    - **A misspecified or over-smoothed mean.**  The mean must be right.  A missing term, or one
      smoothed or shrunk further than the data support, leaves counts more spread than their
      fitted means, and the test rejects for that as it does for any other overdispersion.
    - **Large fitted means.**  The chi-square reference is a large-sample one.  Each row weighs
      in proportion to ``exp(mu)``, so a few rows with large means can carry the variance and make
      the test conservative: in simulation it kept its rate with means up to about 4, and rejected
      0.2% of null samples with means from 0.3 to 12.

    Penalised fits (smooth terms, ``fit_reml``, a selection penalty), offsets and other links use
    the same statistic, subject to the same accuracy requirements on ``mu`` (see *Notes*).

    Parameters
    ----------
    model : SuperGLM
        A fitted Poisson model.
    X : pandas or eager Polars DataFrame
        Design matrix.
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights.  Currently unsupported.
    offset : array-like or None
        Optional offset.  It enters through ``mu`` only.

    Returns
    -------
    ScoreTestZIResult
        ``statistic`` is infinite and ``p_value`` is 0 when the statistic exceeds float64's limit:
        a zero observed at a fitted mean above about 355 plus half the largest fitted mean.

    Raises
    ------
    NoPredictedMeanError
        If the model gives a row no predicted mean: no mean exists to read the statistic from.
    ValueError
        If ``model`` is not Poisson, or the statistic is undefined (a predicted mean is not
        finite, or every predicted mean is zero).
    NotImplementedError
        If ``sample_weight`` is provided.

    Notes
    -----
    The statistic is the squared score for ``p`` at ``p = 0`` over the score's variance::

        S = sum((1{y = 0} - p0) / p0 + (y - mu)),    p0 = exp(-mu)
        V = sum((1 - p0) / p0 - mu)
        statistic = S**2 / V

    With a log link this is van den Broek's (1995) score test in its regression form (Jansakul and
    Hinde, 2002, with a constant inflation probability; Deng and Paul, 2005).  Their variance is
    ``sum((1 - p0) / p0) - mu' D (D' diag(mu) D)^-1 D' mu`` for the design ``D`` of the mean;
    with an intercept in ``D`` the subtracted term is ``sum(mu)``, and at a maximum-likelihood fit
    ``sum(y - mu) = 0``, leaving ``S`` and ``V``.

    The fit enters through ``mu`` alone, so ``S`` and ``V`` take the same form for a penalised fit:
    the penalty leaves the intercept free, the one direction the variance correction removes, so
    the chi-square reference holds as far as the penalised means are accurate.  They also take the
    same form for another link, where ``p`` is the inflation probability with ``mu`` held as each
    row's mean; this is the usual zero-inflated Poisson whenever rescaling every mean by one factor
    stays inside the model (a log link with an intercept, or a power link without an offset).

    **Fitted-mean error.**  On an unseen row, an error ``d`` in the fitted mean raises the expected
    score by ``exp(d) - 1 - d``, about ``d**2 / 2``, whichever way ``d`` goes; ``y - mu`` in ``S``
    removes the first-order effect of ``d`` and not this one.  On a fitted row, whose own count
    pulled its fitted mean toward it, the expected score falls by about as much.  ``V`` allows for
    neither.  Summed over the rows passed, the shift's size is about half the summed squared error
    of ``mu`` (for an unpenalised fit on its own rows, half its number of coefficients times a
    typical mean); the stated rejection rate holds only while that is small beside ``sqrt(V)``.

    References
    ----------
    Deng, D. and Paul, S. R. (2005). Score tests for zero-inflation and over-dispersion in
    generalized linear models. *Statistica Sinica* 15, 257-276.

    Hsu, W.-W., Todem, D., Mawella, N. R., Kim, K. and Rosenkranz, R. R. (2020). A robust score
    test of homogeneity for zero-inflated count data. *Statistical Methods in Medical Research*
    29(12), 3653-3665.

    Baillo, A., Berrendero, J. R. and Carcamo, J. (2008). Tests for zero-inflation and
    overdispersion. arXiv:0809.4205.

    van den Broek, J. (1995), *Biometrics* 51, 738-743, and Jansakul, N. and Hinde, J. P. (2002),
    *Computational Statistics and Data Analysis* 40, 75-96, are the original papers; they were not
    read directly, and the statement here follows the three references above and the documentation
    of statsmodels' ``test_poisson_zeroinflation_jh``.
    """
    import scipy.spatial
    from scipy import stats

    _check_family(model, {"Poisson"}, "score_test_zi")

    if sample_weight is not None:
        raise NotImplementedError(
            "sample_weight is not yet supported for score_test_zi. "
            "Pass sample_weight=None or omit it."
        )

    y = np.asarray(y, dtype=float)
    mu = _get_mu(model, X, y, offset, what="the zero-inflation score test")

    # exp(mu) - 1 - mu is each row's variance and the score of a row with y = 0;
    # a row with y > 0 scores y - 1 - mu.  The largest of them, exp(top), divides
    # the variance and its square root divides the score, which keeps both sums
    # in range and leaves score / sqrt(variance) as it is.
    log_information = _log_zi_information(mu)
    top = float(np.max(log_information, initial=-np.inf))
    if not np.isfinite(top):
        raise ValueError(
            "score_test_zi has no statistic for these rows: a predicted mean is not "
            "finite, or every predicted mean is 0 and the Poisson model expects nothing "
            "but zeros. Check the rows and the offset passed."
        )
    with np.errstate(over="ignore"):
        zero_rows = np.exp(log_information - 0.5 * top)
        other_rows = (y - 1.0 - mu) / np.exp(0.5 * top)
    score = float(np.sum(np.where(y == 0, zero_rows, other_rows)))
    variance = float(np.sum(np.exp(log_information - top)))

    # Squared after it is standardised, so that the square overflows only where
    # the statistic does.
    standardised = score / float(np.sqrt(variance))
    statistic = standardised * standardised
    return ScoreTestZIResult(
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df=1)),
        direction="inflated" if score > 0.0 else "deflated",
    )


def dispersion_test(
    model: SuperGLM,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    alternative: str = "greater",
) -> DispersionTestResult:
    """Cameron & Trivedi (1990) regression-based overdispersion test.

    Tests H0: Var(Y) = mu (equidispersion) vs H1: Var(Y) = mu + alpha*mu^2.

    Parameters
    ----------
    model : SuperGLM
        A fitted Poisson model.
    X : pandas or eager Polars DataFrame
        Design matrix.
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights. Currently unsupported.
    offset : array-like or None
        Optional offset.
    alternative : str
        One of ``"greater"`` (overdispersion), ``"less"`` (underdispersion),
        or ``"two-sided"``.

    Returns
    -------
    DispersionTestResult

    Raises
    ------
    NoPredictedMeanError
        Where the model gives a row no predicted mean: the statistic would be
        read from a mean that does not exist.
    """
    import scipy.spatial
    from scipy import stats

    _check_family(model, {"Poisson"}, "dispersion_test")

    if sample_weight is not None:
        raise NotImplementedError(
            "sample_weight is not yet supported for dispersion_test. "
            "Pass sample_weight=None or omit it."
        )

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(
            f"alternative must be 'greater', 'less', or 'two-sided', got {alternative!r}."
        )

    y = np.asarray(y, dtype=float)
    mu = _get_mu(model, X, y, offset, what="the dispersion test")
    n = len(y)

    # Auxiliary regression: ((y - mu)^2 - y) / mu  on  mu  (no intercept)
    dep_var = ((y - mu) ** 2 - y) / np.maximum(mu, 1e-10)
    reg_var = mu

    # OLS no intercept: alpha_hat = sum(x*y) / sum(x^2)
    alpha_hat = float(np.sum(reg_var * dep_var) / np.sum(reg_var**2))

    # Standard error of the slope
    resid = dep_var - alpha_hat * reg_var
    s2 = float(np.sum(resid**2) / (n - 1))
    se = np.sqrt(s2 / np.sum(reg_var**2))

    t_stat = alpha_hat / se if se > 0 else 0.0

    if alternative == "greater":
        p_value = float(stats.t.sf(t_stat, df=n - 1))
    elif alternative == "less":
        p_value = float(stats.t.cdf(t_stat, df=n - 1))
    else:  # two-sided
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))

    return DispersionTestResult(
        statistic=float(t_stat),
        p_value=p_value,
        alpha_hat=alpha_hat,
        alternative=alternative,
    )


def vuong_test(
    model_a: SuperGLM,
    model_b: SuperGLM,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    correction: str = "aic",
) -> VuongTestResult:
    """Vuong (1989) likelihood-ratio test for non-nested model comparison.

    Compares two models' per-observation log-likelihoods. Under H0 (models
    are equally close to the true DGP), the test statistic is asymptotically
    N(0,1).

    Parameters
    ----------
    model_a, model_b : SuperGLM
        Two fitted SuperGLM models.
    X : pandas or eager Polars DataFrame
        Design matrix (same for both models).
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights, read under the contract both models declare.
        Under ``"frequency"`` integer values are exactly equivalent to row
        replication; under ``"prior"`` they enter each density through
        ``phi_i = phi / w_i`` or the equivalent scaling of the family's own
        parameters. A weighted comparison of two models that declare different
        contracts is rejected, because a single vector cannot represent both.
    offset : array-like or None
        Optional offset.
    correction : str
        Complexity correction: ``"none"``, ``"aic"``, or ``"bic"``.

    Returns
    -------
    VuongTestResult

    Raises
    ------
    NoPredictedMeanError
        Where either model gives a row of positive weight no predicted mean:
        its log-likelihood, and so the comparison, does not exist.
    """
    import scipy.spatial
    from scipy import stats

    if correction not in {"none", "aic", "bic"}:
        raise ValueError(f"correction must be 'none', 'aic', or 'bic', got {correction!r}.")

    from superglm._utils import contract_weights
    from superglm.distributions import prior_weights_in_normalizer
    from superglm.model.input_validation import _finite_vector

    y_raw = np.asarray(y)
    if y_raw.ndim != 1 or y_raw.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    n = len(y_raw)
    y_arr = _finite_vector("y", y_raw, n)

    # Both models must read the weights the same way, a declared contract: models
    # of different families may be compared with weights when they agree on it,
    # models of one family may not when they differ.
    contract_a = model_weight_semantics(model_a)
    contract_b = model_weight_semantics(model_b)
    mixed_weight_contracts = contract_a != contract_b
    if mixed_weight_contracts and sample_weight is not None:
        raise ValueError(
            "sample_weight has incompatible semantics for a weighted Vuong "
            f"comparison: model_a declares {contract_a!r} and model_b {contract_b!r}"
        )

    if contract_a == PRIOR_WEIGHTS and contract_b == PRIOR_WEIGHTS:
        # Either model being Tweedie makes the weights strictly positive.
        family = getattr(model_a, "_distribution")
        if not prior_weights_in_normalizer(family):
            family = getattr(model_b, "_distribution")
        prior_weights = (
            None
            if sample_weight is None
            else contract_weights(sample_weight, n, family=family, weight_semantics=PRIOR_WEIGHTS)
        )
        evaluated_weights = np.ones(n, dtype=np.float64) if prior_weights is None else prior_weights
        ll_a = _per_obs_log_likelihood(
            model_a,
            X,
            y_arr,
            offset,
            sample_weight=prior_weights,
            what=_VUONG_A,
        )
        ll_b = _per_obs_log_likelihood(
            model_b,
            X,
            y_arr,
            offset,
            sample_weight=prior_weights,
            what=_VUONG_B,
        )
        # A zero prior weight is an unobserved row with density exactly 0; it must
        # leave the mean and variance, as ``likelihood_size`` below counts only
        # carried rows. ``prior_weights`` is None without a sample_weight, when
        # every row carries weight.
        d = ll_a - ll_b
        measure = np.ones(n) if prior_weights is None else (prior_weights > 0.0).astype(float)
        if prior_weights is not None:
            d = d[prior_weights > 0.0]
        likelihood_size = (
            float(n)
            if prior_weights is None
            else dispersion_likelihood_size(prior_weights, weight_semantics=PRIOR_WEIGHTS)
        )
        # Sized before the moments: one carried row would make ``np.std(ddof=1)``
        # warn ahead of the intended ValueError.
        if likelihood_size <= 1.0:
            raise ValueError("Vuong test requires an effective likelihood size greater than one")
        mean_m = float(np.mean(d))
        omega = float(np.std(d, ddof=1))
    elif mixed_weight_contracts:
        # Reachable only with sample_weight None (what makes a mixed-contract
        # comparison legal); unit weights still define a lattice for a counting
        # model, so this arm is checked like the others.
        evaluated_weights = np.ones(n, dtype=np.float64)
        measure = evaluated_weights
        ll_a = _per_obs_log_likelihood(model_a, X, y_arr, offset, what=_VUONG_A)
        ll_b = _per_obs_log_likelihood(model_b, X, y_arr, offset, what=_VUONG_B)
        d = ll_a - ll_b
        mean_m = float(np.mean(d))
        omega = float(np.std(d, ddof=1))
        likelihood_size = float(n)
    else:
        frequency_weights, likelihood_size = _frequency_weights(sample_weight, n)
        evaluated_weights = measure = frequency_weights
        ll_a = _per_obs_log_likelihood(
            model_a, X, y_arr, offset, measure=frequency_weights, what=_VUONG_A
        )
        ll_b = _per_obs_log_likelihood(
            model_b, X, y_arr, offset, measure=frequency_weights, what=_VUONG_B
        )
        d = ll_a - ll_b
        mean_m = float(np.sum(frequency_weights * d) / likelihood_size)
        if likelihood_size <= 1.0:
            raise ValueError("Vuong test requires an effective likelihood size greater than one")
        omega = float(
            np.sqrt(np.sum(frequency_weights * (d - mean_m) ** 2) / (likelihood_size - 1.0))
        )

    if likelihood_size <= 1.0:
        raise ValueError("Vuong test requires an effective likelihood size greater than one")

    # A response that cannot be honoured under the declared contract (off the
    # counting lattice, a non-unit prior binomial weight, a fractional
    # replication count) would give a finite pseudo-log-likelihood and an
    # apparently valid result. Checked once, after every arm has resolved its
    # weights, against both families, since either may be the counting one.
    from superglm.model.input_validation import check_weight_contract

    for compared in (model_a, model_b):
        check_weight_contract(
            y_arr,
            evaluated_weights,
            compared._distribution,
            model_weight_semantics(compared),
        )

    # Model complexities
    p_a = model_a.result.effective_df
    p_b = model_b.result.effective_df

    # Apply correction
    if correction == "aic":
        mean_m -= (p_a - p_b) / likelihood_size
    elif correction == "bic":
        mean_m -= (p_a - p_b) * np.log(likelihood_size) / (2 * likelihood_size)

    # Essentially identical: the per-row differences are small beside the
    # per-row spread of the log-likelihoods themselves, over the rows and
    # weights the moments read.  The spread is what a change of the response's
    # units leaves alone (it adds a constant to every row) and what weights
    # multiply.
    if omega <= 1e-10 * _log_likelihood_spread(ll_a, ll_b, measure):
        return VuongTestResult(
            statistic=0.0,
            p_value=1.0,
            preferred="indistinguishable",
            correction=correction,
            mean_diff=mean_m,
            omega=omega,
        )

    v_stat = float(np.sqrt(likelihood_size) * mean_m / omega)
    p_value = float(2.0 * stats.norm.sf(abs(v_stat)))

    if p_value < 0.05:
        preferred = "model_a" if v_stat > 0 else "model_b"
    else:
        preferred = "indistinguishable"

    return VuongTestResult(
        statistic=v_stat,
        p_value=p_value,
        preferred=preferred,
        correction=correction,
        mean_diff=mean_m,
        omega=omega,
    )
