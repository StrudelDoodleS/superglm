"""Model adequacy tests for count and overdispersion diagnostics.

Provides zero-inflation detection, score tests for zero-inflation
(van den Broek 1995), Cameron-Trivedi overdispersion test (1990),
and Vuong's non-nested model comparison test (1989).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.special import gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.model import SuperGLM


@dataclass(frozen=True)
class ZeroInflationResult:
    """Result from :func:`zero_inflation_index`."""

    observed_zeros: int
    expected_zeros: float
    zero_inflation_index: float
    ratio: float


@dataclass(frozen=True)
class ScoreTestZIResult:
    """Result from :func:`score_test_zi`."""

    statistic: float
    p_value: float
    direction: str


@dataclass(frozen=True)
class DispersionTestResult:
    """Result from :func:`dispersion_test`."""

    statistic: float
    p_value: float
    alpha_hat: float
    alternative: str


@dataclass(frozen=True)
class VuongTestResult:
    """Result from :func:`vuong_test`."""

    statistic: float
    p_value: float
    preferred: str
    correction: str
    mean_diff: float
    omega: float


# ── Private helpers ──────────────────────────────────────────────


def _get_mu(model: SuperGLM, X, y, offset=None) -> NDArray:
    """Get predicted mu from a fitted model."""
    return np.asarray(model.predict(X, offset=offset), dtype=float)


def _check_family(model: SuperGLM, allowed: set[str], func_name: str) -> str:
    """Check that the model's family is in the allowed set."""
    from superglm.distributions import NegativeBinomial, Poisson

    family = model._distribution
    if isinstance(family, Poisson):
        name = "Poisson"
    elif isinstance(family, NegativeBinomial):
        name = "NB2"
    else:
        name = type(family).__name__

    if name not in allowed:
        raise ValueError(f"{func_name} requires family in {sorted(allowed)}, got {name}.")
    return name


def _per_obs_ll_poisson(y: NDArray, mu: NDArray) -> NDArray:
    """Per-observation Poisson log-likelihood."""
    return y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)


def _per_obs_ll_nb2(y: NDArray, mu: NDArray, theta: float) -> NDArray:
    """Per-observation NB2 log-likelihood."""
    return (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1)
        + theta * np.log(theta / (mu + theta))
        + y * np.log(mu / (mu + theta))
    )


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


def _per_obs_ll_tweedie(y: NDArray, mu: NDArray, phi: float, p: float, weights: NDArray) -> NDArray:
    """Per-observation Tweedie log-likelihood."""
    from superglm.tweedie_profile import tweedie_logpdf

    return tweedie_logpdf(y, mu, phi, p, weights=weights)


def _per_obs_log_likelihood(
    model: SuperGLM, X, y: NDArray, offset=None, sample_weight=None
) -> NDArray:
    """Compute per-observation log-likelihood for any supported family."""
    from superglm.distributions import (
        Binomial,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )

    mu = _get_mu(model, X, y, offset)
    y = np.asarray(y, dtype=float)
    family = model._distribution
    result = model.result

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
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        return _per_obs_ll_tweedie(y, mu, result.phi, family.p, w)
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

    Parameters
    ----------
    y : array-like
        Observed response values.
    mu : array-like
        Fitted mean values.
    sample_weight : array-like or None
        Optional observation weights (used for counting).
    family : str
        Family assumption: ``"poisson"`` or ``"nb2"``.
    theta : float or None
        NB2 theta parameter (required if ``family="nb2"``).

    Returns
    -------
    ZeroInflationResult
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = len(y)

    family_lower = family.lower()
    if family_lower not in {"poisson", "nb2"}:
        raise ValueError(
            f"zero_inflation_index requires family 'poisson' or 'nb2', got {family!r}."
        )

    observed_zeros = int(np.sum(y == 0))

    if family_lower == "poisson":
        p_zero = np.exp(-mu)
    else:  # nb2
        if theta is None:
            raise ValueError("theta must be provided for NB2 family.")
        p_zero = (theta / (theta + mu)) ** theta

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        expected_zeros = float(np.sum(w * p_zero))
    else:
        expected_zeros = float(np.sum(p_zero))

    zi_index = (observed_zeros - expected_zeros) / n
    ratio = observed_zeros / expected_zeros if expected_zeros > 0 else float("inf")

    return ZeroInflationResult(
        observed_zeros=observed_zeros,
        expected_zeros=expected_zeros,
        zero_inflation_index=zi_index,
        ratio=ratio,
    )


def score_test_zi(
    model: SuperGLM,
    X,
    y,
    sample_weight=None,
    offset=None,
) -> ScoreTestZIResult:
    """Van den Broek (1995) score test for zero-inflation.

    Tests H0: no zero-inflation vs H1: zero-inflation, assuming
    a Poisson baseline model.

    Parameters
    ----------
    model : SuperGLM
        A fitted Poisson model.
    X : pd.DataFrame
        Design matrix.
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights.
    offset : array-like or None
        Optional offset.

    Returns
    -------
    ScoreTestZIResult
    """
    _check_family(model, {"Poisson"}, "score_test_zi")

    if sample_weight is not None:
        raise NotImplementedError(
            "sample_weight is not yet supported for score_test_zi. "
            "Pass sample_weight=None or omit it."
        )

    y = np.asarray(y, dtype=float)
    mu = _get_mu(model, X, y, offset)

    p0 = np.exp(-mu)  # P(Y=0|mu) under Poisson
    indicator = (y == 0).astype(float)

    # Score statistic: van den Broek (1995), eq. (3)
    # Numerator: [sum(I(y=0) - p0)]^2
    numer = (np.sum(indicator - p0)) ** 2

    # Denominator: Var under H0 from van den Broek (1995)
    # Full formula: Var = sum(q_i(1-q_i)) - (sum(mu_i*q_i))^2 / sum(mu_i)
    # where q_i = exp(-mu_i)
    correction = (np.sum(mu * p0)) ** 2 / np.sum(mu)
    denom = np.sum(p0 * (1 - p0)) - correction

    if denom <= 0:
        denom = 1e-10  # safety

    statistic = float(numer / denom)
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
    direction = "inflated" if np.sum(indicator) > np.sum(p0) else "deflated"

    return ScoreTestZIResult(
        statistic=statistic,
        p_value=p_value,
        direction=direction,
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
    X : pd.DataFrame
        Design matrix.
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights.
    offset : array-like or None
        Optional offset.
    alternative : str
        One of ``"greater"`` (overdispersion), ``"less"`` (underdispersion),
        or ``"two-sided"``.

    Returns
    -------
    DispersionTestResult
    """
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
    mu = _get_mu(model, X, y, offset)
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
        p_value = float(1.0 - stats.t.cdf(t_stat, df=n - 1))
    elif alternative == "less":
        p_value = float(stats.t.cdf(t_stat, df=n - 1))
    else:  # two-sided
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))

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
    X : pd.DataFrame
        Design matrix (same for both models).
    y : array-like
        Observed response values.
    sample_weight : array-like or None
        Observation weights.
    offset : array-like or None
        Optional offset.
    correction : str
        Complexity correction: ``"none"``, ``"aic"``, or ``"bic"``.

    Returns
    -------
    VuongTestResult
    """
    if correction not in {"none", "aic", "bic"}:
        raise ValueError(f"correction must be 'none', 'aic', or 'bic', got {correction!r}.")

    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)

    ll_a = _per_obs_log_likelihood(model_a, X, y_arr, offset, sample_weight)
    ll_b = _per_obs_log_likelihood(model_b, X, y_arr, offset, sample_weight)

    # Per-observation log-likelihood differences
    m = ll_a - ll_b

    # Compute (weighted) mean and std of per-obs LL differences
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        w_sum = w.sum()
        mean_m = float(np.sum(w * m) / w_sum)
        # Weighted variance with Bessel-like correction
        var_m = float(np.sum(w * (m - mean_m) ** 2) / w_sum)
        omega = float(np.sqrt(var_m))
    else:
        mean_m = float(np.mean(m))
        omega = float(np.std(m, ddof=1))

    # Model complexities
    p_a = model_a.result.effective_df
    p_b = model_b.result.effective_df

    # Apply correction
    if correction == "aic":
        mean_m -= (p_a - p_b) / n
    elif correction == "bic":
        mean_m -= (p_a - p_b) * np.log(n) / (2 * n)

    if omega < 1e-10:
        # Models are essentially identical
        return VuongTestResult(
            statistic=0.0,
            p_value=1.0,
            preferred="indistinguishable",
            correction=correction,
            mean_diff=mean_m,
            omega=omega,
        )

    v_stat = float(np.sqrt(n) * mean_m / omega)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(v_stat))))

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
