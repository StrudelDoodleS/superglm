"""Tweedie profile likelihood — estimate p from data.

For p ∈ (1, 2), the Tweedie distribution is a compound Poisson-Gamma.
This module provides multiple search strategies for estimating the power
parameter p via profile likelihood, plus exact Wright-Bessel logpdf
evaluation and compound Poisson-Gamma simulation.

Search methods:

- ``"brent"`` (default): bounded scalar optimisation via scipy.
- ``"grid"``: exhaustive grid search over p.
- ``"grid_refine"``: coarse grid + local Brent refinement.
- ``"profile_opt"``: general-purpose optimizer (L-BFGS-B, Powell) on
  logit-transformed p.

References
----------
- Dunn & Smyth (2005): Series evaluation of Tweedie EDMs
- Yang, Qian & Zou (2018): Insurance Premium Prediction via Tweedie CPMs
- Jørgensen (1997): Theory of dispersion models
"""

from __future__ import annotations

import logging
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, wright_bessel

from superglm._utils import _validate_strict_prior_weights
from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.penalties.base import penalty_has_targets
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compound Poisson-Gamma simulation
# ---------------------------------------------------------------------------


def generate_tweedie_cpg(
    n: int,
    mu: float | NDArray,
    phi: float,
    p: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate Tweedie(mu, phi, p) via compound Poisson-Gamma.

    Parameters
    ----------
    n : int
        Number of samples.
    mu : float or array of shape (n,)
        Mean parameter.
    phi : float
        Dispersion parameter (>0).
    p : float
        Power parameter, must be in (1, 2).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    y : ndarray of shape (n,)
        Simulated responses (non-negative, with exact zeros).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()

    # CPG parameters (Jørgensen 1997)
    lam = np.power(mu, 2 - p) / ((2 - p) * phi)  # Poisson rate
    alpha = (2 - p) / (p - 1)  # Gamma shape per claim
    beta = phi * (p - 1) * np.power(mu, p - 1)  # Gamma scale per claim

    # Vectorised: draw N ~ Poisson(lam), then Y|N ~ Gamma(alpha*N, beta)
    N = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    pos = N > 0
    if np.any(pos):
        # Gamma additive property: sum of N iid Gamma(alpha, beta) = Gamma(N*alpha, beta)
        y[pos] = rng.gamma(alpha * N[pos], scale=beta[pos])

    return y


# ---------------------------------------------------------------------------
# Tweedie log-pdf
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TweedieLogpdfDiagnostics:
    """Diagnostics for the Tweedie log-density evaluator."""

    n_positive: int = 0
    n_saddlepoint: int = 0

    @property
    def saddlepoint_fraction(self) -> float:
        if self.n_positive == 0:
            return 0.0
        return float(self.n_saddlepoint) / float(self.n_positive)


@dataclass(frozen=True)
class _PreparedTweedieDensity:
    """Phi-independent terms for repeated Tweedie density evaluations."""

    y: NDArray
    mu: NDArray
    weights: NDArray
    p: float
    t_arg_limit: float
    log_t_arg_limit: float
    a: float
    zero_mask: NDArray
    positive_mask: NDArray
    positive_indices: NDArray
    zero_rate_numerator: NDArray
    log_weight: NDArray
    positive_log_y: NDArray
    positive_canonical_c: NDArray
    positive_saddlepoint_deviance: NDArray
    positive_saddlepoint_log_base: NDArray
    positive_log_t_phi_independent: NDArray


@dataclass(frozen=True)
class _TweedieDensityEvaluation:
    """One Tweedie density evaluation, optionally including its NLL score."""

    logpdf: NDArray
    log_phi_score: NDArray | None
    diagnostics: _TweedieLogpdfDiagnostics
    score_valid: bool


def _readonly_copy(values: NDArray, *, dtype: Any | None = None) -> NDArray:
    """Return an owning, read-only array for an immutable evaluation record."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_tweedie_inputs(
    y: NDArray,
    mu: NDArray,
    p: float,
    weights: NDArray | None,
) -> tuple[NDArray, NDArray, float, NDArray | None]:
    """Validate and convert common Tweedie density and dispersion inputs."""
    y_raw = np.asarray(y)
    if np.iscomplexobj(y_raw):
        raise ValueError("y must be finite and non-negative")
    mu_raw = np.asarray(mu)
    if np.iscomplexobj(mu_raw):
        raise ValueError("mu must be finite and strictly positive")
    y_arr = np.asarray(y_raw, dtype=np.float64)
    mu_arr = np.asarray(mu_raw, dtype=np.float64)
    if y_arr.ndim != 1 or mu_arr.ndim != 1 or y_arr.shape != mu_arr.shape or y_arr.size == 0:
        raise ValueError("y and mu must be one-dimensional arrays with the same non-empty shape")
    if not np.all(np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and non-negative")
    if not np.all(np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")

    p_arr = np.asarray(p)
    if p_arr.ndim != 0:
        raise ValueError("p must be finite and in the open interval (1, 2)")
    try:
        p_float = float(p_arr)
    except (TypeError, ValueError) as exc:
        raise ValueError("p must be finite and in the open interval (1, 2)") from exc
    if not np.isfinite(p_float) or not 1.0 < p_float < 2.0:
        raise ValueError("p must be finite and in the open interval (1, 2)")

    validated_weights = None
    if weights is not None:
        validated_weights = _validate_strict_prior_weights(weights, len(y_arr))
    return y_arr, mu_arr, p_float, validated_weights


def _validate_tweedie_phi(phi: float) -> float:
    """Validate and convert a scalar Tweedie dispersion parameter."""
    phi_arr = np.asarray(phi)
    if phi_arr.ndim != 0:
        raise ValueError("phi must be finite and strictly positive")
    try:
        phi_float = float(phi_arr)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi must be finite and strictly positive") from exc
    if not np.isfinite(phi_float) or phi_float <= 0.0:
        raise ValueError("phi must be finite and strictly positive")
    return phi_float


_TWEEDIE_DEVIANCE_SERIES_THRESHOLD = 1e-3
_TWEEDIE_DEVIANCE_SERIES_TERMS = 8


def _tweedie_positive_unit_deviance(y: NDArray, mu: NDArray, p: float) -> NDArray:
    """Compute positive-response unit deviance without close-mean cancellation."""
    y_array, mu_array = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        delta = (y_array - mu_array) / mu_array
    extreme_positive = np.isposinf(delta)
    near = np.abs(delta) <= _TWEEDIE_DEVIANCE_SERIES_THRESHOLD
    g = np.empty_like(delta)

    if np.any(near):
        delta_near = delta[near]
        term = np.full_like(delta_near, 0.5)
        series = term.copy()
        # The recurrence adds k=1,...,8 from the integral expansion. At the
        # 1e-3 threshold, the first omitted term is O(1e-27), below binary64
        # rounding even for the largest coefficient over 1 < p < 2.
        for k in range(_TWEEDIE_DEVIANCE_SERIES_TERMS):
            term *= -delta_near * (p + k) / (k + 3.0)
            series += term
        g[near] = delta_near**2 * series

    regular = ~near & ~extreme_positive & (delta > -1.0)
    if np.any(regular):
        delta_regular = delta[regular]
        with np.errstate(all="ignore"):
            log_ratio = np.log1p(delta_regular)
            first = (1.0 + delta_regular) * np.expm1((1.0 - p) * log_ratio) / (1.0 - p)
            second = np.expm1((2.0 - p) * log_ratio) / (2.0 - p)
        g[regular] = first - second

    # For a positive y many orders below mu, delta can round to exactly -1.
    # Recover log(y / mu) from the original values; this branch is far from
    # the cancellation region, so the power-scale formula is well-conditioned.
    rounded_to_minus_one = ~near & ~regular & ~extreme_positive
    if np.any(rounded_to_minus_one):
        with np.errstate(all="ignore"):
            log_ratio = np.log(y_array[rounded_to_minus_one]) - np.log(
                mu_array[rounded_to_minus_one]
            )
            ratio = np.exp(log_ratio)
            ratio_two_minus_p = np.exp((2.0 - p) * log_ratio)
            first = (ratio_two_minus_p - ratio) / (1.0 - p)
            second = (ratio_two_minus_p - 1.0) / (2.0 - p)
        g[rounded_to_minus_one] = first - second

    deviance = np.empty_like(delta)
    ordinary_ratio = ~extreme_positive
    with np.errstate(all="ignore"):
        deviance[ordinary_ratio] = (
            2.0 * np.power(mu_array[ordinary_ratio], 2.0 - p) * g[ordinary_ratio]
        )

    if np.any(extreme_positive):
        # For y >> mu, factor the expanded positive half-deviance by
        # A = y * mu**(1-p) / (p-1). The remaining ratios use mu/y and
        # cannot create inf-inf or 0*inf. -expm1 keeps 1 - B/A accurate
        # when p is itself very close to one.
        with np.errstate(all="ignore"):
            log_y = np.log(y_array[extreme_positive])
            log_mu = np.log(mu_array[extreme_positive])
            log_mu_over_y = log_mu - log_y
            log_b_over_a = (p - 1.0) * log_mu_over_y - np.log(2.0 - p)
            log_c_over_a = np.log(p - 1.0) - np.log(2.0 - p) + log_mu_over_y
            correction = -np.expm1(log_b_over_a) + np.exp(log_c_over_a)
            log_deviance = (
                np.log(2.0) + log_y + (1.0 - p) * log_mu - np.log(p - 1.0) + np.log(correction)
            )
            deviance[extreme_positive] = np.exp(log_deviance)
    negative_roundoff = np.isfinite(deviance) & (deviance < 0.0)
    if np.any(negative_roundoff):
        deviance = deviance.copy()
        deviance[negative_roundoff] = 0.0
    return deviance


def _prepare_tweedie_density(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    t_arg_limit: float = 1e14,
) -> _PreparedTweedieDensity:
    """Prepare fixed terms for repeated density evaluations over ``phi``."""
    y, mu, p, validated_weights = _validate_tweedie_inputs(y, mu, p, weights)
    if validated_weights is None:
        weights_array = np.ones(len(y), dtype=np.float64)
    else:
        weights_array = validated_weights

    t_arg_limit_array = np.asarray(t_arg_limit)
    if t_arg_limit_array.ndim != 0:
        raise ValueError("t_arg_limit must be a scalar")
    try:
        t_arg_limit_float = float(t_arg_limit_array)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_arg_limit must be a scalar") from exc
    if t_arg_limit_float > 0.0:
        log_t_arg_limit = float(np.log(t_arg_limit_float))
    else:
        # A non-positive limit intentionally forces every positive term onto
        # the saddlepoint branch. NaN retains the old all-saddle behavior.
        log_t_arg_limit = np.nan if np.isnan(t_arg_limit_float) else -np.inf

    zero_mask = y == 0.0
    positive_mask = ~zero_mask
    positive_indices = np.flatnonzero(positive_mask)
    y_positive = y[positive_mask]

    with np.errstate(all="ignore"):
        mu_one_minus_p = np.power(mu, 1.0 - p)
        mu_two_minus_p = np.power(mu, 2.0 - p)
        zero_rate_numerator = mu_two_minus_p / (2.0 - p)
        log_weight = np.log(weights_array)
        positive_log_y = np.log(y_positive)

        positive_canonical_c = y_positive * mu_one_minus_p[positive_mask] / (
            1.0 - p
        ) - mu_two_minus_p[positive_mask] / (2.0 - p)
        positive_saddlepoint_deviance = _tweedie_positive_unit_deviance(
            y_positive,
            mu[positive_mask],
            p,
        )
        positive_saddlepoint_log_base = np.log(2.0 * np.pi) + p * positive_log_y

        a = (2.0 - p) / (p - 1.0)
        alpha = -a
        positive_log_t_phi_independent = (
            alpha * (np.log(p - 1.0) - positive_log_y)
            - np.log(2.0 - p)
            + (a + 1.0) * log_weight[positive_mask]
        )

    return _PreparedTweedieDensity(
        y=_readonly_copy(y, dtype=np.float64),
        mu=_readonly_copy(mu, dtype=np.float64),
        weights=_readonly_copy(weights_array, dtype=np.float64),
        p=p,
        t_arg_limit=t_arg_limit_float,
        log_t_arg_limit=log_t_arg_limit,
        a=a,
        zero_mask=_readonly_copy(zero_mask, dtype=np.bool_),
        positive_mask=_readonly_copy(positive_mask, dtype=np.bool_),
        positive_indices=_readonly_copy(positive_indices, dtype=np.intp),
        zero_rate_numerator=_readonly_copy(zero_rate_numerator, dtype=np.float64),
        log_weight=_readonly_copy(log_weight, dtype=np.float64),
        positive_log_y=_readonly_copy(positive_log_y, dtype=np.float64),
        positive_canonical_c=_readonly_copy(positive_canonical_c, dtype=np.float64),
        positive_saddlepoint_deviance=_readonly_copy(
            positive_saddlepoint_deviance,
            dtype=np.float64,
        ),
        positive_saddlepoint_log_base=_readonly_copy(
            positive_saddlepoint_log_base,
            dtype=np.float64,
        ),
        positive_log_t_phi_independent=_readonly_copy(
            positive_log_t_phi_independent,
            dtype=np.float64,
        ),
    )


def _evaluate_tweedie_density(
    prepared: _PreparedTweedieDensity,
    phi: float,
    *,
    compute_score: bool = False,
) -> _TweedieDensityEvaluation:
    """Evaluate a prepared density and its optional mean-NLL log-phi score."""
    phi = _validate_tweedie_phi(phi)
    log_phi = float(np.log(phi))
    inverse_phi_eff = prepared.weights / phi
    log_phi_eff = log_phi - prepared.log_weight

    logpdf = np.empty(len(prepared.y), dtype=np.float64)
    log_phi_score = np.empty(len(prepared.y), dtype=np.float64) if compute_score else None

    zero = prepared.zero_mask
    if np.any(zero):
        zero_logpdf = -prepared.zero_rate_numerator[zero] * inverse_phi_eff[zero]
        logpdf[zero] = zero_logpdf
        if log_phi_score is not None:
            log_phi_score[zero] = zero_logpdf

    n_saddlepoint = 0
    positive = prepared.positive_mask
    if np.any(positive):
        inverse_phi_positive = inverse_phi_eff[positive]
        log_phi_eff_positive = log_phi_eff[positive]
        log_t = prepared.positive_log_t_phi_independent - (prepared.a + 1.0) * log_phi

        # Select the numerical branch in log space. In particular, do not
        # clip log(t): clipping can move an observation across t_arg_limit.
        try_exact = log_t < prepared.log_t_arg_limit
        exact = np.zeros(len(log_t), dtype=np.bool_)
        t_positive = np.full(len(log_t), np.nan, dtype=np.float64)
        wright_a_plus_one = np.full(len(log_t), np.nan, dtype=np.float64)
        positive_logpdf = np.empty(len(log_t), dtype=np.float64)

        if np.any(try_exact):
            try_exact_indices = np.flatnonzero(try_exact)
            with np.errstate(all="ignore"):
                t_exact = np.exp(log_t[try_exact])
                wright_recurrence = wright_bessel(
                    prepared.a,
                    prepared.a + 1.0,
                    t_exact,
                )
                candidate_logpdf = (
                    np.log(prepared.a)
                    + log_t[try_exact]
                    + np.log(wright_recurrence)
                    - prepared.positive_log_y[try_exact]
                    + prepared.positive_canonical_c[try_exact] * inverse_phi_positive[try_exact]
                )

            density_valid = (
                np.isfinite(wright_recurrence)
                & (wright_recurrence > 0.0)
                & np.isfinite(candidate_logpdf)
            )
            valid_indices = try_exact_indices[density_valid]
            exact[valid_indices] = True
            t_positive[valid_indices] = t_exact[density_valid]
            wright_a_plus_one[valid_indices] = wright_recurrence[density_valid]
            positive_logpdf[valid_indices] = candidate_logpdf[density_valid]

        # This assignment is intentionally independent of np.any(exact): an
        # all-invalid Wright batch still needs every saddlepoint fallback.
        saddlepoint = ~exact
        n_saddlepoint = int(np.count_nonzero(saddlepoint))
        if np.any(saddlepoint):
            positive_logpdf[saddlepoint] = (
                -0.5
                * (
                    prepared.positive_saddlepoint_log_base[saddlepoint]
                    + log_phi_eff_positive[saddlepoint]
                )
                - 0.5
                * prepared.positive_saddlepoint_deviance[saddlepoint]
                * inverse_phi_positive[saddlepoint]
            )
        logpdf[positive] = positive_logpdf

        if log_phi_score is not None:
            positive_score = np.full(len(log_t), np.nan, dtype=np.float64)
            if np.any(saddlepoint):
                positive_score[saddlepoint] = (
                    0.5
                    - 0.5
                    * prepared.positive_saddlepoint_deviance[saddlepoint]
                    * inverse_phi_positive[saddlepoint]
                )

            if np.any(exact):
                with np.errstate(all="ignore"):
                    wright_a = wright_bessel(
                        prepared.a,
                        prepared.a,
                        t_positive[exact],
                    )
                    ratio = wright_a / (prepared.a * wright_a_plus_one[exact])

                # Wright evaluations accumulate parameter-scaled rounding as
                # a grows near p=1. Cap the allowance so a materially sub-unit
                # ratio still invalidates the analytic score.
                ratio_tolerance = min(
                    1e-10,
                    64.0 * np.finfo(np.float64).eps * max(1.0, prepared.a),
                )
                ratio_valid = (
                    np.isfinite(wright_a)
                    & (wright_a > 0.0)
                    & np.isfinite(ratio)
                    & (ratio >= 1.0 - ratio_tolerance)
                )
                exact_indices = np.flatnonzero(exact)
                valid_score_indices = exact_indices[ratio_valid]
                if np.any(ratio_valid):
                    stable_ratio = np.maximum(ratio[ratio_valid], 1.0)
                    exact_score = (
                        stable_ratio / (prepared.p - 1.0)
                        + prepared.positive_canonical_c[valid_score_indices]
                        * inverse_phi_positive[valid_score_indices]
                    )
                    finite_score = np.isfinite(exact_score)
                    positive_score[valid_score_indices[finite_score]] = exact_score[finite_score]

            log_phi_score[positive] = positive_score

    diagnostics = _TweedieLogpdfDiagnostics(
        n_positive=int(np.count_nonzero(positive)),
        n_saddlepoint=n_saddlepoint,
    )
    score_valid = log_phi_score is not None and bool(np.all(np.isfinite(log_phi_score)))
    return _TweedieDensityEvaluation(
        logpdf=_readonly_copy(logpdf, dtype=np.float64),
        log_phi_score=(
            _readonly_copy(log_phi_score, dtype=np.float64) if log_phi_score is not None else None
        ),
        diagnostics=diagnostics,
        score_valid=score_valid,
    )


def _tweedie_logpdf_impl(
    y: NDArray,
    mu: NDArray,
    phi: float,
    p: float,
    *,
    weights: NDArray | None = None,
    t_arg_limit: float = 1e14,
) -> tuple[NDArray, _TweedieLogpdfDiagnostics]:
    """Compatibility wrapper over the prepared Tweedie density evaluator."""
    prepared = _prepare_tweedie_density(
        y,
        mu,
        p,
        weights=weights,
        t_arg_limit=t_arg_limit,
    )
    evaluation = _evaluate_tweedie_density(prepared, phi)
    return evaluation.logpdf.copy(), evaluation.diagnostics


def tweedie_logpdf(
    y: NDArray,
    mu: NDArray,
    phi: float,
    p: float,
    *,
    weights: NDArray | None = None,
    t_arg_limit: float = 1e14,
) -> NDArray:
    """Exact Tweedie log-density with saddlepoint fallback.

    Parameters
    ----------
    y, mu : arrays of shape (n,)
        Observations and fitted means.
    phi : float
        Dispersion parameter.
    p : float
        Power parameter in (1, 2).
    weights : array of shape (n,), optional
        Observation weights (e.g. sample_weight). Effective phi = phi / w.
    t_arg_limit : float
        Switch to saddlepoint when wright_bessel argument t >= this.
        A high default keeps the exact Wright-Bessel branch active deeper
        into the low-p region, where the saddlepoint can be noticeably
        biased.

    Returns
    -------
    logpdf : ndarray of shape (n,)
    """
    logpdf, _ = _tweedie_logpdf_impl(
        y,
        mu,
        phi,
        p,
        weights=weights,
        t_arg_limit=t_arg_limit,
    )
    return logpdf


def _saddlepoint(y: NDArray, mu: NDArray, phi: NDArray, p: float) -> NDArray:
    """Saddlepoint approximation to the Tweedie log-density."""
    y_safe = np.maximum(y, 1e-300)
    deviance = _tweedie_positive_unit_deviance(y_safe, mu, p)
    return -0.5 * (np.log(2.0 * np.pi) + np.log(phi) + p * np.log(y_safe)) - deviance / (2.0 * phi)


# ---------------------------------------------------------------------------
# Dispersion estimation
# ---------------------------------------------------------------------------


def estimate_phi(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    df_resid: float | None = None,
) -> float:
    """Weighted Pearson estimate of dispersion parameter phi.

    phi_hat = sum(w * (y - mu)^2 / mu^p) / denom

    Under the prior-weight convention used here,
    ``Var(Y_i) = phi * mu_i^p / w_i``. Therefore
    ``E[w_i * (Y_i - mu_i)^2 / mu_i^p] = phi`` for each observation, and the
    natural denominator is the residual observation count rather than the sum
    of weights.

    where denom = df_resid if provided, else n_obs (i.e. no df correction).

    For the prior-weight convention used by ``sample_weight`` in SuperGLM,
    callers should pass the residual observation count
    ``df_resid = n_obs - edf``.
    """
    y, mu, p, weights = _validate_tweedie_inputs(y, mu, p, weights)
    mu_safe = np.maximum(mu, 1e-10)
    variance_fn = np.power(mu_safe, p)
    pearson = (y - mu) ** 2 / variance_fn

    denom = float(df_resid if df_resid is not None else len(y))
    if weights is not None:
        numer = float(np.sum(weights * pearson))
    else:
        numer = float(np.sum(pearson))
    return numer / denom


def _profile_phi(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    df_resid: float | None = None,
    phi_method: str = "pearson",
) -> tuple[float, float]:
    """Profile out phi and return ``(phi_hat, mean_nll)`` for fixed ``(mu, p)``."""
    if phi_method == "pearson":
        phi_hat = max(estimate_phi(y, mu, p, weights=weights, df_resid=df_resid), 1e-10)
        ll = tweedie_logpdf(y, mu, phi_hat, p, weights=weights)
        return phi_hat, float(-np.mean(ll))

    if phi_method != "mle":
        raise ValueError(
            f"phi_method={phi_method!r} is not valid, expected one of ['mle', 'pearson']"
        )

    phi_init = max(estimate_phi(y, mu, p, weights=weights, df_resid=df_resid), 1e-10)
    log_phi_init = float(np.log(phi_init))
    log_lo = max(np.log(1e-12), log_phi_init - 8.0)
    log_hi = min(np.log(1e12), log_phi_init + 8.0)

    def objective(log_phi: float) -> float:
        ll = tweedie_logpdf(y, mu, float(np.exp(log_phi)), p, weights=weights)
        return float(-np.mean(ll))

    opt = minimize_scalar(
        objective,
        bounds=(log_lo, log_hi),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 50},
    )
    phi_hat = float(np.exp(opt.x))
    return phi_hat, float(opt.fun)


# ---------------------------------------------------------------------------
# Profile likelihood result
# ---------------------------------------------------------------------------

_TRACE_COLUMNS = [
    "step",
    "p",
    "phi",
    "nll",
    "n_iter",
    "fit_converged",
    "source",
    "fit_trace",
    "fit_trace_kind",
]
_SADDLEPOINT_NOTE_THRESHOLD = 0.10
_SADDLEPOINT_WARN_THRESHOLD = 0.25


@dataclass
class TweedieProfileResult:
    """Result of Tweedie power parameter estimation.

    Attributes
    ----------
    p_hat : float
        Estimated power parameter.
    phi_hat : float
        Estimated dispersion at p_hat.
    nll : float
        Mean negative log-likelihood at (p_hat, phi_hat).
    n_evaluations : int
        Total number of profile evaluations.
    converged : bool
        Whether the search converged.
    method : str
        Search method used (``"brent"``, ``"grid"``, etc.).
    phi_method : str
        How phi was profiled (``"pearson"`` or ``"mle"``).
    search_trace : DataFrame
        Per-evaluation record with columns:
        ``step, p, phi, nll, n_iter, fit_converged, source``.
    saddlepoint_fraction : float
        Fraction of positive density evaluations that used the saddlepoint
        approximation at the final ``(p_hat, phi_hat)``.
    """

    p_hat: float
    phi_hat: float
    nll: float
    n_evaluations: int
    converged: bool
    method: str
    phi_method: str
    search_trace: pd.DataFrame
    saddlepoint_fraction: float = 0.0
    n_saddlepoint: int = 0
    n_positive: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def cache(self) -> dict[float, float]:
        """Deprecated: use ``search_trace`` instead.

        Returns a dict mapping ``p → nll`` reconstructed from the search
        trace for backward compatibility.
        """
        import warnings as _w

        _w.warn(
            "TweedieProfileResult.cache is deprecated; use .search_trace instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dict(zip(self.search_trace["p"], self.search_trace["nll"]))

    # Stored for CI/plot
    _objective: Any = field(default=None, repr=False)
    _ll_scale: float = field(default=0.0, repr=False)
    _ci_cache: dict[float, tuple[float, float]] = field(default_factory=dict, repr=False)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Profile likelihood confidence interval for Tweedie p.

        Requires that the result was produced by ``estimate_tweedie_p``.
        Results are cached so repeated calls (e.g. from summary()) are free.
        """
        if alpha in self._ci_cache:
            return self._ci_cache[alpha]
        if self._objective is None:
            raise RuntimeError(
                "Profile CI requires the objective function. Use "
                "estimate_tweedie_p() to produce this result."
            )
        result = profile_ci_p(self._objective, self.p_hat, self.nll, self._ll_scale, alpha=alpha)
        self._ci_cache[alpha] = result
        return result

    def profile_plot(
        self,
        *,
        alpha: float = 0.05,
        n_points: int = 50,
        ax=None,
    ):
        """Profile deviance plot for Tweedie power parameter p.

        Evaluates the profile objective on a dense grid for the curve, and
        overlays the search evaluation points from ``search_trace``.

        Parameters
        ----------
        alpha : float
            Significance level for CI (default 0.05).
        n_points : int
            Number of grid points for the smooth curve.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._objective is None:
            raise RuntimeError(
                "Profile plot requires the objective function. Use "
                "estimate_tweedie_p() to produce this result."
            )

        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        ci_lo, ci_hi = self.ci(alpha=alpha)

        # Grid must cover CI and all search evaluation points
        trace_ps = self.search_trace["p"].values
        margin = 0.2 * (ci_hi - ci_lo)
        grid_lo = max(1.01, ci_lo - margin)
        grid_hi = min(1.99, ci_hi + margin)
        if len(trace_ps) > 0:
            grid_lo = min(grid_lo, float(trace_ps.min()) - 0.005)
            grid_hi = max(grid_hi, float(trace_ps.max()) + 0.005)
            grid_lo = max(1.01, grid_lo)
            grid_hi = min(1.99, grid_hi)
        p_grid = np.linspace(grid_lo, grid_hi, n_points)

        nll_values = np.array([self._objective(p) for p in p_grid])
        deviance = 2.0 * self._ll_scale * (nll_values - self.nll)

        cutoff = chi2.ppf(1.0 - alpha, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        ax.plot(p_grid, deviance, color="steelblue", linewidth=1.5)

        # Mark search evaluation points from trace
        if len(trace_ps) > 0:
            trace_nll = self.search_trace["nll"].values
            trace_dev = 2.0 * self._ll_scale * (trace_nll - self.nll)
            ax.scatter(
                trace_ps,
                trace_dev,
                color="darkorange",
                s=35,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
                label=f"Evaluations ({len(trace_ps)})",
            )

        ax.axhline(
            cutoff,
            linestyle="--",
            color="grey",
            linewidth=0.8,
            label=f"{100 * (1 - alpha):.0f}% cutoff",
        )
        ax.axvline(
            self.p_hat,
            linestyle=":",
            color="black",
            linewidth=0.8,
            label=f"MLE = {self.p_hat:.3f}",
        )
        ax.fill_betweenx(
            [0, cutoff],
            ci_lo,
            ci_hi,
            alpha=0.10,
            color="firebrick",
            label=f"{100 * (1 - alpha):.0f}% CI: [{ci_lo:.3f}, {ci_hi:.3f}]",
        )

        ax.set_xlabel("p")
        ax.set_ylabel("Profile deviance")
        ax.set_title("Tweedie p profile likelihood")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        return fig


def _build_saddlepoint_messages(p: float, diagnostics: _TweedieLogpdfDiagnostics) -> list[str]:
    """Build thresholded saddlepoint diagnostics for the final profile result."""
    frac = diagnostics.saddlepoint_fraction
    if diagnostics.n_positive == 0 or frac < _SADDLEPOINT_NOTE_THRESHOLD:
        return []

    message = (
        "Saddlepoint approximation used for "
        f"{diagnostics.n_saddlepoint}/{diagnostics.n_positive} positive Tweedie "
        f"density terms ({frac:.0%}) at p={p:.3f}; profile likelihood may be "
        "approximation-sensitive near the lower p bound."
    )
    if frac >= _SADDLEPOINT_WARN_THRESHOLD:
        _warnings.warn(message, UserWarning, stacklevel=3)
    return [message]


def _fit_iteration_trace(iteration_log) -> list[dict[str, float | int]]:
    """Convert solver diagnostics into a compact frontend learning curve."""
    if not iteration_log:
        return []
    return [
        {
            "iteration": int(row.iteration),
            "loss": float(row.deviance),
        }
        for row in iteration_log
    ]


def _reml_iteration_trace(reml_result) -> list[dict[str, float | int]]:
    """Convert REML objective history into the same compact curve shape."""
    history = getattr(reml_result, "objective_history", None)
    if not history:
        return []
    return [
        {
            "iteration": int(i),
            "loss": float(value),
        }
        for i, value in enumerate(history)
        if np.isfinite(value)
    ]


# ---------------------------------------------------------------------------
# Profile context — fit path
# ---------------------------------------------------------------------------


@dataclass
class _ProfileContext:
    """One-time setup + per-evaluation logic for profile p estimation (fit path).

    All search methods share this context. It manages the design matrix,
    solver dispatch, warm starts, and trace accumulation.
    """

    y_arr: NDArray
    w_arr: NDArray
    offset_arr: NDArray
    dm: Any  # DesignMatrix
    groups: list
    link: Any
    penalty: Any
    use_direct: bool
    lambda2: Any
    direct_solve: str
    phi_method: str
    verbose: bool
    ll_scale: float
    trace_callback: Any = field(default=None, repr=False)
    trace_iterations: bool = False

    # Mutable warm-start state
    warm_beta: NDArray | None = field(default=None, repr=False)
    warm_intercept: float | None = field(default=None, repr=False)
    last_p_eval: float | None = field(default=None, repr=False)
    last_mu: NDArray | None = field(default=None, repr=False)
    last_edf: float | None = field(default=None, repr=False)

    # Trace accumulator
    trace_rows: list[dict] = field(default_factory=list, repr=False)
    _nll_cache: dict[float, float] = field(default_factory=dict, repr=False)
    n_evals: int = field(default=0, repr=False)

    def evaluate(self, p: float, source: str = "") -> float:
        """Fit at p, profile phi, record trace row, return mean NLL."""
        import time as _time

        from superglm.distributions import Tweedie

        key = round(p, 12)
        if key in self._nll_cache:
            return self._nll_cache[key]

        _t0 = _time.perf_counter()
        dist = Tweedie(p)
        if self.use_direct:
            result, _ = fit_irls_direct(
                X=self.dm,
                y=self.y_arr,
                weights=self.w_arr,
                family=dist,
                link=self.link,
                groups=self.groups,
                lambda2=self.lambda2,
                offset=self.offset_arr,
                beta_init=self.warm_beta,
                intercept_init=self.warm_intercept,
                direct_solve=self.direct_solve,
                record_diagnostics=self.trace_iterations,
            )
        else:
            result = fit_pirls(
                X=self.dm,
                y=self.y_arr,
                weights=self.w_arr,
                family=dist,
                link=self.link,
                groups=self.groups,
                penalty=self.penalty,
                offset=self.offset_arr,
                beta_init=self.warm_beta,
                intercept_init=self.warm_intercept,
                record_diagnostics=self.trace_iterations,
            )

        eta = stabilize_eta(
            self.dm.matvec(result.beta) + result.intercept + self.offset_arr, self.link
        )
        mu = clip_mu(self.link.inverse(eta), dist)
        df_resid = max(float(len(self.y_arr)) - float(result.effective_df), 1.0)

        phi, nll = _profile_phi(
            self.y_arr,
            mu,
            p,
            weights=self.w_arr,
            df_resid=df_resid,
            phi_method=self.phi_method,
        )

        # Update warm starts
        self.warm_beta = result.beta
        self.warm_intercept = result.intercept
        self.last_p_eval = p
        self.last_mu = mu
        self.last_edf = float(result.effective_df)

        # Record trace
        row = {
            "step": self.n_evals,
            "p": p,
            "phi": phi,
            "nll": nll,
            "n_iter": result.n_iter,
            "fit_converged": result.converged,
            "source": source,
            "fit_trace": _fit_iteration_trace(result.iteration_log),
            "fit_trace_kind": "weighted deviance" if self.trace_iterations else "",
        }
        self.trace_rows.append(row)
        if self.trace_callback is not None and source:
            self.trace_callback(dict(row))
        self._nll_cache[key] = nll
        self.n_evals += 1
        _elapsed = _time.perf_counter() - _t0

        logger.info(
            f"  estimate_p eval={self.n_evals:2d}  p={p:.4f}  phi={phi:.4f}  "
            f"nll={nll:.4f}  iters={result.n_iter}  {_elapsed:.2f}s"
        )
        if self.verbose:
            print(
                f"  p={p:.4f}  phi={phi:.4f}  nll={nll:.4f}  iters={result.n_iter}  {_elapsed:.2f}s"
            )

        return nll

    def finalize(self, p_hat: float, method: str, converged: bool) -> TweedieProfileResult:
        """Build result with final phi at p_hat and search_trace DataFrame."""
        p_hat = round(p_hat, 12)
        nll = self._nll_cache.get(p_hat, self.evaluate(p_hat, source="final"))

        # Get phi at p_hat
        if self.last_p_eval is not None and round(self.last_p_eval, 12) == p_hat:
            mu_final = self.last_mu
            edf_final = self.last_edf
        else:
            # One final fit at p_hat
            self.evaluate(p_hat, source="final")
            mu_final = self.last_mu
            edf_final = self.last_edf

        df_resid_final = max(float(len(self.y_arr)) - float(edf_final), 1.0)
        phi_hat, _ = _profile_phi(
            self.y_arr,
            mu_final,
            p_hat,
            weights=self.w_arr,
            df_resid=df_resid_final,
            phi_method=self.phi_method,
        )
        _, diagnostics = _tweedie_logpdf_impl(
            self.y_arr,
            mu_final,
            phi_hat,
            p_hat,
            weights=self.w_arr,
        )
        warnings_list = _build_saddlepoint_messages(p_hat, diagnostics)

        trace = pd.DataFrame(self.trace_rows, columns=_TRACE_COLUMNS)

        return TweedieProfileResult(
            p_hat=p_hat,
            phi_hat=phi_hat,
            nll=nll,
            n_evaluations=self.n_evals,
            converged=converged,
            method=method,
            phi_method=self.phi_method,
            search_trace=trace,
            saddlepoint_fraction=diagnostics.saddlepoint_fraction,
            n_saddlepoint=diagnostics.n_saddlepoint,
            n_positive=diagnostics.n_positive,
            warnings=warnings_list,
            _objective=self.evaluate,
            _ll_scale=self.ll_scale,
        )


def _build_profile_context(
    model,
    X,
    y,
    sample_weight,
    offset,
    phi_method: str,
    verbose: bool,
    trace_callback=None,
    trace_iterations: bool = False,
) -> _ProfileContext:
    """One-time setup: build design matrix, calibrate lambda, create context."""
    from superglm.distributions import Tweedie

    y_arr = np.asarray(y, dtype=np.float64)

    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, sample_weight)

    # Temporary p so _build_design_matrix can resolve the distribution.
    # The design matrix itself doesn't depend on p.
    saved_family = model.family
    model.family = Tweedie(p=1.5)
    try:
        y_arr, w_arr, offset_arr = model._build_design_matrix(X, y_arr, sample_weight, offset)
    finally:
        model.family = saved_family

    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = model._compute_lambda_max(y_arr, w_arr) * 0.1

    if offset_arr is None:
        offset_arr = np.zeros(len(y_arr))

    penalty = model.penalty
    groups = model._groups
    use_direct = penalty.lambda1 is not None and (
        penalty.lambda1 == 0 or not penalty_has_targets(penalty, groups)
    )

    return _ProfileContext(
        y_arr=y_arr,
        w_arr=w_arr,
        offset_arr=offset_arr,
        dm=model._dm,
        groups=groups,
        link=model._link,
        penalty=penalty,
        use_direct=use_direct,
        lambda2=model.lambda2,
        direct_solve=getattr(model, "_direct_solve", "auto"),
        phi_method=phi_method,
        verbose=verbose,
        ll_scale=float(len(y_arr)),
        trace_callback=trace_callback,
        trace_iterations=trace_iterations,
    )


# ---------------------------------------------------------------------------
# Profile context — REML path
# ---------------------------------------------------------------------------


@dataclass
class _ProfileContextREML:
    """Per-evaluation logic for profile p estimation (REML path).

    Each evaluation calls ``model.fit_reml()`` — no solver-level warm starts,
    but shares the same dispatch interface as ``_ProfileContext``.
    """

    model: Any
    X: Any
    y: NDArray
    sample_weight: Any
    offset: Any
    w_arr: NDArray
    phi_method: str
    verbose: bool
    ll_scale: float
    trace_callback: Any = field(default=None, repr=False)
    trace_iterations: bool = False

    # Mutable state
    last_p_eval: float | None = field(default=None, repr=False)
    last_mu: NDArray | None = field(default=None, repr=False)
    last_edf: float | None = field(default=None, repr=False)

    # Trace accumulator
    trace_rows: list[dict] = field(default_factory=list, repr=False)
    _nll_cache: dict[float, float] = field(default_factory=dict, repr=False)
    n_evals: int = field(default=0, repr=False)

    def evaluate(self, p: float, source: str = "") -> float:
        """Fit REML at p, profile phi, record trace row, return mean NLL."""
        import time as _time

        from superglm.distributions import Tweedie

        key = round(p, 12)
        if key in self._nll_cache:
            return self._nll_cache[key]

        _t0 = _time.perf_counter()
        self.model.family = Tweedie(p=p)
        self.model.fit_reml(self.X, self.y, sample_weight=self.sample_weight, offset=self.offset)

        mu = np.maximum(self.model.predict(self.X), 1e-10)
        df_resid = max(float(len(self.y)) - float(self.model.result.effective_df), 1.0)
        phi, nll = _profile_phi(
            self.y,
            mu,
            p,
            weights=self.w_arr,
            df_resid=df_resid,
            phi_method=self.phi_method,
        )

        self.last_p_eval = p
        self.last_mu = mu
        self.last_edf = float(self.model.result.effective_df)

        n_iter = (
            self.model._reml_result.n_reml_iter
            if hasattr(self.model, "_reml_result") and self.model._reml_result is not None
            else 0
        )

        row = {
            "step": self.n_evals,
            "p": p,
            "phi": phi,
            "nll": nll,
            "n_iter": n_iter,
            "fit_converged": self.model.result.converged,
            "source": source,
            "fit_trace": (
                _reml_iteration_trace(getattr(self.model, "_reml_result", None))
                if self.trace_iterations
                else []
            ),
            "fit_trace_kind": "REML objective" if self.trace_iterations else "",
        }
        self.trace_rows.append(row)
        if self.trace_callback is not None and source:
            self.trace_callback(dict(row))
        self._nll_cache[key] = nll
        self.n_evals += 1
        _elapsed = _time.perf_counter() - _t0

        logger.info(
            f"  estimate_p eval={self.n_evals:2d}  p={p:.4f}  phi={phi:.4f}  "
            f"nll={nll:.4f}  reml_iters={n_iter}  {_elapsed:.2f}s"
        )
        if self.verbose:
            print(
                f"  p={p:.4f}  phi={phi:.4f}  nll={nll:.4f}  reml_iters={n_iter}  {_elapsed:.2f}s"
            )

        return nll

    def finalize(self, p_hat: float, method: str, converged: bool) -> TweedieProfileResult:
        """Build result with final phi at p_hat and search_trace DataFrame."""
        p_hat = round(p_hat, 12)

        # Ensure we have mu at p_hat
        if self.last_p_eval is None or round(self.last_p_eval, 12) != p_hat:
            self.evaluate(p_hat, source="final")

        nll = self._nll_cache[p_hat]

        df_resid_final = max(float(len(self.y)) - float(self.last_edf), 1.0)
        phi_hat, _ = _profile_phi(
            self.y,
            self.last_mu,
            p_hat,
            weights=self.w_arr,
            df_resid=df_resid_final,
            phi_method=self.phi_method,
        )
        _, diagnostics = _tweedie_logpdf_impl(
            self.y,
            self.last_mu,
            phi_hat,
            p_hat,
            weights=self.w_arr,
        )
        warnings_list = _build_saddlepoint_messages(p_hat, diagnostics)

        trace = pd.DataFrame(self.trace_rows, columns=_TRACE_COLUMNS)

        return TweedieProfileResult(
            p_hat=p_hat,
            phi_hat=phi_hat,
            nll=nll,
            n_evaluations=self.n_evals,
            converged=converged,
            method=method,
            phi_method=self.phi_method,
            search_trace=trace,
            saddlepoint_fraction=diagnostics.saddlepoint_fraction,
            n_saddlepoint=diagnostics.n_saddlepoint,
            n_positive=diagnostics.n_positive,
            warnings=warnings_list,
            _objective=self.evaluate,
            _ll_scale=self.ll_scale,
        )


def _build_profile_context_reml(
    model,
    X,
    y,
    sample_weight,
    offset,
    phi_method: str,
    verbose: bool,
    trace_callback=None,
    trace_iterations: bool = False,
) -> _ProfileContextREML:
    """Build context for REML-based profile estimation."""
    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, sample_weight)

    # REML profile evaluations call fit_reml(), which rewrites the fitted model
    # state. Keep that mutation inside an isolated scratch model so result.ci()
    # and profile plots cannot leave the caller's model at a probe p.
    profile_model = model._clone_without_features(set())
    if getattr(model, "_last_fit_meta", None) is not None:
        profile_model._last_fit_meta = dict(model._last_fit_meta)

    y_np = np.asarray(y, dtype=np.float64)
    w_arr = (
        np.asarray(sample_weight, dtype=np.float64)
        if sample_weight is not None
        else np.ones(len(y_np))
    )
    return _ProfileContextREML(
        model=profile_model,
        X=X,
        y=y_np,
        sample_weight=sample_weight,
        offset=offset,
        w_arr=w_arr,
        phi_method=phi_method,
        verbose=verbose,
        ll_scale=float(len(y_np)),
        trace_callback=trace_callback,
        trace_iterations=trace_iterations,
    )


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------


def _search_brent(
    ctx: _ProfileContext | _ProfileContextREML,
    p_bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> TweedieProfileResult:
    """Bounded scalar Brent search over p."""
    result = minimize_scalar(
        lambda p: ctx.evaluate(p, source="brent"),
        bounds=p_bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    converged = result.success if hasattr(result, "success") else True
    return ctx.finalize(result.x, method="brent", converged=converged)


def _search_grid(
    ctx: _ProfileContext | _ProfileContextREML,
    p_bounds: tuple[float, float],
    n_grid: int,
    grid: NDArray | None,
) -> TweedieProfileResult:
    """Exhaustive grid search over p."""
    if grid is not None:
        p_grid = np.asarray(grid, dtype=np.float64)
    else:
        p_grid = np.linspace(p_bounds[0], p_bounds[1], n_grid)

    nll_values = np.array([ctx.evaluate(p, source="grid") for p in p_grid])
    best_idx = int(np.argmin(nll_values))
    p_hat = float(p_grid[best_idx])

    return ctx.finalize(p_hat, method="grid", converged=True)


def _search_grid_refine(
    ctx: _ProfileContext | _ProfileContextREML,
    p_bounds: tuple[float, float],
    n_grid_coarse: int,
    xatol: float,
    maxiter: int,
) -> TweedieProfileResult:
    """Coarse grid search + local Brent refinement."""
    # Stage 1: coarse grid
    p_coarse = np.linspace(p_bounds[0], p_bounds[1], n_grid_coarse)
    nll_coarse = np.array([ctx.evaluate(p, source="grid_coarse") for p in p_coarse])
    best_idx = int(np.argmin(nll_coarse))
    p_best = float(p_coarse[best_idx])

    # Stage 2: refine around best region
    step = (p_bounds[1] - p_bounds[0]) / max(n_grid_coarse - 1, 1)
    refine_lo = max(p_bounds[0], p_best - step)
    refine_hi = min(p_bounds[1], p_best + step)

    result = minimize_scalar(
        lambda p: ctx.evaluate(p, source="brent_refine"),
        bounds=(refine_lo, refine_hi),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    converged = result.success if hasattr(result, "success") else True
    return ctx.finalize(result.x, method="grid_refine", converged=converged)


def _search_profile_opt(
    ctx: _ProfileContext | _ProfileContextREML,
    p_bounds: tuple[float, float],
    optimizer: str,
    xatol: float,
    maxiter: int,
) -> TweedieProfileResult:
    """Optimizer-driven profile search with logit-transformed p."""
    _VALID_OPTIMIZERS = {"L-BFGS-B", "Powell"}
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer={optimizer!r} is not valid, expected one of {sorted(_VALID_OPTIMIZERS)}"
        )

    lo, hi = p_bounds

    def p_to_t(p: float) -> float:
        """Map p ∈ (lo, hi) → t ∈ ℝ via logit."""
        return float(logit((p - lo) / (hi - lo)))

    def t_to_p(t: float) -> float:
        """Map t ∈ ℝ → p ∈ (lo, hi) via expit."""
        return float(lo + (hi - lo) * expit(t))

    # 3-point initialization grid to pick starting point
    init_ps = [lo + 0.1 * (hi - lo), 0.5 * (lo + hi), hi - 0.1 * (hi - lo)]
    init_nlls = [ctx.evaluate(p, source="init") for p in init_ps]
    best_init = init_ps[int(np.argmin(init_nlls))]
    t0 = p_to_t(best_init)

    def objective(t_arr):
        t = float(t_arr[0]) if hasattr(t_arr, "__len__") else float(t_arr)
        p = t_to_p(t)
        return ctx.evaluate(p, source="optimizer")

    opts: dict[str, Any] = {"maxiter": maxiter}
    if optimizer == "L-BFGS-B":
        opts["ftol"] = 1e-8
        opts["gtol"] = 1e-6
    elif optimizer == "Powell":
        opts["ftol"] = 1e-8
        opts["xtol"] = xatol

    result = minimize(
        objective,
        x0=[t0],
        method=optimizer,
        options=opts,
    )

    p_hat = t_to_p(float(result.x[0]))
    converged = bool(result.success)

    return ctx.finalize(p_hat, method="profile_opt", converged=converged)


# ---------------------------------------------------------------------------
# Profile likelihood optimiser — public entry point
# ---------------------------------------------------------------------------


def estimate_tweedie_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    p_bounds: tuple[float, float] = (1.05, 1.95),
    xatol: float = 1e-3,
    maxiter: int = 30,
    verbose: bool = False,
    fit_mode: str = "fit",
    phi_method: str = "pearson",
    method: str = "brent",
    n_grid: int = 20,
    grid: NDArray | None = None,
    n_grid_coarse: int = 10,
    optimizer: str = "L-BFGS-B",
    trace_callback=None,
    trace_iterations: bool = False,
) -> TweedieProfileResult:
    """Estimate the Tweedie power parameter via profile likelihood.

    Builds the design matrix once and searches over candidate p values,
    fitting the GLM at each candidate with warm starts.

    Parameters
    ----------
    model : SuperGLM
        A configured but *unfitted* model with features already added.
        Must have a Tweedie family (e.g. ``families.tweedie(p=1.5)``).
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Frequency weights. Must be frequency weights, not variance weights.
    offset : array-like, optional
        Offset added to the linear predictor.
    p_bounds : tuple
        Bounds for p search, default (1.05, 1.95).
    xatol : float
        Tolerance for scalar optimisers (Brent).
    maxiter : int
        Maximum iterations for the optimiser.
    verbose : bool
        Print progress.
    fit_mode : {"fit", "fit_reml"}
        Fitting regime for each candidate p evaluation.
    phi_method : {"pearson", "mle"}
        How to profile out ``phi`` at each candidate ``p``.
    method : {"brent", "grid", "grid_refine", "profile_opt", "joint_ml", "integrated"}
        Search strategy. ``"brent"`` (default) uses bounded scalar
        optimisation. ``"grid"`` does exhaustive grid search.
        ``"grid_refine"`` does a coarse grid + local Brent refinement.
        ``"profile_opt"`` uses a general-purpose optimizer (L-BFGS-B or
        Powell) on logit-transformed p.
    n_grid : int
        Number of grid points for ``method="grid"`` (default 20).
    grid : array-like, optional
        Explicit p grid for ``method="grid"``. Overrides ``n_grid``.
    n_grid_coarse : int
        Number of coarse grid points for ``method="grid_refine"``
        (default 10).
    optimizer : str
        Optimizer backend for ``method="profile_opt"``. One of
        ``"L-BFGS-B"`` (default) or ``"Powell"``.
    trace_iterations : bool
        If True, include the nested fit learning curve for each candidate
        ``p`` evaluation in ``search_trace["fit_trace"]``.

    Returns
    -------
    TweedieProfileResult
    """
    from superglm.distributions import Tweedie

    y_arr = np.asarray(y)
    if sample_weight is not None:
        sample_weight = _validate_strict_prior_weights(sample_weight, len(y_arr))

    # Validate family
    family = model.family
    if not isinstance(family, Tweedie):
        raise ValueError(
            f"estimate_tweedie_p requires a Tweedie family, got {family!r}. "
            "Use families.tweedie(p=...) to create one."
        )

    _VALID_METHODS = {"brent", "grid", "grid_refine", "profile_opt", "joint_ml", "integrated"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method={method!r} is not valid, expected one of {sorted(_VALID_METHODS)}"
        )

    _VALID_FIT_MODES = {"fit", "fit_reml"}
    if fit_mode not in _VALID_FIT_MODES:
        raise ValueError(
            f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(_VALID_FIT_MODES)}"
        )
    _VALID_PHI_METHODS = {"pearson", "mle"}
    if phi_method not in _VALID_PHI_METHODS:
        raise ValueError(
            f"phi_method={phi_method!r} is not valid, expected one of {sorted(_VALID_PHI_METHODS)}"
        )

    if method in ("joint_ml", "integrated"):
        raise NotImplementedError(
            f"method={method!r} is not yet implemented. "
            f"Use one of: 'brent', 'grid', 'grid_refine', 'profile_opt'."
        )

    # Build context
    if fit_mode == "fit_reml":
        ctx = _build_profile_context_reml(
            model,
            X,
            y,
            sample_weight,
            offset,
            phi_method,
            verbose,
            trace_callback,
            trace_iterations,
        )
    else:
        ctx = _build_profile_context(
            model,
            X,
            y,
            sample_weight,
            offset,
            phi_method,
            verbose,
            trace_callback,
            trace_iterations,
        )

    # Dispatch search
    if method == "brent":
        return _search_brent(ctx, p_bounds, xatol, maxiter)
    if method == "grid":
        return _search_grid(ctx, p_bounds, n_grid, grid)
    if method == "grid_refine":
        return _search_grid_refine(ctx, p_bounds, n_grid_coarse, xatol, maxiter)
    return _search_profile_opt(ctx, p_bounds, optimizer, xatol, maxiter)


# ---------------------------------------------------------------------------
# Profile likelihood confidence interval
# ---------------------------------------------------------------------------


def profile_ci_p(
    objective,
    p_hat: float,
    nll_hat: float,
    ll_scale: float,
    *,
    alpha: float = 0.05,
    p_range: tuple[float, float] = (1.02, 1.98),
) -> tuple[float, float]:
    """Profile likelihood confidence interval for Tweedie power p.

    Each evaluation calls ``objective(p)`` which refits the GLM via PIRLS.
    The objective returns mean NLL per observation, so the LRT
    statistic is ``2 * ll_scale * (mean_nll(p) - mean_nll(p_hat))``.

    Parameters
    ----------
    objective : callable
        Profile objective ``p -> mean_nll``.
    p_hat : float
        MLE of p.
    nll_hat : float
        Mean NLL at p_hat.
    ll_scale : float
        Number of effective observations used to convert mean NLL to total.
    alpha : float
        Significance level.
    p_range : tuple
        Search range for CI endpoints.

    Returns
    -------
    (ci_lower, ci_upper) : tuple of float
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    cutoff = chi2.ppf(1.0 - alpha, 1)

    def g(p: float) -> float:
        nll = objective(p)
        return 2.0 * ll_scale * (nll - nll_hat) - cutoff

    # Lower bound
    lo = p_range[0]
    try:
        ci_lower = brentq(g, lo, p_hat, xtol=1e-3)
    except ValueError:
        ci_lower = lo

    # Upper bound
    hi = p_range[1]
    try:
        ci_upper = brentq(g, p_hat, hi, xtol=1e-3)
    except ValueError:
        ci_upper = hi

    return (ci_lower, ci_upper)
