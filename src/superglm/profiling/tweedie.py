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

import copy
import logging
import warnings as _warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize, minimize_scalar
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
    positive_saddlepoint_mask: NDArray
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
    positive_saddlepoint_mask = np.zeros(
        prepared.positive_indices.size,
        dtype=np.bool_,
    )
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
        positive_saddlepoint_mask = saddlepoint
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
        positive_saddlepoint_mask=_readonly_copy(
            positive_saddlepoint_mask,
            dtype=np.bool_,
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


_PHI_LOWER_BOUND = 1e-12
_PHI_UPPER_BOUND = 1e12
_LOG_PHI_LOWER_BOUND = float(np.log(_PHI_LOWER_BOUND))
_LOG_PHI_UPPER_BOUND = float(np.log(_PHI_UPPER_BOUND))
_PHI_SCORE_TOLERANCE = 1e-6
_PHI_ROOT_PROBE = 1e-5
_PHI_BOUNDED_XATOL = 1e-6
_PHI_FALLBACK_GRID_STEP = 1.0
_PHI_BRANCH_XTOL = 1e-12
_PHI_MAX_ANALYTIC_BRANCH_EDGES = 64
_PHI_MAX_NUMERIC_BRANCH_PROBES = 128
_PHI_MAX_FALLBACK_REFINEMENTS = 8
_PHI_MAX_LARGE_FALLBACK_REFINEMENTS = 4
_PHI_LARGE_PROFILE_THRESHOLD = 64
_PHI_ROOT_BRANCH_GRID_STEP = 1.0
_PHI_MAX_ROOT_BRANCH_PROBES = 256
_PHI_BRANCH_VERIFY_CHUNK_SIZE = 65_536
_PHI_NLL_ATOL = 1e-10
_PHI_NLL_RTOL = 1e-10


@dataclass(frozen=True)
class _PhiProfileResult:
    """Detailed result from profiling Tweedie dispersion at fixed ``(mu, p)``."""

    phi: float
    nll: float
    converged: bool
    objective_finite: bool
    n_evaluations: int
    n_score_evaluations: int
    n_value_only_evaluations: int
    n_fallback_evaluations: int
    optimizer: str
    score: float | None
    used_fallback: bool
    fallback_reason: str | None
    branch_switch_detected: bool
    lower_boundary: bool
    upper_boundary: bool
    diagnostics: _TweedieLogpdfDiagnostics
    message: str


@dataclass(frozen=True)
class _PhiBranchMask:
    """Collision-free packed density-branch mask retained by cached points."""

    size: int
    packed: bytes

    @classmethod
    def from_array(cls, values: NDArray) -> _PhiBranchMask:
        mask = np.asarray(values, dtype=np.bool_)
        packed = np.packbits(mask, bitorder="little").tobytes()
        return cls(size=int(mask.size), packed=packed)

    def unpack(self) -> NDArray:
        packed = np.frombuffer(self.packed, dtype=np.uint8)
        mask = np.unpackbits(
            packed,
            count=self.size,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        mask.setflags(write=False)
        return mask

    def changed_bits(self, other: _PhiBranchMask) -> NDArray:
        if self.size != other.size:
            raise ValueError("branch masks must have the same size")
        left = np.frombuffer(self.packed, dtype=np.uint8)
        right = np.frombuffer(other.packed, dtype=np.uint8)
        changed = np.unpackbits(
            np.bitwise_xor(left, right),
            count=self.size,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        changed.setflags(write=False)
        return changed


@dataclass(frozen=True)
class _PhiProfilePoint:
    """One cached exact objective point, keyed by its exact float ``u``."""

    u: float
    phi: float
    nll: float
    objective_finite: bool
    score: float | None
    score_attempted: bool
    score_valid: bool
    branch_mask: _PhiBranchMask
    diagnostics: _TweedieLogpdfDiagnostics

    @property
    def positive_saddlepoint_mask(self) -> NDArray:
        """Unpack the immutable branch bits only for per-observation comparisons."""
        return self.branch_mask.unpack()

    @property
    def branch_signature(self) -> tuple[int, bytes]:
        """Return the collision-free packed signature without duplicating its bytes."""
        return self.branch_mask.size, self.branch_mask.packed


@dataclass(frozen=True)
class _PhiCandidate:
    """A finite cached point and the method that established it."""

    point: _PhiProfilePoint
    source: str
    validated: bool = False


@dataclass(frozen=True)
class _PhiScoreSearchResult:
    """Candidates and safeguards discovered by analytic-score searching."""

    seed_candidates: tuple[_PhiCandidate, ...]
    root_candidates: tuple[_PhiCandidate, ...]
    fallback_reasons: tuple[str, ...]
    branch_switch_detected: bool


@dataclass(frozen=True)
class _PhiBoundedResult:
    """Outcome of the exact value-only safeguard."""

    candidate: _PhiCandidate | None
    success: bool
    message: str
    branch_switch_detected: bool = False


class _PhiRootAbortError(RuntimeError):
    """Abort a score root as soon as its analytic branch becomes untrustworthy."""


class _PhiEvaluationCache:
    """Cache exact value/score passes and account for actual density evaluations."""

    def __init__(self, prepared: _PreparedTweedieDensity):
        self.prepared = prepared
        self.points: dict[float, _PhiProfilePoint] = {}
        self.n_evaluations = 0
        self.n_score_evaluations = 0
        self.n_value_only_evaluations = 0
        self.n_fallback_evaluations = 0

    def evaluate(
        self,
        u: float,
        *,
        compute_score: bool,
        fallback: bool = False,
        phi_override: float | None = None,
    ) -> _PhiProfilePoint:
        key = float(u)
        cached = self.points.get(key)
        if cached is not None and (not compute_score or cached.score_attempted):
            return cached

        if phi_override is not None:
            phi = float(phi_override)
        elif key == _LOG_PHI_LOWER_BOUND:
            phi = _PHI_LOWER_BOUND
        elif key == _LOG_PHI_UPPER_BOUND:
            phi = _PHI_UPPER_BOUND
        else:
            phi = float(np.exp(key))
        evaluation = _evaluate_tweedie_density(
            self.prepared,
            phi,
            compute_score=compute_score,
        )
        self.n_evaluations += 1
        if compute_score:
            self.n_score_evaluations += 1
        else:
            self.n_value_only_evaluations += 1
        if fallback:
            self.n_fallback_evaluations += 1

        with np.errstate(all="ignore"):
            nll = -float(np.mean(evaluation.logpdf))
        objective_finite = bool(np.isfinite(nll))
        score: float | None = None
        score_valid = False
        if compute_score and evaluation.score_valid and evaluation.log_phi_score is not None:
            with np.errstate(all="ignore"):
                candidate_score = float(np.mean(evaluation.log_phi_score))
            if np.isfinite(candidate_score):
                score = candidate_score
                score_valid = True

        point = _PhiProfilePoint(
            u=key,
            phi=phi,
            nll=nll if objective_finite else np.inf,
            objective_finite=objective_finite,
            score=score,
            score_attempted=compute_score,
            score_valid=score_valid,
            branch_mask=_PhiBranchMask.from_array(evaluation.positive_saddlepoint_mask),
            diagnostics=evaluation.diagnostics,
        )
        self.points[key] = point
        return point


def _phi_nll_no_worse(candidate: float, reference: float) -> bool:
    """Return whether an exact NLL is no worse within the profiling tolerance."""
    tolerance = _PHI_NLL_ATOL + _PHI_NLL_RTOL * abs(reference)
    return bool(candidate <= reference + tolerance)


def _phi_profile_denominator(
    prepared: _PreparedTweedieDensity,
    df_resid: float | None,
) -> float:
    """Validate and return the observation-count denominator used for seeds."""
    if df_resid is None:
        return float(len(prepared.y))
    df_array = np.asarray(df_resid)
    if df_array.ndim != 0:
        raise ValueError("df_resid must be finite and strictly positive")
    try:
        denominator = float(df_array)
    except (TypeError, ValueError) as exc:
        raise ValueError("df_resid must be finite and strictly positive") from exc
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("df_resid must be finite and strictly positive")
    return denominator


def _pearson_phi_from_prepared(
    prepared: _PreparedTweedieDensity,
    denominator: float,
) -> float:
    """Compute the Pearson dispersion seed from already validated arrays."""
    with np.errstate(all="ignore"):
        numerator = float(
            np.sum(
                prepared.weights
                * (prepared.y - prepared.mu) ** 2
                / np.power(np.maximum(prepared.mu, 1e-10), prepared.p)
            )
        )
    return numerator / denominator


def _phi_profile_seeds(
    prepared: _PreparedTweedieDensity,
    denominator: float,
    pearson_phi: float,
    phi_start: float | None,
) -> list[tuple[float, str]]:
    """Build distinct clipped warm, Pearson, and stable data seeds in priority order."""
    seeds: list[tuple[float, str]] = []

    def add(candidate_phi: Any, source: str) -> bool:
        candidate_array = np.asarray(candidate_phi)
        if candidate_array.ndim != 0:
            return False
        try:
            candidate = float(candidate_array)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(candidate) or candidate <= 0.0:
            return False
        u = float(np.clip(np.log(candidate), _LOG_PHI_LOWER_BOUND, _LOG_PHI_UPPER_BOUND))
        if any(abs(u - existing_u) <= _PHI_BOUNDED_XATOL for existing_u, _ in seeds):
            return False
        seeds.append((u, source))
        return True

    add(phi_start, "warm start")
    pearson_usable = add(pearson_phi, "Pearson seed")
    if not pearson_usable:
        deviance = np.empty(len(prepared.y), dtype=np.float64)
        deviance[prepared.zero_mask] = 2.0 * prepared.zero_rate_numerator[prepared.zero_mask]
        deviance[prepared.positive_mask] = prepared.positive_saddlepoint_deviance
        with np.errstate(all="ignore"):
            data_phi = float(np.sum(prepared.weights * deviance)) / denominator
        add(data_phi, "mean-deviance seed")
    if not seeds:
        add(1.0, "unit seed")
    return seeds


def _record_phi_fallback(reasons: list[str], reason: str) -> None:
    """Append a fallback diagnosis once while retaining discovery order."""
    if reason not in reasons:
        reasons.append(reason)


def _positive_saddlepoint_mask_at_u_values(
    prepared: _PreparedTweedieDensity,
    positive_indices: NDArray,
    u_values: NDArray,
) -> NDArray:
    """Evaluate exact-density validity at one log-phi value per selected observation."""
    indices = np.asarray(positive_indices, dtype=np.intp)
    u_array = np.asarray(u_values, dtype=np.float64)
    if u_array.shape != indices.shape:
        raise ValueError("u_values must match positive_indices")
    log_t = prepared.positive_log_t_phi_independent[indices] - (prepared.a + 1.0) * u_array
    try_exact = log_t < prepared.log_t_arg_limit
    exact = np.zeros(indices.size, dtype=np.bool_)
    if np.any(try_exact):
        local_indices = np.flatnonzero(try_exact)
        selected_indices = indices[try_exact]
        with np.errstate(all="ignore"):
            t_exact = np.exp(log_t[try_exact])
            recurrence = wright_bessel(
                prepared.a,
                prepared.a + 1.0,
                t_exact,
            )
            original_indices = prepared.positive_indices[selected_indices]
            inverse_phi = prepared.weights[original_indices] / np.exp(u_array[try_exact])
            candidate_logpdf = (
                np.log(prepared.a)
                + log_t[try_exact]
                + np.log(recurrence)
                - prepared.positive_log_y[selected_indices]
                + prepared.positive_canonical_c[selected_indices] * inverse_phi
            )
        exact[local_indices] = (
            np.isfinite(recurrence) & (recurrence > 0.0) & np.isfinite(candidate_logpdf)
        )
    return ~exact


def _positive_saddlepoint_mask_subset(
    prepared: _PreparedTweedieDensity,
    u: float,
    positive_indices: NDArray,
) -> NDArray:
    """Evaluate exact-density validity for a bounded subset without a full density pass."""
    indices = np.asarray(positive_indices, dtype=np.intp)
    return _positive_saddlepoint_mask_at_u_values(
        prepared,
        indices,
        np.full(indices.size, u, dtype=np.float64),
    )


def _locate_first_realized_phi_branch_transition(
    prepared: _PreparedTweedieDensity,
    threshold: float,
    positive_indices: NDArray,
    remaining_probes: int,
) -> tuple[tuple[float, float] | None, int, bool]:
    """Locate the first realized transition after a nominal edge with bounded scalar work."""
    scale = max(1.0, abs(threshold))
    delta = max(_PHI_BRANCH_XTOL, 16.0 * np.finfo(np.float64).eps * scale)
    left_u = float(max(_LOG_PHI_LOWER_BOUND, threshold - delta))
    right_u = float(min(_LOG_PHI_UPPER_BOUND, threshold + delta))
    if remaining_probes < 2:
        return None, remaining_probes, False
    left_mask = _positive_saddlepoint_mask_subset(prepared, left_u, positive_indices)
    right_mask = _positive_saddlepoint_mask_subset(prepared, right_u, positive_indices)
    remaining_probes -= 2
    if not np.array_equal(left_mask, right_mask):
        return (left_u, right_u), remaining_probes, True

    previous_u = right_u
    previous_mask = right_mask
    while previous_u < _LOG_PHI_UPPER_BOUND:
        if remaining_probes <= 0:
            return None, remaining_probes, False
        current_u = float(min(_LOG_PHI_UPPER_BOUND, previous_u + _PHI_ROOT_BRANCH_GRID_STEP))
        current_mask = _positive_saddlepoint_mask_subset(
            prepared,
            current_u,
            positive_indices,
        )
        remaining_probes -= 1
        if not np.array_equal(previous_mask, current_mask):
            interval_left_u = previous_u
            interval_right_u = current_u
            interval_left_mask = previous_mask
            while interval_right_u - interval_left_u > _PHI_BRANCH_XTOL:
                if remaining_probes <= 0:
                    return (interval_left_u, interval_right_u), remaining_probes, False
                midpoint_u = float(0.5 * (interval_left_u + interval_right_u))
                midpoint_mask = _positive_saddlepoint_mask_subset(
                    prepared,
                    midpoint_u,
                    positive_indices,
                )
                remaining_probes -= 1
                if np.array_equal(midpoint_mask, interval_left_mask):
                    interval_left_u = midpoint_u
                else:
                    interval_right_u = midpoint_u
            return (interval_left_u, interval_right_u), remaining_probes, True
        previous_u = current_u
        previous_mask = current_mask
    return None, remaining_probes, True


def _clean_root_phi_branch_edges_verified(
    prepared: _PreparedTweedieDensity,
    thresholds_by_observation: NDArray,
) -> bool:
    """Verify every calibrated edge before allowing a clean root to certify."""
    if thresholds_by_observation.size != prepared.positive_indices.size:
        return False
    if np.any(np.isnan(thresholds_by_observation)):
        return False

    in_range = (
        np.isfinite(thresholds_by_observation)
        & (thresholds_by_observation > _LOG_PHI_LOWER_BOUND + _PHI_BRANCH_XTOL)
        & (thresholds_by_observation < _LOG_PHI_UPPER_BOUND - _PHI_BRANCH_XTOL)
    )
    verified_transitions = _verified_phi_branch_transitions_at_thresholds(
        prepared,
        thresholds_by_observation,
    )
    if np.any(verified_transitions[in_range] != -1):
        return False

    below_lower = thresholds_by_observation <= _LOG_PHI_LOWER_BOUND + _PHI_BRANCH_XTOL
    if np.any(below_lower):
        below_indices = np.flatnonzero(below_lower)
        lower_mask = _positive_saddlepoint_mask_subset(
            prepared,
            _LOG_PHI_LOWER_BOUND + _PHI_BRANCH_XTOL,
            below_indices,
        )
        if np.any(lower_mask):
            return False
    return True


def _better_phi_branch_edge_probes(
    cache: _PhiEvaluationCache,
    root_candidates: list[_PhiCandidate],
) -> tuple[list[_PhiCandidate], bool, bool]:
    """Probe realized branch edges nearest accepted roots with bounded scalar work."""
    prepared = cache.prepared
    thresholds_by_observation, calibrated = _phi_realized_wright_thresholds(prepared)
    if not calibrated:
        return [], False, True
    if not _clean_root_phi_branch_edges_verified(prepared, thresholds_by_observation):
        return [], False, True
    thresholds = thresholds_by_observation[
        np.isfinite(thresholds_by_observation)
        & (thresholds_by_observation > _LOG_PHI_LOWER_BOUND + _PHI_BRANCH_XTOL)
        & (thresholds_by_observation < _LOG_PHI_UPPER_BOUND - _PHI_BRANCH_XTOL)
    ]
    if thresholds.size == 0:
        return [], False, False

    selected_thresholds: set[float] = set()
    for candidate in root_candidates:
        below = thresholds[thresholds < candidate.point.u]
        above = thresholds[thresholds > candidate.point.u]
        if below.size:
            selected_thresholds.add(float(np.max(below)))
        if above.size:
            selected_thresholds.add(float(np.min(above)))

    best_root_nll = min(candidate.point.nll for candidate in root_candidates)
    better_candidates: list[_PhiCandidate] = []
    branch_switch_detected = False
    requires_fallback = False
    remaining_probes = _PHI_MAX_ROOT_BRANCH_PROBES
    for threshold in sorted(selected_thresholds):
        positive_indices = np.flatnonzero(thresholds_by_observation == threshold)
        scale = max(1.0, abs(threshold))
        delta = max(_PHI_BRANCH_XTOL, 16.0 * np.finfo(np.float64).eps * scale)
        left = cache.evaluate(threshold - delta, compute_score=False)
        right = cache.evaluate(threshold + delta, compute_score=False)
        targeted_change = left.branch_mask.changed_bits(right.branch_mask)[positive_indices]
        if not np.any(targeted_change):
            realized_bounds, remaining_probes, completed = (
                _locate_first_realized_phi_branch_transition(
                    prepared,
                    threshold,
                    positive_indices,
                    remaining_probes,
                )
            )
            if not completed:
                requires_fallback = True
                continue
            if realized_bounds is None:
                continue
            left = cache.evaluate(realized_bounds[0], compute_score=False)
            right = cache.evaluate(realized_bounds[1], compute_score=False)
            targeted_change = left.branch_mask.changed_bits(right.branch_mask)[positive_indices]
            if not np.any(targeted_change):
                requires_fallback = True
                continue
        branch_switch_detected = True
        finite_sides = [point for point in (left, right) if point.objective_finite]
        if not finite_sides:
            requires_fallback = True
            continue
        better_side = min(finite_sides, key=lambda point: point.nll)
        if _phi_nll_no_worse(best_root_nll, better_side.nll):
            continue
        better_candidates.append(_PhiCandidate(better_side, "seed"))
    return better_candidates, branch_switch_detected, requires_fallback


def _search_phi_score_candidates(
    cache: _PhiEvaluationCache,
    seeds: list[tuple[float, str]],
) -> _PhiScoreSearchResult:
    """Bracket, solve, and validate every trustworthy score minimum from the seeds."""
    fallback_reasons: list[str] = []
    branch_switch_detected = False
    seed_candidates: list[_PhiCandidate] = []
    seed_points: list[_PhiProfilePoint] = []
    brackets: list[tuple[_PhiProfilePoint, _PhiProfilePoint]] = []

    def add_bracket(left: _PhiProfilePoint, right: _PhiProfilePoint) -> None:
        key = (left.u, right.u)
        if not any(
            (existing_left.u, existing_right.u) == key for existing_left, existing_right in brackets
        ):
            brackets.append((left, right))

    for seed_u, _ in seeds:
        seed_point = cache.evaluate(seed_u, compute_score=True)
        seed_points.append(seed_point)
        if seed_point.objective_finite:
            seed_candidates.append(_PhiCandidate(seed_point, "seed"))
        if not seed_point.score_valid or seed_point.score is None:
            _record_phi_fallback(
                fallback_reasons,
                "analytic derivative unavailable at a profiling seed",
            )
            continue

        found_local_bracket = False
        if (
            abs(seed_point.score) <= _PHI_SCORE_TOLERANCE
            and seed_u - _PHI_ROOT_PROBE > _LOG_PHI_LOWER_BOUND
            and seed_u + _PHI_ROOT_PROBE < _LOG_PHI_UPPER_BOUND
        ):
            left_probe = cache.evaluate(seed_u - _PHI_ROOT_PROBE, compute_score=True)
            right_probe = cache.evaluate(seed_u + _PHI_ROOT_PROBE, compute_score=True)
            if (
                seed_point.objective_finite
                and left_probe.objective_finite
                and right_probe.objective_finite
                and left_probe.score_valid
                and right_probe.score_valid
                and left_probe.branch_signature
                == seed_point.branch_signature
                == right_probe.branch_signature
                and left_probe.score is not None
                and right_probe.score is not None
                and left_probe.score < 0.0 < right_probe.score
            ):
                add_bracket(left_probe, right_probe)
                found_local_bracket = True
            elif (
                left_probe.branch_signature != seed_point.branch_signature
                or right_probe.branch_signature != seed_point.branch_signature
            ):
                branch_switch_detected = True
                _record_phi_fallback(
                    fallback_reasons,
                    "density branch switched around a near-zero score",
                )
            else:
                _record_phi_fallback(
                    fallback_reasons,
                    "near-zero score failed local minimum orientation",
                )

        if found_local_bracket:
            continue

        direction = 1.0 if seed_point.score < 0.0 else -1.0
        previous = seed_point
        distance = 1.0
        while True:
            target_u = float(
                np.clip(
                    seed_u + direction * distance,
                    _LOG_PHI_LOWER_BOUND,
                    _LOG_PHI_UPPER_BOUND,
                )
            )
            if target_u == previous.u:
                break
            current = cache.evaluate(target_u, compute_score=True)
            if not current.score_valid or current.score is None:
                _record_phi_fallback(
                    fallback_reasons,
                    "analytic derivative unavailable during score bracketing",
                )
                break
            if not previous.objective_finite or not current.objective_finite:
                _record_phi_fallback(
                    fallback_reasons,
                    "exact objective became non-finite during score bracketing",
                )
                break
            left, right = (previous, current) if previous.u < current.u else (current, previous)
            if left.branch_signature != right.branch_signature:
                branch_switch_detected = True
                _record_phi_fallback(
                    fallback_reasons,
                    "density branch switched during score bracketing",
                )
                break
            if left.score is not None and right.score is not None:
                if left.score < 0.0 < right.score:
                    add_bracket(left, right)
                    break
                if left.score > 0.0 > right.score:
                    _record_phi_fallback(
                        fallback_reasons,
                        "score bracket has maximum rather than minimum orientation",
                    )
                    break
            previous = current
            if target_u in {_LOG_PHI_LOWER_BOUND, _LOG_PHI_UPPER_BOUND}:
                break
            distance *= 2.0

    if not brackets:
        _record_phi_fallback(fallback_reasons, "no trustworthy minimum score bracket")

    finite_seed_points = [point for point in seed_points if point.objective_finite]
    best_seed_nll = min(point.nll for point in finite_seed_points) if finite_seed_points else np.inf
    root_candidates: list[_PhiCandidate] = []

    for left, right in brackets:
        expected_signature = left.branch_signature

        def score_callback(u: float) -> float:
            nonlocal branch_switch_detected
            point = cache.evaluate(float(u), compute_score=True)
            if not point.objective_finite:
                raise _PhiRootAbortError("exact objective became non-finite inside brentq")
            if not point.score_valid or point.score is None:
                raise _PhiRootAbortError("analytic derivative became unavailable inside brentq")
            if point.branch_signature != expected_signature:
                branch_switch_detected = True
                raise _PhiRootAbortError("density branch switched inside brentq")
            return point.score

        try:
            root_u, root_info = brentq(
                score_callback,
                left.u,
                right.u,
                full_output=True,
                disp=False,
                xtol=1e-10,
                rtol=8.0 * np.finfo(np.float64).eps,
                maxiter=100,
            )
        except _PhiRootAbortError as exc:
            _record_phi_fallback(fallback_reasons, str(exc))
            continue
        except (RuntimeError, ValueError) as exc:
            _record_phi_fallback(fallback_reasons, f"brentq failed: {exc}")
            continue

        root_point = cache.evaluate(float(root_u), compute_score=True)
        valid_root = bool(root_info.converged)
        valid_root &= root_point.objective_finite
        valid_root &= root_point.score_valid and root_point.score is not None
        valid_root &= root_point.branch_signature == expected_signature
        valid_root &= root_point.score is not None and abs(root_point.score) <= _PHI_SCORE_TOLERANCE
        valid_root &= _phi_nll_no_worse(root_point.nll, left.nll)
        valid_root &= _phi_nll_no_worse(root_point.nll, right.nll)
        if np.isfinite(best_seed_nll):
            valid_root &= _phi_nll_no_worse(root_point.nll, best_seed_nll)

        if valid_root:
            probe_left_u = root_point.u - _PHI_ROOT_PROBE
            probe_right_u = root_point.u + _PHI_ROOT_PROBE
            if probe_left_u <= _LOG_PHI_LOWER_BOUND or probe_right_u >= _LOG_PHI_UPPER_BOUND:
                valid_root = False
            else:
                probe_left = cache.evaluate(probe_left_u, compute_score=True)
                probe_right = cache.evaluate(probe_right_u, compute_score=True)
                if (
                    probe_left.branch_signature != expected_signature
                    or probe_right.branch_signature != expected_signature
                ):
                    branch_switch_detected = True
                    valid_root = False
                valid_root &= probe_left.score_valid and probe_right.score_valid
                valid_root &= probe_left.score is not None and probe_right.score is not None
                valid_root &= (
                    probe_left.score is not None
                    and probe_right.score is not None
                    and probe_left.score < 0.0 < probe_right.score
                )
                valid_root &= probe_left.objective_finite and probe_right.objective_finite
                valid_root &= _phi_nll_no_worse(root_point.nll, probe_left.nll)
                valid_root &= _phi_nll_no_worse(root_point.nll, probe_right.nll)

        if not valid_root:
            _record_phi_fallback(
                fallback_reasons,
                "score root failed exact local-minimum validation",
            )
            continue
        if not any(abs(root_point.u - candidate.point.u) <= 1e-8 for candidate in root_candidates):
            root_candidates.append(_PhiCandidate(root_point, "brentq", validated=True))

    if len(root_candidates) > 1:
        _record_phi_fallback(
            fallback_reasons,
            "multiple distinct score minima require exact candidate comparison",
        )
    if root_candidates and not fallback_reasons:
        edge_candidates, edge_switch, edge_requires_fallback = _better_phi_branch_edge_probes(
            cache,
            root_candidates,
        )
        branch_switch_detected |= edge_switch
        if edge_candidates or edge_requires_fallback:
            seed_candidates.extend(edge_candidates)
            _record_phi_fallback(
                fallback_reasons,
                "realized branch-edge probe cannot certify the analytic score root",
            )
    return _PhiScoreSearchResult(
        seed_candidates=tuple(seed_candidates),
        root_candidates=tuple(root_candidates),
        fallback_reasons=tuple(fallback_reasons),
        branch_switch_detected=branch_switch_detected,
    )


def _run_phi_bounded_interval(
    cache: _PhiEvaluationCache,
    bounds: tuple[float, float],
) -> _PhiBoundedResult:
    """Run one exact bounded refinement and validate SciPy's returned value."""

    def value_objective(u: float) -> float:
        point = cache.evaluate(float(u), compute_score=False, fallback=True)
        return point.nll if point.objective_finite else np.inf

    try:
        bounded_result = minimize_scalar(
            value_objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": _PHI_BOUNDED_XATOL, "maxiter": 200},
        )
        message = str(getattr(bounded_result, "message", ""))
        bounded_u = float(bounded_result.x)
        in_bounds = bool(np.isfinite(bounded_u) and bounds[0] <= bounded_u <= bounds[1])
        candidate = None
        fun_consistent = False
        if in_bounds:
            point = cache.evaluate(
                bounded_u,
                compute_score=False,
                fallback=True,
            )
            fun_consistent = bool(
                point.objective_finite
                and np.isfinite(float(bounded_result.fun))
                and np.isclose(
                    point.nll,
                    float(bounded_result.fun),
                    rtol=_PHI_NLL_RTOL,
                    atol=_PHI_NLL_ATOL,
                )
            )
            if fun_consistent:
                candidate = _PhiCandidate(point, "bounded", validated=True)
        success = bool(getattr(bounded_result, "success", False) and in_bounds and fun_consistent)
        return _PhiBoundedResult(candidate=candidate, success=success, message=message)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        return _PhiBoundedResult(candidate=None, success=False, message=str(exc))


def _phi_analytic_branch_thresholds(prepared: _PreparedTweedieDensity) -> NDArray:
    """Return each positive observation's known exact/saddle log-phi threshold."""
    with np.errstate(all="ignore"):
        thresholds = (prepared.positive_log_t_phi_independent - prepared.log_t_arg_limit) / (
            prepared.a + 1.0
        )
    return np.asarray(thresholds, dtype=np.float64)


def _wright_density_recurrence_is_valid(a: float, log_t: float) -> bool:
    """Check the common Wright-recurrence validity predicate at one scalar ``log(t)``."""
    with np.errstate(all="ignore"):
        t = np.exp(log_t)
        recurrence = np.asarray(wright_bessel(a, a + 1.0, np.array([t]))).reshape(-1)[0]
    return bool(np.isfinite(recurrence) and recurrence > 0.0)


def _calibrate_wright_log_t_ceiling(prepared: _PreparedTweedieDensity) -> float | None:
    """Locate the realized Wright-validity ceiling once for this fixed power ``p``."""
    nominal = prepared.log_t_arg_limit
    if not np.isfinite(nominal):
        return None
    invalid_u = float(np.nextafter(nominal, -np.inf))
    if _wright_density_recurrence_is_valid(prepared.a, invalid_u):
        return nominal

    distance = 1.0
    valid_u: float | None = None
    for _ in range(16):
        candidate = float(invalid_u - distance)
        if _wright_density_recurrence_is_valid(prepared.a, candidate):
            valid_u = candidate
            break
        distance *= 2.0
    if valid_u is None:
        return None

    while invalid_u - valid_u > 1e-13:
        midpoint = float(0.5 * (valid_u + invalid_u))
        if midpoint in {valid_u, invalid_u}:
            break
        if _wright_density_recurrence_is_valid(prepared.a, midpoint):
            valid_u = midpoint
        else:
            invalid_u = midpoint
    return float(0.5 * (valid_u + invalid_u))


def _phi_realized_wright_thresholds(
    prepared: _PreparedTweedieDensity,
) -> tuple[NDArray, bool]:
    """Derive per-observation log-phi edges from one calibrated Wright ceiling."""
    ceiling = _calibrate_wright_log_t_ceiling(prepared)
    if ceiling is None:
        return _phi_analytic_branch_thresholds(prepared), False
    with np.errstate(all="ignore"):
        thresholds = (prepared.positive_log_t_phi_independent - ceiling) / (prepared.a + 1.0)
    return np.asarray(thresholds, dtype=np.float64), True


def _select_phi_branch_thresholds(
    thresholds_by_observation: NDArray,
    anchors: list[float],
) -> tuple[NDArray, bool, int]:
    """Select a bounded, range-covering set of known analytic branch edges."""
    in_range = thresholds_by_observation[
        np.isfinite(thresholds_by_observation)
        & (thresholds_by_observation > _LOG_PHI_LOWER_BOUND + _PHI_BRANCH_XTOL)
        & (thresholds_by_observation < _LOG_PHI_UPPER_BOUND - _PHI_BRANCH_XTOL)
    ]
    unique = np.unique(in_range)
    n_unique = int(unique.size)
    if n_unique <= _PHI_MAX_ANALYTIC_BRANCH_EDGES:
        return unique, True, n_unique

    n_cover = _PHI_MAX_ANALYTIC_BRANCH_EDGES // 2
    cover_indices = np.unique(np.rint(np.linspace(0, n_unique - 1, n_cover)).astype(np.intp))
    selected_indices = set(int(index) for index in cover_indices)
    finite_anchors = np.asarray([anchor for anchor in anchors if np.isfinite(anchor)])
    if finite_anchors.size:
        sorted_anchors = np.sort(finite_anchors)
        insertions = np.searchsorted(sorted_anchors, unique, side="left")
        left_indices = np.maximum(insertions - 1, 0)
        right_indices = np.minimum(insertions, sorted_anchors.size - 1)
        distances = np.minimum(
            np.abs(unique - sorted_anchors[left_indices]),
            np.abs(unique - sorted_anchors[right_indices]),
        )
        priority = np.argsort(distances, kind="stable")
    else:
        priority = np.arange(n_unique, dtype=np.intp)
    for index in priority:
        selected_indices.add(int(index))
        if len(selected_indices) >= _PHI_MAX_ANALYTIC_BRANCH_EDGES:
            break
    selected = unique[np.asarray(sorted(selected_indices), dtype=np.intp)]
    return selected, False, n_unique


def _phi_branch_edge_points(
    cache: _PhiEvaluationCache,
    thresholds: NDArray,
) -> list[tuple[_PhiProfilePoint, _PhiProfilePoint]]:
    """Evaluate controlled exact points immediately around analytic branch edges."""
    sides: list[tuple[_PhiProfilePoint, _PhiProfilePoint]] = []
    for threshold in thresholds:
        scale = max(1.0, abs(float(threshold)))
        delta = max(_PHI_BRANCH_XTOL, 16.0 * np.finfo(np.float64).eps * scale)
        left_u = float(max(_LOG_PHI_LOWER_BOUND, float(threshold) - delta))
        right_u = float(min(_LOG_PHI_UPPER_BOUND, float(threshold) + delta))
        left = cache.evaluate(left_u, compute_score=False, fallback=True)
        right = cache.evaluate(right_u, compute_score=False, fallback=True)
        sides.append((left, right))
    return sides


def _verified_phi_branch_transitions(
    thresholds_by_observation: NDArray,
    selected_thresholds: NDArray,
    edge_sides: list[tuple[_PhiProfilePoint, _PhiProfilePoint]],
) -> NDArray:
    """Record only per-observation nominal edges whose controlled sides toggled."""
    transitions = np.zeros(thresholds_by_observation.size, dtype=np.int8)
    for threshold, (left, right) in zip(selected_thresholds, edge_sides, strict=True):
        at_threshold = thresholds_by_observation == threshold
        changed = left.branch_mask.changed_bits(right.branch_mask)
        verified = at_threshold & changed
        if not np.any(verified):
            continue
        left_mask = left.positive_saddlepoint_mask
        right_mask = right.positive_saddlepoint_mask
        transitions[verified] = right_mask[verified].astype(np.int8) - left_mask[verified].astype(
            np.int8
        )
    return transitions


def _verified_phi_branch_transitions_at_thresholds(
    prepared: _PreparedTweedieDensity,
    thresholds_by_observation: NDArray,
) -> NDArray:
    """Verify every calibrated edge in chunks without retaining full density masks."""
    transitions = np.zeros(thresholds_by_observation.size, dtype=np.int8)
    for start in range(0, thresholds_by_observation.size, _PHI_BRANCH_VERIFY_CHUNK_SIZE):
        stop = min(start + _PHI_BRANCH_VERIFY_CHUNK_SIZE, thresholds_by_observation.size)
        indices = np.arange(start, stop, dtype=np.intp)
        thresholds = thresholds_by_observation[start:stop]
        scale = np.maximum(1.0, np.abs(thresholds))
        delta = np.maximum(
            _PHI_BRANCH_XTOL,
            16.0 * np.finfo(np.float64).eps * scale,
        )
        valid = (
            np.isfinite(thresholds)
            & (thresholds - delta >= _LOG_PHI_LOWER_BOUND)
            & (thresholds + delta <= _LOG_PHI_UPPER_BOUND)
        )
        if not np.any(valid):
            continue
        valid_indices = indices[valid]
        left_mask = _positive_saddlepoint_mask_at_u_values(
            prepared,
            valid_indices,
            thresholds[valid] - delta[valid],
        )
        right_mask = _positive_saddlepoint_mask_at_u_values(
            prepared,
            valid_indices,
            thresholds[valid] + delta[valid],
        )
        changed = left_mask != right_mask
        local_transitions = np.zeros(valid_indices.size, dtype=np.int8)
        local_transitions[changed] = right_mask[changed].astype(np.int8) - left_mask[
            changed
        ].astype(np.int8)
        transitions[valid_indices] = local_transitions
    return transitions


def _phi_branch_change_is_unexplained(
    left: _PhiProfilePoint,
    right: _PhiProfilePoint,
    thresholds_by_observation: NDArray,
    verified_transitions_by_observation: NDArray | None = None,
) -> bool:
    """Return whether a mask change lacks a verified nominal-edge transition."""
    changed = left.branch_mask.changed_bits(right.branch_mask)
    if not np.any(changed):
        return False
    if verified_transitions_by_observation is None:
        verified_transitions_by_observation = np.zeros(
            thresholds_by_observation.size,
            dtype=np.int8,
        )
    left_mask = left.positive_saddlepoint_mask
    right_mask = right.positive_saddlepoint_mask
    observed_transitions = right_mask[changed].astype(np.int8) - left_mask[changed].astype(np.int8)
    changed_thresholds = thresholds_by_observation[changed]
    explained = (
        np.isfinite(changed_thresholds)
        & (changed_thresholds >= left.u - _PHI_BRANCH_XTOL)
        & (changed_thresholds <= right.u + _PHI_BRANCH_XTOL)
        & (verified_transitions_by_observation[changed] == observed_transitions)
    )
    return bool(np.any(~explained))


def _locate_unexplained_phi_branch_sides(
    cache: _PhiEvaluationCache,
    left: _PhiProfilePoint,
    right: _PhiProfilePoint,
    thresholds_by_observation: NDArray,
    remaining_probes: int,
    verified_transitions_by_observation: NDArray | None = None,
) -> tuple[list[tuple[_PhiProfilePoint, _PhiProfilePoint]], int, bool]:
    """Boundedly bisect numerical Wright-validity transitions not known analytically."""
    pending = [(left, right)]
    sides: list[tuple[_PhiProfilePoint, _PhiProfilePoint]] = []
    completed = True
    while pending:
        interval_left, interval_right = pending.pop()
        if not _phi_branch_change_is_unexplained(
            interval_left,
            interval_right,
            thresholds_by_observation,
            verified_transitions_by_observation,
        ):
            continue
        if interval_right.u - interval_left.u <= _PHI_BRANCH_XTOL:
            sides.append((interval_left, interval_right))
            continue
        if remaining_probes <= 0:
            completed = False
            continue
        midpoint_u = float(0.5 * (interval_left.u + interval_right.u))
        if midpoint_u in {interval_left.u, interval_right.u}:
            sides.append((interval_left, interval_right))
            continue
        midpoint = cache.evaluate(midpoint_u, compute_score=False, fallback=True)
        remaining_probes -= 1
        pending.append((midpoint, interval_right))
        pending.append((interval_left, midpoint))
    return sides, remaining_probes, completed


def _finite_phi_fallback_segments(
    points: list[_PhiProfilePoint],
) -> list[list[_PhiProfilePoint]]:
    """Partition ordered finite points by full density-branch signature."""
    segments: list[list[_PhiProfilePoint]] = []
    current: list[_PhiProfilePoint] = []
    for point in points:
        if not point.objective_finite:
            if current:
                segments.append(current)
                current = []
            continue
        if current and point.branch_signature != current[-1].branch_signature:
            segments.append(current)
            current = []
        current.append(point)
    if current:
        segments.append(current)
    return segments


def _run_phi_bounded_fallback(
    cache: _PhiEvaluationCache,
    *,
    required: bool,
) -> _PhiBoundedResult:
    """Globally safeguard fallback profiles with a branch-partitioned exact scan."""
    if not required:
        return _PhiBoundedResult(candidate=None, success=True, message="")

    full_range = _run_phi_bounded_interval(
        cache,
        (_LOG_PHI_LOWER_BOUND, _LOG_PHI_UPPER_BOUND),
    )
    n_intervals = int(
        np.ceil((_LOG_PHI_UPPER_BOUND - _LOG_PHI_LOWER_BOUND) / _PHI_FALLBACK_GRID_STEP)
    )
    grid = np.linspace(
        _LOG_PHI_LOWER_BOUND,
        _LOG_PHI_UPPER_BOUND,
        n_intervals + 1,
    )
    grid_points = [cache.evaluate(float(u), compute_score=False, fallback=True) for u in grid]
    all_points = {point.u: point for point in grid_points}
    finite_grid_points = [point for point in grid_points if point.objective_finite]
    anchors = [point.u for point in finite_grid_points]
    if full_range.candidate is not None:
        anchors.append(full_range.candidate.point.u)
    if finite_grid_points:
        anchors.append(min(finite_grid_points, key=lambda point: point.nll).u)

    thresholds_by_observation, wright_calibrated = _phi_realized_wright_thresholds(cache.prepared)
    selected_thresholds, analytic_scan_completed, n_analytic_thresholds = (
        _select_phi_branch_thresholds(thresholds_by_observation, anchors)
    )
    branch_switch_detected = any(
        left.branch_signature != right.branch_signature
        for left, right in zip(grid_points[:-1], grid_points[1:])
    )
    edge_sides = _phi_branch_edge_points(cache, selected_thresholds)
    selected_verified_transitions = _verified_phi_branch_transitions(
        thresholds_by_observation,
        selected_thresholds,
        edge_sides,
    )
    verified_transitions = _verified_phi_branch_transitions_at_thresholds(
        cache.prepared,
        thresholds_by_observation,
    )
    for threshold in selected_thresholds:
        at_threshold = thresholds_by_observation == threshold
        verified_transitions[at_threshold] = selected_verified_transitions[at_threshold]
    for side_left, side_right in edge_sides:
        all_points[side_left.u] = side_left
        all_points[side_right.u] = side_right
        branch_switch_detected |= side_left.branch_signature != side_right.branch_signature

    remaining_numeric_probes = _PHI_MAX_NUMERIC_BRANCH_PROBES
    numeric_scan_completed = True
    initially_ordered = sorted(all_points.values(), key=lambda point: point.u)
    for left, right in zip(initially_ordered[:-1], initially_ordered[1:]):
        if left.branch_signature == right.branch_signature:
            continue
        branch_switch_detected = True
        sides, remaining_numeric_probes, interval_completed = _locate_unexplained_phi_branch_sides(
            cache,
            left,
            right,
            thresholds_by_observation,
            remaining_numeric_probes,
            verified_transitions,
        )
        numeric_scan_completed &= interval_completed
        for side_left, side_right in sides:
            all_points[side_left.u] = side_left
            all_points[side_right.u] = side_right

    ordered_points = sorted(all_points.values(), key=lambda point: point.u)
    segments = _finite_phi_fallback_segments(ordered_points)
    candidates = [
        _PhiCandidate(point, "bounded", validated=True)
        for point in ordered_points
        if point.objective_finite
    ]
    if full_range.candidate is not None:
        candidates.append(full_range.candidate)

    refinement_priorities: dict[tuple[float, float], float] = {}

    def add_refinement(bounds: tuple[float, float], priority: float) -> None:
        if bounds[1] - bounds[0] <= _PHI_BOUNDED_XATOL:
            return
        existing = refinement_priorities.get(bounds, np.inf)
        refinement_priorities[bounds] = min(existing, priority)

    for segment in segments:
        if len(segment) < 2:
            continue
        segment_bounds_before = len(refinement_priorities)
        for index, point in enumerate(segment):
            left_nll = segment[index - 1].nll if index > 0 else np.inf
            right_nll = segment[index + 1].nll if index + 1 < len(segment) else np.inf
            if point.nll <= left_nll and point.nll <= right_nll:
                lower = segment[max(0, index - 1)].u
                upper = segment[min(len(segment) - 1, index + 1)].u
                add_refinement((lower, upper), point.nll)
        if len(refinement_priorities) == segment_bounds_before:
            lower, upper = segment[0].u, segment[-1].u
            add_refinement((lower, upper), min(point.nll for point in segment))

    ordered_refinements = sorted(
        refinement_priorities,
        key=lambda bounds: (refinement_priorities[bounds], bounds),
    )
    max_refinements = (
        _PHI_MAX_LARGE_FALLBACK_REFINEMENTS
        if len(cache.prepared.y) > _PHI_LARGE_PROFILE_THRESHOLD
        else _PHI_MAX_FALLBACK_REFINEMENTS
    )
    refinement_scan_completed = len(ordered_refinements) <= max_refinements
    selected_refinements = ordered_refinements[:max_refinements]

    refinement_results = [
        _run_phi_bounded_interval(cache, bounds) for bounds in selected_refinements
    ]
    for refinement in refinement_results:
        if refinement.candidate is not None:
            candidates.append(refinement.candidate)

    finite_grid_indices = [
        index for index, point in enumerate(grid_points) if point.objective_finite
    ]
    finite_gap = False
    if finite_grid_indices:
        first_finite = finite_grid_indices[0]
        last_finite = finite_grid_indices[-1]
        finite_gap = any(
            not point.objective_finite for point in grid_points[first_finite : last_finite + 1]
        )
    scan_completed = bool(
        candidates
        and full_range.success
        and all(result.success for result in refinement_results)
        and not finite_gap
        and analytic_scan_completed
        and numeric_scan_completed
        and refinement_scan_completed
    )
    best_candidate = (
        min(candidates, key=lambda candidate: candidate.point.nll) if candidates else None
    )
    messages = [message for message in [full_range.message] if message]
    failed_messages = {
        result.message for result in refinement_results if not result.success and result.message
    }
    messages.extend(sorted(failed_messages))
    messages.append(
        f"Global fallback scan covered {len(ordered_points)} exact points "
        f"in {len(segments)} finite branch segments; probed "
        f"{len(selected_thresholds)} of {n_analytic_thresholds} analytic branch edges "
        f"and refined {len(selected_refinements)} of {len(ordered_refinements)} basins; "
        f"Wright ceiling calibrated={wright_calibrated}."
    )
    return _PhiBoundedResult(
        candidate=best_candidate,
        success=scan_completed,
        message=" ".join(messages),
        branch_switch_detected=branch_switch_detected,
    )


def _finalize_phi_mle_result(
    cache: _PhiEvaluationCache,
    score_search: _PhiScoreSearchResult,
    bounded: _PhiBoundedResult,
    default_diagnostics: _TweedieLogpdfDiagnostics,
) -> _PhiProfileResult:
    """Compare exact candidates and propagate convergence and boundary diagnostics."""
    fallback_reasons = list(score_search.fallback_reasons)
    branch_switch_detected = bool(
        score_search.branch_switch_detected or bounded.branch_switch_detected
    )
    if bounded.branch_switch_detected and not score_search.branch_switch_detected:
        _record_phi_fallback(
            fallback_reasons,
            "density branch switched during value-only fallback scan",
        )
    need_fallback = bool(fallback_reasons)
    lower_point = cache.evaluate(_LOG_PHI_LOWER_BOUND, compute_score=False)
    upper_point = cache.evaluate(_LOG_PHI_UPPER_BOUND, compute_score=False)
    candidates = [*score_search.seed_candidates, *score_search.root_candidates]
    if bounded.candidate is not None:
        candidates.append(bounded.candidate)
    if lower_point.objective_finite:
        candidates.append(_PhiCandidate(lower_point, "lower boundary", validated=True))
    if upper_point.objective_finite:
        candidates.append(_PhiCandidate(upper_point, "upper boundary", validated=True))
    finite_candidates = [candidate for candidate in candidates if candidate.point.objective_finite]

    if not finite_candidates:
        diagnostics = next(
            (point.diagnostics for point in cache.points.values()),
            default_diagnostics,
        )
        return _PhiProfileResult(
            phi=np.nan,
            nll=np.inf,
            converged=False,
            objective_finite=False,
            n_evaluations=cache.n_evaluations,
            n_score_evaluations=cache.n_score_evaluations,
            n_value_only_evaluations=cache.n_value_only_evaluations,
            n_fallback_evaluations=cache.n_fallback_evaluations,
            optimizer="bounded" if need_fallback else "brentq",
            score=None,
            used_fallback=need_fallback,
            fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
            branch_switch_detected=branch_switch_detected,
            lower_boundary=False,
            upper_boundary=False,
            diagnostics=diagnostics,
            message=bounded.message or "No finite exact phi-profile objective was found.",
        )

    raw_best = min(finite_candidates, key=lambda candidate: candidate.point.nll)
    minimum_nll = raw_best.point.nll
    equivalent_best = [
        candidate
        for candidate in finite_candidates
        if _phi_nll_no_worse(candidate.point.nll, minimum_nll)
    ]
    source_priority = {
        "brentq": 0,
        "bounded": 1,
        "seed": 2,
    }
    if raw_best.source in {"lower boundary", "upper boundary"}:
        best = raw_best
    else:
        equivalent_interior = [
            candidate
            for candidate in equivalent_best
            if candidate.source not in {"lower boundary", "upper boundary"}
        ]
        best = min(
            equivalent_interior,
            key=lambda candidate: (source_priority[candidate.source], candidate.point.nll),
        )
    if (
        best.point.u - _LOG_PHI_LOWER_BOUND <= 4.0 * _PHI_BOUNDED_XATOL
        and lower_point.objective_finite
        and _phi_nll_no_worse(lower_point.nll, best.point.nll)
    ):
        best = _PhiCandidate(lower_point, "lower boundary", validated=True)
    elif (
        _LOG_PHI_UPPER_BOUND - best.point.u <= 4.0 * _PHI_BOUNDED_XATOL
        and upper_point.objective_finite
        and _phi_nll_no_worse(upper_point.nll, best.point.nll)
    ):
        best = _PhiCandidate(upper_point, "upper boundary", validated=True)

    lower_boundary = best.point.u == _LOG_PHI_LOWER_BOUND
    upper_boundary = best.point.u == _LOG_PHI_UPPER_BOUND
    final_point = best.point
    boundary_kkt = not (lower_boundary or upper_boundary)
    if lower_boundary or upper_boundary:
        final_point = cache.evaluate(final_point.u, compute_score=True)
        if final_point.score_valid and final_point.score is not None:
            if lower_boundary:
                boundary_kkt = final_point.score >= -_PHI_SCORE_TOLERANCE
            else:
                boundary_kkt = final_point.score <= _PHI_SCORE_TOLERANCE
        else:
            inward_u = (
                final_point.u + _PHI_ROOT_PROBE
                if lower_boundary
                else final_point.u - _PHI_ROOT_PROBE
            )
            inward_point = cache.evaluate(inward_u, compute_score=False)
            boundary_kkt = bool(
                inward_point.objective_finite
                and _phi_nll_no_worse(final_point.nll, inward_point.nll)
            )

    if lower_boundary or upper_boundary:
        candidate_converged = best.validated and boundary_kkt
    elif best.source == "brentq":
        candidate_converged = best.validated
    elif best.source == "bounded":
        # A deterministic scan is a strong global safeguard, but it cannot
        # prove that a narrow switch-and-back was not hidden between two
        # equal-signature grid points. Preserve the best exact value without
        # claiming global convergence for an interior value-only fallback.
        candidate_converged = False
    else:
        candidate_converged = False
    converged = bool(final_point.objective_finite and candidate_converged and not need_fallback)

    if best.source == "brentq":
        optimizer = "brentq"
    elif need_fallback or best.source == "bounded" or lower_boundary or upper_boundary:
        optimizer = "bounded"
    else:
        optimizer = "brentq"
    message_parts = []
    if fallback_reasons:
        message_parts.append("Fallback: " + "; ".join(fallback_reasons))
    if bounded.message:
        message_parts.append(bounded.message)
    if need_fallback:
        message_parts.append(
            "The deterministic fallback is best-effort and does not certify the global minimum."
        )
    if lower_boundary:
        message_parts.append("Selected the exact hard lower phi boundary.")
    if upper_boundary:
        message_parts.append("Selected the exact hard upper phi boundary.")
    if not message_parts:
        message_parts.append("Analytic score root passed exact local-minimum validation.")

    return _PhiProfileResult(
        phi=final_point.phi,
        nll=final_point.nll,
        converged=converged,
        objective_finite=final_point.objective_finite,
        n_evaluations=cache.n_evaluations,
        n_score_evaluations=cache.n_score_evaluations,
        n_value_only_evaluations=cache.n_value_only_evaluations,
        n_fallback_evaluations=cache.n_fallback_evaluations,
        optimizer=optimizer,
        score=final_point.score if final_point.score_valid else None,
        used_fallback=need_fallback,
        fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
        branch_switch_detected=branch_switch_detected,
        lower_boundary=lower_boundary,
        upper_boundary=upper_boundary,
        diagnostics=final_point.diagnostics,
        message=" ".join(message_parts),
    )


def _profile_phi_detailed(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    df_resid: float | None = None,
    phi_method: str = "pearson",
    phi_start: float | None = None,
) -> _PhiProfileResult:
    """Profile Tweedie dispersion with cached exact values and analytic scores."""
    if phi_method not in {"mle", "pearson"}:
        raise ValueError(
            f"phi_method={phi_method!r} is not valid, expected one of ['mle', 'pearson']"
        )

    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)
    denominator = _phi_profile_denominator(prepared, df_resid)
    pearson_phi = _pearson_phi_from_prepared(prepared, denominator)

    cache = _PhiEvaluationCache(prepared)
    default_diagnostics = _TweedieLogpdfDiagnostics(
        n_positive=int(prepared.positive_indices.size),
        n_saddlepoint=0,
    )

    if phi_method == "pearson":
        phi_hat = max(pearson_phi, 1e-10)
        if not np.isfinite(phi_hat) or phi_hat <= 0.0:
            return _PhiProfileResult(
                phi=float(phi_hat),
                nll=np.inf,
                converged=False,
                objective_finite=False,
                n_evaluations=0,
                n_score_evaluations=0,
                n_value_only_evaluations=0,
                n_fallback_evaluations=0,
                optimizer="pearson",
                score=None,
                used_fallback=False,
                fallback_reason=None,
                branch_switch_detected=False,
                lower_boundary=False,
                upper_boundary=False,
                diagnostics=default_diagnostics,
                message="Pearson plug-in dispersion is not finite.",
            )
        point = cache.evaluate(
            float(np.log(phi_hat)),
            compute_score=False,
            phi_override=phi_hat,
        )
        return _PhiProfileResult(
            phi=point.phi,
            nll=point.nll,
            converged=point.objective_finite,
            objective_finite=point.objective_finite,
            n_evaluations=cache.n_evaluations,
            n_score_evaluations=cache.n_score_evaluations,
            n_value_only_evaluations=cache.n_value_only_evaluations,
            n_fallback_evaluations=cache.n_fallback_evaluations,
            optimizer="pearson",
            score=None,
            used_fallback=False,
            fallback_reason=None,
            branch_switch_detected=False,
            lower_boundary=False,
            upper_boundary=False,
            diagnostics=point.diagnostics,
            message="Pearson plug-in dispersion evaluated with the exact objective.",
        )

    seeds = _phi_profile_seeds(prepared, denominator, pearson_phi, phi_start)
    score_search = _search_phi_score_candidates(cache, seeds)
    need_fallback = bool(score_search.fallback_reasons)
    bounded = _run_phi_bounded_fallback(cache, required=need_fallback)
    return _finalize_phi_mle_result(
        cache,
        score_search,
        bounded,
        default_diagnostics,
    )


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
    result = _profile_phi_detailed(
        y,
        mu,
        p,
        weights=weights,
        df_resid=df_resid,
        phi_method=phi_method,
    )
    return result.phi, result.nll


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
    "solver_converged",
    "reml_converged",
    "source",
    "fit_trace",
    "fit_trace_kind",
    "edf",
    "phi_converged",
    "phi_n_evaluations",
    "phi_n_score_evaluations",
    "phi_n_value_only_evaluations",
    "phi_n_fallback_evaluations",
    "phi_boundary",
    "phi_optimizer",
    "phi_score",
    "objective_finite",
    "n_saddlepoint",
    "n_positive",
    "saddlepoint_fraction",
    "phi_used_fallback",
    "phi_fallback_reason",
    "phi_branch_switch_detected",
    "phi_message",
    "density_method",
    "density_exact",
]
_SADDLEPOINT_WARN_THRESHOLD = 0.10
_SADDLEPOINT_HIGH_THRESHOLD = 0.50
_NEAR_POWER_LOWER = 1.08
_NEAR_POWER_UPPER = 1.98

_DensityMethod = Literal["exact", "hybrid_exact_saddlepoint", "saddlepoint"]
_DensityWarningSeverity = Literal["none", "label", "warning", "high"]
_DENSITY_SEVERITY_RANK: dict[_DensityWarningSeverity, int] = {
    "none": 0,
    "label": 1,
    "warning": 2,
    "high": 3,
}


@dataclass(frozen=True)
class _DensitySummary:
    """Validated classification of one record's density diagnostics."""

    n_positive: int
    n_saddlepoint: int
    fraction: float
    method: _DensityMethod
    exact: bool
    saddle_severity: _DensityWarningSeverity
    severity: _DensityWarningSeverity
    near_power_boundary: bool
    inconsistent: bool


def _density_count(value: Any) -> int | None:
    """Return one finite integer count, or ``None`` for malformed diagnostics."""
    try:
        values = np.asarray(value)
        if (
            values.ndim != 0
            or np.iscomplexobj(values)
            or np.issubdtype(values.dtype, np.bool_)
            or not np.issubdtype(values.dtype, np.number)
        ):
            return None
        parsed = float(values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(parsed) or parsed != np.floor(parsed):
        return None
    return int(parsed)


def _saddlepoint_warning_severity(
    n_saddlepoint: int,
    fraction: float,
) -> _DensityWarningSeverity:
    """Map one valid record's saddlepoint fraction to the public policy."""
    if n_saddlepoint == 0:
        return "none"
    if fraction >= _SADDLEPOINT_HIGH_THRESHOLD:
        return "high"
    if fraction >= _SADDLEPOINT_WARN_THRESHOLD:
        return "warning"
    return "label"


def _classify_density_diagnostics(p: float, diagnostics: Any) -> _DensitySummary:
    """Classify exact and saddlepoint terms without trusting malformed counts."""
    n_positive = _density_count(getattr(diagnostics, "n_positive", None))
    n_saddlepoint = _density_count(getattr(diagnostics, "n_saddlepoint", None))
    inconsistent = bool(
        n_positive is None
        or n_saddlepoint is None
        or n_positive < 0
        or n_saddlepoint < 0
        or n_saddlepoint > n_positive
    )
    if inconsistent:
        return _DensitySummary(
            n_positive=-1 if n_positive is None else n_positive,
            n_saddlepoint=-1 if n_saddlepoint is None else n_saddlepoint,
            fraction=1.0,
            method="hybrid_exact_saddlepoint",
            exact=False,
            saddle_severity="high",
            severity="high",
            near_power_boundary=False,
            inconsistent=True,
        )

    assert n_positive is not None
    assert n_saddlepoint is not None
    fraction = 0.0 if n_positive == 0 else float(n_saddlepoint) / float(n_positive)
    if n_saddlepoint == 0:
        method: _DensityMethod = "exact"
    elif n_saddlepoint == n_positive:
        method = "saddlepoint"
    else:
        method = "hybrid_exact_saddlepoint"
    saddle_severity = _saddlepoint_warning_severity(n_saddlepoint, fraction)
    near_power_boundary = bool(
        n_saddlepoint > 0 and (p <= _NEAR_POWER_LOWER or p >= _NEAR_POWER_UPPER)
    )
    severity: _DensityWarningSeverity = "high" if near_power_boundary else saddle_severity
    return _DensitySummary(
        n_positive=n_positive,
        n_saddlepoint=n_saddlepoint,
        fraction=fraction,
        method=method,
        exact=method == "exact",
        saddle_severity=saddle_severity,
        severity=severity,
        near_power_boundary=near_power_boundary,
        inconsistent=False,
    )


@dataclass(frozen=True)
class TweedieProfileCIEvaluation:
    """One finite fixed-p likelihood-ratio evaluation used by a profile CI."""

    p: float
    nll: float
    lr_statistic: float


@dataclass(frozen=True)
class TweedieProfileCIDensityProvenance:
    """Density method retained for one evaluated point in the connected LR region."""

    p: float
    source: str
    n_positive: int
    n_saddlepoint: int
    fraction: float
    method: _DensityMethod
    lr_statistic: float
    counts_valid: bool = True


@dataclass(frozen=True)
class TweedieProfileCIEndpoint:
    """One profile-CI endpoint and how it was obtained."""

    value: float
    status: Literal["root_found", "truncated"]
    at_range_boundary: bool
    lr_statistic: float


@dataclass(frozen=True)
class TweedieProfileCIDetails:
    """Immutable evidence and diagnostics for one Tweedie profile CI.

    ``density_provenance`` covers retained, actually evaluated points in the
    connected LR region; it is not a claim about every unevaluated power value.
    """

    alpha: float
    cutoff: float
    p_range: tuple[float, float]
    lower: TweedieProfileCIEndpoint
    upper: TweedieProfileCIEndpoint
    interval: tuple[float, float]
    n_new_evaluations: int
    evaluations: tuple[TweedieProfileCIEvaluation, ...]
    warnings: tuple[str, ...]
    density_provenance: tuple[TweedieProfileCIDensityProvenance, ...] = ()
    density_method: _DensityMethod | None = None
    density_exact: bool | None = None
    density_warning_severity: _DensityWarningSeverity = "none"
    near_power_boundary: bool = False
    max_saddlepoint_fraction: float = 0.0
    max_saddlepoint_p: float | None = None
    any_saddlepoint: bool = False
    n_density_records: int = 0
    n_saddlepoint_records: int = 0
    n_invalid_density_records: int = 0
    n_positive: int | None = None
    n_saddlepoint: int | None = None
    density_warnings: tuple[str, ...] = ()
    density_warning_signatures: tuple[str, ...] = ()


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
        Completed distinct fixed-p records in the immutable search snapshot.
        Later CI or plotting probes do not change this value.
    n_total_evaluations : int
        Dynamic count of all completed distinct fixed-p records, including
        successful post-search CI or plotting probes.
    n_post_search_evaluations : int
        Number of completed distinct records added after the search snapshot.
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
    density_method : {"exact", "hybrid_exact_saddlepoint", "saddlepoint"} or None
        Density evaluation method in the immutable winning record.
    density_exact : bool or None
        Whether every winning-record density term was evaluated exactly.
    density_warning_severity : {"none", "label", "warning", "high"}
        Highest density or saddle-qualified near-power diagnostic severity.
    near_power_boundary : bool
        Whether saddlepoint terms were used at ``p <= 1.08`` or ``p >= 1.98``.
    outer_converged : bool
        Whether the requested outer search itself converged.
    outer_message : str
        Diagnostic message returned by the outer optimizer.
    outer_boundary : {"lower", "upper"} or None
        Configured search endpoint selected as the winning record, if any.
    fit_converged : bool
        Whether the winning fixed-p fit converged, including both REML and
        final solver convergence for ``fit_mode="fit_reml"``.
    objective_finite : bool
        Whether the winning profiled objective is finite and valid.
    phi_converged : bool
        Whether the winning inner dispersion profile converged.
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
    outer_converged: bool = True
    outer_message: str = ""
    outer_boundary: str | None = None
    fit_converged: bool = True
    solver_converged: bool = True
    reml_converged: bool | None = None
    objective_finite: bool = True
    phi_converged: bool = True
    phi_n_evaluations: int = 0
    phi_n_score_evaluations: int = 0
    phi_n_value_only_evaluations: int = 0
    phi_n_fallback_evaluations: int = 0
    phi_optimizer: str = ""
    phi_score: float | None = None
    phi_used_fallback: bool = False
    phi_fallback_reason: str | None = None
    phi_branch_switch_detected: bool = False
    phi_boundary: str = ""
    phi_message: str = ""
    density_method: _DensityMethod | None = field(default=None, kw_only=True)
    density_exact: bool | None = field(default=None, kw_only=True)
    density_warning_severity: _DensityWarningSeverity = field(default="none", kw_only=True)
    near_power_boundary: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Derive new density fields for legacy positional construction."""
        self._ensure_density_compat_state()

    def _ensure_density_compat_state(self) -> None:
        """Restore density fields absent from legacy construction or pickle state."""
        summary = _classify_density_diagnostics(
            getattr(self, "p_hat", np.nan),
            _TweedieLogpdfDiagnostics(
                n_positive=getattr(self, "n_positive", None),
                n_saddlepoint=getattr(self, "n_saddlepoint", None),
            ),
        )
        density_was_derived = bool(
            self.__dict__.get("density_method") is None
            or self.__dict__.get("density_exact") is None
        )
        if self.__dict__.get("density_method") is None:
            self.density_method = summary.method
        if self.__dict__.get("density_exact") is None:
            self.density_exact = summary.exact
        if "density_warning_severity" not in self.__dict__ or (
            density_was_derived and self.density_warning_severity == "none"
        ):
            self.density_warning_severity = summary.severity
        if "near_power_boundary" not in self.__dict__ or density_was_derived:
            self.near_power_boundary = bool(
                self.__dict__.get("near_power_boundary", False) or summary.near_power_boundary
            )
        signatures = self.__dict__.get("_emitted_ci_density_warning_signatures")
        if not isinstance(signatures, set):
            try:
                signatures = set(()) if signatures is None else set(signatures)
            except TypeError:
                signatures = set()
            self._emitted_ci_density_warning_signatures = signatures

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore results pickled before density provenance fields existed."""
        self.__dict__.update(state)
        self._ensure_density_compat_state()

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
    _ci_details_cache: dict[float, TweedieProfileCIDetails] = field(
        default_factory=dict, repr=False
    )
    _ci_p_range: tuple[float, float] = field(default=(1.02, 1.98), repr=False)
    _ci_seed_points: tuple[float, ...] = field(default=(), repr=False)
    _evaluation_count: Any = field(default=None, repr=False)
    _evaluation_record: Any = field(default=None, repr=False)
    _emitted_ci_density_warning_signatures: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    @property
    def n_total_evaluations(self) -> int:
        """Completed distinct fixed-p records, including later CI/plot probes."""
        if self._evaluation_count is None:
            return int(self.n_evaluations)
        return max(int(self.n_evaluations), int(self._evaluation_count()))

    @property
    def n_post_search_evaluations(self) -> int:
        """Completed distinct fixed-p records added after the search snapshot."""
        return max(0, self.n_total_evaluations - int(self.n_evaluations))

    def _validate_ci_winner(self) -> None:
        """Reject likelihood-ratio inference from an invalid winning record."""
        for name in ("objective_finite", "fit_converged", "phi_converged"):
            if not bool(getattr(self, name)):
                raise RuntimeError(
                    f"Tweedie profile CI requires {name}=True for the winning "
                    f"record at p={self.p_hat:g}."
                )

    def _validate_ci_phi_method(self) -> None:
        """Require exact dispersion profiling before likelihood-ratio inference."""
        if self.phi_method != "mle":
            raise RuntimeError(
                "Tweedie likelihood-ratio profile CI requires exact MLE dispersion "
                "profiling (phi_method='mle'); use bootstrap/sandwich inference for "
                "Pearson plug-in profiles."
            )

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Profile likelihood confidence interval for Tweedie p.

        Requires that the result was produced by ``estimate_tweedie_p``.
        Results are cached so repeated calls (e.g. from summary()) are free.
        The interval targets the nearest detected connected LR component.
        Its finite max-gap scan can miss a narrower unsampled LR island.
        """
        self._validate_ci_phi_method()
        alpha_value, _, _, _, _ = _validate_profile_ci_inputs(
            self.p_hat,
            self.nll,
            self._ll_scale,
            alpha,
            self._ci_p_range,
        )
        self._validate_ci_winner()
        if alpha_value in self._ci_cache:
            return self._ci_cache[alpha_value]
        if self._objective is None:
            raise RuntimeError(
                "Profile CI requires the objective function. Use "
                "estimate_tweedie_p() to produce this result."
            )
        details = _profile_ci_p_detailed(
            self._objective,
            self.p_hat,
            self.nll,
            self._ll_scale,
            alpha=alpha_value,
            p_range=self._ci_p_range,
            seed_points=self._ci_seed_points,
            evaluation_count=self._evaluation_count,
            evaluation_record=self._evaluation_record,
        )
        density_signatures = dict(
            zip(
                getattr(details, "density_warnings", ()),
                getattr(details, "density_warning_signatures", ()),
            )
        )
        for message in details.warnings:
            signature = density_signatures.get(message)
            if signature is not None:
                already_emitted = signature in self._emitted_ci_density_warning_signatures
                if signature == "saddle:warning":
                    already_emitted |= "saddle:high" in self._emitted_ci_density_warning_signatures
                if already_emitted:
                    continue
            _warnings.warn(message, UserWarning, stacklevel=2)
            if signature is not None:
                self._emitted_ci_density_warning_signatures.add(signature)
                if signature == "saddle:high":
                    self._emitted_ci_density_warning_signatures.add("saddle:warning")
        # Preserve this exact tuple object for compatibility consumers.
        self._ci_cache[alpha_value] = details.interval
        self._ci_details_cache[alpha_value] = details
        return details.interval

    def ci_details(self, alpha: float = 0.05) -> TweedieProfileCIDetails:
        """Return immutable endpoint status and evaluation evidence for ``ci``."""
        self._validate_ci_phi_method()
        alpha_value, _, _, _, _ = _validate_profile_ci_inputs(
            self.p_hat,
            self.nll,
            self._ll_scale,
            alpha,
            self._ci_p_range,
        )
        self._validate_ci_winner()
        if alpha_value in self._ci_cache and alpha_value not in self._ci_details_cache:
            raise RuntimeError(
                "Tweedie profile CI details are unavailable for a pre-populated "
                "tuple-only cache entry."
            )
        if alpha_value not in self._ci_details_cache:
            self.ci(alpha=alpha_value)
        return self._ci_details_cache[alpha_value]

    def profile_plot(
        self,
        *,
        alpha: float = 0.05,
        n_points: int = 50,
        ax=None,
    ):
        """Profile-objective plot for Tweedie power parameter p.

        Evaluates the profile objective on a dense grid for the curve, and
        overlays the search evaluation points from ``search_trace``. MLE
        profiles use likelihood-ratio wording; Pearson profiles use neutral
        objective and interval wording.

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

        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha must be finite and strictly between 0 and 1") from exc
        if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")

        is_mle_profile = self.phi_method == "mle"
        trace_ps = self.search_trace["p"].values
        cached_interval = self._ci_cache.get(alpha_value) if is_mle_profile else None
        details: TweedieProfileCIDetails | None = None
        if cached_interval is not None:
            cached_details = self._ci_details_cache.get(alpha_value)
            if (
                cached_details is not None
                and getattr(cached_details, "alpha", None) == alpha_value
                and getattr(cached_details, "interval", None) == cached_interval
            ):
                details = cached_details
            ci_lo, ci_hi = cached_interval
            margin = 0.2 * (ci_hi - ci_lo)
            grid_lo = max(1.01, ci_lo - margin)
            grid_hi = min(1.99, ci_hi + margin)
            if len(trace_ps) > 0:
                grid_lo = min(grid_lo, float(trace_ps.min()) - 0.005)
                grid_hi = max(grid_hi, float(trace_ps.max()) + 0.005)
                grid_lo = max(1.01, grid_lo)
                grid_hi = min(1.99, grid_hi)
        else:
            support = np.append(np.asarray(trace_ps, dtype=np.float64), self.p_hat)
            support_lo = float(np.min(support))
            support_hi = float(np.max(support))
            margin = max(0.05, 0.2 * (support_hi - support_lo))
            grid_lo = max(1.01, support_lo - margin)
            grid_hi = min(1.99, support_hi + margin)
        p_grid = np.linspace(grid_lo, grid_hi, n_points)

        nll_values = np.array([self._objective(p) for p in p_grid])
        deviance = 2.0 * self._ll_scale * (nll_values - self.nll)

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

        if not is_mle_profile:
            estimate_label = f"profile estimate = {self.p_hat:.3f}"
        elif self.density_exact is None:
            estimate_label = f"profile estimate (density provenance unavailable) = {self.p_hat:.3f}"
        elif self.density_exact is False:
            estimate_label = f"approximation-based profile estimate = {self.p_hat:.3f}"
        else:
            estimate_label = f"MLE = {self.p_hat:.3f}"

        ax.axvline(
            self.p_hat,
            linestyle=":",
            color="black",
            linewidth=0.8,
            label=estimate_label,
        )
        if cached_interval is not None:
            from scipy.stats import chi2

            cutoff = float(chi2.ppf(1.0 - alpha_value, 1))
            ax.axhline(
                cutoff,
                linestyle="--",
                color="grey",
                linewidth=0.8,
                label=f"{100 * (1 - alpha_value):.0f}% cutoff",
            )
            if details is None or details.density_exact is None:
                interval_kind = "profile interval (density provenance unavailable)"
            elif self.density_exact is False or details.density_exact is False:
                interval_kind = "approximation-based LR interval"
            else:
                interval_kind = "LR interval"
            truncated = []
            if details is not None:
                truncated = [
                    side
                    for side, endpoint in (("lower", details.lower), ("upper", details.upper))
                    if endpoint.status == "truncated"
                ]
            truncation_label = ""
            if truncated:
                sides = " and ".join(truncated)
                truncation_label = f"; {sides} truncated at configured bound"
            ax.fill_betweenx(
                [0, cutoff],
                ci_lo,
                ci_hi,
                alpha=0.10,
                color="firebrick",
                label=(
                    f"{100 * (1 - alpha_value):.0f}% {interval_kind}: "
                    f"[{ci_lo:.3f}, {ci_hi:.3f}]{truncation_label}"
                ),
            )

        ax.set_xlabel("p")
        if cached_interval is not None:
            ax.set_ylabel("Profile deviance")
            ax.set_title("Tweedie p profile likelihood")
        else:
            ax.set_ylabel("Profile objective difference")
            ax.set_title("Tweedie p profile objective")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        return fig


def _build_density_messages(p: float, summary: _DensitySummary) -> list[str]:
    """Build and emit non-duplicated final-record density diagnostics."""
    messages: list[str] = []
    if summary.inconsistent:
        messages.append(
            "High-severity: inconsistent density diagnostics at "
            f"p={p:.3f} (n_saddlepoint={summary.n_saddlepoint}, "
            f"n_positive={summary.n_positive}); exact density evaluation cannot be certified."
        )
    elif _DENSITY_SEVERITY_RANK[summary.saddle_severity] >= _DENSITY_SEVERITY_RANK["warning"]:
        prefix = "High-severity: " if summary.saddle_severity == "high" else ""
        messages.append(
            f"{prefix}Saddlepoint approximation used for "
            f"{summary.n_saddlepoint}/{summary.n_positive} positive Tweedie density "
            f"terms ({summary.fraction:.0%}) at p={p:.3f}; the profile estimate is "
            "approximation-based."
        )
    if summary.near_power_boundary:
        side = "lower" if p <= _NEAR_POWER_LOWER else "upper"
        messages.append(
            "High-severity: Tweedie saddlepoint use occurs in the documented "
            f"near-power boundary instability region on the {side} side at p={p:.3f}; "
            "the inherent boundary instability is separate from numerical optimizer "
            "convergence."
        )
    for message in messages:
        _warnings.warn(message, UserWarning, stacklevel=3)
    return messages


def _fit_iteration_trace(iteration_log) -> tuple[tuple[int, float], ...]:
    """Freeze solver diagnostics into a compact frontend learning curve."""
    if not iteration_log:
        return ()
    return tuple((int(row.iteration), float(row.deviance)) for row in iteration_log)


def _reml_iteration_trace(reml_result) -> tuple[tuple[int, float], ...]:
    """Freeze REML objective history into the same compact curve shape."""
    history = getattr(reml_result, "objective_history", None)
    if not history:
        return ()
    return tuple((int(i), float(value)) for i, value in enumerate(history) if np.isfinite(value))


@dataclass(frozen=True)
class _ProfileEvaluation:
    """One complete, immutable fixed-p profile evaluation."""

    step: int
    p: float
    mu: NDArray
    edf: float
    n_iter: int
    fit_converged: bool
    source: str
    fit_trace: tuple[tuple[int, float], ...]
    fit_trace_kind: str
    phi_result: _PhiProfileResult
    solver_converged: bool | None = None
    reml_converged: bool | None = None

    @property
    def phi(self) -> float:
        """Profiled dispersion from the authoritative inner result."""
        return self.phi_result.phi

    @property
    def nll(self) -> float:
        """Mean NLL from the authoritative inner result."""
        return self.phi_result.nll


def _owned_readonly_array(values: NDArray) -> NDArray:
    """Copy an array into owning read-only storage for a cached record."""
    owned = np.array(values, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


def _phi_boundary_label(result: _PhiProfileResult) -> str:
    """Return a compact trace label for an inner dispersion boundary."""
    if result.lower_boundary and result.upper_boundary:
        return "both"
    if result.lower_boundary:
        return "lower"
    if result.upper_boundary:
        return "upper"
    return ""


def _materialize_profile_trace_row(record: _ProfileEvaluation) -> dict[str, Any]:
    """Build a fresh mutable public/callback payload from an immutable record."""
    phi_result = record.phi_result
    diagnostics = phi_result.diagnostics
    density = _classify_density_diagnostics(record.p, diagnostics)
    return {
        "step": record.step,
        "p": record.p,
        "phi": record.phi,
        "nll": record.nll,
        "n_iter": record.n_iter,
        "fit_converged": record.fit_converged,
        "solver_converged": (
            record.fit_converged if record.solver_converged is None else record.solver_converged
        ),
        "reml_converged": record.reml_converged,
        "source": record.source,
        "fit_trace": [
            {"iteration": iteration, "loss": loss} for iteration, loss in record.fit_trace
        ],
        "fit_trace_kind": record.fit_trace_kind,
        "edf": record.edf,
        "phi_converged": phi_result.converged,
        "phi_n_evaluations": phi_result.n_evaluations,
        "phi_n_score_evaluations": phi_result.n_score_evaluations,
        "phi_n_value_only_evaluations": phi_result.n_value_only_evaluations,
        "phi_n_fallback_evaluations": phi_result.n_fallback_evaluations,
        "phi_boundary": _phi_boundary_label(phi_result),
        "phi_optimizer": phi_result.optimizer,
        "phi_score": phi_result.score,
        "objective_finite": phi_result.objective_finite,
        "n_saddlepoint": density.n_saddlepoint,
        "n_positive": density.n_positive,
        "saddlepoint_fraction": density.fraction,
        "density_method": density.method,
        "density_exact": density.exact,
        "phi_used_fallback": phi_result.used_fallback,
        "phi_fallback_reason": phi_result.fallback_reason,
        "phi_branch_switch_detected": phi_result.branch_switch_detected,
        "phi_message": phi_result.message,
    }


def _profile_trace_frame(
    evaluation_cache: dict[float, _ProfileEvaluation],
) -> pd.DataFrame:
    """Materialize the immutable cache as a fresh search-trace DataFrame."""
    rows = [_materialize_profile_trace_row(record) for record in evaluation_cache.values()]
    return pd.DataFrame(rows, columns=_TRACE_COLUMNS)


def _previous_finite_phi(evaluation_cache: dict[float, _ProfileEvaluation]) -> float | None:
    """Return the most recently evaluated finite positive dispersion."""
    for record in reversed(tuple(evaluation_cache.values())):
        if np.isfinite(record.phi) and record.phi > 0.0:
            return record.phi
    return None


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
    max_iter: int = 100
    tol: float = 1e-6
    active_set: bool = False
    convergence: str = "deviance"
    trace_callback: Any = field(default=None, repr=False)
    trace_iterations: bool = False

    # Mutable warm-start state
    warm_beta: NDArray | None = field(default=None, repr=False)
    warm_intercept: float | None = field(default=None, repr=False)

    # Complete candidate cache; insertion order is the immutable search trace.
    _evaluation_cache: dict[float, _ProfileEvaluation] = field(default_factory=dict, repr=False)

    @property
    def n_evals(self) -> int:
        """Number of distinct fixed-p evaluations retained by this context."""
        return len(self._evaluation_cache)

    def evaluation_count(self) -> int:
        """Return the completed-record count for result-owned diagnostics."""
        return len(self._evaluation_cache)

    def evaluation_record(self, p: float) -> _ProfileEvaluation | None:
        """Return the authoritative completed record for an exact p key."""
        return self._evaluation_cache.get(float(p))

    def evaluate(self, p: float, source: str = "") -> float:
        """Fit at p, profile phi, record trace row, return mean NLL."""
        import time as _time

        from superglm.distributions import Tweedie

        key = float(p)
        if key in self._evaluation_cache:
            return self._evaluation_cache[key].nll

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
                max_iter=self.max_iter,
                tol=self.tol,
                record_diagnostics=self.trace_iterations,
                convergence=self.convergence,
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
                max_iter_outer=self.max_iter,
                tol=self.tol,
                active_set=self.active_set,
                lambda2=self.lambda2,
                record_diagnostics=self.trace_iterations,
                convergence=self.convergence,
            )

        eta = stabilize_eta(
            self.dm.matvec(result.beta) + result.intercept + self.offset_arr, self.link
        )
        mu = clip_mu(self.link.inverse(eta), dist)
        df_resid = max(float(len(self.y_arr)) - float(result.effective_df), 1.0)

        phi_result = _profile_phi_detailed(
            self.y_arr,
            mu,
            p,
            weights=self.w_arr,
            df_resid=df_resid,
            phi_method=self.phi_method,
            phi_start=_previous_finite_phi(self._evaluation_cache),
        )

        record = _ProfileEvaluation(
            step=self.n_evals,
            p=float(p),
            mu=_owned_readonly_array(mu),
            edf=float(result.effective_df),
            n_iter=int(result.n_iter),
            fit_converged=bool(result.converged),
            source=source,
            fit_trace=(_fit_iteration_trace(result.iteration_log) if self.trace_iterations else ()),
            fit_trace_kind="weighted deviance" if self.trace_iterations else "",
            phi_result=phi_result,
            solver_converged=bool(result.converged),
            reml_converged=None,
        )
        self._evaluation_cache[key] = record

        # Update warm starts
        self.warm_beta = result.beta
        self.warm_intercept = result.intercept

        if self.trace_callback is not None and source:
            self.trace_callback(_materialize_profile_trace_row(record))
        _elapsed = _time.perf_counter() - _t0

        logger.info(
            f"  estimate_p eval={self.n_evals:2d}  p={p:.4f}  phi={record.phi:.4f}  "
            f"nll={record.nll:.4f}  iters={result.n_iter}  {_elapsed:.2f}s"
        )
        if self.verbose:
            print(
                f"  p={p:.4f}  phi={record.phi:.4f}  nll={record.nll:.4f}  "
                f"iters={result.n_iter}  {_elapsed:.2f}s"
            )

        return record.nll

    def finalize(self, p_hat: float, method: str, converged: bool) -> TweedieProfileResult:
        """Build result with final phi at p_hat and search_trace DataFrame."""
        key = float(p_hat)
        if key not in self._evaluation_cache:
            self.evaluate(p_hat, source="final")
        record = self._evaluation_cache[key]
        return _finalize_profile_record(
            self,
            record,
            method=method,
            outer_converged=converged,
        )


def _clone_profile_model(model, X, sample_weight):
    """Clone configured profile state and resolve shorthand only on the clone."""
    profile_model = model._clone_without_features(
        set(),
        lambda2=copy.deepcopy(model.lambda2),
    )
    if model._splines is not None and not model._specs:
        # clone_without_features() normally clones resolved specs. Preserve
        # unresolved shorthand metadata and resolve it only on the scratch model.
        profile_model._splines = copy.deepcopy(model._splines)
        profile_model._n_knots = copy.deepcopy(model._n_knots)
        profile_model._degree = model._degree
        profile_model._categorical_base = model._categorical_base
        profile_model._auto_detect_features(X, sample_weight)
    return profile_model


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
    from superglm.distributions import Tweedie, validate_response

    y_arr = np.asarray(y, dtype=np.float64)

    # Profile fits and later CI probes must not rewrite the caller's fitted
    # design, resolved family/link, penalty, groups, or inference caches.
    profile_model = _clone_profile_model(model, X, sample_weight)

    # Match fit()'s fit-only lambda-policy guard before building the design.
    for name, spec in profile_model._specs.items():
        lambda_policy = getattr(spec, "_lambda_policy", None)
        if lambda_policy is not None:
            raise NotImplementedError(
                f"lambda_policy on feature '{name}' is only supported with "
                f"fit_reml(), not fit(). Use fit_reml() or remove lambda_policy."
            )

    # Temporary p so _build_design_matrix can resolve the distribution.
    # The design matrix itself doesn't depend on p.
    saved_family = profile_model.family
    profile_model.family = Tweedie(p=1.5)
    try:
        y_arr, w_arr, offset_arr = profile_model._build_design_matrix(
            X, y_arr, sample_weight, offset
        )
    finally:
        profile_model.family = saved_family

    validate_response(y_arr, profile_model._distribution)

    if profile_model.penalty.lambda1 is None:
        profile_model.penalty.lambda1 = profile_model._compute_lambda_max(y_arr, w_arr) * 0.1

    if offset_arr is None:
        offset_arr = np.zeros(len(y_arr))

    penalty = profile_model.penalty
    groups = profile_model._groups
    has_lambda1_targets = penalty_has_targets(penalty, groups)

    if (
        any(group.monotone_engine is not None for group in groups)
        and penalty.lambda1 is not None
        and penalty.lambda1 > 0
        and has_lambda1_targets
    ):
        raise NotImplementedError(
            "Monotone fit-time constraints are not supported with selection_penalty > 0. "
            "Set selection_penalty=0 or fit unconstrained and call model.monotonize()."
        )

    monotone_engines = {
        group.monotone_engine for group in groups if group.monotone_engine is not None
    }
    if len(monotone_engines) > 1:
        raise NotImplementedError("SCOP + QP monotone terms in the same model are not supported.")

    has_constraints = any(group.constraints is not None for group in groups)
    has_scop = any(group.monotone_engine == "scop" for group in groups)
    use_direct = (
        has_constraints
        or has_scop
        or (penalty.lambda1 is not None and (penalty.lambda1 == 0 or not has_lambda1_targets))
    )

    return _ProfileContext(
        y_arr=y_arr,
        w_arr=w_arr,
        offset_arr=offset_arr,
        dm=profile_model._dm,
        groups=groups,
        link=profile_model._link,
        penalty=penalty,
        use_direct=use_direct,
        lambda2=profile_model.lambda2,
        direct_solve=profile_model._direct_solve,
        max_iter=profile_model._max_iter,
        tol=profile_model._tol,
        active_set=profile_model._active_set,
        convergence=profile_model._convergence,
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

    # Complete candidate cache; insertion order is the immutable search trace.
    _evaluation_cache: dict[float, _ProfileEvaluation] = field(default_factory=dict, repr=False)

    @property
    def n_evals(self) -> int:
        """Number of distinct fixed-p evaluations retained by this context."""
        return len(self._evaluation_cache)

    def evaluation_count(self) -> int:
        """Return the completed-record count for result-owned diagnostics."""
        return len(self._evaluation_cache)

    def evaluation_record(self, p: float) -> _ProfileEvaluation | None:
        """Return the authoritative completed record for an exact p key."""
        return self._evaluation_cache.get(float(p))

    def evaluate(self, p: float, source: str = "") -> float:
        """Fit REML at p, profile phi, record trace row, return mean NLL."""
        import time as _time

        from superglm.distributions import Tweedie

        key = float(p)
        if key in self._evaluation_cache:
            return self._evaluation_cache[key].nll

        _t0 = _time.perf_counter()
        self.model.family = Tweedie(p=p)
        self.model.fit_reml(self.X, self.y, sample_weight=self.sample_weight, offset=self.offset)

        fit_mu = getattr(self.model, "_fit_mu", None)
        if (
            isinstance(fit_mu, np.ndarray)
            and fit_mu.shape == self.y.shape
            and np.all(np.isfinite(fit_mu))
            and np.all(fit_mu > 0.0)
        ):
            mu = fit_mu.copy()
        else:
            mu = np.asarray(
                self.model.predict(self.X, offset=self.offset),
                dtype=np.float64,
            )
        mu = np.maximum(mu, 1e-10)
        df_resid = max(float(len(self.y)) - float(self.model.result.effective_df), 1.0)
        phi_result = _profile_phi_detailed(
            self.y,
            mu,
            p,
            weights=self.w_arr,
            df_resid=df_resid,
            phi_method=self.phi_method,
            phi_start=_previous_finite_phi(self._evaluation_cache),
        )

        reml_result = getattr(self.model, "_reml_result", None)
        n_iter = reml_result.n_reml_iter if reml_result is not None else 0
        solver_converged = bool(getattr(self.model.result, "converged", False))
        reml_converged = (
            None if reml_result is None else bool(getattr(reml_result, "converged", False))
        )
        fit_converged = solver_converged and (
            reml_converged if reml_converged is not None else True
        )

        record = _ProfileEvaluation(
            step=self.n_evals,
            p=float(p),
            mu=_owned_readonly_array(mu),
            edf=float(self.model.result.effective_df),
            n_iter=int(n_iter),
            fit_converged=fit_converged,
            source=source,
            fit_trace=(
                _reml_iteration_trace(getattr(self.model, "_reml_result", None))
                if self.trace_iterations
                else ()
            ),
            fit_trace_kind="REML objective" if self.trace_iterations else "",
            phi_result=phi_result,
            solver_converged=solver_converged,
            reml_converged=reml_converged,
        )
        self._evaluation_cache[key] = record

        if self.trace_callback is not None and source:
            self.trace_callback(_materialize_profile_trace_row(record))
        _elapsed = _time.perf_counter() - _t0

        logger.info(
            f"  estimate_p eval={self.n_evals:2d}  p={p:.4f}  phi={record.phi:.4f}  "
            f"nll={record.nll:.4f}  reml_iters={n_iter}  {_elapsed:.2f}s"
        )
        if self.verbose:
            print(
                f"  p={p:.4f}  phi={record.phi:.4f}  nll={record.nll:.4f}  "
                f"reml_iters={n_iter}  {_elapsed:.2f}s"
            )

        return record.nll

    def finalize(self, p_hat: float, method: str, converged: bool) -> TweedieProfileResult:
        """Build result with final phi at p_hat and search_trace DataFrame."""
        key = float(p_hat)
        if key not in self._evaluation_cache:
            self.evaluate(p_hat, source="final")
        record = self._evaluation_cache[key]
        return _finalize_profile_record(
            self,
            record,
            method=method,
            outer_converged=converged,
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
    # REML profile evaluations call fit_reml(), which rewrites the fitted model
    # state. Keep that mutation inside an isolated scratch model so result.ci()
    # and profile plots cannot leave the caller's model at a probe p.
    profile_model = _clone_profile_model(model, X, sample_weight)
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


def _profile_record_is_selectable(record: _ProfileEvaluation) -> bool:
    """Whether a cached record is a usable finite profile estimate."""
    return bool(
        np.isfinite(record.p)
        and np.isfinite(record.nll)
        and np.isfinite(record.phi)
        and record.phi > 0.0
        and record.phi_result.objective_finite
    )


def _format_profile_range(bounds: tuple[float, float]) -> str:
    """Format effective search bounds for diagnostics."""
    return f"[{bounds[0]:g}, {bounds[1]:g}]"


def _best_finite_profile_record(
    ctx: _ProfileContext | _ProfileContextREML,
    *,
    method: str,
    searched_bounds: tuple[float, float],
) -> _ProfileEvaluation:
    """Return the earliest best valid cached record, or fail descriptively."""
    selectable = [
        record for record in ctx._evaluation_cache.values() if _profile_record_is_selectable(record)
    ]
    if not selectable:
        raise RuntimeError(
            f"Tweedie profile method={method!r} produced no valid result from "
            f"{len(ctx._evaluation_cache)} evaluations over p range "
            f"{_format_profile_range(searched_bounds)}; candidates require finite p/NLL, "
            "finite positive phi, and objective_finite=True."
        )
    return min(selectable, key=lambda record: record.nll)


def _outer_boundary_label(p: float, bounds: tuple[float, float] | None) -> str | None:
    """Classify an exact winning search endpoint."""
    if bounds is None:
        return None
    lo, hi = bounds
    scale = max(abs(lo), abs(hi), 1.0)
    atol = 16.0 * np.finfo(np.float64).eps * scale
    if np.isclose(lo, hi, rtol=0.0, atol=atol):
        return None
    lower = bool(np.isclose(p, lo, rtol=0.0, atol=atol))
    upper = bool(np.isclose(p, hi, rtol=0.0, atol=atol))
    if lower:
        return "lower"
    if upper:
        return "upper"
    return None


def _finalize_profile_record(
    ctx: _ProfileContext | _ProfileContextREML,
    record: _ProfileEvaluation,
    *,
    method: str,
    outer_converged: bool,
    outer_message: str = "",
    searched_bounds: tuple[float, float] | None = None,
) -> TweedieProfileResult:
    """Materialize a result directly from its immutable winning record."""
    if not _profile_record_is_selectable(record):
        raise RuntimeError(
            f"Tweedie profile method={method!r} cannot finalize invalid cached record "
            f"at p={record.p:g}; candidates require finite p/NLL, finite positive phi, "
            "and objective_finite=True."
        )
    phi_result = record.phi_result
    diagnostics = phi_result.diagnostics
    density = _classify_density_diagnostics(record.p, diagnostics)
    boundary = _outer_boundary_label(record.p, searched_bounds)
    warnings_list = _build_density_messages(record.p, density)
    if boundary:
        bounds_text = (
            _format_profile_range(searched_bounds) if searched_bounds is not None else "the search"
        )
        warnings_list.append(
            f"Profile optimum is at the {boundary} boundary of {bounds_text}; "
            "the optimum may lie outside the configured search range."
        )
    if not outer_converged:
        detail = f" ({outer_message})" if outer_message else ""
        warnings_list.append(f"Outer p search did not converge{detail}.")
    if not record.fit_converged:
        warnings_list.append("Winning fixed-p model fit did not converge.")
    if not phi_result.converged:
        warnings_list.append("Winning inner phi profile did not converge.")
    phi_boundary = _phi_boundary_label(phi_result)
    if phi_boundary:
        warnings_list.append(f"Winning phi estimate is at the {phi_boundary} dispersion boundary.")
    trace = _profile_trace_frame(ctx._evaluation_cache)
    solver_converged = (
        record.fit_converged if record.solver_converged is None else record.solver_converged
    )
    aggregate_converged = bool(
        phi_result.objective_finite
        and outer_converged
        and record.fit_converged
        and phi_result.converged
    )

    return TweedieProfileResult(
        p_hat=record.p,
        phi_hat=record.phi,
        nll=record.nll,
        n_evaluations=len(trace),
        converged=aggregate_converged,
        method=method,
        phi_method=ctx.phi_method,
        search_trace=trace,
        saddlepoint_fraction=density.fraction,
        n_saddlepoint=density.n_saddlepoint,
        n_positive=density.n_positive,
        density_method=density.method,
        density_exact=density.exact,
        density_warning_severity=density.severity,
        near_power_boundary=density.near_power_boundary,
        warnings=warnings_list,
        outer_converged=bool(outer_converged),
        outer_message=outer_message,
        outer_boundary=boundary,
        fit_converged=record.fit_converged,
        solver_converged=solver_converged,
        reml_converged=record.reml_converged,
        objective_finite=phi_result.objective_finite,
        phi_converged=phi_result.converged,
        phi_n_evaluations=phi_result.n_evaluations,
        phi_n_score_evaluations=phi_result.n_score_evaluations,
        phi_n_value_only_evaluations=phi_result.n_value_only_evaluations,
        phi_n_fallback_evaluations=phi_result.n_fallback_evaluations,
        phi_optimizer=phi_result.optimizer,
        phi_score=phi_result.score,
        phi_used_fallback=phi_result.used_fallback,
        phi_fallback_reason=phi_result.fallback_reason,
        phi_branch_switch_detected=phi_result.branch_switch_detected,
        phi_boundary=phi_boundary,
        phi_message=phi_result.message,
        _objective=ctx.evaluate,
        _ll_scale=ctx.ll_scale,
        _ci_seed_points=tuple(float(value) for value in trace["p"]),
        _evaluation_count=ctx.evaluation_count,
        _evaluation_record=ctx.evaluation_record,
    )


def _finalize_best_profile(
    ctx: _ProfileContext | _ProfileContextREML,
    *,
    method: str,
    outer_converged: bool,
    outer_message: str,
    searched_bounds: tuple[float, float],
) -> TweedieProfileResult:
    """Select and finalize the global best valid record in the search cache."""
    record = _best_finite_profile_record(
        ctx,
        method=method,
        searched_bounds=searched_bounds,
    )
    return _finalize_profile_record(
        ctx,
        record,
        method=method,
        outer_converged=outer_converged,
        outer_message=outer_message,
        searched_bounds=searched_bounds,
    )


def _evaluate_reported_candidate(
    ctx: _ProfileContext | _ProfileContextREML,
    candidate: float,
    *,
    source: str,
    bounds: tuple[float, float],
) -> tuple[bool, str]:
    """Evaluate and validate one parsed optimizer-reported candidate."""
    lo, hi = bounds
    if not lo <= candidate <= hi:
        return (
            False,
            f"optimizer result.x candidate {candidate:g} is outside applicable bounds "
            f"{_format_profile_range(bounds)}",
        )
    try:
        ctx.evaluate(float(candidate), source=source)
    except Exception as exc:
        return False, f"optimizer result.x candidate evaluation failed: {type(exc).__name__}: {exc}"
    record = ctx._evaluation_cache.get(float(candidate))
    if record is None:
        return False, "optimizer result.x candidate evaluation was not cached"
    if not _profile_record_is_selectable(record):
        return False, "optimizer result.x candidate evaluation was not finite and valid"
    return True, ""


def _parse_optimizer_result_x(result: Any) -> tuple[float | None, str]:
    """Safely parse a scalar finite optimizer ``result.x`` value."""
    if not hasattr(result, "x"):
        return None, "optimizer result.x is missing"
    try:
        values = np.asarray(result.x)
    except (TypeError, ValueError) as exc:
        return None, f"optimizer result.x could not be parsed as one scalar: {exc}"
    if values.size != 1:
        return None, f"optimizer result.x must contain one scalar, got shape {values.shape}"
    if np.iscomplexobj(values):
        return None, "optimizer result.x must contain one real scalar"
    try:
        candidate = float(values.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"optimizer result.x could not be parsed as one scalar: {exc}"
    if not np.isfinite(candidate):
        return None, "optimizer result.x must be finite"
    return candidate, ""


def _explicit_optimizer_success(result: Any) -> tuple[bool, str]:
    """Require an optimizer to report an explicit boolean success value."""
    if not hasattr(result, "success"):
        return False, "optimizer result.success is missing"
    success = result.success
    if not isinstance(success, bool | np.bool_):
        return False, "optimizer result.success must be boolean True"
    if not bool(success):
        return False, "optimizer reported success=False"
    return True, ""


def _optimizer_outer_diagnostics(result: Any, *issues: str) -> str:
    """Combine the backend message with outer-search validation failures."""
    message = str(getattr(result, "message", "") or "")
    parts = [part for part in (message, *issues) if part]
    return "; ".join(parts)


def _search_brent(
    ctx: _ProfileContext | _ProfileContextREML,
    p_bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> TweedieProfileResult:
    """Bounded scalar Brent search over p."""
    ctx.evaluate(p_bounds[0], source="brent")
    ctx.evaluate(p_bounds[1], source="brent")
    result = minimize_scalar(
        lambda p: ctx.evaluate(p, source="brent"),
        bounds=p_bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    candidate, candidate_issue = _parse_optimizer_result_x(result)
    candidate_valid = False
    if candidate is not None:
        candidate_valid, candidate_issue = _evaluate_reported_candidate(
            ctx,
            candidate,
            source="brent",
            bounds=p_bounds,
        )
    success, success_issue = _explicit_optimizer_success(result)
    converged = bool(success and candidate_valid)
    return _finalize_best_profile(
        ctx,
        method="brent",
        outer_converged=converged,
        outer_message=_optimizer_outer_diagnostics(result, success_issue, candidate_issue),
        searched_bounds=p_bounds,
    )


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

    for p in p_grid:
        ctx.evaluate(p, source="grid")
    finite_grid = p_grid[np.isfinite(p_grid)]
    searched_bounds = (
        (float(np.min(finite_grid)), float(np.max(finite_grid))) if finite_grid.size else p_bounds
    )

    return _finalize_best_profile(
        ctx,
        method="grid",
        outer_converged=True,
        outer_message="Grid search completed.",
        searched_bounds=searched_bounds,
    )


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
    for p in p_coarse:
        ctx.evaluate(p, source="grid_coarse")
    coarse_records = [ctx._evaluation_cache[float(p)] for p in p_coarse]
    selectable = [record for record in coarse_records if _profile_record_is_selectable(record)]
    if selectable:
        p_best = min(selectable, key=lambda record: record.nll).p
    else:
        finite_nll = [
            record for record in coarse_records if np.isfinite(record.p) and np.isfinite(record.nll)
        ]
        p_best = (
            min(finite_nll, key=lambda record: record.nll).p
            if finite_nll
            else float(p_coarse[len(p_coarse) // 2])
        )

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

    candidate, candidate_issue = _parse_optimizer_result_x(result)
    candidate_valid = False
    if candidate is not None:
        candidate_valid, candidate_issue = _evaluate_reported_candidate(
            ctx,
            candidate,
            source="brent_refine",
            bounds=(refine_lo, refine_hi),
        )
    success, success_issue = _explicit_optimizer_success(result)
    converged = bool(success and candidate_valid)
    return _finalize_best_profile(
        ctx,
        method="grid_refine",
        outer_converged=converged,
        outer_message=_optimizer_outer_diagnostics(result, success_issue, candidate_issue),
        searched_bounds=p_bounds,
    )


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
    for p in init_ps:
        ctx.evaluate(p, source="init")
    init_records = [ctx._evaluation_cache[float(p)] for p in init_ps]
    selectable = [record for record in init_records if _profile_record_is_selectable(record)]
    if selectable:
        best_init = min(selectable, key=lambda record: record.nll).p
    else:
        finite_nll = [
            record for record in init_records if np.isfinite(record.p) and np.isfinite(record.nll)
        ]
        best_init = (
            min(finite_nll, key=lambda record: record.nll).p if finite_nll else 0.5 * (lo + hi)
        )
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

    reported_t, candidate_issue = _parse_optimizer_result_x(result)
    candidate_valid = False
    if reported_t is not None:
        p_hat = t_to_p(reported_t)
        candidate_valid, candidate_issue = _evaluate_reported_candidate(
            ctx,
            p_hat,
            source="optimizer",
            bounds=p_bounds,
        )
    success, success_issue = _explicit_optimizer_success(result)
    converged = bool(success and candidate_valid)

    return _finalize_best_profile(
        ctx,
        method="profile_opt",
        outer_converged=converged,
        outer_message=_optimizer_outer_diagnostics(result, success_issue, candidate_issue),
        searched_bounds=p_bounds,
    )


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
    phi_method: str = "mle",
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


_CI_SCAN_SUBINTERVALS = 16
_CI_ROOT_XTOL = 1e-4
_CI_ZERO_ATOL = 1e-10
_CI_BETTER_LR_ATOL = 1e-6


class _TweedieProfileCIEvaluationError(Exception):
    """Common base for failures caused by one fixed-p CI evaluation."""


class _TweedieProfileCIEvaluationValueError(
    _TweedieProfileCIEvaluationError,
    ValueError,
):
    """ValueError-compatible fixed-p CI evaluation failure."""


class _TweedieProfileCIEvaluationRuntimeError(
    _TweedieProfileCIEvaluationError,
    RuntimeError,
):
    """RuntimeError-compatible fixed-p CI evaluation failure."""


def _validate_profile_ci_inputs(
    p_hat: float,
    nll_hat: float,
    ll_scale: float,
    alpha: float,
    p_range: tuple[float, float],
) -> tuple[float, float, float, float, tuple[float, float]]:
    """Validate and normalize scalar profile-CI inputs without evaluating it."""
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    normalized: dict[str, float] = {}
    for name, value in (("p_hat", p_hat), ("nll_hat", nll_hat), ("ll_scale", ll_scale)):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be a finite scalar")
        normalized[name] = parsed
    if normalized["ll_scale"] <= 0.0:
        raise ValueError("ll_scale must be finite and strictly positive")

    try:
        range_values = np.asarray(p_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_range must be two ordered finite bounds") from exc
    if range_values.shape != (2,) or np.iscomplexobj(range_values):
        raise ValueError("p_range must be two ordered finite bounds")
    try:
        lo, hi = (float(range_values[0]), float(range_values[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("p_range must be two ordered finite bounds") from exc
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError("p_range must be two ordered finite bounds")
    if not lo <= normalized["p_hat"] <= hi:
        raise ValueError("p_range must contain p_hat")
    return (
        alpha_value,
        normalized["p_hat"],
        normalized["nll_hat"],
        normalized["ll_scale"],
        (lo, hi),
    )


def _validate_ci_profile_record(record: Any, p: float) -> None:
    """Require an authoritative completed fixed-p record to be CI-usable."""
    if record is None:
        raise _TweedieProfileCIEvaluationRuntimeError(
            f"Tweedie profile CI did not retain a completed fixed-p record at p={p:g}."
        )
    try:
        phi_result = getattr(record, "phi_result", None)
        nll_values = np.asarray(getattr(record, "nll", np.nan))
        nll_finite = bool(
            nll_values.size == 1
            and not np.iscomplexobj(nll_values)
            and np.isfinite(float(nll_values.reshape(-1)[0]))
        )
        objective_finite = bool(
            phi_result is not None and getattr(phi_result, "objective_finite", False) and nll_finite
        )
    except Exception as exc:
        raise _TweedieProfileCIEvaluationRuntimeError(
            f"Tweedie profile CI candidate record is invalid at p={p:g}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    checks = (
        ("objective_finite", objective_finite),
        ("fit_converged", bool(getattr(record, "fit_converged", False))),
        ("phi_converged", bool(getattr(phi_result, "converged", False))),
    )
    for name, valid in checks:
        if not valid:
            raise _TweedieProfileCIEvaluationRuntimeError(
                f"Tweedie profile CI candidate has {name}=False at p={p:g}."
            )


def _ci_lr_tolerance(cutoff: float, ll_scale: float, nll: float, nll_hat: float) -> float:
    """Tolerance for profile-minimum and LR-cutoff consistency, in LR units."""
    return max(
        _CI_BETTER_LR_ATOL,
        1e-3 * cutoff,
        128.0 * np.finfo(np.float64).eps * ll_scale * max(abs(nll), abs(nll_hat), 1.0),
    )


@dataclass(frozen=True)
class _CIDensityAggregate:
    """Density provenance summarized over evaluated points in one LR component."""

    provenance: tuple[TweedieProfileCIDensityProvenance, ...]
    method: _DensityMethod | None
    exact: bool | None
    severity: _DensityWarningSeverity
    near_power_boundary: bool
    max_fraction: float
    max_p: float | None
    any_saddlepoint: bool
    n_saddlepoint_records: int
    n_invalid_density_records: int
    n_positive: int | None
    n_saddlepoint: int | None
    warnings: tuple[str, ...]
    warning_signatures: tuple[str, ...]


def _near_power_side(p: float, summary: _DensitySummary) -> str | None:
    """Return the affected near-power side only when saddlepoint terms were used."""
    if not summary.near_power_boundary:
        return None
    return "lower" if p <= _NEAR_POWER_LOWER else "upper"


def _aggregate_ci_density_provenance(
    provenance_with_summaries: list[tuple[TweedieProfileCIDensityProvenance, _DensitySummary]],
    *,
    p_hat: float,
) -> _CIDensityAggregate:
    """Aggregate connected-region records and build only new/escalated warnings."""
    if not provenance_with_summaries:
        return _CIDensityAggregate(
            provenance=(),
            method=None,
            exact=None,
            severity="none",
            near_power_boundary=False,
            max_fraction=0.0,
            max_p=None,
            any_saddlepoint=False,
            n_saddlepoint_records=0,
            n_invalid_density_records=0,
            n_positive=None,
            n_saddlepoint=None,
            warnings=(),
            warning_signatures=(),
        )

    summaries = tuple(summary for _, summary in provenance_with_summaries)
    winner_index = next(
        (index for index, (item, _) in enumerate(provenance_with_summaries) if item.p == p_hat),
        None,
    )
    valid_indices = [index for index, summary in enumerate(summaries) if not summary.inconsistent]
    if winner_index is not None and winner_index in valid_indices:
        positive_count_reference = summaries[winner_index].n_positive
    elif valid_indices:
        positive_count_reference = summaries[valid_indices[0]].n_positive
    else:
        positive_count_reference = None
    mismatched_positive_count_indices = {
        index
        for index in valid_indices
        if (
            positive_count_reference is not None
            and summaries[index].n_positive != positive_count_reference
        )
    }
    positive_count_inconsistent = bool(mismatched_positive_count_indices)
    items = tuple(
        replace(item, counts_valid=False) if index in mismatched_positive_count_indices else item
        for index, (item, _) in enumerate(provenance_with_summaries)
    )
    any_saddlepoint = any(summary.n_saddlepoint > 0 for summary in summaries)
    exact = bool(
        not positive_count_inconsistent
        and all(summary.exact and not summary.inconsistent for summary in summaries)
    )
    method: _DensityMethod
    if positive_count_inconsistent:
        method = "hybrid_exact_saddlepoint"
    elif exact:
        method = "exact"
    elif all(summary.method == "saddlepoint" for summary in summaries):
        method = "saddlepoint"
    else:
        method = "hybrid_exact_saddlepoint"

    saddle_items = [
        (item, summary)
        for item, summary in provenance_with_summaries
        if summary.n_saddlepoint > 0 or summary.inconsistent
    ]
    if saddle_items:
        max_item, max_summary = max(saddle_items, key=lambda pair: pair[1].fraction)
        max_fraction = max_summary.fraction
        max_p: float | None = max_item.p
    else:
        max_summary = _classify_density_diagnostics(
            p_hat,
            _TweedieLogpdfDiagnostics(n_positive=0, n_saddlepoint=0),
        )
        max_fraction = 0.0
        max_p = None

    near_sides = {
        side
        for item, summary in provenance_with_summaries
        if (side := _near_power_side(item.p, summary)) is not None
    }
    if winner_index is None:
        winner_summary = _classify_density_diagnostics(
            p_hat,
            _TweedieLogpdfDiagnostics(n_positive=0, n_saddlepoint=0),
        )
        winner_sides: set[str] = set()
    else:
        winner_item, winner_summary = provenance_with_summaries[winner_index]
        winner_side = _near_power_side(winner_item.p, winner_summary)
        winner_sides = set() if winner_side is None else {winner_side}

    severity: _DensityWarningSeverity = (
        "high"
        if positive_count_inconsistent
        else max(
            summaries,
            key=lambda summary: _DENSITY_SEVERITY_RANK[summary.severity],
        ).severity
    )
    warning_messages: list[str] = []
    warning_signatures: list[str] = []
    if positive_count_inconsistent:
        warning_messages.append(
            "High-severity: the evaluated LR region has inconsistent positive-response "
            "counts across authoritative fixed-p records; exact interval density evaluation "
            "cannot be certified."
        )
        warning_signatures.append("invalid:positive_count")
    if any(summary.inconsistent for summary in summaries) and not winner_summary.inconsistent:
        warning_messages.append(
            "High-severity: the evaluated LR region contains inconsistent density diagnostics; "
            "exact interval density evaluation cannot be certified."
        )
        warning_signatures.append("invalid")
    elif not positive_count_inconsistent and (
        _DENSITY_SEVERITY_RANK[max_summary.saddle_severity]
        > _DENSITY_SEVERITY_RANK[winner_summary.saddle_severity]
        and _DENSITY_SEVERITY_RANK[max_summary.saddle_severity] >= _DENSITY_SEVERITY_RANK["warning"]
    ):
        prefix = "High-severity: " if max_summary.saddle_severity == "high" else ""
        warning_messages.append(
            f"{prefix}Saddlepoint approximation in the evaluated LR region reaches "
            f"{max_summary.n_saddlepoint}/{max_summary.n_positive} positive density terms "
            f"({max_fraction:.0%}) at p={max_p:.3f}; the reported LR interval is "
            "approximation-based."
        )
        warning_signatures.append(f"saddle:{max_summary.saddle_severity}")

    new_sides = near_sides - winner_sides
    for side in sorted(new_sides):
        warning_messages.append(
            "High-severity: the evaluated LR region adds Tweedie saddlepoint use in the "
            f"documented near-power boundary instability region on the {side} side; "
            "this is separate from numerical optimizer convergence."
        )
        warning_signatures.append(f"boundary:{side}")

    n_invalid_density_records = sum(summary.inconsistent for summary in summaries) + len(
        mismatched_positive_count_indices
    )
    if n_invalid_density_records:
        n_positive: int | None = None
        n_saddlepoint: int | None = None
    else:
        n_positive = sum(summary.n_positive for summary in summaries)
        n_saddlepoint = sum(summary.n_saddlepoint for summary in summaries)

    return _CIDensityAggregate(
        provenance=items,
        method=method,
        exact=exact,
        severity=severity,
        near_power_boundary=bool(near_sides),
        max_fraction=max_fraction,
        max_p=max_p,
        any_saddlepoint=any_saddlepoint,
        n_saddlepoint_records=sum(summary.n_saddlepoint > 0 for summary in summaries),
        n_invalid_density_records=n_invalid_density_records,
        n_positive=n_positive,
        n_saddlepoint=n_saddlepoint,
        warnings=tuple(warning_messages),
        warning_signatures=tuple(warning_signatures),
    )


def _profile_ci_p_detailed(
    objective,
    p_hat: float,
    nll_hat: float,
    ll_scale: float,
    *,
    alpha: float = 0.05,
    p_range: tuple[float, float] = (1.02, 1.98),
    seed_points: tuple[float, ...] = (),
    evaluation_count=None,
    evaluation_record=None,
) -> TweedieProfileCIDetails:
    """Compute the connected LR interval with explicit endpoint semantics.

    The bounded scan walks outward from ``p_hat`` and roots the first sampled
    barrier it encounters. It fills gaps between cached search points to at
    most one sixteenth of the full configured range, avoiding a doubled
    per-side budget. As with any finite scan, a narrower unsampled likelihood
    island cannot be guaranteed detectable. ``n_new_evaluations`` counts
    completed distinct records when a context count callback is supplied.
    """
    from scipy.stats import chi2

    alpha, p_hat, nll_hat, ll_scale, p_range = _validate_profile_ci_inputs(
        p_hat, nll_hat, ll_scale, alpha, p_range
    )
    if not callable(objective):
        raise ValueError("objective must be callable")
    lo, hi = p_range
    cutoff = float(chi2.ppf(1.0 - alpha, 1))
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise RuntimeError(f"Tweedie profile CI produced an invalid LR cutoff for alpha={alpha:g}.")

    before_count = int(evaluation_count()) if evaluation_count is not None else 0
    evidence: dict[float, TweedieProfileCIEvaluation] = {}
    evidence_records: dict[float, Any] = {}

    def evaluate(p: float) -> TweedieProfileCIEvaluation:
        try:
            key = float(p)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI received an invalid p probe {p!r}: {exc}"
            ) from exc
        if key in evidence:
            return evidence[key]
        try:
            raw_nll = objective(key)
        except ValueError as exc:
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI objective failed at p={key:g}: {exc}"
            ) from exc
        except Exception as exc:
            raise _TweedieProfileCIEvaluationRuntimeError(
                f"Tweedie profile CI objective failed at p={key:g}: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            values = np.asarray(raw_nll)
            if values.size != 1 or np.iscomplexobj(values):
                raise ValueError("objective did not return one real scalar")
            nll = float(values.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI objective returned an invalid value at p={key:g}: {exc}"
            ) from exc
        if not np.isfinite(nll):
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI objective returned non-finite NLL at p={key:g}."
            )
        if evaluation_record is not None:
            try:
                record = evaluation_record(key)
            except _TweedieProfileCIEvaluationError:
                raise
            except Exception as exc:
                raise _TweedieProfileCIEvaluationRuntimeError(
                    f"Tweedie profile CI record lookup failed at p={key:g}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _validate_ci_profile_record(record, key)
            evidence_records[key] = record

        try:
            lr_statistic = float(2.0 * ll_scale * (nll - nll_hat))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI could not compute the LR statistic at p={key:g}: {exc}"
            ) from exc
        if not np.isfinite(lr_statistic):
            raise _TweedieProfileCIEvaluationValueError(
                f"Tweedie profile CI produced a non-finite LR statistic at p={key:g}."
            )
        better_tolerance = _ci_lr_tolerance(
            cutoff,
            ll_scale,
            nll,
            nll_hat,
        )
        if lr_statistic < -better_tolerance:
            raise _TweedieProfileCIEvaluationRuntimeError(
                f"Tweedie profile CI found a better profile value; rerun/expand search "
                f"(p={key:g}, LR={lr_statistic:.6g})."
            )
        point = TweedieProfileCIEvaluation(
            p=key,
            nll=nll,
            lr_statistic=lr_statistic,
        )
        evidence[key] = point
        return point

    def criterion(point: TweedieProfileCIEvaluation) -> float:
        return point.lr_statistic - cutoff

    def is_zero(value: float) -> bool:
        return bool(abs(value) <= _CI_ZERO_ATOL * max(1.0, cutoff))

    # A connected interval only depends on the path from the estimate to the
    # first LR crossing. Remote points beyond that crossing are neither part
    # of the interval nor relevant density evidence. A bound is evaluated
    # later only when the outward scan reaches it without an earlier root.
    center = evaluate(p_hat)
    center_criterion = criterion(center)
    center_tolerance = _ci_lr_tolerance(cutoff, ll_scale, center.nll, nll_hat)
    if abs(center.lr_statistic) > center_tolerance:
        raise _TweedieProfileCIEvaluationRuntimeError(
            f"Tweedie profile CI objective(p_hat) is inconsistent with nll_hat "
            f"at p={p_hat:g} (LR={center.lr_statistic:.6g}, "
            f"tolerance={center_tolerance:.6g})."
        )

    finite_seeds: set[float] = set()
    for value in seed_points:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(parsed) and lo <= parsed <= hi and parsed != p_hat:
            finite_seeds.add(parsed)

    anchors = sorted({lo, p_hat, hi} | finite_seeds)
    scan_candidates = set(anchors)
    max_gap = (hi - lo) / float(_CI_SCAN_SUBINTERVALS)
    for left, right in zip(anchors[:-1], anchors[1:]):
        ratio = (right - left) / max_gap
        roundoff = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
        n_intervals = max(1, int(np.ceil(ratio - roundoff)))
        if n_intervals > 1:
            scan_candidates.update(
                float(value)
                for value in np.linspace(left, right, n_intervals + 1, dtype=np.float64)[1:-1]
            )

    def locate(bound: float, *, side: Literal["lower", "upper"]) -> TweedieProfileCIEndpoint:
        if side == "lower":
            ordered = sorted((p for p in scan_candidates if bound <= p < p_hat), reverse=True)
        else:
            ordered = sorted(p for p in scan_candidates if p_hat < p <= bound)

        previous = center
        previous_value = center_criterion
        for candidate in ordered:
            current = evaluate(candidate)
            current_value = criterion(current)
            if is_zero(current_value):
                return TweedieProfileCIEndpoint(
                    value=current.p,
                    status="root_found",
                    at_range_boundary=bool(current.p == bound),
                    lr_statistic=current.lr_statistic,
                )
            if previous_value < 0.0 < current_value:
                bracket = tuple(sorted((previous.p, current.p)))
                root = np.nan
                root_point = None
                root_residual = np.inf
                root_tolerance = np.nan
                for attempt, xtol in enumerate((_CI_ROOT_XTOL, 1e-10), start=1):
                    try:
                        root_value, root_result = brentq(
                            lambda p: criterion(evaluate(p)),
                            bracket[0],
                            bracket[1],
                            xtol=xtol,
                            rtol=4.0 * np.finfo(np.float64).eps,
                            full_output=True,
                            disp=False,
                        )
                    except _TweedieProfileCIEvaluationError:
                        raise
                    except (ValueError, RuntimeError) as exc:
                        raise RuntimeError(
                            f"Tweedie numerical CI root failed on {side} bracket "
                            f"[{bracket[0]:g}, {bracket[1]:g}]: {exc}"
                        ) from exc

                    converged = getattr(root_result, "converged", None)
                    if not isinstance(converged, bool | np.bool_) or not bool(converged):
                        raise RuntimeError(
                            f"Tweedie numerical CI root did not converge on the {side} side."
                        )
                    try:
                        root = float(root_value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeError(
                            f"Tweedie numerical CI root was not a finite scalar on the {side} side."
                        ) from exc
                    if not np.isfinite(root):
                        raise RuntimeError(
                            f"Tweedie numerical CI root was not finite on the {side} side."
                        )
                    if not bracket[0] <= root <= bracket[1]:
                        raise RuntimeError(
                            f"Tweedie numerical CI root p={root:g} lies outside bracket "
                            f"[{bracket[0]:g}, {bracket[1]:g}] on the {side} side."
                        )

                    root_point = evaluate(root)
                    root_residual = abs(criterion(root_point))
                    root_tolerance = _ci_lr_tolerance(
                        cutoff,
                        ll_scale,
                        root_point.nll,
                        nll_hat,
                    )
                    if root_residual <= root_tolerance:
                        break
                    if attempt == 2:
                        raise RuntimeError(
                            f"Tweedie profile CI has an unresolved or discontinuous LR cutoff "
                            f"on the {side} side at p={root:g} "
                            f"(residual={root_residual:.6g}, "
                            f"tolerance={root_tolerance:.6g})."
                        )
                assert root_point is not None
                return TweedieProfileCIEndpoint(
                    value=root,
                    status="root_found",
                    at_range_boundary=bool(root == bound),
                    lr_statistic=root_point.lr_statistic,
                )
            previous = current
            previous_value = current_value

        bound_point = evaluate(bound)
        return TweedieProfileCIEndpoint(
            value=bound,
            status="truncated",
            at_range_boundary=True,
            lr_statistic=bound_point.lr_statistic,
        )

    lower = locate(lo, side="lower")
    upper = locate(hi, side="upper")
    interval = (lower.value, upper.value)
    warning_messages: list[str] = []
    for label, endpoint in (("Lower", lower), ("Upper", upper)):
        if endpoint.status == "truncated":
            warning_messages.append(
                f"{label} Tweedie profile CI is truncated at the configured p_range "
                f"boundary p={endpoint.value:g}; the LR cutoff was not reached on the "
                "connected interval."
            )

    provenance_with_summaries: list[tuple[TweedieProfileCIDensityProvenance, _DensitySummary]] = []
    if evaluation_record is not None:
        p_tolerance = 32.0 * np.finfo(np.float64).eps * max(abs(lower.value), abs(upper.value), 1.0)
        for point in evidence.values():
            if not lower.value - p_tolerance <= point.p <= upper.value + p_tolerance:
                continue
            lr_tolerance = _ci_lr_tolerance(cutoff, ll_scale, point.nll, nll_hat)
            if point.lr_statistic > cutoff + lr_tolerance:
                continue
            record = evidence_records.get(point.p)
            if record is None:
                continue
            diagnostics = getattr(getattr(record, "phi_result", None), "diagnostics", None)
            summary = _classify_density_diagnostics(point.p, diagnostics)
            provenance_with_summaries.append(
                (
                    TweedieProfileCIDensityProvenance(
                        p=point.p,
                        source=str(getattr(record, "source", "")),
                        n_positive=summary.n_positive,
                        n_saddlepoint=summary.n_saddlepoint,
                        fraction=summary.fraction,
                        method=summary.method,
                        lr_statistic=point.lr_statistic,
                        counts_valid=not summary.inconsistent,
                    ),
                    summary,
                )
            )
    density = _aggregate_ci_density_provenance(
        provenance_with_summaries,
        p_hat=p_hat,
    )
    warning_messages.extend(density.warnings)

    if evaluation_count is None:
        n_new_evaluations = len(evidence)
    else:
        after_count = int(evaluation_count())
        n_new_evaluations = max(0, after_count - before_count)
    return TweedieProfileCIDetails(
        alpha=alpha,
        cutoff=cutoff,
        p_range=p_range,
        lower=lower,
        upper=upper,
        interval=interval,
        n_new_evaluations=n_new_evaluations,
        evaluations=tuple(evidence.values()),
        warnings=tuple(warning_messages),
        density_provenance=density.provenance,
        density_method=density.method,
        density_exact=density.exact,
        density_warning_severity=density.severity,
        near_power_boundary=density.near_power_boundary,
        max_saddlepoint_fraction=density.max_fraction,
        max_saddlepoint_p=density.max_p,
        any_saddlepoint=density.any_saddlepoint,
        n_density_records=len(density.provenance),
        n_saddlepoint_records=density.n_saddlepoint_records,
        n_invalid_density_records=density.n_invalid_density_records,
        n_positive=density.n_positive,
        n_saddlepoint=density.n_saddlepoint,
        density_warnings=density.warnings,
        density_warning_signatures=density.warning_signatures,
    )


def profile_ci_p(
    objective,
    p_hat: float,
    nll_hat: float,
    ll_scale: float,
    *,
    alpha: float = 0.05,
    p_range: tuple[float, float] = (1.02, 1.98),
) -> tuple[float, float]:
    """Return the connected profile likelihood confidence interval for p.

    Use :meth:`TweedieProfileResult.ci_details` for root/truncation status and
    the immutable fixed-p evidence behind a result-owned interval. The interval
    targets the nearest detected connected LR component. Its finite max-gap
    scan can miss a narrower unsampled LR island.
    """
    details = _profile_ci_p_detailed(
        objective,
        p_hat,
        nll_hat,
        ll_scale,
        alpha=alpha,
        p_range=p_range,
    )
    for message in details.warnings:
        _warnings.warn(message, UserWarning, stacklevel=2)
    return details.interval
