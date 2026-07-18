"""Tweedie profile likelihood — estimate p from data.

For p ∈ (1, 2), the Tweedie distribution is a compound Poisson-Gamma.
This module provides multiple search strategies for estimating the power
parameter p via profile likelihood, plus certified compound-Poisson/Gamma
log-density evaluation and simulation.

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
import operator
import threading
import warnings as _warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal, Protocol
from uuid import UUID, SafeUUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logit

import superglm._tweedie_density as _tweedie_density
from superglm._tweedie_numerics import (
    PHI_LOWER_BOUND,
    TweedieNumericalError,
    _as_real_float64_array,
    _contains_masked_array,
    compound_poisson_gamma_parameters,
    normalize_boolean,
    normalize_numeric_vector,
    normalize_optional_callable,
    normalize_positive_int,
    normalize_positive_scalar,
    normalize_tweedie_bounds,
    normalize_tweedie_grid,
    normalize_tweedie_power,
    pearson_dispersion,
    tweedie_unit_deviance,
)
from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.penalties.base import penalty_has_targets
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compound Poisson-Gamma simulation
# ---------------------------------------------------------------------------

_POISSON_LAM_MAX = float(np.iinfo(np.int64).max) - 10.0 * np.sqrt(float(np.iinfo(np.int64).max))


class _CPGRNG(Protocol):
    """Structural type for the random sampling calls used by the CPG generator."""

    def poisson(self, lam: NDArray) -> Any: ...

    def gamma(self, shape: NDArray, *, scale: NDArray) -> Any: ...


def _normalize_cpg_size(n: int) -> int:
    """Return a non-negative integer sample count, excluding booleans."""
    if isinstance(n, bool | np.bool_) or _contains_masked_array(n):
        raise TypeError("n must be a non-negative integer")
    try:
        normalized = operator.index(n)
    except TypeError as exc:
        raise TypeError("n must be a non-negative integer") from exc
    if normalized < 0:
        raise ValueError("n must be non-negative")
    return normalized


def _normalize_cpg_power(p: float) -> float:
    """Return a finite real scalar power in the open interval (1, 2)."""
    if _contains_masked_array(p):
        raise ValueError("p must be a finite real numeric scalar in (1, 2)")
    try:
        raw = np.asarray(p)
    except (TypeError, ValueError) as exc:
        raise ValueError("p must be a finite real numeric scalar in (1, 2)") from exc
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError("p must be a finite real numeric scalar in (1, 2)")
    try:
        normalized = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("p must be a finite real numeric scalar in (1, 2)") from exc
    if not np.isfinite(normalized) or not 1.0 < normalized < 2.0:
        raise ValueError("p must be a finite real numeric scalar in (1, 2)")
    return normalized


def _normalize_cpg_parameter(name: str, value: float | NDArray, n: int) -> NDArray:
    """Return an owned positive float64 vector with exact shape ``(n,)``."""
    message = (
        f"{name} must be finite, strictly positive, and either a real numeric scalar "
        f"or an array with shape ({n},)"
    )
    if _contains_masked_array(value):
        raise ValueError(message)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if raw.dtype.kind not in "iuf" or (raw.ndim != 0 and (raw.ndim != 1 or raw.shape != (n,))):
        raise ValueError(message)
    normalized: NDArray
    if raw.ndim == 0:
        try:
            scalar = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(message)
        normalized = np.full(n, scalar, dtype=np.float64)
    else:
        try:
            normalized = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise ValueError(message)
    return normalized


def _resolve_cpg_rng(rng: _CPGRNG | None) -> _CPGRNG:
    """Return a generator-like object with both required sampling methods."""
    if rng is None:
        return np.random.default_rng()
    if not callable(getattr(rng, "poisson", None)) or not callable(getattr(rng, "gamma", None)):
        raise TypeError("rng must provide callable poisson and gamma methods")
    return rng


def _prepare_cpg_parameters(
    mu: NDArray,
    phi: NDArray,
    p: float,
) -> tuple[NDArray, float, NDArray]:
    """Return finite, representable compound Poisson-Gamma parameters."""
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        try:
            parameters = compound_poisson_gamma_parameters(mu, phi, p)
        except TweedieNumericalError as exc:
            raise ValueError(str(exc)) from exc
        lam = parameters.rate
        alpha = parameters.shape
        beta = parameters.scale

    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0) or np.any(lam > _POISSON_LAM_MAX):
        raise ValueError(
            "Poisson rate must be finite, strictly positive, and within NumPy's int64-safe limit"
        )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Gamma shape must be finite and strictly positive")
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0.0):
        raise ValueError("Gamma scale must be finite and strictly positive")
    return lam, alpha, beta


def _draw_cpg_counts(rng: _CPGRNG, lam: NDArray) -> NDArray:
    """Draw and structurally validate one Poisson count per rate."""
    try:
        raw_counts = rng.poisson(lam)
    except TypeError as exc:
        raise TypeError(
            "Poisson sampler call to poisson(lam) has an incompatible signature"
        ) from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError("Poisson sampler failed for validated Poisson rate parameters") from exc

    if _contains_masked_array(raw_counts):
        raise RuntimeError("Poisson sampler output must not contain a mask")
    try:
        counts = np.asarray(raw_counts)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Poisson sampler output must be an integer array") from exc
    if counts.shape != lam.shape:
        raise RuntimeError(
            f"Poisson sampler output must have shape {lam.shape}; got {counts.shape}"
        )
    if counts.dtype.kind not in "iu":
        raise RuntimeError("Poisson sampler output must have an integer dtype excluding bool")
    if np.any(counts < 0):
        raise RuntimeError("Poisson sampler output must be non-negative")
    if np.any(counts > np.iinfo(np.int64).max):
        raise RuntimeError("Poisson sampler output must fit within int64")
    return np.array(counts, dtype=np.int64, copy=True)


def _draw_cpg_positive_values(
    rng: _CPGRNG,
    counts: NDArray,
    positive: NDArray,
    alpha: float,
    beta: NDArray,
) -> NDArray:
    """Draw and validate Gamma totals for strictly positive Poisson counts."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shapes = alpha * counts[positive]
    if not np.all(np.isfinite(shapes)) or np.any(shapes <= 0.0):
        raise ValueError("Gamma shape for positive events must be finite and strictly positive")

    try:
        raw_values = rng.gamma(shapes, scale=beta[positive])
    except TypeError as exc:
        raise TypeError(
            "Gamma sampler call to gamma(shape, scale=...) has an incompatible signature"
        ) from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(
            "Gamma sampler failed for validated Gamma shape and scale parameters"
        ) from exc

    if _contains_masked_array(raw_values):
        raise RuntimeError("Gamma sampler output must not contain a mask")
    try:
        raw = np.asarray(raw_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Gamma sampler output must be a real numeric array") from exc
    expected_shape = (int(np.count_nonzero(positive)),)
    if raw.shape != expected_shape:
        raise RuntimeError(
            f"Gamma sampler output must have shape {expected_shape}; got {raw.shape}"
        )
    if raw.dtype.kind not in "iuf":
        raise RuntimeError("Gamma sampler output must have a real numeric dtype excluding bool")
    if np.any(raw < 0.0):
        raise RuntimeError("Gamma sampler output must not contain negative values")
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "Gamma sampler output must be finite; non-finite values indicate numerical overflow"
        )

    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Gamma sampler output could not be represented as finite float64") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Gamma sampler output must be finite; non-finite values indicate numerical overflow"
        )
    if np.any(values == 0.0):
        raise ValueError("Gamma sampler output underflowed to zero for a positive event")
    return values


def generate_tweedie_cpg(
    n: int,
    mu: float | NDArray,
    phi: float | NDArray,
    p: float,
    rng: _CPGRNG | None = None,
) -> NDArray:
    """Simulate Tweedie(mu, phi, p) via compound Poisson-Gamma.

    Parameters
    ----------
    n : int
        Non-negative number of samples. Booleans are not accepted.
    mu : float or array of shape (n,)
        Finite, strictly positive mean parameter. Arrays must have exact shape ``(n,)``.
    phi : float or array of shape (n,)
        Finite, strictly positive dispersion parameter. Arrays must have exact shape ``(n,)``.
    p : float
        Finite real scalar power parameter in the open interval ``(1, 2)``.
    rng : numpy Generator, optional
        Random number generator for reproducibility. A supplied object must provide callable
        ``poisson`` and ``gamma`` methods.

    Returns
    -------
    y : ndarray of shape (n,)
        Newly allocated float64 responses (non-negative, with exact zeros).
    """
    n = _normalize_cpg_size(n)
    p = _normalize_cpg_power(p)
    mu = _normalize_cpg_parameter("mu", mu, n)
    phi = _normalize_cpg_parameter("phi", phi, n)
    rng = _resolve_cpg_rng(rng)

    lam, alpha, beta = _prepare_cpg_parameters(mu, phi, p)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Vectorised: draw N ~ Poisson(lam), then Y|N ~ Gamma(alpha*N, beta)
    counts = _draw_cpg_counts(rng, lam)
    y = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    if np.any(positive):
        # Gamma additive property: sum of N iid Gamma(alpha, beta) = Gamma(N*alpha, beta)
        y[positive] = _draw_cpg_positive_values(
            rng,
            counts,
            positive,
            alpha,
            beta,
        )

    if y.shape != (n,) or y.dtype != np.dtype(np.float64):
        raise RuntimeError("Tweedie generator output must have shape (n,) and float64 dtype")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise RuntimeError("Tweedie generator output must be finite and non-negative")
    if np.any((y == 0.0) != (counts == 0)):
        raise RuntimeError(
            "Tweedie structural zeros must correspond exactly to zero Poisson counts"
        )
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
    zero_mask: NDArray
    positive_mask: NDArray
    positive_indices: NDArray


@dataclass(frozen=True)
class _TweedieDensityEvaluation:
    """One certified Tweedie density evaluation and its log-likelihood score."""

    logpdf: NDArray
    log_phi_score: NDArray
    diagnostics: _TweedieLogpdfDiagnostics
    score_valid: bool


def _readonly_copy(values: NDArray, *, dtype: Any | None = None) -> NDArray:
    """Return an owning, read-only array for an immutable evaluation record."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


class _TweedieArrayTypeError(TypeError, ValueError):
    """Strict array type error compatible with legacy value-error checks."""


def _validate_strict_tweedie_array(
    value: object,
    *,
    name: str,
    legacy_message: str,
) -> NDArray[np.float64]:
    """Reject coercive public array inputs before shared shape validation."""
    if _contains_masked_array(value):
        raise _TweedieArrayTypeError(
            f"{legacy_message}; {name} must be a real numeric array without a mask"
        )
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _TweedieArrayTypeError(
            f"{legacy_message}; {name} must be a real numeric array"
        ) from exc
    if raw.dtype.kind not in "iuf":
        raise _TweedieArrayTypeError(f"{legacy_message}; {name} must be a real numeric array")
    return _as_real_float64_array(name, raw)


def _validate_tweedie_inputs(
    y: NDArray,
    mu: NDArray,
    p: float,
    weights: NDArray | None,
) -> tuple[NDArray, NDArray, float, NDArray | None]:
    """Validate and convert common Tweedie density and dispersion inputs."""
    y_arr = _validate_strict_tweedie_array(
        y,
        name="y",
        legacy_message="y must be finite and non-negative",
    )
    mu_arr = _validate_strict_tweedie_array(
        mu,
        name="mu",
        legacy_message="mu must be finite and strictly positive",
    )
    if y_arr.ndim != 1 or mu_arr.ndim != 1 or y_arr.shape != mu_arr.shape or y_arr.size == 0:
        raise ValueError("y and mu must be one-dimensional arrays with the same non-empty shape")
    if not np.all(np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and non-negative")
    if not np.all(np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")

    try:
        p_float = normalize_tweedie_power(p)
    except ValueError as exc:
        raise ValueError("p must be finite and in the open interval (1, 2)") from exc

    validated_weights = None
    if weights is not None:
        validated_weights = _validate_strict_tweedie_array(
            weights,
            name="weights",
            legacy_message="weights must be finite and strictly positive",
        )
        if (
            validated_weights.ndim != 1
            or validated_weights.shape != y_arr.shape
            or not np.all(np.isfinite(validated_weights))
            or np.any(validated_weights <= 0.0)
        ):
            raise ValueError(
                "weights must be finite and strictly positive, one-dimensional, "
                f"and have length {len(y_arr)}"
            )
    return y_arr, mu_arr, p_float, validated_weights


def _validate_tweedie_phi(phi: float) -> float:
    """Validate and convert a scalar Tweedie dispersion parameter."""
    try:
        return normalize_positive_scalar("phi", phi)
    except ValueError as exc:
        raise ValueError("phi must be finite and strictly positive") from exc


def _prepare_tweedie_density(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
) -> _PreparedTweedieDensity:
    """Prepare fixed terms for repeated density evaluations over ``phi``."""
    y, mu, p, validated_weights = _validate_tweedie_inputs(y, mu, p, weights)
    weights_array: NDArray[np.float64]
    if validated_weights is None:
        weights_array = np.ones(len(y), dtype=np.float64)
    else:
        weights_array = validated_weights

    zero_mask = y == 0.0
    positive_mask = ~zero_mask
    positive_indices = np.flatnonzero(positive_mask)

    return _PreparedTweedieDensity(
        y=_readonly_copy(y, dtype=np.float64),
        mu=_readonly_copy(mu, dtype=np.float64),
        weights=_readonly_copy(weights_array, dtype=np.float64),
        p=p,
        zero_mask=_readonly_copy(zero_mask, dtype=np.bool_),
        positive_mask=_readonly_copy(positive_mask, dtype=np.bool_),
        positive_indices=_readonly_copy(positive_indices, dtype=np.intp),
    )


def _evaluate_tweedie_density(
    prepared: _PreparedTweedieDensity,
    phi: float,
    *,
    compute_score: bool = False,
) -> _TweedieDensityEvaluation:
    """Evaluate the shared certified kernel for prepared inputs.

    ``compute_score`` is retained for private-call compatibility.  The certified
    kernel always returns the value and score together so cache consumers never
    perform a second density pass.
    """
    del compute_score
    phi = _validate_tweedie_phi(phi)
    evaluation = _tweedie_density.evaluate_tweedie_density(
        prepared.y,
        prepared.mu,
        phi,
        prepared.p,
        weights=prepared.weights,
    )
    diagnostics = _TweedieLogpdfDiagnostics(
        n_positive=evaluation.diagnostics.n_positive,
        n_saddlepoint=evaluation.diagnostics.n_approximate,
    )
    return _TweedieDensityEvaluation(
        logpdf=evaluation.logpdf,
        log_phi_score=evaluation.log_phi_score,
        diagnostics=diagnostics,
        score_valid=bool(np.all(np.isfinite(evaluation.log_phi_score))),
    )


def _tweedie_logpdf_impl(
    y: NDArray,
    mu: NDArray,
    phi: float,
    p: float,
    *,
    weights: NDArray | None = None,
) -> tuple[NDArray, _TweedieLogpdfDiagnostics]:
    """Compatibility wrapper over the shared certified density evaluator."""
    prepared = _prepare_tweedie_density(
        y,
        mu,
        p,
        weights=weights,
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
) -> NDArray:
    """Certified exact Tweedie log-density.

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
    )
    return logpdf


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
    try:
        strict_p = normalize_tweedie_power(p)
    except ValueError as exc:
        raise ValueError("p must be finite and in the open interval (1, 2)") from exc
    strict_y = _validate_strict_tweedie_array(
        y,
        name="y",
        legacy_message="y must be finite and non-negative",
    )
    strict_mu = _validate_strict_tweedie_array(
        mu,
        name="mu",
        legacy_message="mu must be finite and strictly positive",
    )
    strict_weights = (
        None
        if weights is None
        else _validate_strict_tweedie_array(
            weights,
            name="weights",
            legacy_message="weights must be finite and strictly positive",
        )
    )
    y, mu, p, weights = _validate_tweedie_inputs(
        strict_y,
        strict_mu,
        strict_p,
        strict_weights,
    )
    denominator = len(y) if df_resid is None else df_resid
    return pearson_dispersion(y, mu, p, weights, denominator)


_PHI_LOWER_BOUND = PHI_LOWER_BOUND
_PHI_UPPER_BOUND = 1e12
_LOG_PHI_LOWER_BOUND = float(np.log(_PHI_LOWER_BOUND))
_LOG_PHI_UPPER_BOUND = float(np.log(_PHI_UPPER_BOUND))
_PHI_SCORE_TOLERANCE = 1e-6
_PHI_ROOT_PROBE = 1e-5
_PHI_BOUNDED_XATOL = 1e-6
_PHI_NLL_ATOL = 1e-10
_PHI_NLL_RTOL = 1e-10


@dataclass(frozen=True)
class _PhiProfileResult:
    """Detailed result from profiling Tweedie dispersion at fixed ``(mu, p)``.

    Evaluation counters describe actual kernel outputs, not which optimizer
    requested them.  The certified kernel returns a score with every density
    pass, so current exact profiles report every pass in
    ``n_score_evaluations`` and none in ``n_value_only_evaluations``.
    ``n_fallback_evaluations`` separately records value-objective optimizer
    provenance.
    """

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
class _PhiProfilePoint:
    """One cached exact objective point, keyed by its exact float ``u``."""

    u: float
    phi: float
    nll: float
    objective_finite: bool
    score: float | None
    score_attempted: bool
    score_valid: bool
    diagnostics: _TweedieLogpdfDiagnostics


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
    """Abort a score root as soon as its analytic score/root becomes untrustworthy."""


class _PhiEvaluationCache:
    """Cache exact passes and report the value and score work each pass performs."""

    def __init__(self, prepared: _PreparedTweedieDensity):
        self.prepared = prepared
        self.points: dict[float, _PhiProfilePoint] = {}
        self.n_evaluations = 0
        self.n_score_evaluations = 0
        self.n_value_only_evaluations = 0
        self.n_fallback_evaluations = 0
        self.density_errors: dict[float, _tweedie_density.TweedieDensityError] = {}

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
        if cached is not None:
            return cached

        if phi_override is not None:
            phi = float(phi_override)
        elif key == _LOG_PHI_LOWER_BOUND:
            phi = _PHI_LOWER_BOUND
        elif key == _LOG_PHI_UPPER_BOUND:
            phi = _PHI_UPPER_BOUND
        else:
            phi = float(np.exp(key))
        self.n_evaluations += 1
        del compute_score
        # The certified series always evaluates and certifies both outputs.
        # Fallback provenance is counted separately below; there is no cheaper
        # value-only kernel pass to report.
        self.n_score_evaluations += 1
        if fallback:
            self.n_fallback_evaluations += 1

        try:
            evaluation = _evaluate_tweedie_density(self.prepared, phi)
        except _tweedie_density.TweedieDensityError as exc:
            self.density_errors[key] = exc
            point = _PhiProfilePoint(
                u=key,
                phi=phi,
                nll=np.inf,
                objective_finite=False,
                score=None,
                score_attempted=True,
                score_valid=False,
                diagnostics=_TweedieLogpdfDiagnostics(
                    n_positive=int(self.prepared.positive_indices.size),
                    n_saddlepoint=0,
                ),
            )
            self.points[key] = point
            return point

        with np.errstate(all="ignore"):
            nll = -float(np.mean(evaluation.logpdf))
        objective_finite = bool(np.isfinite(nll))
        score: float | None = None
        score_valid = evaluation.score_valid
        if score_valid:
            with np.errstate(all="ignore"):
                candidate_score = -float(np.mean(evaluation.log_phi_score))
            if np.isfinite(candidate_score):
                score = candidate_score
            else:
                score_valid = False

        point = _PhiProfilePoint(
            u=key,
            phi=phi,
            nll=nll if objective_finite else np.inf,
            objective_finite=objective_finite,
            score=score,
            score_attempted=True,
            score_valid=score_valid,
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
    return pearson_dispersion(
        prepared.y,
        prepared.mu,
        prepared.p,
        prepared.weights,
        denominator,
    )


def _phi_profile_seeds(
    prepared: _PreparedTweedieDensity,
    denominator: float,
    pearson_phi: float,
    phi_start: float | None,
) -> list[tuple[float, str]]:
    """Build distinct interior warm, Pearson, and stable data seeds in priority order."""
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
        u = float(np.log(candidate))
        if not _LOG_PHI_LOWER_BOUND < u < _LOG_PHI_UPPER_BOUND:
            return False
        if any(abs(u - existing_u) <= _PHI_BOUNDED_XATOL for existing_u, _ in seeds):
            return False
        seeds.append((u, source))
        return True

    add(phi_start, "warm start")
    pearson_usable = add(pearson_phi, "Pearson seed")
    if not pearson_usable:
        deviance = tweedie_unit_deviance(
            prepared.y,
            prepared.mu,
            prepared.p,
        )
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
                and left_probe.score is not None
                and right_probe.score is not None
                and left_probe.score < 0.0 < right_probe.score
            ):
                add_bracket(left_probe, right_probe)
                found_local_bracket = True
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

        def score_callback(u: float) -> float:
            point = cache.evaluate(float(u), compute_score=True)
            if not point.objective_finite:
                raise _PhiRootAbortError("exact objective became non-finite inside brentq")
            if not point.score_valid or point.score is None:
                raise _PhiRootAbortError("analytic derivative became unavailable inside brentq")
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
        except _tweedie_density.TweedieDensityError:
            raise
        except (RuntimeError, ValueError) as exc:
            _record_phi_fallback(fallback_reasons, f"brentq failed: {exc}")
            continue

        root_point = cache.evaluate(float(root_u), compute_score=True)
        valid_root = bool(root_info.converged)
        valid_root &= root_point.objective_finite
        valid_root &= root_point.score_valid and root_point.score is not None
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
    except _tweedie_density.TweedieDensityError:
        raise
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        return _PhiBoundedResult(candidate=None, success=False, message=str(exc))


def _run_phi_bounded_fallback(
    cache: _PhiEvaluationCache,
    *,
    required: bool,
) -> _PhiBoundedResult:
    """Run one value-only exact rescue when analytic score search is unavailable."""
    if not required:
        return _PhiBoundedResult(candidate=None, success=True, message="")
    return _run_phi_bounded_interval(
        cache,
        (_LOG_PHI_LOWER_BOUND, _LOG_PHI_UPPER_BOUND),
    )


def _competitive_phi_boundary_point(
    cache: _PhiEvaluationCache,
    bounded: _PhiBoundedResult,
    boundary_u: float,
) -> _PhiProfilePoint | None:
    """Return an already reached or bounded-search-competitive hard boundary."""
    cached = cache.points.get(boundary_u)
    if cached is not None:
        if error := cache.density_errors.get(boundary_u):
            raise error
        return cached
    if (
        bounded.candidate is not None
        and abs(bounded.candidate.point.u - boundary_u) <= 4.0 * _PHI_BOUNDED_XATOL
    ):
        point = cache.evaluate(boundary_u, compute_score=True)
        if error := cache.density_errors.get(boundary_u):
            raise error
        return point
    return None


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
    need_fallback = bool(fallback_reasons)
    lower_point = _competitive_phi_boundary_point(
        cache,
        bounded,
        _LOG_PHI_LOWER_BOUND,
    )
    upper_point = _competitive_phi_boundary_point(
        cache,
        bounded,
        _LOG_PHI_UPPER_BOUND,
    )
    candidates = [*score_search.seed_candidates, *score_search.root_candidates]
    if bounded.candidate is not None:
        candidates.append(bounded.candidate)
    if lower_point is not None and lower_point.objective_finite:
        candidates.append(_PhiCandidate(lower_point, "lower boundary", validated=True))
    if upper_point is not None and upper_point.objective_finite:
        candidates.append(_PhiCandidate(upper_point, "upper boundary", validated=True))
    finite_candidates = [candidate for candidate in candidates if candidate.point.objective_finite]

    if not finite_candidates:
        if cache.density_errors:
            raise next(iter(cache.density_errors.values()))
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
        and lower_point is not None
        and lower_point.objective_finite
        and _phi_nll_no_worse(lower_point.nll, best.point.nll)
    ):
        best = _PhiCandidate(lower_point, "lower boundary", validated=True)
    elif (
        _LOG_PHI_UPPER_BOUND - best.point.u <= 4.0 * _PHI_BOUNDED_XATOL
        and upper_point is not None
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
        # Preserve the best certified value without claiming an interior
        # derivative optimum from a value-only rescue.
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
        phi_hat = max(pearson_phi, PHI_LOWER_BOUND)
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
        if phi_hat == PHI_LOWER_BOUND:
            return _PhiProfileResult(
                phi=phi_hat,
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
                lower_boundary=True,
                upper_boundary=False,
                diagnostics=default_diagnostics,
                message=(
                    "Pearson plug-in dispersion is at the hard lower boundary; "
                    "the concentrated exact density was not evaluated."
                ),
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


_TWEEDIE_PROFILE_RESULT_TOKEN = object()


@dataclass
class _TweedieProfileEvaluator:
    """One detachable owner for all post-search profile operations."""

    context: Any = field(repr=False)
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def evaluate(self, p: float, source: str = "") -> float:
        """Evaluate and retain one fixed-power profile record."""
        with self._lock:
            return self.context.evaluate(p, source=source)

    def evaluation_count(self) -> int:
        """Return the number of completed distinct profile records."""
        with self._lock:
            return int(self.context.evaluation_count())

    def evaluation_record(self, p: float):
        """Return the authoritative completed record at an exact power."""
        with self._lock:
            return self.context.evaluation_record(p)


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
    _validation_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _evaluator: _TweedieProfileEvaluator | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _frozen_evaluation_count: int | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        """Derive new density fields for legacy positional construction."""
        self._ensure_density_compat_state()

    def _ensure_density_compat_state(self) -> None:
        """Restore density fields absent from legacy construction or pickle state."""
        legacy_n_positive: Any = getattr(self, "n_positive", None)
        legacy_n_saddlepoint: Any = getattr(self, "n_saddlepoint", None)
        summary = _classify_density_diagnostics(
            getattr(self, "p_hat", np.nan),
            _TweedieLogpdfDiagnostics(
                n_positive=legacy_n_positive,
                n_saddlepoint=legacy_n_saddlepoint,
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
        """Restore a detached result without reviving row-owning callbacks."""
        self.__dict__.update(state)
        self._ensure_density_compat_state()
        if "_ci_details_cache" not in self.__dict__:
            self._ci_details_cache = {}
        frozen_count = self.__dict__.get("_frozen_evaluation_count")
        if type(frozen_count) is not int or frozen_count < int(self.n_evaluations):
            self._frozen_evaluation_count = int(self.n_evaluations)
        self._evaluator = None
        self._objective = None
        self._evaluation_count = None
        self._evaluation_record = None

    def __getstate__(self) -> dict[str, Any]:
        """Serialize a detached copy of state without racing a live profile operation."""
        evaluator = self.__dict__.get("_evaluator")
        if type(evaluator) is _TweedieProfileEvaluator:
            with evaluator._lock:
                return self._detached_pickle_state()
        return self._detached_pickle_state()

    def _detached_pickle_state(self) -> dict[str, Any]:
        """Build serialization state while the evaluator lock is held, when live."""
        state = self.__dict__.copy()
        evaluator = state.get("_evaluator")
        if type(evaluator) is _TweedieProfileEvaluator:
            try:
                frozen_count = evaluator.evaluation_count()
            except Exception:
                frozen_count = int(self.n_evaluations)
        else:
            existing = state.get("_frozen_evaluation_count")
            frozen_count = (
                existing
                if type(existing) is int and existing >= int(self.n_evaluations)
                else int(self.n_evaluations)
            )
        state["_frozen_evaluation_count"] = max(int(self.n_evaluations), int(frozen_count))
        state["_evaluator"] = None
        state["_objective"] = None
        state["_evaluation_count"] = None
        state["_evaluation_record"] = None
        # Only these library-owned containers can change during a guarded
        # profile operation. Snapshot them while holding the evaluator lock so
        # serialization observes one coherent cache state without recursively
        # copying arbitrary user-attached result attributes.
        state["_ci_cache"] = dict(state.get("_ci_cache", {}))
        state["_ci_details_cache"] = dict(state.get("_ci_details_cache", {}))
        state["_emitted_ci_density_warning_signatures"] = set(
            state.get("_emitted_ci_density_warning_signatures", set())
        )
        return state

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

    def _profile_evaluator_callbacks(self):
        """Resolve modern evaluator callbacks or legacy constructor fields."""
        if type(self._evaluator) is _TweedieProfileEvaluator:
            return (
                self._evaluator.evaluate,
                self._evaluator.evaluation_count,
                self._evaluator.evaluation_record,
            )
        return self._objective, self._evaluation_count, self._evaluation_record

    @contextmanager
    def _profile_operation_guard(self):
        """Serialize multi-evaluation operations against detachment and peers."""
        evaluator = self._evaluator
        if type(evaluator) is _TweedieProfileEvaluator:
            with evaluator._lock:
                if self._evaluator is not evaluator:
                    raise RuntimeError("Tweedie profile evaluator was detached concurrently")
                yield
            return
        yield

    def detach_evaluator(self) -> None:
        """Release all profile training rows while preserving completed caches."""
        evaluator = self._evaluator
        if type(evaluator) is _TweedieProfileEvaluator:
            with evaluator._lock:
                self._detach_evaluator_locked(evaluator)
            return
        self._detach_evaluator_locked(None)

    def _detach_evaluator_locked(self, evaluator: _TweedieProfileEvaluator | None) -> None:
        """Detach after excluding live operations on the supplied evaluator."""
        if evaluator is not None:
            try:
                frozen_count = evaluator.evaluation_count()
            except Exception:
                frozen_count = int(self.n_evaluations)
            context = evaluator.context
            if hasattr(context, "trace_callback"):
                context.trace_callback = None
        elif type(self._frozen_evaluation_count) is int:
            frozen_count = self._frozen_evaluation_count
        else:
            # Do not execute arbitrary legacy callbacks during detachment.
            frozen_count = int(self.n_evaluations)
        self._frozen_evaluation_count = max(int(self.n_evaluations), int(frozen_count))
        self._evaluator = None
        self._objective = None
        self._evaluation_count = None
        self._evaluation_record = None

    @property
    def n_total_evaluations(self) -> int:
        """Completed distinct fixed-p records, including later CI/plot probes."""
        evaluator = self._evaluator
        if type(evaluator) is _TweedieProfileEvaluator:
            with evaluator._lock:
                if self._evaluator is not evaluator:
                    return max(
                        int(self.n_evaluations),
                        self._frozen_evaluation_count
                        if type(self._frozen_evaluation_count) is int
                        else 0,
                    )
                return self._n_total_evaluations_unlocked()
        return self._n_total_evaluations_unlocked()

    def _n_total_evaluations_unlocked(self) -> int:
        """Update the count high-water mark while any live evaluator lock is held."""
        _, evaluation_count, _ = self._profile_evaluator_callbacks()
        frozen_count = (
            self._frozen_evaluation_count
            if type(self._frozen_evaluation_count) is int
            else int(self.n_evaluations)
        )
        if callable(evaluation_count):
            observed_count = int(evaluation_count())
            frozen_count = max(int(self.n_evaluations), frozen_count, observed_count)
            self._frozen_evaluation_count = frozen_count
        return max(int(self.n_evaluations), frozen_count)

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
        """Return the nearest detected connected profile-likelihood interval.

        Live evaluator use is serialized. Its finite scan can miss a narrower unsampled
        likelihood-ratio island.
        """
        with self._profile_operation_guard():
            return self._ci_locked(alpha)

    def _ci_locked(self, alpha: float = 0.05) -> tuple[float, float]:
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
        objective, evaluation_count, evaluation_record = self._profile_evaluator_callbacks()
        if not callable(objective):
            raise RuntimeError(
                "Tweedie profile evaluator is detached, whether explicitly, after fit-state "
                "release, or after serialization. Cached intervals remain available; for an "
                "uncached interval, pass eager_ci_alpha=<alpha> to SuperGLM.estimate_p(...) "
                "or use retain_fit_state=True before fitting, and call ci(...) before "
                "serialization."
            )
        details = _profile_ci_p_detailed(
            objective,
            self.p_hat,
            self.nll,
            self._ll_scale,
            alpha=alpha_value,
            p_range=self._ci_p_range,
            seed_points=self._ci_seed_points,
            evaluation_count=evaluation_count,
            evaluation_record=evaluation_record,
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
        """Return cached endpoint evidence, serializing live evaluator use."""
        with self._profile_operation_guard():
            return self._ci_details_locked(alpha)

    def _ci_details_locked(self, alpha: float = 0.05) -> TweedieProfileCIDetails:
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

    def trace_plot(self, *, ax=None):
        """Plot cached Tweedie p-search evaluations without fitting new models."""
        import matplotlib.pyplot as plt

        from superglm.plotting.common import (
            _LINE_COLOR,
            _LINE_WIDTH,
            _PW_FILL,
            _REF_COLOR,
            _REF_LW,
        )

        try:
            trace_p = np.asarray(self.search_trace["p"], dtype=np.float64)
            trace_nll = np.asarray(self.search_trace["nll"], dtype=np.float64)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Tweedie search trace must contain numeric p/nll columns") from exc
        if trace_p.ndim != 1 or trace_nll.ndim != 1 or trace_p.shape != trace_nll.shape:
            raise RuntimeError(
                "Tweedie search trace must contain one-dimensional numeric p/nll columns "
                "of equal length"
            )

        finite = np.isfinite(trace_p) & np.isfinite(trace_nll)
        if not np.any(finite):
            raise RuntimeError("Tweedie search trace contains no finite p/nll evaluations")

        finite_p = trace_p[finite]
        finite_nll = trace_nll[finite]
        order = np.argsort(finite_p, kind="stable")
        plotted_p = finite_p[order]
        plotted_difference = 2.0 * self._ll_scale * (finite_nll[order] - self.nll)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.get_figure()

        ax.plot(
            plotted_p,
            plotted_difference,
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            marker="o",
            markersize=5.5,
            markerfacecolor=_PW_FILL,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"Search evaluations ({len(plotted_p)})",
            zorder=4,
        )
        ax.axvline(
            self.p_hat,
            linestyle=":",
            color=_REF_COLOR,
            linewidth=_REF_LW,
            label=rf"$\hat{{p}}$ = {self.p_hat:.3f}",
        )
        ax.set_xlabel("p")
        ax.set_ylabel(
            "Profile deviance" if self.phi_method == "mle" else "Profile objective difference"
        )
        ax.set_title("Tweedie p profile search trace")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, loc="upper right")
        return fig

    def profile_plot(
        self,
        *,
        alpha: float = 0.05,
        n_points: int = 50,
        ax=None,
    ):
        """Plot a dense profile curve while serializing live evaluator use."""
        with self._profile_operation_guard():
            return self._profile_plot_locked(alpha=alpha, n_points=n_points, ax=ax)

    def _profile_plot_locked(
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
        objective, _, _ = self._profile_evaluator_callbacks()
        if not callable(objective):
            raise RuntimeError(
                "Tweedie profile evaluator is detached, whether explicitly, after fit-state "
                "release, or after serialization. Dense profile plots require a live evaluator; "
                "call profile_plot() before serialization or retain fit state before release. "
                "trace_plot() remains available from the immutable search trace."
            )

        if _contains_masked_array(n_points):
            raise ValueError("n_points must be an unmasked integer")
        n_points_value = normalize_positive_int(
            n_points,
            name="n_points",
            minimum=2,
            maximum=_MAX_PROFILE_GRID_POINTS,
        )

        import matplotlib.pyplot as plt

        if _contains_masked_array(alpha):
            raise ValueError("alpha must be finite and strictly between 0 and 1")
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
            support: NDArray[np.float64] = np.append(
                np.asarray(trace_ps, dtype=np.float64), self.p_hat
            )
            support_lo = float(np.min(support))
            support_hi = float(np.max(support))
            margin = max(0.05, 0.2 * (support_hi - support_lo))
            grid_lo = max(1.01, support_lo - margin)
            grid_hi = min(1.99, support_hi + margin)
        p_grid = np.linspace(grid_lo, grid_hi, n_points_value)

        raw_nll_values = []
        for p in p_grid:
            raw_nll = objective(p)
            if _contains_masked_array(raw_nll):
                raise ValueError("Tweedie profile plot objective values must not contain a mask")
            raw_nll_values.append(raw_nll)
        nll_values = np.array(raw_nll_values)
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
    profile_x: Any = field(default=None, repr=False, kw_only=True)
    profile_y: Any = field(default=None, repr=False, kw_only=True)
    profile_sample_weight: Any = field(default=None, repr=False, kw_only=True)
    profile_offset: Any = field(default=None, repr=False, kw_only=True)

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
    from superglm.distributions import Tweedie
    from superglm.model.profile_ops import (
        _validate_tweedie_profile_clone_isolation,
        _validate_tweedie_profile_copy_protocols,
    )

    _validate_tweedie_profile_copy_protocols(model)
    profile_model = model._clone_without_features(
        set(),
        lambda2=copy.deepcopy(model.lambda2),
    )
    profile_model.family = Tweedie(p=model.family.p)
    profile_model.link = copy.deepcopy(model.link)
    profile_model._interaction_specs = copy.deepcopy(model._interaction_specs)
    profile_model._interaction_order = list(model._interaction_order)
    profile_model._pending_interactions = copy.deepcopy(model._pending_interactions)
    profile_model._splines = copy.deepcopy(model._splines)
    profile_model._n_knots = copy.deepcopy(model._n_knots)
    profile_model._n_bins = copy.deepcopy(model._n_bins)
    profile_model._degree = model._degree
    profile_model._categorical_base = model._categorical_base
    if model._splines is not None and not model._specs:
        # clone_without_features() normally clones resolved specs. Preserve
        # unresolved shorthand metadata and resolve it only on the scratch model.
        profile_model._auto_detect_features(X, sample_weight)
    _validate_tweedie_profile_clone_isolation(model, profile_model)
    return profile_model


def _snapshot_profile_inputs(X, y, sample_weight, offset):
    """Own the inputs retained by profile contexts and their lazy probes."""
    return (
        _snapshot_tweedie_profile_dataframe(X),
        np.array(y, dtype=np.float64, copy=True),
        (None if sample_weight is None else np.array(sample_weight, dtype=np.float64, copy=True)),
        None if offset is None else np.array(offset, dtype=np.float64, copy=True),
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
    *,
    _inputs_owned: bool = False,
) -> _ProfileContext:
    """One-time setup: build design matrix, calibrate lambda, create context."""
    from superglm.distributions import Tweedie, validate_response

    if _inputs_owned:
        X_snapshot, y_arr, weight_snapshot, offset_snapshot = X, y, sample_weight, offset
    else:
        X_snapshot, y_arr, weight_snapshot, offset_snapshot = _snapshot_profile_inputs(
            X,
            y,
            sample_weight,
            offset,
        )
    y_snapshot = y_arr

    # Profile fits and later CI probes must not rewrite the caller's fitted
    # design, resolved family/link, penalty, groups, or inference caches.
    profile_model = _clone_profile_model(model, X_snapshot, weight_snapshot)

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
            X_snapshot,
            y_arr,
            weight_snapshot,
            offset_snapshot,
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
        profile_x=X_snapshot,
        profile_y=y_snapshot,
        profile_sample_weight=weight_snapshot,
        profile_offset=offset_snapshot,
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
    *,
    _inputs_owned: bool = False,
) -> _ProfileContextREML:
    """Build context for REML-based profile estimation."""
    if _inputs_owned:
        X_snapshot, y_snapshot, weight_snapshot, offset_snapshot = X, y, sample_weight, offset
    else:
        X_snapshot, y_snapshot, weight_snapshot, offset_snapshot = _snapshot_profile_inputs(
            X,
            y,
            sample_weight,
            offset,
        )

    # REML profile evaluations call fit_reml(), which rewrites the fitted model
    # state. Keep that mutation inside an isolated scratch model so result.ci()
    # and profile plots cannot leave the caller's model at a probe p.
    profile_model = _clone_profile_model(model, X_snapshot, weight_snapshot)
    if getattr(model, "_last_fit_meta", None) is not None:
        profile_model._last_fit_meta = dict(model._last_fit_meta)

    w_arr = weight_snapshot if weight_snapshot is not None else np.ones(len(y_snapshot))
    return _ProfileContextREML(
        model=profile_model,
        X=X_snapshot,
        y=y_snapshot,
        sample_weight=weight_snapshot,
        offset=offset_snapshot,
        w_arr=w_arr,
        phi_method=phi_method,
        verbose=verbose,
        ll_scale=float(len(y_snapshot)),
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
        _ll_scale=ctx.ll_scale,
        _ci_seed_points=tuple(float(value) for value in trace["p"]),
        _evaluator=_TweedieProfileEvaluator(ctx),
        _validation_token=_TWEEDIE_PROFILE_RESULT_TOKEN,
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


_MAX_PROFILE_INTEGER_CONTROL = int(np.iinfo(np.intp).max)
# Every grid point runs a complete model fit and dispersion profile.  This is
# intentionally generous while preventing accidental allocation/runaway work.
_MAX_PROFILE_GRID_POINTS = 10_000
_PREPARED_TWEEDIE_PROFILE_TOKEN = object()


@dataclass(frozen=True)
class _PreparedTweedieProfileInputs:
    """Validated controls and owned numeric rows for one profile call."""

    X: pd.DataFrame
    y: NDArray[np.float64]
    sample_weight: NDArray[np.float64] | None
    offset: NDArray[np.float64] | None
    p_bounds: tuple[float, float]
    xatol: float
    maxiter: int
    verbose: bool
    fit_mode: str
    phi_method: str
    method: str
    n_grid: int
    grid: NDArray[np.float64] | None
    n_grid_coarse: int
    optimizer: str
    trace_callback: Any
    trace_iterations: bool
    _model_identity: int = field(repr=False)
    _validation_token: object = field(repr=False, compare=False)


_PREPARED_TWEEDIE_PROFILE_CALL: ContextVar[_PreparedTweedieProfileInputs | None] = ContextVar(
    "_PREPARED_TWEEDIE_PROFILE_CALL", default=None
)


@contextmanager
def _use_prepared_tweedie_profile_inputs(prepared: _PreparedTweedieProfileInputs):
    """Offer one prepared snapshot to the next matching public estimator call."""
    token = _PREPARED_TWEEDIE_PROFILE_CALL.set(prepared)
    try:
        yield
    finally:
        _PREPARED_TWEEDIE_PROFILE_CALL.reset(token)


def _claim_prepared_tweedie_profile_inputs(
    model,
    X,
    y,
    sample_weight,
    offset,
    p_bounds,
    xatol,
    maxiter,
    verbose,
    fit_mode,
    phi_method,
    method,
    n_grid,
    grid,
    n_grid_coarse,
    optimizer,
    trace_callback,
    trace_iterations,
) -> _PreparedTweedieProfileInputs | None:
    """Consume the context-local snapshot only for its exact public call."""
    prepared = _PREPARED_TWEEDIE_PROFILE_CALL.get()
    if prepared is None:
        return None
    actual = (
        X,
        y,
        sample_weight,
        offset,
        p_bounds,
        xatol,
        maxiter,
        verbose,
        fit_mode,
        phi_method,
        method,
        n_grid,
        grid,
        n_grid_coarse,
        optimizer,
        trace_callback,
        trace_iterations,
    )
    expected = (
        prepared.X,
        prepared.y,
        prepared.sample_weight,
        prepared.offset,
        prepared.p_bounds,
        prepared.xatol,
        prepared.maxiter,
        prepared.verbose,
        prepared.fit_mode,
        prepared.phi_method,
        prepared.method,
        prepared.n_grid,
        prepared.grid,
        prepared.n_grid_coarse,
        prepared.optimizer,
        prepared.trace_callback,
        prepared.trace_iterations,
    )
    if (
        prepared._validation_token is _PREPARED_TWEEDIE_PROFILE_TOKEN
        and prepared._model_identity == id(model)
        and all(value is expected_value for value, expected_value in zip(actual, expected))
    ):
        # Remove it before profiling can invoke a callback and re-enter this API.
        _PREPARED_TWEEDIE_PROFILE_CALL.set(None)
        return prepared
    return None


def _normalize_profile_choice(value: object, *, name: str, choices: set[str]) -> str:
    """Return one supported string choice with a stable public error."""
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name}={value!r} is not valid, expected one of {sorted(choices)}")
    return value


def _normalize_profile_rows(
    X: object,
    y: object,
    sample_weight: object,
    offset: object,
) -> tuple[
    pd.DataFrame, NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
]:
    """Validate row-oriented public inputs before building a profile context."""
    if type(X) is not pd.DataFrame:
        raise TypeError("X must be a plain pandas DataFrame")

    y_array = normalize_numeric_vector(y, name="y", nonnegative=True)
    if y_array.size == 0:
        raise ValueError("y must contain at least one observation")
    if len(X) != len(y_array):
        raise ValueError(
            f"X and y must have the same number of rows; got {len(X)} and {len(y_array)}"
        )

    weight_array = None
    if sample_weight is not None:
        weight_message = (
            "weights must be finite and strictly positive, one-dimensional, "
            f"and have length {len(y_array)}"
        )
        try:
            weight_array = normalize_numeric_vector(
                sample_weight,
                name="weights",
                length=len(y_array),
                positive=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(weight_message) from exc

    offset_array = None
    if offset is not None:
        offset_array = normalize_numeric_vector(
            offset,
            name="offset",
            length=len(y_array),
        )
    return X, y_array, weight_array, offset_array


_PROFILE_NUMPY_SCALAR_TYPES = {
    scalar_type
    for scalar_type in np.sctypeDict.values()
    if isinstance(scalar_type, type)
    and issubclass(scalar_type, np.generic)
    and not issubclass(scalar_type, np.void)
}
_PROFILE_EXACT_ATOMIC_TYPES = {
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    Decimal,
    date,
    timedelta,
    range,
}
_PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES = (
    pd.BooleanDtype,
    pd.CategoricalDtype,
    pd.DatetimeTZDtype,
    pd.Float32Dtype,
    pd.Float64Dtype,
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.IntervalDtype,
    pd.StringDtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
)
_PROFILE_NULLABLE_NUMERIC_DTYPE_TYPES = (
    pd.Float32Dtype,
    pd.Float64Dtype,
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
)
_PROFILE_INDEX_TYPES = (
    pd.CategoricalIndex,
    pd.DatetimeIndex,
    pd.Index,
    pd.IntervalIndex,
    pd.MultiIndex,
    pd.RangeIndex,
    pd.TimedeltaIndex,
)
_PROFILE_NULLABLE_NUMERIC_ARRAY_TYPES = (
    pd.arrays.FloatingArray,
    pd.arrays.IntegerArray,
)
_PROFILE_MASKED_ARRAY_TYPES = (
    pd.arrays.BooleanArray,
    *_PROFILE_NULLABLE_NUMERIC_ARRAY_TYPES,
)


def _profile_dtype_has_no_metadata(dtype: np.dtype) -> bool:
    """Return whether a NumPy dtype contains no metadata or structured references."""
    return dtype.metadata is None and dtype.subdtype is None and dtype.fields is None


def _is_share_safe_profile_timezone(value: object) -> bool:
    """Return whether retaining this exact timezone object is safe."""
    if value is None:
        return True
    if type(value) is timezone:
        try:
            return type(value.utcoffset(None)) is timedelta and type(value.tzname(None)) is str
        except (TypeError, ValueError, OverflowError):
            return False
    if type(value) is ZoneInfo:
        return type(value.key) is str
    return False


def _canonical_tweedie_profile_timezone(value: object) -> object:
    """Return a canonical safe timezone without retaining mutable third-party state."""
    if _is_share_safe_profile_timezone(value):
        return value
    zone = getattr(value, "zone", None)
    if type(value).__module__.partition(".")[0] == "pytz" and type(zone) is str:
        try:
            return ZoneInfo(zone)
        except (KeyError, ValueError) as exc:
            raise TypeError("timezone does not identify an installed IANA zone") from exc
    raise TypeError("timezone must be datetime.timezone, ZoneInfo, or a named pytz zone")


def _is_known_immutable_profile_value(value: object) -> bool:
    """Return whether sharing this exact scalar across a profile snapshot is safe."""
    value_type = type(value)
    if value is None or value is Ellipsis or value is NotImplemented:
        return True
    if value is pd.NA or value is pd.NaT:
        return True
    if value_type in _PROFILE_EXACT_ATOMIC_TYPES:
        return True
    if value_type is UUID:
        return type(value.int) is int and (value.is_safe is None or type(value.is_safe) is SafeUUID)
    if value_type in (datetime, time):
        return _is_share_safe_profile_timezone(value.tzinfo)
    if value_type in (tuple, frozenset):
        return all(_is_known_immutable_profile_value(item) for item in value)
    if value_type is slice:
        return all(
            _is_known_immutable_profile_value(item)
            for item in (value.start, value.stop, value.step)
        )
    if value_type in _PROFILE_NUMPY_SCALAR_TYPES:
        return not value.dtype.hasobject
    return False


def _validate_tweedie_profile_dtype(dtype: object, *, name: str) -> None:
    """Accept only native NumPy and understood built-in pandas column storage."""
    if isinstance(dtype, np.dtype):
        if not _profile_dtype_has_no_metadata(dtype) or dtype.kind == "V":
            raise TypeError(f"{name} has an unsupported structured or metadata-bearing dtype")
        return
    if type(dtype) not in _PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES:
        raise TypeError(f"{name} has an unsupported custom extension dtype")
    if type(dtype) in _PROFILE_NULLABLE_NUMERIC_DTYPE_TYPES:
        _validate_tweedie_profile_dtype(dtype.numpy_dtype, name=name)
    if type(dtype) is pd.StringDtype:
        missing_value = getattr(dtype, "na_value", pd.NA)
        if (
            type(dtype.storage) is not str
            or dtype.storage not in {"python", "pyarrow"}
            or not _is_known_immutable_profile_value(missing_value)
        ):
            raise TypeError(f"{name} has unsupported string storage or missing-value state")
    if type(dtype) is pd.DatetimeTZDtype:
        try:
            _canonical_tweedie_profile_timezone(dtype.tz)
        except TypeError as exc:
            raise TypeError(f"{name} has an unsupported mutable or custom timezone") from exc
        if type(dtype.unit) is not str:
            raise TypeError(f"{name} has an unsupported mutable or custom timezone")
    if type(dtype) is pd.IntervalDtype:
        if type(dtype.closed) is not str or dtype.closed not in {
            "left",
            "right",
            "both",
            "neither",
        }:
            raise TypeError(f"{name} has unsupported interval closure state")
        _validate_tweedie_profile_dtype(dtype.subtype, name=name)


def _copy_tweedie_profile_dtype(dtype: object) -> object:
    """Construct a fresh canonical dtype from already-validated public state."""
    if isinstance(dtype, np.dtype):
        return np.dtype(dtype.str)
    if type(dtype) in _PROFILE_NULLABLE_NUMERIC_DTYPE_TYPES:
        return np.dtype(dtype.numpy_dtype)
    if type(dtype) is pd.StringDtype:
        missing_value = getattr(dtype, "na_value", pd.NA)
        try:
            return pd.StringDtype(storage="python", na_value=missing_value)
        except TypeError:
            if missing_value is not pd.NA:
                raise TypeError("This pandas version cannot preserve the string missing value")
            return pd.StringDtype(storage="python")
    if type(dtype) is pd.DatetimeTZDtype:
        return pd.DatetimeTZDtype(
            unit=dtype.unit,
            tz=_canonical_tweedie_profile_timezone(dtype.tz),
        )
    if type(dtype) is pd.IntervalDtype:
        return pd.IntervalDtype(
            subtype=_copy_tweedie_profile_dtype(dtype.subtype),
            closed=dtype.closed,
        )
    if type(dtype) is pd.CategoricalDtype:
        raise TypeError("categorical dtypes are reconstructed from categories and codes")
    return type(dtype)()


def _require_base_ndarray(value: object, *, name: str) -> None:
    """Reject ndarray subclasses whose copy/read operations are user-controlled."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must use an exact NumPy ndarray backing buffer")


def _validate_tweedie_profile_array_storage(values: object, *, name: str) -> None:
    """Validate the concrete pandas array and every mutable backing buffer."""
    values_type = type(values)
    if values_type is np.ndarray:
        return
    if values_type is pd.Categorical:
        _require_base_ndarray(values._codes, name=f"{name} categorical codes")
        if type(values.ordered) is not bool:
            raise TypeError(f"{name} has unsupported categorical ordering state")
        _validate_tweedie_profile_axis(values.categories, name=f"{name} categories")
        return
    if values_type in _PROFILE_MASKED_ARRAY_TYPES:
        _require_base_ndarray(values._data, name=f"{name} data")
        _require_base_ndarray(values._mask, name=f"{name} mask")
        return
    if values_type is pd.arrays.StringArray:
        _require_base_ndarray(values._ndarray, name=f"{name} string data")
        if any(not _is_known_immutable_profile_value(value) for value in values._ndarray):
            raise TypeError(f"{name} contains unsupported custom string scalar values")
        return
    arrow_string_array = getattr(pd.arrays, "ArrowStringArray", None)
    if arrow_string_array is not None and values_type is arrow_string_array:
        # Arrow buffers are immutable; reconstruction iterates into Python-backed
        # string storage so the prepared frame retains no caller-owned Arrow graph.
        return
    if values_type in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray):
        _require_base_ndarray(values._ndarray, name=f"{name} datetime data")
        return
    if values_type is pd.arrays.IntervalArray:
        _validate_tweedie_profile_array_storage(values._left, name=f"{name} left endpoints")
        _validate_tweedie_profile_array_storage(values._right, name=f"{name} right endpoints")
        return
    raise TypeError(f"{name} uses an unsupported or custom pandas array implementation")


def _materialize_tweedie_nullable_numeric(values: object, *, name: str) -> np.ndarray:
    """Own nullable numeric values in native storage without silent integer rounding."""
    target_dtype = np.dtype(values._data.dtype)
    mask = values._mask
    if np.any(mask) and target_dtype.kind in "iu":
        present = values._data[~mask]
        as_float = np.asarray(present, dtype=np.float64)
        if any(
            not np.isfinite(converted) or int(converted) != int(original)
            for original, converted in zip(present, as_float, strict=True)
        ):
            raise TypeError(
                f"{name} has missing nullable integers that cannot be represented exactly"
            )
        target_dtype = np.dtype(np.float64)
    owned = np.array(values._data, dtype=target_dtype, copy=True, subok=False)
    if np.any(mask):
        owned[mask] = np.nan
    return owned


def _validate_tweedie_profile_axis(axis: pd.Index, *, name: str) -> None:
    """Reject mutable/custom labels whose state could leak into a snapshot."""
    if type(axis) not in _PROFILE_INDEX_TYPES:
        raise TypeError(f"{name} must use a built-in pandas Index type")
    _validate_tweedie_profile_dtype(axis.dtype, name=name)
    if any(not _is_known_immutable_profile_value(value) for value in axis.names):
        raise TypeError(f"{name} names must be immutable scalar values for Tweedie profiling")
    if type(axis) in (pd.DatetimeIndex, pd.TimedeltaIndex):
        frequency = axis.freqstr
        if frequency is not None and type(frequency) is not str:
            raise TypeError(f"{name} has unsupported frequency metadata")
    if isinstance(axis, pd.CategoricalIndex):
        _validate_tweedie_profile_array_storage(axis.array, name=name)
        _validate_tweedie_profile_axis(axis.categories, name=f"{name} categories")
        return
    if isinstance(axis, pd.MultiIndex):
        for position, code in enumerate(axis.codes):
            _require_base_ndarray(code, name=f"{name} code {position}")
        for position, level in enumerate(axis.levels):
            _validate_tweedie_profile_axis(level, name=f"{name} level {position}")
        return
    _validate_tweedie_profile_array_storage(axis._values, name=name)
    if pd.api.types.is_object_dtype(axis.dtype) or type(axis.dtype) is pd.StringDtype:
        if any(not _is_known_immutable_profile_value(value) for value in axis):
            raise TypeError(f"{name} labels must be immutable scalar values for Tweedie profiling")


def _tweedie_profile_dataframe_column(X: pd.DataFrame, position: int) -> pd.Series:
    """Return one trusted base-Series view from an exact base DataFrame."""
    column = X.iloc[:, position]
    if type(column) is not pd.Series:
        raise TypeError(f"X column {position} did not produce a plain pandas Series")
    return column


def _validate_tweedie_profile_dataframe_values(X: pd.DataFrame) -> None:
    """Validate object-backed values before retaining any shared scalar references."""
    attributes = X.attrs
    if type(attributes) is not dict:
        raise TypeError("X.attrs must use a plain dict for Tweedie profiling")
    if attributes:
        raise TypeError("X.attrs must be empty for Tweedie profiling")
    _validate_tweedie_profile_axis(X.index, name="X index")
    _validate_tweedie_profile_axis(X.columns, name="X column")
    for position in range(len(X.columns)):
        column = _tweedie_profile_dataframe_column(X, position)
        dtype = column.dtype
        _validate_tweedie_profile_dtype(dtype, name=f"X column {position}")
        values = column._values
        _validate_tweedie_profile_array_storage(values, name=f"X column {position}")
        if pd.api.types.is_object_dtype(dtype) or type(dtype) is pd.StringDtype:
            scalar_values = values._ndarray if type(values) is pd.arrays.StringArray else values
        else:
            continue
        if any(not _is_known_immutable_profile_value(value) for value in scalar_values):
            raise TypeError(
                "X object and categorical values must be immutable scalar values "
                "for Tweedie profiling"
            )


def _copy_tweedie_profile_axis(axis: pd.Index) -> pd.Index:
    """Reconstruct a validated pandas axis without retaining caller storage."""
    if isinstance(axis, pd.CategoricalIndex):
        source = axis.array
        categorical = pd.Categorical.from_codes(
            np.array(source.codes, copy=True, subok=False),
            categories=_copy_tweedie_profile_axis(source.categories),
            ordered=source.ordered,
        )
        return pd.CategoricalIndex(categorical, name=axis.name)
    if isinstance(axis, pd.MultiIndex):
        return pd.MultiIndex(
            levels=[_copy_tweedie_profile_axis(level) for level in axis.levels],
            codes=[np.array(code, copy=True, subok=False) for code in axis.codes],
            sortorder=axis.sortorder,
            names=list(axis.names),
            verify_integrity=False,
        )
    if isinstance(axis, pd.RangeIndex):
        return pd.RangeIndex(axis.start, axis.stop, axis.step, name=axis.name)
    values = axis._values
    if type(values) in _PROFILE_NULLABLE_NUMERIC_ARRAY_TYPES:
        owned = _materialize_tweedie_nullable_numeric(values, name="profile axis")
        return pd.Index(owned, dtype=owned.dtype, name=axis.name, tupleize_cols=False)
    dtype = _copy_tweedie_profile_dtype(axis.dtype)
    if isinstance(dtype, np.dtype):
        if type(values) is np.ndarray:
            owned = np.array(values, dtype=dtype, copy=True, subok=False)
        else:
            owned = np.array(values._ndarray, dtype=dtype, copy=True, subok=False)
    else:
        owned = pd.array(list(axis), dtype=dtype, copy=True)
    if type(axis) is pd.DatetimeIndex:
        return pd.DatetimeIndex(
            owned,
            dtype=dtype,
            freq=axis.freqstr,
            name=axis.name,
        )
    if type(axis) is pd.TimedeltaIndex:
        return pd.TimedeltaIndex(
            owned,
            dtype=dtype,
            freq=axis.freqstr,
            name=axis.name,
        )
    return pd.Index(owned, dtype=dtype, name=axis.name, tupleize_cols=False)


def _copy_tweedie_profile_column(
    values: object,
    dtype: object,
    *,
    index: pd.Index,
) -> pd.Series:
    """Reconstruct one validated column from owned canonical storage."""
    if isinstance(dtype, pd.CategoricalDtype):
        categorical = pd.Categorical.from_codes(
            np.array(values.codes, copy=True, subok=False),
            categories=_copy_tweedie_profile_axis(values.categories),
            ordered=values.ordered,
        )
        return pd.Series(categorical, index=index, copy=False)
    if type(values) in _PROFILE_NULLABLE_NUMERIC_ARRAY_TYPES:
        owned = _materialize_tweedie_nullable_numeric(values, name="profile column")
        return pd.Series(owned, index=index, dtype=owned.dtype, copy=False)
    owned_dtype = _copy_tweedie_profile_dtype(dtype)
    if isinstance(owned_dtype, np.dtype):
        if type(values) is np.ndarray:
            owned = np.array(values, dtype=owned_dtype, copy=True, subok=False)
        else:
            owned = np.array(values._ndarray, dtype=owned_dtype, copy=True, subok=False)
    else:
        owned = pd.array(list(values), dtype=owned_dtype, copy=True)
    return pd.Series(owned, index=index, dtype=owned_dtype, copy=False)


def _reconstruct_tweedie_profile_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Build the owned frame after source validation has completed."""
    snapshot_index = _copy_tweedie_profile_axis(X.index)
    snapshot_columns = _copy_tweedie_profile_axis(X.columns)
    snapshot = pd.DataFrame(index=snapshot_index)
    for position in range(len(X.columns)):
        column = _tweedie_profile_dataframe_column(X, position)
        source = column._values
        snapshot.insert(
            position,
            position,
            _copy_tweedie_profile_column(
                source,
                column.dtype,
                index=snapshot_index,
            ),
        )
    snapshot.columns = snapshot_columns
    return snapshot


def _snapshot_tweedie_profile_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a validated frame entirely from owned canonical storage."""
    try:
        _validate_tweedie_profile_dataframe_values(X)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise TypeError("Could not safely snapshot values in X") from exc
    if not X.columns.is_unique:
        raise ValueError("X column labels must be unique")
    try:
        return _reconstruct_tweedie_profile_dataframe(X)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise TypeError("Could not safely snapshot values in X") from exc


def _prepare_tweedie_profile_inputs(
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
    **unexpected,
) -> _PreparedTweedieProfileInputs:
    """Validate one public profile call before any model fit or user callback."""
    from superglm.distributions import Tweedie

    if unexpected:
        name = next(iter(unexpected))
        raise TypeError(f"estimate_tweedie_p() got an unexpected keyword argument {name!r}")

    family = model.family
    if not isinstance(family, Tweedie):
        raise ValueError(
            f"estimate_tweedie_p requires a Tweedie family, got {family!r}. "
            "Use families.tweedie(p=...) to create one."
        )

    method = _normalize_profile_choice(
        method,
        name="method",
        choices={"brent", "grid", "grid_refine", "profile_opt", "joint_ml", "integrated"},
    )
    fit_mode = _normalize_profile_choice(
        fit_mode,
        name="fit_mode",
        choices={"fit", "fit_reml"},
    )
    phi_method = _normalize_profile_choice(
        phi_method,
        name="phi_method",
        choices={"pearson", "mle"},
    )
    optimizer = _normalize_profile_choice(
        optimizer,
        name="optimizer",
        choices={"L-BFGS-B", "Powell"},
    )
    p_bounds = normalize_tweedie_bounds(p_bounds)
    xatol = normalize_positive_scalar("xatol", xatol)
    maxiter = normalize_positive_int(
        maxiter,
        name="maxiter",
        maximum=_MAX_PROFILE_INTEGER_CONTROL,
    )
    n_grid = normalize_positive_int(
        n_grid,
        name="n_grid",
        minimum=2,
        maximum=_MAX_PROFILE_GRID_POINTS,
    )
    n_grid_coarse = normalize_positive_int(
        n_grid_coarse,
        name="n_grid_coarse",
        minimum=2,
        maximum=_MAX_PROFILE_GRID_POINTS,
    )
    verbose = normalize_boolean(verbose, name="verbose")
    trace_iterations = normalize_boolean(trace_iterations, name="trace_iterations")
    trace_callback = normalize_optional_callable(trace_callback, name="trace_callback")
    if grid is not None:
        grid = normalize_tweedie_grid(grid, maximum=_MAX_PROFILE_GRID_POINTS)

    X, y_array, sample_weight, offset = _normalize_profile_rows(
        X,
        y,
        sample_weight,
        offset,
    )
    if method in ("joint_ml", "integrated"):
        raise NotImplementedError(
            f"method={method!r} is not yet implemented. "
            "Use one of: 'brent', 'grid', 'grid_refine', 'profile_opt'."
        )
    X = _snapshot_tweedie_profile_dataframe(X)

    return _PreparedTweedieProfileInputs(
        X=X,
        y=y_array,
        sample_weight=sample_weight,
        offset=offset,
        p_bounds=p_bounds,
        xatol=xatol,
        maxiter=maxiter,
        verbose=verbose,
        fit_mode=fit_mode,
        phi_method=phi_method,
        method=method,
        n_grid=n_grid,
        grid=grid,
        n_grid_coarse=n_grid_coarse,
        optimizer=optimizer,
        trace_callback=trace_callback,
        trace_iterations=trace_iterations,
        _model_identity=id(model),
        _validation_token=_PREPARED_TWEEDIE_PROFILE_TOKEN,
    )


def _estimate_tweedie_p_prepared(
    model,
    prepared: _PreparedTweedieProfileInputs,
) -> TweedieProfileResult:
    """Execute a profile search from one already validated, owned input set."""
    ctx: _ProfileContext | _ProfileContextREML
    if prepared.fit_mode == "fit_reml":
        ctx = _build_profile_context_reml(
            model,
            prepared.X,
            prepared.y,
            prepared.sample_weight,
            prepared.offset,
            prepared.phi_method,
            prepared.verbose,
            prepared.trace_callback,
            prepared.trace_iterations,
            _inputs_owned=True,
        )
    else:
        ctx = _build_profile_context(
            model,
            prepared.X,
            prepared.y,
            prepared.sample_weight,
            prepared.offset,
            prepared.phi_method,
            prepared.verbose,
            prepared.trace_callback,
            prepared.trace_iterations,
            _inputs_owned=True,
        )

    try:
        if prepared.method == "brent":
            return _search_brent(ctx, prepared.p_bounds, prepared.xatol, prepared.maxiter)
        if prepared.method == "grid":
            return _search_grid(ctx, prepared.p_bounds, prepared.n_grid, prepared.grid)
        if prepared.method == "grid_refine":
            return _search_grid_refine(
                ctx,
                prepared.p_bounds,
                prepared.n_grid_coarse,
                prepared.xatol,
                prepared.maxiter,
            )
        return _search_profile_opt(
            ctx,
            prepared.p_bounds,
            prepared.optimizer,
            prepared.xatol,
            prepared.maxiter,
        )
    finally:
        # Trace callbacks are synchronous search instrumentation, not part of
        # the lazy objective retained for confidence intervals or later probes.
        ctx.trace_callback = None


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
        Feature matrix. Profiling snapshots one exact ``pandas.DataFrame`` with
        empty ``attrs`` and built-in NumPy/pandas dtypes. Object values,
        categorical levels, and axis labels must be ordinary deeply immutable
        scalars; convert custom numeric wrappers or category objects to plain
        numbers or strings before profiling. Built-in nullable numerics are
        accepted when they can be materialized exactly as owned NumPy storage.
        Sparse, period, Arrow-backed, and custom extension storage must first
        be converted to dense native NumPy, Python-string, boolean, or
        categorical columns.
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Finite, strictly positive EDM prior weights, not replication or frequency weights.
        The Tweedie variance convention is
        ``Var(Y_i | x_i) = phi * mu_i**p / w_i``.
        Remove zero-weight rows consistently from ``X``, ``y``, ``sample_weight``,
        and ``offset`` before calling this function.
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
        How to profile out ``phi`` at each candidate ``p``. ``"mle"`` is the
        default; ``"pearson"`` is an explicit faster plug-in that does not
        support likelihood-ratio confidence intervals.
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
    prepared = _claim_prepared_tweedie_profile_inputs(
        model,
        X,
        y,
        sample_weight,
        offset,
        p_bounds,
        xatol,
        maxiter,
        verbose,
        fit_mode,
        phi_method,
        method,
        n_grid,
        grid,
        n_grid_coarse,
        optimizer,
        trace_callback,
        trace_iterations,
    )
    if prepared is None:
        prepared = _prepare_tweedie_profile_inputs(
            model,
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            p_bounds=p_bounds,
            xatol=xatol,
            maxiter=maxiter,
            verbose=verbose,
            fit_mode=fit_mode,
            phi_method=phi_method,
            method=method,
            n_grid=n_grid,
            grid=grid,
            n_grid_coarse=n_grid_coarse,
            optimizer=optimizer,
            trace_callback=trace_callback,
            trace_iterations=trace_iterations,
        )
    return _estimate_tweedie_p_prepared(model, prepared)


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
    if _contains_masked_array(alpha):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    normalized: dict[str, float] = {}
    for name, value in (("p_hat", p_hat), ("nll_hat", nll_hat), ("ll_scale", ll_scale)):
        if _contains_masked_array(value):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be a finite scalar")
        normalized[name] = parsed
    if normalized["ll_scale"] <= 0.0:
        raise ValueError("ll_scale must be finite and strictly positive")

    if _contains_masked_array(p_range):
        raise ValueError("p_range must be two ordered finite bounds")
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
            if _contains_masked_array(raw_nll):
                raise ValueError("objective returned a value containing a mask")
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
