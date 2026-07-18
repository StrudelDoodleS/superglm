"""Observed-information geometry for Wood's LAML/REML criterion.

The coefficient solver may reach the penalized likelihood optimum by Fisher
scoring, but the Laplace determinant and its smoothing-parameter derivatives
must use the negative *observed* likelihood Hessian.  The two coincide for
canonical links and differ for non-canonical links such as Gamma/log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_well_scaled,
    centered_gram_rhs,
    centered_rhs,
)
from superglm.distributions import (
    _VARIANCE_FLOOR,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import DesignMatrix
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    LogitLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
    stabilize_eta,
)
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    build_centered_system,
    grouped_augmented_factor,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification


def _readonly(values: NDArray, *, dtype: Any = np.float64) -> NDArray:
    # Every caller passes either a newly computed array or an already-frozen
    # centered-system array.  Avoid duplicating O(n) curvature rows and O(p²)
    # geometry at this short-lived outer-iteration boundary.
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _validate_rows(
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.ndim != 1 or mu.shape != y.shape or eta.shape != y.shape:
        raise ValueError("y, mu, and eta must be one-dimensional arrays with equal shape")
    if sample_weight.shape != y.shape:
        raise ValueError("sample_weight must match y")
    if (
        not np.all(np.isfinite(y))
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(eta))
        or not np.all(np.isfinite(sample_weight))
    ):
        raise ValueError("observed-information inputs must be finite")
    if np.any(sample_weight < 0.0):
        raise ValueError("sample_weight must be non-negative")
    return y, mu, eta, sample_weight


def compute_observed_information_weights(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Return row weights for the negative observed log-likelihood Hessian.

    With ``u = dmu/deta``, ``v = d2mu/deta2``, variance ``V`` and response
    residual ``r = y - mu``, the unit-dispersion curvature is

    ``W_obs = w*u**2/V + w*r*(u**2*V'/V**2 - v/V)``.

    This is Wood's Newton weight written using inverse-link derivatives.  For
    Gamma/log it reduces to ``w*y/mu``; its Fisher counterpart is merely
    ``w``.  The dispersion factor is deliberately absent because Wood's
    criterion factors the common ``1/phi`` out of both likelihood and penalty
    curvature.
    """
    y, mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
    if isinstance(link, LogLink):
        observed: NDArray | None = None
        if isinstance(distribution, Gamma):
            # GLUM uses the same closed form in its specialized Gamma/log
            # Newton rows.  It avoids five temporary O(n) arrays here.
            observed = sample_weight * y / mu
        elif isinstance(distribution, Poisson):
            observed = sample_weight * mu
        elif isinstance(distribution, NegativeBinomial) and distribution.theta != "auto":
            theta = float(distribution.theta)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                observed = sample_weight * theta * mu * (theta + y) / (theta + mu) ** 2
        elif isinstance(distribution, Tweedie):
            power = float(distribution.p)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                observed = (
                    sample_weight * mu ** (1.0 - power) * ((2.0 - power) * mu + (power - 1.0) * y)
                )
        if observed is not None:
            if not np.all(np.isfinite(observed)):
                raise ValueError("observed-information weights are not finite")
            return observed
    if not hasattr(link, "deriv2_inverse"):
        raise NotImplementedError("observed curvature requires link.deriv2_inverse")
    if not hasattr(distribution, "variance_derivative"):
        raise NotImplementedError("observed curvature requires distribution.variance_derivative")

    u = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    v = np.asarray(link.deriv2_inverse(eta), dtype=np.float64)
    variance = np.maximum(
        np.asarray(distribution.variance(mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    variance_prime = np.asarray(distribution.variance_derivative(mu), dtype=np.float64)
    residual = y - mu
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fisher = sample_weight * u**2 / variance
        noncanonical = (
            sample_weight * residual * (u**2 * variance_prime / variance**2 - v / variance)
        )
        observed = fisher + noncanonical
    if not np.all(np.isfinite(observed)):
        raise ValueError("observed-information weights are not finite")
    return observed


REMLCurvature = Literal["fisher", "observed"]
SCOPREMLCurvature = REMLCurvature

_BUILTIN_REML_DISTRIBUTIONS = (
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    NegativeBinomial,
    Tweedie,
)
_BUILTIN_REML_LINKS = (
    IdentityLink,
    LogLink,
    LogitLink,
    ProbitLink,
    CloglogLink,
    CauchitLink,
    InverseLink,
    InverseSquaredLink,
    SqrtLink,
    PowerLink,
    NegativeBinomialLink,
)


def _parameters_match(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 32.0 * np.finfo(np.float64).eps * scale


def _builtin_fisher_equals_observed(distribution: Any, link: Any) -> bool:
    """Return whether the residual term in Wood's Newton rows is identically zero."""
    distribution_type = type(distribution)
    link_type = type(link)
    if distribution_type is Gaussian:
        return link_type is IdentityLink or (
            link_type is PowerLink and _parameters_match(float(link.power), 1.0)
        )
    if distribution_type is Poisson:
        return link_type is LogLink
    if distribution_type is Binomial:
        return link_type is LogitLink
    if distribution_type is Gamma:
        return link_type is InverseLink or (
            link_type is PowerLink and _parameters_match(float(link.power), -1.0)
        )
    if distribution_type is NegativeBinomial and link_type is NegativeBinomialLink:
        theta = distribution.theta
        return theta != "auto" and _parameters_match(float(theta), float(link.theta))
    if distribution_type is Tweedie and link_type is PowerLink:
        return _parameters_match(float(link.power), 1.0 - float(distribution.p))
    return False


def _explicit_curvature_protocol(
    distribution: Any,
    link: Any,
    *,
    hook_name: str,
    description: str,
) -> REMLCurvature | None:
    declarations: list[str] = []
    distribution_hook = getattr(distribution, hook_name, None)
    if callable(distribution_hook):
        declarations.append(str(distribution_hook(link)))
    link_hook = getattr(link, hook_name, None)
    if callable(link_hook):
        declarations.append(str(link_hook(distribution)))
    if not declarations:
        return None
    if any(value not in {"fisher", "observed"} for value in declarations):
        raise ValueError(f"{hook_name} must return 'fisher' or 'observed'")
    if len(set(declarations)) != 1:
        raise ValueError(f"family and link {description} curvature protocols disagree")
    return declarations[0]  # type: ignore[return-value]


def classify_reml_curvature(distribution: Any, link: Any) -> REMLCurvature:
    """Select the exact coefficient geometry for ordinary direct LAML."""
    distribution_is_builtin = type(distribution) in _BUILTIN_REML_DISTRIBUTIONS
    link_is_builtin = type(link) in _BUILTIN_REML_LINKS
    if distribution_is_builtin and link_is_builtin:
        return "fisher" if _builtin_fisher_equals_observed(distribution, link) else "observed"

    declared = _explicit_curvature_protocol(
        distribution,
        link,
        hook_name="reml_curvature",
        description="ordinary REML",
    )
    if declared is not None:
        return declared
    raise NotImplementedError(
        "custom family/link combinations require an explicit ordinary REML curvature "
        "protocol; define reml_curvature(counterpart) returning 'fisher' or 'observed'"
    )


def classify_scop_reml_curvature(distribution: Any, link: Any) -> SCOPREMLCurvature:
    """Select the coefficient Hessian used by shape-constrained LAML.

    Fisher rows are valid only for built-in combinations whose residual
    curvature term vanishes analytically.  Other built-in combinations use
    exact order-zero observed rows.  Custom families or links must opt in via
    ``scop_reml_curvature(counterpart)`` so an unverified Fisher approximation
    can never be presented as Pya--Wood geometry.
    """
    distribution_is_builtin = type(distribution) in _BUILTIN_REML_DISTRIBUTIONS
    link_is_builtin = type(link) in _BUILTIN_REML_LINKS
    if distribution_is_builtin and link_is_builtin:
        return "fisher" if _builtin_fisher_equals_observed(distribution, link) else "observed"

    declared = _explicit_curvature_protocol(
        distribution,
        link,
        hook_name="scop_reml_curvature",
        description="SCOP REML",
    )
    if declared is not None:
        return declared
    raise NotImplementedError(
        "custom family/link combinations require an explicit SCOP REML curvature protocol; "
        "define scop_reml_curvature(counterpart) returning 'fisher' or 'observed'"
    )


def compute_scop_observed_information_weights(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Return exact non-negative observed rows supported by SCOP moment kernels."""
    y, mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
    distribution_hook = getattr(distribution, "scop_observed_information_weights", None)
    link_hook = getattr(link, "scop_observed_information_weights", None)
    if callable(distribution_hook) and callable(link_hook):
        raise ValueError("family and link both define SCOP observed-information row protocols")
    if callable(distribution_hook):
        observed = distribution_hook(link, y, mu, eta, sample_weight)
    elif callable(link_hook):
        observed = link_hook(distribution, y, mu, eta, sample_weight)
    else:
        observed = compute_observed_information_weights(
            distribution,
            link,
            y,
            mu,
            eta,
            sample_weight,
        )
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != y.shape or not np.all(np.isfinite(observed)):
        raise ValueError("SCOP observed-information rows must be finite and match y")
    if np.any(observed < 0.0):
        minimum = float(np.min(observed))
        raise ValueError(
            "signed observed-information rows are not supported by the current stable SCOP "
            f"moment kernels (minimum row={minimum:.3e})"
        )
    return observed


def requires_observed_reml_geometry(distribution: Any, link: Any) -> bool:
    """Whether direct REML must replace its fitted Fisher geometry.

    Canonical/equal-curvature combinations must reuse the solver geometry to
    avoid an unnecessary full data pass.  Gamma/log is the first enabled
    noncanonical specialization: its positive closed-form rows make the
    observed replacement exact and keep it on the accelerated centered path.
    Other noncanonical combinations are enabled separately once their mode
    convergence and indefinite-curvature policies are fully gated.
    """
    return classify_reml_curvature(distribution, link) == "observed"


def compute_observed_dW_deta(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Differentiate the observed-information row weights w.r.t. ``eta``."""
    y, mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
    if isinstance(link, LogLink):
        if isinstance(distribution, Gamma):
            return -(sample_weight * y / mu)
        if isinstance(distribution, Poisson):
            return sample_weight * mu
    if not hasattr(link, "deriv3_inverse"):
        raise NotImplementedError("observed-weight derivatives require link.deriv3_inverse")
    if not hasattr(distribution, "variance_second_derivative"):
        raise NotImplementedError(
            "observed-weight derivatives require distribution.variance_second_derivative"
        )

    u = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    v = np.asarray(link.deriv2_inverse(eta), dtype=np.float64)
    t = np.asarray(link.deriv3_inverse(eta), dtype=np.float64)
    variance = np.maximum(
        np.asarray(distribution.variance(mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    variance_prime = np.asarray(distribution.variance_derivative(mu), dtype=np.float64)
    variance_second = np.asarray(
        distribution.variance_second_derivative(mu),
        dtype=np.float64,
    )
    residual = y - mu

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fisher_prime = sample_weight * (
            2.0 * u * v / variance - u**3 * variance_prime / variance**2
        )
        noncanonical_factor = u**2 * variance_prime / variance**2 - v / variance
        factor_prime = (
            3.0 * u * v * variance_prime / variance**2
            + u**3 * variance_second / variance**2
            - 2.0 * u**3 * variance_prime**2 / variance**3
            - t / variance
        )
        derivative = fisher_prime + sample_weight * (
            -u * noncanonical_factor + residual * factor_prime
        )
    if not np.all(np.isfinite(derivative)):
        raise ValueError("observed-information weight derivatives are not finite")
    return derivative


def compute_observed_d2W_deta2(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    *,
    allow_approximate: bool = False,
) -> NDArray:
    """Return the second ``eta`` derivative of observed row curvature.

    Gamma/log and Poisson/log have exact closed forms. Other combinations
    require inverse-link fourth and variance third derivatives for Wood's
    analytic expression; those are not part of the current family protocol.
    They therefore fail by default rather than silently presenting a fixed-step
    finite difference as exact. ``allow_approximate=True`` is an explicit
    diagnostic escape hatch and is not used by production LAML.
    """
    y, _mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
    if isinstance(link, LogLink):
        if isinstance(distribution, Gamma):
            return sample_weight * y / _mu
        if isinstance(distribution, Poisson):
            return sample_weight * _mu
    if not allow_approximate:
        raise NotImplementedError(
            "exact second observed-weight derivatives are unavailable for this family/link"
        )
    eps = 1e-5
    eta_plus = eta + eps
    eta_minus = eta - eps
    mu_plus = clip_mu(link.inverse(eta_plus), distribution)
    mu_minus = clip_mu(link.inverse(eta_minus), distribution)
    plus = compute_observed_dW_deta(
        distribution,
        link,
        y,
        mu_plus,
        eta_plus,
        sample_weight,
    )
    minus = compute_observed_dW_deta(
        distribution,
        link,
        y,
        mu_minus,
        eta_minus,
        sample_weight,
    )
    derivative = (plus - minus) / (2.0 * eps)
    if not np.all(np.isfinite(derivative)):
        raise ValueError("second observed-information weight derivatives are not finite")
    return derivative


@dataclass(frozen=True)
class ObservedREMLGeometry:
    """Observed slope geometry after stable profiling of the intercept."""

    eta: NDArray
    mu: NDArray
    weights: NDArray
    weight_derivative: NDArray | None
    weight_second_derivative: NDArray | None
    sum_w: float
    mean_x: NDArray
    centered_data_gram: NDArray
    centered_hessian: NDArray
    hessian_inverse: NDArray | None
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class ObservedModeScore:
    """Penalized likelihood score and a unitless KKT residual."""

    intercept: float
    slopes: NDArray
    max_abs: float
    relative_max: float


def _stable_signed_sum(values: NDArray) -> float:
    """Sum signed finite rows without cancellation or intermediate overflow."""
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0
    normalized_sum = math.fsum(float(value / scale) for value in values)
    return float(scale * normalized_sum)


def observed_penalized_mode_score(
    *,
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    y: NDArray,
    sample_weight: NDArray,
    result: PIRLSResult,
    penalty: NDArray,
    geometry: ObservedREMLGeometry,
) -> ObservedModeScore:
    """Evaluate the full penalized score at a proposed Laplace mode.

    Fisher scoring and Newton iteration target the same score root, but a
    deviance-change stopping rule alone does not certify that the root is
    accurate enough for implicit differentiation.  This residual is evaluated
    from the retained coefficients and is scale-relative in both intercept and
    slope equations.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape:
        raise ValueError("mode-score rows must match the design")
    if penalty.shape != (dm.p, dm.p):
        raise ValueError("mode-score penalty must match slope coordinates")
    if geometry.mu.shape != y.shape:
        raise ValueError("observed geometry does not match mode-score rows")

    variance = np.maximum(
        np.asarray(distribution.variance(geometry.mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    dmu_deta = np.asarray(link.deriv_inverse(geometry.eta), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_score = sample_weight * (y - geometry.mu) * dmu_deta / variance
    if not np.all(np.isfinite(row_score)):
        raise ValueError("penalized mode score is not finite")

    intercept_score = float(np.sum(row_score, dtype=np.float64))
    centered_diagonal = np.diag(geometry.centered_data_gram)
    with np.errstate(invalid="ignore", divide="ignore"):
        centered_scale = np.sqrt(np.abs(centered_diagonal) / geometry.sum_w)
    raw_centering_safe = np.all(np.isfinite(centered_scale)) and _raw_centering_well_scaled(
        geometry.mean_x,
        centered_scale,
    )
    if raw_centering_safe:
        data_slope_score = dm.rmatvec(row_score) - geometry.mean_x * intercept_score
    else:
        data_slope_score = centered_rhs(
            dm=dm,
            W=np.ones(dm.n, dtype=np.float64),
            mean_x=geometry.mean_x,
            z_centered=row_score,
        )
    penalty_score = penalty @ result.beta
    slope_score = data_slope_score - penalty_score
    max_abs = max(
        abs(intercept_score),
        float(np.max(np.abs(slope_score), initial=0.0)),
    )
    tiny = np.finfo(np.float64).tiny
    intercept_scale = max(
        tiny,
        float(np.sum(np.abs(row_score), dtype=np.float64)),
    )
    # Normalize by a pre-cancellation bound, not by ``data_slope_score``
    # itself.  At a converged mode that aggregate is (nearly) zero, so using
    # it as its own denominator turns any round-off residue into a relative
    # error of one.  The absolute row-score mass times the centered predictor
    # scale has the right score units and is invariant to feature translation
    # and to a common rescaling of observation weights.
    slope_scale = np.maximum(
        tiny,
        intercept_scale * centered_scale + np.abs(penalty_score),
    )
    slope_relative_max = float(np.max(np.abs(slope_score) / slope_scale, initial=0.0))
    relative_max = max(
        abs(intercept_score) / intercept_scale,
        slope_relative_max,
    )
    return ObservedModeScore(
        intercept=intercept_score,
        slopes=_readonly(slope_score),
        max_abs=max_abs,
        relative_max=relative_max,
    )


def _stable_signed_mean(dm: DesignMatrix, weights: NDArray, sum_w: float) -> NDArray:
    """Compute a signed weighted mean without subtracting large raw moments."""
    if dm.p == 0:
        return np.zeros(0, dtype=np.float64)
    anchor = np.asarray(dm.row_subset(np.array([0], dtype=np.intp)).toarray()[0], dtype=float)
    total = np.zeros(dm.p, dtype=np.float64)
    compensation = np.zeros(dm.p, dtype=np.float64)
    chunk_size = 8192
    for start in range(0, dm.n, chunk_size):
        stop = min(start + chunk_size, dm.n)
        rows = np.arange(start, stop, dtype=np.intp)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=np.float64)
        contribution = (block - anchor).T @ weights[start:stop]
        corrected = contribution - compensation
        updated = total + corrected
        compensation[...] = (updated - total) - corrected
        total[...] = updated
    return anchor + total / sum_w


def build_observed_reml_geometry(
    *,
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result: PIRLSResult,
    penalty: NDArray,
    tabmat_state: TabmatCenteringState | None = None,
    compute_inverse: bool = True,
    derivative_order: int = 0,
) -> ObservedREMLGeometry:
    """Build Wood's observed LAML Hessian without altering fit inference state.

    Non-negative observed rows use the shared centered-system execution layer,
    including its Tabmat/discrete kernels.  The uncommon negative-row case
    uses bounded, compensated centered chunks; the final penalized curvature
    must still be positive semidefinite for a valid Laplace mode.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    offset_arr = np.asarray(offset_arr, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape or offset_arr.shape != y.shape:
        raise ValueError("REML geometry row arrays must match the design")
    penalty = np.asarray(penalty, dtype=np.float64)
    if penalty.shape != (dm.p, dm.p) or not np.all(np.isfinite(penalty)):
        raise ValueError("penalty must be a finite square matrix in slope coordinates")
    penalty = 0.5 * (penalty + penalty.T)
    # S is a Gaussian-prior precision and must be PSD.  The shared centered
    # solver may project tiny round-off in a declared-PSD system, so reject a
    # materially invalid penalty before entering that path.
    decompose_gram(penalty)
    if derivative_order not in (0, 1, 2):
        raise ValueError("derivative_order must be 0, 1, or 2")

    beta = np.asarray(result.beta, dtype=np.float64)
    if beta.shape != (dm.p,) or not np.all(np.isfinite(beta)):
        raise ValueError("result.beta must be a finite vector matching the design columns")
    if not np.isfinite(result.intercept):
        raise ValueError("result.intercept must be finite")

    eta = stabilize_eta(dm.matvec(beta) + result.intercept + offset_arr, link)
    mu = clip_mu(link.inverse(eta), distribution)
    observed_w = compute_observed_information_weights(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )
    weight_derivative = (
        compute_observed_dW_deta(
            distribution,
            link,
            y,
            mu,
            eta,
            sample_weight,
        )
        if derivative_order >= 1
        else None
    )
    weight_second_derivative = (
        compute_observed_d2W_deta2(
            distribution,
            link,
            y,
            mu,
            eta,
            sample_weight,
        )
        if derivative_order >= 2
        else None
    )
    nonnegative = bool(np.all(observed_w >= 0.0))
    sum_w = (
        float(np.sum(observed_w, dtype=np.float64))
        if nonnegative
        else _stable_signed_sum(observed_w)
    )
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("observed intercept curvature must have a positive finite sum")

    if nonnegative:
        centered = build_centered_system(
            dm=dm,
            W=observed_w,
            z_off=np.zeros(dm.n, dtype=np.float64),
            penalty=penalty,
            tabmat_split=dm.tabmat_centering_split,
            tabmat_state=tabmat_state,
        )
        mean_x = centered.mean_x
        data_gram = centered.data_gram
        hessian = centered.hessian
    else:
        mean_x = _stable_signed_mean(dm, observed_w, sum_w)
        data_gram, _ = centered_gram_rhs(
            dm=dm,
            W=observed_w,
            mean_x=mean_x,
            z_centered=np.zeros(dm.n, dtype=np.float64),
        )
        hessian = 0.5 * (data_gram + data_gram.T) + penalty

    decomposition = decompose_gram(hessian)
    if nonnegative and needs_factor_certification(decomposition):
        factor_certified = decompose_factor(
            grouped_augmented_factor(
                dm,
                observed_w,
                penalty,
                center=mean_x,
            )
        )
        if factor_certified.rank != decomposition.rank:
            decomposition = factor_certified

    return ObservedREMLGeometry(
        eta=_readonly(eta),
        mu=_readonly(mu),
        weights=_readonly(observed_w),
        weight_derivative=(_readonly(weight_derivative) if weight_derivative is not None else None),
        weight_second_derivative=(
            _readonly(weight_second_derivative) if weight_second_derivative is not None else None
        ),
        sum_w=sum_w,
        mean_x=_readonly(mean_x),
        centered_data_gram=_readonly(data_gram),
        centered_hessian=_readonly(hessian),
        hessian_inverse=(_readonly(decomposition.pseudo_inverse()) if compute_inverse else None),
        log_det_H=float(np.log(sum_w) + decomposition.log_pdet),
        hessian_rank=1 + decomposition.rank,
    )
