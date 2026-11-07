"""Observed-information geometry for Wood's LAML/REML criterion.

The coefficient solver may reach the penalized likelihood optimum by Fisher
scoring, but the Laplace determinant and its smoothing-parameter derivatives
must use the negative *observed* likelihood Hessian.  The two coincide for
canonical links and differ for non-canonical links such as Gamma/log.
"""

from __future__ import annotations

import dataclasses
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
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
    _own_deriv4_inverse,
    stabilize_eta,
)
from superglm.reml.penalty_algebra import (
    penalty_component_magnitude_matvec,
    penalty_product_rounding,
    total_penalty_matvec,
)
from superglm.solvers._structured.assembly import (
    build_augmented_structured_factor,
    build_penalized_structured_operator,
)
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
    SumToZeroLeafSystem,
    SumToZeroPenalizedOperator,
    SumToZeroTreeFactor,
)
from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafFactor,
    FactorSmoothLeafSystem,
    FactorSmoothPenalizedOperator,
    ProfiledFactorSmoothLeafFactor,
)
from superglm.solvers._structured.layout import (
    get_structured_layout,
)
from superglm.solvers._structured.moments import (
    NestedStructuredSystem,
    build_structured_system,
)
from superglm.solvers._structured.nested import (
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    compact_operator_diagonal,
)
from superglm.solvers._structured.state import (
    centred_data_operator,
)
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    _InitialDataReuse,
    build_centered_system,
    grouped_augmented_factor,
)
from superglm.solvers.hessian_factor import HessianFactor
from superglm.solvers.mode_score import (
    centre_offset_mean,
    centred_matvec,
    corrected_two_pass_pair,
    dense_columns,
    linear_predictor,
    penalized_mode_residual,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification
from superglm.types import GroupSlice, PenaltyComponent


def _readonly(values: NDArray, *, dtype: Any = np.float64) -> NDArray:
    # Callers pass fresh or already-frozen arrays, so no copy is needed.
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
        raise ObservedGeometryInfeasibleError("observed-information inputs must be finite")
    if np.any(sample_weight < 0.0):
        raise ValueError("sample_weight must be non-negative")
    return y, mu, eta, sample_weight


def _deriv4_inverse(link: Any, eta: NDArray) -> NDArray:
    """Return the link's exact fourth inverse derivative (``link.deriv4_inverse``).

    Every built-in link supplies it; a custom link must implement the
    protocol, and a subclass that overrides ``inverse`` or a lower derivative
    must override it as well (``links._own_deriv4_inverse``).
    """
    protocol = _own_deriv4_inverse(link)
    if protocol is None:
        raise NotImplementedError(
            "exact second observed-weight derivatives require link.deriv4_inverse"
        )
    result = protocol(eta)
    result = np.asarray(result, dtype=np.float64)
    # Every built-in link is elementwise in eta, so a shape mismatch is a bug
    # in a custom link rather than a condition this iterate reached.
    if result.shape != eta.shape:
        raise ValueError("link.deriv4_inverse must return an array shaped like eta")
    if not np.all(np.isfinite(result)):
        raise ObservedGeometryInfeasibleError("fourth inverse-link derivatives must be finite")
    return result


def _variance_third_derivative(distribution: Any, mu: NDArray) -> NDArray:
    """Return an exact third variance derivative for built-ins or protocols."""
    distribution_type = type(distribution)
    if distribution_type in (Gaussian, Poisson, Gamma, NegativeBinomial, Binomial):
        result = np.zeros_like(mu)
    elif distribution_type is Tweedie:
        power = float(distribution.p)
        result = power * (power - 1.0) * (power - 2.0) * mu ** (power - 3.0)
    else:
        protocol = getattr(distribution, "variance_third_derivative", None)
        if not callable(protocol):
            raise NotImplementedError(
                "exact second observed-weight derivatives require "
                "distribution.variance_third_derivative"
            )
        result = protocol(mu)
    result = np.asarray(result, dtype=np.float64)
    # As in `_deriv4_inverse`: a wrong shape is a custom family's bug.
    if result.shape != mu.shape:
        raise ValueError(
            "distribution.variance_third_derivative must return an array shaped like mu"
        )
    if not np.all(np.isfinite(result)):
        raise ObservedGeometryInfeasibleError("third variance derivatives must be finite")
    return result


def _validate_observed_bundle(
    values: tuple[NDArray, NDArray | None, NDArray | None],
) -> tuple[NDArray, NDArray | None, NDArray | None]:
    labels = ("weights", "first derivatives", "second derivatives")
    for label, rows in zip(labels, values, strict=True):
        if rows is not None and not np.all(np.isfinite(rows)):
            raise ObservedGeometryInfeasibleError(f"observed-information {label} are not finite")
    return values


def _compute_observed_row_bundle(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    *,
    derivative_order: int,
) -> tuple[NDArray, NDArray | None, NDArray | None]:
    """Return observed rows and exact eta derivatives through the requested order."""
    if derivative_order not in (0, 1, 2):
        raise ValueError("observed row derivative_order must be 0, 1, or 2")
    validate_observed_derivative_capability(distribution, link, derivative_order)
    y, mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)

    if type(link) is LogLink:
        if type(distribution) is Gamma:
            observed = sample_weight * y / mu
            first = -observed if derivative_order >= 1 else None
            second = observed if derivative_order >= 2 else None
            return _validate_observed_bundle((observed, first, second))
        if type(distribution) is Poisson:
            observed = sample_weight * mu
            first = observed if derivative_order >= 1 else None
            second = observed if derivative_order >= 2 else None
            return _validate_observed_bundle((observed, first, second))
        if type(distribution) is NegativeBinomial and distribution.theta != "auto":
            theta = float(distribution.theta)
            denominator = theta + mu
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                observed = sample_weight * theta * mu * (theta + y) / denominator**2
                ratio = (theta - mu) / denominator
                first = observed * ratio if derivative_order >= 1 else None
                second = (
                    observed * (theta**2 - 4.0 * theta * mu + mu**2) / denominator**2
                    if derivative_order >= 2
                    else None
                )
            return _validate_observed_bundle((observed, first, second))
        if type(distribution) is Tweedie:
            power = float(distribution.p)
            left = 2.0 - power
            right = power - 1.0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mu_left = mu ** (2.0 - power)
                y_right = y * mu ** (1.0 - power)
                observed = sample_weight * (left * mu_left + right * y_right)
                first = (
                    sample_weight * (left**2 * mu_left - right**2 * y_right)
                    if derivative_order >= 1
                    else None
                )
                second = (
                    sample_weight * (left**3 * mu_left + right**3 * y_right)
                    if derivative_order >= 2
                    else None
                )
            return _validate_observed_bundle((observed, first, second))

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
        fisher = u**2 / variance
        noncanonical_factor = u**2 * variance_prime / variance**2 - v / variance
        observed = sample_weight * (fisher + residual * noncanonical_factor)
    if derivative_order == 0:
        return _validate_observed_bundle((observed, None, None))

    if not hasattr(link, "deriv3_inverse"):
        raise NotImplementedError("first observed-weight derivatives require link.deriv3_inverse")
    if not hasattr(distribution, "variance_second_derivative"):
        raise NotImplementedError(
            "first observed-weight derivatives require distribution.variance_second_derivative"
        )
    t = np.asarray(link.deriv3_inverse(eta), dtype=np.float64)
    variance_second = np.asarray(
        distribution.variance_second_derivative(mu),
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fisher_first = 2.0 * u * v / variance - u**3 * variance_prime / variance**2
        factor_first = (
            3.0 * u * v * variance_prime / variance**2
            + u**3 * variance_second / variance**2
            - 2.0 * u**3 * variance_prime**2 / variance**3
            - t / variance
        )
        first = sample_weight * (fisher_first - u * noncanonical_factor + residual * factor_first)
    if derivative_order == 1:
        return _validate_observed_bundle((observed, first, None))

    fourth = _deriv4_inverse(link, eta)
    variance_third = _variance_third_derivative(distribution, mu)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fisher_second = (
            2.0 * (v**2 + u * t) / variance
            - 5.0 * u**2 * v * variance_prime / variance**2
            - u**4 * variance_second / variance**2
            + 2.0 * u**4 * variance_prime**2 / variance**3
        )
        factor_second = (
            -fourth / variance
            + (3.0 * v**2 + 4.0 * u * t) * variance_prime / variance**2
            + 6.0 * u**2 * v * variance_second / variance**2
            + u**4 * variance_third / variance**2
            - 12.0 * u**2 * v * variance_prime**2 / variance**3
            - 6.0 * u**4 * variance_prime * variance_second / variance**3
            + 6.0 * u**4 * variance_prime**3 / variance**4
        )
        second = sample_weight * (
            fisher_second
            - v * noncanonical_factor
            - 2.0 * u * factor_first
            + residual * factor_second
        )
    return _validate_observed_bundle((observed, first, second))


def compute_observed_information_weights(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Return row weights for Wood's negative observed likelihood Hessian."""
    return _compute_observed_row_bundle(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
        derivative_order=0,
    )[0]


def observed_row_error_scale(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    observed: NDArray,
) -> NDArray:
    """Per-row scale ``e_r >= |w_r|`` of the rounding in the observed rows ``observed``.

    A row ``w0 (u^2/V + (y - mu) (a - b))``, ``a = u^2 V'/V^2``, ``b = v/V``,
    is accurate to a few ulps of ``w0 (|u^2/V| + |y - mu| (|a| + |b|))``, not
    of ``|w|``, since the terms can cancel. The closed-form log-link rows are
    sums of non-negative terms, so there ``e = w``. The factor charges its
    error bound with ``sum_r e_r`` (``moments._nested_pass``).
    """
    y, mu, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
    observed = np.asarray(observed, dtype=np.float64)
    magnitude = np.abs(observed)
    if type(link) is LogLink and type(distribution) in (Gamma, Poisson, Tweedie):
        return magnitude
    if (
        type(link) is LogLink
        and type(distribution) is NegativeBinomial
        and distribution.theta != "auto"
    ):
        return magnitude
    u = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    v = np.asarray(link.deriv2_inverse(eta), dtype=np.float64)
    variance = np.maximum(np.asarray(distribution.variance(mu), dtype=np.float64), _VARIANCE_FLOOR)
    variance_prime = np.asarray(distribution.variance_derivative(mu), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fisher = u**2 / variance
        terms = np.abs(u**2 * variance_prime / variance**2) + np.abs(v / variance)
        scale = sample_weight * (np.abs(fisher) + np.abs(y - mu) * terms)
    return np.maximum(scale, magnitude)


REMLCurvature = Literal["fisher", "observed"]

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


# Frames under this directory are superglm's own; a warning skips them to land
# on the caller's line.
_PACKAGE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The 0.41.0 name of ``reml_curvature``, still honoured: same argument, same
# 'fisher' or 'observed' answer.
_LEGACY_CURVATURE_HOOK = "scop_reml_curvature"
_warned_legacy_curvature: set[type] = set()


def _curvature_hook(owner: Any, hook_name: str) -> Any:
    """``owner``'s curvature declaration, reading the 0.41.0 name where only it exists."""
    hook = getattr(owner, hook_name, None)
    if callable(hook) or hook_name != "reml_curvature":
        return hook
    legacy = getattr(owner, _LEGACY_CURVATURE_HOOK, None)
    if not callable(legacy):
        return None
    owner_type = type(owner)
    if owner_type not in _warned_legacy_curvature:
        _warned_legacy_curvature.add(owner_type)
        warnings.warn(
            f"{owner_type.__name__} declares its REML curvature through "
            f"{_LEGACY_CURVATURE_HOOK}, which the retired SCOP shape engine read; the "
            "hook is now reml_curvature, with the same argument and values. Rename "
            f"{_LEGACY_CURVATURE_HOOK} to reml_curvature to silence this warning.",
            FutureWarning,
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    return legacy


def _explicit_curvature_protocol(
    distribution: Any,
    link: Any,
    *,
    hook_name: str,
    description: str,
) -> REMLCurvature | None:
    declarations: list[str] = []
    distribution_hook = _curvature_hook(distribution, hook_name)
    if callable(distribution_hook):
        declarations.append(str(distribution_hook(link)))
    link_hook = _curvature_hook(link, hook_name)
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


def classify_binned_reml_curvature(distribution: Any, link: Any) -> REMLCurvature:
    """The REML curvature of the binned (``discrete=True``) route: a function of family and link.

    Observed where the exact route's is and the pair declares observed rows
    that stay non-negative in floating point (``concave_observed_rows``:
    Gamma/log, Tweedie/log, NB2/log, binomial probit and cloglog), since the
    performance iteration's working linear model needs non-negative weights
    (Wood, Pya & Saefken 2016, JASA 111(516)). Every other pair is Fisher,
    including binomial/log, whose maximum can sit on the mean-space boundary,
    and custom families and links.
    """
    from superglm.solvers.working_rows import concave_observed_rows

    if concave_observed_rows(distribution, link) and (
        classify_reml_curvature(distribution, link) == "observed"
    ):
        return "observed"
    return "fisher"


def validate_observed_derivative_capability(
    distribution: Any,
    link: Any,
    derivative_order: int,
) -> None:
    """Fail before fitting when exact observed rows cannot reach ``derivative_order``."""
    if derivative_order not in (0, 1, 2):
        raise ValueError("observed row derivative_order must be 0, 1, or 2")

    required = [
        (link, "deriv_inverse", "link.deriv_inverse"),
        (link, "deriv2_inverse", "link.deriv2_inverse"),
        (distribution, "variance", "distribution.variance"),
        (distribution, "variance_derivative", "distribution.variance_derivative"),
    ]
    if derivative_order >= 1:
        required.extend(
            [
                (link, "deriv3_inverse", "link.deriv3_inverse"),
                (
                    distribution,
                    "variance_second_derivative",
                    "distribution.variance_second_derivative",
                ),
            ]
        )

    missing = [label for owner, name, label in required if not callable(getattr(owner, name, None))]
    if derivative_order >= 2:
        if type(link) not in _BUILTIN_REML_LINKS and _own_deriv4_inverse(link) is None:
            missing.append("link.deriv4_inverse")
        if type(distribution) not in _BUILTIN_REML_DISTRIBUTIONS and not callable(
            getattr(distribution, "variance_third_derivative", None)
        ):
            missing.append("distribution.variance_third_derivative")

    if missing:
        methods = ", ".join(missing)
        raise NotImplementedError(
            f"exact order-{derivative_order} observed REML rows require {methods}"
        )


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
    centered_data_gram: NDArray | CompactSymmetricOperator
    centered_hessian: NDArray | CompactSymmetricOperator
    hessian_inverse: NDArray | HessianFactor | None
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class ObservedModeScore:
    """Penalized likelihood score and a unitless KKT residual.

    ``relative_max`` is the bar times the largest ``relative / bar_effective``
    over the intercept and identified slopes, so it exceeds the bar exactly
    when the mode is not certified. ``raw_relative_max`` covers every
    coefficient; ``weakly_identified`` holds the slopes flagged and kept;
    ``floor_binding`` says a backward-error floor raised the bar for an
    identified coefficient, which the fit discloses.
    """

    intercept: float
    slopes: NDArray
    max_abs: float
    relative_max: float
    raw_relative_max: float = 0.0
    weakly_identified: tuple[int, ...] = ()
    floor_binding: bool = False


# Observed geometry differentiates implicitly through the penalized mode, so
# it needs a tighter solve than the caller may have asked for, and a fixed bar
# to judge the result against. Both derive from this one ceiling.
OBSERVED_PIRLS_TOL_CEILING = 1e-10


def observed_mode_certification_bar() -> float:
    """The bar at the default REML tolerance: ``mode_score.MODE_CERTIFICATION_BAR``.

    Used only when no ``bar`` is passed. REML callers pass
    ``mode_score.mode_certification_bar(reml_tol)``, one bar for the PIRLS
    stop and the certificate, so ``reml_tol`` moves it and ``pirls_tol`` does
    not.
    """
    from superglm.solvers.mode_score import MODE_CERTIFICATION_BAR

    return MODE_CERTIFICATION_BAR


def stopped_on_iteration_budget(result: Any, link: Any = None) -> bool:
    """Did PIRLS stop short of its certificate without a structural failure?

    True for ``"max_iter"``, ``"step_rejected"`` and ``"score_stagnated"``:
    an uncertified mode the caller discloses as not converged, never refuses.
    So is ``"mean_space_boundary"`` under the identity ``link``: its step
    halving keeps every binomial mean inside (0, 1), so the state is finite.
    Every other termination reason is something the mode score cannot judge
    and keeps its own handling.
    """
    return result.termination_reason in ("max_iter", "step_rejected", "score_stagnated") or (
        result.termination_reason == "mean_space_boundary" and isinstance(link, IdentityLink)
    )


class ObservedGeometryInfeasibleError(ValueError):
    """No usable observed geometry exists AT THIS ITERATE.

    Non-finite observed rows, a non-positive intercept curvature or an
    indefinite penalized Hessian: a shorter step or another lambda can clear
    it, so a sweeping caller routes around the point. A caller-contract
    violation (misshapen design, non-PSD penalty) stays a bare ``ValueError``.
    """


class ObservedModeNotCertifiedError(RuntimeError):
    """The penalized mode is not accurate enough to differentiate through.

    Raised when observed REML geometry needs implicit differentiation and the
    mode score misses the certification bar. It concerns one ``(data, power,
    lambda)`` point, not the model, so a caller evaluating many points treats
    it as an infeasible point. Subclasses ``RuntimeError``.
    """

    def __init__(self, relative_max: float, tolerance: float, *, hint: str = "") -> None:
        self.relative_max = float(relative_max)
        self.tolerance = float(tolerance)
        self.hint = hint
        # Family-agnostic: the raise site, which holds the distribution, adds
        # any family `hint`. No `tol` advice: conditioning sets the score.
        message = (
            "REML could not certify the penalized coefficient mode: score "
            f"{self.relative_max:.3e} exceeds the {self.tolerance:.3e} needed to "
            "differentiate the Laplace determinant through it.\n"
            "  This reflects how this data and parameterisation condition the "
            "penalized fit, not a solver setting -- changing `tol` does not move it."
        )
        if hint:
            message = f"{message}\n  {hint}"
        super().__init__(message)


#: What a power search records when this error carries no more specific reason:
#: PIRLS itself never reached a mode. Most raise sites mean exactly this.
_NO_MODE_REASON = "no converged penalized mode (PIRLS found no mode to certify)"
_NO_MODE_DETAIL = "no converged penalized mode"


class ObservedModeNotConvergedError(ObservedModeNotCertifiedError):
    """No mode score exists here, as opposed to one that missed the bar.

    ``relative_max`` is non-finite here, finite on the parent; readers test
    ``np.isfinite(relative_max)`` to choose between quoting a score and
    describing a condition. A subclass of the certification error, so every
    handler that routes the parent around routes this too, while bare
    ``RuntimeError`` invariant violations still propagate.
    ``infeasible_detail`` names the stage that refused when PIRLS did reach
    a mode (a geometry build, a mode score, the W(rho) correction).
    """

    def __init__(
        self,
        message: str = "",
        *,
        hint: str = "",
        infeasible_detail: str | None = None,
    ) -> None:
        # No score: the attributes exist for parent-typed handlers.
        self.relative_max = float("inf")
        self.tolerance = float("nan")
        self.hint = hint
        # None, not the default string: readers resolve it with `or`.
        self.infeasible_detail = infeasible_detail
        body = message or (
            "observed REML requires a converged penalized coefficient mode "
            "and PIRLS did not reach one at this point."
        )
        if hint:
            body = f"{body}\n  {hint}"
        RuntimeError.__init__(self, body)


def mode_certification_hint(distribution: Any) -> str:
    """Name a parameterisation the caller chose and could change, if one exists.

    Empty for families with no such knob: no untested remedy is offered.
    """
    if isinstance(distribution, Tweedie):
        return (
            "`estimate_p()` scores powers where PIRLS reaches no penalized mode as "
            "infeasible and searches the rest, rather than requiring a workable p "
            "to be found by hand."
        )
    return ""


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
    penalty: NDArray | None,
    geometry: ObservedREMLGeometry,
    lambdas: dict[str, float] | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    excluded: NDArray | tuple[int, ...] = (),
    bar: float | None = None,
) -> ObservedModeScore:
    """Evaluate the full penalized score at a proposed Laplace mode.

    A deviance-change stop does not certify the root accurately enough for
    implicit differentiation. The residual (``mode_score.penalized_mode_residual``)
    is scale-relative in intercept and slopes, with a backward-error floor per
    coefficient; weakly identified coefficients, and the ``excluded`` slopes
    the Laplace approximation leaves out, are flagged and kept out of the
    certificate. PIRLS stops on the same quantity.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape:
        raise ValueError("mode-score rows must match the design")
    if penalty is not None:
        penalty = np.asarray(penalty, dtype=np.float64)
        if penalty.shape != (dm.p, dm.p):
            raise ValueError("mode-score penalty must match slope coordinates")
    elif reml_penalties is None or lambdas is None:
        raise ValueError("mode-score requires either a dense penalty or compact penalty components")
    if geometry.mu.shape != y.shape:
        raise ValueError("observed geometry does not match mode-score rows")

    variance = np.maximum(
        np.asarray(distribution.variance(geometry.mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    dmu_deta = np.asarray(link.deriv_inverse(geometry.eta), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_score = sample_weight * (y - geometry.mu) * dmu_deta / variance
    # An accepted geometry does not bound this quotient: finite observed rows
    # can still overflow residual times derivative (Gaussian/sqrt at eta ~1e150),
    # which means no usable mode at this point, not a malformed call.
    if not np.all(np.isfinite(row_score)):
        raise ObservedGeometryInfeasibleError("penalized mode score is not finite")

    centered_diagonal = (
        np.diag(geometry.centered_data_gram)
        if isinstance(geometry.centered_data_gram, np.ndarray)
        else compact_operator_diagonal(geometry.centered_data_gram)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        centered_scale = np.sqrt(np.abs(centered_diagonal) / geometry.sum_w)
    centered_scale = np.where(np.isfinite(centered_scale), centered_scale, 0.0)
    beta = np.asarray(result.beta, dtype=np.float64)
    group_matrices = list(dm.group_matrices)
    if penalty is not None:
        penalty_score = penalty @ beta
        penalty_magnitude = np.abs(penalty) @ np.abs(beta)
        penalty_curvature = np.diag(penalty).astype(np.float64, copy=True)
    else:
        assert lambdas is not None and reml_penalties is not None
        penalty_score = total_penalty_matvec(beta, lambdas, reml_penalties, group_matrices)
        penalty_magnitude = total_penalty_magnitude_matvec(
            beta, lambdas, reml_penalties, group_matrices
        )
        from superglm.reml.identified import penalty_diagonal

        penalty_curvature = penalty_diagonal(dm.p, lambdas, reml_penalties)

    mean_x = np.asarray(geometry.mean_x, dtype=np.float64)
    fisher = np.maximum(sample_weight * dmu_deta**2 / variance, 0.0)
    fisher = np.where(np.isfinite(fisher), fisher, 0.0)
    factor = geometry.hessian_inverse
    excluded_indices = tuple(getattr(factor, "weakly_identified_slopes", ()) or ())
    excluded_mask = np.zeros(dm.p, dtype=bool)
    excluded_mask[list(excluded_indices)] = True
    excluded_mask[np.asarray(excluded, dtype=np.intp)] = True
    bar = observed_mode_certification_bar() if bar is None else float(bar)
    # The floors read the intercept about mean_x and the centred rows X~ beta.
    # A centred state forms them from its own alpha and (X - 1 c') beta, since
    # the raw ones cancel c' beta at a large offset.
    alpha_c = getattr(result, "centred_intercept", None)
    centre = getattr(result, "state_center", None)
    if alpha_c is None or centre is None:
        shift = float(mean_x @ beta)
        alpha = float(result.intercept) + shift
        eta_tilde = dm.matvec(beta) - shift
    else:
        centre = np.asarray(centre, dtype=np.float64)
        offset_mean = centre_offset_mean(
            dm, geometry.weights, float(geometry.sum_w), centre, mean_x
        )
        shift = float(offset_mean @ beta)
        alpha = float(alpha_c) + shift
        eta_tilde = centred_matvec(dm, beta, centre) - shift
    residual = penalized_mode_residual(
        dm=dm,
        row_score=row_score,
        fisher_weights=fisher,
        positive_prior=sample_weight > 0.0,
        mean_x=mean_x,
        centered_scale=centered_scale,
        alpha=alpha,
        eta_tilde=eta_tilde,
        penalty_score=penalty_score,
        penalty_magnitude=penalty_magnitude,
        penalty_curvature=penalty_curvature,
        sum_w=float(geometry.sum_w),
        bar=bar,
        excluded=excluded_mask,
        penalty_rounding=penalty_product_rounding(
            dm.p, None if penalty is not None else reml_penalties
        ),
    )
    slope_score = residual.slope_score
    max_abs = max(
        abs(residual.intercept_score),
        float(np.max(np.abs(slope_score), initial=0.0)),
    )
    return ObservedModeScore(
        intercept=residual.intercept_score,
        slopes=_readonly(slope_score),
        max_abs=max_abs,
        relative_max=bar * residual.ratio(),
        raw_relative_max=float(np.max(residual.relative, initial=0.0)),
        weakly_identified=tuple(int(index) for index in np.flatnonzero(residual.excluded)),
        floor_binding=residual.floor_binding,
    )


def total_penalty_magnitude_matvec(
    beta: NDArray,
    lambdas: float | dict[str, float],
    penalties: list[PenaltyComponent],
    group_matrices: list,
) -> NDArray:
    """``|S| |beta|`` from compact components, the magnitude ``total_penalty_matvec`` sums."""
    magnitude = np.abs(np.asarray(beta, dtype=np.float64))
    product = np.zeros_like(magnitude)
    for component in penalties:
        lam = lambdas[component.name] if isinstance(lambdas, dict) else lambdas
        if lam == 0:
            continue
        product[component.group_sl] += abs(float(lam)) * penalty_component_magnitude_matvec(
            component,
            magnitude[component.group_sl],
            group_matrices[component.group_index],
        )
    return product


def _stable_signed_mean_pair(
    dm: DesignMatrix, weights: NDArray, sum_w: float
) -> tuple[NDArray, NDArray, NDArray | None]:
    """``(mean_x, hi, lo)``: a signed weighted mean and, on dense columns, its exact pair.

    The mean is ``anchor + lo`` from the corrected two-pass algorithm
    (``mode_score.corrected_two_pass_pair``), never from large raw moments.
    A ``DenseGroupMatrix`` column keeps the pair so its rows centre as
    ``(x - hi) - lo``; other columns centre about the rounded mean
    (``lo = 0``). ``lo`` is ``None`` without a dense column.
    """
    anchor, offset_mean = _stable_signed_offset(dm, weights, sum_w)
    mean = anchor + offset_mean
    dense = dense_columns(dm)
    if not np.any(dense):
        return mean, mean, None
    return mean, np.where(dense, anchor, mean), np.where(dense, offset_mean, 0.0)


def _stable_signed_offset(
    dm: DesignMatrix, weights: NDArray, sum_w: float
) -> tuple[NDArray, NDArray]:
    """``(anchor, lo)``: the signed weighted mean as an exact pair, every column.

    The anchor is the rounded mean from a pass shifted by the first row that
    carries weight, never row 0 as such: a zero-weight row far from the data
    would set the remainder's scale.
    """
    if dm.p == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    chunk_size = 8192

    def chunks():
        for start in range(0, dm.n, chunk_size):
            stop = min(start + chunk_size, dm.n)
            rows = np.arange(start, stop, dtype=np.intp)
            yield start, stop, np.asarray(dm.row_subset(rows).toarray(), dtype=np.float64)

    return corrected_two_pass_pair(chunks, np.asarray(weights, dtype=np.float64), sum_w, dm.p)


def schur_curvature_is_negative(eigenvalues: NDArray, certificate) -> bool:
    """Whether a structured factor's scaled Schur eigenvalues prove negative curvature.

    ``eigenvalues`` are those of the factor's Jacobi-scaled deflated border
    matrix, ``certificate`` its ``BorderCertificate``. An eigenvalue below
    ``-certificate.curvature_floor(||Q_s||_2)`` (the certified uncertainty
    ``tau`` plus eigensolver rounding ``2 width eps ||Q_s||_2``) is material
    negative curvature. Run only on a factor its own check accepted, so each
    refusal certifies an indefinite observed Hessian.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if not values.size:
        return False
    norm = float(np.max(np.abs(values)))
    return bool(np.any(values < -certificate.curvature_floor(norm)))


def _observed_centered_system(
    dm: DesignMatrix,
    weights: NDArray,
    penalty: NDArray,
    tabmat_state: TabmatCenteringState | None,
    reuse: _InitialDataReuse | None,
) -> CenteredSystem:
    """The centred system at the observed weights, its data served by ``reuse`` when it holds them.

    The data of the last build serve a build at bit-identical weights under an equal centring
    state (``_SystemData``): an accepted trial's geometry and the next iteration's candidate
    geometry at the same coefficients read the same system.
    """
    zero = np.zeros(dm.n, dtype=np.float64)
    keep = reuse is not None and tabmat_state is not None
    key = (dm, dm.execution_plan)
    if keep:
        served = reuse.take(key, weights, zero, tabmat_state, penalty)
        if served is not None:
            return served
        before = dataclasses.replace(tabmat_state)
    system = build_centered_system(
        dm=dm,
        W=weights,
        z_off=zero,
        penalty=penalty,
        tabmat_split=dm.tabmat_centering_split,
        tabmat_state=tabmat_state,
    )
    if keep:
        reuse.remember(key, weights, zero, before, tabmat_state, system)
    return system


def build_observed_reml_geometry(
    *,
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result: PIRLSResult,
    penalty: NDArray | None,
    tabmat_state: TabmatCenteringState | None = None,
    compute_inverse: bool = True,
    derivative_order: int = 0,
    groups: list[GroupSlice] | None = None,
    lambdas: dict[str, float] | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    structured_group_index: int | None = None,
    structured_chain_group_indices: tuple[int, ...] = (),
    _data_reuse: _InitialDataReuse | None = None,
) -> ObservedREMLGeometry:
    """Build Wood's observed LAML Hessian without altering fit inference state.

    Non-negative rows use the shared centered-system execution layer; signed
    rows use bounded, compensated centred chunks, and the penalized curvature
    must still be PSD. A random effect's nested chain
    (``structured_chain_group_indices``) builds the nested factor through the
    same refusal seams.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    offset_arr = np.asarray(offset_arr, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape or offset_arr.shape != y.shape:
        raise ValueError("REML geometry row arrays must match the design")
    if penalty is not None:
        penalty = np.asarray(penalty, dtype=np.float64)
        if penalty.shape != (dm.p, dm.p) or not np.all(np.isfinite(penalty)):
            raise ValueError("penalty must be a finite square matrix in slope coordinates")
        penalty = 0.5 * (penalty + penalty.T)
        # S is a Gaussian-prior precision: reject a materially non-PSD penalty
        # before the centred solver projects away round-off.
        decompose_gram(penalty)
    elif (
        structured_group_index is None
        or groups is None
        or lambdas is None
        or reml_penalties is None
    ):
        raise ValueError(
            "compact observed geometry requires groups, lambdas, penalties, "
            "and a structured group index"
        )
    if derivative_order not in (0, 1, 2):
        raise ValueError("derivative_order must be 0, 1, or 2")

    beta = np.asarray(result.beta, dtype=np.float64)
    # A wrong-width beta belongs to another design (a bug); a non-finite one
    # is this iterate diverging, a point to route around.
    if beta.shape != (dm.p,):
        raise ValueError("result.beta must match the design columns")
    if not np.all(np.isfinite(beta)):
        raise ObservedGeometryInfeasibleError("result.beta must be a finite vector")
    if not np.isfinite(result.intercept):
        raise ObservedGeometryInfeasibleError("result.intercept must be finite")

    eta = stabilize_eta(linear_predictor(dm, result, offset_arr), link)
    mu = link.inverse(eta)
    # A binomial/log mode certified on the model's own score
    # (``mean_space_true_mode``) is read at the unclipped mean; only its
    # zero-weight rows keep the clip, whose unclipped mean can pass one.
    if getattr(result, "mean_space_true_mode", False):
        held = ~(np.asarray(sample_weight, dtype=np.float64) > 0.0)
        if np.any(held):
            mu = np.where(held, clip_mu(mu, distribution), mu)
    else:
        mu = clip_mu(mu, distribution)
    observed_w, weight_derivative, weight_second_derivative = _compute_observed_row_bundle(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
        derivative_order=derivative_order,
    )
    nonnegative = bool(np.all(observed_w >= 0.0))
    sum_w = (
        float(np.sum(observed_w, dtype=np.float64))
        if nonnegative
        else _stable_signed_sum(observed_w)
    )
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ObservedGeometryInfeasibleError(
            "observed intercept curvature must have a positive finite sum"
        )

    if structured_group_index is not None:
        if groups is None or lambdas is None or reml_penalties is None:
            raise RuntimeError("Structured observed geometry inputs were not validated.")
        structured_layout = get_structured_layout(
            dm,
            groups,
            dominant_group_index=structured_group_index,
            chain_group_indices=structured_chain_group_indices,
        )
        # Structured operators raise LinAlgError for non-finite blocks (the
        # iterate); their ValueErrors are caller bugs and stay bare.
        try:
            system = build_structured_system(
                list(dm.group_matrices),
                groups,
                observed_w,
                np.zeros(dm.n, dtype=np.float64),
                dominant_group_index=structured_group_index,
                layout=structured_layout,
                prior_weights=sample_weight,
                signed=True,
                error=observed_row_error_scale(
                    distribution, link, y, mu, eta, sample_weight, observed_w
                ),
            )
        except np.linalg.LinAlgError as error:
            raise ObservedGeometryInfeasibleError(
                "structured observed geometry has no usable blocks at the fitted coefficients"
            ) from error
        # The structured moments' own `sum_w` can differ in sign from the one
        # certified above for signed rows; it is the one the centring divides
        # by, so it is tested here rather than by retyping the factors' refusal.
        if not np.isfinite(system.sum_w) or system.sum_w <= 0.0:
            raise ObservedGeometryInfeasibleError(
                "structured observed intercept curvature must have a positive finite sum"
            )
        # The penalized operator has the same block types and the same seam.
        try:
            penalized = build_penalized_structured_operator(
                system,
                list(dm.group_matrices),
                groups,
                lambdas,
                reml_penalties=reml_penalties,
            )
        except np.linalg.LinAlgError as error:
            raise ObservedGeometryInfeasibleError(
                "penalized structured observed geometry has no usable blocks "
                "at the fitted coefficients"
            ) from error
        xtw = np.empty(dm.p, dtype=np.float64)
        xtw[system.operator.small_indices] = system.xtw_small
        xtw[system.operator.structured_indices] = system.xtw_structured
        mean_x = xtw / system.sum_w
        if isinstance(system, FactorSmoothLeafSystem | SumToZeroLeafSystem):
            # On the c0-shifted moments: the mode score's centred scale reads
            # this diagonal, which raw moments cancel to noise at a large offset.
            data_gram = centred_data_operator(system)
            raw = data_gram.raw
            small, local = (
                getattr(penalized, "penalty_small", None),
                getattr(penalized, "penalty_local", None),
            )
            if small is None or local is None:  # pragma: no cover - built with its parts
                raise TypeError("An fs or sz system's penalized operator keeps its penalty parts.")
            if isinstance(raw, SumToZeroBlockOperator):
                penalized_raw = SumToZeroPenalizedOperator.with_penalties(raw, small, local)
            elif isinstance(raw, BlockSymmetricOperator):
                penalized_raw = FactorSmoothPenalizedOperator.with_penalties(raw, small, local)
            else:  # pragma: no cover - structured dispatch invariant
                raise TypeError("An fs or sz centred data operator is a block operator.")
            hessian = dataclasses.replace(data_gram, raw=penalized_raw)
        else:
            data_gram = CenteredBlockOperator(
                raw=system.operator,
                cross=xtw,
                total=system.sum_w,
                center=mean_x,
            )
            hessian = CenteredBlockOperator(
                raw=penalized,
                cross=xtw,
                total=system.sum_w,
                center=mean_x,
            )
        # The factors refuse a level with negative or non-finite curvature by
        # LinAlgError, raised before any structural check (finiteness before
        # symmetry, since NaN fails `np.allclose`). Their plain ValueErrors are
        # misassembled inputs, which the preamble does not check here, so only
        # LinAlgError is retyped as an infeasible point.
        try:
            augmented_factor, _ = build_augmented_structured_factor(system, penalized)
        except np.linalg.LinAlgError as error:
            raise ObservedGeometryInfeasibleError(
                "the structured solver refused the observed REML coefficient Hessian "
                f"({error}); the fitted coefficients do not define a valid Laplace mode"
            ) from error
        # Guard before the eigencheck, so an unknown factor gets a TypeError
        # naming it. Curvature is read on the Jacobi-scaled Schur complement,
        # where the factor takes its rank decisions.
        if not isinstance(  # pragma: no cover - structured dispatch invariant
            augmented_factor,
            FactorSmoothLeafFactor | SumToZeroTreeFactor | NestedSchurFactor,
        ):
            raise TypeError("Unsupported structured observed factor geometry.")
        try:
            schur_eigenvalues = augmented_factor.scaled_schur_eigenvalues()
        except np.linalg.LinAlgError as error:
            # Non-convergence here describes the iterate's curvature: infeasible.
            raise ObservedGeometryInfeasibleError(
                "observed REML Schur complement has no usable eigendecomposition "
                "at the fitted coefficients"
            ) from error
        if schur_curvature_is_negative(schur_eigenvalues, augmented_factor.border_certificate):
            raise ObservedGeometryInfeasibleError(
                "observed REML coefficient Hessian is indefinite; "
                "the fitted coefficients do not define a valid Laplace mode"
            )
        if isinstance(augmented_factor, FactorSmoothLeafFactor):
            profiled_factor = ProfiledFactorSmoothLeafFactor(
                augmented_factor=augmented_factor,
                sum_w=system.sum_w,
                xtw=xtw,
            )
        elif isinstance(augmented_factor, SumToZeroTreeFactor):
            profiled_factor = ProfiledSumToZeroTreeFactor(
                augmented_factor=augmented_factor,
                sum_w=system.sum_w,
                xtw=xtw,
            )
        elif isinstance(system, NestedStructuredSystem):
            profiled_factor = ProfiledNestedSchurFactor(
                augmented_factor=augmented_factor,
                sum_w=system.sum_w,
                xtw=xtw,
                data_operator=system.operator,
            )
        else:  # pragma: no cover - structured dispatch invariant
            raise TypeError("Unsupported structured observed factor geometry.")
        return ObservedREMLGeometry(
            eta=_readonly(eta),
            mu=_readonly(mu),
            weights=_readonly(observed_w),
            weight_derivative=(
                _readonly(weight_derivative) if weight_derivative is not None else None
            ),
            weight_second_derivative=(
                _readonly(weight_second_derivative)
                if weight_second_derivative is not None
                else None
            ),
            sum_w=system.sum_w,
            mean_x=_readonly(mean_x),
            centered_data_gram=data_gram,
            centered_hessian=hessian,
            hessian_inverse=profiled_factor if compute_inverse else None,
            log_det_H=augmented_factor.logdet(),
            hessian_rank=augmented_factor.rank,
        )

    if nonnegative:
        if penalty is None:  # pragma: no cover - dense branch invariant
            raise RuntimeError("Dense observed geometry is missing its penalty.")
        # The preamble has validated every caller input this call checks, so
        # what it can still refuse belongs to the iterate; retyping is safe.
        try:
            centered = _observed_centered_system(dm, observed_w, penalty, tabmat_state, _data_reuse)
        except ValueError as error:
            raise ObservedGeometryInfeasibleError(
                "observed working weights do not define a centered system at these coefficients"
            ) from error
        mean_x = centered.mean_x
        centre, centre_lo = centered.centre_pair()
        data_gram = centered.data_gram
        hessian = centered.hessian
    else:
        if penalty is None:  # pragma: no cover - dense branch invariant
            raise RuntimeError("Dense observed geometry is missing its penalty.")
        mean_x, centre, centre_lo = _stable_signed_mean_pair(dm, observed_w, sum_w)
        # No retype: `centered_gram_rhs` checks only shapes supplied here.
        data_gram, _ = centered_gram_rhs(
            dm=dm,
            W=observed_w,
            mean_x=centre,
            z_centered=np.zeros(dm.n, dtype=np.float64),
            mean_lo=centre_lo,
        )
        hessian = 0.5 * (data_gram + data_gram.T) + penalty

    try:
        decomposition = decompose_gram(hessian)
    except ValueError as error:
        raise ObservedGeometryInfeasibleError(
            "observed REML coefficient Hessian is indefinite; "
            "the fitted coefficients do not define a valid Laplace mode"
        ) from error
    if nonnegative and needs_factor_certification(decomposition):
        # As above: the inputs are validated, so a refusal is the iterate's.
        try:
            factor_certified = decompose_factor(
                grouped_augmented_factor(
                    dm,
                    observed_w,
                    penalty,
                    center=centre,
                    center_lo=centre_lo,
                )
            )
        except ValueError as error:
            raise ObservedGeometryInfeasibleError(
                "terminal observed REML coefficient factor could not be certified; "
                "the fitted coefficients do not define a valid Laplace mode"
            ) from error
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
