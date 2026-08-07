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
from superglm.reml.penalty_algebra import total_penalty_matvec
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    build_centered_system,
    grouped_augmented_factor,
)
from superglm.solvers.hessian_factor import HessianFactor
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification
from superglm.solvers.structured import (
    BlockSchurFactor,
    CenteredBlockOperator,
    CompactSymmetricOperator,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
    build_augmented_structured_factor,
    build_penalized_structured_operator,
    build_structured_system,
    compact_operator_diagonal,
    get_structured_layout,
)
from superglm.solvers.sum_to_zero import (
    ProfiledSumToZeroBlockFactor,
    SumToZeroBlockFactor,
)
from superglm.types import GroupSlice, PenaltyComponent


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


def _deriv4_inverse(link: Any, eta: NDArray) -> NDArray:
    """Return an exact fourth inverse-link derivative for built-ins or protocols."""
    link_type = type(link)
    if link_type is LogLink:
        result = np.exp(eta)
    elif link_type is IdentityLink or link_type is SqrtLink:
        result = np.zeros_like(eta)
    elif link_type is LogitLink:
        from scipy.special import expit

        probability = expit(eta)
        result = (
            probability
            * (1.0 - probability)
            * (1.0 - 14.0 * probability + 36.0 * probability**2 - 24.0 * probability**3)
        )
    elif link_type is ProbitLink:
        from scipy.stats import norm

        result = (3.0 * eta - eta**3) * norm.pdf(eta)
    elif link_type is CloglogLink:
        exp_eta = np.exp(eta)
        result = exp_eta * np.exp(-exp_eta) * (1.0 - 7.0 * exp_eta + 6.0 * exp_eta**2 - exp_eta**3)
    elif link_type is CauchitLink:
        result = 24.0 * eta * (1.0 - eta**2) / (np.pi * (1.0 + eta**2) ** 4)
    elif link_type is InverseLink:
        result = 24.0 / eta**5
    elif link_type is InverseSquaredLink:
        result = 6.5625 * eta ** (-4.5)
    elif link_type is PowerLink:
        exponent = 1.0 / float(link.power)
        result = (
            exponent
            * (exponent - 1.0)
            * (exponent - 2.0)
            * (exponent - 3.0)
            * np.maximum(eta, 1e-15) ** (exponent - 4.0)
        )
    elif link_type is NegativeBinomialLink:
        exp_eta = np.exp(np.clip(eta, -30.0, -1e-10))
        result = (
            float(link.theta)
            * exp_eta
            * (1.0 + 11.0 * exp_eta + 11.0 * exp_eta**2 + exp_eta**3)
            / (1.0 - exp_eta) ** 5
        )
    else:
        protocol = getattr(link, "deriv4_inverse", None)
        if not callable(protocol):
            raise NotImplementedError(
                "exact second observed-weight derivatives require link.deriv4_inverse"
            )
        result = protocol(eta)
    result = np.asarray(result, dtype=np.float64)
    if result.shape != eta.shape or not np.all(np.isfinite(result)):
        raise ValueError("fourth inverse-link derivatives must be finite and match eta")
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
    if result.shape != mu.shape or not np.all(np.isfinite(result)):
        raise ValueError("third variance derivatives must be finite and match mu")
    return result


def _validate_observed_bundle(
    values: tuple[NDArray, NDArray | None, NDArray | None],
) -> tuple[NDArray, NDArray | None, NDArray | None]:
    labels = ("weights", "first derivatives", "second derivatives")
    for label, rows in zip(labels, values, strict=True):
        if rows is not None and not np.all(np.isfinite(rows)):
            raise ValueError(f"observed-information {label} are not finite")
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
        if type(link) not in _BUILTIN_REML_LINKS and not callable(
            getattr(link, "deriv4_inverse", None)
        ):
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

    Canonical/equal-curvature combinations reuse the solver geometry and avoid
    an unnecessary full data pass.  Every other supported built-in pair uses
    Wood's negative observed likelihood Hessian.
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
    _, derivative, _ = _compute_observed_row_bundle(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
        derivative_order=1,
    )
    assert derivative is not None
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

    Production LAML always requests the exact analytic bundle.  The explicit
    ``allow_approximate`` escape hatch is retained only for diagnostics with a
    custom family or link that has not implemented the fourth/third derivative
    protocol; built-ins never use it.
    """
    try:
        _, _, derivative = _compute_observed_row_bundle(
            distribution,
            link,
            y,
            mu,
            eta,
            sample_weight,
            derivative_order=2,
        )
    except NotImplementedError:
        if not allow_approximate:
            raise
        y, _, eta, sample_weight = _validate_rows(y, mu, eta, sample_weight)
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
    assert derivative is not None
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
    centered_data_gram: NDArray | CompactSymmetricOperator
    centered_hessian: NDArray | CompactSymmetricOperator
    hessian_inverse: NDArray | HessianFactor | None
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class ObservedModeScore:
    """Penalized likelihood score and a unitless KKT residual."""

    intercept: float
    slopes: NDArray
    max_abs: float
    relative_max: float


class ObservedModeNotCertifiedError(RuntimeError):
    """The penalized mode is not accurate enough to differentiate through.

    Raised where observed REML geometry needs implicit differentiation and the
    achieved mode score misses the certification bar. This is a statement about
    one ``(data, power, lambda)`` point, not about the model: a power search can
    meet the bar comfortably across most of its range and miss it near a bound
    where the working weights are worse conditioned. Callers that evaluate many
    such points should treat it as an infeasible point rather than a failure,
    which is why it is distinguishable from a bare ``RuntimeError``.

    Subclasses ``RuntimeError`` so existing handlers keep working.
    """

    def __init__(self, relative_max: float, tolerance: float, *, hint: str = "") -> None:
        self.relative_max = float(relative_max)
        self.tolerance = float(tolerance)
        self.hint = hint
        # The body stays family-agnostic. Observed geometry is the DEFAULT here
        # -- only canonical-link pairs take the Fisher branch -- so this reaches
        # gamma/log, gaussian/log, poisson/sqrt, nb2/log, binomial/probit and
        # every custom link. Advice naming one family's parameters would be
        # wrong for most callers who see it, so a family-specific `hint` is
        # supplied by the raise site, which holds the distribution.
        #
        # Deliberately absent everywhere: "loosen or tighten `tol`". The achieved
        # score is set by conditioning and does not move with it, and tightening
        # used to make matters strictly worse by raising this bar in step.
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


class ObservedModeNotConvergedError(ObservedModeNotCertifiedError):
    """PIRLS reached no penalized mode at all at this point.

    The sibling condition to certification failure: there is nothing to
    certify. To a power search the two are one situation -- this point has no
    usable penalized mode -- so this subclasses the certification error and
    every handler that routes the parent around routes this too. It is NOT a
    plain ``RuntimeError`` sibling, deliberately: ``optimize_direct_reml``
    raises bare ``RuntimeError`` for genuine invariant violations that must
    propagate, so no caller should ever be tempted into a blanket catch.
    """

    def __init__(self, message: str = "", *, hint: str = "") -> None:
        # No mode exists, so no score was achieved. The attributes exist so
        # parent-typed handlers can format them; non-finite is the signal to
        # describe the condition rather than quote a score.
        self.relative_max = float("inf")
        self.tolerance = float("nan")
        self.hint = hint
        body = message or (
            "observed REML requires a converged penalized coefficient mode "
            "and PIRLS did not reach one at this point."
        )
        if hint:
            body = f"{body}\n  {hint}"
        RuntimeError.__init__(self, body)


def mode_certification_hint(distribution: Any) -> str:
    """Name a parameterisation the caller chose and could change, if one exists.

    Returns empty for families with no such knob. A caller staring at a
    conditioning failure is better served by silence than by a plausible remedy
    that has not been shown to work.
    """
    if isinstance(distribution, Tweedie):
        return (
            "Tweedie conditioning worsens as p approaches 2. `estimate_p()` scores "
            "uncertifiable powers infeasible and searches the rest, rather than "
            "requiring a workable p to be found by hand."
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
    if not np.all(np.isfinite(row_score)):
        raise ValueError("penalized mode score is not finite")

    intercept_score = float(np.sum(row_score, dtype=np.float64))
    centered_diagonal = (
        np.diag(geometry.centered_data_gram)
        if isinstance(geometry.centered_data_gram, np.ndarray)
        else compact_operator_diagonal(geometry.centered_data_gram)
    )
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
    penalty_score = (
        penalty @ result.beta
        if penalty is not None
        else total_penalty_matvec(
            result.beta,
            lambdas,
            reml_penalties,
            list(dm.group_matrices),
        )
    )
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
    penalty: NDArray | None,
    tabmat_state: TabmatCenteringState | None = None,
    compute_inverse: bool = True,
    derivative_order: int = 0,
    groups: list[GroupSlice] | None = None,
    lambdas: dict[str, float] | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    structured_group_index: int | None = None,
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
    if penalty is not None:
        penalty = np.asarray(penalty, dtype=np.float64)
        if penalty.shape != (dm.p, dm.p) or not np.all(np.isfinite(penalty)):
            raise ValueError("penalty must be a finite square matrix in slope coordinates")
        penalty = 0.5 * (penalty + penalty.T)
        # S is a Gaussian-prior precision and must be PSD.  The shared centered
        # solver may project tiny round-off in a declared-PSD system, so reject a
        # materially invalid penalty before entering that path.
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
    if beta.shape != (dm.p,) or not np.all(np.isfinite(beta)):
        raise ValueError("result.beta must be a finite vector matching the design columns")
    if not np.isfinite(result.intercept):
        raise ValueError("result.intercept must be finite")

    eta = stabilize_eta(dm.matvec(beta) + result.intercept + offset_arr, link)
    mu = clip_mu(link.inverse(eta), distribution)
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
        raise ValueError("observed intercept curvature must have a positive finite sum")

    if structured_group_index is not None:
        if groups is None or lambdas is None or reml_penalties is None:
            raise RuntimeError("Structured observed geometry inputs were not validated.")
        structured_layout = get_structured_layout(
            dm,
            groups,
            dominant_group_index=structured_group_index,
        )
        system = build_structured_system(
            list(dm.group_matrices),
            groups,
            observed_w,
            np.zeros(dm.n, dtype=np.float64),
            dominant_group_index=structured_group_index,
            layout=structured_layout,
        )
        penalized = build_penalized_structured_operator(
            system,
            list(dm.group_matrices),
            groups,
            lambdas,
            reml_penalties=reml_penalties,
        )
        xtw = np.empty(dm.p, dtype=np.float64)
        xtw[system.operator.small_indices] = system.xtw_small
        xtw[system.operator.structured_indices] = system.xtw_structured
        mean_x = xtw / system.sum_w
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
        augmented_factor, _ = build_augmented_structured_factor(system, penalized)
        if isinstance(augmented_factor, SumToZeroBlockFactor):
            if not augmented_factor.public_positive_definite:
                raise ValueError(
                    "terminal observed REML coefficient Hessian is indefinite; "
                    "the fitted coefficients do not define a valid Laplace mode"
                )
            profiled_factor = ProfiledSumToZeroBlockFactor(
                augmented_factor=augmented_factor,
                sum_w=system.sum_w,
                xtw=xtw,
            )
        else:
            schur_eigenvalues = np.linalg.eigvalsh(augmented_factor._Q)
            schur_scale = max(
                float(np.max(np.abs(schur_eigenvalues), initial=0.0)),
                1.0,
            )
            if np.any(schur_eigenvalues < -1e-10 * schur_scale):
                raise ValueError(
                    "terminal observed REML coefficient Hessian is indefinite; "
                    "the fitted coefficients do not define a valid Laplace mode"
                )
            if isinstance(augmented_factor, BlockSchurFactor):
                profiled_factor = ProfiledBlockSchurFactor(
                    augmented_factor=augmented_factor,
                    sum_w=system.sum_w,
                    xtw=xtw,
                )
            elif isinstance(augmented_factor, ScalarSchurFactor):
                profiled_factor = ProfiledScalarSchurFactor(
                    augmented_factor=augmented_factor,
                    sum_w=system.sum_w,
                    xtw=xtw,
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
        if penalty is None:  # pragma: no cover - dense branch invariant
            raise RuntimeError("Dense observed geometry is missing its penalty.")
        mean_x = _stable_signed_mean(dm, observed_w, sum_w)
        data_gram, _ = centered_gram_rhs(
            dm=dm,
            W=observed_w,
            mean_x=mean_x,
            z_centered=np.zeros(dm.n, dtype=np.float64),
        )
        hessian = 0.5 * (data_gram + data_gram.T) + penalty

    try:
        decomposition = decompose_gram(hessian)
    except ValueError as error:
        raise ValueError(
            "terminal observed REML coefficient Hessian is indefinite; "
            "the fitted coefficients do not define a valid Laplace mode"
        ) from error
    if nonnegative and needs_factor_certification(decomposition):
        factor_certified = decompose_factor(
            grouped_augmented_factor(
                dm,
                observed_w,
                penalty,
                center=mean_x,
            )
        )
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
