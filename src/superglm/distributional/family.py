"""Family metadata and natural-parameter likelihood contracts."""

from __future__ import annotations

import copy
import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link

CurvatureKind = Literal["observed", "fisher", "hybrid"]


@dataclass(frozen=True)
class ObservationContract:
    """Versioned shape of one family observation."""

    kind: Literal["complete"]
    schema_version: int

    def __post_init__(self) -> None:
        if self.kind != "complete" or self.schema_version != 1:
            raise UnsupportedLikelihoodContractError(
                "only complete observations under schema version 1 are supported"
            )


COMPLETE_OBSERVATION = ObservationContract(kind="complete", schema_version=1)


@runtime_checkable
class FamilyLikelihoodPlan(Protocol):
    """Family-bound likelihood law and immutable positional weights."""

    @property
    def weights(self) -> ResolvedLikelihoodWeights: ...

    @property
    def plan_identifier(self) -> str: ...

    def take(self, indices: NDArray[np.integer]) -> FamilyLikelihoodPlan: ...


@runtime_checkable
class LikelihoodPlanValidatingFamily(Protocol):
    """Optional one-shot validation for a family-owned likelihood plan.

    The returned canonical response must be an exact-shape, finite, family-owned
    read-only float64 array, as required by the chunk dispatcher.
    """

    def validate_likelihood_plan(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]: ...


def _readonly_float_array(value: NDArray, *, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ParameterSupport:
    """Open or closed scalar bounds for one natural parameter."""

    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = False
    upper_inclusive: bool = False

    def __post_init__(self) -> None:
        for name in ("lower_inclusive", "upper_inclusive"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when specified")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("lower must be strictly less than upper")

    def contains(self, values: NDArray) -> NDArray[np.bool_]:
        """Return the elementwise finite-support mask."""
        array = np.asarray(values)
        valid = np.isfinite(array)
        if self.lower is not None:
            lower_ok = array >= self.lower if self.lower_inclusive else array > self.lower
            valid &= lower_ok
        if self.upper is not None:
            upper_ok = array <= self.upper if self.upper_inclusive else array < self.upper
            valid &= upper_ok
        return valid


@dataclass(frozen=True)
class ParameterSpec:
    """Ordered metadata for one family-defined natural parameter."""

    name: str
    default_link: str | Link
    role: str
    support: ParameterSupport
    curvature: CurvatureKind = "observed"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or keyword.iskeyword(self.name)
        ):
            raise ValueError("parameter name must be a non-keyword identifier")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("parameter role must be a non-empty string")
        if not isinstance(self.support, ParameterSupport):
            raise TypeError("support must be a ParameterSupport")
        if self.curvature not in ("observed", "fisher", "hybrid"):
            raise ValueError(f"invalid curvature declaration: {self.curvature!r}")
        if isinstance(self.default_link, str):
            if not self.default_link.strip():
                raise ValueError("default_link string must not be empty")
        elif not isinstance(self.default_link, Link):
            raise TypeError("default_link must be a link name or Link")


@dataclass(frozen=True)
class FamilyCapabilities:
    """Operations and derivative orders implemented by a family."""

    max_derivative_order: int
    expected_information: bool
    cdf: bool
    quantile: bool
    random: bool
    response_mean: bool
    censored_response: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_derivative_order, bool)
            or not isinstance(self.max_derivative_order, int)
            or self.max_derivative_order < 0
        ):
            raise ValueError("max_derivative_order must be a non-negative integer")
        for name in (
            "expected_information",
            "cdf",
            "quantile",
            "random",
            "response_mean",
            "censored_response",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class InitialParameterState:
    """Owned, finite natural-parameter initialization for all rows."""

    theta: NDArray
    coefficient_hints: tuple[NDArray | None, ...] | None = None

    def __post_init__(self) -> None:
        theta = _readonly_float_array(self.theta, name="theta")
        if theta.ndim != 2:
            raise ValueError("theta must be two-dimensional")
        if theta.shape[1] < 1:
            raise ValueError("theta must contain at least one parameter")
        object.__setattr__(self, "theta", theta)

        if self.coefficient_hints is not None:
            hints: list[NDArray | None] = []
            for index, hint in enumerate(self.coefficient_hints):
                if hint is None:
                    hints.append(None)
                    continue
                owned = _readonly_float_array(hint, name=f"coefficient_hints[{index}]")
                if owned.ndim != 1:
                    raise ValueError("each coefficient hint must be one-dimensional")
                hints.append(owned)
            object.__setattr__(self, "coefficient_hints", tuple(hints))

    def validate_shape(self, *, n_observations: int, k_parameters: int) -> None:
        """Require the initialization to match the current chunk contract."""
        expected = (n_observations, k_parameters)
        if self.theta.shape != expected:
            raise ValueError(f"theta shape {self.theta.shape} does not match {expected}")
        if self.coefficient_hints is not None and len(self.coefficient_hints) != k_parameters:
            raise ValueError("coefficient_hints must have one entry per parameter")


@dataclass(frozen=True)
class NaturalLikelihoodEvaluation:
    """Weighted likelihood derivatives with a raw signed Hessian.

    ``hessian_packed`` is exactly the raw signed Hessian of the weighted
    per-observation log likelihood in canonical upper-triangular order.  It is
    neither negated curvature nor Fisher information. Absent score and Hessian
    arrays represent exact derivative orders zero and one, respectively.
    """

    optimizing_log_likelihood: NDArray
    parameter_independent_carrier: NDArray
    score: NDArray | None
    hessian_packed: NDArray | None
    valid: NDArray | None = None

    def __post_init__(self) -> None:
        optimizing = _readonly_float_array(
            self.optimizing_log_likelihood,
            name="optimizing_log_likelihood",
        )
        carrier = _readonly_float_array(
            self.parameter_independent_carrier,
            name="parameter_independent_carrier",
        )
        if optimizing.ndim != 1:
            raise ValueError("optimizing_log_likelihood must be one-dimensional")
        if carrier.ndim != 1:
            raise ValueError("parameter_independent_carrier must be one-dimensional")
        n_observations = len(optimizing)
        if len(carrier) != n_observations:
            raise ValueError("likelihood arrays must have the same row count")

        score = None
        if self.score is not None:
            score = _readonly_float_array(self.score, name="score")
            if score.ndim != 2:
                raise ValueError("score must be two-dimensional")
            if score.shape[1] < 1:
                raise ValueError("score must contain at least one parameter")
            if score.shape[0] != n_observations:
                raise ValueError("likelihood derivative arrays must have the same row count")

        hessian = None
        if self.hessian_packed is not None:
            if score is None:
                raise ValueError("hessian_packed requires a score")
            hessian = _readonly_float_array(self.hessian_packed, name="hessian_packed")
            if hessian.ndim != 2:
                raise ValueError("hessian_packed must be two-dimensional")
            expected_channels = score.shape[1] * (score.shape[1] + 1) // 2
            if hessian.shape[0] != n_observations:
                raise ValueError("likelihood derivative arrays must have the same row count")
            if hessian.shape[1] != expected_channels:
                raise ValueError(
                    f"hessian_packed has {hessian.shape[1]} packed channels; "
                    f"expected {expected_channels}"
                )

        valid = self.valid
        if valid is not None:
            valid_array = np.asarray(valid)
            if valid_array.dtype != np.bool_:
                raise TypeError("valid must be a boolean array")
            if valid_array.shape != (n_observations,):
                raise ValueError("valid must have shape (n_observations,)")
            valid_array = np.array(valid_array, dtype=bool, copy=True)
            valid_array.setflags(write=False)
            valid = valid_array

        object.__setattr__(self, "optimizing_log_likelihood", optimizing)
        object.__setattr__(self, "parameter_independent_carrier", carrier)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "hessian_packed", hessian)
        object.__setattr__(self, "valid", valid)

    @property
    def derivative_order(self) -> int:
        """Highest natural derivative represented by this result."""

        if self.score is None:
            return 0
        if self.hessian_packed is None:
            return 1
        return 2

    @property
    def reported_log_likelihood(self) -> NDArray[np.float64]:
        """Normalized row likelihood, including its fixed carrier."""

        return _readonly_float_array(
            self.optimizing_log_likelihood + self.parameter_independent_carrier,
            name="reported_log_likelihood",
        )

    def log_likelihood_delta(self, reference: NaturalLikelihoodEvaluation) -> float:
        """Sum rowwise optimizing differences without subtracting large totals."""

        if not isinstance(reference, NaturalLikelihoodEvaluation):
            raise TypeError("reference must be a NaturalLikelihoodEvaluation")
        if reference.optimizing_log_likelihood.shape != self.optimizing_log_likelihood.shape:
            raise ValueError("likelihood deltas require matching row shapes")
        delta = np.sum(
            self.optimizing_log_likelihood - reference.optimizing_log_likelihood,
            dtype=np.float64,
        )
        if not np.isfinite(delta):
            raise ValueError("optimizing likelihood delta must be finite")
        return float(delta)


@runtime_checkable
class DistributionalFamily(Protocol):
    """Array-oriented family boundary for a regular multi-parameter likelihood."""

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]: ...

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> FamilyLikelihoodPlan: ...

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState: ...

    def evaluate_natural(
        self,
        y: NDArray,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        """Evaluate one likelihood target; derivative order only controls work.

        At fixed rows, parameters and plan, order zero must return the same
        optimizing value and carrier as order two wherever the latter is valid.
        A value-only screen must not refuse a point that has valid derivatives.
        """
        ...


@runtime_checkable
class ConfigurableDistributionalFamily(Protocol):
    """Optional complete-fit configuration metadata."""

    def to_config(self) -> Mapping[str, object]: ...


@runtime_checkable
class DefaultPredictionFamily(Protocol):
    """Optional default response prediction from natural parameters."""

    @property
    def default_prediction_name(self) -> str: ...

    def default_prediction(self, theta: NDArray) -> NDArray: ...


@runtime_checkable
class DistributionFunctionFamily(Protocol):
    """Optional row-wise distribution function and quantile from natural parameters.

    ``theta`` is the ``(n, k)`` natural-parameter matrix; ``y`` and ``p``
    broadcast to ``n``.  ``p`` is strictly inside ``(0, 1)``.
    """

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray: ...

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray: ...


@runtime_checkable
class PriorWeightedDistributionFunctionFamily(Protocol):
    """Optional row law under non-unit prior weights.

    A prior weight is part of the row's own distribution -- it scales the
    Gaussian variance as ``sigma^2 / w``, the gamma shape and scale, the
    Tweedie rate and its compound scale -- so the probability-integral
    transform of a prior-weighted row is not the unit-weight one.  ``weights``
    is the positive prior weight of each row and broadcasts to ``n``; a family
    that cannot represent the weighted law simply does not implement this
    protocol, and the caller refuses rather than silently reading the
    unit-weight distribution function.
    """

    def cdf_prior_weighted(self, y: NDArray, theta: NDArray, weights: NDArray) -> NDArray: ...

    def quantile_prior_weighted(self, p: NDArray, theta: NDArray, weights: NDArray) -> NDArray: ...


@runtime_checkable
class ExpectedShortfallFamily(Protocol):
    """Optional row-wise upper conditional tail mean from natural parameters.

    ``expected_shortfall(p, theta)`` is ``E[Y | Y > q_p]`` for each row and
    ``p`` is strictly inside ``(0, 1)``.  This is independent of
    :class:`DistributionFunctionFamily`: endpoint-singular quantile quadrature
    is not a certified implementation of a family's tail mean.
    """

    def expected_shortfall(self, p: NDArray, theta: NDArray) -> NDArray: ...


@runtime_checkable
class PriorWeightedExpectedShortfallFamily(Protocol):
    """Optional upper conditional tail mean under non-unit prior weights.

    A prior weight changes the row law for reproductive families, so the
    weighted tail mean is a separate structural capability.  Callers refuse
    when this method is absent rather than substituting the unit-weight law.
    """

    def expected_shortfall_prior_weighted(
        self,
        p: NDArray,
        theta: NDArray,
        weights: NDArray,
    ) -> NDArray: ...


@runtime_checkable
class VarianceFamily(Protocol):
    """Optional closed-form predictive variance of one row from its parameters.

    A family that knows ``Var(Y | theta)`` in closed form spares every caller
    that needs a second moment -- an actual-versus-expected standard error, a
    sharpness table -- the cost and the Monte-Carlo error of simulating one.
    The method is the law at unit prior weight; the weighted law is the
    business of :class:`PriorWeightedVarianceFamily`.
    """

    def variance(self, theta: NDArray) -> NDArray: ...


@runtime_checkable
class PriorWeightedVarianceFamily(Protocol):
    """Optional closed-form predictive variance under non-unit prior weights.

    A prior weight is part of the row's own law, so a row at a fifth of a
    year's exposure does not have the variance of a full year's.  For the
    reproductive families the weight divides it -- ``sigma^2 / w`` for the
    Gaussian, ``phi mu^p / w`` for the Tweedie -- but that is a property of the
    family and not a rule a caller may apply on its own, which is why this is
    a separate protocol from :class:`VarianceFamily`: a family that knows the
    unit-weight variance and has no weighted law implements only the first.
    """

    def variance_prior_weighted(self, theta: NDArray, weights: NDArray) -> NDArray: ...


@runtime_checkable
class AtomFamily(Protocol):
    """A family with point masses: the CDF's left limit at ``y`` for randomised PIT.

    ``weights`` carries the positive prior weights when the caller is working
    under a non-unit prior-weight contract and is ``None`` otherwise; an
    implementation whose atoms do not move with the prior weight accepts the
    keyword and ignores it.
    """

    def cdf_left_limit(
        self,
        y: NDArray,
        theta: NDArray,
        weights: NDArray | None = None,
    ) -> NDArray: ...


@runtime_checkable
class FitFailureDiagnosingFamily(Protocol):
    """Optional diagnosis after repeated complete-fit curvature failure."""

    def diagnose_repeated_curvature_failure(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
    ) -> Exception | None: ...


@runtime_checkable
class ExpectedInformationFamily(Protocol):
    """Optional family capability for natural-scale Fisher information."""

    def expected_information_natural(
        self,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray: ...


@runtime_checkable
class PredictorCurvatureDirectionalFamily(Protocol):
    """Optional exact directional derivative of observed predictor curvature."""

    def predictor_curvature_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        eta_direction: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray: ...


@runtime_checkable
class ResponseBoundaryFamily(Protocol):
    """Optional declaration of the response boundaries a predictor can escape to.

    One tuple per natural parameter, in ``parameters`` order, each a subset of
    ``("zero", "one")``.  A categorical level on that predictor whose rows all
    sit on a listed boundary has no finite maximum-likelihood effect: the
    likelihood increases as the predictor walks to infinity.  An empty inner
    tuple means that predictor cannot separate under the supplied link.
    """

    def response_boundaries(self, links: Sequence[Link]) -> tuple[tuple[str, ...], ...]: ...


def _validated_complete_fit_configuration(
    family: object,
) -> Mapping[str, object]:
    serializer = getattr(family, "to_config", None)
    if not callable(serializer) or not isinstance(family, ConfigurableDistributionalFamily):
        raise TypeError(
            "complete-fit distributional family must implement "
            "ConfigurableDistributionalFamily.to_config()"
        )
    raw = serializer()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("family to_config() must return a non-empty mapping")
    config = copy.deepcopy(dict(raw))
    if any(not isinstance(key, str) or not key for key in config):
        raise ValueError("family configuration keys must be non-empty strings")
    return MappingProxyType(config)


def _validated_derivative_order(value: object) -> int:
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or not 0 <= int(value) <= 2
    ):
        raise ValueError("derivative_order must be an integer from zero through two")
    return int(value)


def _validated_parameter_matrix(
    value: NDArray,
    *,
    n_observations: int | None,
    parameters: tuple[ParameterSpec, ...],
    family_name: str,
) -> NDArray[np.float64]:
    try:
        theta = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{family_name} theta must be a finite parameter matrix") from exc
    expected_rows = theta.shape[0] if n_observations is None and theta.ndim == 2 else n_observations
    expected = (expected_rows, len(parameters))
    if theta.ndim != 2 or theta.shape != expected or theta.shape[0] == 0:
        raise ValueError(f"{family_name} theta must have shape {expected}; got {theta.shape}")
    for index, parameter in enumerate(parameters):
        if not np.all(parameter.support.contains(theta[:, index])):
            raise ValueError(
                f"{family_name} parameter {parameter.name!r} is outside its finite support"
            )
    return theta


validated_derivative_order = _validated_derivative_order
validated_parameter_matrix = _validated_parameter_matrix


def validate_family(family: DistributionalFamily) -> tuple[ParameterSpec, ...]:
    """Validate static family metadata and return its ordered parameters."""
    if not isinstance(family, DistributionalFamily):
        raise TypeError("family does not implement DistributionalFamily")
    parameters = family.parameters
    if not isinstance(parameters, tuple) or not parameters:
        raise ValueError("family parameters must be a non-empty tuple")
    if not all(isinstance(parameter, ParameterSpec) for parameter in parameters):
        raise TypeError("family parameters must contain only ParameterSpec values")
    names = tuple(parameter.name for parameter in parameters)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate family parameter name: {', '.join(duplicates)}")
    return parameters
