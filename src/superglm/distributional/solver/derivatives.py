"""Chain-rule transformation from natural to predictor derivatives.

Families apply observation weights exactly once in
``NaturalLikelihoodEvaluation``.  This layer never receives weights and
therefore cannot apply them again.  It forms the complete signed predictor
Hessian first, then negates exactly once to publish negative curvature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import NaturalLikelihoodEvaluation
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.solver.packing import packed_pairs
from superglm.links import Link


def _readonly_float_array(value: NDArray, *, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PredictorLikelihoodEvaluation:
    """Weighted likelihood state expressed in predictor coordinates."""

    optimizing_log_likelihood: NDArray[np.float64]
    parameter_independent_carrier: NDArray[np.float64]
    score_eta: NDArray[np.float64]
    hessian_eta_packed: NDArray[np.float64]
    curvature_packed: NDArray[np.float64]
    valid: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        optimizing = _readonly_float_array(
            self.optimizing_log_likelihood,
            name="optimizing_log_likelihood",
        )
        carrier = _readonly_float_array(
            self.parameter_independent_carrier,
            name="parameter_independent_carrier",
        )
        score = _readonly_float_array(self.score_eta, name="score_eta")
        hessian = _readonly_float_array(self.hessian_eta_packed, name="hessian_eta_packed")
        curvature = _readonly_float_array(self.curvature_packed, name="curvature_packed")
        if optimizing.ndim != 1 or carrier.ndim != 1 or score.ndim != 2:
            raise ValueError("predictor likelihood arrays have invalid dimensions")
        n_observations, k_parameters = score.shape
        n_channels = k_parameters * (k_parameters + 1) // 2
        if (
            len(optimizing) != n_observations
            or len(carrier) != n_observations
            or hessian.shape != (n_observations, n_channels)
            or curvature.shape != (n_observations, n_channels)
        ):
            raise ValueError("predictor likelihood arrays have inconsistent shapes")
        if not np.array_equal(curvature, -hessian):
            raise ValueError("curvature_packed must negate hessian_eta_packed exactly once")

        valid = self.valid
        if valid is not None:
            valid = np.array(valid, dtype=np.bool_, copy=True)
            if valid.shape != (n_observations,):
                raise ValueError("valid must have shape (n_observations,)")
            valid.setflags(write=False)

        object.__setattr__(self, "optimizing_log_likelihood", optimizing)
        object.__setattr__(self, "parameter_independent_carrier", carrier)
        object.__setattr__(self, "score_eta", score)
        object.__setattr__(self, "hessian_eta_packed", hessian)
        object.__setattr__(self, "curvature_packed", curvature)
        object.__setattr__(self, "valid", valid)

    @property
    def reported_log_likelihood(self) -> NDArray[np.float64]:
        """Normalized row likelihood, including its fixed carrier."""

        return _readonly_float_array(
            self.optimizing_log_likelihood + self.parameter_independent_carrier,
            name="reported_log_likelihood",
        )


def _inverse_link_derivatives(
    link: Link,
    eta: NDArray[np.float64],
    *,
    parameter_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(link, Link):
        raise TypeError(f"link {parameter_index} does not implement Link")
    second_derivative = getattr(link, "deriv2_inverse", None)
    if not callable(second_derivative):
        raise NotImplementedError(
            f"link {parameter_index} must implement deriv2_inverse for observed derivatives"
        )
    first = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    second = np.asarray(second_derivative(eta), dtype=np.float64)
    expected = eta.shape
    if first.shape != expected:
        raise ValueError(
            f"link {parameter_index} deriv_inverse returned shape {first.shape}; expected {expected}"
        )
    if second.shape != expected:
        raise ValueError(
            f"link {parameter_index} deriv2_inverse returned shape {second.shape}; expected {expected}"
        )
    if not np.all(np.isfinite(first)):
        raise ValueError(f"finite values required from link {parameter_index} deriv_inverse")
    if not np.all(np.isfinite(second)):
        raise ValueError(f"finite values required from link {parameter_index} deriv2_inverse")
    return first, second


def _scaled_triple_product(
    left: NDArray[np.float64],
    middle: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multiply finite row vectors without losing range in an intermediate.

    Normal mantissa products incur at most the usual two-multiply relative
    error. Exact rounding at the exponent boundaries preserves subnormals and
    distinguishes final overflow without assuming a wider floating dtype.
    """
    tiny = np.finfo(np.float64).tiny
    upper_boundary = np.finfo(np.float64).max / 2.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        intermediate = left * middle
        result = intermediate * right
    nonzero = (left != 0.0) & (middle != 0.0) & (right != 0.0)
    result[~nonzero] = 0.0
    exceptional = nonzero & (
        ~np.isfinite(intermediate)
        | (np.abs(intermediate) < tiny)
        | ~np.isfinite(result)
        | (np.abs(result) < tiny)
        | (np.abs(result) >= upper_boundary)
    )
    if not np.any(exceptional):
        return result

    left_values = left[exceptional]
    middle_values = middle[exceptional]
    right_values = right[exceptional]
    left_mantissa, left_exponent = np.frexp(left_values)
    middle_mantissa, middle_exponent = np.frexp(middle_values)
    right_mantissa, right_exponent = np.frexp(right_values)
    mantissa = left_mantissa * middle_mantissa * right_mantissa
    exponent = left_exponent + middle_exponent + right_exponent
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.ldexp(mantissa, exponent)
    boundary = ~np.isfinite(scaled) | (np.abs(scaled) < tiny) | (np.abs(scaled) >= upper_boundary)
    for index in np.flatnonzero(boundary):
        exact = (
            Fraction.from_float(float(left_values[index]))
            * Fraction.from_float(float(middle_values[index]))
            * Fraction.from_float(float(right_values[index]))
        )
        try:
            scaled[index] = float(exact)
        except OverflowError as exc:
            raise _NumericalEvaluationError(
                "inverse-link derivative product has no finite floating-point result"
            ) from exc
    result[exceptional] = scaled
    return result


def transform_natural_derivatives(
    evaluation: NaturalLikelihoodEvaluation,
    eta: NDArray,
    links: Sequence[Link],
) -> PredictorLikelihoodEvaluation:
    """Apply inverse-link chain rules to one weighted family evaluation."""
    if not isinstance(evaluation, NaturalLikelihoodEvaluation):
        raise TypeError("evaluation must be a NaturalLikelihoodEvaluation")
    if evaluation.derivative_order != 2:
        raise ValueError("natural derivative transformation requires derivative order 2")
    assert evaluation.score is not None
    assert evaluation.hessian_packed is not None
    eta_array = np.asarray(eta, dtype=np.float64)
    if eta_array.shape != evaluation.score.shape:
        raise ValueError(
            f"eta shape {eta_array.shape} does not match natural score shape {evaluation.score.shape}"
        )
    if not np.all(np.isfinite(eta_array)):
        raise ValueError("eta must contain only finite values")

    link_tuple = tuple(links)
    n_observations, k_parameters = eta_array.shape
    if len(link_tuple) != k_parameters:
        raise ValueError("one link per parameter is required")

    first = np.empty((n_observations, k_parameters), dtype=np.float64)
    second = np.empty_like(first)
    for parameter_index, link in enumerate(link_tuple):
        first[:, parameter_index], second[:, parameter_index] = _inverse_link_derivatives(
            link,
            eta_array[:, parameter_index],
            parameter_index=parameter_index,
        )

    score_eta = evaluation.score * first
    hessian_eta = np.empty_like(evaluation.hessian_packed)
    for packed_index, (left, right) in enumerate(packed_pairs(k_parameters)):
        transformed = _scaled_triple_product(
            evaluation.hessian_packed[:, packed_index], first[:, left], first[:, right]
        )
        if left == right:
            transformed = transformed + evaluation.score[:, left] * second[:, left]
        hessian_eta[:, packed_index] = transformed

    return PredictorLikelihoodEvaluation(
        optimizing_log_likelihood=evaluation.optimizing_log_likelihood,
        parameter_independent_carrier=evaluation.parameter_independent_carrier,
        score_eta=score_eta,
        hessian_eta_packed=hessian_eta,
        curvature_packed=-hessian_eta,
        valid=evaluation.valid,
    )


def transform_natural_information(
    information_packed: NDArray,
    eta: NDArray,
    links: Sequence[Link],
) -> NDArray[np.float64]:
    """Map natural Fisher information to predictor coordinates.

    Expected scores vanish, so only products of first inverse-link
    derivatives enter this congruence transform; there is no observed-score
    diagonal chain term.
    """
    eta_array = np.asarray(eta, dtype=np.float64)
    if eta_array.ndim != 2 or not np.all(np.isfinite(eta_array)):
        raise ValueError("eta must be a finite two-dimensional array")
    n_observations, k_parameters = eta_array.shape
    link_tuple = tuple(links)
    if len(link_tuple) != k_parameters:
        raise ValueError("one link per parameter is required")
    information = np.asarray(information_packed, dtype=np.float64)
    expected_shape = (n_observations, k_parameters * (k_parameters + 1) // 2)
    if information.shape != expected_shape or not np.all(np.isfinite(information)):
        raise ValueError(f"information_packed must be finite with shape {expected_shape}")

    first = np.empty((n_observations, k_parameters), dtype=np.float64)
    for parameter_index, link in enumerate(link_tuple):
        if not isinstance(link, Link):
            raise TypeError(f"link {parameter_index} does not implement Link")
        derivative = np.asarray(
            link.deriv_inverse(eta_array[:, parameter_index]),
            dtype=np.float64,
        )
        if derivative.shape != (n_observations,) or not np.all(np.isfinite(derivative)):
            raise ValueError(
                f"link {parameter_index} deriv_inverse must return a finite row vector"
            )
        first[:, parameter_index] = derivative

    transformed = np.empty_like(information)
    for packed_index, (left, right) in enumerate(packed_pairs(k_parameters)):
        transformed[:, packed_index] = _scaled_triple_product(
            information[:, packed_index], first[:, left], first[:, right]
        )
    return _readonly_float_array(transformed, name="predictor_information_packed")
