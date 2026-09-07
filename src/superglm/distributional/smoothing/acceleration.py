from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AccelerationRefusalReason = Literal[
    "warming",
    "zero_raw_residual",
    "zero_history_rank",
    "no_model_reduction",
    "current_duplicate",
    "raw_duplicate",
    "box_blocked",
    "nonfinite",
    "non_gfs_proposal",
]


class _NumericalProposalError(Exception):
    pass


def _readonly_copy(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validated_float_vector(values: NDArray, *, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
        if raw.ndim != 1 or np.issubdtype(raw.dtype, np.complexfloating):
            raise ValueError
        with np.errstate(over="raise", invalid="raise"):
            result = np.array(raw, dtype=np.float64, copy=True)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real vector") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    result.setflags(write=False)
    return result


def _scaled_norm(values: NDArray) -> float:
    absolute = np.abs(values)
    if absolute.size == 0:
        return 0.0
    scale = float(np.max(absolute))
    if not np.isfinite(scale):
        raise _NumericalProposalError
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled = absolute / scale
            result = scale * np.sqrt(np.sum(scaled * scaled))
    except FloatingPointError as error:
        raise _NumericalProposalError from error
    if not np.isfinite(result):
        raise _NumericalProposalError
    return float(result)


def _nonnegative_product(*factors: float) -> float:
    mantissa = 1.0
    exponent = 0
    for factor in factors:
        if not np.isfinite(factor) or factor < 0.0:
            raise _NumericalProposalError
        if factor == 0.0:
            return 0.0
        part, part_exponent = np.frexp(factor)
        mantissa = float(np.nextafter(mantissa * part, np.inf))
        exponent += int(part_exponent)
        mantissa, normalization = np.frexp(mantissa)
        exponent += int(normalization)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.ldexp(mantissa, exponent)
    except (FloatingPointError, OverflowError) as error:
        raise _NumericalProposalError from error
    if result == 0.0:
        return float(np.nextafter(0.0, np.inf))
    if not np.isfinite(result):
        raise _NumericalProposalError
    upward = float(np.nextafter(result, np.inf))
    if not np.isfinite(upward):
        raise _NumericalProposalError
    return upward


def _nonnegative_sum(*terms: float) -> float:
    total = 0.0
    for term in terms:
        if not np.isfinite(term) or term < 0.0:
            raise _NumericalProposalError
        try:
            with np.errstate(over="raise", invalid="raise"):
                total += term
        except FloatingPointError as error:
            raise _NumericalProposalError from error
        if total > 0.0:
            total = float(np.nextafter(total, np.inf))
    return total


def _truncated_svd_solution(delta_f: NDArray, residual: NDArray) -> tuple[NDArray[np.float64], int]:
    try:
        raw_matrix = np.asarray(delta_f)
        raw_vector = np.asarray(residual)
        work_dtype = np.result_type(raw_matrix.dtype, raw_vector.dtype, np.float64)
        if not np.issubdtype(work_dtype, np.floating):
            raise TypeError
        matrix = np.asarray(raw_matrix, dtype=work_dtype)
        vector = np.asarray(raw_vector, dtype=work_dtype)
        finite = np.all(np.isfinite(matrix)) and np.all(np.isfinite(vector))
        eps = np.finfo(work_dtype).eps
    except (TypeError, ValueError, OverflowError) as error:
        raise _NumericalProposalError from error
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[0] != vector.size:
        raise _NumericalProposalError
    if not finite:
        raise _NumericalProposalError

    n = max(matrix.shape)
    if n * eps >= 1.0:
        raise _NumericalProposalError
    gamma_n = (n * eps) / (1.0 - n * eps)
    try:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    except (FloatingPointError, np.linalg.LinAlgError, TypeError, ValueError) as error:
        raise _NumericalProposalError from error
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        raise _NumericalProposalError
    cutoff = gamma_n * singular[0]
    keep = singular > cutoff
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return np.zeros(matrix.shape[1], dtype=np.float64), 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            gamma = vh[keep].T @ ((u[:, keep].T @ vector) / singular[keep])
            result = np.asarray(gamma, dtype=np.float64)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as error:
        raise _NumericalProposalError from error
    if not np.all(np.isfinite(result)):
        raise _NumericalProposalError
    return result, rank


def _model_reduction_bound(
    delta_f: NDArray,
    residual: NDArray,
    gamma: NDArray,
    model_residual: NDArray,
) -> float:
    try:
        raw_matrix = np.asarray(delta_f)
        raw_vector = np.asarray(residual)
        raw_coefficients = np.asarray(gamma)
        raw_model = np.asarray(model_residual)
        work_dtype = np.result_type(
            raw_matrix.dtype,
            raw_vector.dtype,
            raw_coefficients.dtype,
            raw_model.dtype,
            np.float64,
        )
        if not np.issubdtype(work_dtype, np.floating):
            raise TypeError
        matrix = np.asarray(raw_matrix, dtype=work_dtype)
        vector = np.asarray(raw_vector, dtype=work_dtype)
        coefficients = np.asarray(raw_coefficients, dtype=work_dtype)
        model = np.asarray(raw_model, dtype=work_dtype)
        eps = np.finfo(work_dtype).eps
    except (TypeError, ValueError, OverflowError) as error:
        raise _NumericalProposalError from error
    if (
        matrix.ndim != 2
        or vector.ndim != 1
        or coefficients.ndim != 1
        or model.ndim != 1
        or matrix.shape != (vector.size, coefficients.size)
        or model.shape != vector.shape
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(model))
    ):
        raise _NumericalProposalError

    m, p = matrix.shape
    if (p + 1) * eps >= 1.0 or (m + 1) * eps >= 1.0:
        raise _NumericalProposalError
    gamma_mv = ((p + 1) * eps) / (1.0 - (p + 1) * eps)
    gamma_norm = ((m + 1) * eps) / (1.0 - (m + 1) * eps)
    try:
        with np.errstate(over="raise", invalid="raise"):
            vector_norm = _scaled_norm(vector)
            matrix_norm = _scaled_norm(matrix)
            coefficient_norm = _scaled_norm(coefficients)
            model_norm = _scaled_norm(model)
            bound = _nonnegative_sum(
                _nonnegative_product(gamma_mv, vector_norm),
                _nonnegative_product(gamma_mv, matrix_norm, coefficient_norm),
                _nonnegative_product(gamma_norm, vector_norm),
                _nonnegative_product(gamma_norm, model_norm),
            )
    except FloatingPointError as error:
        raise _NumericalProposalError from error
    if not np.isfinite(bound):
        raise _NumericalProposalError
    return float(bound)


def _common_scaled_step(
    current: NDArray,
    step: NDArray,
    raw_residual: NDArray,
    *,
    max_log_step: float,
    max_amplification: float,
    lower: float,
    upper: float,
) -> NDArray[np.float64] | None:
    work_dtype = np.result_type(current.dtype, step.dtype, raw_residual.dtype, np.float64)
    current_values = np.asarray(current, dtype=work_dtype)
    step_values = np.asarray(step, dtype=work_dtype)
    residual_values = np.asarray(raw_residual, dtype=work_dtype)
    if (
        current_values.ndim != 1
        or step_values.shape != current_values.shape
        or residual_values.shape != current_values.shape
        or not np.all(np.isfinite(current_values))
        or not np.all(np.isfinite(step_values))
        or not np.all(np.isfinite(residual_values))
        or not np.isfinite(max_log_step)
        or max_log_step <= 0.0
        or not np.isfinite(max_amplification)
        or max_amplification <= 0.0
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower > upper
    ):
        return None
    if current_values.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(current_values < lower) or np.any(current_values > upper):
        return None

    raw_inf = float(np.linalg.norm(residual_values, ord=np.inf))
    trial_inf = float(np.linalg.norm(step_values, ord=np.inf))
    if not np.isfinite(raw_inf) or not np.isfinite(trial_inf):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        amplified_limit = max_amplification * raw_inf
    limit = min(max_log_step, amplified_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        return None
    trust_scale = 1.0 if trial_inf == 0.0 else limit / trial_inf
    if not np.isfinite(trust_scale) or trust_scale <= 0.0:
        return None

    box_scale = 1.0
    for value, direction in zip(current_values, step_values, strict=True):
        if direction > 0.0:
            room = upper - value
            if room <= 0.0:
                return None
            with np.errstate(over="ignore", invalid="ignore"):
                coordinate_scale = room / direction
            box_scale = min(box_scale, float(coordinate_scale))
        elif direction < 0.0:
            room = lower - value
            if room >= 0.0:
                return None
            with np.errstate(over="ignore", invalid="ignore"):
                coordinate_scale = room / direction
            box_scale = min(box_scale, float(coordinate_scale))

    scale = min(1.0, trust_scale, box_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = scale * step_values
        candidate = current_values + scaled
    if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(candidate)):
        return None
    while (
        np.linalg.norm(scaled, ord=np.inf) > limit
        or np.any(candidate < lower)
        or np.any(candidate > upper)
    ):
        smaller = float(np.nextafter(scale, 0.0))
        if smaller <= 0.0 or smaller == scale:
            return None
        scale = smaller
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = scale * step_values
            candidate = current_values + scaled
        if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(candidate)):
            return None
    return np.asarray(scaled, dtype=np.float64)


@dataclass(frozen=True)
class MultisecantProposal:
    log_lambdas: NDArray[np.float64]
    log_step: NDArray[np.float64]
    raw_residual_norm: float
    model_residual_norm: float
    numerical_rank: int
    secant_depth: int


@dataclass(frozen=True)
class MultisecantDecision:
    proposal: MultisecantProposal | None
    refusal_reason: AccelerationRefusalReason | None


@dataclass(frozen=True)
class _AcceptedPair:
    log_lambdas: NDArray[np.float64]
    raw_residual: NDArray[np.float64]


def _validated_provenance(provenance: Hashable) -> int:
    try:
        first_hash = hash(provenance)
        second_hash = hash(provenance)
        first_equal = bool(provenance == provenance)
        second_equal = bool(provenance == provenance)
    except (TypeError, ValueError) as error:
        raise TypeError("provenance must be hashable with scalar equality") from error
    if first_hash != second_hash or not first_equal or not second_equal:
        raise TypeError("provenance must have stable hash and equality")
    return first_hash


def _provenance_equal(left: Hashable, right: Hashable) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError) as error:
        raise TypeError("provenance must have scalar equality") from error


class WindowedTypeIIAnderson:
    def __init__(self, *, history: int, max_amplification: float) -> None:
        if isinstance(history, bool) or not isinstance(history, int):
            raise TypeError("history must be a positive integer")
        if history <= 0:
            raise ValueError("history must be a positive integer")
        if (
            isinstance(max_amplification, (bool, np.bool_))
            or not np.isfinite(max_amplification)
            or max_amplification <= 0.0
        ):
            raise ValueError("max_amplification must be finite and positive")
        self._history = history
        self._max_amplification = float(max_amplification)
        self._pairs: list[_AcceptedPair] = []
        self._provenance: Hashable | None = None
        self._provenance_hash: int | None = None

    def record_accepted(
        self,
        *,
        log_lambdas: NDArray,
        raw_residual: NDArray,
        provenance: Hashable,
    ) -> None:
        provenance_hash = _validated_provenance(provenance)
        log_values = _validated_float_vector(log_lambdas, name="log_lambdas")
        residual_values = _validated_float_vector(raw_residual, name="raw_residual")
        if log_values.shape != residual_values.shape:
            raise ValueError("log_lambdas and raw_residual must be finite equal-shape vectors")

        if self._provenance is not None and (
            provenance_hash != self._provenance_hash
            or not _provenance_equal(self._provenance, provenance)
        ):
            self.reset()
        if self._pairs and self._pairs[-1].log_lambdas.shape != log_values.shape:
            raise ValueError("accepted vectors must keep the same shape")

        self._provenance = provenance
        self._provenance_hash = provenance_hash
        self._pairs.append(
            _AcceptedPair(
                log_lambdas=log_values,
                raw_residual=residual_values,
            )
        )
        del self._pairs[: max(0, len(self._pairs) - self._history - 1)]

    def propose(
        self,
        *,
        max_log_step: float,
        minimum_log_lambda: float,
        maximum_log_lambda: float,
    ) -> MultisecantDecision:
        if not np.isfinite(max_log_step) or max_log_step <= 0.0:
            raise ValueError("max_log_step must be finite and positive")
        if (
            not np.isfinite(minimum_log_lambda)
            or not np.isfinite(maximum_log_lambda)
            or minimum_log_lambda > maximum_log_lambda
        ):
            raise ValueError("log-lambda bounds must be finite and ordered")
        if len(self._pairs) < 2:
            return MultisecantDecision(proposal=None, refusal_reason="warming")

        current = self._pairs[-1]
        if not np.any(current.raw_residual):
            return MultisecantDecision(proposal=None, refusal_reason="zero_raw_residual")
        try:
            raw_norm = _scaled_norm(current.raw_residual)
        except _NumericalProposalError:
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")

        with np.errstate(over="ignore", invalid="ignore"):
            delta_x = np.column_stack(
                [
                    right.log_lambdas - left.log_lambdas
                    for left, right in zip(self._pairs[:-1], self._pairs[1:], strict=True)
                ]
            )
            delta_f = np.column_stack(
                [
                    right.raw_residual - left.raw_residual
                    for left, right in zip(self._pairs[:-1], self._pairs[1:], strict=True)
                ]
            )
        if not np.all(np.isfinite(delta_x)) or not np.all(np.isfinite(delta_f)):
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        try:
            gamma, rank = _truncated_svd_solution(delta_f, current.raw_residual)
        except _NumericalProposalError:
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        if rank == 0:
            return MultisecantDecision(proposal=None, refusal_reason="zero_history_rank")

        with np.errstate(over="ignore", invalid="ignore"):
            model_residual = current.raw_residual - delta_f @ gamma
        try:
            model_norm = _scaled_norm(model_residual)
        except _NumericalProposalError:
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        try:
            reduction_bound = _model_reduction_bound(
                delta_f, current.raw_residual, gamma, model_residual
            )
        except _NumericalProposalError:
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        if raw_norm - model_norm <= reduction_bound:
            return MultisecantDecision(proposal=None, refusal_reason="no_model_reduction")
        with np.errstate(over="ignore", invalid="ignore"):
            type_ii_step = current.raw_residual - (delta_x + delta_f) @ gamma
        if not np.all(np.isfinite(type_ii_step)):
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        log_step = _common_scaled_step(
            current.log_lambdas,
            type_ii_step,
            current.raw_residual,
            max_log_step=max_log_step,
            max_amplification=self._max_amplification,
            lower=minimum_log_lambda,
            upper=maximum_log_lambda,
        )
        if log_step is None:
            return MultisecantDecision(proposal=None, refusal_reason="box_blocked")
        with np.errstate(over="ignore", invalid="ignore"):
            log_lambdas = current.log_lambdas + log_step
            raw_lower = np.maximum(-max_log_step, minimum_log_lambda - current.log_lambdas)
            raw_upper = np.minimum(max_log_step, maximum_log_lambda - current.log_lambdas)
            raw_step = np.clip(current.raw_residual, raw_lower, raw_upper)
            raw_lambdas = current.log_lambdas + raw_step
        if (
            not np.all(np.isfinite(log_step))
            or not np.all(np.isfinite(log_lambdas))
            or not np.all(np.isfinite(raw_lower))
            or not np.all(np.isfinite(raw_upper))
            or not np.all(np.isfinite(raw_step))
            or not np.all(np.isfinite(raw_lambdas))
        ):
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        if np.array_equal(log_lambdas, current.log_lambdas):
            return MultisecantDecision(proposal=None, refusal_reason="current_duplicate")
        accelerated_inf = float(np.linalg.norm(log_step, ord=np.inf))
        raw_step_inf = float(np.linalg.norm(raw_step, ord=np.inf))
        eps = np.finfo(np.float64).eps
        gamma_four = (4.0 * eps) / (1.0 - 4.0 * eps)
        with np.errstate(over="ignore", invalid="ignore"):
            duplicate_bound = gamma_four * (accelerated_inf + raw_step_inf)
            duplicate_distance = float(np.linalg.norm(log_step - raw_step, ord=np.inf))
        if not np.isfinite(duplicate_bound) or not np.isfinite(duplicate_distance):
            return MultisecantDecision(proposal=None, refusal_reason="nonfinite")
        if np.array_equal(log_lambdas, raw_lambdas) or duplicate_distance <= duplicate_bound:
            return MultisecantDecision(proposal=None, refusal_reason="raw_duplicate")

        return MultisecantDecision(
            proposal=MultisecantProposal(
                log_lambdas=_readonly_copy(log_lambdas),
                log_step=_readonly_copy(log_step),
                raw_residual_norm=raw_norm,
                model_residual_norm=model_norm,
                numerical_rank=rank,
                secant_depth=delta_f.shape[1],
            ),
            refusal_reason=None,
        )

    def reject(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pairs.clear()
        self._provenance = None
        self._provenance_hash = None
