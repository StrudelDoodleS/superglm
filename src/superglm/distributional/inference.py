"""Joint covariance and solve-based EDF attribution for distributional fits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import CurvatureKind
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import DenseSolverResult
from superglm.distributional.telemetry import CurvatureTelemetry


def _readonly(values: NDArray, *, name: str) -> NDArray[np.float64]:
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _frozen_mapping(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite numbers")
        frozen[key] = number
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class JointInference:
    """Immutable full-coordinate covariance and qualified EDF attribution."""

    covariance: NDArray[np.float64]
    influence: NDArray[np.float64]
    coefficient_edf: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    total_edf: float
    predictor_edf: Mapping[str, float]
    intercept_edf: Mapping[str, float]
    term_edf: Mapping[str, float]
    covariance_curvature_source: CurvatureKind
    edf_curvature_source: CurvatureKind
    curvature_telemetry: CurvatureTelemetry
    rank: int
    reconciliation_tolerance: float
    slice_reconciliation_error: float
    predictor_reconciliation_error: float

    def __post_init__(self) -> None:
        covariance = _readonly(self.covariance, name="covariance")
        influence = _readonly(self.influence, name="influence")
        coefficient_edf = _readonly(self.coefficient_edf, name="coefficient_edf")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square")
        width = covariance.shape[0]
        if influence.shape != (width, width) or coefficient_edf.shape != (width,):
            raise ValueError("influence and coefficient EDF must match covariance width")
        symmetry_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.linalg.norm(covariance, ord=np.inf)),
            )
        )
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=symmetry_tolerance):
            raise ValueError("covariance must be symmetric")
        if not np.allclose(
            coefficient_edf,
            np.diag(influence),
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        ):
            raise ValueError("coefficient EDF must equal the influence diagonal")

        coefficient_names = tuple(self.coefficient_names)
        if (
            len(coefficient_names) != width
            or len(set(coefficient_names)) != width
            or any(not isinstance(name, str) or not name for name in coefficient_names)
        ):
            raise ValueError("coefficient_names must uniquely name every covariance coordinate")
        total_edf = float(self.total_edf)
        if not math.isfinite(total_edf):
            raise ValueError("total_edf must be finite")
        trace_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(total_edf))
        if abs(total_edf - float(np.trace(influence))) > trace_tolerance:
            raise ValueError("total_edf must equal the influence trace")

        if not isinstance(self.curvature_telemetry, CurvatureTelemetry):
            raise TypeError("curvature_telemetry must be CurvatureTelemetry")
        if self.covariance_curvature_source != self.edf_curvature_source:
            raise ValueError("covariance and EDF must use the same terminal curvature source")
        if self.covariance_curvature_source != self.curvature_telemetry.actual_source:
            raise ValueError("inference source must equal the recorded actual terminal curvature")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank <= width
        ):
            raise ValueError("rank must be a non-negative integer no larger than covariance width")

        tolerance = float(self.reconciliation_tolerance)
        slice_error = float(self.slice_reconciliation_error)
        predictor_error = float(self.predictor_reconciliation_error)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("reconciliation_tolerance must be finite and positive")
        if any(not math.isfinite(value) or value < 0.0 for value in (slice_error, predictor_error)):
            raise ValueError("reconciliation errors must be finite and non-negative")
        if slice_error > tolerance or predictor_error > tolerance:
            raise ValueError("qualified EDF slices do not reconcile to total EDF")

        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "influence", influence)
        object.__setattr__(self, "coefficient_edf", coefficient_edf)
        object.__setattr__(self, "coefficient_names", coefficient_names)
        object.__setattr__(self, "total_edf", total_edf)
        object.__setattr__(
            self, "predictor_edf", _frozen_mapping(self.predictor_edf, name="predictor_edf")
        )
        object.__setattr__(
            self, "intercept_edf", _frozen_mapping(self.intercept_edf, name="intercept_edf")
        )
        object.__setattr__(self, "term_edf", _frozen_mapping(self.term_edf, name="term_edf"))
        object.__setattr__(self, "reconciliation_tolerance", tolerance)
        object.__setattr__(self, "slice_reconciliation_error", slice_error)
        object.__setattr__(self, "predictor_reconciliation_error", predictor_error)

    @property
    def curvature_source(self) -> CurvatureKind:
        """The single terminal curvature source shared by covariance and EDF."""
        return self.covariance_curvature_source

    @property
    def coefficient_edf_by_name(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                name: float(value)
                for name, value in zip(
                    self.coefficient_names,
                    self.coefficient_edf,
                    strict=True,
                )
            }
        )

    @property
    def negative_coefficient_edf(self) -> Mapping[str, float]:
        """Return every retained negative coordinate contribution without clipping."""
        return MappingProxyType(
            {name: value for name, value in self.coefficient_edf_by_name.items() if value < 0.0}
        )


def _global_slice(value: slice, *, width: int, name: str) -> slice:
    if not isinstance(value, slice) or value.step not in (None, 1):
        raise ValueError(f"{name} must use one contiguous global slice")
    start = value.start
    stop = value.stop
    if (
        not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
        or stop > width
    ):
        raise ValueError(f"{name} has an invalid global slice")
    return slice(start, stop)


def _attributed_edf(
    layout: StackedLayout,
    diagonal: NDArray[np.float64],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    width = layout.n_coefficients
    coverage = np.zeros(width, dtype=np.int64)
    intercept_edf: dict[str, float] = {}
    for state in layout.predictors:
        state_slice = _global_slice(
            state.coefficient_slice,
            width=width,
            name=f"predictor {state.name!r}",
        )
        if state.intercept_index is None:
            continue
        if state.intercept_index != state_slice.start:
            raise ValueError("predictor intercept must be first in its global slice")
        coverage[state.intercept_index] += 1
        intercept_edf[f"{state.name}:(intercept)"] = float(diagonal[state.intercept_index])

    term_edf: dict[str, float] = {}
    for name, raw_slice in layout.term_slices.items():
        term_slice = _global_slice(raw_slice, width=width, name=f"term {name!r}")
        predictor_name, separator, _ = name.partition(":")
        if not separator:
            raise ValueError(f"term {name!r} is not predictor-qualified")
        try:
            state = layout.predictor(predictor_name)
        except KeyError as exc:
            raise ValueError(f"term {name!r} has an unknown predictor namespace") from exc
        if (
            term_slice.start < state.coefficient_slice.start
            or term_slice.stop > state.coefficient_slice.stop
        ):
            raise ValueError(f"term {name!r} lies outside its predictor slice")
        coverage[term_slice] += 1
        term_edf[name] = float(np.sum(diagonal[term_slice], dtype=np.float64))

    if width == 0 or np.any(coverage != 1):
        raise ValueError(
            "intercepts and qualified terms must form a complete non-overlapping partition"
        )
    predictor_edf = {
        state.name: float(np.sum(diagonal[state.coefficient_slice], dtype=np.float64))
        for state in layout.predictors
    }
    return predictor_edf, intercept_edf, term_edf


def compute_joint_inference(
    layout: StackedLayout,
    result: DenseSolverResult,
) -> JointInference:
    """Compute stabilized covariance and EDF from one accepted terminal geometry."""
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    if not isinstance(result, DenseSolverResult):
        raise TypeError("result must be a DenseSolverResult")
    width = layout.n_coefficients
    if width <= 0 or result.coefficients.shape != (width,):
        raise ValueError("result coefficients must match a non-empty global layout")
    if result.terminal_rank.width != width:
        raise ValueError("terminal rank decomposition does not match the global layout")

    covariance = result.terminal_pseudo_inverse()
    influence = np.column_stack(
        tuple(
            result.solve_terminal(result.terminal_data_curvature[:, column])
            for column in range(width)
        )
    )
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(influence)):
        raise ValueError("terminal rank solve produced non-finite inference state")
    diagonal = np.diag(influence).copy()
    total_edf = float(np.trace(influence))
    predictor_edf, intercept_edf, term_edf = _attributed_edf(layout, diagonal)

    scale = max(1.0, abs(total_edf), float(np.sum(np.abs(diagonal), dtype=np.float64)))
    reconciliation_tolerance = (
        max(
            result.config.residual_tolerance,
            128.0 * np.finfo(float).eps,
        )
        * scale
    )
    slice_total = float(sum(intercept_edf.values()) + sum(term_edf.values()))
    predictor_total = float(sum(predictor_edf.values()))
    source = result.terminal_curvature.actual_source
    rank = (
        result.terminal_rank.rank
        if result.terminal_reduced_rank is None
        else result.terminal_reduced_rank.rank
    )
    return JointInference(
        covariance=covariance,
        influence=influence,
        coefficient_edf=diagonal,
        coefficient_names=layout.coefficient_names,
        total_edf=total_edf,
        predictor_edf=predictor_edf,
        intercept_edf=intercept_edf,
        term_edf=term_edf,
        covariance_curvature_source=source,
        edf_curvature_source=source,
        curvature_telemetry=result.terminal_curvature,
        rank=rank,
        reconciliation_tolerance=reconciliation_tolerance,
        slice_reconciliation_error=abs(slice_total - total_edf),
        predictor_reconciliation_error=abs(predictor_total - total_edf),
    )


__all__ = ["JointInference", "compute_joint_inference"]
