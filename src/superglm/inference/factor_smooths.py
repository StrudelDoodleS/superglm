"""Compact inference and actuarial reporting for fitted factor smooths."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from superglm._reporting_state import ReportingSupportState
from superglm.features.factor_smooth import FactorSmooth
from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.inference.covariance import (
    covariance_factor_smooth_raw_level_block,
    covariance_selected_block,
)
from superglm.model.state_ops import _solver_space_working_weights
from superglm.solvers.structured import (
    FactorSmoothLevelSupport,
    StructuredLinearSystemState,
)
from superglm.types import GroupSlice, PenaltyComponent

_LAMBDA_LOWER_BOUND = 1.0e-6
_LAMBDA_UPPER_BOUND = 1.0e10


@dataclass(frozen=True)
class FactorSmoothResult:
    """Shared smoothing parameters and fitted level curves.

    Fully penalized ``fs`` reports include local credibility diagnostics.
    Sum-to-zero ``sz`` reports instead describe centered deviation curves.
    """

    name: str
    variable: str
    grouping_variable: str
    basis: Literal["fs", "sz"]
    lambdas: dict[str, float]
    phi: float
    variance_components: dict[str, float]
    effective_df: float
    collapsed: bool | None
    at_lower_boundary: dict[str, bool]
    at_upper_boundary: dict[str, bool]
    table: pd.DataFrame
    curves: pd.DataFrame
    diagnostics: dict[str, Any]

    @property
    def smoothing_lambdas(self) -> dict[str, float]:
        """Named alias emphasizing that lambdas are shared across all levels."""
        return dict(self.lambdas)


def _resolve_factor_smooth(
    model,
    name: str,
) -> tuple[GroupSlice, FactorSmooth]:
    matches: list[tuple[GroupSlice, FactorSmooth]] = []
    for interaction_name in model._interaction_order:
        spec = model._interaction_specs[interaction_name]
        if not isinstance(spec, FactorSmooth):
            continue
        if name not in (interaction_name, spec.name):
            continue
        group = next(
            (candidate for candidate in model._groups if candidate.name == interaction_name),
            None,
        )
        if group is not None:
            matches.append((group, spec))
    if not matches:
        raise KeyError(f"No fitted FactorSmooth term named {name!r}.")
    if len(matches) > 1:
        raise ValueError(f"FactorSmooth name {name!r} is ambiguous.")
    return matches[0]


def _factor_penalties(
    model,
    group: GroupSlice,
    spec: FactorSmooth,
) -> list[PenaltyComponent]:
    lambdas = getattr(model, "_reml_lambdas", None)
    penalties = getattr(model, "_reml_penalties", None)
    if lambdas is None or penalties is None:
        raise RuntimeError("factor_smooth() requires a fit_reml() result")
    group_components = [component for component in penalties if component.group_name == group.name]
    expected_kind = "repeated" if spec.basis == "fs" else "sum_to_zero"
    matching = [
        component for component in group_components if component.penalty_kind == expected_kind
    ]
    if not matching or len(matching) != len(group_components):
        raise RuntimeError(
            f"FactorSmooth term {group.name!r} with basis={spec.basis!r} "
            f"requires only {expected_kind!r} REML components."
        )
    suffixes = {_component_suffix(group.name, component.name) for component in matching}
    if spec.basis == "sz" and suffixes != {"wiggle"}:
        raise RuntimeError(
            f"FactorSmooth term {group.name!r} with basis='sz' requires exactly "
            "one shared 'wiggle' REML component."
        )
    return matching


def _component_suffix(group_name: str, component_name: str) -> str:
    prefix = f"{group_name}:"
    return component_name[len(prefix) :] if component_name.startswith(prefix) else component_name


def _support_from_retained_design(
    model,
    group: GroupSlice,
) -> FactorSmoothLevelSupport | None:
    if model._dm is None or model._fit_weights is None:
        return None
    group_index = model._groups.index(group)
    group_matrix = model._dm.group_matrices[group_index]
    if not isinstance(group_matrix, FactorSmoothGroupMatrix):
        return None
    working_weights = _solver_space_working_weights(model)
    information, _xtw, _xtwz = group_matrix.factor_smooth_sufficient_stats(
        working_weights,
        np.zeros_like(working_weights),
    )
    return FactorSmoothLevelSupport(
        count=np.bincount(
            group_matrix.codes,
            minlength=group_matrix.n_levels,
        ),
        fit_weight=np.bincount(
            group_matrix.codes,
            weights=model._fit_weights,
            minlength=group_matrix.n_levels,
        ),
        information=information,
    )


def _stored_support(
    model,
    group: GroupSlice,
) -> FactorSmoothLevelSupport | None:
    reporting = getattr(model, "_reporting_support_state", None)
    if isinstance(reporting, ReportingSupportState):
        support = reporting.support_totals.get(group.name)
        if isinstance(support, FactorSmoothLevelSupport):
            return support
    state = getattr(model, "_linear_system_state", None)
    if isinstance(state, StructuredLinearSystemState):
        support = state.support_totals.get(group.name)
        if isinstance(support, FactorSmoothLevelSupport):
            return support
    return _support_from_retained_design(model, group)


def _grid_values(spec: FactorSmooth, grid: int | NDArray | None) -> NDArray:
    if spec._spline is None:
        raise RuntimeError("FactorSmooth marginal state is unavailable.")
    if grid is None:
        grid = 100
    if isinstance(grid, bool):
        raise TypeError("grid must be an integer or a one-dimensional numeric array")
    if isinstance(grid, int):
        if grid < 2:
            raise ValueError("integer grid must contain at least two points")
        return np.linspace(float(spec._spline._lo), float(spec._spline._hi), grid)
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("grid must be a finite, non-empty one-dimensional array")
    return values


def factor_smooth_result(
    model,
    name: str,
    *,
    grid: int | NDArray | None = 100,
    levels: list[Any] | tuple[Any, ...] | None = None,
    confidence_level: float = 0.95,
) -> FactorSmoothResult:
    """Build compact per-level inference for one fitted factor smooth."""
    if model._result is None:
        raise RuntimeError("factor_smooth() requires a fitted model")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    group, spec = _resolve_factor_smooth(model, name)
    penalties = _factor_penalties(model, group, spec)
    fitted_lambdas = model._reml_lambdas
    lambdas = {
        _component_suffix(group.name, component.name): float(fitted_lambdas[component.name])
        for component in penalties
    }
    phi = float(model.result.phi)
    variance_components = {
        component: float(np.inf if value == 0.0 else phi / value)
        for component, value in lambdas.items()
    }
    at_lower_boundary = {
        component: value <= _LAMBDA_LOWER_BOUND * (1.0 + 1.0e-8)
        for component, value in lambdas.items()
    }
    at_upper_boundary = {
        component: value >= _LAMBDA_UPPER_BOUND * (1.0 - 1.0e-8)
        for component, value in lambdas.items()
    }
    collapsed: bool | None = (
        bool(at_upper_boundary and all(at_upper_boundary.values())) if spec.basis == "fs" else None
    )
    if collapsed is True:
        warnings.warn(
            f"FactorSmooth term {group.name!r} is collapsed at the upper lambda boundary.",
            UserWarning,
            stacklevel=3,
        )

    support = _stored_support(model, group)
    if support is None:
        raise RuntimeError(
            f"FactorSmooth support for {group.name!r} is unavailable; refit with "
            "retain_fit_state=True or direct_solve='structured'."
        )
    n_levels = len(spec._levels)
    if support.information.shape != (n_levels, spec.k, spec.k):
        raise RuntimeError("Retained FactorSmooth support does not match its fitted basis.")

    inference = model._fit_inference_info
    active_group = next(
        (active for active in inference["active_groups"] if active.name == group.name),
        None,
    )
    if active_group is None:
        raise RuntimeError(f"FactorSmooth term {group.name!r} is absent from fitted inference.")
    coefficient_levels = n_levels if spec.basis == "fs" else n_levels - 1
    if active_group.size != coefficient_levels * spec.k:
        raise RuntimeError("FactorSmooth inference block has inconsistent dimensions.")

    component_omegas: list[tuple[PenaltyComponent, NDArray]] = []
    for component in penalties:
        if component.omega_ssp is None:
            raise RuntimeError(f"FactorSmooth penalty {component.name!r} has no marginal geometry.")
        omega = np.asarray(component.omega_ssp, dtype=np.float64)
        component_omegas.append((component, omega))

    information_rank = np.asarray(
        [
            np.linalg.matrix_rank(support.information[level_index])
            for level_index in range(n_levels)
        ],
        dtype=np.int64,
    )
    information_trace = np.trace(
        support.information,
        axis1=1,
        axis2=2,
    )
    has_information = (
        (np.asarray(support.fit_weight) > 0.0)
        & np.isfinite(information_trace)
        & (information_trace > 0.0)
    )
    sufficient_support = information_rank == spec.k

    if levels is None:
        selected_level_indices = np.arange(n_levels, dtype=np.intp)
    else:
        requested = list(levels)
        selected_level_indices = (
            pd.Index(spec._levels)
            .get_indexer(requested)
            .astype(
                np.intp,
                copy=False,
            )
        )
        if np.any(selected_level_indices < 0):
            missing = [
                level
                for level, index in zip(requested, selected_level_indices, strict=True)
                if index < 0
            ]
            raise KeyError(f"Unknown FactorSmooth levels: {missing!r}.")

    grid_values = _grid_values(spec, grid)
    basis = spec.marginal_basis(grid_values)
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    common_diagnostics = {
        "n_levels_without_information": int(np.count_nonzero(~has_information)),
        "n_levels_with_insufficient_support": int(np.count_nonzero(~sufficient_support)),
        "block_size": spec.k,
        "n_levels": n_levels,
        "lambda_names": {
            _component_suffix(group.name, component.name): component.name
            for component, _omega in component_omegas
        },
        "lambda_lower_bound": _LAMBDA_LOWER_BOUND,
        "lambda_upper_bound": _LAMBDA_UPPER_BOUND,
        "backend": getattr(model.result, "direct_backend", None),
        "fallback_reason": getattr(model.result, "direct_fallback_reason", None),
    }

    if spec.basis == "fs":
        local_penalty = np.zeros((spec.k, spec.k), dtype=np.float64)
        for component, omega in component_omegas:
            local_penalty += float(fitted_lambdas[component.name]) * omega
        local_penalized = support.information + local_penalty[None, :, :]
        local_credibility = np.empty(n_levels, dtype=np.float64)
        for level_index in range(n_levels):
            try:
                shrinkage = np.linalg.solve(
                    local_penalized[level_index],
                    local_penalty,
                )
            except np.linalg.LinAlgError:
                shrinkage = np.linalg.pinv(local_penalized[level_index]) @ local_penalty
            local_credibility[level_index] = float(np.trace(np.eye(spec.k) - shrinkage) / spec.k)

        coefficient_edf = np.asarray(inference["edf"], dtype=np.float64)[active_group.sl]
        level_edf = coefficient_edf.reshape(n_levels, spec.k).sum(axis=1)
        coefficients = np.asarray(model.result.beta[group.sl], dtype=np.float64).reshape(
            n_levels,
            spec.k,
        )
        table = pd.DataFrame(
            {
                "level": spec._levels,
                "count": np.asarray(support.count, dtype=np.int64),
                "fit_weight": np.asarray(support.fit_weight, dtype=np.float64),
                "information_trace": information_trace,
                "information_rank": information_rank,
                "effective_df": level_edf,
                "credibility": local_credibility,
                "shrinkage": 1.0 - local_credibility,
                "coefficient_norm": np.linalg.norm(coefficients, axis=1),
                "has_information": has_information,
                "sufficient_support": sufficient_support,
                "collapsed": np.full(n_levels, collapsed, dtype=bool),
            }
        )

        curve_frames: list[pd.DataFrame] = []
        for level_index in selected_level_indices:
            local_start = active_group.start + int(level_index) * spec.k
            augmented_indices = np.arange(
                1 + local_start,
                1 + local_start + spec.k,
                dtype=np.intp,
            )
            covariance = phi * covariance_selected_block(
                inference["XtWX_inv_aug"],
                augmented_indices,
            )
            effect = basis @ coefficients[level_index]
            variance = np.einsum(
                "ij,jk,ik->i",
                basis,
                covariance,
                basis,
                optimize=True,
            )
            posterior_se = np.sqrt(np.maximum(variance, 0.0))
            curve_frames.append(
                pd.DataFrame(
                    {
                        "level": np.full(
                            len(grid_values),
                            spec._levels[int(level_index)],
                            dtype=object,
                        ),
                        spec.variable: grid_values,
                        "effect": effect,
                        "posterior_se": posterior_se,
                        "lower": effect - critical_value * posterior_se,
                        "upper": effect + critical_value * posterior_se,
                    }
                )
            )
        curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        credibility_violation = bool(
            np.any(local_credibility < -1.0e-10) or np.any(local_credibility > 1.0 + 1.0e-10)
        )
        diagnostics = {
            "credibility_definition": "trace(I - D_level^-1 P_level) / block_size",
            "credibility_out_of_bounds": credibility_violation,
            **common_diagnostics,
        }
    else:
        coefficients = spec._level_blocks(np.asarray(model.result.beta[group.sl], dtype=np.float64))
        public_group_indices = np.arange(
            1 + active_group.start,
            1 + active_group.end,
            dtype=np.intp,
        )
        raw_inverse_blocks = np.stack(
            [
                covariance_factor_smooth_raw_level_block(
                    inference["XtWX_inv_aug"],
                    public_group_indices,
                    level=level_index,
                    n_levels=n_levels,
                    block_size=spec.k,
                    term_name=group.name,
                )
                for level_index in range(n_levels)
            ]
        )
        base_level_df = active_group.size / n_levels
        level_edf = np.asarray(
            [
                base_level_df
                - sum(
                    float(fitted_lambdas[component.name])
                    * float(np.trace(raw_inverse_blocks[level_index] @ omega))
                    for component, omega in component_omegas
                )
                for level_index in range(n_levels)
            ],
            dtype=np.float64,
        )
        coefficient_group_edf = float(
            np.sum(np.asarray(inference["edf"], dtype=np.float64)[active_group.sl])
        )
        edf_reconciliation = (coefficient_group_edf - float(np.sum(level_edf))) / n_levels
        # The two algebraically identical EDF paths can differ by a few ulps
        # amplified by lambda near the 1e10 boundary.  Share that numerical
        # residual across raw levels so the attribution remains symmetric and
        # sums exactly to the fitted term EDF.
        level_edf += edf_reconciliation
        table = pd.DataFrame(
            {
                "level": spec._levels,
                "count": np.asarray(support.count, dtype=np.int64),
                "fit_weight": np.asarray(support.fit_weight, dtype=np.float64),
                "information_trace": information_trace,
                "information_rank": information_rank,
                "effective_df": level_edf,
                "coefficient_norm": np.linalg.norm(coefficients, axis=1),
                "has_information": has_information,
                "sufficient_support": sufficient_support,
            }
        )

        curve_frames = []
        for level_index in selected_level_indices:
            covariance = phi * raw_inverse_blocks[int(level_index)]
            effect = basis @ coefficients[int(level_index)]
            variance = np.einsum(
                "ij,jk,ik->i",
                basis,
                covariance,
                basis,
                optimize=True,
            )
            posterior_se = np.sqrt(np.maximum(variance, 0.0))
            curve_frames.append(
                pd.DataFrame(
                    {
                        "level": np.full(
                            len(grid_values),
                            spec._levels[int(level_index)],
                            dtype=object,
                        ),
                        spec.variable: grid_values,
                        "effect": effect,
                        "posterior_se": posterior_se,
                        "lower": effect - critical_value * posterior_se,
                        "upper": effect + critical_value * posterior_se,
                    }
                )
            )
        curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        all_effects = basis @ coefficients.T
        diagnostics = {
            "max_abs_level_effect_sum": float(
                np.max(np.abs(np.sum(all_effects, axis=1)), initial=0.0)
            ),
            "level_edf_numerical_reconciliation": float(edf_reconciliation),
            **common_diagnostics,
        }

    return FactorSmoothResult(
        name=group.name,
        variable=spec.variable,
        grouping_variable=spec.group,
        basis=spec.basis,
        lambdas=lambdas,
        phi=phi,
        variance_components=variance_components,
        effective_df=float(np.sum(level_edf)),
        collapsed=collapsed,
        at_lower_boundary=at_lower_boundary,
        at_upper_boundary=at_upper_boundary,
        table=table,
        curves=curves,
        diagnostics=diagnostics,
    )


__all__ = ["FactorSmoothResult", "factor_smooth_result"]
