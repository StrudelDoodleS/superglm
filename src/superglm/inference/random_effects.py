"""Actuarial reporting for fitted scalar random effects."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributions import _VARIANCE_FLOOR, Poisson, clip_mu
from superglm.features.random_effect import RandomEffect
from superglm.group_matrix import RandomEffectGroupMatrix
from superglm.inference.covariance import covariance_selected_diagonal
from superglm.links import LogLink, stabilize_eta
from superglm.model.fit_data_guard import require_unchanged_fit_data
from superglm.model.state_ops import _solver_space_working_weights
from superglm.solvers.structured import (
    StructuredLevelSupport,
    StructuredLinearSystemState,
)
from superglm.types import GroupSlice

_LAMBDA_LOWER_BOUND = 1.0e-6
_LAMBDA_UPPER_BOUND = 1.0e10


@dataclass(frozen=True)
class RandomEffectResult:
    """Variance-component summary and one row per fitted factor level."""

    name: str
    lambda_value: float
    phi: float
    tau_squared: float
    standard_deviation: float
    effective_df: float
    collapsed: bool
    at_lower_boundary: bool
    at_upper_boundary: bool
    table: pd.DataFrame
    diagnostics: dict[str, Any]

    @property
    def variance_component(self) -> float:
        """Alias for the random-effect variance ``phi / lambda``."""
        return self.tau_squared

    @property
    def smoothing_lambda(self) -> float:
        """Named alias for the fitted penalty multiplier."""
        return self.lambda_value


def vectorized_conditional_unpooled_effect(
    *,
    codes: NDArray,
    n_levels: int,
    y: NDArray,
    sample_weight: NDArray,
    base_eta: NDArray,
    distribution,
    link,
    initial: NDArray | None = None,
    max_iter: int = 100,
    tolerance: float = 1.0e-11,
) -> NDArray:
    """Solve every unpenalized conditional level update simultaneously."""
    codes = np.asarray(codes, dtype=np.intp)
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    base_eta = np.asarray(base_eta, dtype=np.float64)
    if (
        codes.ndim != 1
        or y.shape != codes.shape
        or sample_weight.shape != codes.shape
        or base_eta.shape != codes.shape
    ):
        raise ValueError("unpooled-effect row arrays must be one-dimensional and aligned")
    if np.any((codes < 0) | (codes >= n_levels)):
        raise ValueError("unpooled-effect codes must refer to fitted levels")
    if np.any(sample_weight < 0.0) or not np.all(np.isfinite(sample_weight)):
        raise ValueError("sample_weight must be finite and non-negative")

    # The canonical Poisson/log case has an exact aggregate update. Keeping
    # this branch explicit makes the actuarial actual/expected identity exact,
    # including the meaningful -inf result for zero-actual levels.
    if isinstance(distribution, Poisson) and isinstance(link, LogLink):
        stable_base_eta = stabilize_eta(base_eta, link)
        expected = np.bincount(
            codes,
            weights=sample_weight * np.exp(stable_base_eta),
            minlength=n_levels,
        )
        actual = np.bincount(
            codes,
            weights=sample_weight * y,
            minlength=n_levels,
        )
        result = np.full(n_levels, np.nan, dtype=np.float64)
        positive_expected = expected > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            result[positive_expected] = np.log(
                actual[positive_expected] / expected[positive_expected]
            )
        return result

    effects = (
        np.zeros(n_levels, dtype=np.float64)
        if initial is None
        else np.array(initial, dtype=np.float64, copy=True)
    )
    if effects.shape != (n_levels,):
        raise ValueError(f"initial must have shape ({n_levels},)")
    informed = (
        np.bincount(
            codes,
            weights=sample_weight,
            minlength=n_levels,
        )
        > 0.0
    )

    for _ in range(max_iter):
        eta = stabilize_eta(base_eta + effects[codes], link)
        mu = clip_mu(link.inverse(eta), distribution)
        variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
        derivative = link.deriv_inverse(eta)
        score = np.bincount(
            codes,
            weights=sample_weight * (y - mu) * derivative / variance,
            minlength=n_levels,
        )
        information = np.bincount(
            codes,
            weights=sample_weight * derivative**2 / variance,
            minlength=n_levels,
        )
        valid = informed & np.isfinite(score) & np.isfinite(information) & (information > 0.0)
        update = np.zeros(n_levels, dtype=np.float64)
        update[valid] = score[valid] / information[valid]
        effects[valid] += update[valid]
        if np.max(np.abs(update[valid]), initial=0.0) <= tolerance:
            break

    effects[~informed] = np.nan
    return effects


def _resolve_random_effect(model, name: str) -> tuple[GroupSlice, RandomEffect]:
    matches: list[tuple[GroupSlice, RandomEffect]] = []
    for group in model._groups:
        if group.name != name and group.feature_name != name:
            continue
        spec = model._specs.get(group.feature_name or group.name)
        if isinstance(spec, RandomEffect):
            matches.append((group, spec))
    if not matches:
        raise KeyError(f"No fitted RandomEffect term named {name!r}.")
    if len(matches) > 1:
        raise ValueError(f"RandomEffect name {name!r} is ambiguous.")
    return matches[0]


def _lambda_for_group(model, group: GroupSlice) -> float:
    lambdas = getattr(model, "_reml_lambdas", None)
    penalties = getattr(model, "_reml_penalties", None)
    if lambdas is None or penalties is None:
        raise RuntimeError("random_effects() requires a fit_reml() result")
    matching = [
        component
        for component in penalties
        if component.group_name == group.name and component.penalty_kind == "identity"
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"RandomEffect term {group.name!r} does not have one identity REML component."
        )
    return float(lambdas[matching[0].name])


def _support_from_retained_design(
    model,
    group: GroupSlice,
) -> StructuredLevelSupport | None:
    if model._dm is None or model._fit_weights is None:
        return None
    group_index = model._groups.index(group)
    group_matrix = model._dm.group_matrices[group_index]
    if not isinstance(group_matrix, RandomEffectGroupMatrix):
        return None
    working_weights = _solver_space_working_weights(model)
    return StructuredLevelSupport(
        count=np.bincount(
            group_matrix.codes,
            minlength=group_matrix.n_levels,
        ),
        fit_weight=np.bincount(
            group_matrix.codes,
            weights=model._fit_weights,
            minlength=group_matrix.n_levels,
        ),
        information=group_matrix.rmatvec(working_weights),
    )


def _stored_support(model, group: GroupSlice) -> StructuredLevelSupport | None:
    state = getattr(model, "_linear_system_state", None)
    if isinstance(state, StructuredLinearSystemState):
        support = state.support_totals.get(group.name)
        if support is not None:
            return support
    return _support_from_retained_design(model, group)


def _reporting_rows(
    model,
    group: GroupSlice,
    spec: RandomEffect,
    *,
    X,
    y,
    sample_weight,
    offset,
) -> tuple[NDArray, NDArray, NDArray, NDArray] | None:
    explicit = any(value is not None for value in (X, y, sample_weight, offset))
    if explicit and (X is None or y is None):
        raise ValueError("X and y must be supplied together for random-effect row diagnostics.")

    if X is None:
        if model._fit_X_ref is None or model._fit_y_ref is None or model._fit_weights is None:
            return None
        require_unchanged_fit_data(model, model._fit_X_ref, model._fit_y_ref)
        X = model._fit_X_ref
        y = model._fit_y_ref
        sample_weight = model._fit_weights
        offset = model._fit_offset

    frame = as_eager_frame(X)
    feature_name = group.feature_name or group.name
    codes = spec._prediction_codes(frame.column_array(feature_name))
    if np.any(codes < 0):
        raise ValueError("random-effect reporting rows contain unseen fitted levels")
    y_values = np.asarray(y, dtype=np.float64)
    n = len(codes)
    if y_values.shape != (n,):
        raise ValueError("y must be one-dimensional and match X")
    weights = (
        np.ones(n, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != (n,):
        raise ValueError("sample_weight must be one-dimensional and match X")
    offset_values = (
        np.zeros(n, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    )
    if offset_values.shape != (n,):
        raise ValueError("offset must be one-dimensional and match X")
    eta = model._predict_eta_exact(
        frame,
        offset=offset_values,
        random_effects="conditional",
    )
    base_eta = eta - model.result.beta[group.sl][codes]
    return codes, y_values, weights, base_eta


def random_effect_result(
    model,
    name: str,
    *,
    exposure: NDArray | None = None,
    X=None,
    y: NDArray | None = None,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
) -> RandomEffectResult:
    """Build a compact actuarial report for one fitted random effect."""
    if model._result is None:
        raise RuntimeError("random_effects() requires a fitted model")
    group, spec = _resolve_random_effect(model, name)
    lambda_value = _lambda_for_group(model, group)
    phi = float(model.result.phi)
    tau_squared = float(np.inf if lambda_value == 0.0 else phi / lambda_value)
    at_lower_boundary = lambda_value <= _LAMBDA_LOWER_BOUND * (1.0 + 1.0e-8)
    at_upper_boundary = lambda_value >= _LAMBDA_UPPER_BOUND * (1.0 - 1.0e-8)
    collapsed = at_upper_boundary
    if collapsed:
        warnings.warn(
            f"RandomEffect term {group.name!r} is collapsed at the upper lambda boundary.",
            UserWarning,
            stacklevel=3,
        )

    support = _stored_support(model, group)
    if support is None:
        raise RuntimeError(
            f"RandomEffect support for {group.name!r} is unavailable; refit with "
            "retain_fit_state=True or direct_solve='structured'."
        )
    rows = _reporting_rows(
        model,
        group,
        spec,
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    if rows is None:
        if support.unpooled_effect is None:
            raise RuntimeError(
                "Training rows were released and unpooled effects were not precomputed; "
                "supply X and y (plus the original sample_weight and offset when used)."
            )
        codes = None
        unpooled = np.asarray(support.unpooled_effect, dtype=np.float64)
        count = np.asarray(support.count)
        fit_weight = np.asarray(support.fit_weight)
    else:
        codes, y_values, weights, base_eta = rows
        count = np.bincount(codes, minlength=len(spec._levels))
        fit_weight = np.bincount(
            codes,
            weights=weights,
            minlength=len(spec._levels),
        )
        unpooled = vectorized_conditional_unpooled_effect(
            codes=codes,
            n_levels=len(spec._levels),
            y=y_values,
            sample_weight=weights,
            base_eta=base_eta,
            distribution=model._distribution,
            link=model._link,
            initial=model.result.beta[group.sl],
        )

    if exposure is None:
        exposure_total = np.full(len(spec._levels), np.nan, dtype=np.float64)
    else:
        exposure_values = np.asarray(exposure, dtype=np.float64)
        if exposure_values.ndim != 1:
            raise ValueError("exposure must be one-dimensional")
        if codes is None:
            raise ValueError(
                "exposure aggregation requires X after row-scale fit state has been released"
            )
        if exposure_values.shape != codes.shape:
            raise ValueError("exposure must have the same length as X")
        exposure_total = np.bincount(
            codes,
            weights=exposure_values,
            minlength=len(spec._levels),
        )

    inference = model._fit_inference_info
    active_group = next(
        (active for active in inference["active_groups"] if active.name == group.name),
        None,
    )
    if active_group is None:
        raise RuntimeError(f"RandomEffect term {group.name!r} is absent from fitted inference.")
    augmented_indices = np.arange(
        1 + active_group.start,
        1 + active_group.end,
        dtype=np.intp,
    )
    posterior_variance = phi * covariance_selected_diagonal(
        inference["XtWX_inv_aug"],
        augmented_indices,
    )
    posterior_se = np.sqrt(np.maximum(posterior_variance, 0.0))

    information = np.asarray(support.information, dtype=np.float64)
    credibility = information / (information + lambda_value)
    credibility_violation = bool(
        np.any(credibility < -1.0e-10) or np.any(credibility > 1.0 + 1.0e-10)
    )
    effect = np.asarray(model.result.beta[group.sl], dtype=np.float64)
    relativity = (
        np.exp(effect)
        if isinstance(model._link, LogLink)
        else np.full(effect.shape, np.nan, dtype=np.float64)
    )
    has_information = np.isfinite(information) & (information > 0.0)
    finite = (
        np.isfinite(unpooled)
        & np.isfinite(effect)
        & np.isfinite(posterior_se)
        & np.isfinite(credibility)
    )

    table = pd.DataFrame(
        {
            "level": spec._levels,
            "count": count.astype(np.int64, copy=False),
            "fit_weight": fit_weight,
            "exposure": exposure_total,
            "unpooled_effect": unpooled,
            "effect": effect,
            "relativity": relativity,
            "posterior_se": posterior_se,
            "credibility": credibility,
            "shrinkage": 1.0 - credibility,
            "finite": finite,
            "has_information": has_information,
            "collapsed": np.full(len(effect), collapsed, dtype=bool),
        }
    )
    group_edf = model._group_edf or {}
    diagnostics = {
        "credibility_out_of_bounds": credibility_violation,
        "unpooled_finite": bool(np.all(np.isfinite(unpooled))),
        "n_levels_without_information": int(np.count_nonzero(~has_information)),
        "lambda_lower_bound": _LAMBDA_LOWER_BOUND,
        "lambda_upper_bound": _LAMBDA_UPPER_BOUND,
        "backend": getattr(model.result, "direct_backend", None),
        "fallback_reason": getattr(model.result, "direct_fallback_reason", None),
    }
    return RandomEffectResult(
        name=group.name,
        lambda_value=lambda_value,
        phi=phi,
        tau_squared=tau_squared,
        standard_deviation=float(np.sqrt(tau_squared)),
        effective_df=float(group_edf.get(group.name, np.nan)),
        collapsed=collapsed,
        at_lower_boundary=at_lower_boundary,
        at_upper_boundary=at_upper_boundary,
        table=table,
        diagnostics=diagnostics,
    )


__all__ = [
    "RandomEffectResult",
    "random_effect_result",
    "vectorized_conditional_unpooled_effect",
]
