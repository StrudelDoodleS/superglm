"""Reference-level adjustments for ordered spline reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.types import GroupSlice


def _ordered_spline_specs(
    feature_order: Sequence[str],
    specs: Mapping[str, Any],
):
    from superglm.features.ordered_categorical import OrderedCategorical

    for name in feature_order:
        spec = specs.get(name)
        if isinstance(spec, OrderedCategorical) and spec.basis == "spline":
            yield name, spec


def ordered_reference_intercept(
    intercept: float,
    beta: NDArray,
    feature_order: Sequence[str],
    specs: Mapping[str, Any],
    groups: Sequence[GroupSlice],
) -> float:
    """Return the intercept corresponding to ordered-spline reference levels.

    Ordered spline relativities are reported after subtracting each feature's
    fitted effect at its reference level. The same effects therefore have to
    be added to the fitted intercept for the reported factors to reconstruct
    the model prediction. This helper is pure and never changes ``beta`` or a
    fitted result object.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    adjustment = 0.0
    for name, spec in _ordered_spline_specs(feature_order, specs):
        feature_groups = [group for group in groups if group.feature_name == name]
        if not feature_groups:
            continue
        feature_beta = np.concatenate([beta_arr[group.sl] for group in feature_groups])
        adjustment += spec._base_log_effect(feature_beta)
    return float(intercept + adjustment)


def ordered_reference_beta_contrast(
    n_coefficients: int,
    feature_order: Sequence[str],
    specs: Mapping[str, Any],
    groups: Sequence[GroupSlice],
) -> NDArray[np.float64]:
    """Return ``c`` where ``c @ beta`` is the ordered reference adjustment.

    This companion to :func:`ordered_reference_intercept` is useful for
    propagating the full intercept/feature covariance into reporting standard
    errors without duplicating ordered-spline basis logic.
    """
    contrast: NDArray[np.float64] = np.zeros(n_coefficients, dtype=np.float64)
    for name, spec in _ordered_spline_specs(feature_order, specs):
        feature_groups = [group for group in groups if group.feature_name == name]
        if not feature_groups:
            continue
        base_row = np.asarray(spec.transform([spec._base_level]), dtype=np.float64).ravel()
        expected_size = sum(group.size for group in feature_groups)
        if base_row.size != expected_size:
            raise RuntimeError(
                f"Ordered spline reference row for {name!r} has {base_row.size} columns; "
                f"expected {expected_size}."
            )
        start = 0
        for group in feature_groups:
            stop = start + group.size
            contrast[group.sl] = base_row[start:stop]
            start = stop
    return contrast


__all__ = ["ordered_reference_beta_contrast", "ordered_reference_intercept"]
