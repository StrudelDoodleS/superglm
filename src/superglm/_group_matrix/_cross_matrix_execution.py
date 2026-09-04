"""Rectangular weighted moments between two grouped matrix plans."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_algebra import (
    _BlockWeightCache,
    _cross_gram,
    _profile_add,
    _profile_count,
    _profile_elapsed,
)
from ._group_matrix_execution import GroupSpan, MatrixExecutionPlan


@dataclass(frozen=True)
class CrossGroupPair:
    """One immutable left/right block placement in rectangular coordinates."""

    left: GroupSpan
    right: GroupSpan


class CrossMatrixExecutionPlan:
    """Immutable rectangular execution layout for ``X_left.T W X_right``."""

    _IMMUTABLE_LAYOUT_FIELDS = frozenset(
        {
            "left",
            "right",
            "n",
            "shape",
            "group_pairs",
            "_pair_entries",
        }
    )

    def __setattr__(self, name, value):
        if getattr(self, "_layout_frozen", False) and name in self._IMMUTABLE_LAYOUT_FIELDS:
            raise AttributeError(f"{name} is immutable after plan construction")
        super().__setattr__(name, value)

    def __init__(self, left: MatrixExecutionPlan, right: MatrixExecutionPlan) -> None:
        if not isinstance(left, MatrixExecutionPlan) or not isinstance(right, MatrixExecutionPlan):
            raise TypeError("left and right must be MatrixExecutionPlan instances")
        if left.n != right.n:
            raise ValueError("left and right plans must have the same row count")
        self.left = left
        self.right = right
        self.n = left.n
        self.shape = (left.p, right.p)
        self.group_pairs = tuple(
            CrossGroupPair(left_span, right_span)
            for left_span in left.group_spans
            for right_span in right.group_spans
        )
        self._pair_entries = tuple(
            (
                pair,
                left.group_matrices[pair.left.index],
                right.group_matrices[pair.right.index],
            )
            for pair in self.group_pairs
        )
        self._layout_frozen = True

    def cross_moment(
        self,
        weights: NDArray,
        *,
        signed: bool = True,
        profile: dict | None = None,
    ) -> NDArray[np.float64]:
        """Return ``X_left.T @ diag(weights) @ X_right`` in global coordinates."""
        if not isinstance(signed, bool):
            raise TypeError("signed must be bool")
        try:
            W = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weights must be a finite vector matching the plan row count") from exc
        if W.shape != (self.n,) or not np.all(np.isfinite(W)):
            raise ValueError("weights must be a finite vector matching the plan row count")
        if not signed and np.any(W < 0.0):
            raise ValueError("negative weights require signed=True")

        _profile_count(profile, "cross_calls")
        for key in (
            "cross_route_specialized_calls",
            "cross_route_fallback_calls",
            "block_hist2d_builds",
            "block_hist2d_reuses",
        ):
            _profile_count(profile, key, 0)
        started = perf_counter() if profile is not None else 0.0
        result = np.zeros(self.shape, dtype=np.float64)
        cache = _BlockWeightCache(profile)
        for pair, left_group, right_group in self._pair_entries:
            category = f"{type(left_group).__name__}__{type(right_group).__name__}"
            _profile_count(profile, f"cross_pair_{category}_calls")
            pair_started = perf_counter() if profile is not None else 0.0
            pair_profile: dict | None = {} if profile is not None else None
            cross = _cross_gram(
                left_group,
                right_group,
                W,
                cache,
                pair_profile,
            )
            if profile is not None:
                assert pair_profile is not None
                for key, value in pair_profile.items():
                    _profile_add(profile, key, float(value))
                route = "fallback" if "block_cross_fallback_s" in pair_profile else "specialized"
                _profile_count(profile, f"cross_route_{route}_calls")
            result[pair.left.columns, pair.right.columns] = cross
            _profile_elapsed(profile, f"cross_pair_{category}_s", pair_started)
        _profile_elapsed(profile, "cross_total_s", started)
        return result


__all__ = ["CrossGroupPair", "CrossMatrixExecutionPlan"]
