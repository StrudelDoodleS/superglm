"""Serializable curvature-source and fallback telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from superglm.distributional.family import CurvatureKind


@dataclass(frozen=True)
class CurvatureTelemetry:
    """Required audit record for one accepted or retried curvature request."""

    requested_source: CurvatureKind
    actual_source: CurvatureKind
    reason: str | None
    minimum_eigenvalue: float
    rank: int
    condition_estimate: float | None
    fallback_count: int
    # Absent in schema 8, whose scope depended on the family's Fisher support.
    matrix_kind: Literal["data", "penalized"] | None = None

    def __post_init__(self) -> None:
        valid_sources = ("observed", "fisher", "hybrid")
        if self.requested_source not in valid_sources:
            raise ValueError(f"invalid requested curvature source: {self.requested_source!r}")
        if self.actual_source not in valid_sources:
            raise ValueError(f"invalid actual curvature source: {self.actual_source!r}")
        if self.matrix_kind not in (None, "data", "penalized"):
            raise ValueError("matrix_kind must be 'data', 'penalized', or None for legacy state")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("reason must be None or a non-empty string")
        if isinstance(self.minimum_eigenvalue, bool) or not np.isfinite(self.minimum_eigenvalue):
            raise ValueError("minimum_eigenvalue must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if self.condition_estimate is not None:
            if isinstance(self.condition_estimate, bool) or not np.isfinite(
                self.condition_estimate
            ):
                raise ValueError("condition_estimate must be finite when present")
            if self.condition_estimate < 0.0:
                raise ValueError("condition_estimate must be non-negative")
        if (
            isinstance(self.fallback_count, bool)
            or not isinstance(self.fallback_count, int)
            or self.fallback_count < 0
        ):
            raise ValueError("fallback_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return exactly the versioned curvature-fallback fields."""
        result = {
            "requested_source": self.requested_source,
            "actual_source": self.actual_source,
            "reason": self.reason,
            "minimum_eigenvalue": float(self.minimum_eigenvalue),
            "rank": self.rank,
            "condition_estimate": (
                None if self.condition_estimate is None else float(self.condition_estimate)
            ),
            "fallback_count": self.fallback_count,
        }
        if self.matrix_kind is not None:
            result["matrix_kind"] = self.matrix_kind
        return result

    def assert_no_fallback(self) -> None:
        """Reject telemetry that cannot certify an algorithm-matched fixture."""
        if self.fallback_count > 0 or self.actual_source != self.requested_source:
            raise RuntimeError(
                "algorithm-matched certification forbids curvature fallback "
                f"({self.requested_source} -> {self.actual_source}, "
                f"fallback_count={self.fallback_count})"
            )
