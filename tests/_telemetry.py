"""Assertion helper for curvature telemetry in algorithm-matched fixtures."""

from __future__ import annotations

from superglm.distributional.telemetry import CurvatureTelemetry


def assert_no_fallback(telemetry: CurvatureTelemetry) -> None:
    """Reject telemetry that cannot certify an algorithm-matched fixture."""
    if telemetry.fallback_count > 0 or telemetry.actual_source != telemetry.requested_source:
        raise RuntimeError(
            "algorithm-matched certification forbids curvature fallback "
            f"({telemetry.requested_source} -> {telemetry.actual_source}, "
            f"fallback_count={telemetry.fallback_count})"
        )
