"""Auditable terminal-curvature retry and fallback policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import CurvatureKind
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankPolicy,
    decompose_gram,
)


class CurvaturePolicyError(ValueError):
    """Raised when no policy-permitted curvature result can be published."""


class RepeatedCurvatureIndefinitenessError(CurvaturePolicyError):
    """Requested non-Fisher curvature remained materially indefinite.

    Its retry was exhausted and no Fisher fallback was available.
    """


@dataclass(frozen=True)
class CurvaturePolicyState:
    """Per-request retry state plus the model's cumulative fallback count."""

    retry_attempted: bool = False
    fallback_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.retry_attempted, bool):
            raise TypeError("retry_attempted must be bool")
        if (
            isinstance(self.fallback_count, bool)
            or not isinstance(self.fallback_count, int)
            or self.fallback_count < 0
        ):
            raise ValueError("fallback_count must be a non-negative integer")


@dataclass(frozen=True)
class CurvatureDecision:
    """Accepted curvature or an explicit request for one tighter inner solve."""

    matrix: NDArray[np.float64] | None
    decomposition: RankDecomposition | None
    telemetry: CurvatureTelemetry
    retry_required: bool
    state: CurvaturePolicyState

    def __post_init__(self) -> None:
        if not isinstance(self.retry_required, bool):
            raise TypeError("retry_required must be bool")
        if self.retry_required:
            if self.matrix is not None or self.decomposition is not None:
                raise ValueError("retry decisions cannot publish curvature")
        elif self.matrix is None or self.decomposition is None:
            raise ValueError("accepted decisions must publish curvature and rank state")


@dataclass(frozen=True)
class _CurvatureAnalysis:
    matrix: NDArray[np.float64]
    decomposition: RankDecomposition
    materially_indefinite: bool
    minimum_eigenvalue: float
    condition_estimate: float | None


def _analyze_curvature(
    matrix: NDArray,
    *,
    label: str,
    policy: RankPolicy,
) -> _CurvatureAnalysis:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CurvaturePolicyError(f"{label} curvature must be a finite square matrix") from exc
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise CurvaturePolicyError(f"{label} curvature must be a non-empty square matrix")
    if not np.all(np.isfinite(values)):
        raise CurvaturePolicyError(f"{label} curvature contains non-finite values")

    symmetric = np.array(0.5 * (values + values.T), dtype=np.float64, copy=True)
    symmetric.setflags(write=False)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise CurvaturePolicyError(f"{label} curvature eigendecomposition failed") from exc
    minimum_eigenvalue = float(eigenvalues[0])

    materially_indefinite = False
    try:
        decomposition = decompose_gram(symmetric, policy=policy)
    except ValueError as exc:
        if "materially" not in str(exc).lower():
            raise CurvaturePolicyError(f"{label} curvature rank analysis failed: {exc}") from exc
        materially_indefinite = True
        try:
            decomposition = decompose_gram(
                symmetric,
                policy=policy,
                allow_indefinite=True,
            )
        except (ValueError, np.linalg.LinAlgError) as fallback_exc:
            raise CurvaturePolicyError(
                f"{label} indefinite-curvature diagnostics failed: {fallback_exc}"
            ) from fallback_exc

    raw_condition = decomposition.pre_truncation_condition
    condition_estimate = float(raw_condition) if np.isfinite(raw_condition) else None
    return _CurvatureAnalysis(
        matrix=symmetric,
        decomposition=decomposition,
        materially_indefinite=materially_indefinite,
        minimum_eigenvalue=minimum_eigenvalue,
        condition_estimate=condition_estimate,
    )


def _telemetry(
    requested_source: CurvatureKind,
    actual_source: CurvatureKind,
    *,
    reason: str | None,
    minimum_eigenvalue: float,
    analysis: _CurvatureAnalysis,
    fallback_count: int,
) -> CurvatureTelemetry:
    return CurvatureTelemetry(
        requested_source=requested_source,
        actual_source=actual_source,
        reason=reason,
        minimum_eigenvalue=minimum_eigenvalue,
        rank=analysis.decomposition.rank,
        condition_estimate=analysis.condition_estimate,
        fallback_count=fallback_count,
    )


def resolve_curvature(
    requested_source: CurvatureKind,
    requested_matrix: NDArray,
    *,
    fisher_matrix: NDArray | None = None,
    state: CurvaturePolicyState | None = None,
    policy: RankPolicy = SHARED_RANK_POLICY,
) -> CurvatureDecision:
    """Apply the binding one-retry policy to a family-supplied curvature.

    ``hybrid`` is already constructed and documented by the family.  Generic
    policy code diagnoses that matrix as supplied; it does not invent a new
    block combination.
    """
    if requested_source not in ("observed", "fisher", "hybrid"):
        raise ValueError(f"invalid curvature source: {requested_source!r}")
    current_state = CurvaturePolicyState() if state is None else state
    if not isinstance(current_state, CurvaturePolicyState):
        raise TypeError("state must be a CurvaturePolicyState")
    if not isinstance(policy, RankPolicy):
        raise TypeError("policy must be a RankPolicy")

    requested = _analyze_curvature(
        requested_matrix,
        label=requested_source.capitalize(),
        policy=policy,
    )
    if not requested.materially_indefinite:
        reason = "accepted_after_retry" if current_state.retry_attempted else None
        telemetry = _telemetry(
            requested_source,
            requested_source,
            reason=reason,
            minimum_eigenvalue=requested.minimum_eigenvalue,
            analysis=requested,
            fallback_count=current_state.fallback_count,
        )
        return CurvatureDecision(
            matrix=requested.matrix,
            decomposition=requested.decomposition,
            telemetry=telemetry,
            retry_required=False,
            state=CurvaturePolicyState(
                retry_attempted=False,
                fallback_count=current_state.fallback_count,
            ),
        )

    if requested_source == "fisher":
        raise CurvaturePolicyError("Fisher curvature is materially indefinite")

    if not current_state.retry_attempted:
        telemetry = _telemetry(
            requested_source,
            requested_source,
            reason="material_indefiniteness_retry_required",
            minimum_eigenvalue=requested.minimum_eigenvalue,
            analysis=requested,
            fallback_count=current_state.fallback_count,
        )
        return CurvatureDecision(
            matrix=None,
            decomposition=None,
            telemetry=telemetry,
            retry_required=True,
            state=CurvaturePolicyState(
                retry_attempted=True,
                fallback_count=current_state.fallback_count,
            ),
        )

    if fisher_matrix is None:
        raise RepeatedCurvatureIndefinitenessError(
            "Fisher curvature is required after repeated material indefiniteness"
        )
    fisher = _analyze_curvature(fisher_matrix, label="Fisher", policy=policy)
    if fisher.materially_indefinite:
        raise CurvaturePolicyError("Fisher curvature is materially indefinite")

    fallback_count = current_state.fallback_count + 1
    telemetry = _telemetry(
        requested_source,
        "fisher",
        reason="material_indefiniteness_after_retry",
        minimum_eigenvalue=requested.minimum_eigenvalue,
        analysis=fisher,
        fallback_count=fallback_count,
    )
    return CurvatureDecision(
        matrix=fisher.matrix,
        decomposition=fisher.decomposition,
        telemetry=telemetry,
        retry_required=False,
        state=CurvaturePolicyState(retry_attempted=False, fallback_count=fallback_count),
    )
