"""Auditable terminal-curvature retry and fallback policy.

Observed curvature that is materially indefinite at a terminal coefficient
state gets one tighter retry. If it is still indefinite, the policy accepts
the family's expected (Fisher) information when that is supplied and
semidefinite, and otherwise the nearest positive semidefinite matrix to the
observed curvature, as the generalized Fellner-Schall method prescribes for
an indefinite observed Hessian (Wood and Fasiolo, *Biometrics* 73(4), 2017,
section 3; Wood, Pya and Saefken, *JASA* 111, 2016, section 3.3.1). The
solver supplies that matrix; this module only decides which one is accepted
and records the decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import CurvatureKind
from superglm.distributional.telemetry import AcceptedCurvatureKind, CurvatureTelemetry
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankPolicy,
    _symmetric_part,
    decompose_gram,
)


class CurvaturePolicyError(ValueError):
    """Raised when no policy-permitted curvature result can be published."""


class RepeatedCurvatureIndefinitenessError(CurvaturePolicyError):
    """Requested non-Fisher curvature remained materially indefinite.

    Its retry was exhausted and the caller supplied neither a Fisher nor a
    nearest-semidefinite fallback. The coefficient solver always supplies the
    second, so a fit never raises this.
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

    symmetric = np.array(_symmetric_part(values), dtype=np.float64, copy=True)
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
    actual_source: AcceptedCurvatureKind,
    *,
    reason: str | None,
    minimum_eigenvalue: float,
    analysis: _CurvatureAnalysis,
    fallback_count: int,
    matrix_kind: Literal["data", "penalized"] | None,
) -> CurvatureTelemetry:
    return CurvatureTelemetry(
        requested_source=requested_source,
        actual_source=actual_source,
        reason=reason,
        minimum_eigenvalue=minimum_eigenvalue,
        rank=analysis.decomposition.rank,
        condition_estimate=analysis.condition_estimate,
        fallback_count=fallback_count,
        matrix_kind=matrix_kind,
    )


def resolve_curvature(
    requested_source: CurvatureKind,
    requested_matrix: NDArray,
    *,
    fisher_matrix: NDArray | None = None,
    semidefinite_fallback: Callable[[], NDArray] | None = None,
    state: CurvaturePolicyState | None = None,
    policy: RankPolicy = SHARED_RANK_POLICY,
    matrix_kind: Literal["data", "penalized"] | None = None,
) -> CurvatureDecision:
    """Apply the binding one-retry policy to a family-supplied curvature.

    ``hybrid`` is already constructed and documented by the family.  Generic
    policy code diagnoses that matrix as supplied; it does not invent a new
    block combination.

    After the retry, ``fisher_matrix`` is accepted when it is semidefinite.
    Otherwise ``semidefinite_fallback`` is called, once, for the caller's
    nearest positive semidefinite matrix to the requested one, which is then
    accepted with ``actual_source="nearest_semidefinite"``. It is called only
    on that path, so a caller can defer building the matrix until it is
    needed. With neither fallback usable the policy raises, as before.
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
            matrix_kind=matrix_kind,
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
            matrix_kind=matrix_kind,
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

    accepted: _CurvatureAnalysis | None = None
    actual_source: AcceptedCurvatureKind = "fisher"
    if fisher_matrix is not None:
        fisher = _analyze_curvature(fisher_matrix, label="Fisher", policy=policy)
        if not fisher.materially_indefinite:
            accepted = fisher
        elif semidefinite_fallback is None:
            raise CurvaturePolicyError("Fisher curvature is materially indefinite")
    if accepted is None:
        if semidefinite_fallback is None:
            raise RepeatedCurvatureIndefinitenessError(
                "Fisher curvature is required after repeated material indefiniteness"
            )
        accepted = _analyze_curvature(
            semidefinite_fallback(),
            label="Nearest semidefinite",
            policy=policy,
        )
        if accepted.materially_indefinite:
            raise CurvaturePolicyError("nearest semidefinite curvature is materially indefinite")
        actual_source = "nearest_semidefinite"

    fallback_count = current_state.fallback_count + 1
    telemetry = _telemetry(
        requested_source,
        actual_source,
        reason="material_indefiniteness_after_retry",
        minimum_eigenvalue=requested.minimum_eigenvalue,
        analysis=accepted,
        fallback_count=fallback_count,
        matrix_kind=matrix_kind,
    )
    return CurvatureDecision(
        matrix=accepted.matrix,
        decomposition=accepted.decomposition,
        telemetry=telemetry,
        retry_required=False,
        state=CurvaturePolicyState(retry_attempted=False, fallback_count=fallback_count),
    )
