"""Eligibility and cost policy for the structured direct backend."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.overrides import (
    _factor_smooth_override_local_blocks,
    _structured_override_incompatibility,
)
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredGroupSelection:
    """Dominant structured group choice or a recorded dense-fallback reason."""

    group_index: int | None
    group_name: str | None
    fallback_reason: str | None


@dataclass(frozen=True)
class StructuredBackendDecision:
    """Resolved direct backend and the selected dominant block.

    ``auto_cost_ratio`` carries the crossover model's predicted
    structured/dense factorization flop ratio whenever ``direct_solve="auto"``
    reached the cost comparison, for either outcome.  It is ``None`` for
    forced backends and for eligibility (non-cost) fallbacks.  Callers put it
    in the fit profile beside the realized timings so the crossover constants
    can be recalibrated against real fits (issue #343).
    """

    use_structured: bool
    group_index: int | None
    group_name: str | None
    fallback_reason: str | None
    auto_cost_ratio: float | None = None


_AUTO_MIN_COEFFICIENT_WIDTH = 32
# Measured crossover, August 2026 (issue #343).  The flop ratio compares only
# the two factorizations, but on n >> p fits the factorization is a small
# minority of per-iteration work -- the shared O(n)-row moment build dominates
# (the standard operations count for this fitting problem is O(n p^2 + M p^3),
# Wood 2015, JRSS-C 64(1); both terms belong in any method comparison).  The
# structured path also carries per-outer-iteration derivative machinery the
# dense path does not.  So a "25% cheaper factorization" prediction is a
# prediction about a term that does not decide the fit time, and the previous
# 0.75 bound admitted wide-border cases where the structured backend was
# measured ~2x slower end to end.  Block elimination pays off when it
# eliminates most of the matrix and keeps the dense border small -- the regime
# every established user of this factorization occupies (lme4's sparse
# Cholesky, Bates et al. 2015, JSS 67(1); doubly-bordered block-diagonal
# solvers).  End-to-end anchors on a real ~67k-row Tweedie(1.5) log-link
# pricing workload, exact REML path, single-threaded:
#
#   ratio 0.596 (K=23  beside q=77):  structured 2.03x slower
#   ratio 0.316 (K=39  beside q=49):  structured 1.66x slower
#   ratio 0.148 (K=80  beside q=49):  structured 1.41x slower
#   ratio 0.104 (K=105 beside q=49):  structured 1.10x slower
#   ratio 0.033 (K=225 beside q=49):  structured 1.56x FASTER
#
# Synthetic FactorSmooth ("fs" and "sz") sweeps reproduce the same ordering
# (mid-ratio loses, tiny-ratio wins), and the discrete cached-W path is
# insensitive at mid ratio (measured a tie), so one constant governs all three
# geometries.  0.05 splits the measured bracket [0.033, 0.104] with margin on
# both sides.  For the scalar geometry the ratio has the closed form
# ((q+1)/(p+1))^2, so this bound equivalently requires the dominant block to
# span at least ~78% of the augmented width.
_AUTO_MAX_STRUCTURED_COST_RATIO = 0.05


def _structured_auto_is_beneficial(
    dominant_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Apply the measured scalar-Schur crossover and return its cost ratio.

    The width floor keeps tiny systems dense regardless of shape.  The ratio
    bound (see ``_AUTO_MAX_STRUCTURED_COST_RATIO``) demands that the scalar
    Schur elimination remove the overwhelming majority of the augmented
    width; for this geometry the ratio reduces to ``((q+1)/(p+1))**2``.  The
    intercept is included in both algebra estimates.
    """
    if dominant_size < 1 or small_size < 0:
        raise ValueError("Structured auto dimensions must be non-negative with a dominant block.")
    coefficient_width = dominant_size + small_size
    dense_dimension = coefficient_width + 1
    schur_small_dimension = small_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(schur_small_dimension**3 + dominant_size * schur_small_dimension**2)
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _block_structured_auto_is_beneficial(
    n_levels: int,
    block_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Estimate the block-Schur crossover from the actual ``K``, ``k``, and ``q``.

    The estimate counts local factorizations, local solves against the
    dense-small block, Schur accumulation, and the final dense-small
    factorization.  It intentionally ignores shared row-moment work, so auto
    selection only chooses the block backend when its linear algebra alone has
    a material cubic-cost advantage.
    """
    if n_levels < 1 or block_size < 1 or small_size < 0:
        raise ValueError(
            "Block structured auto dimensions require positive K and k and non-negative q."
        )
    coefficient_width = n_levels * block_size + small_size
    dense_dimension = coefficient_width + 1
    schur_small_dimension = small_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(
        n_levels * block_size**3
        + n_levels * block_size**2 * schur_small_dimension
        + n_levels * block_size * schur_small_dimension**2
        + schur_small_dimension**3
    )
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _sum_to_zero_structured_auto_is_beneficial(
    n_levels: int,
    block_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Estimate constrained SZ work from local blocks and its dense border."""
    if n_levels < 2 or block_size < 1 or small_size < 0:
        raise ValueError(
            "SZ structured auto dimensions require K >= 2, positive k, and non-negative q."
        )
    coefficient_width = (n_levels - 1) * block_size + small_size
    dense_dimension = coefficient_width + 1
    border_width = small_size + block_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(
        n_levels * block_size**3 + n_levels * block_size**2 * border_width + border_width**3
    )
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _structured_auto_cost_decision(
    dominant_matrix: GroupMatrix,
    selection: StructuredGroupSelection,
    coefficient_width: int,
    small_size: int,
) -> StructuredBackendDecision:
    """Return the measured automatic crossover decision for one selected block."""
    if isinstance(dominant_matrix, FactorSmoothGroupMatrix):
        if dominant_matrix.factor_basis == "sz":
            use_structured, cost_ratio = _sum_to_zero_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
        else:
            use_structured, cost_ratio = _block_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
        geometry_name = "FactorSmooth"
        dimensions = f"K={dominant_matrix.n_levels}, k={dominant_matrix.block_size}, q={small_size}"
    elif isinstance(dominant_matrix, RandomEffectGroupMatrix):
        use_structured, cost_ratio = _structured_auto_is_beneficial(
            dominant_matrix.shape[1],
            small_size,
        )
        geometry_name = "RandomEffect"
        dimensions = f"K={dominant_matrix.shape[1]}, q={small_size}"
    else:  # pragma: no cover - StructuredGroupSelection invariant
        raise RuntimeError("structured selection chose an unsupported group matrix")

    fallback_reason = None
    if not use_structured:
        fallback_reason = (
            f"{geometry_name} geometry is below the measured structured crossover "
            f"(p={coefficient_width}, {dimensions}, estimated_cost_ratio={cost_ratio:.3f}; "
            f"require p >= {_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= "
            f"{_AUTO_MAX_STRUCTURED_COST_RATIO:.2f})"
        )
    logger.debug(
        "structured auto crossover: %s %s (p=%d, %s, estimated_cost_ratio=%.4f)",
        geometry_name,
        "selected" if use_structured else "declined",
        coefficient_width,
        dimensions,
        cost_ratio,
    )
    return StructuredBackendDecision(
        use_structured=use_structured,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=fallback_reason,
        auto_cost_ratio=cost_ratio,
    )


def record_auto_backend_decision(
    profile: dict | None,
    direct_solve: str,
    decision: StructuredBackendDecision,
    *,
    log: bool = True,
) -> None:
    """Record one automatic crossover decision for offline recalibration.

    Writes the predicted factorization cost ratio and the choice into the fit
    profile, where they sit beside the realized per-phase timings
    (``irls_gram_s``, ``irls_solve_s``, ``reml_*_s``) that a forced-backend
    rerun can be compared against.  Emits one INFO line when ``auto`` commits
    to the structured backend; drivers that resolve once per fit pass
    ``log=True``, per-solve callers pass ``log=False`` so repeated inner
    resolutions stay quiet.
    """
    if direct_solve != "auto" or decision.auto_cost_ratio is None:
        return
    if profile is not None:
        profile["structured_auto_cost_ratio"] = decision.auto_cost_ratio
        profile["structured_auto_selected"] = decision.use_structured
    if log and decision.use_structured:
        logger.info(
            "direct_solve='auto' chose the structured backend for group %r "
            "(estimated factorization cost ratio %.4f <= %.2f). The fit "
            "profile records this prediction beside realized timings; compare "
            "against a direct_solve='gram' rerun to recalibrate the crossover.",
            decision.group_name,
            decision.auto_cost_ratio,
            _AUTO_MAX_STRUCTURED_COST_RATIO,
        )


def _selection_failure(
    reason: str,
    mode: Literal["auto", "structured"],
) -> StructuredGroupSelection:
    if mode == "structured":
        raise ValueError(f"direct_solve='structured' is ineligible: {reason}")
    return StructuredGroupSelection(
        group_index=None,
        group_name=None,
        fallback_reason=reason,
    )


def select_structured_group(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    *,
    mode: Literal["auto", "structured"],
) -> StructuredGroupSelection:
    """Select one algebraically supported dominant structured block."""
    if mode not in ("auto", "structured"):
        raise ValueError("Structured selection mode must be 'auto' or 'structured'.")
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")

    for group in groups:
        if group.constraints is not None:
            return _selection_failure(
                f"group {group.name!r} has coefficient constraints",
                mode,
            )
        if group.scop_reparameterization is not None:
            return _selection_failure(
                f"group {group.name!r} has unsupported SCOP geometry",
                mode,
            )

    factor_smooth_indices = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, FactorSmoothGroupMatrix)
    ]
    random_effect_indices = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix)
    ]
    if len(factor_smooth_indices) > 1:
        names = [groups[index].name for index in factor_smooth_indices]
        return _selection_failure(
            f"the structured backend supports at most one FactorSmooth term; found {names!r}",
            mode,
        )
    if factor_smooth_indices:
        dominant_index = factor_smooth_indices[0]
    elif random_effect_indices:
        dominant_index = max(
            random_effect_indices,
            key=lambda index: group_matrices[index].shape[1],
        )
    else:
        return _selection_failure("the model has no RandomEffect or FactorSmooth term", mode)

    dominant_group = groups[dominant_index]
    dominant_matrix = group_matrices[dominant_index]
    if dominant_group.size != dominant_matrix.shape[1]:
        term_kind = (
            "FactorSmooth"
            if isinstance(dominant_matrix, FactorSmoothGroupMatrix)
            else "RandomEffect"
        )
        return _selection_failure(
            f"{term_kind} group {dominant_group.name!r} has inconsistent coefficient geometry",
            mode,
        )
    return StructuredGroupSelection(
        group_index=dominant_index,
        group_name=dominant_group.name,
        fallback_reason=None,
    )


def _factor_smooth_component_lambda(
    group_name: str,
    suffix: str,
    lambda2: float | dict[str, float],
) -> float:
    """Resolve one repeated factor-smooth component lambda."""
    from superglm.reml.penalty_algebra import resolve_component_lambda

    return resolve_component_lambda(lambda2, group_name, suffix)


def _factor_smooth_local_penalty(
    matrix: FactorSmoothGroupMatrix,
    group_name: str,
    lambda2: float | dict[str, float],
) -> tuple[NDArray, tuple[tuple[str, float], ...]]:
    """Build the exact lambda-scaled local penalty and its cache identity."""
    local_penalty = np.zeros((matrix.block_size, matrix.block_size), dtype=np.float64)
    resolved_components: list[tuple[str, float]] = []
    for suffix, omega in matrix.repeated_penalty_components:
        lam = _factor_smooth_component_lambda(group_name, suffix, lambda2)
        resolved_components.append((suffix, lam))
        values = np.asarray(omega, dtype=np.float64)
        local_penalty += lam * (0.5 * (values + values.T))
    return local_penalty, tuple(resolved_components)


def _factor_smooth_singular_local_level(
    matrix: FactorSmoothGroupMatrix,
    row_weights: NDArray,
    local_penalty: NDArray,
    penalty_identity: tuple[tuple[str, float], ...],
) -> int | None:
    """Return the first numerically singular weighted local block, if any."""
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("row_weights must match the structured design row count.")
    contiguous_weights = np.ascontiguousarray(weights)
    weight_digest = hashlib.blake2b(
        contiguous_weights.data,
        digest_size=16,
    ).digest()
    cache_key = (penalty_identity, weight_digest)
    if getattr(matrix, "_structured_feasibility_key", None) == cache_key:
        return getattr(matrix, "_structured_feasibility_level", None)

    information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
        weights,
        np.zeros_like(weights),
    )
    local_blocks = np.asarray(information, dtype=np.float64) + local_penalty[None, :, :]
    singular_level = _first_singular_factor_smooth_block(local_blocks)
    matrix._structured_feasibility_level = singular_level
    matrix._structured_feasibility_key = cache_key
    return singular_level


def _first_singular_factor_smooth_block(
    local_blocks: NDArray,
    *,
    scale_floor: float = 1.0,
) -> int | None:
    """Return the first local block that is not numerically positive definite."""
    symmetric = 0.5 * (local_blocks + local_blocks.transpose(0, 2, 1))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    block_size = local_blocks.shape[1]
    scales = np.maximum(np.max(np.abs(eigenvalues), axis=1), scale_floor)
    thresholds = np.finfo(np.float64).eps * max(block_size, 1) * scales * 10.0
    singular = eigenvalues[:, 0] <= thresholds
    return int(np.flatnonzero(singular)[0]) if np.any(singular) else None


def _factor_smooth_override_singular_local_level(
    matrix: FactorSmoothGroupMatrix,
    row_weights: NDArray,
    local_penalties: NDArray,
) -> int | None:
    """Check override-defined local blocks against positive-weight row support."""
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("row_weights must match the structured design row count.")
    expected_shape = (matrix.n_levels, matrix.block_size, matrix.block_size)
    if local_penalties.shape != expected_shape:
        raise ValueError(f"FactorSmooth override local penalties must have shape {expected_shape}.")
    information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
        weights,
        np.zeros_like(weights),
    )
    return _first_singular_factor_smooth_block(
        np.asarray(information, dtype=np.float64) + local_penalties
    )


def _backend_ineligibility(
    reason: str,
    mode: Literal["auto", "structured"],
    selection: StructuredGroupSelection,
) -> StructuredBackendDecision:
    """Return an automatic fallback or reject an unsafe forced backend."""
    if mode == "structured":
        raise ValueError(f"direct_solve='structured' is ineligible: {reason}")
    return StructuredBackendDecision(
        use_structured=False,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=reason,
    )


def _factor_smooth_zero_penalty_component(
    matrix: FactorSmoothGroupMatrix,
    group_name: str,
    lambda2: float | dict[str, float],
) -> str | None:
    """Return the first repeated component whose requested penalty is zero."""
    for suffix, _omega in matrix.repeated_penalty_components:
        lam = _factor_smooth_component_lambda(group_name, suffix, lambda2)
        if lam == 0.0:
            return suffix
    return None


def resolve_structured_backend(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    *,
    direct_solve: str,
    coefficient_width: int,
    row_weights: NDArray | None = None,
    lambda2: float | dict[str, float] | None = None,
    S_override: NDArray | None = None,
) -> StructuredBackendDecision:
    """Resolve forced/automatic scalar Schur use once for a direct fit."""
    if direct_solve not in ("auto", "structured"):
        return StructuredBackendDecision(
            use_structured=False,
            group_index=None,
            group_name=None,
            fallback_reason=None,
        )

    mode: Literal["auto", "structured"] = "structured" if direct_solve == "structured" else "auto"
    selection = select_structured_group(group_matrices, groups, mode=mode)
    if selection.group_index is None:
        return StructuredBackendDecision(
            use_structured=False,
            group_index=None,
            group_name=None,
            fallback_reason=selection.fallback_reason,
        )
    group_name = selection.group_name
    if group_name is None:  # pragma: no cover - StructuredGroupSelection invariant
        raise RuntimeError("structured group selection omitted its group name")

    dominant_matrix = group_matrices[selection.group_index]
    dominant_size = dominant_matrix.shape[1]
    small_size = coefficient_width - dominant_size
    auto_cost_decision = (
        _structured_auto_cost_decision(
            dominant_matrix,
            selection,
            coefficient_width,
            small_size,
        )
        if mode == "auto"
        else None
    )
    dominant_group = groups[selection.group_index]
    override_penalty: NDArray | None = None
    override_local_penalties: NDArray | None = None
    if S_override is not None:
        override_penalty = np.asarray(S_override, dtype=np.float64)
        if override_penalty.shape != (coefficient_width, coefficient_width):
            raise ValueError(
                f"S_override must have shape ({coefficient_width}, {coefficient_width})."
            )
        flat_structured = np.arange(
            dominant_group.start,
            dominant_group.end,
            dtype=np.intp,
        )
        small_mask = np.ones(coefficient_width, dtype=bool)
        small_mask[flat_structured] = False
        small_indices = np.flatnonzero(small_mask)
        override_geometry: Literal[
            "random_effect",
            "factor_smooth",
            "sum_to_zero",
        ]
        if isinstance(dominant_matrix, RandomEffectGroupMatrix):
            structured_indices = flat_structured
            override_geometry = "random_effect"
        elif isinstance(dominant_matrix, FactorSmoothGroupMatrix):
            public_levels = (
                dominant_matrix.n_levels - 1
                if dominant_matrix.factor_basis == "sz"
                else dominant_matrix.n_levels
            )
            structured_indices = flat_structured.reshape(
                public_levels,
                dominant_matrix.block_size,
            )
            override_geometry = (
                "sum_to_zero" if dominant_matrix.factor_basis == "sz" else "factor_smooth"
            )
        else:  # pragma: no cover - StructuredGroupSelection invariant
            raise RuntimeError("structured selection chose an unsupported group matrix")
        incompatibility = _structured_override_incompatibility(
            override_penalty,
            small_indices=small_indices,
            structured_indices=structured_indices,
            geometry=override_geometry,
        )
        if incompatibility is not None:
            return _backend_ineligibility(
                incompatibility,
                mode,
                selection,
            )
        if (
            isinstance(dominant_matrix, FactorSmoothGroupMatrix)
            and dominant_matrix.factor_basis != "sz"
        ):
            override_local_penalties = _factor_smooth_override_local_blocks(
                override_penalty,
                structured_indices,
                sum_to_zero=False,
            )
    if isinstance(dominant_matrix, RandomEffectGroupMatrix) and (
        lambda2 is not None or S_override is not None
    ):
        override_diagonal: NDArray | None = None
        if override_penalty is not None:
            override_diagonal = np.diag(override_penalty[dominant_group.sl, dominant_group.sl])
            has_dominant_penalty = bool(np.any(override_diagonal > 0.0))
        elif isinstance(lambda2, dict):
            has_dominant_penalty = float(lambda2.get(group_name, 0.0)) != 0.0
        else:
            has_dominant_penalty = lambda2 is not None and float(lambda2) != 0.0
        if not has_dominant_penalty:
            if row_weights is not None:
                weights = np.asarray(row_weights, dtype=np.float64)
                if weights.shape != (dominant_matrix.shape[0],):
                    raise ValueError("row_weights must match the structured design row count.")
                level_weight = np.bincount(
                    dominant_matrix.codes,
                    weights=weights,
                    minlength=dominant_matrix.n_levels,
                )
                if np.any(level_weight <= 0.0):
                    return _backend_ineligibility(
                        (
                            f"RandomEffect group {group_name!r} has a level with "
                            "zero total weight and zero penalty"
                        ),
                        mode,
                        selection,
                    )
            return _backend_ineligibility(
                (
                    f"RandomEffect group {group_name!r} has zero penalty and is "
                    "aliased with the fitted intercept"
                ),
                mode,
                selection,
            )
        if row_weights is not None and override_diagonal is not None:
            weights = np.asarray(row_weights, dtype=np.float64)
            if weights.shape != (dominant_matrix.shape[0],):
                raise ValueError("row_weights must match the structured design row count.")
            level_weight = np.bincount(
                dominant_matrix.codes,
                weights=weights,
                minlength=dominant_matrix.n_levels,
            )
            if np.any(level_weight + override_diagonal <= 0.0):
                return _backend_ineligibility(
                    (
                        f"RandomEffect group {group_name!r} has non-positive local "
                        "information under the authoritative S_override"
                    ),
                    mode,
                    selection,
                )
    if (
        isinstance(dominant_matrix, FactorSmoothGroupMatrix)
        and dominant_matrix.factor_basis != "sz"
    ):
        if override_local_penalties is not None:
            structurally_singular_level = _first_singular_factor_smooth_block(
                override_local_penalties,
                scale_floor=0.0,
            )
            if structurally_singular_level is not None:
                level_label = dominant_matrix.levels[structurally_singular_level]
                return _backend_ineligibility(
                    (
                        f"FactorSmooth group {group_name!r} authoritative S_override "
                        f"has a zero penalty component or singular local penalty block "
                        f"for level {level_label!r}, which can alias the intercept "
                        "or population smooth"
                    ),
                    mode,
                    selection,
                )
            numerically_singular_level = _first_singular_factor_smooth_block(
                override_local_penalties
            )
            if row_weights is not None and numerically_singular_level is not None:
                if auto_cost_decision is not None and not auto_cost_decision.use_structured:
                    return auto_cost_decision
                singular_level = _factor_smooth_override_singular_local_level(
                    dominant_matrix,
                    row_weights,
                    override_local_penalties,
                )
                if singular_level is not None:
                    level_label = dominant_matrix.levels[singular_level]
                    return _backend_ineligibility(
                        (
                            f"FactorSmooth group {group_name!r} has a singular local block "
                            f"for level {level_label!r} under the authoritative S_override"
                        ),
                        mode,
                        selection,
                    )
        elif lambda2 is not None:
            local_penalty, penalty_identity = _factor_smooth_local_penalty(
                dominant_matrix,
                group_name,
                lambda2,
            )
            zero_component = _factor_smooth_zero_penalty_component(
                dominant_matrix,
                group_name,
                lambda2,
            )
            if zero_component is not None:
                return _backend_ineligibility(
                    (
                        f"FactorSmooth group {group_name!r} has zero penalty component "
                        f"{zero_component!r}, which can alias the intercept "
                        "or population smooth"
                    ),
                    mode,
                    selection,
                )
            numerically_singular_penalty = (
                _first_singular_factor_smooth_block(local_penalty[None, :, :]) is not None
            )
            if row_weights is not None and numerically_singular_penalty:
                if auto_cost_decision is not None and not auto_cost_decision.use_structured:
                    return auto_cost_decision
                singular_level = _factor_smooth_singular_local_level(
                    dominant_matrix,
                    row_weights,
                    local_penalty,
                    penalty_identity,
                )
                if singular_level is not None:
                    level_label = dominant_matrix.levels[singular_level]
                    return _backend_ineligibility(
                        (
                            f"FactorSmooth group {group_name!r} has a singular local block "
                            f"for level {level_label!r} under the requested weights "
                            "and penalties"
                        ),
                        mode,
                        selection,
                    )
    if mode == "structured":
        return StructuredBackendDecision(
            use_structured=True,
            group_index=selection.group_index,
            group_name=group_name,
            fallback_reason=None,
        )
    if auto_cost_decision is None:  # pragma: no cover - mode invariant
        raise RuntimeError("automatic structured resolution omitted its cost decision")
    return auto_cost_decision
