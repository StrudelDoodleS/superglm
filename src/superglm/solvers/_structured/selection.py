"""Eligibility and cost policy for the structured direct backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.overrides import _structured_override_incompatibility
from superglm.types import GroupSlice


@dataclass(frozen=True)
class StructuredGroupSelection:
    """Dominant structured group choice or a recorded dense-fallback reason."""

    group_index: int | None
    group_name: str | None
    fallback_reason: str | None


@dataclass(frozen=True)
class StructuredBackendDecision:
    """Resolved direct backend and the selected dominant block."""

    use_structured: bool
    group_index: int | None
    group_name: str | None
    fallback_reason: str | None


_AUTO_MIN_COEFFICIENT_WIDTH = 32
_AUTO_MAX_STRUCTURED_COST_RATIO = 0.75


def _structured_auto_is_beneficial(
    dominant_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Apply the measured scalar-Schur crossover and return its cost ratio.

    July 2026 end-to-end REML profiles found dense wins below 32 slope
    coefficients, while scalar Schur wins materially at and above that width
    when its leading factor/solve work is no more than 75% of dense work.
    The intercept is included in both algebra estimates.
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


def _factor_smooth_singular_local_level(
    matrix: FactorSmoothGroupMatrix,
    group_name: str,
    row_weights: NDArray,
    lambda2: float | dict[str, float],
) -> int | None:
    """Return the first structurally singular penalized local block, if any."""
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("row_weights must match the structured design row count.")
    active_components: list[str] = []
    local_penalty = np.zeros((matrix.block_size, matrix.block_size), dtype=np.float64)
    for suffix, omega in matrix.repeated_penalty_components:
        if isinstance(lambda2, dict):
            lam = float(
                lambda2.get(
                    f"{group_name}:{suffix}",
                    lambda2.get(group_name, 0.0),
                )
            )
        else:
            lam = float(lambda2)
        if lam:
            active_components.append(suffix)
            local_penalty += np.asarray(omega, dtype=np.float64)

    penalty_eigenvalues = np.linalg.eigvalsh(local_penalty)
    penalty_scale = max(float(np.max(np.abs(penalty_eigenvalues), initial=0.0)), 1.0)
    penalty_threshold = np.finfo(np.float64).eps * max(matrix.block_size, 1) * penalty_scale * 10.0
    if penalty_eigenvalues[0] > penalty_threshold:
        return None

    positive_rows = weights > 0.0
    packed_support = np.packbits(positive_rows)
    support_digest = hashlib.blake2b(
        packed_support.data,
        digest_size=16,
    ).digest()
    cache_key = (tuple(active_components), support_digest)
    if getattr(matrix, "_structured_feasibility_key", None) == cache_key:
        return getattr(matrix, "_structured_feasibility_level", None)

    information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
        positive_rows.astype(np.float64),
        np.zeros_like(weights),
    )
    local_blocks = np.asarray(information, dtype=np.float64) + local_penalty[None, :, :]

    eigenvalues = np.linalg.eigvalsh(local_blocks)
    scales = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    thresholds = np.finfo(np.float64).eps * max(matrix.block_size, 1) * scales * 10.0
    singular = eigenvalues[:, 0] <= thresholds
    singular_level = int(np.flatnonzero(singular)[0]) if np.any(singular) else None
    matrix._structured_feasibility_key = cache_key
    matrix._structured_feasibility_level = singular_level
    return singular_level


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
        if isinstance(lambda2, dict):
            lam = float(
                lambda2.get(
                    f"{group_name}:{suffix}",
                    lambda2.get(group_name, 0.0),
                )
            )
        else:
            lam = float(lambda2)
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
    dominant_group = groups[selection.group_index]
    override_penalty: NDArray | None = None
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
    if isinstance(dominant_matrix, FactorSmoothGroupMatrix) and lambda2 is not None:
        if dominant_matrix.factor_basis != "sz":
            zero_component = _factor_smooth_zero_penalty_component(
                dominant_matrix,
                group_name,
                lambda2,
            )
            if zero_component is not None:
                return _backend_ineligibility(
                    (
                        f"FactorSmooth group {group_name!r} has zero penalty component "
                        f"{zero_component!r}, which can alias the intercept or population smooth"
                    ),
                    mode,
                    selection,
                )
        if row_weights is not None:
            singular_level = _factor_smooth_singular_local_level(
                dominant_matrix,
                group_name,
                row_weights,
                lambda2,
            )
            if singular_level is not None:
                level_label = dominant_matrix.levels[singular_level]
                return _backend_ineligibility(
                    (
                        f"FactorSmooth group {group_name!r} has a singular local block "
                        f"for level {level_label!r} under the requested weights and penalties"
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
    else:
        use_structured, cost_ratio = _structured_auto_is_beneficial(
            dominant_size,
            small_size,
        )
    fallback_reason = None
    if not use_structured:
        geometry_name = (
            "FactorSmooth"
            if isinstance(
                group_matrices[selection.group_index],
                FactorSmoothGroupMatrix,
            )
            else "RandomEffect"
        )
        if isinstance(dominant_matrix, FactorSmoothGroupMatrix):
            dimensions = (
                f"K={dominant_matrix.n_levels}, k={dominant_matrix.block_size}, q={small_size}"
            )
        else:
            dimensions = f"K={dominant_size}, q={small_size}"
        fallback_reason = (
            f"{geometry_name} geometry is below the measured structured crossover "
            f"(p={coefficient_width}, {dimensions}, estimated_cost_ratio={cost_ratio:.3f}; "
            f"require p >= {_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= "
            f"{_AUTO_MAX_STRUCTURED_COST_RATIO:.2f})"
        )
    return StructuredBackendDecision(
        use_structured=use_structured,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=fallback_reason,
    )
