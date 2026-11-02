"""Eligibility and cost policy for the structured direct backend."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
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
    """Dominant structured group choice, or why the model's terms take gram.

    ``chain_group_indices`` is ``()`` when no group was selected and
    ``(group_index,)`` otherwise; the nested chain is resolved later, by
    ``resolve_structured_backend``.
    """

    group_index: int | None
    group_name: str | None
    fallback_reason: str | None
    chain_group_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class StructuredBackendDecision:
    """Resolved direct backend and the selected dominant block.

    Every field is decided by the model's terms, their sizes and penalties
    and the call's configuration, never by values, weights or a caught error;
    a fit never switches solver after it.  ``fallback_reason`` says why
    ``auto`` takes gram.  ``auto_cost_ratio`` is the predicted
    structured/dense cost ratio when ``auto`` reached the cost comparison,
    else ``None``.  ``chain_group_indices`` is a RandomEffect leaf's nested
    chain, coarsest to finest, ending with ``group_index``; ``(g,)`` for a
    FactorSmooth block; ``()`` without a structured group.
    ``nested_fallback_reason`` says why a found chain was declined to its leaf.
    """

    use_structured: bool
    group_index: int | None
    group_name: str | None
    fallback_reason: str | None
    auto_cost_ratio: float | None = None
    chain_group_indices: tuple[int, ...] = ()
    nested_fallback_reason: str | None = None


_AUTO_MIN_COEFFICIENT_WIDTH = 32
# RandomEffect crossover.  With w = p + 1, a structured factor that leaves a
# border of b = w - (chain levels) columns costs K b^2 + b^3 = w b^2 flops per
# factorization against dense w^3, a ratio of (b / w)^2.  A chain also makes
# dense passes of n b^2 over the rows at every iterate, priced as
# _AUTO_NESTED_ROW_PASSES passes: ratio (b / w)^2 (1 + passes n / w).  A lone
# level is a chain of one.  Memory (w b against w^2) never reverses the order.
# Both constants are calibrated on complete fit_reml fits (issue #343 and the
# nested-chain work); 0.75 keeps near-degenerate shapes dense.  No chain is
# declined for its family, link, working rows or memory.
_AUTO_MAX_NESTED_COST_RATIO = 0.75
_AUTO_NESTED_ROW_PASSES = 0.08
# FactorSmooth and sum-to-zero blocks take the structured route below this
# factorization cost ratio (issue #343); a sum-to-zero block takes the balance
# tree.
_AUTO_MAX_STRUCTURED_COST_RATIO = 0.05


def _random_effect_auto_cost_ratio(
    n_rows: int,
    coefficient_width: int,
    level_sizes: Sequence[int],
) -> float:
    """Predicted structured/dense cost ratio of a RandomEffect leaf's chain.

    ``level_sizes`` runs coarsest to finest over the chain; the model is the
    comment above ``_AUTO_MAX_NESTED_COST_RATIO``.
    """
    width = coefficient_width + 1
    border = width - sum(level_sizes)
    return (border / width) ** 2 * (1.0 + _AUTO_NESTED_ROW_PASSES * n_rows / width)


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
    groups: list[GroupSlice],
    coefficient_width: int,
    chain: tuple[int, ...],
    nested_fallback_reason: str | None,
) -> StructuredBackendDecision:
    """Return the automatic crossover decision for one selected block."""
    small_size = coefficient_width - dominant_matrix.shape[1]
    bound = _AUTO_MAX_STRUCTURED_COST_RATIO
    if isinstance(dominant_matrix, RandomEffectGroupMatrix):
        sizes = [groups[index].size for index in chain]
        cost_ratio = _random_effect_auto_cost_ratio(
            dominant_matrix.shape[0], coefficient_width, sizes
        )
        bound = _AUTO_MAX_NESTED_COST_RATIO
        use_structured = coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH and cost_ratio <= bound
        geometry_name = "RandomEffect"
        dimensions = f"n={dominant_matrix.shape[0]}, levels={sizes}"
    elif isinstance(dominant_matrix, FactorSmoothGroupMatrix):
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
    else:  # pragma: no cover - StructuredGroupSelection invariant
        raise RuntimeError("structured selection chose an unsupported group matrix")

    fallback_reason = None
    if not use_structured:
        fallback_reason = (
            f"{geometry_name} geometry is below the measured structured crossover "
            f"(p={coefficient_width}, {dimensions}, estimated_cost_ratio={cost_ratio:.3f}; "
            f"require p >= {_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= {bound:.2f})"
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
        chain_group_indices=chain,
        nested_fallback_reason=nested_fallback_reason,
    )


def record_auto_backend_decision(
    profile: dict | None,
    direct_solve: str,
    decision: StructuredBackendDecision,
    *,
    log: bool = True,
) -> None:
    """Record one automatic crossover decision for offline recalibration.

    Writes the predicted cost ratio and the choice into the fit profile,
    beside the realized phase timings.  ``log=True`` emits one INFO line when
    ``auto`` chooses the structured backend; per-solve callers pass
    ``log=False``.
    """
    if direct_solve != "auto" or decision.auto_cost_ratio is None:
        return
    if profile is not None:
        profile["structured_auto_cost_ratio"] = decision.auto_cost_ratio
        profile["structured_auto_selected"] = decision.use_structured
    if log and decision.use_structured:
        logger.info(
            "direct_solve='auto' chose the structured backend for group %r "
            "(predicted cost ratio %.4f). The fit "
            "profile records this prediction beside realized timings; compare "
            "against a direct_solve='gram' rerun to recalibrate the crossover.",
            decision.group_name,
            decision.auto_cost_ratio,
        )


def has_structured_term(group_matrices: Sequence) -> bool:
    """Whether the design has a term the structured backend eliminates (RandomEffect, FactorSmooth).

    A design without one always takes the dense Gram route.
    """
    return any(
        isinstance(matrix, RandomEffectGroupMatrix | FactorSmoothGroupMatrix)
        for matrix in group_matrices
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
    if dominant_group.size != dominant_matrix.shape[1]:  # pragma: no cover - design invariant
        term_kind = (
            "FactorSmooth"
            if isinstance(dominant_matrix, FactorSmoothGroupMatrix)
            else "RandomEffect"
        )
        raise RuntimeError(
            f"{term_kind} group {dominant_group.name!r} has inconsistent coefficient geometry."
        )
    return StructuredGroupSelection(
        group_index=dominant_index,
        group_name=dominant_group.name,
        fallback_reason=None,
        chain_group_indices=(dominant_index,),
    )


def nested_parent_codes(
    child: RandomEffectGroupMatrix,
    parent: RandomEffectGroupMatrix,
) -> NDArray | None:
    """Return each child level's parent code, or ``None`` when not strictly nested.

    The strict-nesting count test in O(n) without a sort: each observed child
    code must meet a single parent code.  Unobserved child levels point at
    parent 0; their weight is exactly zero, so the pointer is never read.
    """
    return _parent_codes(child.codes, parent.codes, child.n_levels)


def _parent_codes(child: NDArray, parent: NDArray, n_child_levels: int) -> NDArray | None:
    first = np.zeros(n_child_levels, dtype=np.intp)
    first[child] = parent
    if not np.array_equal(first[child], parent):
        return None
    return first


def _group_spans(groups: list[GroupSlice]) -> tuple[tuple[str, int, int], ...]:
    return tuple((group.name, group.start, group.end) for group in groups)


# The nesting cache: pair tests, chains and the chains' trees with their leaf
# row orders, keyed by group indices and spans.  One dict in a design's
# ``_structured_layout_cache``, shared by reference with every lambda rebuild
# (``carry_nesting_cache``), so its O(n) passes run once per design lineage.
# Owner: the lineage's RandomEffect codes.  Lifetime: the first design built on
# them and all its rebuilds; not pickled.  Invalidation: none, since a lambda,
# weight or border change moves no code.  Layouts hold border matrices, so they
# stay per design.
_NESTING_CACHE_KEY = "nesting"


def shared_nesting_cache(layout_cache: dict | None) -> dict:
    """Return the nesting cache inside a design's layout cache (a fresh one for ``None``)."""
    return {} if layout_cache is None else layout_cache.setdefault(_NESTING_CACHE_KEY, {})


def carry_nesting_cache(source: dict, target: dict) -> None:
    """Share one design's nesting cache with its lambda rebuild (layout caches)."""
    target[_NESTING_CACHE_KEY] = shared_nesting_cache(source)


def cached_nested_parent_codes(
    random_effects: dict[int, RandomEffectGroupMatrix],
    groups: list[GroupSlice],
    child_index: int,
    parent_index: int,
    cache: dict,
) -> NDArray | None:
    """``nested_parent_codes`` for two design groups through the nesting cache.

    ``random_effects`` maps design group indices to their RandomEffect
    matrices; ``cache`` is the nesting cache (contract above
    ``_NESTING_CACHE_KEY``).
    """
    key = ("nested_parent", child_index, parent_index, _group_spans(groups))
    if key not in cache:
        cache[key] = nested_parent_codes(random_effects[child_index], random_effects[parent_index])
    return cache[key]


def _declared_ancestors(
    random_effects: dict[int, RandomEffectGroupMatrix],
    groups: list[GroupSlice],
    leaf_index: int,
) -> frozenset[int]:
    """The RandomEffect terms the leaf is declared nested in, transitively.

    Follows ``RandomEffect(nested_in=)`` from the leaf through each declared
    parent's own declaration, by feature name; a name that is not a
    RandomEffect term of this design ends the walk.
    """
    by_name = {groups[index].feature_name: index for index in random_effects}
    ancestors: list[int] = []
    parent = random_effects[leaf_index].declared_parent
    while parent is not None and parent in by_name and by_name[parent] not in ancestors:
        index = by_name[parent]
        if index == leaf_index:
            break
        ancestors.append(index)
        parent = random_effects[index].declared_parent
    return frozenset(ancestors)


def find_nested_chain(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    *,
    leaf_index: int,
    excluded: frozenset[int] = frozenset(),
    cache: dict | None = None,
) -> tuple[int, ...]:
    """Return the nested RandomEffect chain above a leaf that eliminates most levels.

    Each other RandomEffect term takes one count test against the leaf
    (nesting is transitive).  Of the totally ordered subsets of the passing
    terms, the rule takes the one holding most of the leaf's declared
    ancestors (``RandomEffect(nested_in=)``; a declared hierarchy is always
    used), then most levels: a heaviest path, by dynamic programming from the
    coarsest term.  Ties go to the chain met first, finest term first.
    Without declarations it reads the level-code pattern alone, as a sparse
    Cholesky's symbolic analysis does.  Pairs of passing terms are tested on
    the leaf's observed levels, not the rows, with the same result.  Returns
    the chain coarsest to finest, ending with ``leaf_index``.
    """
    cache = {} if cache is None else cache
    random_effects = {
        index: matrix
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix)
    }
    leaf = random_effects[leaf_index]
    declared = _declared_ancestors(random_effects, groups, leaf_index)
    candidate_maps = {
        index: cached_nested_parent_codes(random_effects, groups, leaf_index, index, cache)
        for index, matrix in random_effects.items()
        if index != leaf_index and index not in excluded and groups[index].size == matrix.n_levels
    }
    observed = np.zeros(leaf.n_levels, dtype=bool)
    observed[leaf.codes] = True
    maps = {index: codes[observed] for index, codes in candidate_maps.items() if codes is not None}
    order = sorted(
        maps,
        key=lambda index: (
            -np.count_nonzero(np.bincount(maps[index])),
            -random_effects[index].n_levels,
            index,
        ),
    )
    parents = {
        (child, parent): _parent_codes(maps[child], maps[parent], random_effects[child].n_levels)
        for child, parent in combinations(order, 2)
    }
    spans = _group_spans(groups)
    cache.update(
        (("nested_parent", child, parent, spans), codes)
        for (child, parent), codes in parents.items()
    )
    # best[t]: (declared ancestors, levels, chain coarsest to finest) of the
    # heaviest chain ending at t
    best: dict[int, tuple[int, int, tuple[int, ...]]] = {}
    for child in reversed(order):
        n_declared, levels, above = max(
            (best[parent] for parent in order if parents.get((child, parent)) is not None),
            key=lambda entry: entry[:2],
            default=(0, 0, ()),
        )
        best[child] = (
            n_declared + (child in declared),
            levels + random_effects[child].n_levels,
            (*above, child),
        )
    _, _, chain = max(
        (best[index] for index in order), key=lambda entry: entry[:2], default=(0, 0, ())
    )
    return (*chain, leaf_index)


def _zero_penalty_random_effects(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float] | None,
    S_override: NDArray | None,
) -> frozenset[int]:
    """Return the RandomEffect groups whose whole penalty is zero.

    Such a level leaves the chain for the border; the chain is re-grown over
    the others, so its parents compose through the removed level.
    """
    random_effects = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix)
    ]
    if S_override is not None:
        diagonal = np.diag(np.asarray(S_override, dtype=np.float64))
        return frozenset(
            index for index in random_effects if not np.any(diagonal[groups[index].sl] > 0.0)
        )
    lambdas = {
        index: lambda2.get(groups[index].name, 0.0) if isinstance(lambda2, dict) else lambda2
        for index in random_effects
    }
    return frozenset(
        index for index in random_effects if not groups[index].penalized or lambdas[index] == 0.0
    )


def _resolve_nested_chain(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    leaf_index: int,
    *,
    coefficient_width: int,
    lambda2: float | dict[str, float] | None,
    S_override: NDArray | None,
    cache: dict,
) -> tuple[tuple[int, ...], str | None]:
    """Return the chain for a RandomEffect leaf and why a found chain was declined.

    A lone leaf is a chain of one.  The chain never depends on the family,
    the link or the rows, since the factor takes signed working rows.  An
    override that is not diagonal on a longer chain declines it to its leaf.
    """
    excluded = _zero_penalty_random_effects(group_matrices, groups, lambda2, S_override)
    chain_key = ("nested_chain", leaf_index, tuple(sorted(excluded)), _group_spans(groups))
    if chain_key not in cache:
        cache[chain_key] = find_nested_chain(
            group_matrices,
            groups,
            leaf_index=leaf_index,
            excluded=excluded - {leaf_index},
            cache=cache,
        )
    chain = cache[chain_key]
    names = [groups[index].name for index in chain]
    if S_override is not None and len(chain) >= 2:
        chain_indices = np.concatenate([np.arange(groups[g].start, groups[g].end) for g in chain])
        border = np.ones(coefficient_width, dtype=bool)
        border[chain_indices] = False
        incompatibility = _structured_override_incompatibility(
            np.asarray(S_override, dtype=np.float64),
            small_indices=np.flatnonzero(border),
            structured_indices=chain_indices,
            geometry="random_effect",
        )
        if incompatibility is not None:
            return (leaf_index,), f"nested chain {names!r} declined: {incompatibility}"
    return chain, None


def _factor_smooth_component_lambda(
    group_name: str,
    suffix: str,
    lambda2: float | dict[str, float],
) -> float:
    """Resolve one repeated factor-smooth component lambda."""
    from superglm.reml.penalty_algebra import resolve_component_lambda

    return resolve_component_lambda(lambda2, group_name, suffix)


def _first_singular_factor_smooth_block(local_blocks: NDArray) -> int | None:
    """Return the first local block that is not numerically positive definite."""
    symmetric = 0.5 * (local_blocks + local_blocks.transpose(0, 2, 1))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    block_size = local_blocks.shape[1]
    scales = np.max(np.abs(eigenvalues), axis=1)
    thresholds = np.finfo(np.float64).eps * max(block_size, 1) * scales * 10.0
    singular = eigenvalues[:, 0] <= thresholds
    return int(np.flatnonzero(singular)[0]) if np.any(singular) else None


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
    lambda2: float | dict[str, float] | None = None,
    S_override: NDArray | None = None,
    nesting_cache: dict | None = None,
) -> StructuredBackendDecision:
    """Resolve forced/automatic structured use once for a direct fit.

    A RandomEffect leaf grows a nested chain over the other RandomEffect terms
    with a non-zero penalty.  The decision reads the model's terms, sizes and
    penalties, never values, weights, family, link or the sign of the working
    rows.  A chain on which an ``S_override`` is not diagonal is declined to
    its leaf (``nested_fallback_reason``).  Under ``auto`` a term the
    structured solvers cannot take (an override coupling the leaf to the
    border, a zero penalty on the leaf or a FactorSmooth component) is fitted
    on gram with the reason in ``fallback_reason``;
    ``direct_solve="structured"`` raises instead.  ``nesting_cache`` is the
    design's layout cache.
    """
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
    chain, nested_fallback_reason = selection.chain_group_indices, None
    if isinstance(dominant_matrix, RandomEffectGroupMatrix):
        chain, nested_fallback_reason = _resolve_nested_chain(
            group_matrices,
            groups,
            selection.group_index,
            coefficient_width=coefficient_width,
            lambda2=lambda2,
            S_override=S_override,
            cache=shared_nesting_cache(nesting_cache),
        )
    auto_cost_decision = (
        _structured_auto_cost_decision(
            dominant_matrix,
            selection,
            groups,
            coefficient_width,
            chain,
            nested_fallback_reason,
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
        if override_penalty is not None:
            has_dominant_penalty = bool(
                np.any(np.diag(override_penalty[dominant_group.sl, dominant_group.sl]) > 0.0)
            )
        elif isinstance(lambda2, dict):
            has_dominant_penalty = float(lambda2.get(group_name, 0.0)) != 0.0
        else:
            has_dominant_penalty = lambda2 is not None and float(lambda2) != 0.0
        if not has_dominant_penalty:
            return _backend_ineligibility(
                (
                    f"RandomEffect group {group_name!r} has zero penalty and is "
                    "aliased with the fitted intercept"
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
                override_local_penalties
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
            # No data-dependent check: the fs leaf factor certifies each level's
            # pivot block from the rows with the penalty's square root inside.
        elif lambda2 is not None:
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
    if mode == "structured":
        return StructuredBackendDecision(
            use_structured=True,
            group_index=selection.group_index,
            group_name=group_name,
            fallback_reason=None,
            chain_group_indices=chain,
            nested_fallback_reason=nested_fallback_reason,
        )
    if auto_cost_decision is None:  # pragma: no cover - mode invariant
        raise RuntimeError("automatic structured resolution omitted its cost decision")
    # A basis="sz" FactorSmooth takes the balance tree on the same size rule as
    # fs: no route reads its weights, lambdas or data.
    return auto_cost_decision
