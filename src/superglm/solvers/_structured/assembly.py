"""Penalty assembly and cached solves for structured systems."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    GroupMatrix,
)
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
    SumToZeroLeafSystem,
    SumToZeroPenalizedOperator,
    SumToZeroTreeFactor,
)
from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafFactor,
    FactorSmoothLeafSystem,
    FactorSmoothPenalizedOperator,
    ProfiledFactorSmoothLeafFactor,
)
from superglm.solvers._structured.moments import NestedStructuredSystem
from superglm.solvers._structured.nested import (
    NestedPenalizedOperator,
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
)
from superglm.solvers._structured.overrides import (
    _factor_smooth_override_local_blocks,
    _structured_override_incompatibility,
)
from superglm.solvers.hessian_factor import _component_indices
from superglm.types import GroupSlice, PenaltyComponent


@dataclass(frozen=True)
class CachedBlockStructuredSolution:
    """One lambda-only solve against a cached ``fs`` leaf system."""

    beta: NDArray
    intercept: float
    factor: ProfiledFactorSmoothLeafFactor
    penalized_operator: FactorSmoothPenalizedOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int
    # with ``centred``, the solve's state (one-engine design §3.8): ``alpha``
    # about the factor's border centre ``c0``, written into slope coordinates
    # (0 off the border); ``intercept = alpha - fsum(c * beta)``
    centred_intercept: float | None = None
    state_center: NDArray | None = None


@dataclass(frozen=True)
class CachedNestedStructuredSolution:
    """One lambda-only solve against cached nested-chain working moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledNestedSchurFactor
    penalized_operator: NestedPenalizedOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int
    # with ``centred``, the solve's state (one-engine design §3.8): ``alpha``
    # about the factor's border centre ``c0``, written into slope coordinates
    # (0 off the border); ``intercept = alpha - fsum(c * beta)``
    centred_intercept: float | None = None
    state_center: NDArray | None = None


@dataclass(frozen=True)
class CachedSumToZeroStructuredSolution:
    """One lambda-only solve against cached constrained SZ moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledSumToZeroTreeFactor
    penalized_operator: SumToZeroPenalizedOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int
    # with ``centred``, the solve's state (one-engine design §3.8): ``alpha``
    # about the factor's border centre ``c0``, written into slope coordinates
    # (0 off the border); ``intercept = alpha - fsum(c * beta)``
    centred_intercept: float | None = None
    state_center: NDArray | None = None


_PENALTIES_REQUIRED = (
    "A structured penalized operator needs the fit's compact penalty components "
    "(reml_penalties) or an authoritative S_override."
)


def _lambda_for_component(
    lambda2: float | dict[str, float],
    name: str,
) -> float:
    return float(lambda2[name]) if isinstance(lambda2, dict) else float(lambda2)


def _dense_component_omega(
    component: PenaltyComponent,
    group_matrix: GroupMatrix,
) -> NDArray:
    if component.omega_ssp is not None:
        return np.asarray(component.omega_ssp, dtype=np.float64)
    if component.omega_raw is None or not hasattr(group_matrix, "R_inv"):
        raise ValueError(f"Dense penalty component {component.name!r} has no solver-space matrix.")
    return np.asarray(
        group_matrix.R_inv.T @ component.omega_raw @ group_matrix.R_inv,
        dtype=np.float64,
    )


def build_penalized_block_operator(
    system: FactorSmoothLeafSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> FactorSmoothPenalizedOperator:
    """Add compact penalties to an ``fs`` system's moment operator, keeping them apart.

    ``A`` and ``D`` below are the penalty parts alone; the returned operator
    carries them as ``penalty_small`` and ``penalty_local`` (the square roots
    the factor takes) beside the penalized moments every trace reads.
    """
    operator = system.operator
    p = operator.shape[0]
    A = np.zeros_like(operator.A)
    D = np.zeros_like(operator.D)
    small_position = np.full(p, -1, dtype=np.intp)
    small_position[operator.small_indices] = np.arange(len(operator.small_indices))
    structured_position = np.full(p, -1, dtype=np.intp)
    structured_position[operator.structured_indices.ravel()] = np.arange(
        operator.n_levels * operator.block_size
    )

    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="factor_smooth",
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        D += _factor_smooth_override_local_blocks(
            penalty,
            operator.structured_indices,
            sum_to_zero=False,
        )
        return _penalized_leaf_operator(operator, A, D)

    if reml_penalties is None:
        raise ValueError(_PENALTIES_REQUIRED)
    for component in reml_penalties:
        lam = _lambda_for_component(lambda2, component.name)
        if lam == 0.0:
            continue
        indices = _component_indices(component, p)
        local_small = small_position[indices]
        local_structured = structured_position[indices]
        wholly_small = np.all(local_small >= 0)
        wholly_structured = np.all(local_structured >= 0)
        if not wholly_small and not wholly_structured:
            raise ValueError(f"Penalty component {component.name!r} crosses structured partitions.")
        if component.penalty_kind == "identity":
            if wholly_small:
                A[local_small, local_small] += lam
            else:
                levels = local_structured // operator.block_size
                coordinates = local_structured % operator.block_size
                D[levels, coordinates, coordinates] += lam
            continue
        if component.penalty_kind == "repeated":
            if not wholly_structured:
                raise ValueError(
                    f"Repeated penalty component {component.name!r} must lie in "
                    "the dominant factor-smooth block."
                )
            if (
                component.repeat_count != operator.n_levels
                or component.block_width != operator.block_size
                or not np.array_equal(
                    indices.reshape(operator.n_levels, operator.block_size),
                    operator.structured_indices,
                )
            ):
                raise ValueError(
                    f"Repeated penalty component {component.name!r} does not match "
                    "the dominant factor-smooth geometry."
                )
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (operator.block_size, operator.block_size):
                raise ValueError(
                    f"Repeated penalty component {component.name!r} has shape "
                    f"{omega.shape}; expected "
                    f"({operator.block_size}, {operator.block_size})."
                )
            D += lam * omega[None, :, :]
            continue

        omega = _dense_component_omega(
            component,
            group_matrices[component.group_index],
        )
        if omega.shape != (len(indices), len(indices)):
            raise ValueError(
                f"Penalty component {component.name!r} has shape {omega.shape}; "
                f"expected ({len(indices)}, {len(indices)})."
            )
        if not wholly_small:
            raise ValueError(
                f"Dense penalty component {component.name!r} cannot span the "
                "dominant factor-smooth block."
            )
        A[np.ix_(local_small, local_small)] += lam * omega

    return _penalized_leaf_operator(operator, A, D)


def _penalized_leaf_operator(
    operator: BlockSymmetricOperator, A: NDArray, D: NDArray
) -> FactorSmoothPenalizedOperator:
    """The moments plus the penalty parts ``A``, ``D``, with the parts kept for the factor."""
    return FactorSmoothPenalizedOperator.with_penalties(operator, A, D)


def build_penalized_sum_to_zero_operator(
    system: SumToZeroLeafSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> SumToZeroPenalizedOperator:
    """Add public penalties to their all-level ``sz`` geometry, keeping the penalty parts apart.

    ``A`` and ``D`` below are the penalty parts alone (``D`` one block per
    level, all ``K``): the returned operator carries them as
    ``penalty_small``/``penalty_local``, the square roots the balance tree
    takes, beside the penalized moments every trace reads.
    """
    operator = system.operator
    p = operator.shape[0]
    A = np.zeros_like(operator.A)
    D = np.zeros_like(operator.D)
    small_position = np.full(p, -1, dtype=np.intp)
    small_position[operator.small_indices] = np.arange(len(operator.small_indices))
    structured_position = np.full(p, -1, dtype=np.intp)
    structured_position[operator.structured_indices.ravel()] = np.arange(
        (operator.n_levels - 1) * operator.block_size
    )

    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="sum_to_zero",
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        D += _factor_smooth_override_local_blocks(
            penalty,
            operator.structured_indices,
            sum_to_zero=True,
        )
        return SumToZeroPenalizedOperator.with_penalties(operator, A, D)

    if reml_penalties is None:
        raise ValueError(_PENALTIES_REQUIRED)
    for component in reml_penalties:
        lam = _lambda_for_component(lambda2, component.name)
        if lam == 0.0:
            continue
        indices = _component_indices(component, p)
        local_small = small_position[indices]
        local_structured = structured_position[indices]
        wholly_small = np.all(local_small >= 0)
        wholly_structured = np.all(local_structured >= 0)
        if not wholly_small and not wholly_structured:
            raise ValueError(f"Penalty component {component.name!r} crosses structured partitions.")
        if component.penalty_kind == "identity":
            if not wholly_small:
                raise ValueError("The dominant SZ block accepts only a sum-to-zero penalty.")
            A[local_small, local_small] += lam
            continue
        if component.penalty_kind == "sum_to_zero":
            if (
                not wholly_structured
                or component.repeat_count != operator.n_levels
                or component.block_width != operator.block_size
                or not np.array_equal(
                    indices.reshape(operator.n_levels - 1, operator.block_size),
                    operator.structured_indices,
                )
            ):
                raise ValueError(
                    f"Sum-to-zero penalty component {component.name!r} does not "
                    "match the dominant SZ geometry."
                )
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (operator.block_size, operator.block_size):
                raise ValueError(
                    f"Sum-to-zero penalty component {component.name!r} has the wrong local shape."
                )
            D += lam * omega[None, :, :]
            continue
        if wholly_structured:
            raise ValueError("The dominant SZ block accepts only penalty_kind='sum_to_zero'.")
        omega = _dense_component_omega(
            component,
            group_matrices[component.group_index],
        )
        if omega.shape != (len(indices), len(indices)):
            raise ValueError(
                f"Penalty component {component.name!r} has shape {omega.shape}; "
                f"expected ({len(indices)}, {len(indices)})."
            )
        A[np.ix_(local_small, local_small)] += lam * omega

    return SumToZeroPenalizedOperator.with_penalties(operator, A, D)


def _nested_penalty_terms(
    group_matrices: list[GroupMatrix],
    lambda2: float | dict[str, float],
    reml_penalties: list[PenaltyComponent],
    p: int,
):
    """Yield ``(name, indices, scale, omega)`` for each compact penalty component.

    ``omega=None`` is an identity; a component whose lambda is zero is skipped.
    """
    for component in reml_penalties:
        lam = _lambda_for_component(lambda2, component.name)
        if lam == 0.0:
            continue
        omega = (
            None
            if component.penalty_kind == "identity"
            else _dense_component_omega(component, group_matrices[component.group_index])
        )
        yield component.name, _component_indices(component, p), lam, omega


def build_penalized_nested_operator(
    system: NestedStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> NestedPenalizedOperator:
    """Assemble per-node ridges and the border penalty ``S_b`` of a nested system.

    The whole chain is the dominant diagonal block: a chain term must be
    diagonal and lands on the node penalties, a border term lands in ``S_b``,
    and a term that straddles the two raises ``ValueError``.  An authoritative
    ``S_override`` must be diagonal on the chain with no chain-to-border mass
    (§3.7); its diagonal gives the per-node ``lambda_u``.  ``S_b`` is kept
    apart from ``A``: the factor adds it to ``Q`` as its own PSD term (§3.4).
    """
    operator = system.operator
    p, q = operator.shape[0], len(operator.small_indices)
    border_penalty = np.zeros((q, q), dtype=np.float64)
    node_penalty = np.zeros(operator.tree.n_nodes, dtype=np.float64)
    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="random_effect",
        )
        if incompatibility is not None:
            raise ValueError(f"Nested chain penalty: {incompatibility}")
        border_penalty += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        node_penalty += np.diag(penalty)[operator.structured_indices]
    else:
        if reml_penalties is None:
            raise ValueError(_PENALTIES_REQUIRED)
        small_position = np.full(p, -1, dtype=np.intp)
        small_position[operator.small_indices] = np.arange(q)
        node_position = np.full(p, -1, dtype=np.intp)
        node_position[operator.structured_indices] = np.arange(operator.tree.n_nodes)
        for name, indices, scale, omega in _nested_penalty_terms(
            group_matrices, lambda2, reml_penalties, p
        ):
            local_small, local_node = small_position[indices], node_position[indices]
            if np.all(local_small >= 0):
                if omega is None:
                    border_penalty[local_small, local_small] += scale
                else:
                    border_penalty[np.ix_(local_small, local_small)] += scale * omega
            elif not np.all(local_node >= 0):
                raise ValueError(
                    f"Penalty component {name!r} crosses the nested chain and the border."
                )
            elif omega is not None:
                raise ValueError(f"Nested chain penalty component {name!r} is not an identity.")
            else:
                node_penalty[local_node] += scale
    return NestedPenalizedOperator(
        data=operator,
        node_penalty=operator.tree.split(node_penalty),
        border_penalty=0.5 * (border_penalty + border_penalty.T),
    )


def build_penalized_structured_operator(
    system: FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator | NestedPenalizedOperator:
    """Dispatch compact penalty assembly by structured-system geometry."""
    if isinstance(system, NestedStructuredSystem):
        return build_penalized_nested_operator(
            system,
            group_matrices,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
    if isinstance(system, SumToZeroLeafSystem):
        return build_penalized_sum_to_zero_operator(
            system,
            group_matrices,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
    if isinstance(system, FactorSmoothLeafSystem):
        return build_penalized_block_operator(
            system,
            group_matrices,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
    raise TypeError(f"Unsupported structured system {type(system).__name__}.")


def build_augmented_block_factor(
    system: FactorSmoothLeafSystem,
    penalized_operator: FactorSmoothPenalizedOperator,
) -> tuple[FactorSmoothLeafFactor, NDArray]:
    """Factor an ``fs`` leaf system with the intercept as its super-root; return it and the raw RHS.

    The factor solves the normal equations from the right-hand side inside its
    leaf factorization (``FactorSmoothLeafFactor.solve_data``); the raw RHS is
    returned for callers that add a further right-hand side.  The last factor
    built on the system is reused for bitwise the same penalty parts (perf
    F1); the memo holds it weakly, so a system and its factor form no
    reference cycle (perf F17).
    """
    operator = system.operator
    if not np.array_equal(penalized_operator.small_indices, operator.small_indices):
        raise ValueError("Penalized and unpenalized operators must use identical partitions.")
    factor = None
    if penalized_operator.penalty_small is None or penalized_operator.penalty_local is None:
        raise ValueError("An fs factor needs the penalized operator's penalty parts.")
    for small, local, held in system.factor_memo:
        if np.array_equal(small, penalized_operator.penalty_small) and np.array_equal(
            local, penalized_operator.penalty_local
        ):
            factor = held()
    if factor is None:
        factor = FactorSmoothLeafFactor(system, penalized_operator)
        system.factor_memo[:] = [
            (
                np.array(penalized_operator.penalty_small, copy=True),
                np.array(penalized_operator.penalty_local, copy=True),
                weakref.ref(factor),
            )
        ]
    p = operator.shape[0]
    rhs = np.empty(p + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def build_augmented_sum_to_zero_factor(
    system: SumToZeroLeafSystem,
    penalized_operator: SumToZeroPenalizedOperator,
) -> tuple[SumToZeroTreeFactor, NDArray]:
    """Factor an ``sz`` leaf system on the balance tree, the intercept its super-root.

    The factor solves the normal equations from the right-hand side inside its
    leaf factorization (``SumToZeroTreeFactor.solve_data``); the raw RHS is
    returned for callers that add a further right-hand side.  The last factor
    built on the system is reused for bitwise the same penalty parts (as the
    fs factor, perf F1); the memo holds it weakly, so a system and its factor
    form no reference cycle (perf F17).
    """
    operator = system.operator
    if not np.array_equal(penalized_operator.small_indices, operator.small_indices):
        raise ValueError("Penalized and unpenalized operators must use identical partitions.")
    if penalized_operator.penalty_small is None or penalized_operator.penalty_local is None:
        raise ValueError("An sz factor needs the penalized operator's penalty parts.")
    factor = None
    for small, local, held in system.factor_memo:
        if np.array_equal(small, penalized_operator.penalty_small) and np.array_equal(
            local, penalized_operator.penalty_local
        ):
            factor = held()
    if factor is None:
        factor = SumToZeroTreeFactor(system, penalized_operator)
        system.factor_memo[:] = [
            (
                np.array(penalized_operator.penalty_small, copy=True),
                np.array(penalized_operator.penalty_local, copy=True),
                weakref.ref(factor),
            )
        ]
    p = operator.shape[0]
    rhs = np.empty(p + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def build_augmented_nested_factor(
    system: NestedStructuredSystem,
    penalized: NestedPenalizedOperator,
) -> tuple[NestedSchurFactor, NDArray]:
    """Add the unpenalized intercept to a nested system; return its factor and RHS.

    The augmentation is exact (``NestedPenalizedOperator.augmented``): the
    intercept becomes border column 0 with leaf mean 1 and a zero penalty.
    """
    if penalized.data is not system.operator:
        raise ValueError("The nested penalized operator must wrap the system's data operator.")
    operator = system.operator
    factor = NestedSchurFactor(
        penalized.augmented(),
        chain_group_names=system.chain_group_names,
        chain_group_indices=system.chain_group_indices,
        intercept=True,
    )
    rhs = np.empty(operator.shape[0] + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def solve_augmented_normal_equations(
    system, factor, rhs: NDArray, *, centred: bool = False, extra: NDArray | None = None
) -> NDArray:
    """Solve the augmented normal equations ``H [b_0; b] = [1 X]' W z`` of a structured system.

    A nested factor takes its data-side solve with the border right-hand side
    in its centred coordinates (``NestedSchurFactor.solve_data``,
    ``NestedStructuredSystem.xtwz_small_centred``), and the fs and sz leaf
    factors theirs from the right-hand side inside the leaf factorization;
    with ``centred`` the first entry is then the centred intercept ``alpha``
    about the factor's border centre instead of the raw one.  Any other
    factor takes its ``solve`` (``centred`` does not apply to it).  ``extra``
    ``(p + 1,)``, zero in the intercept entry, is a further right-hand side
    (a Levenberg shift's ``E beta``, design §3.11): it is not data, so it
    takes the factor's full solve beside the data-side one, in the same
    intercept coordinate.
    """
    if (
        isinstance(factor, FactorSmoothLeafFactor) and isinstance(system, FactorSmoothLeafSystem)
    ) or (isinstance(factor, SumToZeroTreeFactor) and isinstance(system, SumToZeroLeafSystem)):
        solution = factor.solve_data(centred=centred)
        if extra is not None:
            # both solves in the same intercept coordinate: with ``centred``
            # the data-side entry 0 is ``alpha``, so the extra one must be too
            solution = solution + factor.solve(extra, centred=centred)
        return solution
    if isinstance(factor, NestedSchurFactor) and isinstance(system, NestedStructuredSystem):
        border_rhs = system.xtwz_small_centred
        border = None if border_rhs is None else np.concatenate(([system.sum_wz], border_rhs))
        solution = factor.solve_data(rhs, border_centred=border, centred=centred)
        if extra is not None:
            solution = solution + factor.solve(extra, centred=centred)
        return solution
    if extra is not None:
        rhs = rhs + extra
    if centred:
        raise ValueError("Only a nested factor solves in centred coordinates.")
    return factor.solve(rhs)


def _cached_centred_solution(
    system, factor, rhs: NDArray, centred: bool
) -> tuple[NDArray, float, float | None, NDArray | None]:
    """``(beta, intercept, alpha, c)`` of a cached lambda trial in the PIRLS state's coordinates.

    The data-side solve keeps its intercept as the centred ``alpha`` about the
    factor's border centre ``c0`` (``solve_augmented_normal_equations`` with
    ``centred``), as a PIRLS iterate holds it, and the raw intercept is read
    from it as PIRLS reads it, ``alpha - fsum(c * beta)``.  A trial that took
    ``alpha`` back from the raw intercept instead, ``intercept + c' beta``,
    cancelled ``c' beta`` at a column's offset: at 1e16 discrete REML ran to
    ``max_reml_iter`` with lambda 2.84% off.  ``c`` is ``(p,)``, zero off the
    border; a border without a centred column gives ``alpha`` bit for bit as
    the raw solve's intercept.  Without ``centred`` the raw solve, as
    before, and no centred state (``alpha`` and ``c`` are ``None``).
    """
    if not centred:
        coefficients = solve_augmented_normal_equations(system, factor, rhs)
        return coefficients[1:], float(coefficients[0]), None, None
    coefficients = solve_augmented_normal_equations(system, factor, rhs, centred=True)
    beta = coefficients[1:]
    leaf = system.operator.leaf if isinstance(system, NestedStructuredSystem) else system.leaf
    centre = np.zeros(len(beta), dtype=np.float64)
    centre[system.operator.small_indices] = leaf.center
    alpha = float(coefficients[0])
    return beta, alpha - math.fsum(centre * beta), alpha, centre


def build_augmented_structured_factor(
    system: FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem,
    penalized_operator: (
        FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator | NestedPenalizedOperator
    ),
):
    """Dispatch intercept augmentation and Schur factorization."""
    if isinstance(system, NestedStructuredSystem):
        if not isinstance(penalized_operator, NestedPenalizedOperator):
            raise TypeError("Nested structured systems require a nested penalized operator.")
        return build_augmented_nested_factor(system, penalized_operator)
    if isinstance(system, SumToZeroLeafSystem):
        if not isinstance(penalized_operator, SumToZeroPenalizedOperator):
            raise TypeError("sz leaf systems require the sz penalized operator.")
        return build_augmented_sum_to_zero_factor(system, penalized_operator)
    if isinstance(system, FactorSmoothLeafSystem):
        if not isinstance(penalized_operator, FactorSmoothPenalizedOperator):
            raise TypeError("fs leaf systems require the fs penalized operator.")
        return build_augmented_block_factor(system, penalized_operator)
    raise TypeError(f"Unsupported structured system {type(system).__name__}.")


def solve_cached_block_structured(
    system: FactorSmoothLeafSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    centred: bool = False,
) -> CachedBlockStructuredSolution:
    """Solve a factor-smooth lambda trial from a cached ``fs`` leaf system (no data pass)."""
    penalized = build_penalized_block_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_block_factor(system, penalized)
    beta, intercept, alpha, centre = _cached_centred_solution(
        system, augmented_factor, rhs, centred
    )
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledFactorSmoothLeafFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
    )
    return CachedBlockStructuredSolution(
        beta=beta,
        intercept=intercept,
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
        centred_intercept=alpha,
        state_center=centre,
    )


def solve_cached_sum_to_zero_structured(
    system: SumToZeroLeafSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    centred: bool = False,
) -> CachedSumToZeroStructuredSolution:
    """Solve an ``sz`` lambda trial from a cached leaf system on the balance tree (no data pass)."""
    penalized = build_penalized_sum_to_zero_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_sum_to_zero_factor(system, penalized)
    beta, intercept, alpha, centre = _cached_centred_solution(
        system, augmented_factor, rhs, centred
    )
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledSumToZeroTreeFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
    )
    return CachedSumToZeroStructuredSolution(
        beta=beta,
        intercept=intercept,
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
        centred_intercept=alpha,
        state_center=centre,
    )


def solve_cached_nested_structured(
    system: NestedStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    centred: bool = False,
) -> CachedNestedStructuredSolution:
    """Solve a lambda trial from cached nested working moments (no data pass)."""
    penalized = build_penalized_nested_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_nested_factor(system, penalized)
    beta, intercept, alpha, centre = _cached_centred_solution(
        system, augmented_factor, rhs, centred
    )
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledNestedSchurFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
        data_operator=system.operator,
    )
    return CachedNestedStructuredSolution(
        beta=beta,
        intercept=intercept,
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
        centred_intercept=alpha,
        state_center=centre,
    )


def solve_cached_structured(
    system: FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    centred: bool = False,
) -> (
    CachedBlockStructuredSolution
    | CachedSumToZeroStructuredSolution
    | CachedNestedStructuredSolution
):
    """Dispatch a cached lambda-only solve by dominant structured geometry.

    With ``centred`` the solution also carries the factor's centred state
    (``centred_intercept`` about ``state_center``, ``_cached_centred_solution``).
    """
    if isinstance(system, NestedStructuredSystem):
        return solve_cached_nested_structured(
            system,
            group_matrices,
            groups,
            lambdas,
            reml_penalties=reml_penalties,
            centred=centred,
        )
    if isinstance(system, SumToZeroLeafSystem):
        return solve_cached_sum_to_zero_structured(
            system,
            group_matrices,
            groups,
            lambdas,
            reml_penalties=reml_penalties,
            centred=centred,
        )
    if isinstance(system, FactorSmoothLeafSystem):
        return solve_cached_block_structured(
            system,
            group_matrices,
            groups,
            lambdas,
            reml_penalties=reml_penalties,
            centred=centred,
        )
    raise TypeError(f"Unsupported structured system {type(system).__name__}.")
