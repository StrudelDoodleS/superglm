"""Block Gram assembly from explicit group matrices: a test fixture of the Gram-invariant tests.

Each entry point builds a ``MatrixExecutionPlan`` over the given blocks, checks the
solver-column spans, and reads one moment bundle from it, as ``DesignMatrix.execution_plan``
does for the engine.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import GroupMatrix


def _execution_plan_for_blocks(gms, groups, W: NDArray, tabmat_split):
    """Build the plan and verify the solver-column spans."""
    plan = MatrixExecutionPlan(
        gms,
        n=len(W),
        ordinary_tabmat=tabmat_split is not None,
        prepared_ordinary_split=tabmat_split,
    )
    plan.validate_group_spans(groups)
    return plan


def _block_xtwx(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """A weighted design Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=False, profile=profile).gram


def _block_xtwx_rhs(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    Wz: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """A Gram, ``X'W``, and ``X'Wz``."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    moments = plan.moments(
        W,
        rhs=(Wz,),
        include_xtw=True,
        signed=False,
        profile=profile,
    )
    assert moments.xtw is not None
    return moments.gram, moments.xtw, moments.xt_rhs[0]


def _block_xtwx_signed(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """An arbitrary-sign weighted Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=True, profile=profile).gram
