"""Private group-matrix helpers package."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from ._cross_matrix_execution import CrossGroupPair, CrossMatrixExecutionPlan

# Each name is read from its module on first access (``superglm._lazy``).
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm._group_matrix._cross_matrix_execution": (
        "CrossGroupPair",
        "CrossMatrixExecutionPlan",
    ),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = ["CrossGroupPair", "CrossMatrixExecutionPlan"]
