"""The coefficient solver; its implementation lives in ``superglm.distributional.solver.solver``."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
    from superglm.distributional.solver.solver import DenseSolverError, fit_dense_fixed_lambda

# Each name is read from its module on first access (``superglm._lazy``), so importing the
# package, or one of its modules, loads only what that module imports.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.distributional.results.solver": ("DenseSolverConfig", "DenseSolverResult"),
    "superglm.distributional.solver.solver": ("DenseSolverError", "fit_dense_fixed_lambda"),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = ["DenseSolverConfig", "DenseSolverError", "DenseSolverResult", "fit_dense_fixed_lambda"]
