"""The coefficient solver; its implementation lives in ``superglm.distributional.solver.solver``."""

from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
from superglm.distributional.solver.solver import DenseSolverError, fit_dense_fixed_lambda

__all__ = ["DenseSolverConfig", "DenseSolverError", "DenseSolverResult", "fit_dense_fixed_lambda"]
