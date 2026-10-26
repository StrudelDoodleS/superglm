"""Direct penalised IRLS solver (no BCD).

Solves the penalised GLM via iteratively reweighted least squares with a
single dense system solve per iteration:

    β = (X'WX + S)⁻¹ X'Wz

where p is ~50-80 (total model columns), making the p×p solve trivially
fast.  Uses gram-based operations (per-group gram + cross_gram) to form
X'WX without materialising the full (n, p) dense matrix.  For discretized
groups this reduces the O(n·p²) bottleneck to O(n_bins·K²) per group.

This replaces BCD when lambda1=0 (no L1/group lasso penalty), which is
the standard REML workflow where smoothing and optional term selection
are handled through the penalty structure. Without BCD, the 33-iteration aliasing from
shared B matrices between select=True subgroups vanishes entirely.

The fit lives in ``_fit`` (``fit_irls_direct`` and its method object), its
recorder in ``_recorder``, and its step methods behind the interface in
``_step``: the QR step in ``_qr``, the structured step (random-effect chains,
``fs`` and ``sz`` terms) in ``_structured``, the shape-constrained (SCOP) step
with its own line search in ``_scop``; the dense and QP routes are still one
step in ``_fit``.  ``_centred`` holds what more than one step reads
of the centred system.  This module re-exports the names other modules import.  A name the
fit reads is patched where the fit reads it, in the submodule: patching it
here would change nothing the fit sees.
"""

from superglm.solvers.irls_direct._fit import (
    _QP_FEASIBILITY_TOL,
    _SUBNORMAL_SPACING,
    _build_penalty_matrix,
    _safe_decompose_H,
    _separation_weight_ratio,
    _solve_constrained_qp_with_cold_retry,
    _underflow_allowance,
    fit_irls_direct,
)
from superglm.solvers.irls_direct._scop import (
    SCOPRunCentring,
    _centred_system_matvec,
    _evaluate_scop_trial,
    _SCOPGroupState,
    _SCOPTrialState,
)
from superglm.solvers.irls_direct._structured import (
    StructuredSolverError,
    _build_iterate_factor,
    _levenberg_shifted_leaf_operator,
    _structured_solver_errors,
)

__all__ = [
    "_QP_FEASIBILITY_TOL",
    "_SUBNORMAL_SPACING",
    "SCOPRunCentring",
    "StructuredSolverError",
    "_build_iterate_factor",
    "_build_penalty_matrix",
    "_centred_system_matvec",
    "_evaluate_scop_trial",
    "_levenberg_shifted_leaf_operator",
    "_safe_decompose_H",
    "_SCOPGroupState",
    "_SCOPTrialState",
    "_separation_weight_ratio",
    "_solve_constrained_qp_with_cold_retry",
    "_structured_solver_errors",
    "_underflow_allowance",
    "fit_irls_direct",
]
