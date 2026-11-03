"""Direct penalised IRLS solver, for fits with no L1 or group-lasso penalty.

Each iteration solves ``beta = (X'WX + S)^-1 X'Wz`` as one small dense system,
forming ``X'WX`` from per-group Grams and cross-Grams so the ``(n, p)`` design
is never materialised; a discretised group costs ``O(n_bins K^2)``.

The fit lives in ``_fit``, its recorder in ``_recorder``, its stop rule in
``_convergence``, its line search in ``_search``, and its step methods behind
the interface in ``_step``: QR in ``_qr``, dense in ``_dense``, linearly
constrained (QP) in ``_qp``, structured (random-effect chains, ``fs`` and
``sz``) in ``_structured``.  Finalisers and the constrained active face live
in ``_finalise``; ``_centred`` holds what several steps read of the centred
system, and ``_mean_space`` the binomial/log mean-space overlay.

This module only re-exports.  Patch a name in the submodule that reads it:
patching it here changes nothing the fit sees.
"""

from superglm.solvers.irls_direct._finalise import (
    _active_face_basis,
    _binding_rank,
    _face_decomposition,
    _feature_edf,
)
from superglm.solvers.irls_direct._fit import (
    _build_penalty_matrix,
    _safe_decompose_H,
    _separation_weight_ratio,
    fit_irls_direct,
)
from superglm.solvers.irls_direct._mean_space import _SUBNORMAL_SPACING, _underflow_allowance
from superglm.solvers.irls_direct._qp import (
    _QP_FEASIBILITY_TOL,
    _binding_rows,
    _solve_constrained_qp_with_cold_retry,
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
    "StructuredSolverError",
    "_active_face_basis",
    "_binding_rank",
    "_binding_rows",
    "_build_iterate_factor",
    "_build_penalty_matrix",
    "_face_decomposition",
    "_feature_edf",
    "_levenberg_shifted_leaf_operator",
    "_safe_decompose_H",
    "_separation_weight_ratio",
    "_solve_constrained_qp_with_cold_retry",
    "_structured_solver_errors",
    "_underflow_allowance",
    "fit_irls_direct",
]
