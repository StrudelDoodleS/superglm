"""Deprecated: use superglm.reml instead."""

import warnings

warnings.warn(
    "Importing from superglm.reml_optimizer is deprecated. Use superglm.reml instead.",
    DeprecationWarning,
    stacklevel=2,
)
from superglm.reml import (  # noqa: E402,F401
    compute_d2W_deta2,
    compute_dW_deta,
    optimize_direct_reml,
    optimize_discrete_reml_cached_w,
    optimize_efs_reml,
    reml_direct_gradient,
    reml_direct_hessian,
    reml_laml_objective,
    reml_w_correction,
    run_reml_once,
)
