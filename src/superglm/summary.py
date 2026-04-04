"""Deprecated: use superglm.inference.summary instead."""

import warnings

warnings.warn(
    "Importing from superglm.summary is deprecated. Use superglm.inference.summary instead.",
    DeprecationWarning,
    stacklevel=2,
)
from superglm.inference.summary import (  # noqa: E402,F401
    ModelSummary,
    _BasisDetailRow,
    _CoefRow,
    _compute_coef_stats,
    _sig_stars,
)
