"""Backward-compat stub: discretize moved to superglm.diagnostics.discretize."""

import warnings as _w

_w.warn(
    "superglm.discretize is deprecated — use superglm.diagnostics.discretize",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.diagnostics.discretize import (  # noqa: E402, F401
    DiscretizationResult,
    discretization_impact,
)
