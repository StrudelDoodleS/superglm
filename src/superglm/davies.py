"""Backward-compat stub — use ``superglm.stats.davies`` instead."""

import warnings as _warnings

_warnings.warn(
    "superglm.davies is deprecated, use superglm.stats.davies",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.stats.davies import psum_chisq, satterthwaite  # noqa: F401, E402
