"""Backward-compat stub — use ``superglm.stats.wood_pvalue`` instead."""

import warnings as _warnings

_warnings.warn(
    "superglm.wood_pvalue is deprecated, use superglm.stats.wood_pvalue",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.stats.wood_pvalue import (  # noqa: F401, E402
    _mixture_pvalue,
    wood_test_smooth,
)
