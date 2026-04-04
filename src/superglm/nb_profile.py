"""Backward-compat stub — use ``superglm.profiling.nb`` instead."""

import warnings as _warnings

_warnings.warn(
    "superglm.nb_profile is deprecated, use superglm.profiling.nb",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.profiling.nb import (  # noqa: F401, E402
    NBProfileResult,
    estimate_nb_theta,
    profile_ci_theta,
)
