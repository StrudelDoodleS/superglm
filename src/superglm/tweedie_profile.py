"""Backward-compat stub — use ``superglm.profiling.tweedie`` instead."""

import warnings as _warnings

_warnings.warn(
    "superglm.tweedie_profile is deprecated, use superglm.profiling.tweedie",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.profiling.tweedie import (  # noqa: F401, E402
    TweedieProfileResult,
    _profile_phi,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    profile_ci_p,
    tweedie_logpdf,
)
