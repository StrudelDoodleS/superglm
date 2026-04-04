"""Distribution parameter profiling (Tweedie p, NB theta).

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.profiling.nb import NBProfileResult, estimate_nb_theta, profile_ci_theta
from superglm.profiling.tweedie import (
    TweedieProfileResult,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    profile_ci_p,
    tweedie_logpdf,
)

__all__ = [
    "NBProfileResult",
    "TweedieProfileResult",
    "estimate_nb_theta",
    "estimate_phi",
    "estimate_tweedie_p",
    "generate_tweedie_cpg",
    "profile_ci_p",
    "profile_ci_theta",
    "tweedie_logpdf",
]
