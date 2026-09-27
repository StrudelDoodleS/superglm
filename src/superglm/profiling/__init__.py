"""Distribution parameter profiling (Tweedie p, NB theta).

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.profiling.nb import (
    NBProfileResult,
    NBThetaBoundWarning,
    estimate_nb_theta,
)
from superglm.profiling.tweedie import TweedieProfileResult

__all__ = [
    "NBProfileResult",
    "NBThetaBoundWarning",
    "TweedieProfileResult",
    "estimate_nb_theta",
]
