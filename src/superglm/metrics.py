"""Deprecated: use superglm.inference.metrics instead."""

import warnings

warnings.warn(
    "Importing from superglm.metrics is deprecated. Use superglm.inference.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)
from superglm.inference.covariance import (  # noqa: E402,F401
    _penalised_xtwx_inv,
    _penalised_xtwx_inv_gram,
    _second_diff_penalty,
)
from superglm.inference.metrics import *  # noqa: E402,F401,F403
