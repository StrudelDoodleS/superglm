"""Deprecated: use superglm.reml.multi_penalty instead."""

import warnings

warnings.warn(
    "Importing from superglm.multi_penalty is deprecated. Use superglm.reml.multi_penalty instead.",
    DeprecationWarning,
    stacklevel=2,
)
from superglm.reml.multi_penalty import *  # noqa: E402,F401,F403
