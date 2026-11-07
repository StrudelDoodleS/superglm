"""A warning on the caller's line, importing nothing a scoring process would otherwise skip."""

from __future__ import annotations

import os
import warnings

# Frames under this directory are superglm's own; a warning skips them to land
# on the caller's line however deep the call (Python 3.12, skip_file_prefixes).
_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


def warn_caller(message: str, category: type[Warning] = UserWarning) -> None:
    """Warn at the first frame outside superglm."""
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_PREFIX,))
