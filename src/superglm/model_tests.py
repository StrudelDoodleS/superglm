"""Backward-compat stub — use ``superglm.stats.model_tests`` instead."""

import warnings as _warnings

_warnings.warn(
    "superglm.model_tests is deprecated, use superglm.stats.model_tests",
    DeprecationWarning,
    stacklevel=2,
)

from superglm.stats.model_tests import (  # noqa: F401, E402
    DispersionTestResult,
    ScoreTestZIResult,
    VuongTestResult,
    ZeroInflationResult,
    dispersion_test,
    score_test_zi,
    vuong_test,
    zero_inflation_index,
)
