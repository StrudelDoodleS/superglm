"""Statistical hypothesis tests for fitted models.

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.stats.davies import psum_chisq, satterthwaite
from superglm.stats.model_tests import (
    DispersionTestResult,
    ScoreTestZIResult,
    VuongTestResult,
    ZeroInflationResult,
    dispersion_test,
    score_test_zi,
    vuong_test,
    zero_inflation_index,
)
from superglm.stats.wood_pvalue import wood_test_smooth

__all__ = [
    "DispersionTestResult",
    "ScoreTestZIResult",
    "VuongTestResult",
    "ZeroInflationResult",
    "dispersion_test",
    "psum_chisq",
    "satterthwaite",
    "score_test_zi",
    "vuong_test",
    "wood_test_smooth",
    "zero_inflation_index",
]
