"""Statistical hypothesis tests for fitted models.

# Internal submodules: import siblings directly, not through this __init__.
"""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
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

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.stats.davies": ("psum_chisq", "satterthwaite"),
    "superglm.stats.model_tests": (
        "DispersionTestResult",
        "ScoreTestZIResult",
        "VuongTestResult",
        "ZeroInflationResult",
        "dispersion_test",
        "score_test_zi",
        "vuong_test",
        "zero_inflation_index",
    ),
    "superglm.stats.wood_pvalue": ("wood_test_smooth",),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

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
