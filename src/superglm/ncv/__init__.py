"""Neighbourhood cross-validation: smoothing chosen by held-out prediction.

The package's names, in the order a fit uses them: the neighbourhoods of the rows being fitted
(:func:`build_neighbourhoods`, or one of the builders, and the checks on them), the engine's
problem for a built design (:func:`build_problem`, its inner fits certified to
:func:`ncv_mode_bar` of the search's tolerance), the search over the smoothing parameters
(:func:`search_smoothing`, which returns a :class:`SearchOutcome` holding an
:class:`NCVResult`), and the criterion with its held-out pairs at the fit that is published
(:func:`evaluate_fit` on a :class:`ModeFit`).  ``superglm.ncv.search`` is the module;
the function it defines is exported as ``search_smoothing``.

Internal submodules import their siblings directly, not through this package.
"""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.ncv.engine import (  # noqa: F401
        DEFAULT_NCV_TOL,
        Evaluation,
        HeldOutPairs,
        LeaveOutRankError,
        ModeFit,
        NCVProblem,
        WorkCounts,
        build_problem,
        evaluate,
        evaluate_fit,
        ncv_mode_bar,
    )
    from superglm.ncv.neighbourhoods import (  # noqa: F401
        FEW_NEIGHBOURHOODS,
        LevelConflict,
        Neighbourhoods,
        build_neighbourhoods,
        check_leaves_weighted_rows,
        check_scores_weighted_rows,
        from_groups,
        from_neighbours,
        from_splitter,
        single_rows,
        unidentified_levels,
        warn_few_neighbourhoods,
    )
    from superglm.ncv.result import (  # noqa: F401
        STOP_REASONS,
        TRIAL_FAILURES,
        NCVResult,
    )
    from superglm.ncv.search import (  # noqa: F401
        DEFAULT_MAX_NCV_ITER,
        SearchOutcome,
        search_smoothing,
    )

# Each name is read from its module on first access (``superglm._lazy``), so importing the
# package, or one of its modules, loads only what that module imports.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.ncv.engine": (
        "DEFAULT_NCV_TOL",
        "Evaluation",
        "HeldOutPairs",
        "LeaveOutRankError",
        "ModeFit",
        "NCVProblem",
        "WorkCounts",
        "build_problem",
        "evaluate",
        "evaluate_fit",
        "ncv_mode_bar",
    ),
    "superglm.ncv.neighbourhoods": (
        "FEW_NEIGHBOURHOODS",
        "LevelConflict",
        "Neighbourhoods",
        "build_neighbourhoods",
        "check_leaves_weighted_rows",
        "check_scores_weighted_rows",
        "from_groups",
        "from_neighbours",
        "from_splitter",
        "single_rows",
        "unidentified_levels",
        "warn_few_neighbourhoods",
    ),
    "superglm.ncv.result": ("STOP_REASONS", "TRIAL_FAILURES", "NCVResult"),
    "superglm.ncv.search": ("DEFAULT_MAX_NCV_ITER", "SearchOutcome", "search_smoothing"),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = sorted(name for names in _LAZY_EXPORTS.values() for name in names)
