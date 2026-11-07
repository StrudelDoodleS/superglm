"""Four small canonical fits make a counted number of passes over the rows and repeat only the passes this file names.

The work ledger flags a call whose input an earlier call of the same operation had bit for bit.
This file fails on what does not depend on the floating-point trajectory of a fit, so that no BLAS
build's rounding changes the verdict:

* a pass over the rows (the ledger's ``PASS`` targets: Gram builds, likelihood evaluations, design
  products) that repeats outside ``ROW_PASS_REPEATS``, or whose Gram repeats exceed a fraction of the
  centred systems built (a ceiling, so a count below it passes), or that the ledger cannot identify;
* the number of passes: once-per-fit stages, one row pass per centred system built, and two moment
  passes per predictor however many smoothing iterations the fit takes.

A call is a repeat when its arrays are bit for bit those of an earlier call and its objects the same
objects; the ledger leaves out the arguments that only carry state or output (``carriers`` of each
target in ``TARGETS``), so a call handed a new carrier at every solve still counts.  Each entry of
``ROW_PASS_REPEATS`` must be repeated by some fit, so the entry of a repeat that is removed fails
the file until it is deleted.

Every other repeat, such as a penalty decomposed again or a rank decision on a Gram matrix, takes an
input that is itself the outcome of an iteration, so whether it appears can change with the kernel.
Those are reported in the warnings summary of the run and fail nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, GaussianLS, PSpline, SuperGLM, SuperLSS, cat, s
from superglm.diagnostics.work_ledger import PASS, TARGETS, work_ledger
from superglm.features.constraint import Constraint

PENALTY = (
    "the penalty matrices are fixed for the fit, yet each smoothing-parameter evaluation "
    "decomposes them again; the cost is the block size cubed, not the row count"
)
ITERATE = (
    "the centred system, a score or a linear predictor is rebuilt at weights or coefficients an "
    "earlier evaluation already visited (an accepted iterate re-entered by the next evaluation, "
    "or the final one by the diagnostics); for the Gram pass this is repeated work over every row"
)
RANK = "a rank decision on a Gram matrix or factor an earlier decision already saw"
BLAS = "equal to an earlier input only under some BLAS kernels' rounding"

ROW_PASSES = frozenset(target.kind for target in TARGETS if target.role == PASS)

#: ``(operation, caller prefix)`` -> reason, for the passes over the rows that repeat today.  Each
#: entry must still match a repeated row in one of the four fits, so a repeat that is fixed takes its
#: entry with it.
ROW_PASS_REPEATS = {
    ("centered_gram_rhs", "solvers.centered_system:build_centered_system"): ITERATE,
    ("packed_centered_gram_rhs", "solvers.centered_system:_raw_rung_system"): ITERATE,
    ("anchor_support_centered_gram_rhs", "solvers.centered_system:_raw_rung_system"): ITERATE,
    ("DesignMatrix.rmatvec", "solvers.centered_system:build_centered_system"): ITERATE,
    ("centred_matvec", "solvers.irls_direct._fit:evaluate_state"): ITERATE,
    ("centred_matvec", "solvers.mode_score:linear_predictor"): ITERATE,
    ("DesignMatrix.rmatvec", "solvers.mode_score:"): ITERATE,
    ("DesignMatrix.rmatvec", "reml.identified:"): ITERATE,
}

#: The other repeats the fits show, to label the report; ``"*"`` matches every operation.
OTHER_REPEATS = {
    ("*", "reml.penalty_algebra:"): PENALTY,
    ("*", "dm_builder:_whitening_map"): PENALTY,
    ("numpy.linalg.eigh", "solvers.centered_system:_equilibrated_block_root"): PENALTY,
    ("numpy.linalg.eigvalsh", "reml.efs_update:"): PENALTY,
    ("numpy.linalg.eigvalsh", "features._spline_penalties:_interval_blocks"): PENALTY,
    ("build_centered_system", "solvers.irls_direct._fit:get_centered_system"): ITERATE,
    ("numpy.linalg.cholesky", "solvers.centered_system:_attach_centered_penalty"): ITERATE,
    ("*", "solvers.rank:_decompose_gram"): RANK,
    ("decompose_gram", "distributional.solver.solver:_initial_coefficients"): RANK,
    ("numpy.linalg.svd", "solvers.rank:_selection_amplification"): BLAS,
    ("numpy.linalg.lstsq", "reml.result:_map_beta_between_bases"): BLAS,
    ("numpy.linalg.matrix_rank", "solvers.constrained_qp:_stationarity_certificate"): BLAS,
    ("numpy.linalg.eigh", "distributional.smoothing.objective:_stable_isolated"): BLAS,
    ("numpy.linalg.cholesky", "distributional.smoothing.newton:newton_direction"): BLAS,
}

#: The exact routes repeat a quarter to a third of their Gram passes (the iterate an accepted
#: step re-enters), the discrete route none on this data.  A ceiling is a fraction of the centred
#: systems built, and a count below it passes.  A route that builds every system twice repeats
#: half of its passes, so the discrete ceiling sits under a half; the exact routes already repeat
#: a third, so doubling takes them past theirs.
GRAM_PASSES = ("packed_centered_gram_rhs", "anchor_support_centered_gram_rhs", "centered_gram_rhs")
GRAM_REPEAT_CEILING = {"glm_exact": 0.5, "glm_shape_constrained": 0.5, "glm_discrete": 0.25}
ROUTE_GRAM_PASSES = {
    "glm_exact": GRAM_PASSES,
    "glm_shape_constrained": GRAM_PASSES,
    "glm_discrete": ("packed_centered_gram_rhs",),
}

PREDICTORS = 2  # location and scale of the binned SuperLSS fit
#: Fellner-Schall updates alone from a far start to strict tolerances: many more smoothing iterations.
FORCED = {
    "outer": "efs",
    "initial_lambda": 1e-4,
    "practical_reml": False,
    "reml_tol": 1e-12,
    "reml_plateau_tol": 1e-14,
    "inner_tol": 1e-12,
}


def _frame():
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame(
        {"x": rng.uniform(size=n), "z": rng.uniform(size=n), "g": rng.choice(list("abcd"), n)}
    )
    counts = rng.poisson(np.exp(0.3 + np.sin(3 * frame["x"]) + 0.5 * frame["z"])).astype(float)
    gaussian = np.sin(3 * frame["x"]) + np.exp(-1.0 + 0.5 * frame["z"]) * rng.normal(size=n)
    return frame, counts, gaussian


def _glm(discrete=False, increasing=False):
    frame, counts, _ = _frame()
    z = PSpline(n_knots=6, constraint=Constraint.fit.increasing if increasing else None)
    model = SuperGLM(
        family="poisson",
        discrete=discrete,
        features={"x": PSpline(n_knots=6), "z": z, "g": Categorical()},
    )
    return model.fit_reml(frame, counts)


def _lss_binned(**options):
    frame, _, gaussian = _frame()
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(s("x", kind="cr", k=6), cat("g")),
        family.scale(s("z", kind="cr", k=5)),
        discrete=True,
    )
    return model.fit_reml(frame, gaussian, **options)


FITS = {
    "glm_exact": _glm,
    "glm_discrete": lambda: _glm(discrete=True),
    "lss_binned": _lss_binned,
    "glm_shape_constrained": lambda: _glm(increasing=True),
}


def _recorded(fit):
    with work_ledger() as ledger:
        fit()
    assert ledger.errors == []
    return ledger


@pytest.fixture(scope="module")
def ledgers():
    return {name: _recorded(fit) for name, fit in FITS.items()}


@pytest.fixture(scope="module")
def forced_lss_ledger():
    return _recorded(lambda: _lss_binned(**FORCED))


def _reason(row, table):
    for (operation, prefix), reason in table.items():
        if operation in ("*", row.operation) and row.caller.startswith(prefix):
            return reason
    return None


def _repeat(row) -> str:
    return f"{row.operation} @ {row.caller}: {row.repeated} of {row.count} calls repeat"


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("name", list(FITS))
def test_a_fit_repeats_no_pass_over_the_rows_that_is_not_listed(ledgers, name):
    repeated = [row for row in ledgers[name].rows() if row.repeated]
    unlisted = [
        _repeat(row)
        for row in repeated
        if row.operation in ROW_PASSES and _reason(row, ROW_PASS_REPEATS) is None
    ]
    others = [
        f"{_repeat(row)} ({'known' if _reason(row, OTHER_REPEATS) else 'not listed'})"
        for row in repeated
        if row.operation not in ROW_PASSES
    ]
    if others:
        warnings.warn(f"{name}: repeats outside the row passes, not failed:\n" + "\n".join(others))
    unidentified = [
        f"{row.operation} @ {row.caller}: {row.uncounted} of {row.count} calls"
        for row in ledgers[name].rows()
        if row.operation in ROW_PASSES and row.uncounted
    ]

    assert unlisted == []
    # such a call is never a repeat, so a pass handed one could repeat without this file seeing it:
    # declare the argument a carrier of its target (TARGETS) if it carries no input
    assert unidentified == []


@pytest.mark.pins_behaviour
def test_every_listed_repeat_of_a_pass_over_the_rows_still_happens(ledgers):
    repeated = [row for ledger in ledgers.values() for row in ledger.rows() if row.repeated]
    stale = [
        f"{operation} @ {prefix}"
        for operation, prefix in ROW_PASS_REPEATS
        if not any(row.operation == operation and row.caller.startswith(prefix) for row in repeated)
    ]

    assert stale == [], (
        f"no fit repeats these any more; delete the entries from ROW_PASS_REPEATS: {stale}"
    )


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("name", ["glm_exact", "glm_shape_constrained"])
def test_the_exact_route_builds_one_system_and_one_gram_pass_per_iteration(ledgers, name):
    ledger = ledgers[name]
    systems = ledger.count("build_centered_system")

    assert systems > 0
    for passes in GRAM_PASSES:
        assert ledger.count(passes) == systems, passes
    assert ledger.count("build_design_matrix") == ledger.count("finalize_reml_fit") == 1


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("name", list(GRAM_REPEAT_CEILING))
def test_a_route_repeats_at_most_a_fraction_of_its_gram_passes(ledgers, name):
    ledger = ledgers[name]
    systems = ledger.count("build_centered_system")

    assert systems > 0
    for passes in ROUTE_GRAM_PASSES[name]:
        repeated = sum(row.repeated for row in ledger.find(passes))
        assert repeated <= GRAM_REPEAT_CEILING[name] * systems, (
            f"{passes}: {repeated} of {systems} centred systems"
        )


@pytest.mark.pins_behaviour
def test_the_discrete_route_passes_over_the_bins_once_per_system(ledgers):
    ledger = ledgers["glm_discrete"]
    systems = ledger.count("build_centered_system")

    assert systems > 0
    assert ledger.count("packed_centered_gram_rhs") == systems
    assert ledger.count("centered_gram_rhs") == 0  # no pass over the rows themselves
    assert ledger.count("build_design_matrix") == ledger.count("finalize_reml_fit") == 1


@pytest.mark.pins_behaviour
def test_a_binned_superlss_fit_builds_each_stage_once_and_makes_two_moment_passes_per_predictor(
    ledgers, forced_lss_ledger
):
    ledger = ledgers["lss_binned"]
    once = ("compile_predictors", "fit_joint_null_model", "prepare_distributional_fit_state")
    solves = "_fit_dense_fixed_lambda_core"

    assert forced_lss_ledger.count(solves) > ledger.count(
        solves
    )  # the forced fit does iterate more
    for each in (ledger, forced_lss_ledger):
        assert [each.count(stage) for stage in once] == [1, 1, 1]
        assert each.count("MatrixExecutionPlan.moments") == 2 * PREDICTORS


@pytest.mark.pins_behaviour
def test_a_binned_superlss_fit_evaluates_the_whole_likelihood_at_most_every_other_solve(ledgers):
    # The first trial of a solve is fused and not counted, so what remains is the initial and
    # terminal state: two for a whole default fit, however many solves it makes (6 to 52 over 27
    # data sets).  Backtracked trials add more when the fit is forced, a count that follows the
    # line search and so has no ceiling; an evaluation added to every solve would pass this one.
    ledger = ledgers["lss_binned"]
    evaluations = ledger.count("_evaluate_state_unmeasured")

    assert 0 < evaluations <= 0.5 * ledger.count("_fit_dense_fixed_lambda_core")
