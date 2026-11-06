"""Four small canonical fits repeat only the work this file names, and pass over the rows a counted number of times.

The work ledger flags a call whose input an earlier call of the same operation had bit for bit.
A new flag outside ``ALLOWED`` is new repeated work; an entry here is a known repeat with its
reason, to be removed when the repeat is.  Counts are of events that do not depend on how many
iterations a BLAS build's rounding lets a fit take: once-per-fit work, and one row pass per
centred system built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, GaussianLS, PSpline, SuperGLM, SuperLSS, cat, s
from superglm.diagnostics.work_ledger import work_ledger
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

#: ``(operation, caller prefix)`` -> reason.  ``"*"`` matches every operation at that caller.
ALLOWED = {
    ("*", "reml.penalty_algebra:"): PENALTY,
    ("*", "dm_builder:_whitening_map"): PENALTY,
    ("numpy.linalg.eigh", "solvers.centered_system:_equilibrated_block_root"): PENALTY,
    ("numpy.linalg.eigvalsh", "reml.efs_update:"): PENALTY,
    ("numpy.linalg.eigvalsh", "features._spline_penalties:_interval_blocks"): PENALTY,
    ("centered_gram_rhs", "solvers.centered_system:build_centered_system"): ITERATE,
    ("anchor_support_centered_gram_rhs", "solvers.centered_system:_raw_rung_system"): ITERATE,
    ("DesignMatrix.rmatvec", "solvers.centered_system:build_centered_system"): ITERATE,
    ("numpy.linalg.cholesky", "solvers.centered_system:_attach_centered_penalty"): ITERATE,
    ("centred_matvec", "solvers.irls_direct._fit:evaluate_state"): ITERATE,
    ("centred_matvec", "solvers.mode_score:linear_predictor"): ITERATE,
    ("DesignMatrix.rmatvec", "solvers.mode_score:"): ITERATE,
    ("DesignMatrix.rmatvec", "reml.identified:"): ITERATE,
    ("*", "solvers.rank:_decompose_gram"): RANK,
    ("decompose_gram", "distributional.solver.solver:_initial_coefficients"): RANK,
    ("numpy.linalg.svd", "solvers.rank:_selection_amplification"): BLAS,
    ("numpy.linalg.lstsq", "reml.result:_map_beta_between_bases"): BLAS,
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


def _lss_binned():
    frame, _, gaussian = _frame()
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(s("x", kind="cr", k=6), cat("g")),
        family.scale(s("z", kind="cr", k=5)),
        discrete=True,
    )
    return model.fit_reml(frame, gaussian)


FITS = {
    "glm_exact": _glm,
    "glm_discrete": lambda: _glm(discrete=True),
    "lss_binned": _lss_binned,
    "glm_shape_constrained": lambda: _glm(increasing=True),
}


@pytest.fixture(scope="module")
def ledgers():
    recorded = {}
    for name, fit in FITS.items():
        with work_ledger() as ledger:
            fit()
        assert ledger.errors == []
        recorded[name] = ledger
    return recorded


def _allowed(row) -> bool:
    return any(
        operation in ("*", row.operation) and row.caller.startswith(prefix)
        for operation, prefix in ALLOWED
    )


@pytest.mark.parametrize("name", list(FITS))
def test_a_fit_repeats_no_work_that_is_not_listed(ledgers, name):
    unlisted = [
        f"{row.operation} @ {row.caller}: {row.repeated} of {row.count} calls repeat"
        for row in ledgers[name].rows()
        if row.repeated and not _allowed(row)
    ]

    assert unlisted == []


@pytest.mark.parametrize("name", ["glm_exact", "glm_shape_constrained"])
def test_the_exact_route_builds_one_system_and_one_gram_pass_per_iteration(ledgers, name):
    ledger = ledgers[name]
    systems = ledger.count("build_centered_system")

    assert systems > 0
    for passes in (
        "packed_centered_gram_rhs",
        "anchor_support_centered_gram_rhs",
        "centered_gram_rhs",
    ):
        assert ledger.count(passes) == systems, passes
    assert ledger.count("build_design_matrix") == ledger.count("finalize_reml_fit") == 1


def test_the_discrete_route_passes_over_the_bins_once_per_system_and_repeats_none(ledgers):
    ledger = ledgers["glm_discrete"]
    systems = ledger.count("build_centered_system")

    assert systems > 0
    assert ledger.count("packed_centered_gram_rhs") == systems
    assert ledger.count("centered_gram_rhs") == 0  # no pass over the rows themselves
    assert sum(row.repeated for row in ledger.find("packed_centered_gram_rhs")) == 0
    assert ledger.count("build_design_matrix") == ledger.count("finalize_reml_fit") == 1


def test_a_binned_superlss_fit_builds_each_stage_once_and_makes_a_fixed_number_of_passes(ledgers):
    ledger = ledgers["lss_binned"]
    once = ("compile_predictors", "fit_joint_null_model", "prepare_distributional_fit_state")

    assert [ledger.count(stage) for stage in once] == [1, 1, 1]
    assert ledger.count("MatrixExecutionPlan.moments") == 4  # the diagonal-moment passes
    assert (
        ledger.count("_evaluate_state_unmeasured") == 2
    )  # likelihood evaluations of the whole fit
