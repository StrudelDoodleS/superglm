"""The debug work ledger: what it records, what it flags, and that it leaves nothing behind."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.diagnostics.work_ledger as work_ledger_module
from superglm import Categorical, PSpline, SuperGLM
from superglm.diagnostics.work_ledger import TARGETS, work_ledger
from superglm.solvers.constrained_qp import solve_constrained_qp
from tests.bound_predictor_fixtures import model_from_templates


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, n))
    return factor @ factor.T + n * np.eye(n)


def _capped_problem(p=40, seed=0):
    """A coefficient vector with every entry capped: the active set grows one row at a time."""
    rng = np.random.default_rng(seed)
    H = np.eye(p) + 0.5 * np.ones((p, p)) / p
    A = np.vstack([np.eye(p), -np.eye(p)])
    b = np.full(2 * p, -0.35)
    g = H @ rng.normal(0.0, 0.6, p)
    return H, g, A, b


def _bindings():
    """Every binding the ledger may replace: the declared entry points and the names that alias them."""
    import sys

    seen = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(("superglm", "numpy.linalg", "scipy.linalg")) and module is not None:
            seen.update({(name, key): value for key, value in vars(module).items()})
    return seen


def test_qp_saddle_solves_are_counted_with_the_iterations_and_flagged_as_updates():
    H, g, A, b = _capped_problem()
    with work_ledger() as ledger:
        result = solve_constrained_qp(H, g, A, b, max_iter=200)

    solves = ledger.find("numpy.linalg.solve")
    equilibrations = ledger.find("scipy.linalg.lapack.dsyequb")
    assert result.n_iter > 10
    # One saddle solve per iteration that took a step, plus at most the final certificate.
    assert result.n_iter - 3 <= sum(row.count for row in solves) <= result.n_iter + 1
    assert sum(row.count for row in equilibrations) == sum(row.count for row in solves)
    # Each equilibrated system differs from the last by the one constraint row entering or leaving.
    flagged = sum(row.update_candidates for row in equilibrations)
    assert flagged >= sum(row.count for row in equilibrations) - 2
    assert all(row.caller.endswith("_solve_saddle_direct") for row in solves + equilibrations)


def test_an_exact_repeat_is_flagged_and_distinct_inputs_are_not():
    first, second = _spd(8, 1), _spd(8, 2)
    with work_ledger() as ledger:
        for matrix in (first, second, first.copy(), first, second.copy()):
            np.linalg.cholesky(matrix)
        scipy.linalg.cholesky(first)  # another operation: its own history

    (row,) = ledger.find("numpy.linalg.cholesky")
    assert (row.count, row.repeated) == (5, 3)
    assert ledger.find("scipy.linalg.cholesky")[0].repeated == 0
    assert [flagged.operation for _, flagged in ledger.flagged()] == ["numpy.linalg.cholesky"]


def test_mode_and_other_arguments_make_inputs_distinct():
    matrix = np.random.default_rng(3).normal(size=(9, 5))
    with work_ledger() as ledger:
        np.linalg.qr(matrix, mode="reduced")
        np.linalg.qr(matrix, mode="r")

    assert ledger.find("numpy.linalg.qr")[0].repeated == 0


def test_a_one_row_or_one_border_change_is_an_update_candidate_and_a_rebuild_is_not():
    base = _spd(12, 4)
    row_changed = base.copy()
    row_changed[3, :] += 1.0
    row_changed[:, 3] = row_changed[3, :]
    bordered = np.block([[base, np.ones((12, 1))], [np.ones((1, 12)), np.full((1, 1), 50.0)]])
    unrelated = _spd(12, 5)

    def flagged(*matrices):
        with work_ledger() as ledger:
            for matrix in matrices:
                np.linalg.eigvalsh(matrix)
        return ledger.find("numpy.linalg.eigvalsh")[0].update_candidates

    assert flagged(base, row_changed) == 1
    assert flagged(base, bordered) == 1
    assert flagged(bordered, base) == 1
    assert flagged(base, unrelated) == 0
    assert flagged(base, base * 1.0001) == 0


def test_calls_inside_a_library_routine_are_not_counted_again():
    matrix = _spd(10, 6)
    with work_ledger() as ledger:
        scipy.linalg.cholesky(matrix)
        scipy.linalg.eigh(matrix)

    assert {row.operation for row in ledger.rows()} == {
        "scipy.linalg.cholesky",
        "scipy.linalg.eigh",
    }


def test_only_the_last_input_of_each_call_site_is_kept():
    with work_ledger() as ledger:
        for seed in range(30):
            np.linalg.cholesky(_spd(40, seed))
            np.linalg.eigvalsh(_spd(40, seed))

    assert len(ledger._last) == 2
    assert ledger._last_bytes == sum(kept.nbytes for kept in ledger._last.values())


def test_every_declared_entry_point_resolves_and_is_put_back_after_an_exception():
    before = _bindings()
    with pytest.raises(ZeroDivisionError):
        with work_ledger() as ledger:
            assert np.linalg.cholesky is not before[("numpy.linalg", "cholesky")]
            raise ZeroDivisionError
    after = _bindings()

    assert ledger.skipped == []
    assert len(TARGETS) > 40
    assert {key for key in before if before[key] is not after.get(key)} == set()
    assert work_ledger_module._ACTIVE is None
    with work_ledger():  # a new ledger starts cleanly
        pass


def test_one_ledger_records_at_a_time():
    with work_ledger():
        with pytest.raises(RuntimeError, match="already recording"):
            with work_ledger():
                pass


def _glm_frame():
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(size=n), "g": rng.choice(list("abcd"), n)})
    y = rng.poisson(np.exp(0.3 + np.sin(3 * frame["x"]))).astype(float)
    return frame, y


def test_a_superglm_fit_is_recorded_and_unchanged_by_the_ledger():
    frame, y = _glm_frame()

    def fit():
        model = SuperGLM(family="poisson", features={"x": PSpline(n_knots=6), "g": Categorical()})
        return model.fit_reml(frame, y)

    plain = fit().result.beta
    with work_ledger() as ledger:
        recorded = fit().result.beta

    np.testing.assert_array_equal(recorded, plain)
    assert ledger.errors == []
    assert ledger.count("centered_gram_rhs") + ledger.count("MatrixExecutionPlan.moments") > 0
    assert ledger.count("numpy.linalg.cholesky") + ledger.count("scipy.linalg.cholesky") > 0


def test_a_superlss_fit_is_recorded_and_unchanged_by_the_ledger():
    from superglm.distributional.families.gaussian import GaussianLS
    from superglm.distributional.predictor import Predictor
    from superglm.features import Spline

    rng = np.random.default_rng(8)
    x = rng.uniform(size=300)
    response = np.sin(3 * x) + np.exp(-1.0 + 0.5 * x) * rng.normal(size=300)
    frame = pd.DataFrame({"x": x})

    def fit():
        model = model_from_templates(
            family=GaussianLS(),
            predictors=(
                Predictor("location", {"x": Spline(n_knots=5)}),
                Predictor("scale", {"x": Spline(n_knots=4)}),
            ),
        )
        return model.fit_reml(frame, response).coef_

    plain = dict(fit())
    with work_ledger() as ledger:
        recorded = dict(fit())

    assert ledger.errors == []
    assert recorded.keys() == plain.keys()
    np.testing.assert_array_equal(list(recorded.values()), list(plain.values()))
    assert ledger.count("_evaluate_state_unmeasured") > 0
    assert ledger.count("_assemble_dense_geometry_from_matrices") > 0
    assert "numpy.linalg.eigh" in {row.operation for row in ledger.rows()}


def test_the_report_names_each_operation_once_per_caller():
    with work_ledger() as ledger:
        for seed in range(3):
            np.linalg.cholesky(_spd(6, seed))
    text = ledger.report()

    assert text.count("numpy.linalg.cholesky") == 1
    assert "repeat" in text and "update" in text
    assert ledger.as_dict()["rows"][0]["count"] == 3
