"""The debug work ledger: what it records, what it flags, and that it leaves nothing behind."""

from __future__ import annotations

import importlib
import subprocess
import sys
import threading
import types

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


def _flagged_updates(*matrices):
    with work_ledger() as ledger:
        for matrix in matrices:
            np.linalg.eigvalsh(matrix)
    return ledger.find("numpy.linalg.eigvalsh")[0].update_candidates


def test_dropping_or_inserting_an_interior_row_and_column_is_an_update_candidate():
    """The saddle system ``[[H, -A'], [A, 0]]`` loses a constraint that is not the last one."""
    p = 8
    H = np.eye(p) + 0.5 * np.ones((p, p)) / p  # constant diagonal
    A = np.random.default_rng(30).normal(size=(3, p))

    def saddle(rows):
        return np.block([[H, -rows.T], [rows, np.zeros((len(rows), len(rows)))]])

    kkt3, kkt2 = saddle(A), saddle(A[[0, 2]])
    assert _flagged_updates(kkt3, kkt2) == 1
    assert _flagged_updates(kkt2, kkt3) == 1

    flat = np.full((6, 6), 0.1) + 1.9 * np.eye(6)  # every diagonal entry equal
    inserted = np.insert(np.insert(flat, 2, 0.0, axis=0), 2, 0.0, axis=1)
    inserted[2, 2] = 2.0
    assert _flagged_updates(flat, inserted) == 1
    assert _flagged_updates(inserted, flat) == 1


def test_calls_inside_a_library_routine_are_not_counted_again():
    matrix = _spd(10, 6)
    with work_ledger() as ledger:
        scipy.linalg.cholesky(matrix)
        scipy.linalg.eigh(matrix)

    assert {row.operation for row in ledger.rows()} == {
        "scipy.linalg.cholesky",
        "scipy.linalg.eigh",
    }


def test_only_the_last_input_of_each_call_site_is_kept_and_dropped_on_exit():
    with work_ledger() as ledger:
        for seed in range(30):
            np.linalg.cholesky(_spd(40, seed))
            np.linalg.eigvalsh(_spd(40, seed))
        assert len(ledger._last) == 2
        assert ledger._last_bytes == sum(kept.nbytes for kept in ledger._last.values())

    assert (ledger._last, ledger._last_bytes, ledger._seen) == ({}, 0, {})
    assert ledger.count("numpy.linalg.cholesky") == 30  # the rows stay


def test_the_input_budget_evicts_the_oldest_site_and_an_oversize_input_is_not_copied(monkeypatch):
    monkeypatch.setattr(work_ledger_module, "_INPUT_BYTES", 14_000)  # holds four 20x20 copies
    matrix = _spd(20, 0)
    with work_ledger() as ledger:
        for name in ("cholesky", "eigvalsh", "eigh", "svd", "inv", "qr"):
            getattr(np.linalg, name)(matrix)  # looked up inside the block, where it is wrapped
        kept = {site[0] for site in ledger._last}
        assert ledger._last_bytes <= 14_000
        assert kept == {f"numpy.linalg.{name}" for name in ("eigh", "svd", "inv", "qr")}

    with work_ledger() as oversize:
        np.linalg.cholesky(_spd(40, 1))  # 12,800 bytes, over a quarter of the budget
        assert oversize._last == {}
    assert oversize.find("numpy.linalg.cholesky")[0].untracked == 1


def test_a_call_site_is_compared_with_its_own_previous_input_not_a_neighbour_in_the_function():
    def one_row_changed(matrix):
        changed = matrix.copy()
        changed[2, :] += 1.0
        changed[:, 2] = changed[2, :]
        return changed

    def two_sites(first, second):
        np.linalg.eigvalsh(first)
        np.linalg.eigvalsh(second)

    first, second = _spd(10, 20), _spd(10, 21)
    with work_ledger() as ledger:
        two_sites(first, second)
        two_sites(one_row_changed(first), one_row_changed(second))

    assert ledger.find("numpy.linalg.eigvalsh")[0].update_candidates == 2


def test_the_same_value_under_another_keyword_is_another_input():
    matrix = _spd(6, 22)
    with work_ledger() as ledger:
        scipy.linalg.cholesky(matrix.copy(), lower=True)
        scipy.linalg.cholesky(matrix.copy(), overwrite_a=True)

    assert ledger.find("scipy.linalg.cholesky")[0].repeated == 0


def test_an_r_only_qr_costs_half_of_one_that_forms_q_and_syrk_reads_trans_by_position():
    matrix = np.random.default_rng(23).normal(size=(60, 6))

    def flops(call):
        with work_ledger() as ledger:
            call()
        return ledger.rows()[0].flops

    assert flops(lambda: np.linalg.qr(matrix)) == 2 * flops(lambda: np.linalg.qr(matrix, mode="r"))
    assert flops(lambda: scipy.linalg.blas.dsyrk(1.0, matrix, 0.0, np.zeros((6, 6)), 1)) == (
        60 * 6 * 7
    )


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


def test_a_routine_reached_through_the_warm_get_funcs_cache_is_recorded():
    """SciPy memoizes ``get_blas_funcs``; a lookup before the block holds the original routine."""
    matrix = np.random.default_rng(9).normal(size=(30, 6))
    scipy.linalg.get_blas_funcs("syrk", (matrix,))  # warms the cache outside the block

    with work_ledger() as ledger:
        scipy.linalg.get_blas_funcs("syrk", (matrix,))(1.0, matrix.T)

    assert ledger.count("scipy.linalg.blas.dsyrk") == 1


def test_no_wrapper_stays_in_the_get_funcs_cache_after_the_block():
    """A first lookup inside the block must not leave the wrapper in SciPy's memo for the process."""
    code = (
        "import numpy as np, scipy.linalg\n"
        "from superglm.diagnostics.work_ledger import work_ledger\n"
        "a = np.eye(3) * 2.0\n"
        "with work_ledger():\n"
        "    scipy.linalg.cholesky(a)\n"
        "    scipy.linalg.get_blas_funcs('syrk', (a,))\n"
        "memos = [scipy.linalg.get_lapack_funcs.memo, scipy.linalg.get_blas_funcs.memo]\n"
        "assert not [v for memo in memos for v in memo.values() if hasattr(v, '__wrapped__')]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


def test_a_warm_get_funcs_cache_is_back_after_the_block():
    matrix = np.random.default_rng(24).normal(size=(30, 6))
    syrk = scipy.linalg.get_blas_funcs("syrk", (matrix,))
    with work_ledger():
        scipy.linalg.get_blas_funcs("syrk", (matrix,))
    assert scipy.linalg.get_blas_funcs("syrk", (matrix,)) is syrk


def test_a_module_imported_during_the_block_does_not_keep_the_wrapper(tmp_path, monkeypatch):
    (tmp_path / "late_ledger_module.py").write_text("from numpy.linalg import cholesky\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    original = np.linalg.cholesky
    try:
        with work_ledger():
            late = importlib.import_module("late_ledger_module")
            assert late.cholesky is not original
        assert late.cholesky is original
    finally:
        sys.modules.pop("late_ledger_module", None)


def test_bookkeeping_on_worker_threads_is_not_taken_from_the_wall_clock():
    matrix = np.random.default_rng(25).normal(size=(5000, 1))
    with work_ledger() as ledger:
        threads = [
            threading.Thread(target=np.linalg.qr, args=(matrix,), kwargs={"mode": "r"})
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert ledger.count("numpy.linalg.qr") == 8
    assert ledger.bookkeeping > 0 and ledger.main_bookkeeping == 0
    assert ledger.errors == []


def test_exclusive_seconds_leave_out_the_listed_operations_inside(monkeypatch):
    import superglm.solvers.rank as rank

    clock = [0.0]
    monkeypatch.setattr(
        work_ledger_module, "time", types.SimpleNamespace(perf_counter=lambda: clock[0])
    )

    def eigh(matrix):
        clock[0] += 2.0

    def decompose_gram(matrix):
        clock[0] += 1.0
        np.linalg.eigh(matrix)
        clock[0] += 4.0

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    monkeypatch.setattr(rank, "decompose_gram", decompose_gram)
    with work_ledger() as ledger:
        rank.decompose_gram(_spd(5, 26))

    assert ledger.find("decompose_gram")[0].seconds == 5.0
    assert ledger.find("numpy.linalg.eigh")[0].seconds == 2.0
    assert ledger.elapsed == 7.0
