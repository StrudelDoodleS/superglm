"""fs models saved by superglm v0.35.0 load, predict and rebuild their inference (design §3.12, T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_fs_fixtures.py`` and
``direct_solve="auto"``: their retained linear system names the
retired block-Schur family (``BlockSchurFactor``, ``ProfiledBlockSchurFactor``,
``BlockStructuredSystem``).  Each record holds the model, its training rows,
its predictions and, for models that retained their fit state, the standard
errors v0.35.0 reported.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers._structured.block_leaves import FactorSmoothLeafFactor

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"
NOTICE = "rebuilt with the current solver"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _se_tolerance(model) -> float:
    """Relative SE agreement both solvers certify: ``n u kappa_s`` for the variances.

    Each route's ``diag(H^-1)`` is within ``gamma_n kappa_s(H)`` of the exact
    one relative to itself (the moment route forms ``X'WX`` in ``n``-term sums,
    Higham 2002 §3.1 and Theorem 10.4 on the Jacobi-scaled matrix), so the two
    variances differ by at most twice that, and their square roots by half of
    it again.
    """
    from superglm.model.state_ops import _solver_space_working_weights

    dm = model._dm
    X = np.hstack([np.ones((dm.n, 1)), dm.toarray()])
    weights = _solver_space_working_weights(model)
    penalty = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    H = X.T @ (weights[:, None] * X)
    H[1:, 1:] += penalty
    scale = 1.0 / np.sqrt(np.diag(H))
    kappa = np.linalg.cond(scale[:, None] * H * scale[None, :])
    u = np.finfo(float).eps / 2
    return float(dm.n * u / (1.0 - dm.n * u) * kappa)


@pytest.mark.parametrize("name", ["fs_gaussian_exact", "fs_poisson_discrete"])
def test_an_fs_model_saved_by_v0_35_0_loads_predicts_and_rebuilds_its_inference(name) -> None:
    """T8 row "retired-class shim removed": without the module ``__getattr__`` of
    ``factors`` and ``moments`` the pickle does not load at all.

    Predictions never read the solver state and stay bitwise.  The first
    inference call rebuilds it with the fs leaf factor at the saved
    coefficients and smoothing parameters, once, with the notice; its standard
    errors agree with the ones v0.35.0 reported.
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "BlockSchurFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]

    np.testing.assert_array_equal(np.asarray(model.predict(frame)), record["prediction"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    assert sum(NOTICE in str(item.message) for item in caught) == 1
    assert isinstance(model._linear_system_state.augmented_factor, FactorSmoothLeafFactor)

    tolerance = _se_tolerance(model)
    for term, saved in record["se"].items():
        rebuilt = np.asarray(se[term], dtype=float)
        np.testing.assert_array_equal(np.isfinite(rebuilt), np.isfinite(saved))
        finite = np.isfinite(saved)
        relative = np.abs(rebuilt[finite] - saved[finite]) / np.abs(saved[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.metrics(frame, y)
    assert not any(NOTICE in str(item.message) for item in caught)


def test_a_released_fs_model_saved_by_v0_35_0_predicts_and_asks_for_a_refit() -> None:
    record = _load("fs_gaussian_released")
    assert record["version"] == "0.35.0"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    np.testing.assert_array_equal(np.asarray(model.predict(frame)), record["prediction"])
    with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
        model.metrics(frame, y).coefficient_se
