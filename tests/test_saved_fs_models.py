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


def dense_unit_se(model, se: dict) -> dict:
    """``sqrt(diag(H^-1))`` per term, ``H`` the penalised Hessian of the model's own design rows.

    An oracle independent of the solver: the dense design at the saved
    coefficients' working weights, ``X'WX`` plus the saved smoothing
    parameters' penalty, inverted densely.  ``coefficient_se_raw`` reads the
    same covariance at unit dispersion.
    """
    from superglm.model.state_ops import _solver_space_working_weights

    dm = model._dm
    X = np.hstack([np.ones((dm.n, 1)), dm.toarray()])
    weights = _solver_space_working_weights(model)
    H = X.T @ (weights[:, None] * X)
    H[1:, 1:] += build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    unit = np.sqrt(np.diag(np.linalg.inv(H))[1:])
    offsets = np.cumsum([0] + [np.asarray(value).size for value in se.values()])
    assert offsets[-1] == unit.size
    return {term: unit[offsets[i] : offsets[i + 1]] for i, term in enumerate(se)}


def assert_rebuilt_se(model, frame, y, se: dict, saved: dict) -> None:
    """The rebuilt errors are the penalised Hessian's, on every coordinate, and v0.35.0's where it reported one.

    The penalised Hessian of these fits is positive definite, so every
    coefficient is identified (``solvers.identifiability``) and every error is
    finite: the mask is the whole vector, including the coordinates v0.35.0's
    data-only rule blanked.  Every error at unit dispersion is the dense
    oracle's within ``_se_tolerance``; the dispersion ``coefficient_se``
    applies is pinned by v0.35.0's own reported errors (``saved_se_scale``).
    """
    tolerance = _se_tolerance(model)
    raw = model.metrics(frame, y).coefficient_se_raw
    oracle = dense_unit_se(model, se)
    scale = saved_se_scale(model, frame, y)
    for term, expected in oracle.items():
        assert np.all(np.isfinite(expected)), term
        for values in (se[term], raw[term]):
            assert np.array_equal(
                np.isfinite(np.asarray(values, dtype=float)), np.isfinite(expected)
            )
        relative = np.abs(np.asarray(raw[term], dtype=float) - expected) / expected
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)
        reported = scale * np.asarray(saved[term], dtype=float)
        finite = np.isfinite(reported)
        rebuilt = np.asarray(se[term], dtype=float)
        relative = np.abs(rebuilt[finite] - reported[finite]) / np.abs(reported[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)
        # the dispersion of the restored errors is the reported ones' dispersion
        ratio = rebuilt / np.asarray(raw[term], dtype=float)
        assert np.all(np.abs(ratio - ratio[0]) <= tolerance * ratio[0]), term


_U = np.finfo(np.float64).eps / 2
# numpy's documented worst case for a float64 SIMD transcendental (its SVML
# kernels, numpy PR #19478); its own tests hold exp and log to 2 (PR #20991)
_TRANSCENDENTAL_ULPS = 4


def _gamma(count: int) -> float:
    return count * _U / (1.0 - count * _U)


def _term_magnitude(term, frame, beta_all) -> np.ndarray:
    """``sum_j |X_ij| |beta_j|`` over one prediction term's columns, one column at a time."""
    from superglm.model.base import _score_prediction_term_local_exact

    width = len(term["beta_idx"])
    beta = np.abs(beta_all[term["beta_idx"]])
    magnitude = np.zeros(len(frame))
    for j in np.flatnonzero(beta):
        unit = np.zeros(width)
        unit[j] = 1.0
        magnitude += np.abs(_score_prediction_term_local_exact(term, frame, unit)) * beta[j]
    return magnitude


def _prediction_rounding(model, frame) -> np.ndarray:
    """The bound two float64 evaluations of ``model.predict(frame)`` are held to, per row.

    ``eta_i = b0 + sum_j X_ij beta_j`` is a dot product of ``p + 1`` terms, so
    an evaluation in any summation order is within ``gamma_{p+1} a_i`` of the
    exact one, ``a_i = |b0| + sum_j |X_ij| |beta_j|`` (Higham 2002, eq. 3.5):
    two evaluations -- the saved one, and this one on another BLAS kernel or
    SIMD target -- differ by twice that.  The inverse link carries it at its
    slope, ``|h'(eta)| + |h''(eta)| d`` over the interval (``mu`` for the log
    link), and adds each evaluation's own error, at most
    ``_TRANSCENDENTAL_ULPS`` ulp of ``mu``.  Nothing here is the round-off of
    one machine: a wrong coefficient or basis moves a prediction by many
    orders of magnitude more.
    """
    from superglm._frame import as_eager_frame
    from superglm.model.base import _prediction_plan

    rows = as_eager_frame(frame)
    beta = np.asarray(model.result.beta, dtype=np.float64)
    plan = _prediction_plan(model)
    magnitude = np.full(len(rows), abs(float(model.result.intercept)))
    for term in plan["features"] + plan["interactions"]:
        magnitude += _term_magnitude(term, rows, beta)
    d_eta = 2.0 * _gamma(beta.size + 1) * magnitude
    eta = np.asarray(model._predict_eta_raw_exact(frame), dtype=np.float64)
    link = model._link
    slope = np.abs(link.deriv_inverse(eta)) + np.abs(link.deriv2_inverse(eta)) * d_eta
    mu = np.abs(np.asarray(link.inverse(eta), dtype=np.float64))
    return slope * d_eta + 2.0 * _TRANSCENDENTAL_ULPS * 2.0 * _U * mu


def assert_predicts_as_saved(model, frame, saved) -> None:
    """``model.predict(frame)`` is the saved prediction within ``_prediction_rounding``."""
    predicted = np.asarray(model.predict(frame), dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    assert predicted.shape == saved.shape
    excess = np.abs(predicted - saved) / _prediction_rounding(model, frame)
    assert np.all(excess <= 1.0), (int(np.argmax(excess)), float(np.max(excess)))


def saved_se_scale(model, frame, y) -> float:
    """The factor that restates v0.35.0's ``coefficient_se`` at today's dispersion.

    For a known-scale family ``coefficient_se`` scales the covariance by
    ``pearson_chi2 / residual_df``.  Given the fit's own objects, v0.35.0 read
    ``pearson_chi2`` from the fit's statistics (``model._fit_stats``), which on
    a discrete fit belong to the binned design.  ``metrics`` now evaluates it
    on ``predict``'s mean (#441).  The factor is formed here from an
    independent Pearson sum on that mean, not from ``metrics``, so the
    comparison still pins the dispersion ``coefficient_se`` applies.

    Rounding: each Pearson sum has ``n`` non-negative terms and is within
    ``gamma_(n+3)`` of the exact sum, so the two agree within
    ``2 gamma_(n+3)`` relative; the ratio, square root and the multiply add a
    few units of roundoff.  Both are far below ``_se_tolerance``'s
    ``n u kappa``, which they leave unchanged.  The fixtures carry no weights or
    offsets.
    """
    if not model._distribution.scale_known:
        return 1.0
    mu = np.asarray(model.predict(frame), dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    pearson = float(np.sum((response - mu) ** 2 / model._distribution.variance(mu)))
    return float(np.sqrt(pearson / model._fit_stats.pearson_chi2))


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("name", ["fs_gaussian_exact", "fs_poisson_discrete"])
def test_an_fs_model_saved_by_v0_35_0_loads_predicts_and_rebuilds_its_inference(name) -> None:
    """T8 row "retired-class shim removed": without the module ``__getattr__`` of
    ``factors`` and ``moments`` the pickle does not load at all.

    Predictions never read the solver state: they are v0.35.0's within two
    evaluations' rounding (``assert_predicts_as_saved``).  The first
    inference call rebuilds it with the fs leaf factor at the saved
    coefficients and smoothing parameters, once, with the notice; every
    standard error is the penalised Hessian's, including those v0.35.0's
    data-only rule blanked, and agrees with v0.35.0's where it reported one
    (``assert_rebuilt_se``).
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "BlockSchurFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]

    assert_predicts_as_saved(model, frame, record["prediction"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    assert sum(NOTICE in str(item.message) for item in caught) == 1
    assert isinstance(model._linear_system_state.augmented_factor, FactorSmoothLeafFactor)

    assert_rebuilt_se(model, frame, y, se, record["se"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.metrics(frame, y)
    assert not any(NOTICE in str(item.message) for item in caught)


def test_a_released_fs_model_saved_by_v0_35_0_predicts_and_asks_for_a_refit() -> None:
    """A model saved with ``retain_fit_state=False`` predicts; ``metrics``, ``summary`` and ``factor_smooth``
    each ask for a refit (``summary`` and the term reports raised an unrelated
    ``AttributeError`` before the retained state was resolved first).
    """
    record = _load("fs_gaussian_released")
    assert record["version"] == "0.35.0"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
        model.metrics(frame, y).coefficient_se
    for call in (model.summary, lambda: model.factor_smooth("x:g:fs")):
        with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
            call()
