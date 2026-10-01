"""sz models saved by superglm v0.35.0 load, predict and rebuild their inference (design §3.12, T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_sz_fixtures.py`` and
``direct_solve="structured"``: their retained linear system names the
retired range-space family (``SumToZeroBlockFactor``,
``ProfiledSumToZeroBlockFactor``, ``SumToZeroBlockStructuredSystem`` and the
factor's ``_LocalPSD`` and ``_SymmetricBorderFactor``).  Each record holds the
model, its training rows, its predictions and, for models that retained
their fit state, the standard errors and every level's curve standard error
(the implied last level's included) v0.35.0 reported.  The saved models keep
their raw B-spline coordinates: the balance tree takes their penalty's square
root as it is.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from superglm.solvers._structured.balance_tree import SumToZeroTreeFactor

from .test_saved_fs_models import _gamma, _se_tolerance, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"
NOTICE = "rebuilt with the current solver"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _curve_rounding(model, curves) -> np.ndarray:
    """The bound two float64 evaluations of the term's curves are held to, per row.

    A level's curve is ``B (M beta)``: the grid basis ``B`` (``k`` columns)
    times that level's block of the term's coefficients ``M beta`` (the
    implied last level's included).  Chained dot products of ``w`` and ``k``
    terms are within ``gamma_{k+w} |B| |M| |beta|`` of the exact product
    (Higham 2002, eq. 3.5 twice), so two evaluations differ by twice that.
    ``|M| |beta|`` is formed a coefficient at a time from the map itself.
    """
    from superglm.inference.factor_smooths import _resolve_factor_smooth

    group, spec = _resolve_factor_smooth(model, "x:g:sz")
    beta = np.asarray(model.result.beta[group.sl], dtype=np.float64)
    blocks = np.zeros_like(spec._level_blocks(beta))
    for j in np.flatnonzero(beta):
        unit = np.zeros(beta.size)
        unit[j] = 1.0
        blocks += np.abs(spec._level_blocks(unit)) * abs(beta[j])
    levels = {str(level): index for index, level in enumerate(spec._levels)}
    basis = np.abs(spec.marginal_basis(curves[spec.variable].to_numpy()))
    rows = np.asarray([levels[str(level)] for level in curves["level"]])
    magnitude = np.einsum("ij,ij->i", basis, blocks[rows])
    return 2.0 * _gamma(beta.size + spec.k) * magnitude


@pytest.mark.parametrize("name", ["sz_gaussian_exact", "sz_poisson_discrete"])
def test_an_sz_model_saved_by_v0_35_0_loads_predicts_and_rebuilds_its_inference(name) -> None:
    """T8 row "retired-class shim removed": without the module ``__getattr__`` of
    ``sum_to_zero`` and ``moments`` the pickle does not load at all.

    Predictions never read the solver state: they are v0.35.0's within two
    evaluations' rounding (``assert_predicts_as_saved``).  The first
    inference call rebuilds it with the balance tree at the saved
    coefficients and smoothing parameters, once, with the notice; its
    standard errors, and every level's curve standard errors, agree with the
    ones v0.35.0 reported.
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "SumToZeroBlockFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]

    assert_predicts_as_saved(model, frame, record["prediction"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    assert sum(NOTICE in str(item.message) for item in caught) == 1
    assert isinstance(model._linear_system_state.augmented_factor, SumToZeroTreeFactor)

    tolerance = _se_tolerance(model)
    for term, saved in record["se"].items():
        rebuilt = np.asarray(se[term], dtype=float)
        np.testing.assert_array_equal(np.isfinite(rebuilt), np.isfinite(saved))
        finite = np.isfinite(saved)
        relative = np.abs(rebuilt[finite] - saved[finite]) / np.abs(saved[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = model.factor_smooth("x:g:sz", grid=7).curves
    excess = np.abs(curves["effect"].to_numpy() - record["curve_effect"]) / _curve_rounding(
        model, curves
    )
    assert np.all(excess <= 1.0), float(np.max(excess))
    saved_curve_se = record["curve_se"]
    relative = np.abs(curves["posterior_se"].to_numpy() - saved_curve_se) / saved_curve_se
    assert np.all(relative <= tolerance), (float(np.max(relative)), tolerance)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.metrics(frame, y)
    assert not any(NOTICE in str(item.message) for item in caught)


def test_a_released_sz_model_saved_by_v0_35_0_predicts_and_asks_for_a_refit() -> None:
    """A model saved with ``retain_fit_state=False`` predicts; ``metrics``, ``summary`` and ``factor_smooth``
    each ask for a refit (``summary`` and the term reports raised an unrelated
    ``AttributeError`` before the retained state was resolved first).
    """
    record = _load("sz_gaussian_released")
    assert record["version"] == "0.35.0"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
        model.metrics(frame, y).coefficient_se
    for call in (model.summary, lambda: model.factor_smooth("x:g:sz")):
        with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
            call()
