"""Random-effect models saved by superglm v0.35.0 load, predict and rebuild their inference (T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_re_fixtures.py`` and
``direct_solve="auto"``: a single ``RandomEffect`` beside a narrow border took
the scalar Schur factor there, so their retained linear system names the
retired scalar family (``ScalarSchurFactor``, ``ProfiledScalarSchurFactor``,
``ScalarStructuredSystem`` and the factor's ``_DiagonalLowRank``), which the
one engine deleted (design §3.12, decision 13).  Each record holds the model,
its training rows, its predictions and, for models that retained their fit
state, the standard errors v0.35.0 reported.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from superglm.solvers._structured.nested import NestedSchurFactor

from .test_saved_fs_models import _se_tolerance, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"
NOTICE = "rebuilt with the current solver"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


@pytest.mark.parametrize("name", ["re_gaussian_exact", "re_poisson_discrete"])
def test_a_random_effect_model_saved_by_v0_35_0_loads_predicts_and_rebuilds(name) -> None:
    """T8 row "retired-class shim removed": without the module ``__getattr__`` of
    ``factors``, ``moments`` and ``operators`` the pickle does not load at all.

    Predictions never read the solver state: they are v0.35.0's within two
    evaluations' rounding (``assert_predicts_as_saved``).  The first
    inference call rebuilds it once, with the notice, as the random effect's
    chain of one on the nested factor at the saved coefficients and smoothing
    parameters; its standard errors agree with the ones v0.35.0 reported
    within the variances' ``n u kappa_s`` (``_se_tolerance``), with the same
    pattern of reported and unreported values, and the summary is formed.
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "ScalarSchurFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]

    assert_predicts_as_saved(model, frame, record["prediction"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    assert sum(NOTICE in str(item.message) for item in caught) == 1
    factor = model._linear_system_state.augmented_factor
    assert isinstance(factor, NestedSchurFactor)
    assert factor.chain_group_names == ("g",)

    tolerance = _se_tolerance(model)
    for term, saved in record["se"].items():
        rebuilt = np.asarray(se[term], dtype=float)
        np.testing.assert_array_equal(np.isfinite(rebuilt), np.isfinite(saved))
        finite = np.isfinite(saved)
        relative = np.abs(rebuilt[finite] - saved[finite]) / np.abs(saved[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert str(model.summary())
        model.metrics(frame, y)
    assert not any(NOTICE in str(item.message) for item in caught)


def test_a_released_random_effect_model_saved_by_v0_35_0_predicts_and_asks_for_a_refit() -> None:
    """A model saved with ``retain_fit_state=False`` predicts; ``metrics``, ``summary`` and ``random_effects``
    each ask for a refit (``summary`` and the term reports raised an unrelated
    ``AttributeError`` before the retained state was resolved first).
    """
    record = _load("re_gaussian_released")
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "ScalarSchurFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
        model.metrics(frame, y).coefficient_se
    for call in (model.summary, lambda: model.random_effects("g")):
        with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
            call()
