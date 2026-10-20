"""Never break userspace: what v0.40.0 accepted keeps working (the 0.41 audit's guard).

The fixtures in ``fixtures/saved_v0_40_0`` were written by the v0.40.0 release
(commit 542f5826) with ``scripts/make_saved_v040_fixtures.py``.  A model saved
by that release loads, predicts what it predicted then, and refits.

``fixtures/release_differential_v0_40_0.jsonl`` holds v0.40.0's outcomes on the
release differential's compact subset (``benchmarks/release_differential.py
run --subset`` on 542f5826): every case converged there, so each must converge
here, at an objective and held-out results no worse than the harness's
tolerances.
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from benchmarks import release_differential

from superglm import SuperLSS

from .test_saved_fs_models import _gamma, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_40_0"
GLM_RECORDS = sorted(path.stem for path in FIXTURES.glob("glm_*.pkl")) + ["sklearn_regressor"]
LSS_RECORDS = sorted(path.stem for path in FIXTURES.glob("lss_*.pkl"))


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def test_the_fixture_set_covers_every_saved_route() -> None:
    assert len(GLM_RECORDS) == 11
    assert LSS_RECORDS == ["lss_gamma", "lss_gaussian", "lss_nb", "lss_tweedie", "lss_two_piece"]


@pytest.mark.parametrize("name", GLM_RECORDS)
def test_a_model_saved_by_v0_40_0_loads_predicts_and_refits(name: str) -> None:
    """Predictions are v0.40.0's within two evaluations' rounding
    (``assert_predicts_as_saved``), and the loaded model refits the same data
    with the fit method that made it."""
    record = _load(name)
    assert record["version"] == "0.40.0"
    loaded, frame, y = record["model"], record["frame"], record["y"]
    model = getattr(loaded, "_model", loaded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is loaded:
            assert_predicts_as_saved(model, frame, record["prediction"])
        else:  # the sklearn wrapper predicts through its SuperGLM
            np.testing.assert_array_equal(loaded.predict(frame), model.predict(frame))
            assert_predicts_as_saved(model, frame, record["prediction"])
        assert str(model.summary())
        getattr(loaded, record["fit"])(frame, y)
    assert np.all(np.isfinite(np.asarray(loaded.predict(frame), dtype=np.float64)))


def _lss_eta_rounding(model: SuperLSS, frame) -> np.ndarray:
    """Per row and parameter, the bound two float64 evaluations of the linear predictors meet.

    Each predictor is ``b0 + sum_t X_t beta_t`` over its terms, a dot product
    of at most ``p + 1`` terms, so two evaluations differ by at most
    ``2 gamma_{p+1} (|b0| + sum_j |X_ij| |beta_j|)`` (Higham 2002, eq. 3.5),
    the magnitude formed one basis column at a time.
    """
    from superglm._frame import as_eager_frame
    from superglm.distributional.model import _score_feature, _term_indices

    fitted = model._require_fitted()
    rows = as_eager_frame(frame)
    columns = []
    for predictor in fitted.compiled_predictors:
        state = fitted.layout.predictors[predictor.parameter_index]
        local = np.asarray(fitted.result.coefficients[state.coefficient_slice], dtype=np.float64)
        width = int(state.intercept_index is not None)
        slopes = local[width:]
        magnitude = np.full(len(rows), abs(float(local[0])) if width else 0.0)
        assert not predictor.compiled.interaction_order
        for name in predictor.compiled.feature_order:
            indices = np.atleast_1d(
                np.arange(len(slopes))[_term_indices(predictor.compiled.groups, name)]
            )
            spec, values = predictor.compiled.specs[name], rows.column_array(name)
            for position, j in enumerate(indices):
                unit = np.zeros(len(indices))
                unit[position] = 1.0
                contribution = np.asarray(_score_feature(spec, values, unit), dtype=np.float64)
                magnitude += np.abs(contribution).reshape(-1) * abs(slopes[j])
        columns.append(2.0 * _gamma(local.size + 1) * magnitude)
    return np.column_stack(columns)


@pytest.mark.parametrize("name", LSS_RECORDS)
def test_a_superlss_model_saved_by_v0_40_0_loads_and_predicts(name: str) -> None:
    """``from_bytes`` restores v0.40.0's coefficients bit for bit, and the linear
    predictors are v0.40.0's within two evaluations' rounding."""
    record = _load(name)
    assert record["version"] == "0.40.0"
    model = SuperLSS.from_bytes(record["serialized"])
    frame = record["frame"]
    restored = dict(model.coef_)
    assert list(restored) == list(record["coefficients"])
    assert all(
        math.copysign(1.0, restored[key]) == math.copysign(1.0, value) and restored[key] == value
        for key, value in record["coefficients"].items()
    )
    eta = model.predict_link(frame).to_numpy(dtype=np.float64)
    assert eta.shape == record["eta"].shape
    excess = np.abs(eta - record["eta"]) / _lss_eta_rounding(model, frame)
    assert np.all(excess <= 1.0), float(np.max(excess))
    assert np.all(np.isfinite(np.asarray(model.predict(frame), dtype=np.float64)))


RELEASE_OUTCOMES = release_differential.load(FIXTURES.parent / "release_differential_v0_40_0.jsonl")


def test_the_pinned_outcomes_are_the_subset_and_all_converged_on_v0_40_0() -> None:
    assert list(RELEASE_OUTCOMES) == list(release_differential.SUBSET)
    assert all(release_differential.sensible(record) for record in RELEASE_OUTCOMES.values())


@pytest.mark.slow
@pytest.mark.parametrize("name", release_differential.SUBSET)
def test_a_fit_v0_40_0_converged_still_converges_no_worse(name: str) -> None:
    """The fit is not refused, converges, and its objective (where the criterion
    is unchanged), held-out NLL, deviance and Gini are no worse than v0.40.0's
    beyond the harness's tolerances (``release_differential.flags``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = {"status": "ok", **release_differential.run_case(name)}
    assert release_differential.flags(name, RELEASE_OUTCOMES[name], record) == []
