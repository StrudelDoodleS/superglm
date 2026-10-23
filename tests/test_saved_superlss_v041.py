"""A ``SuperLSS`` that v0.41.0 saved with ``joblib.dump`` still loads and predicts.

The fixtures in ``fixtures/saved_v0_41_0`` were written by the v0.41.0 release
(commit 7b1732b7) with ``scripts/make_saved_v041_lss_fixtures.py``. Loading one
runs the restore hooks the pickle names (``_restore_superlss`` and
``_restore_mapping_proxy``), so renaming or moving either one fails here
rather than in a user's saved file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

import superglm as sg
from superglm._frame import as_eager_frame
from superglm.distributional.model import _score_feature, _term_indices

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_41_0"
CASES = ["lss_gaussian", "lss_gamma"]
_U = np.finfo(np.float64).eps / 2.0
# Roundings a basis column may differ by on another machine, counted as extra terms.
_BASIS_ULPS = 8


def _gamma(count: int) -> float:
    """Higham's accumulation bound gamma_k for float64."""
    return count * _U / (1.0 - count * _U)


def _eta_rounding(model: sg.SuperLSS, frame) -> np.ndarray:
    """Per row and parameter, the bound two float64 evaluations of the linear predictors meet.

    Each predictor is ``b0 + sum_t X_t beta_t`` over its terms, a dot product
    of at most ``p + 1`` terms, so two evaluations differ by at most
    ``2 gamma_{p+1} (|b0| + sum_j |X_ij| |beta_j|)`` (Higham 2002, eq. 3.5),
    the magnitude formed one basis column at a time. A basis function may also
    round differently on another machine (``sin``, ``exp``), which the
    ``_BASIS_ULPS`` term covers.
    """
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
        columns.append(2.0 * _gamma(local.size + 1 + _BASIS_ULPS) * magnitude)
    return np.column_stack(columns)


def _mean_rounding(prediction: np.ndarray, eta_rounding: np.ndarray) -> np.ndarray:
    """The bound on a conditional mean built from the first linear predictor.

    The mean is the inverse link of that predictor: the identity or ``exp``.
    A change ``d`` in eta moves the identity mean by ``d`` and the ``exp`` mean
    by ``expm1(d) |mean|``, and ``exp`` itself rounds within a few units of
    roundoff. The sum bounds either link.
    """
    shift = eta_rounding[:, 0]
    return shift + (np.expm1(shift) + 8.0 * _U) * np.abs(prediction)


@pytest.mark.parametrize("name", CASES)
def test_a_superlss_saved_by_v0_41_0_loads_predicts_and_reads_its_fit(name: str) -> None:
    reference = joblib.load(FIXTURES / f"{name}.reference.joblib.xz")
    assert reference["version"] == "0.41.0"
    model = joblib.load(FIXTURES / f"{name}.model.joblib.xz")
    assert isinstance(model, sg.SuperLSS)
    frame = reference["frame"]

    eta = model.predict_link(frame).to_numpy(dtype=np.float64)
    assert eta.shape == reference["eta"].shape
    rounding = _eta_rounding(model, frame)
    assert np.all(np.abs(eta - reference["eta"]) <= rounding), float(
        np.max(np.abs(eta - reference["eta"]) / rounding)
    )
    mean = np.asarray(model.predict(frame), dtype=np.float64)
    saved = reference["prediction"]
    assert np.all(np.abs(mean - saved) <= _mean_rounding(saved, rounding))

    # A restored model carries no training frame; the readers that need the
    # fitted state take it through X_train=.
    assert len(model.summary(X_train=frame)) > 0
    assert str(model.diagnose())
