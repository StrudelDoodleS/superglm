"""The checks behind exporting an edited SuperLSS as joblib."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.api import SuperLSS
from superglm.distributional.terms import _compiled_predictor
from superglm.editor.persistence import _slice_rows, _spread_row_indices


def _term_keys(model: SuperLSS) -> dict[str, tuple[str, ...]]:
    fitted = model._require_fitted()
    keys: dict[str, tuple[str, ...]] = {}
    for parameter in model.parameter_names_:
        compiled = _compiled_predictor(fitted, parameter).compiled
        keys[parameter] = (*compiled.feature_order, *compiled.interaction_order)
    return keys


def validate_lss_artifact(model: SuperLSS, loaded: object) -> None:
    """Raise ``ValueError`` unless ``loaded`` is ``model`` as a joblib round trip leaves it.

    The type, the parameters, each predictor's term keys, the shape and finiteness
    of the coefficients and the hand-edited terms must all agree.
    """
    try:
        if type(loaded) is not type(model):
            raise ValueError("loaded object has the wrong model type")
        if tuple(loaded.parameter_names_) != tuple(model.parameter_names_):
            raise ValueError("loaded model parameters differ")
        if _term_keys(loaded) != _term_keys(model):
            raise ValueError("loaded model terms differ")
        original = np.fromiter(model.coef_.values(), dtype=np.float64)
        restored = np.fromiter(loaded.coef_.values(), dtype=np.float64)
        if restored.shape != original.shape:
            raise ValueError("loaded model coefficient shape differs")
        if not np.all(np.isfinite(restored)):
            raise ValueError("loaded model coefficients are non-finite")
        if tuple(loaded._hand_edited_terms) != tuple(model._hand_edited_terms):
            raise ValueError("loaded model hand-edited terms differ")
    except ValueError as exc:
        raise ValueError(f"artifact validation failed: {exc}") from exc
    except Exception as exc:
        raise ValueError("artifact validation failed: invalid fitted model contract") from exc


def lss_prediction_rows(model: SuperLSS, dataset: Any, max_rows: int) -> NDArray[np.float64]:
    """Every parameter's prediction on at most ``max_rows`` rows spread across ``dataset``.

    Rows of zero weight are left out, as scoring leaves them out. The offsets are
    the dataset's, a dict keyed by parameter name, cut to the same rows.
    """
    weights = dataset.sample_weight
    kept = (
        np.arange(dataset.n_obs)
        if weights is None
        else np.flatnonzero(np.asarray(weights, dtype=np.float64) > 0.0)
    )
    indices = kept[_spread_row_indices(kept.size, max_rows)]
    offsets = (
        None
        if dataset.offset is None
        else {name: np.asarray(values)[indices] for name, values in dataset.offset.items()}
    )
    rows = _slice_rows(dataset.X, indices)
    return model.predict_parameters(rows, offsets=offsets).to_numpy(dtype=np.float64)
