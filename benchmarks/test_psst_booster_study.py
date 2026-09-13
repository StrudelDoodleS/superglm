"""Check categorical meaning, selection and prediction scale at real backends."""

import numpy as np
import pandas as pd
import pytest
from psst_booster_study import category_frames, fit_candidate, select_candidate


def test_category_vocabulary_comes_only_from_training():
    train = pd.DataFrame({"x": [0.0, 1.0], "c00": ["b", "a"]})
    valid = pd.DataFrame({"x": [2.0], "c00": ["z"]})
    frames = category_frames(train, valid)
    assert list(frames[0]["c00"].cat.categories) == ["a", "b"]
    assert pd.isna(frames[1]["c00"].iloc[0])


def test_validation_selects_depth_without_test_information():
    candidates = {
        "constant": {"validation_loss": 4.0, "test_loss": 0.0},
        "depth2": {"validation_loss": 2.0, "test_loss": 9.0},
        "depth4": {"validation_loss": 3.0, "test_loss": 1.0},
        "depth6": {"error": "failed fit"},
    }
    assert select_candidate(candidates) == "depth2"


def data():
    rng = np.random.default_rng(99)
    x = np.linspace(-1, 1, 160)
    frame = pd.DataFrame({"x": x, "c00": np.where(np.arange(len(x)) % 2, "a", "b")})
    return frame, rng


@pytest.mark.parametrize("backend", ["xgboost", "catboost", "lightgbm"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_prediction_scale_matches_backend_response_mean(backend, family):
    frame, rng = data()
    y = 2 * frame["x"].to_numpy() if family == "gaussian" else rng.poisson(np.exp(frame["x"]))
    train, valid = category_frames(frame, frame)
    fitted = fit_candidate(backend, family, train, y, valid, y, depth=2, seed=9, rounds=12)
    mean = fitted.predict(valid)
    if backend == "xgboost":
        import xgboost as xgb

        direct = fitted.model.predict(
            xgb.DMatrix(valid, enable_categorical=True), iteration_range=(0, fitted.best_rounds)
        )
    elif backend == "lightgbm":
        direct = fitted.model.predict(valid, num_iteration=fitted.best_rounds)
    else:
        direct = fitted.model.predict(valid, ntree_end=fitted.best_rounds)
    np.testing.assert_allclose(mean, direct, rtol=16 * np.finfo(np.float32).eps, atol=0)


def test_xgboost_uses_best_iteration_not_patience_trees():
    frame, _ = data()
    train, valid = category_frames(frame, frame)
    y = frame["x"].to_numpy()
    fitted = fit_candidate("xgboost", "gaussian", train, y, valid, -y, depth=2, seed=9, rounds=12)
    assert fitted.best_rounds < fitted.trained_rounds
    import xgboost as xgb

    matrix = xgb.DMatrix(valid, enable_categorical=True)
    first = fitted.model.predict(matrix, iteration_range=(0, 1))
    all_trees = fitted.model.predict(matrix)
    np.testing.assert_array_equal(fitted.predict(valid), first)
    assert np.max(np.abs(first - all_trees)) > 0.01


@pytest.mark.parametrize("backend", ["lightgbm", "catboost"])
@pytest.mark.parametrize(
    "target_scale", [pytest.param(-1.0, id="first"), pytest.param(0.3, id="interior")]
)
def test_booster_iteration_selection_matches_retained_model(backend, target_scale):
    frame, _ = data()
    train, valid = category_frames(frame, frame)
    y = frame["x"].to_numpy()
    fitted = fit_candidate(
        backend, "gaussian", train, y, valid, target_scale * y, depth=2, seed=9, rounds=60
    )
    # Both libraries retain only their selected trees with these fit settings.
    # Count the actual trees, independently of the adapter's best_rounds value.
    if backend == "lightgbm":
        retained = fitted.model.current_iteration()
    else:
        retained = fitted.model.tree_count_
    if target_scale < 0:
        assert retained == 1
    else:
        assert retained > 1
    assert retained < fitted.trained_rounds
    assert fitted.best_rounds == retained
    np.testing.assert_array_equal(fitted.predict(valid), fitted.model.predict(valid))
