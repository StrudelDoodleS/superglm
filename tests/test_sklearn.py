"""Tests for the sklearn-compatible SuperGLMRegressor wrapper."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from superglm.sklearn import SuperGLMRegressor


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    age = rng.uniform(18, 85, n)
    region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
    density = rng.normal(5, 2, n)
    exposure = rng.uniform(0.3, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"age": age, "region": region, "density": density})
    return X, y, exposure


class TestFitPredict:
    def test_basic(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, lambda1=0.01)
        model.fit(X, y, sample_weight=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_sklearn_attributes(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, lambda1=0.01)
        model.fit(X, y, sample_weight=exposure)
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert model.n_features_in_ == 3
        assert list(model.feature_names_in_) == ["age", "region", "density"]


class TestAutoDetect:
    def test_categorical_from_dtype(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(lambda1=0.01)
        model.fit(X, y)
        assert "Categorical" in model._feature_types["region"]

    def test_numeric_default(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(lambda1=0.01)
        model.fit(X, y)
        assert "Numeric" in model._feature_types["density"]
        assert "Numeric" in model._feature_types["age"]

    def test_spline_override(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, lambda1=0.01)
        model.fit(X, y)
        assert "Spline" in model._feature_types["age"]


class TestNKnots:
    def test_int_broadcast(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(spline_features=["age", "density"], n_knots=12, lambda1=0.01)
        model.fit(X, y)
        assert "n_knots=12" in model._feature_types["age"]
        assert "n_knots=12" in model._feature_types["density"]

    def test_list_per_feature(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(
            spline_features=["age", "density"],
            n_knots=[10, 20],
            lambda1=0.01,
        )
        model.fit(X, y)
        assert "n_knots=10" in model._feature_types["age"]
        assert "n_knots=20" in model._feature_types["density"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_knots has length"):
            model = SuperGLMRegressor(spline_features=["a", "b"], n_knots=[10])
            model.fit(
                pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                np.array([1, 2, 3]),
            )


class TestOffset:
    def test_single_col(self, sample_data):
        X, y, exposure = sample_data
        X = X.copy()
        X["log_exp"] = np.log(exposure)
        model = SuperGLMRegressor(
            spline_features=["age"],
            n_knots=10,
            offset="log_exp",
            lambda1=0.01,
        )
        model.fit(X, y)
        assert "log_exp" not in model._feature_types
        assert model.n_features_in_ == 3
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_multi_col(self, sample_data):
        X, y, exposure = sample_data
        X = X.copy()
        X["off1"] = np.log(exposure) * 0.5
        X["off2"] = np.log(exposure) * 0.5
        model = SuperGLMRegressor(
            spline_features=["age"],
            n_knots=10,
            offset=["off1", "off2"],
            lambda1=0.01,
        )
        model.fit(X, y)
        assert "off1" not in model._feature_types
        assert "off2" not in model._feature_types
        assert model.n_features_in_ == 3

    def test_missing_offset_raises(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(offset="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            model.fit(X, y)


class TestSklearnContract:
    def test_get_set_params(self):
        model = SuperGLMRegressor(family="gamma", n_knots=20)
        params = model.get_params()
        assert params["family"] == "gamma"
        assert params["n_knots"] == 20
        model.set_params(family="poisson")
        assert model.family == "poisson"

    def test_check_is_fitted_before_fit(self):
        model = SuperGLMRegressor()
        with pytest.raises(Exception):
            check_is_fitted(model)

    def test_sample_weight(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLMRegressor(lambda1=0.01)
        model.fit(X, y, sample_weight=exposure)
        assert model.coef_ is not None

    def test_no_sample_weight(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(lambda1=0.01)
        model.fit(X, y)
        assert model.coef_ is not None
