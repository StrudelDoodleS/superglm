"""Tests for SuperGLMClassifier sklearn contract."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from superglm.sklearn import SuperGLMClassifier


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.choice(["A", "B"], n)
    eta = -0.5 + 0.7 * x1 + (x2 == "B") * 0.4
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


class TestClassifierContract:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        result = clf.fit(X, y)
        assert result is clf

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        assert_allclose(clf.classes_, [0, 1])

    def test_predict_returns_binary(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.shape == y.shape

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_decision_function_shape(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (len(y),)
        # Decision function is log-odds, should span negative and positive
        assert df.min() < 0
        assert df.max() > 0

    def test_predict_proba_consistent_with_predict(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        preds = clf.predict(X)
        expected = (proba[:, 1] >= 0.5).astype(int)
        assert_allclose(preds, expected)

    def test_get_params(self, binary_data):
        clf = SuperGLMClassifier(lambda1=0.1, lambda2=0.5)
        params = clf.get_params()
        assert params["lambda1"] == 0.1
        assert params["lambda2"] == 0.5

    def test_set_params(self, binary_data):
        clf = SuperGLMClassifier()
        clf.set_params(lambda1=0.05)
        assert clf.lambda1 == 0.05

    def test_feature_names_in(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        assert hasattr(clf, "feature_names_in_")
        assert list(clf.feature_names_in_) == ["x1", "x2"]

    def test_n_features_in(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        assert clf.n_features_in_ == 2


class TestClassifierWithSampleWeight:
    def test_fit_with_weights(self, binary_data):
        X, y = binary_data
        weights = np.ones(len(y))
        weights[:100] = 2.0
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y, sample_weight=weights)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)


class TestClassifierThreshold:
    def test_custom_threshold(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0, threshold=0.8)
        clf.fit(X, y)
        preds = clf.predict(X)
        # With high threshold, fewer positives
        proba = clf.predict_proba(X)
        expected = (proba[:, 1] >= 0.8).astype(int)
        assert_allclose(preds, expected)


class TestClassifierEdgeCases:
    def test_one_class_raises(self, binary_data):
        X, _ = binary_data
        y_all_zero = np.zeros(len(X))
        clf = SuperGLMClassifier(lambda1=0)
        with pytest.raises(ValueError, match="requires both classes"):
            clf.fit(X, y_all_zero)

    def test_classes_from_data(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X, y)
        assert_allclose(clf.classes_, [0, 1])


class TestClassifierPenaltyAliases:
    """Alias coverage for SuperGLMClassifier."""

    def test_new_names(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(selection_penalty=0.05, spline_penalty=0.2)
        clf.fit(X, y)
        assert clf._model.penalty.lambda1 == 0.05
        assert clf._model.lambda2 == 0.2

    def test_old_names(self, binary_data):
        X, y = binary_data
        clf = SuperGLMClassifier(lambda1=0.05, lambda2=0.2)
        clf.fit(X, y)
        assert clf._model.penalty.lambda1 == 0.05
        assert clf._model.lambda2 == 0.2

    def test_selection_penalty_lambda1_conflict(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="selection_penalty.*lambda1"):
            clf = SuperGLMClassifier(selection_penalty=0.05, lambda1=0.03)
            clf.fit(X, y)

    def test_spline_penalty_lambda2_conflict(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="spline_penalty.*lambda2"):
            clf = SuperGLMClassifier(spline_penalty=0.2, lambda2=0.7)
            clf.fit(X, y)


class TestClassifierNdarrayInput:
    def test_ndarray_fit_predict(self, binary_data):
        X, y = binary_data
        X_arr = X[["x1"]].values  # numeric only
        clf = SuperGLMClassifier(lambda1=0)
        clf.fit(X_arr, y)
        preds = clf.predict(X_arr)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_ndarray_with_feature_names(self, binary_data):
        X, y = binary_data
        X_arr = X[["x1"]].values
        clf = SuperGLMClassifier(lambda1=0, feature_names=["signal"])
        clf.fit(X_arr, y)
        assert list(clf.feature_names_in_) == ["signal"]

    def test_penalty_default_is_none(self):
        clf = SuperGLMClassifier()
        assert clf.penalty is None
