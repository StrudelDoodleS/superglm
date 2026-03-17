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
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, selection_penalty=0.01)
        model.fit(X, y, sample_weight=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_sklearn_attributes(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, selection_penalty=0.01)
        model.fit(X, y, sample_weight=exposure)
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert model.n_features_in_ == 3
        assert list(model.feature_names_in_) == ["age", "region", "density"]


class TestAutoDetect:
    def test_categorical_from_dtype(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(selection_penalty=0.01)
        model.fit(X, y)
        assert "Categorical" in model._feature_types["region"]

    def test_numeric_default(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(selection_penalty=0.01)
        model.fit(X, y)
        assert "Numeric" in model._feature_types["density"]
        assert "Numeric" in model._feature_types["age"]

    def test_spline_override(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(spline_features=["age"], n_knots=10, selection_penalty=0.01)
        model.fit(X, y)
        assert "Spline" in model._feature_types["age"]


class TestNKnots:
    def test_int_broadcast(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(
            spline_features=["age", "density"], n_knots=12, selection_penalty=0.01
        )
        model.fit(X, y)
        assert "n_knots=12" in model._feature_types["age"]
        assert "n_knots=12" in model._feature_types["density"]

    def test_list_per_feature(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(
            spline_features=["age", "density"],
            n_knots=[10, 20],
            selection_penalty=0.01,
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
            selection_penalty=0.01,
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
            selection_penalty=0.01,
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
        model = SuperGLMRegressor(selection_penalty=0.01)
        model.fit(X, y, sample_weight=exposure)
        assert model.coef_ is not None

    def test_no_sample_weight(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(selection_penalty=0.01)
        model.fit(X, y)
        assert model.coef_ is not None


# ── ndarray input ─────────────────────────────────────────────────


class TestNdarrayInput:
    @pytest.fixture
    def array_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        y = rng.poisson(np.exp(0.5 * X[:, 0])).astype(float)
        return X, y

    def test_ndarray_with_feature_names(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(
            selection_penalty=0.01,
            feature_names=["a", "b", "c"],
            spline_features=["a"],
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert list(m.feature_names_in_) == ["a", "b", "c"]

    def test_ndarray_synthetic_names(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(selection_penalty=0.01)
        m.fit(X, y)
        assert list(m.feature_names_in_) == ["x0", "x1", "x2"]

    def test_ndarray_integer_spline_features(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(selection_penalty=0.01, spline_features=[0])
        m.fit(X, y)
        assert "Spline" in m._feature_types["x0"]
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_ndarray_string_ref_without_feature_names_raises(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(spline_features=["age"])
        with pytest.raises(ValueError, match="string ref.*ndarray"):
            m.fit(X, y)

    def test_ndarray_explicit_categorical_numeric(self):
        rng = np.random.default_rng(42)
        n = 200
        # col 0: numeric, col 1: category-like ints, col 2: numeric
        X = np.column_stack(
            [
                rng.standard_normal(n),
                rng.choice([1, 2, 3], n),
                rng.standard_normal(n),
            ]
        )
        y = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
        m = SuperGLMRegressor(
            feature_names=["x", "cat", "z"],
            categorical_features=["cat"],
            numeric_features=["x"],
            # z is unspecified → defaults to numeric
        )
        m.fit(X, y)
        assert "Categorical" in m._feature_types["cat"]
        assert "Numeric" in m._feature_types["x"]
        assert "Numeric" in m._feature_types["z"]

    def test_named_ndarray_unspecified_stays_numeric(self):
        """Regression: named ndarray with categorical_features must not
        auto-detect unspecified columns from DataFrame dtype.

        When _normalize_X converts an ndarray to a DataFrame, the resulting
        dtypes are artefacts of the conversion (e.g. all-float → float64,
        mixed-type → object).  Only real DataFrames should use dtype
        inference for unspecified columns.
        """
        rng = np.random.default_rng(42)
        n = 200
        # All float — col 1 is integer-coded categorical
        X = np.column_stack(
            [
                rng.standard_normal(n),
                rng.choice([1.0, 2.0, 3.0], n),
                rng.standard_normal(n),
            ]
        )
        y = rng.poisson(1.0, n).astype(float)
        m = SuperGLMRegressor(
            feature_names=["x", "cat", "z"],
            categorical_features=["cat"],
            selection_penalty=0.01,
        )
        m.fit(X, y)
        assert "Categorical" in m._feature_types["cat"]
        # z is unspecified — must default to Numeric (ndarray contract)
        assert "Numeric" in m._feature_types["x"]
        assert "Numeric" in m._feature_types["z"]

    def test_ndarray_offset_by_index(self):
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack(
            [
                rng.standard_normal(n),
                rng.standard_normal(n),
                rng.uniform(0.5, 1.5, n),  # offset col
            ]
        )
        y = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
        m = SuperGLMRegressor(selection_penalty=0.01, offset=2)
        m.fit(X, y)
        assert m.n_features_in_ == 2  # offset excluded

    def test_ndarray_predict_shape_matches(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(selection_penalty=0.01)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == y.shape

    def test_feature_names_length_mismatch_raises(self, array_data):
        X, y = array_data
        m = SuperGLMRegressor(feature_names=["a", "b"])
        with pytest.raises(ValueError, match="feature_names has 2.*3 columns"):
            m.fit(X, y)


# ── Penalty default behaviour ─────────────────────────────────────


class TestPenaltyDefault:
    def test_default_penalty_is_none(self):
        m = SuperGLMRegressor()
        assert m.penalty is None

    def test_no_penalty_fits(self, sample_data):
        """penalty=None with no selection_penalty → unpenalised fit."""
        X, y, w = sample_data
        m = SuperGLMRegressor()
        m.fit(X, y, sample_weight=w)
        assert m.coef_ is not None

    def test_selection_penalty_auto_upgrades(self, sample_data):
        """Positive selection_penalty with penalty=None → group_lasso."""
        X, y, w = sample_data
        m = SuperGLMRegressor(selection_penalty=0.05)
        m.fit(X, y, sample_weight=w)
        assert m._model.penalty.lambda1 == 0.05

    def test_explicit_penalty_respected(self, sample_data):
        X, y, w = sample_data
        m = SuperGLMRegressor(penalty="ridge", selection_penalty=0.05)
        m.fit(X, y, sample_weight=w)
        assert type(m._model.penalty).__name__ == "Ridge"


# ── Family validation ────────────────────────────────────────────


class TestRegressorFamilyGuard:
    def test_binomial_raises(self, sample_data):
        X, y, _ = sample_data
        m = SuperGLMRegressor(family="binomial")
        with pytest.raises(ValueError, match="use SuperGLMClassifier"):
            m.fit(X, y)

    def test_unknown_family_raises(self, sample_data):
        X, y, _ = sample_data
        m = SuperGLMRegressor(family="laplace")
        with pytest.raises(ValueError, match="Unknown family 'laplace'"):
            m.fit(X, y)

    def test_gaussian_fits(self, sample_data):
        X, y, _ = sample_data
        m = SuperGLMRegressor(family="gaussian")
        m.fit(X, y)
        assert m.coef_ is not None

    def test_clone_binomial_is_constructable(self):
        """clone() should work — error is deferred to fit()."""
        from sklearn.base import clone

        m = SuperGLMRegressor(family="binomial")
        m2 = clone(m)
        assert m2.family == "binomial"


# ── sklearn clone / get_params ────────────────────────────────────


class TestSklearnClone:
    def test_clone(self, sample_data):
        from sklearn.base import clone

        m = SuperGLMRegressor(
            selection_penalty=0.05,
            spline_features=["age"],
            feature_names=None,
        )
        m2 = clone(m)
        params1 = m.get_params()
        params2 = m2.get_params()
        assert params1 == params2

    def test_get_params_includes_new_fields(self):
        m = SuperGLMRegressor(
            categorical_features=["a"],
            numeric_features=["b"],
            feature_names=["a", "b", "c"],
        )
        p = m.get_params()
        assert p["categorical_features"] == ["a"]
        assert p["numeric_features"] == ["b"]
        assert p["feature_names"] == ["a", "b", "c"]


# ── Bug regression tests ────────────────────────────────────────


class TestDataFrameDtypeAutoDetect:
    """Regression: specifying categorical_features on a DataFrame must
    not disable dtype auto-detection for unspecified string columns."""

    def test_partial_categorical_preserves_auto_detect(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(
            {
                "x": rng.standard_normal(n),
                "cat1": rng.choice(["A", "B"], n),
                "cat2": rng.choice(["X", "Y", "Z"], n),
            }
        )
        y = rng.poisson(1.0, n).astype(float)
        # Only declare cat1 explicitly — cat2 should still be auto-detected
        m = SuperGLMRegressor(categorical_features=["cat1"], selection_penalty=0.01)
        m.fit(X, y)
        assert "Categorical" in m._feature_types["cat1"]
        assert "Categorical" in m._feature_types["cat2"]
        assert "Numeric" in m._feature_types["x"]

    def test_partial_numeric_preserves_auto_detect(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(
            {
                "x": rng.standard_normal(n),
                "region": rng.choice(["A", "B"], n),
                "z": rng.standard_normal(n),
            }
        )
        y = rng.poisson(1.0, n).astype(float)
        # Only declare x explicitly — region should still auto-detect as categorical
        m = SuperGLMRegressor(numeric_features=["x"], selection_penalty=0.01)
        m.fit(X, y)
        assert "Categorical" in m._feature_types["region"]
        assert "Numeric" in m._feature_types["x"]
        assert "Numeric" in m._feature_types["z"]


class TestSparseMatrixInput:
    """Regression: sparse matrix input should be densified, not crash."""

    def test_sparse_csr_fit_predict(self):
        import scipy.sparse

        rng = np.random.default_rng(42)
        n = 200
        X_dense = rng.standard_normal((n, 3))
        X_sparse = scipy.sparse.csr_matrix(X_dense)
        y = rng.poisson(np.exp(0.3 * X_dense[:, 0])).astype(float)
        m = SuperGLMRegressor(selection_penalty=0.01)
        m.fit(X_sparse, y)
        preds = m.predict(X_sparse)
        assert preds.shape == (n,)

    def test_sparse_csc_fit_predict(self):
        import scipy.sparse

        rng = np.random.default_rng(42)
        n = 200
        X_dense = rng.standard_normal((n, 3))
        X_sparse = scipy.sparse.csc_matrix(X_dense)
        y = rng.poisson(np.exp(0.3 * X_dense[:, 0])).astype(float)
        m = SuperGLMRegressor(selection_penalty=0.01)
        m.fit(X_sparse, y)
        preds = m.predict(X_sparse)
        assert preds.shape == (n,)
