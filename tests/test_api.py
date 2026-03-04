"""Tests for the new SuperGLM constructor API."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, GroupLasso, SparseGroupLasso, Ridge, Spline, Categorical, Numeric


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


class TestFeaturesDict:
    """Test explicit features={...} constructor."""

    def test_basic_fit_predict(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_features_registered_at_init(self):
        model = SuperGLM(
            features={
                "a": Numeric(),
                "b": Numeric(),
            },
        )
        assert list(model._specs.keys()) == ["a", "b"]
        assert model._feature_order == ["a", "b"]

    def test_add_feature_extends_dict(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.add_feature("density", Numeric())
        model.fit(X, y, exposure=exposure)
        assert "density" in model._specs


class TestSplinesAutoDetect:
    """Test splines=[...] auto-detect constructor."""

    def test_basic_fit_predict(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_auto_detects_types(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            lambda1=0.01,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, exposure=exposure)
        assert isinstance(model._specs["age"], Spline)
        assert isinstance(model._specs["region"], Categorical)
        assert isinstance(model._specs["density"], Numeric)

    def test_empty_splines_all_auto(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(lambda1=0.01, splines=[])
        model.fit(X, y)
        # age and density are numeric, region is categorical
        assert isinstance(model._specs["region"], Categorical)
        assert isinstance(model._specs["age"], Numeric)
        assert isinstance(model._specs["density"], Numeric)

    def test_knots_per_feature(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(
            lambda1=0.01,
            splines=["age", "density"],
            n_knots=[10, 20],
        )
        model.fit(X, y)
        assert model._specs["age"].n_knots == 10
        assert model._specs["density"].n_knots == 20

    def test_knots_broadcast_int(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(
            lambda1=0.01,
            splines=["age", "density"],
            n_knots=12,
        )
        model.fit(X, y)
        assert model._specs["age"].n_knots == 12
        assert model._specs["density"].n_knots == 12

    def test_knots_length_mismatch(self):
        with pytest.raises(ValueError, match="n_knots has length"):
            model = SuperGLM(splines=["a", "b"], n_knots=[10])
            model.fit(
                pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                np.array([1, 2, 3]),
            )

    def test_categorical_base_no_exposure(self, sample_data):
        """Without exposure, most_exposed falls back to first."""
        X, y, _ = sample_data
        model = SuperGLM(lambda1=0.01, splines=[])
        model.fit(X, y)
        # Should have used base="first" since no exposure
        assert model._specs["region"].base == "first"


class TestMutualExclusivity:
    def test_features_and_splines_raises(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            SuperGLM(features={"a": Numeric()}, splines=["b"])


class TestPenaltyResolution:
    def test_string_group_lasso(self):
        model = SuperGLM(penalty="group_lasso", lambda1=0.05)
        assert isinstance(model.penalty, GroupLasso)
        assert model.penalty.lambda1 == 0.05

    def test_string_sparse_group_lasso(self):
        model = SuperGLM(penalty="sparse_group_lasso", lambda1=0.05)
        assert isinstance(model.penalty, SparseGroupLasso)

    def test_string_ridge(self):
        model = SuperGLM(penalty="ridge", lambda1=0.05)
        assert isinstance(model.penalty, Ridge)

    def test_none_defaults_to_group_lasso(self):
        model = SuperGLM(penalty=None)
        assert isinstance(model.penalty, GroupLasso)

    def test_object_passed_through(self):
        p = GroupLasso(lambda1=0.02)
        model = SuperGLM(penalty=p)
        assert model.penalty is p

    def test_object_with_lambda1_raises(self):
        with pytest.raises(ValueError, match="Cannot set 'lambda1'"):
            SuperGLM(penalty=GroupLasso(lambda1=0.01), lambda1=0.05)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty"):
            SuperGLM(penalty="lasso")

    def test_auto_calibrate(self, sample_data):
        """lambda1=None should auto-calibrate at fit time."""
        X, y, exposure = sample_data
        model = SuperGLM(
            lambda1=None,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, exposure=exposure)
        assert model.penalty.lambda1 is not None
        assert model.penalty.lambda1 > 0


class TestBackwardCompat:
    """Ensure legacy add_feature() usage still works."""

    def test_add_feature_only(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(penalty=GroupLasso(lambda1=0.01))
        model.add_feature("age", Spline(n_knots=10, penalty="ssp"))
        model.add_feature("region", Categorical(base="first"))
        model.add_feature("density", Numeric())
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_duplicate_feature_raises(self):
        model = SuperGLM()
        model.add_feature("x", Numeric())
        with pytest.raises(ValueError, match="Feature already added"):
            model.add_feature("x", Numeric())
