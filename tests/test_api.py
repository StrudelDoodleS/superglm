"""Tests for the new SuperGLM constructor API."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BasisSpline,
    Categorical,
    GroupElasticNet,
    GroupLasso,
    Numeric,
    Ridge,
    SparseGroupLasso,
    Spline,
    SuperGLM,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    age = rng.uniform(18, 85, n)
    region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
    density = rng.normal(5, 2, n)
    sample_weight = rng.uniform(0.3, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age": age, "region": region, "density": density})
    return X, y, sample_weight


class TestFeaturesDict:
    """Test explicit features={...} constructor."""

    def test_basic_fit_predict(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_predict_repeated_calls_match_dataframe_copy(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)

        pred1 = model.predict(X)
        pred2 = model.predict(X)
        pred3 = model.predict(X.copy())

        np.testing.assert_allclose(pred1, pred2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pred1, pred3, rtol=1e-12, atol=1e-12)

    def test_features_registered_at_init(self):
        model = SuperGLM(
            features={
                "a": Numeric(),
                "b": Numeric(),
            },
        )
        assert list(model._specs.keys()) == ["a", "b"]
        assert model._feature_order == ["a", "b"]


class TestSplinesAutoDetect:
    """Test splines=[...] auto-detect constructor."""

    def test_basic_fit_predict(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_auto_detects_types(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            selection_penalty=0.01,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert isinstance(model._specs["age"], BasisSpline)
        assert isinstance(model._specs["region"], Categorical)
        assert isinstance(model._specs["density"], Numeric)

    def test_empty_splines_all_auto(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(selection_penalty=0.01, splines=[])
        model.fit(X, y)
        # age and density are numeric, region is categorical
        assert isinstance(model._specs["region"], Categorical)
        assert isinstance(model._specs["age"], Numeric)
        assert isinstance(model._specs["density"], Numeric)

    def test_knots_per_feature(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(
            selection_penalty=0.01,
            splines=["age", "density"],
            n_knots=[10, 20],
        )
        model.fit(X, y)
        assert model._specs["age"].n_knots == 10
        assert model._specs["density"].n_knots == 20

    def test_knots_broadcast_int(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLM(
            selection_penalty=0.01,
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
        """Without sample_weight, most_exposed falls back to first."""
        X, y, _ = sample_data
        model = SuperGLM(selection_penalty=0.01, splines=[])
        model.fit(X, y)
        # Should have used base="first" since no sample_weight
        assert model._specs["region"].base == "first"


class TestMutualExclusivity:
    def test_features_and_splines_raises(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            SuperGLM(features={"a": Numeric()}, splines=["b"])


class TestPenaltyResolution:
    def test_string_group_lasso(self):
        model = SuperGLM(penalty="group_lasso", selection_penalty=0.05)
        assert isinstance(model.penalty, GroupLasso)
        assert model.penalty.lambda1 == 0.05

    def test_string_sparse_group_lasso(self):
        model = SuperGLM(penalty="sparse_group_lasso", selection_penalty=0.05)
        assert isinstance(model.penalty, SparseGroupLasso)

    def test_string_ridge(self):
        model = SuperGLM(penalty="ridge", selection_penalty=0.05)
        assert isinstance(model.penalty, Ridge)

    def test_string_group_elastic_net(self):
        model = SuperGLM(penalty="group_elastic_net", selection_penalty=0.05)
        assert isinstance(model.penalty, GroupElasticNet)
        assert model.penalty.lambda1 == 0.05
        assert model.penalty.alpha == 0.5

    def test_none_defaults_to_group_lasso(self):
        model = SuperGLM(penalty=None)
        assert isinstance(model.penalty, GroupLasso)

    def test_object_passed_through(self):
        p = GroupLasso(lambda1=0.02)
        model = SuperGLM(penalty=p)
        assert model.penalty is p

    def test_object_with_selection_penalty_raises(self):
        with pytest.raises(ValueError, match="Cannot set 'selection_penalty'"):
            SuperGLM(penalty=GroupLasso(lambda1=0.01), selection_penalty=0.05)

    def test_object_with_penalty_features_raises(self):
        with pytest.raises(ValueError, match="Cannot set 'penalty_features'"):
            SuperGLM(penalty=GroupLasso(lambda1=0.01), penalty_features=["region"])

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty"):
            SuperGLM(penalty="lasso")

    def test_auto_calibrate(self, sample_data):
        """selection_penalty=None should auto-calibrate at fit time."""
        X, y, sample_weight = sample_data
        model = SuperGLM(
            selection_penalty=None,
            splines=["age"],
            n_knots=10,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert model.penalty.lambda1 is not None
        assert model.penalty.lambda1 > 0

    def test_string_penalty_features(self):
        model = SuperGLM(penalty="group_lasso", selection_penalty=0.05, penalty_features=["region"])
        assert isinstance(model.penalty, GroupLasso)
        assert model.penalty.features == frozenset({"region"})

    def test_unknown_penalty_feature_raises_at_fit(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            penalty_features=["missing"],
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(),
                "density": Numeric(),
            },
        )
        with pytest.raises(ValueError, match="Unknown penalty feature/group filter"):
            model.fit(X, y, sample_weight=sample_weight)
