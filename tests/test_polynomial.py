"""Tests for the Polynomial feature spec."""

import numpy as np
import pandas as pd
import pytest

from superglm import Polynomial, SuperGLM
from superglm.features.polynomial import Polynomial as PolynomialDirect


class TestPolynomialSpec:
    def test_group_size_equals_degree(self):
        for deg in [1, 2, 3, 4]:
            info = PolynomialDirect(degree=deg).build(np.linspace(0, 100, 200))
            assert info.n_cols == deg

    def test_no_penalty_matrix(self):
        info = PolynomialDirect(degree=3).build(np.linspace(0, 1, 50))
        assert info.penalty_matrix is None
        assert info.reparametrize is False

    def test_columns_shape(self):
        x = np.linspace(0, 100, 300)
        info = PolynomialDirect(degree=3).build(x)
        assert info.columns.shape == (300, 3)

    def test_columns_near_orthogonal(self):
        """Legendre polynomials on uniformly spaced data should be nearly orthogonal."""
        x = np.linspace(0, 100, 1000)
        info = PolynomialDirect(degree=3).build(x)
        G = info.columns.T @ info.columns
        # Off-diagonal elements should be much smaller than diagonal
        diag = np.diag(G)
        off_diag = G - np.diag(diag)
        assert np.max(np.abs(off_diag)) < 0.1 * np.min(diag)

    def test_degree_zero_raises(self):
        with pytest.raises(ValueError, match="degree must be >= 1"):
            PolynomialDirect(degree=0)

    def test_transform_same_as_build(self):
        x = np.linspace(0, 50, 100)
        spec = PolynomialDirect(degree=2)
        info = spec.build(x)
        transformed = spec.transform(x)
        np.testing.assert_allclose(info.columns, transformed)

    def test_transform_new_data(self):
        spec = PolynomialDirect(degree=2)
        spec.build(np.linspace(0, 100, 200))
        new_x = np.array([25.0, 50.0, 75.0])
        result = spec.transform(new_x)
        assert result.shape == (3, 2)

    def test_reconstruct_keys(self):
        spec = PolynomialDirect(degree=3)
        spec.build(np.linspace(0, 100, 200))
        rec = spec.reconstruct(np.array([0.1, -0.05, 0.01]))
        assert "x" in rec
        assert "log_relativity" in rec
        assert "relativity" in rec
        assert "degree" in rec
        assert "coefficients" in rec
        assert rec["degree"] == 3

    def test_reconstruct_curve_shape(self):
        spec = PolynomialDirect(degree=2)
        spec.build(np.linspace(0, 100, 200))
        rec = spec.reconstruct(np.array([0.0, 0.1]), n_points=50)
        assert rec["x"].shape == (50,)
        assert rec["relativity"].shape == (50,)
        assert np.all(rec["relativity"] > 0)

    def test_constant_feature_no_crash(self):
        """All-same values should not crash (span ≈ 0)."""
        spec = PolynomialDirect(degree=2)
        info = spec.build(np.full(100, 5.0))
        assert info.columns.shape == (100, 2)


class TestPolynomialIntegration:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n = 500
        age = rng.uniform(18, 85, n)
        region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
        exposure = rng.uniform(0.3, 1.0, n)
        mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
        y = rng.poisson(mu * exposure).astype(float)
        X = pd.DataFrame({"age": age, "region": region})
        return X, y, exposure

    def test_fit_predict(self, sample_data):
        X, y, exposure = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Polynomial(degree=3),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_reconstruct_feature(self, sample_data):
        X, y, exposure = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Polynomial(degree=2),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        rec = model.reconstruct_feature("age")
        assert "x" in rec
        assert "relativity" in rec

    def test_summary_shows_polynomial(self, sample_data):
        X, y, exposure = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Polynomial(degree=2),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        s = model.summary()
        assert "age" in s
        assert s["age"]["n_params"] == 2
