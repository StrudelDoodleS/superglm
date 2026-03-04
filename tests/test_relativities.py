"""Tests for relativities extraction and plotting."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, Spline, Categorical, Numeric, plot_relativities


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


@pytest.fixture
def fitted_model(sample_data):
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
    return model


class TestRelativities:
    def test_all_features_present(self, fitted_model):
        rels = fitted_model.relativities()
        assert set(rels.keys()) == {"age", "region", "density"}

    def test_spline_schema(self, fitted_model):
        df = fitted_model.relativities()["age"]
        assert list(df.columns) == ["x", "relativity", "log_relativity"]
        assert len(df) == 200

    def test_categorical_schema(self, fitted_model):
        df = fitted_model.relativities()["region"]
        assert list(df.columns) == ["level", "relativity", "log_relativity"]
        assert set(df["level"]) == {"A", "B", "C"}

    def test_numeric_schema(self, fitted_model):
        df = fitted_model.relativities()["density"]
        assert list(df.columns) == ["label", "relativity", "log_relativity"]
        assert len(df) == 1
        assert df["label"].iloc[0] == "per_unit"

    def test_spline_exp_log_consistency(self, fitted_model):
        df = fitted_model.relativities()["age"]
        np.testing.assert_allclose(
            np.exp(df["log_relativity"]), df["relativity"], rtol=1e-10
        )

    def test_categorical_base_level_is_one(self, fitted_model):
        df = fitted_model.relativities()["region"]
        # base="first" → "A" is the base level (first alphabetically in the data)
        base_row = df[df["level"] == "A"]
        assert len(base_row) == 1
        assert base_row["relativity"].iloc[0] == pytest.approx(1.0)
        assert base_row["log_relativity"].iloc[0] == pytest.approx(0.0)


class TestPlotRelativities:
    def test_returns_figure(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativities()
        assert isinstance(fig, Figure)

    def test_standalone_function(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        rels = fitted_model.relativities()
        fig = plot_relativities(rels)
        assert isinstance(fig, Figure)

    def test_ncols_parameter(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(ncols=3)
        axes = fig.get_axes()
        assert len(axes) == 3  # 3 features, 1 row of 3

    def test_figsize_parameter(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(figsize=(12, 8))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12)
        assert h == pytest.approx(8)

    def test_with_exposure(self, sample_data, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, exposure = sample_data
        fig = fitted_model.plot_relativities(X=X, exposure=exposure)
        assert isinstance(fig, Figure)
        # Twin axes created for spline (age) and categorical (region) → extra axes
        all_axes = fig.get_axes()
        assert len(all_axes) > 3

    def test_standalone_with_exposure(self, sample_data, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, exposure = sample_data
        rels = fitted_model.relativities()
        fig = plot_relativities(rels, X=X, exposure=exposure)
        assert isinstance(fig, Figure)

    def test_empty_dict(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = plot_relativities({})
        assert isinstance(fig, Figure)
