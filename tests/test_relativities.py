"""Tests for relativities extraction and plotting."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Numeric,
    Polynomial,
    Spline,
    SuperGLM,
    plot_relativities,
    plot_term,
)


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


@pytest.fixture
def polynomial_model():
    rng = np.random.default_rng(123)
    n = 400
    age = rng.uniform(18, 90, n)
    exposure = rng.uniform(0.4, 1.2, n)
    age_s = (age - 50.0) / 20.0
    mu = np.exp(-1.8 + 0.35 * age_s - 0.25 * age_s**2)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"age": age})

    model = SuperGLM(features={"age": Polynomial(degree=2)})
    model.fit(X, y, exposure=exposure)
    return X, exposure, model


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
        np.testing.assert_allclose(np.exp(df["log_relativity"]), df["relativity"], rtol=1e-10)

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

    def test_plot_with_ci(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities()
        # Spline subplot should have a PolyCollection from fill_between
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly, "Expected a PolyCollection (CI band) on a spline subplot"

    def test_plot_ci_disabled(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(with_ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No PolyCollection expected when with_ci=False"

    def test_plot_ci_categorical_errorbars(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities()
        # Categorical subplot should have a LineCollection from errorbar
        has_linecoll = any(
            isinstance(child, mcoll.LineCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_linecoll, "Expected a LineCollection (error bars) on a categorical subplot"


class TestPlotRelativitiesNew:
    """Smoke tests for the new TermInference-based plotting path."""

    def test_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativities()
        assert isinstance(fig, Figure)

    def test_interval_pointwise(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(interval="pointwise")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly, "Expected CI band with interval='pointwise'"

    def test_interval_simultaneous(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(interval="simultaneous")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly, "Expected simultaneous band"

    def test_interval_both(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(interval="both")
        # Count PolyCollections — should have at least 2 per spline (pw + sim)
        poly_count = sum(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert poly_count >= 2, f"Expected nested bands, got {poly_count} PolyCollections"

    def test_interval_none(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(interval=None)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected with interval=None"

    def test_show_exposure(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, exposure = sample_data
        fig = fitted_model.plot_relativities(
            X=X,
            exposure=exposure,
            show_exposure=True,
        )
        assert isinstance(fig, Figure)
        # With density strips, there should be more axes than just the main panels
        all_axes = fig.get_axes()
        assert len(all_axes) > 3

    def test_show_knots(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativities(show_knots=True)
        assert isinstance(fig, Figure)

    def test_with_ci_false(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativities(with_ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected when with_ci=False"

    def test_legacy_dict_api(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        rels = fitted_model.relativities()
        fig = plot_relativities(rels)
        assert isinstance(fig, Figure)

    def test_mixed_features(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, exposure = sample_data
        # fitted_model has spline (age), categorical (region), numeric (density)
        fig = fitted_model.plot_relativities(X=X, exposure=exposure, show_exposure=True)
        assert isinstance(fig, Figure)

        visible = [ax for ax in fig.get_axes() if ax.get_visible()]
        # Spline (age): main + density strip = 2
        # Categorical (region): main + twin exposure axis = 2 (spans both grid rows)
        # Numeric (density): main + density strip = 2
        # + 1 hidden unused grid cell
        assert len(visible) >= 5, f"Expected >= 5 visible axes, got {len(visible)}"

        # Categorical panel: vertical orientation — level labels on x-axis
        cat_axes = [
            ax
            for ax in visible
            if any(t.get_text() in ("A", "B", "C") for t in ax.get_xticklabels())
        ]
        assert len(cat_axes) >= 1, "Categorical panel should have visible level labels on x-axis"

        # Numeric panel: continuous flat line — has a density strip below
        density_strips = [
            ax
            for ax in visible
            if any(isinstance(c, mcoll.PolyCollection) for c in ax.get_children())
            and ax.get_xlabel() == "density"
        ]
        assert len(density_strips) >= 1, "Numeric term should have an exposure density strip"


class TestPlotRelativity:
    """Smoke tests for the single-term plot_relativity() entry point."""

    def test_spline_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativity("age")
        assert isinstance(fig, Figure)

    def test_spline_interval_both(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativity("age", interval="both")
        poly_count = sum(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert poly_count >= 2, f"Expected nested bands, got {poly_count} PolyCollections"

    def test_spline_interval_none(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativity("age", interval=None)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected with interval=None"

    def test_spline_with_ci_false(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot_relativity("age", with_ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected when with_ci=False"

    def test_spline_show_knots(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativity("age", show_knots=True)
        assert isinstance(fig, Figure)

    def test_spline_density_strip(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, exposure = sample_data
        fig = fitted_model.plot_relativity("age", X=X, exposure=exposure)
        # Main panel + density strip = 2 axes
        assert len(fig.get_axes()) >= 2

    def test_categorical_vertical(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativity("region")
        assert isinstance(fig, Figure)
        ax = fig.get_axes()[0]
        # Vertical orientation: levels on x-axis
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(x_labels) & {"A", "B", "C"}, f"Expected level labels on x-axis, got {x_labels}"

    def test_categorical_with_exposure_bars(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, exposure = sample_data
        fig = fitted_model.plot_relativity("region", X=X, exposure=exposure)
        # Twin axis for exposure bars → 2 axes total
        assert len(fig.get_axes()) >= 2
        ax2 = fig.get_axes()[1]
        assert ax2.get_ylabel() == "Weight"
        assert len(ax2.get_yticks()) > 0

    def test_numeric_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot_relativity("density")
        assert isinstance(fig, Figure)

    def test_numeric_density_strip(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, exposure = sample_data
        fig = fitted_model.plot_relativity("density", X=X, exposure=exposure)
        # Twin axis for exposure histogram → 2 axes total
        assert len(fig.get_axes()) >= 2

    def test_standalone_plot_term(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        ti = fitted_model.term_inference("age")
        fig = plot_term(ti)
        assert isinstance(fig, Figure)

    def test_polynomial_returns_figure(self, polynomial_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, exposure, model = polynomial_model
        fig = model.plot_relativity("age", X=X, exposure=exposure)
        assert isinstance(fig, Figure)

    def test_polynomial_term_inference_matches_grid(self, polynomial_model):
        X, _, model = polynomial_model
        ti = model.term_inference("age")

        assert ti.kind == "polynomial"
        assert ti.x is not None
        assert ti.se_log_relativity is not None
        assert len(ti.x) == len(ti.relativity) == len(ti.se_log_relativity)
        assert ti.x.min() == pytest.approx(X["age"].min())
        assert ti.x.max() == pytest.approx(X["age"].max())
