"""Tests for relativities extraction and plotting."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
)
from superglm.editor import EditorSession
from superglm.plotting import plot_relativities, plot_term


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


@pytest.fixture
def fitted_model(sample_data):
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
    return model


@pytest.fixture
def polynomial_model():
    rng = np.random.default_rng(123)
    n = 400
    age = rng.uniform(18, 90, n)
    sample_weight = rng.uniform(0.4, 1.2, n)
    age_s = (age - 50.0) / 20.0
    mu = np.exp(-1.8 + 0.35 * age_s - 0.25 * age_s**2)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age": age})

    model = SuperGLM(features={"age": Polynomial(degree=2)})
    model.fit(X, y, sample_weight=sample_weight)
    return X, sample_weight, model


@pytest.fixture
def collapsed_ordered_model():
    rng = np.random.default_rng(20260708)
    levels = ["18-24", "25-34", "35-49", "50-64", "65-80"]
    age_band = rng.choice(levels, 600, p=[0.12, 0.24, 0.26, 0.25, 0.13])
    mileage = rng.normal(0.0, 1.0, len(age_band))
    sample_weight = rng.uniform(0.5, 1.5, len(age_band))
    effects = {
        "18-24": 0.00,
        "25-34": 0.01,
        "35-49": 0.01,
        "50-64": -0.08,
        "65-80": -0.14,
    }
    X = pd.DataFrame({"age_band": age_band, "mileage": mileage})
    y = (
        0.8
        + np.array([effects[value] for value in age_band])
        + 0.04 * mileage
        + rng.normal(0.0, 0.05, len(X))
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(
                order=levels,
                basis=Spline(kind="ps", n_knots=2),
                base="first",
            ),
            "mileage": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    session = EditorSession.from_model(
        model,
        terms=["age_band"],
        train_data=(X, y, sample_weight),
    )
    session.select_levels("age_band", ["18-24", "25-34", "35-49"])
    collapsed = session.replace_with_collapsed_levels("age_band", method="fit")
    return X, sample_weight, collapsed


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

    def test_categorical_base_level_is_one_native(self, fitted_model):
        df = fitted_model.relativities(centering="native")["region"]
        # base="first" → "A" is the base level (first alphabetically in the data)
        base_row = df[df["level"] == "A"]
        assert len(base_row) == 1
        assert base_row["relativity"].iloc[0] == pytest.approx(1.0)
        assert base_row["log_relativity"].iloc[0] == pytest.approx(0.0)

    def test_mean_centering_geometric_mean_one(self, fitted_model):
        df = fitted_model.relativities(centering="mean")["region"]
        # geometric mean of relativities should be 1.0
        assert np.mean(df["log_relativity"].values) == pytest.approx(0.0, abs=1e-10)


class TestPlotRelativities:
    def test_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot()
        assert isinstance(fig, Figure)

    def test_standalone_function(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        terms = [fitted_model.term_inference(n) for n in ("age", "region", "density")]
        fig = plot_relativities(terms)
        assert isinstance(fig, Figure)

    def test_ncols_parameter(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        fig = fitted_model.plot(ncols=3)
        axes = fig.get_axes()
        assert len(axes) == 3  # 3 features, 1 row of 3

    def test_figsize_parameter(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        fig = fitted_model.plot(figsize=(12, 8))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12)
        assert h == pytest.approx(8)

    def test_with_exposure(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, sample_weight = sample_data
        fig = fitted_model.plot(X=X, sample_weight=sample_weight)
        assert isinstance(fig, Figure)
        # Twin axes created for spline (age) and categorical (region) → extra axes
        all_axes = fig.get_axes()
        assert len(all_axes) > 3

    def test_standalone_with_exposure(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, sample_weight = sample_data
        terms = [fitted_model.term_inference(n) for n in ("age", "region", "density")]
        fig = plot_relativities(terms, X=X, sample_weight=sample_weight)
        assert isinstance(fig, Figure)

    def test_empty_list(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = plot_relativities([])
        assert isinstance(fig, Figure)

    def test_plot_with_ci(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot()
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

        fig = fitted_model.plot(ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No PolyCollection expected when ci=False"

    def test_plot_ci_categorical_errorbars(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot()
        # Categorical subplot should have a LineCollection from errorbar
        has_linecoll = any(
            isinstance(child, mcoll.LineCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_linecoll, "Expected a LineCollection (error bars) on a categorical subplot"


class TestPlotRelativitiesNew:
    """Smoke tests for the TermInference-based plotting path."""

    def test_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot()
        assert isinstance(fig, Figure)

    def test_ci_pointwise(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot(ci="pointwise")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly, "Expected CI band with ci='pointwise'"

    def test_ci_simultaneous(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot(ci="simultaneous")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly, "Expected simultaneous band"

    def test_ci_both(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot(ci="both")
        # Count PolyCollections — should have at least 2 per spline (pw + sim)
        poly_count = sum(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert poly_count >= 2, f"Expected nested bands, got {poly_count} PolyCollections"

    def test_ci_none(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot(ci=None)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected with ci=None"

    def test_show_density(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, sample_weight = sample_data
        fig = fitted_model.plot(
            X=X,
            sample_weight=sample_weight,
            show_density=True,
        )
        assert isinstance(fig, Figure)
        # With density strips, there should be more axes than just the main panels
        all_axes = fig.get_axes()
        assert len(all_axes) > 3

    def test_show_knots(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot(show_knots=True)
        assert isinstance(fig, Figure)

    def test_ci_false(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot(ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected when ci=False"

    def test_standalone_term_list(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        terms = [fitted_model.term_inference(n) for n in ("age", "region", "density")]
        fig = plot_relativities(terms)
        assert isinstance(fig, Figure)

    def test_mixed_features(self, sample_data, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, sample_weight = sample_data
        # fitted_model has spline (age), categorical (region), numeric (density)
        fig = fitted_model.plot(X=X, sample_weight=sample_weight, show_density=True)
        assert isinstance(fig, Figure)

        visible = [ax for ax in fig.get_axes() if ax.get_visible()]
        # Spline (age): main + density strip = 2
        # Categorical (region): main + twin sample_weight axis = 2 (spans both grid rows)
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

        # Density strips: axes with PolyCollection + ylim near [0, 1.05] + no yticks
        density_strips = [
            ax
            for ax in visible
            if any(isinstance(c, mcoll.PolyCollection) for c in ax.get_children())
            and len(ax.get_yticks()) == 0
        ]
        # At least 2 density strips: one for spline (age), one for numeric (density)
        assert len(density_strips) >= 2, f"Expected >= 2 density strips, got {len(density_strips)}"


class TestPlotRelativity:
    """Smoke tests for model.plot('term') single-term plotting."""

    def test_spline_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("age")
        assert isinstance(fig, Figure)

    def test_spline_ci_both(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci="both")
        poly_count = sum(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert poly_count >= 2, f"Expected nested bands, got {poly_count} PolyCollections"

    def test_spline_ci_none(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci=None)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected with ci=None"

    def test_spline_ci_false(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly, "No bands expected when ci=False"

    def test_spline_show_knots(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("age", show_knots=True)
        assert isinstance(fig, Figure)

    def test_spline_density_strip(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=sample_weight)
        # Main panel + density strip = 2 axes
        assert len(fig.get_axes()) >= 2

    def test_categorical_vertical(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("region")
        assert isinstance(fig, Figure)
        ax = fig.get_axes()[0]
        # Vertical orientation: levels on x-axis
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(x_labels) & {"A", "B", "C"}, f"Expected level labels on x-axis, got {x_labels}"

    def test_categorical_with_exposure_bars(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("region", X=X, sample_weight=sample_weight)
        # Twin axis for sample_weight bars → 2 axes total
        assert len(fig.get_axes()) >= 2
        ax2 = fig.get_axes()[1]
        assert ax2.get_ylabel() == "Weight"
        assert len(ax2.get_yticks()) > 0
        ti = fitted_model.term_inference("region")
        expected = (
            pd.DataFrame({"level": X["region"], "weight": sample_weight})
            .groupby("level", sort=False)["weight"]
            .sum()
        )
        expected_vals = np.array([expected.get(level, 0.0) for level in ti.levels], dtype=float)
        heights = np.array([patch.get_height() for patch in ax2.patches], dtype=float)
        np.testing.assert_allclose(heights, expected_vals)

    def test_ordered_categorical_plot_defaults_to_collapsed_group_display(
        self,
        collapsed_ordered_model,
    ):
        import matplotlib

        matplotlib.use("Agg")

        X, sample_weight, model = collapsed_ordered_model
        fig = model.plot("age_band", X=X, sample_weight=sample_weight)
        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels()]

        assert labels == ["18-24+25-34+35-49", "50-64", "65-80"]
        ax2 = fig.axes[1]
        heights = np.array([patch.get_height() for patch in ax2.patches], dtype=float)
        expected = (
            pd.DataFrame({"level": X["age_band"], "weight": sample_weight})
            .groupby("level", sort=False)["weight"]
            .sum()
        )
        np.testing.assert_allclose(
            heights,
            [
                sum(expected.get(level, 0.0) for level in ["18-24", "25-34", "35-49"]),
                expected.get("50-64", 0.0),
                expected.get("65-80", 0.0),
            ],
        )

    def test_ordered_categorical_plot_can_show_expanded_group_members(
        self,
        collapsed_ordered_model,
    ):
        import matplotlib

        matplotlib.use("Agg")

        X, sample_weight, model = collapsed_ordered_model
        fig = model.plot(
            "age_band",
            X=X,
            sample_weight=sample_weight,
            grouped_level_display="expanded",
        )
        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels()]

        assert labels == ["18-24", "25-34", "35-49", "50-64", "65-80"]

    def test_plotly_ordered_categorical_group_display_uses_collapsed_labels(
        self,
        collapsed_ordered_model,
    ):
        go = pytest.importorskip("plotly.graph_objects")

        X, sample_weight, model = collapsed_ordered_model
        fig = model.plot(
            ["age_band", "mileage"],
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            grouped_level_display="collapsed",
        )
        marker_traces = [
            trace
            for trace in fig.data
            if isinstance(trace, go.Scatter)
            and trace.name == "Relativity"
            and list(trace.x) == ["18-24+25-34+35-49", "50-64", "65-80"]
        ]

        assert marker_traces

    def test_plotly_collapsed_ordered_categorical_suppresses_stale_knot_diagnostics(self):
        go = pytest.importorskip("plotly.graph_objects")

        rng = np.random.default_rng(20260708)
        levels = ["18-24", "25-34", "35-49", "50-64", "65-80"]
        values = {
            "18-24": 21.0,
            "25-34": 30.0,
            "35-49": 42.0,
            "50-64": 57.0,
            "65-80": 72.0,
        }
        age_band = rng.choice(levels, 700, p=[0.12, 0.23, 0.30, 0.22, 0.13])
        mileage = rng.normal(0.0, 1.0, len(age_band))
        X = pd.DataFrame({"age_band": age_band, "mileage": mileage})
        y = (
            0.8
            + 0.16 * np.sin(np.array([values[level] for level in age_band]) / 10.0)
            + 0.04 * mileage
            + rng.normal(0.0, 0.05, len(age_band))
        )
        sample_weight = np.ones(len(X), dtype=np.float64)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={
                "age_band": OrderedCategorical(values=values, basis=Spline(kind="ps", n_knots=3)),
                "mileage": Numeric(),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        session = EditorSession.from_model(
            model,
            terms=["age_band"],
            train_data=(X, y, sample_weight),
        )
        session.select_levels("age_band", ["18-24", "25-34", "35-49"])
        model = session.replace_with_collapsed_levels("age_band", method="fit")

        fig = model.plot(
            ["age_band", "mileage"],
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            grouped_level_display="collapsed",
            show_knots=True,
        )

        knot_traces = [
            trace
            for trace in fig.data
            if isinstance(trace, go.Scatter) and trace.name == "Interior knots"
        ]

        assert knot_traces == []

    def test_numeric_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("density")
        assert isinstance(fig, Figure)

    def test_numeric_density_strip(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("density", X=X, sample_weight=sample_weight)
        # Twin axis for sample_weight histogram → 2 axes total
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

        X, sample_weight, model = polynomial_model
        fig = model.plot("age", X=X, sample_weight=sample_weight)
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
