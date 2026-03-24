"""Tests for interaction plotting."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline
from superglm.model import SuperGLM
from superglm.plotting import plot_interaction

# ── Shared fixture ────────────────────────────────────────────────


@pytest.fixture
def interaction_data():
    """Synthetic dataset with all feature types needed for interaction tests."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "bm": rng.normal(100, 15, n),
            "density": rng.normal(50, 10, n),
            "region": rng.choice(["A", "B", "C"], n),
            "type": rng.choice(["X", "Y", "Z"], n),
        }
    )
    mu = np.exp(
        -1.0
        + 0.01 * X["age"].to_numpy() / 10
        + 0.3 * (X["region"] == "B").to_numpy()
        + 0.001 * X["bm"].to_numpy()
    )
    y = rng.poisson(mu)
    return X, y


# ── Helper to fit a model with a given interaction ────────────────


def _fit_spline_cat(data):
    X, y = data
    m = SuperGLM(
        features={"age": Spline(n_knots=5), "region": Categorical()},
        interactions=[("age", "region")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


def _fit_poly_cat(data):
    X, y = data
    m = SuperGLM(
        features={"age": Polynomial(degree=2), "region": Categorical()},
        interactions=[("age", "region")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


def _fit_cat_cat(data):
    X, y = data
    m = SuperGLM(
        features={"region": Categorical(), "type": Categorical()},
        interactions=[("region", "type")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


def _fit_num_cat(data):
    X, y = data
    m = SuperGLM(
        features={"bm": Numeric(), "region": Categorical()},
        interactions=[("bm", "region")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


def _fit_num_num(data):
    X, y = data
    m = SuperGLM(
        features={"bm": Numeric(), "density": Numeric()},
        interactions=[("bm", "density")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


def _fit_poly_poly(data):
    X, y = data
    m = SuperGLM(
        features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
        interactions=[("age", "bm")],
        selection_penalty=0.01,
    )
    m.fit(X, y)
    return m


# ── matplotlib tests ──────────────────────────────────────────────


class TestVaryingCoefficientMpl:
    def test_spline_cat_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_spline_cat(interaction_data)
        fig = plot_interaction(model, "age:region")
        assert isinstance(fig, Figure)

    def test_spline_cat_correct_n_lines(self, interaction_data):
        model = _fit_spline_cat(interaction_data)
        fig = plot_interaction(model, "age:region")
        ax = fig.axes[0]
        # Solid lines = non-base curves (base is dashed, reference is dotted)
        solid_lines = [line for line in ax.get_lines() if line.get_linestyle() == "-"]
        n_non_base = len(model._interaction_specs["age:region"]._non_base)
        assert len(solid_lines) == n_non_base

    def test_poly_cat_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_poly_cat(interaction_data)
        fig = plot_interaction(model, "age:region")
        assert isinstance(fig, Figure)


class TestCategoricalHeatmapMpl:
    def test_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_cat_cat(interaction_data)
        fig = plot_interaction(model, "region:type")
        assert isinstance(fig, Figure)

    def test_has_imshow(self, interaction_data):
        model = _fit_cat_cat(interaction_data)
        fig = plot_interaction(model, "region:type")
        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) >= 1


class TestNumericCategoricalBarsMpl:
    def test_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_num_cat(interaction_data)
        fig = plot_interaction(model, "bm:region")
        assert isinstance(fig, Figure)

    def test_correct_n_bars(self, interaction_data):
        model = _fit_num_cat(interaction_data)
        fig = plot_interaction(model, "bm:region")
        ax = fig.axes[0]
        patches = ax.patches
        n_non_base = len(model._interaction_specs["bm:region"]._non_base)
        # +1 for the base level bar
        assert len(patches) == n_non_base + 1


class TestNumericInteractionBarMpl:
    def test_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_num_num(interaction_data)
        fig = plot_interaction(model, "bm:density")
        assert isinstance(fig, Figure)

    def test_single_bar(self, interaction_data):
        model = _fit_num_num(interaction_data)
        fig = plot_interaction(model, "bm:density")
        ax = fig.axes[0]
        assert len(ax.patches) == 1


class TestSurfaceMpl:
    def test_returns_figure(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm")
        assert isinstance(fig, Figure)

    def test_has_contourf(self, interaction_data):
        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm")
        ax = fig.axes[0]
        # contourf creates QuadContourSet children
        assert len(ax.collections) > 0


# ── plotly tests ──────────────────────────────────────────────────


plotly = pytest.importorskip("plotly")


class TestVaryingCoefficientPlotly:
    def test_returns_go_figure(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_spline_cat(interaction_data)
        fig = plot_interaction(model, "age:region", engine="plotly")
        assert isinstance(fig, go.Figure)

    def test_correct_n_traces(self, interaction_data):
        model = _fit_spline_cat(interaction_data)
        fig = plot_interaction(model, "age:region", engine="plotly")
        n_levels = len(model._interaction_specs["age:region"]._non_base)
        scatter_traces = [t for t in fig.data if isinstance(t, plotly.graph_objects.Scatter)]
        assert len(scatter_traces) >= n_levels


class TestCategoricalHeatmapPlotly:
    def test_has_heatmap_trace(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_cat_cat(interaction_data)
        fig = plot_interaction(model, "region:type", engine="plotly")
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)


class TestSurfacePlotly:
    def test_has_surface_trace(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly")
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Surface) for t in fig.data)

    def test_contour_view_has_contour_trace(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly", interaction_view="contour")
        assert any(isinstance(t, go.Contour) for t in fig.data)
        assert not any(isinstance(t, go.Surface) for t in fig.data)

    def test_contour_pair_has_hdr_panel(self, interaction_data):
        import plotly.graph_objects as go

        X, _ = interaction_data
        sample_weight = np.ones(len(X), dtype=np.float64)
        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(
            model,
            "age:bm",
            engine="plotly",
            interaction_view="contour_pair",
            X=X,
            sample_weight=sample_weight,
        )
        assert any(isinstance(t, go.Contour) for t in fig.data)
        assert not any(isinstance(t, go.Surface) for t in fig.data)
        assert fig.layout.xaxis2.title.text == "age"
        assert fig.layout.yaxis2.title.text == "bm"
        assert any(
            isinstance(t, go.Contour)
            and getattr(getattr(t, "colorbar", None), "title", None).text == "HDR<br>Mass"
            for t in fig.data
        )

    def test_contour_pair_requires_density_data(self, interaction_data):
        model = _fit_poly_poly(interaction_data)
        with pytest.raises(ValueError, match="requires X and sample_weight"):
            plot_interaction(
                model,
                "age:bm",
                engine="plotly",
                interaction_view="contour_pair",
            )

    def test_invalid_interaction_view_raises(self, interaction_data):
        model = _fit_poly_poly(interaction_data)
        with pytest.raises(ValueError, match="interaction_view"):
            plot_interaction(model, "age:bm", engine="plotly", interaction_view="bogus")

    def test_respects_n_points(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly", n_points=73)
        surface = next(t for t in fig.data if isinstance(t, go.Surface) and t.name == "Relativity")
        assert len(surface.x) == 73
        assert len(surface.y) == 73
        assert np.asarray(surface.z).shape == (73, 73)

    def test_no_wall_traces_by_default(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly")
        assert not any(
            isinstance(t, go.Scatter3d) and "Main effect:" in (t.name or "") for t in fig.data
        )

    def test_adds_main_effect_wall_traces(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly", show_main_effect_walls=True)
        assert any(
            isinstance(t, go.Scatter3d) and "Main effect:" in (t.name or "") for t in fig.data
        )

    def test_surface_opacity_kwarg(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly", surface_opacity=0.62)
        surface = next(t for t in fig.data if isinstance(t, go.Surface) and t.name == "Relativity")
        assert surface.opacity == pytest.approx(0.62)

    def test_density_plane_sits_above_scene_floor(self, interaction_data):
        import plotly.graph_objects as go

        X, _ = interaction_data
        sample_weight = np.ones(len(X), dtype=np.float64)
        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(
            model,
            "age:bm",
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
        )
        density = next(
            t for t in fig.data if isinstance(t, go.Surface) and t.name == "Exposure density"
        )
        z_density = float(np.asarray(density.z).flat[0])
        z_floor = float(fig.layout.scene.zaxis.range[0])
        assert z_density > z_floor

    def test_wall_traces_not_in_legend(self, interaction_data):
        import plotly.graph_objects as go

        model = _fit_poly_poly(interaction_data)
        fig = plot_interaction(model, "age:bm", engine="plotly", show_main_effect_walls=True)
        walls = [
            t for t in fig.data if isinstance(t, go.Scatter3d) and "Main effect:" in (t.name or "")
        ]
        assert walls
        assert all(t.showlegend is False for t in walls)


# ── API / error tests ────────────────────────────────────────────


class TestInteractionPlotAPI:
    def test_model_plot_dispatches_interaction(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_spline_cat(interaction_data)
        fig = model.plot("age:region")
        assert isinstance(fig, Figure)

    def test_model_plot_data_returns_surface_and_density_grids(self, interaction_data):
        X, _ = interaction_data
        sample_weight = np.ones(len(X), dtype=np.float64)
        model = _fit_poly_poly(interaction_data)
        payload = model.plot_data("age:bm", X=X, sample_weight=sample_weight, n_points=41)
        assert payload["kind"] == "interaction"
        assert payload["plot_kind"] == "surface"
        assert {"age", "bm", "relativity", "log_relativity"} <= set(payload["effect"].columns)
        assert len(payload["grid_axes"]["age"]) == 41
        assert len(payload["grid_axes"]["bm"]) == 41
        assert payload["density"] is not None
        assert {"age", "bm", "density", "hdr_mass"} <= set(payload["density"].columns)

    def test_model_plot_forwards_interaction_view(self, interaction_data):
        import plotly.graph_objects as go

        X, _ = interaction_data
        sample_weight = np.ones(len(X), dtype=np.float64)
        model = _fit_poly_poly(interaction_data)
        fig = model.plot(
            "age:bm",
            engine="plotly",
            interaction_view="contour_pair",
            X=X,
            sample_weight=sample_weight,
        )
        assert any(isinstance(t, go.Contour) for t in fig.data)
        assert not any(isinstance(t, go.Surface) for t in fig.data)

    def test_unknown_interaction_raises(self, interaction_data):
        model = _fit_spline_cat(interaction_data)
        with pytest.raises(KeyError, match="Interaction not found"):
            plot_interaction(model, "nonexistent:feature")

    def test_unknown_engine_raises(self, interaction_data):
        model = _fit_spline_cat(interaction_data)
        with pytest.raises(ValueError, match="Unknown engine"):
            plot_interaction(model, "age:region", engine="bokeh")

    def test_plotly_import_error(self, interaction_data):
        model = _fit_spline_cat(interaction_data)
        with patch.dict("sys.modules", {"plotly": None, "plotly.graph_objects": None}):
            with pytest.raises(ImportError, match="plotly is required"):
                plot_interaction(model, "age:region", engine="plotly")
