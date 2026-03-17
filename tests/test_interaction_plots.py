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


# ── API / error tests ────────────────────────────────────────────


class TestInteractionPlotAPI:
    def test_model_plot_dispatches_interaction(self, interaction_data):
        from matplotlib.figure import Figure

        model = _fit_spline_cat(interaction_data)
        fig = model.plot("age:region")
        assert isinstance(fig, Figure)

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
