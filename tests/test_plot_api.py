"""Tests for the unified model.plot() API."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


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
def interaction_model(sample_data):
    X, y, sample_weight = sample_data
    model = SuperGLM(
        selection_penalty=0.01,
        features={
            "age": Spline(n_knots=8, penalty="ssp"),
            "region": Categorical(base="first"),
        },
        interactions=[("age", "region")],
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ── terms=None: all main effects ───────────────────────────────


class TestPlotAllTerms:
    def test_returns_figure(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot()
        assert isinstance(fig, Figure)

    def test_all_features_plotted(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        fig = fitted_model.plot()
        # 3 features → at least 3 visible main axes
        visible = [ax for ax in fig.get_axes() if ax.get_visible()]
        assert len(visible) >= 3


# ── terms="name": single main effect ──────────────────────────


class TestPlotSingleTerm:
    def test_spline(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("age")
        assert isinstance(fig, Figure)

    def test_categorical(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("region")
        assert isinstance(fig, Figure)

    def test_numeric(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot("density")
        assert isinstance(fig, Figure)


class TestPlotData:
    def test_single_term_payload_has_effect_density_knots_and_bases(
        self, fitted_model, sample_data
    ):
        X, _, sample_weight = sample_data
        payload = fitted_model.plot_data(
            "age",
            X=X,
            sample_weight=sample_weight,
            show_knots=True,
            show_bases=True,
        )
        assert payload["kind"] == "main_effects"
        assert len(payload["terms"]) == 1
        term = payload["terms"][0]
        assert term["name"] == "age"
        assert {"x", "log_relativity", "relativity"} <= set(term["effect"].columns)
        assert term["density"] is not None
        assert {"x", "density"} <= set(term["density"].columns)
        assert term["knots"] is not None
        assert {"x", "relativity", "log_relativity"} <= set(term["knots"].columns)
        assert term["bases"] is not None
        assert {"x", "basis_index", "basis_value", "coefficient", "contribution"} <= set(
            term["bases"].columns
        )

    def test_all_main_effect_terms_follow_model_order(self, fitted_model):
        payload = fitted_model.plot_data()
        assert payload["kind"] == "main_effects"
        assert [term["name"] for term in payload["terms"]] == list(fitted_model._feature_order)


# ── terms=["a", "b"]: subset of main effects ──────────────────


class TestPlotSubset:
    def test_two_features(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot(["age", "region"])
        assert isinstance(fig, Figure)

    def test_single_in_list(self, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = fitted_model.plot(["age"])
        assert isinstance(fig, Figure)


# ── terms="a:b": single interaction ────────────────────────────


class TestPlotInteraction:
    def test_interaction_matplotlib(self, interaction_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = interaction_model.plot("age:region")
        assert isinstance(fig, Figure)


# ── ci parameter ───────────────────────────────────────────────


class TestCIParameter:
    def test_ci_none(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci=None)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly

    def test_ci_false(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci=False)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert not has_poly

    def test_ci_pointwise(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci="pointwise")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly

    def test_ci_simultaneous(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci="simultaneous")
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly

    def test_ci_both(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci="both")
        poly_count = sum(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert poly_count >= 2

    def test_ci_true(self, fitted_model):
        import matplotlib
        import matplotlib.collections as mcoll

        matplotlib.use("Agg")

        fig = fitted_model.plot("age", ci=True)
        has_poly = any(
            isinstance(child, mcoll.PolyCollection)
            for ax in fig.get_axes()
            for child in ax.get_children()
        )
        assert has_poly

    def test_ci_invalid_raises(self, fitted_model):
        with pytest.raises(ValueError, match="ci="):
            fitted_model.plot("age", ci="invalid")


# ── density with and without sample_weight ─────────────────────


class TestDensity:
    def test_density_with_sample_weight(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=sample_weight, show_density=True)
        assert len(fig.get_axes()) >= 2

    def test_density_without_sample_weight(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, _ = sample_data
        fig = fitted_model.plot("age", X=X, show_density=True)
        # Without sample_weight, falls back to observation density → strip shown
        assert len(fig.get_axes()) >= 2

    def test_density_label_with_weight(self, sample_data, fitted_model):
        """When sample_weight is provided, density strip is labeled 'Weight density'."""
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=sample_weight, show_density=True)
        ylabels = [ax.get_ylabel() for ax in fig.get_axes()]
        assert any("Weight" in lbl for lbl in ylabels)

    def test_density_label_without_weight(self, sample_data, fitted_model):
        """Without sample_weight, density strip is labeled 'Obs. density'."""
        import matplotlib

        matplotlib.use("Agg")

        X, y, _ = sample_data
        fig = fitted_model.plot("age", X=X, show_density=True)
        ylabels = [ax.get_ylabel() for ax in fig.get_axes()]
        assert any("Obs." in lbl for lbl in ylabels)
        assert not any("Weight" in lbl for lbl in ylabels)

    def test_categorical_label_without_weight(self, sample_data, fitted_model):
        """Categorical bars are labeled 'Count' when sample_weight is omitted."""
        import matplotlib

        matplotlib.use("Agg")

        X, y, _ = sample_data
        fig = fitted_model.plot("region", X=X, show_density=True)
        all_labels = []
        for ax in fig.get_axes():
            all_labels.append(ax.get_ylabel())
            h, lab = ax.get_legend_handles_labels()
            all_labels.extend(lab)
        assert any("Count" in lbl for lbl in all_labels)

    def test_density_disabled(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=sample_weight, show_density=False)
        # No density strip → single axis
        assert len(fig.get_axes()) == 1


# ── Error cases ────────────────────────────────────────────────


class TestPlotErrors:
    def test_mixed_terms_raises(self, interaction_model):
        with pytest.raises(ValueError, match="Cannot mix"):
            interaction_model.plot(["age", "age:region"])

    def test_multiple_interactions_raises(self, interaction_model):
        with pytest.raises(ValueError, match="one interaction"):
            interaction_model.plot(["age:region", "age:region"])

    def test_unknown_term_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Term not found"):
            fitted_model.plot("nonexistent")

    def test_unknown_term_in_list_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Term.*not found"):
            fitted_model.plot(["age", "nonexistent"])

    def test_unknown_engine_raises(self, interaction_model):
        with pytest.raises(ValueError, match="Unknown engine"):
            interaction_model.plot("age:region", engine="bokeh")

    def test_unfitted_model_raises(self):
        model = SuperGLM(features={"x": Spline(n_knots=5)})
        with pytest.raises(RuntimeError, match="fitted"):
            model.plot()

    def test_ambiguous_term_raises(self):
        """Feature named 'a:b' + interaction ('a','b') → ambiguity error."""
        rng = np.random.default_rng(42)
        n = 200
        a = rng.uniform(0, 10, n)
        b = rng.choice(["X", "Y"], n)
        y = rng.poisson(np.exp(0.5 + 0.1 * a)).astype(float)
        X = pd.DataFrame({"a": a, "b": b, "a:b": a})

        model = SuperGLM(
            features={
                "a": Spline(n_knots=5),
                "b": Categorical(),
                "a:b": Spline(n_knots=5),
            },
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="Ambiguous"):
            model.plot("a:b")

    def test_ambiguous_term_in_list_raises(self):
        """Ambiguity error also fires from list path."""
        rng = np.random.default_rng(42)
        n = 200
        a = rng.uniform(0, 10, n)
        b = rng.choice(["X", "Y"], n)
        y = rng.poisson(np.exp(0.5 + 0.1 * a)).astype(float)
        X = pd.DataFrame({"a": a, "b": b, "a:b": a})

        model = SuperGLM(
            features={
                "a": Spline(n_knots=5),
                "b": Categorical(),
                "a:b": Spline(n_knots=5),
            },
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="Ambiguous"):
            model.plot(["a:b"])

    def test_reconstruct_feature_ambiguous_raises(self):
        """reconstruct_feature() also raises on name collision."""
        rng = np.random.default_rng(42)
        n = 200
        a = rng.uniform(0, 10, n)
        b = rng.choice(["X", "Y"], n)
        y = rng.poisson(np.exp(0.5 + 0.1 * a)).astype(float)
        X = pd.DataFrame({"a": a, "b": b, "a:b": a})

        model = SuperGLM(
            features={
                "a": Spline(n_knots=5),
                "b": Categorical(),
                "a:b": Spline(n_knots=5),
            },
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="Ambiguous"):
            model.reconstruct_feature("a:b")

    def test_term_inference_ambiguous_raises(self):
        """term_inference() also raises on name collision."""
        rng = np.random.default_rng(42)
        n = 200
        a = rng.uniform(0, 10, n)
        b = rng.choice(["X", "Y"], n)
        y = rng.poisson(np.exp(0.5 + 0.1 * a)).astype(float)
        X = pd.DataFrame({"a": a, "b": b, "a:b": a})

        model = SuperGLM(
            features={
                "a": Spline(n_knots=5),
                "b": Categorical(),
                "a:b": Spline(n_knots=5),
            },
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="Ambiguous"):
            model.term_inference("a:b")


# ── Regression: colon in feature names ─────────────────────────


class TestColonInFeatureName:
    def test_colon_feature_plots_as_main_effect(self):
        """Feature named 'a:b' should plot as a main effect, not an interaction."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x)).astype(float)
        X = pd.DataFrame({"a:b": x})

        model = SuperGLM(features={"a:b": Spline(n_knots=5)})
        model.fit(X, y)

        fig = model.plot("a:b")
        assert isinstance(fig, Figure)


# ── Interaction density fallback ──────────────────────────────


class TestInteractionDensity:
    def test_interaction_density_without_sample_weight(self, sample_data, interaction_model):
        """Interaction plot shows density overlay when X given but no sample_weight."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, _ = sample_data
        fig = interaction_model.plot("age:region", X=X)
        assert isinstance(fig, Figure)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
class TestPlotlyMainEffects:
    def test_plotly_single_main_effect_rejected(self, fitted_model):
        with pytest.raises(ValueError, match="multi-term main-effect explorer"):
            fitted_model.plot("age", engine="plotly")

    def test_plotly_single_term_list_rejected(self, fitted_model):
        with pytest.raises(ValueError, match="at least two main effects"):
            fitted_model.plot(["age"], engine="plotly")

    def test_plotly_all_main_effects_has_dropdown(self, sample_data, fitted_model):
        import plotly.graph_objects as go

        X, y, sample_weight = sample_data
        fig = fitted_model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        assert isinstance(fig, go.Figure)
        assert fig.layout.updatemenus
        # updatemenus[0] = scale toggle (Response|Link), [1] = term dropdown
        assert len(fig.layout.updatemenus[0].buttons) == 2  # Response, Link
        assert len(fig.layout.updatemenus[1].buttons) == 3  # 3 features

    def test_plotly_controls_are_left_stacked(self, fitted_model):
        fig = fitted_model.plot(engine="plotly")
        scale_menu = fig.layout.updatemenus[0]
        term_menu = fig.layout.updatemenus[1]

        assert scale_menu.type == "buttons"
        assert term_menu.type == "dropdown"
        assert scale_menu.x == 0.0
        assert scale_menu.xanchor == "left"
        assert term_menu.x == 0.0
        assert term_menu.xanchor == "left"
        assert scale_menu.y > term_menu.y

        annotations = {annotation.text: annotation for annotation in fig.layout.annotations}
        assert annotations["Scale"].x == 0.0
        assert annotations["Scale"].xanchor == "left"
        assert annotations["Term"].x == 0.0
        assert annotations["Term"].xanchor == "left"
        assert annotations["Scale"].y > annotations["Term"].y

    def test_plotly_multi_main_effects(self, sample_data, fitted_model):
        import plotly.graph_objects as go

        X, y, sample_weight = sample_data
        fig = fitted_model.plot(
            ["age", "region"], engine="plotly", X=X, sample_weight=sample_weight
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.updatemenus
        assert len(fig.layout.updatemenus[1].buttons) == 2  # 2 features

    def test_plotly_density_trace(self, sample_data, fitted_model):
        X, y, sample_weight = sample_data
        fig = fitted_model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        trace_names = {trace.name for trace in fig.data}
        assert "Exposure density" in trace_names

    def test_plotly_density_absent_without_X(self, fitted_model):
        fig = fitted_model.plot(engine="plotly")
        trace_names = {trace.name for trace in fig.data}
        assert "Exposure density" not in trace_names

    def test_plotly_ordered_categorical_step_renders(self):
        rng = np.random.default_rng(123)
        n = 400
        levels = ["Low", "Medium", "High", "Very High"]
        X = pd.DataFrame(
            {
                "risk": rng.choice(levels, n, p=[0.35, 0.30, 0.22, 0.13]),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        effect = {"Low": 0.0, "Medium": 0.15, "High": 0.35, "Very High": 0.65}
        mu = np.exp(-1.4 + 0.2 * X["x"].to_numpy() + np.array([effect[v] for v in X["risk"]]))
        y = rng.poisson(mu * sample_weight).astype(float)

        model = SuperGLM(
            features={
                "risk": OrderedCategorical(order=levels, basis="step"),
                "x": Numeric(),
            }
        )
        model.fit(X, y, sample_weight=sample_weight)

        import plotly.graph_objects as go

        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        assert isinstance(fig, go.Figure)

    def test_plotly_categorical_uses_scatter_with_error_bars(self):
        rng = np.random.default_rng(100)
        n = 500
        X = pd.DataFrame(
            {
                "cat": rng.choice(["A", "B", "C", "D", "E"], n),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        mu = np.exp(-1.7 + 0.15 * X["x"].to_numpy() + 0.2 * (X["cat"] == "B").to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)
        model = SuperGLM(features={"cat": Categorical(), "x": Numeric()})
        model.fit(X, y, sample_weight=sample_weight)

        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        top_panel = [t for t in fig.data if getattr(t, "xaxis", None) == "x" and t.visible is True]
        rel = next(t for t in top_panel if t.type == "scatter" and t.name == "Relativity")
        assert rel.error_y.array is not None
        # Exposure is now on y3 (right axis) in top panel
        exposure = next(t for t in fig.data if t.type == "bar" and t.name == "Exposure")
        assert exposure.yaxis == "y3"

    def test_plotly_categorical_uses_scatter_many_levels(self):
        rng = np.random.default_rng(101)
        n = 1200
        levels = [f"L{i:02d}" for i in range(35)]
        X = pd.DataFrame(
            {
                "cat": rng.choice(levels, n),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        mu = np.exp(-1.8 + 0.1 * X["x"].to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)
        model = SuperGLM(features={"cat": Categorical(), "x": Numeric()})
        model.fit(X, y, sample_weight=sample_weight)

        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        top_panel = [t for t in fig.data if getattr(t, "xaxis", None) == "x" and t.visible is True]
        # Now uses scatter (lines+markers) even for many levels
        assert any(t.type == "scatter" and t.name == "Relativity" for t in top_panel)

    def test_plotly_categorical_uses_top_density_overlay(self):
        rng = np.random.default_rng(151)
        n = 500
        X = pd.DataFrame(
            {
                "cat": rng.choice(["A", "B", "C", "D", "E"], n),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        mu = np.exp(-1.7 + 0.15 * X["x"].to_numpy() + 0.2 * (X["cat"] == "B").to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)
        model = SuperGLM(features={"cat": Categorical(), "x": Numeric()})
        model.fit(X, y, sample_weight=sample_weight)

        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)
        # For categoricals, exposure is now on the top panel (y3) with overlay
        assert fig.layout.yaxis3.visible is True
        assert fig.layout.yaxis3.title.text == "Exposure"

    def test_plotly_categorical_display_override_still_uses_scatter(self):
        rng = np.random.default_rng(102)
        n = 1200
        levels = [f"L{i:02d}" for i in range(35)]
        X = pd.DataFrame(
            {
                "cat": rng.choice(levels, n),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        mu = np.exp(-1.8 + 0.1 * X["x"].to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)
        model = SuperGLM(features={"cat": Categorical(), "x": Numeric()})
        model.fit(X, y, sample_weight=sample_weight)

        fig = model.plot(
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            categorical_display="bars+markers",
        )
        top_panel = [t for t in fig.data if getattr(t, "xaxis", None) == "x" and t.visible is True]
        # categorical_display parameter is retained but rendering always uses scatter
        assert any(t.type == "scatter" and t.name == "Relativity" for t in top_panel)

    def test_plotly_ordered_categorical_spline_uses_numeric_axis(self):
        rng = np.random.default_rng(321)
        n = 500
        levels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        midpoints = {
            "18-25": 21.5,
            "26-35": 30.5,
            "36-45": 40.5,
            "46-55": 50.5,
            "56-65": 60.5,
            "65+": 70.0,
        }
        X = pd.DataFrame(
            {
                "age_band": rng.choice(levels, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        x_mid = np.array([midpoints[v] for v in X["age_band"]], dtype=np.float64)
        mu = np.exp(-1.7 + 0.015 * (x_mid - 45.0) ** 2 / 100 + 0.1 * X["x"].to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)

        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
                "x": Numeric(),
            }
        )
        model.fit(X, y, sample_weight=sample_weight)
        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)

        assert fig.layout.xaxis.type == "linear"
        assert list(fig.layout.xaxis.ticktext) == levels

        ordered_scatter = next(
            t for t in fig.data if t.type == "scatter" and t.name == "Relativity"
        )
        smooth_curve = next(t for t in fig.data if t.type == "scatter" and t.name == "Smooth curve")
        np.testing.assert_allclose(
            np.asarray(ordered_scatter.x, dtype=np.float64), list(midpoints.values())
        )
        assert len(smooth_curve.x) == 200
        assert np.issubdtype(np.asarray(smooth_curve.x).dtype, np.number)

    def test_plotly_ordered_categorical_spline_ci_moves_to_hover(self):
        rng = np.random.default_rng(321)
        n = 500
        levels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        midpoints = {
            "18-25": 21.5,
            "26-35": 30.5,
            "36-45": 40.5,
            "46-55": 50.5,
            "56-65": 60.5,
            "65+": 70.0,
        }
        X = pd.DataFrame(
            {
                "age_band": rng.choice(levels, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        x_mid = np.array([midpoints[v] for v in X["age_band"]], dtype=np.float64)
        mu = np.exp(-1.7 + 0.015 * (x_mid - 45.0) ** 2 / 100 + 0.1 * X["x"].to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)

        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
                "x": Numeric(),
            }
        )
        model.fit(X, y, sample_weight=sample_weight)
        fig = model.plot(engine="plotly", X=X, sample_weight=sample_weight)

        ordered_scatter = next(
            t for t in fig.data if t.type == "scatter" and t.name == "Relativity"
        )
        # Markers now carry error bars directly
        assert ordered_scatter.error_y.array is not None
        assert "95% CI:" in (ordered_scatter.hovertemplate or "")
        assert ordered_scatter.hovertext[0] == levels[0]

    def test_plotly_style_overrides_apply(self):
        rng = np.random.default_rng(321)
        n = 500
        levels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        midpoints = {
            "18-25": 21.5,
            "26-35": 30.5,
            "36-45": 40.5,
            "46-55": 50.5,
            "56-65": 60.5,
            "65+": 70.0,
        }
        X = pd.DataFrame(
            {
                "age_band": rng.choice(levels, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
                "x": rng.normal(size=n),
            }
        )
        sample_weight = rng.uniform(0.4, 1.0, n)
        x_mid = np.array([midpoints[v] for v in X["age_band"]], dtype=np.float64)
        mu = np.exp(-1.7 + 0.015 * (x_mid - 45.0) ** 2 / 100 + 0.1 * X["x"].to_numpy())
        y = rng.poisson(mu * sample_weight).astype(float)

        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
                "x": Numeric(),
            }
        )
        model.fit(X, y, sample_weight=sample_weight)
        fig = model.plot(
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            plotly_style={
                "bar_color": "#2244AA",
                "spline_color": "#00AA55",
                "weight_density_color": "#FFAA00",
                "weight_density_edge_color": "#663300",
                "error_bar_color": "#111111",
                "font_color": "#222222",
                "font_border_color": "#FFFFFF",
            },
        )

        ordered_scatter = next(
            t for t in fig.data if t.type == "scatter" and t.name == "Relativity" and len(t.x) > 1
        )
        smooth_curve = next(t for t in fig.data if t.type == "scatter" and t.name == "Smooth curve")
        density_bar = next(t for t in fig.data if t.type == "bar" and t.name == "Exposure")
        numeric_bar = next(t for t in fig.data if t.type == "bar" and t.name == "Relativity")

        assert ordered_scatter.marker.color == "#00AA55"
        assert smooth_curve.line.color == "#00AA55"
        assert density_bar.marker.color == "#FFAA00"
        assert density_bar.marker.line.color == "#663300"
        assert numeric_bar.error_y.color == "#111111"

    def test_plotly_knots_absent_by_default(self, fitted_model):
        """In response mode, knots are absent unless show_knots=True."""
        fig = fitted_model.plot(engine="plotly")
        knot_traces = [t for t in fig.data if t.name == "Interior knots"]
        assert len(knot_traces) == 0

    def test_plotly_knots_shown(self, fitted_model):
        fig = fitted_model.plot(engine="plotly", show_knots=True)
        knot_traces = [t for t in fig.data if t.name == "Interior knots"]
        assert knot_traces
        assert knot_traces[0].visible is True
        assert knot_traces[0].mode == "markers"

    def test_plotly_bases_present_with_show_bases(self, fitted_model):
        """Basis contributions are added when show_bases=True (link-scale data, always)."""
        fig = fitted_model.plot(engine="plotly", show_bases=True)
        basis_traces = [t for t in fig.data if t.name == "Basis contributions"]
        assert len(basis_traces) == 1
        assert basis_traces[0].hoverinfo == "skip"

    def test_plotly_bases_shown_in_link_mode(self, fitted_model):
        """Basis contributions appear on the link scale."""
        fig = fitted_model.plot(engine="plotly", scale="link", show_bases=True)
        basis = [t for t in fig.data if t.name == "Basis contributions"]
        assert basis
        assert basis[0].visible is True
        assert basis[0].hoverinfo == "skip"

    def test_plotly_ci_none(self, fitted_model):
        fig = fitted_model.plot(engine="plotly", ci=None)
        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        assert len(fill_traces) == 0

    def test_plotly_simultaneous_ci(self, fitted_model):
        fig = fitted_model.plot(engine="plotly", ci="simultaneous")
        sim_traces = [
            t
            for t in fig.data
            if getattr(t, "fill", None) == "toself" and t.name == "95% simultaneous band"
        ]
        assert len(sim_traces) == 1

    def test_plotly_ci_lines_use_neutral_styling(self, fitted_model):
        fig = fitted_model.plot(engine="plotly", ci="pointwise", ci_style="lines")
        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        assert len(fill_traces) == 0

        ci_trace = next(t for t in fig.data if t.name == "95% pointwise CI")
        assert ci_trace.line.color == "rgba(24, 33, 43, 0.58)"
        assert ci_trace.line.dash == "dash"

    def test_kind_global_is_default(self, fitted_model):
        import plotly.graph_objects as go

        fig = fitted_model.plot(engine="plotly", kind="global")
        assert isinstance(fig, go.Figure)

    def test_kind_local_not_implemented(self, fitted_model):
        with pytest.raises(NotImplementedError, match="kind=.*local"):
            fitted_model.plot("age", kind="local")

    def test_matplotlib_main_effects_unchanged(self, sample_data, fitted_model):
        """Existing matplotlib path still works after plotly wiring."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, sample_weight = sample_data
        fig = fitted_model.plot("age", engine="matplotlib", X=X, sample_weight=sample_weight)
        assert isinstance(fig, Figure)

    def test_plotly_interaction_still_works(self, sample_data, interaction_model):
        """Interaction plotly dispatch unchanged after main-effects wiring."""
        import plotly.graph_objects as go

        X, y, sample_weight = sample_data
        fig = interaction_model.plot(
            "age:region", engine="plotly", X=X, sample_weight=sample_weight
        )
        assert isinstance(fig, go.Figure)

    def test_plotly_interaction_respects_n_points(self, sample_data, interaction_model):
        import plotly.graph_objects as go

        fig = interaction_model.plot("age:region", engine="plotly", n_points=61)
        traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines" and t.name]
        assert any(len(t.x) == 61 for t in traces)

    # ── Scale toggle ──────────────────────────────────────

    def test_scale_response_default(self, fitted_model):
        """Default scale='response' produces relativity traces."""
        fig = fitted_model.plot(engine="plotly")
        names = {t.name for t in fig.data}
        assert "Relativity" in names

    def test_scale_link_y_axis(self, fitted_model):
        """Link-scale view uses η in the y-axis title."""
        fig = fitted_model.plot(engine="plotly", scale="link")
        y_title = fig.layout.yaxis.title.text
        assert "η" in y_title or "link" in y_title.lower()

    def test_scale_link_fitted_line_name(self, fitted_model):
        """Link-scale fitted line keeps name 'Relativity'; hover shows η."""
        fig = fitted_model.plot(engine="plotly", scale="link")
        names = {t.name for t in fig.data}
        assert "Relativity" in names
        # The hovertemplate should reference η on link scale
        rel_trace = next(t for t in fig.data if t.name == "Relativity")
        assert "η" in (rel_trace.hovertemplate or "")

    def test_scale_link_reference_at_zero(self, fitted_model):
        """Link-scale reference line at 0, not 1."""
        fig = fitted_model.plot(engine="plotly", scale="link")
        hlines = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
        assert any(s.y0 == 0.0 for s in hlines)

    def test_scale_response_reference_at_one(self, fitted_model):
        """Response-scale reference line at 1."""
        fig = fitted_model.plot(engine="plotly")
        hlines = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
        assert any(s.y0 == 1.0 for s in hlines)

    def test_scale_link_has_basis_contributions(self, fitted_model):
        """Link scale with show_bases shows basis contribution traces."""
        fig = fitted_model.plot(engine="plotly", scale="link", show_bases=True)
        basis = [t for t in fig.data if t.name == "Basis contributions"]
        assert len(basis) >= 1

    def test_scale_response_basis_traces_hidden(self, fitted_model):
        """Basis traces must not distort response-scale autorange."""
        fig_no_bases = fitted_model.plot(engine="plotly", scale="response", show_bases=False)
        fig_bases = fitted_model.plot(engine="plotly", scale="response", show_bases=True)

        # Basis traces exist but are invisible on response scale
        basis = [t for t in fig_bases.data if t.name == "Basis contributions"]
        assert len(basis) >= 1
        assert all(t.visible is False for t in basis)

        # Y-axis autorange should be identical with and without bases
        range_no = fig_no_bases.layout.yaxis.range
        range_with = fig_bases.layout.yaxis.range
        if range_no is not None and range_with is not None:
            assert range_no == range_with

    def test_link_toggle_restores_basis_visibility(self, fitted_model):
        """Clicking Link after starting on response must make basis traces visible."""
        fig = fitted_model.plot(engine="plotly", scale="response", show_bases=True)
        # Find the scale toggle buttons
        toggles = [m for m in fig.layout.updatemenus if m.type == "buttons"]
        assert toggles
        link_btn = next(b for b in toggles[0].buttons if b.label == "Link")
        restyle = link_btn.args[0]
        # The Link restyle must include visible entries that restore basis traces
        assert "visible" in restyle
        vis = restyle["visible"]
        basis_indices = [i for i, t in enumerate(fig.data) if t.name == "Basis contributions"]
        assert basis_indices
        for idx in basis_indices:
            assert vis[idx] is not False, f"Basis trace {idx} not restored by Link toggle"

    def test_scale_link_knots_on_link_curve(self, fitted_model):
        """Link-scale knots are markers on the η curve."""
        fig = fitted_model.plot(engine="plotly", scale="link", show_knots=True)
        knots = [t for t in fig.data if t.name == "Interior knots"]
        assert knots
        assert knots[0].mode == "markers"

    def test_scale_invalid_raises(self, fitted_model):
        with pytest.raises(ValueError, match="scale"):
            fitted_model.plot("age", scale="bogus")

    def test_simultaneous_ci_wider_than_pointwise_on_link(self, fitted_model):
        """Simultaneous link-scale CI should be wider than pointwise."""
        fig = fitted_model.plot(engine="plotly", ci="both", scale="link")
        sim_fill = [
            t
            for t in fig.data
            if getattr(t, "fill", None) == "toself" and "simultaneous" in (t.name or "").lower()
        ]
        pw_fill = [
            t
            for t in fig.data
            if getattr(t, "fill", None) == "toself" and "pointwise" in (t.name or "").lower()
        ]
        assert len(sim_fill) == 1 and len(pw_fill) == 1
        import numpy as np

        sim_y = np.asarray(sim_fill[0].y)
        pw_y = np.asarray(pw_fill[0].y)
        n = len(sim_y) // 2
        sim_span = sim_y[:n] - sim_y[n:][::-1]  # upper - lower
        pw_span = pw_y[:n] - pw_y[n:][::-1]
        assert np.all(sim_span >= pw_span - 1e-12)
        assert np.any(sim_span > pw_span + 1e-12)

    def test_term_dropdown_resets_to_response_scale(self, fitted_model):
        """Switching terms via dropdown should reset to response scale."""
        fig = fitted_model.plot(engine="plotly")
        dropdowns = [m for m in fig.layout.updatemenus if m.type == "dropdown"]
        if not dropdowns:
            pytest.skip("Single-term model, no dropdown")
        btn = dropdowns[0].buttons[0]
        relayout = btn.args[1]
        assert relayout.get("yaxis.title.text") == "Relativity"
        # Scale toggle should be reset to index 0 (Response)
        assert relayout.get("updatemenus[0].active") == 0

    def test_plotly_main_effects_import_error(self, fitted_model):
        """Clear ImportError when plotly is missing."""
        from unittest.mock import patch

        with patch.dict(
            "sys.modules",
            {"plotly": None, "plotly.graph_objects": None, "plotly.subplots": None},
        ):
            with pytest.raises(ImportError, match="plotly is required"):
                fitted_model.plot(engine="plotly")
