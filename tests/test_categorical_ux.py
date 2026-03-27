"""Tests for categorical UX improvements: line+markers, exposure bars, show_knots, collapse_levels."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    LevelGrouping,
    OrderedCategorical,
    SuperGLM,
    collapse_levels,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


# ── Shared fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_data():
    """Synthetic data with ordered + unordered categorical features."""
    rng = np.random.default_rng(42)
    n = 2000
    bands = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    age_band = rng.choice(bands, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
    region = rng.choice(["A", "B", "C", "D"], n, p=[0.3, 0.3, 0.25, 0.15])
    sample_weight = rng.uniform(0.3, 1.0, n)
    midpoints = {
        "18-25": 21.5,
        "26-35": 30.5,
        "36-45": 40.5,
        "46-55": 50.5,
        "56-65": 60.5,
        "65+": 70.0,
    }
    x_numeric = np.array([midpoints[v] for v in age_band])
    mu = np.exp(-2.0 + 0.01 * (x_numeric - 45) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age_band": age_band, "region": region})
    return X, y, sample_weight, midpoints


@pytest.fixture
def fitted_model(sample_data):
    X, y, sample_weight, midpoints = sample_data
    model = SuperGLM(
        features={
            "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
            "region": Categorical(base="first"),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, X, sample_weight


# ═══════════════════════════════════════════════════════════════════
# Feature A: Line+markers for categorical relativities
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
class TestFeatureA:
    def _get_plotly_fig(self, model, X, sample_weight, term=None):
        return model.plot(
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            terms=term,
        )

    def test_no_bar_traces_for_relativities(self, fitted_model):
        """T-A1: No go.Bar traces with name='Relativity'."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "Relativity"]
        assert len(bar_traces) == 0

    def test_ordered_has_markers(self, fitted_model):
        """T-A1/T-A2: Ordered categoricals have scatter traces with markers."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="age_band")
        scatter_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter) and t.mode is not None and "markers" in t.mode
        ]
        assert len(scatter_traces) >= 1

    def test_ordered_smooth_curve_retained(self, fitted_model):
        """T-A2: Ordered categoricals retain smooth curve overlay."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="age_band")
        curve_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter) and t.mode == "lines" and t.name == "Smooth curve"
        ]
        assert len(curve_traces) >= 1

    def test_unordered_lines_markers(self, fitted_model):
        """T-A3: Unordered categoricals show lines+markers with error bars."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        lm_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines+markers"]
        assert len(lm_traces) >= 1

    def test_no_text_mode_traces(self, fitted_model):
        """T-A4: No text-label traces (mode='text') in output."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw)
        text_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "text"]
        assert len(text_traces) == 0

    def test_ordered_trace_count(self, fitted_model):
        """T-A4: Ordered categorical: markers + curve = 2 traces (plus exposure bar)."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="age_band")
        # Should have: exposure bar, markers, smooth curve
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # At least markers + curve
        assert len(scatter_traces) >= 2

    def test_unordered_trace_count(self, fitted_model):
        """T-A4: Unordered categorical: 1 lines+markers trace (plus exposure bar)."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        relativity_traces = [
            t for t in fig.data if isinstance(t, go.Scatter) and t.name == "Relativity"
        ]
        assert len(relativity_traces) == 1


# ═══════════════════════════════════════════════════════════════════
# Feature B: Exposure bars on dual y-axis
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
class TestFeatureB:
    def _get_plotly_fig(self, model, X, sample_weight, term=None):
        return model.plot(
            engine="plotly",
            X=X,
            sample_weight=sample_weight,
            terms=term,
        )

    def test_exposure_bars_absolute(self, fitted_model):
        """T-B1: Exposure bar y-values are raw sums (not normalized to [0,1])."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "Exposure"]
        assert len(bar_traces) >= 1
        y_vals = np.array(bar_traces[0].y)
        # Raw sums should be > 1 (since each weight is 0.3-1.0 and n=2000)
        assert np.max(y_vals) > 1.0

    def test_exposure_bars_top_panel(self, fitted_model):
        """T-B2: Exposure bars render on right y-axis (y3) in top panel."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "Exposure"]
        assert len(bar_traces) >= 1
        assert bar_traces[0].yaxis == "y3"

    def test_exposure_bars_before_relativity(self, fitted_model):
        """T-B3: Exposure bar trace appears before relativity scatter (z-order)."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        exposure_idx = None
        relativity_idx = None
        for i, t in enumerate(fig.data):
            if isinstance(t, go.Bar) and t.name == "Exposure" and exposure_idx is None:
                exposure_idx = i
            if isinstance(t, go.Scatter) and t.name == "Relativity" and relativity_idx is None:
                relativity_idx = i
        assert exposure_idx is not None
        assert relativity_idx is not None
        assert exposure_idx < relativity_idx

    def test_exposure_bar_opacity(self, fitted_model):
        """T-B4: Exposure bar opacity <= 0.4."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "Exposure"]
        assert len(bar_traces) >= 1
        assert bar_traces[0].opacity <= 0.4

    def test_yaxis3_no_hardcoded_range(self, fitted_model):
        """T-B5: yaxis3 does NOT have hardcoded range [0.0, 1.05]."""
        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        y3 = fig.layout.yaxis3
        if y3.range is not None:
            assert list(y3.range) != [0.0, 1.05]

    def test_exposure_hovertemplate(self, fitted_model):
        """T-B6: Exposure hovertemplate contains 'Exposure' not 'Relative density'."""
        import plotly.graph_objects as go

        model, X, sw = fitted_model
        fig = self._get_plotly_fig(model, X, sw, term="region")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "Exposure"]
        assert len(bar_traces) >= 1
        ht = bar_traces[0].hovertemplate
        assert "Exposure" in ht
        assert "Relative density" not in ht


# ═══════════════════════════════════════════════════════════════════
# Feature C: show_knots for OrderedCategorical
# ═══════════════════════════════════════════════════════════════════


class TestFeatureC:
    def test_term_inference_spline_metadata(self, sample_data):
        """T-C1: OrderedCategorical(basis='spline') term_inference has spline metadata."""
        from superglm.inference import SplineMetadata

        X, y, sw, midpoints = sample_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sw)
        ti = model.term_inference("age_band")
        assert ti.spline is not None
        assert isinstance(ti.spline, SplineMetadata)
        assert ti.spline.interior_knots is not None
        assert len(ti.spline.interior_knots) > 0

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_show_knots_true(self, sample_data):
        """T-C2: plot(show_knots=True) renders knot diamonds."""
        X, y, sw, midpoints = sample_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sw)
        fig = model.plot(engine="plotly", X=X, sample_weight=sw, terms="age_band", show_knots=True)
        import plotly.graph_objects as go

        knot_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.name == "Interior knots"
            and t.marker is not None
            and t.marker.symbol == "diamond"
        ]
        assert len(knot_traces) >= 1

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_show_knots_false(self, sample_data):
        """T-C3: plot(show_knots=False) produces no knot trace."""
        X, y, sw, midpoints = sample_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sw)
        fig = model.plot(engine="plotly", X=X, sample_weight=sw, terms="age_band", show_knots=False)
        knot_traces = [t for t in fig.data if getattr(t, "name", None) == "Interior knots"]
        assert len(knot_traces) == 0

    def test_step_basis_no_spline_metadata(self, sample_data):
        """T-C4: OrderedCategorical(basis='step') has ti.spline is None."""
        X, y, sw, _ = sample_data
        levels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        model = SuperGLM(
            features={"age_band": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sw)
        ti = model.term_inference("age_band")
        assert ti.spline is None


# ═══════════════════════════════════════════════════════════════════
# Feature D: collapse_levels + grouping
# ═══════════════════════════════════════════════════════════════════


class TestCollapseLevels:
    def test_from_level(self):
        """T-D1: from_level collapses levels at position >= from_level."""
        g = collapse_levels(["1", "2", "3", "4", "5"], from_level="4")
        assert g.original_to_group["4"] == "4+"
        assert g.original_to_group["5"] == "4+"
        assert g.original_to_group["1"] == "1"
        assert g.original_to_group["2"] == "2"
        assert g.original_to_group["3"] == "3"
        assert "4+" in g.grouped_levels
        assert "4" not in g.grouped_levels
        assert "5" not in g.grouped_levels

    def test_below(self):
        """T-D2: below collapses levels before the cutoff."""
        g = collapse_levels(["1", "2", "3", "4", "5"], below="3")
        assert g.original_to_group["1"] == "<3"
        assert g.original_to_group["2"] == "<3"
        assert g.original_to_group["3"] == "3"
        assert g.original_to_group["4"] == "4"

    def test_groups_explicit(self):
        """T-D3: Explicit groups mapping."""
        g = collapse_levels(["CA", "FL", "NY", "TX"], groups={"South": ["TX", "FL"]})
        assert g.original_to_group["TX"] == "South"
        assert g.original_to_group["FL"] == "South"
        assert g.original_to_group["NY"] == "NY"
        assert g.original_to_group["CA"] == "CA"

    def test_duplicate_membership_raises(self):
        """T-D4: Level in multiple groups raises ValueError."""
        with pytest.raises(ValueError, match="multiple groups"):
            collapse_levels(["A", "B"], groups={"G1": ["A"], "G2": ["A"]})

    def test_unknown_level_raises(self):
        """T-D5: Unknown level in groups raises ValueError."""
        with pytest.raises(ValueError, match="not found in data"):
            collapse_levels(["A", "B"], groups={"G1": ["Z"]})

    def test_mixed_modes_raises(self):
        """T-D6: from_level + groups raises ValueError."""
        with pytest.raises(ValueError, match="Cannot mix"):
            collapse_levels(["A", "B"], from_level="A", groups={"G": ["B"]})

    def test_from_level_and_below_combined(self):
        """from_level + below can be combined."""
        g = collapse_levels(["1", "2", "3", "4", "5"], from_level="4", below="2")
        assert g.original_to_group["1"] == "<2"
        assert g.original_to_group["2"] == "2"
        assert g.original_to_group["4"] == "4+"
        assert g.original_to_group["5"] == "4+"

    def test_all_levels_single_group(self):
        """T-D11: All levels collapse to one group."""
        g = collapse_levels(["A", "B", "C"], groups={"All": ["A", "B", "C"]})
        assert g.grouped_levels == ["All"]
        assert all(v == "All" for v in g.original_to_group.values())

    def test_level_grouping_frozen(self):
        """LevelGrouping is a frozen dataclass."""
        g = collapse_levels(["A", "B"], groups={"G": ["A"]})
        with pytest.raises(AttributeError):
            g.grouped_levels = []

    def test_exports(self):
        """T-D7 (partial): LevelGrouping and collapse_levels importable from superglm."""
        from superglm import collapse_levels as cl

        assert LevelGrouping is not None
        assert cl is not None


class TestOrderedCategoricalGrouping:
    def test_fit_predict_with_grouping(self, sample_data):
        """T-D7: OrderedCategorical with grouping fits and predicts."""
        X, y, sw, midpoints = sample_data
        g = collapse_levels(
            X["age_band"],
            from_level="56-65",
            order=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, grouping=g
                ),
            },
        )
        model.fit(X, y, sample_weight=sw)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_term_inference_expanded_levels(self, sample_data):
        """T-D7: term_inference returns ALL original levels, grouped share same relativity."""
        X, y, sw, midpoints = sample_data
        g = collapse_levels(
            X["age_band"],
            from_level="56-65",
            order=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, grouping=g
                ),
            },
        )
        model.fit(X, y, sample_weight=sw)
        ti = model.term_inference("age_band")
        # Should have all 6 original levels
        assert len(ti.levels) == 6
        assert set(ti.levels) == set(g.all_original_levels)
        # 56-65 and 65+ should have same relativity (both map to "56-65+")
        idx_56 = list(ti.levels).index("56-65")
        idx_65 = list(ti.levels).index("65+")
        assert ti.relativity[idx_56] == ti.relativity[idx_65]

    def test_predict_on_original_levels(self, sample_data):
        """T-D9: predict() works on data with original (ungrouped) level names."""
        X, y, sw, midpoints = sample_data
        g = collapse_levels(
            X["age_band"],
            from_level="56-65",
            order=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, grouping=g
                ),
            },
        )
        model.fit(X, y, sample_weight=sw)
        # Predict on original data containing ungrouped level names
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))


class TestCategoricalGrouping:
    def test_fit_predict_with_grouping(self, sample_data):
        """T-D8: Categorical with grouping fits and predicts."""
        X, y, sw, _ = sample_data
        g = collapse_levels(X["region"], groups={"South": ["A", "B"]})
        model = SuperGLM(
            features={"region": Categorical(base="first", grouping=g)},
        )
        model.fit(X, y, sample_weight=sw)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_term_inference_expanded_levels(self, sample_data):
        """T-D8: term_inference returns expanded levels with grouped sharing relativity."""
        X, y, sw, _ = sample_data
        g = collapse_levels(X["region"], groups={"South": ["A", "B"]})
        model = SuperGLM(
            features={"region": Categorical(base="first", grouping=g)},
        )
        model.fit(X, y, sample_weight=sw)
        ti = model.term_inference("region")
        # Should have all 4 original levels
        assert len(ti.levels) == 4
        # A and B should have the same relativity (both map to South)
        idx_a = list(ti.levels).index("A")
        idx_b = list(ti.levels).index("B")
        assert ti.relativity[idx_a] == ti.relativity[idx_b]

    def test_predict_on_original_levels(self, sample_data):
        """T-D9: predict() works on original levels."""
        X, y, sw, _ = sample_data
        g = collapse_levels(X["region"], groups={"South": ["A", "B"]})
        model = SuperGLM(
            features={"region": Categorical(base="first", grouping=g)},
        )
        model.fit(X, y, sample_weight=sw)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
class TestGroupingPlot:
    def test_plot_with_grouping(self, sample_data):
        """T-D10: Plot with grouping produces figure without error."""
        X, y, sw, midpoints = sample_data
        g = collapse_levels(
            X["age_band"],
            from_level="56-65",
            order=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, grouping=g
                ),
            },
        )
        model.fit(X, y, sample_weight=sw)
        fig = model.plot(engine="plotly", X=X, sample_weight=sw, terms="age_band")
        assert fig is not None
