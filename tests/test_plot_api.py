"""Tests for the unified model.plot() API."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, Spline, SuperGLM


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
def interaction_model(sample_data):
    X, y, exposure = sample_data
    model = SuperGLM(
        lambda1=0.01,
        features={
            "age": Spline(n_knots=8, penalty="ssp"),
            "region": Categorical(base="first"),
        },
        interactions=[("age", "region")],
    )
    model.fit(X, y, exposure=exposure)
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

        X, y, exposure = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=exposure, show_density=True)
        assert len(fig.get_axes()) >= 2

    def test_density_without_sample_weight(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X, y, _ = sample_data
        fig = fitted_model.plot("age", X=X, show_density=True)
        # Without sample_weight, no density strip — just the main panel
        assert isinstance(fig, Figure)

    def test_density_disabled(self, sample_data, fitted_model):
        import matplotlib

        matplotlib.use("Agg")

        X, y, exposure = sample_data
        fig = fitted_model.plot("age", X=X, sample_weight=exposure, show_density=False)
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

    def test_unknown_feature_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Feature not found"):
            fitted_model.plot("nonexistent")

    def test_unknown_interaction_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Interaction not found"):
            fitted_model.plot("age:nonexistent")

    def test_plotly_for_main_effects_raises(self, fitted_model):
        with pytest.raises(ValueError, match="engine=.*only supported"):
            fitted_model.plot("age", engine="plotly")

    def test_unknown_engine_raises(self, interaction_model):
        with pytest.raises(ValueError, match="Unknown engine"):
            interaction_model.plot("age:region", engine="bokeh")

    def test_unfitted_model_raises(self):
        model = SuperGLM(features={"x": Spline(n_knots=5)})
        with pytest.raises(RuntimeError, match="fitted"):
            model.plot()
