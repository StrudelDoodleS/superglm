"""Tests for model.plot_diagnostics() — R-style 4-panel residual diagnostic plots (T1-T5)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from superglm import (
    Numeric,
    Spline,
    SuperGLM,
)
from superglm.distributions import (
    Gamma,
    NegativeBinomial,
    Tweedie,
)

# ── Fixtures ─────────────────────────────────────────────────────


def _make_poisson_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_gaussian_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = 2.0 + 1.5 * x
    y = rng.normal(mu, 0.5)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_gamma_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = np.exp(1.0 + 0.2 * x)
    shape = 5.0
    y = rng.gamma(shape, mu / shape, n)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_binomial_data(rng, n=500):
    x = rng.uniform(-2, 2, n)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 0.8 * x)))
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_nb2_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = np.exp(0.5 + 0.3 * x)
    theta = 2.0
    # NB2 parameterisation: Y ~ NB(theta, theta/(theta+mu))
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_tweedie_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = np.exp(1.0 + 0.2 * x)
    # Approximate tweedie with Poisson-Gamma compound
    # For simplicity, use Poisson * Gamma
    lam = mu**0.5
    n_claims = rng.poisson(lam)
    y = np.zeros(n)
    for i in range(n):
        if n_claims[i] > 0:
            y[i] = rng.gamma(2.0, mu[i] / (2.0 * lam[i]), n_claims[i]).sum()
    X = pd.DataFrame({"x": x})
    return X, y


# ── T1: Smoke test all families ──────────────────────────────────


class TestPlotDiagnosticsSmokeAllFamilies:
    """T1: For each family, fit a small model, call plot_diagnostics(),
    assert returns Figure with 4 axes."""

    def test_poisson(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng)
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_gaussian(self):
        rng = np.random.default_rng(42)
        X, y = _make_gaussian_data(rng)
        model = SuperGLM(
            family="gaussian", features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_gamma(self):
        rng = np.random.default_rng(42)
        X, y = _make_gamma_data(rng)
        model = SuperGLM(family=Gamma(), features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_binomial(self):
        rng = np.random.default_rng(42)
        X, y = _make_binomial_data(rng)
        model = SuperGLM(
            family="binomial", features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_nb2(self):
        rng = np.random.default_rng(42)
        X, y = _make_nb2_data(rng)
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            features={"x": Spline(n_knots=5)},
            selection_penalty=0.0,
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_tweedie(self):
        rng = np.random.default_rng(42)
        X, y = _make_tweedie_data(rng)
        model = SuperGLM(
            family=Tweedie(p=1.5), features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4


# ── T2: Residual type forwarding ─────────────────────────────────


class TestResidualTypeForwarding:
    """T2: Call with each valid residual_type; invalid → ValueError."""

    @pytest.fixture
    def fitted(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=200)
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0.0)
        model.fit(X, y)
        return model, X, y

    @pytest.mark.parametrize("rtype", ["deviance", "pearson", "response", "working", "quantile"])
    def test_valid_types(self, fitted, rtype):
        model, X, y = fitted
        fig = model.plot_diagnostics(X, y, residual_type=rtype)
        assert isinstance(fig, Figure)
        # Panel 1 y-axis label should contain residual type name
        ax1 = fig.get_axes()[0]
        assert rtype.lower() in ax1.get_ylabel().lower()

    def test_invalid_type_raises(self, fitted):
        model, X, y = fitted
        with pytest.raises(ValueError, match="Invalid residual_type"):
            model.plot_diagnostics(X, y, residual_type="invalid")


# ── T3: Panel content verification ───────────────────────────────


class TestPanelContent:
    """T3: Verify panel contents are correct."""

    @pytest.fixture
    def fitted_with_metrics(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=300)
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        mu = model.predict(X)
        return fig, mu

    def test_panel1_xdata_matches_mu(self, fitted_with_metrics):
        fig, mu = fitted_with_metrics
        ax1 = fig.get_axes()[0]
        # The scatter plot data should match mu
        scatter = ax1.collections[0]
        offsets = scatter.get_offsets()
        np.testing.assert_allclose(np.sort(offsets[:, 0]), np.sort(mu), rtol=1e-5)

    def test_panel2_has_reference_line(self, fitted_with_metrics):
        fig, _ = fitted_with_metrics
        ax2 = fig.get_axes()[1]
        # Should have at least one line (the 45-degree ref)
        lines = ax2.get_lines()
        assert len(lines) >= 1

    def test_panel4_has_cooks_contours(self, fitted_with_metrics):
        fig, _ = fitted_with_metrics
        ax4 = fig.get_axes()[3]
        # Should have contour lines (Cook's D = 0.5, 1.0 → 4 lines)
        lines = ax4.get_lines()
        assert len(lines) >= 2  # At least the Cook's D contours


# ── T4: Edge case — intercept-only model ─────────────────────────


class TestInterceptOnly:
    """T4: Intercept-only model should not crash."""

    def test_intercept_only_poisson(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.poisson(2.0, n).astype(float)
        X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=10.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4


# ── T5: Custom figsize ───────────────────────────────────────────


class TestCustomFigsize:
    """T5: figsize parameter is respected."""

    def test_figsize(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=200)
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, figsize=(12, 10))
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1
        assert abs(h - 10) < 0.1


# ── T6: sample_weight and offset ─────────────────────────────────


class TestSampleWeightAndOffset:
    """Smoke test: sample_weight and offset do not crash."""

    def test_with_sample_weight(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=300)
        w = rng.uniform(0.5, 2.0, len(y))
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y, sample_weight=w)
        fig = model.plot_diagnostics(X, y, sample_weight=w)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_with_offset(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=300)
        offset = rng.uniform(-0.5, 0.5, len(y))
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y, offset=offset)
        fig = model.plot_diagnostics(X, y, offset=offset)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4
