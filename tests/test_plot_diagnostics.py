"""Tests for model.plot_diagnostics() — GLM/GAM diagnostic plots."""

from __future__ import annotations

import warnings

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
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y


def _make_tweedie_data(rng, n=500):
    x = rng.uniform(0, 5, n)
    mu = np.exp(1.0 + 0.2 * x)
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
    """For each family, fit a small model, call plot_diagnostics(),
    assert returns Figure with 4 axes and correct panel titles."""

    N_SIM = 10  # keep fast

    def _check_panels(self, fig):
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == 4
        # Panel titles
        assert "Q-Q" in axes[0].get_title()
        assert "Calibration" in axes[1].get_title()
        assert "Linear Predictor" in axes[2].get_title()
        assert "Residual Distribution" in axes[3].get_title()

    def test_poisson(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng)
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)

    def test_gaussian(self):
        rng = np.random.default_rng(42)
        X, y = _make_gaussian_data(rng)
        model = SuperGLM(
            family="gaussian", features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)

    def test_gamma(self):
        rng = np.random.default_rng(42)
        X, y = _make_gamma_data(rng)
        model = SuperGLM(family=Gamma(), features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)

    def test_binomial(self):
        rng = np.random.default_rng(42)
        X, y = _make_binomial_data(rng)
        model = SuperGLM(
            family="binomial", features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)

    def test_nb2(self):
        rng = np.random.default_rng(42)
        X, y = _make_nb2_data(rng)
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            features={"x": Spline(n_knots=5)},
            selection_penalty=0.0,
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)

    def test_tweedie(self):
        rng = np.random.default_rng(42)
        X, y = _make_tweedie_data(rng)
        model = SuperGLM(
            family=Tweedie(p=1.5), features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=self.N_SIM)
        self._check_panels(fig)


# ── T2: Deprecation warning for residual_type ────────────────────


class TestResidualTypeDeprecation:
    """residual_type is deprecated; non-default values emit FutureWarning."""

    @pytest.fixture
    def fitted(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=200)
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0.0)
        model.fit(X, y)
        return model, X, y

    def test_auto_no_warning(self, fitted):
        model, X, y = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig = model.plot_diagnostics(X, y, n_sim=5)
        assert isinstance(fig, Figure)

    def test_explicit_type_warns(self, fitted):
        model, X, y = fitted
        with pytest.warns(FutureWarning, match="residual_type is deprecated"):
            model.plot_diagnostics(X, y, residual_type="deviance", n_sim=5)


# ── T3: Panel content verification ───────────────────────────────


class TestPanelContent:
    """Verify panel contents are correct."""

    @pytest.fixture
    def fitted_fig(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=300)
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=10)
        return fig

    def test_panel1_has_envelope(self, fitted_fig):
        ax1 = fitted_fig.get_axes()[0]
        assert "Q-Q Envelope" in ax1.get_title()
        # Should have fill_between (PolyCollection) for the envelope
        assert len(ax1.collections) >= 1

    def test_panel2_calibration(self, fitted_fig):
        ax2 = fitted_fig.get_axes()[1]
        assert "Calibration" in ax2.get_title()
        lines = ax2.get_lines()
        assert len(lines) >= 1  # y=x reference line

    def test_panel3_has_zero_line(self, fitted_fig):
        ax3 = fitted_fig.get_axes()[2]
        lines = ax3.get_lines()
        assert len(lines) >= 1  # zero reference + trend

    def test_panel4_has_normal_overlay(self, fitted_fig):
        ax4 = fitted_fig.get_axes()[3]
        lines = ax4.get_lines()
        assert len(lines) >= 1  # N(0,1) density


# ── T4: Edge case — intercept-only model ─────────────────────────


class TestInterceptOnly:
    """Intercept-only model should not crash."""

    def test_intercept_only_poisson(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.poisson(2.0, n).astype(float)
        X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=10.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, n_sim=5)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4


# ── T5: Custom figsize ───────────────────────────────────────────


class TestCustomFigsize:
    """figsize parameter is respected."""

    def test_figsize(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=200)
        model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0.0)
        model.fit(X, y)
        fig = model.plot_diagnostics(X, y, figsize=(12, 10), n_sim=5)
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
        fig = model.plot_diagnostics(X, y, sample_weight=w, n_sim=5)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4

    def test_with_offset(self):
        rng = np.random.default_rng(42)
        X, y = _make_poisson_data(rng, n=300)
        offset = rng.uniform(-0.5, 0.5, len(y))
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0)
        model.fit(X, y, offset=offset)
        fig = model.plot_diagnostics(X, y, offset=offset, n_sim=5)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4
