"""Tests for model.plot_diagnostics() — GLM/GAM diagnostic plots."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from superglm import (
    Numeric,
    Spline,
    SuperGLM,
)
from superglm.distributions import (
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
)
from superglm.plotting import diagnostics as diagnostics_module

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


class TestWeightContractDiagnostics:
    class _RecordingRng:
        def poisson(self, lam):
            self.poisson_lam = np.asarray(lam)
            return np.zeros_like(self.poisson_lam, dtype=np.int64)

        def normal(self, loc, scale):
            self.normal_loc = np.asarray(loc)
            self.normal_scale = np.asarray(scale)
            return np.asarray(loc, dtype=np.float64)

    def test_non_tweedie_simulation_is_frequency_weight_invariant(self):
        mu = np.array([0.4, 1.2, 3.5])
        weights = np.array([1.0, 4.0, 9.0])
        rng = self._RecordingRng()

        diagnostics_module._simulate_response(Poisson(), mu, 1.0, weights, rng)
        np.testing.assert_array_equal(rng.poisson_lam, mu)

        diagnostics_module._simulate_response(Gaussian(), mu, 2.25, weights, rng)
        np.testing.assert_array_equal(rng.normal_loc, mu)
        assert rng.normal_scale == pytest.approx(1.5)

    def test_tweedie_simulation_uses_prior_weight_dispersion(self, monkeypatch):
        captured = {}

        def fake_generate(n, mu, phi, p, rng=None):
            captured.update(n=n, mu=np.asarray(mu), phi=np.asarray(phi), p=p, rng=rng)
            return np.zeros(n)

        monkeypatch.setattr(
            "superglm.profiling.tweedie.generate_tweedie_cpg",
            fake_generate,
        )
        mu = np.array([0.7, 1.3, 2.1])
        weights = np.array([0.25, 1.0, 4.0])
        rng = np.random.default_rng(219)

        simulated = diagnostics_module._simulate_response(
            Tweedie(p=1.5),
            mu,
            0.8,
            weights,
            rng,
        )

        np.testing.assert_array_equal(simulated, np.zeros(len(mu)))
        np.testing.assert_allclose(captured["phi"], 0.8 / weights)
        assert captured["n"] == len(mu)
        assert captured["p"] == 1.5
        assert captured["rng"] is rng

    @pytest.mark.parametrize(
        ("family", "weights", "match"),
        [
            (Gaussian(), np.array([1.0, -0.1, 2.0]), "nonnegative"),
            (Gaussian(), np.zeros(3), "at least one positive"),
            (Gaussian(), np.array([1.0, np.nan, 2.0]), "finite"),
            (Tweedie(p=1.5), np.array([1.0, 0.0, 2.0]), "strictly positive"),
            (Tweedie(p=1.5), np.array([[1.0], [2.0], [3.0]]), "one-dimensional"),
        ],
    )
    def test_public_plot_rejects_invalid_family_weights(self, family, weights, match):
        class ValidationOnlyModel:
            _distribution = family

        with pytest.raises(ValueError, match=match):
            diagnostics_module.plot_diagnostics(
                ValidationOnlyModel(),
                None,
                np.ones(3),
                sample_weight=weights,
                n_sim=1,
            )

    @pytest.mark.parametrize("family", ["gaussian", "poisson"])
    def test_integer_frequency_weight_plots_match_literal_rows(self, family):
        rng = np.random.default_rng(219)
        x = np.linspace(-1.5, 1.5, 24)
        X = pd.DataFrame({"x": x})
        if family == "gaussian":
            y = 0.4 + 0.7 * x + rng.normal(scale=0.8, size=len(x))
        else:
            y = rng.poisson(np.exp(0.2 + 0.3 * x)).astype(float)
        weights = np.resize(np.array([1, 3, 2, 4]), len(x)).astype(float)
        model = SuperGLM(
            family=family,
            features={"x": Numeric()},
            selection_penalty=0.0,
            weight_semantics="frequency",
        ).fit(X, y, sample_weight=weights)
        repeated_rows = np.repeat(np.arange(len(y)), weights.astype(np.intp))
        repeated_X = X.iloc[repeated_rows].reset_index(drop=True)
        repeated_y = y[repeated_rows]

        weighted_figure = model.plot_diagnostics(
            X,
            y,
            sample_weight=weights,
            n_sim=3,
            seed=219,
        )
        repeated_figure = model.plot_diagnostics(
            repeated_X,
            repeated_y,
            n_sim=3,
            seed=219,
        )
        try:
            weighted_axes = weighted_figure.get_axes()
            repeated_axes = repeated_figure.get_axes()
            assert [axis.get_title() for axis in weighted_axes] == [
                axis.get_title() for axis in repeated_axes
            ]

            for weighted_line, repeated_line in zip(
                weighted_axes[0].lines,
                repeated_axes[0].lines,
                strict=True,
            ):
                np.testing.assert_allclose(weighted_line.get_xdata(), repeated_line.get_xdata())
                np.testing.assert_allclose(weighted_line.get_ydata(), repeated_line.get_ydata())

            np.testing.assert_allclose(
                weighted_axes[1].collections[0].get_offsets(),
                repeated_axes[1].collections[0].get_offsets(),
            )
            np.testing.assert_allclose(
                weighted_axes[2].collections[0].get_offsets(),
                repeated_axes[2].collections[0].get_offsets(),
            )
            weighted_bars = np.array(
                [
                    (bar.get_x(), bar.get_width(), bar.get_height())
                    for bar in weighted_axes[3].patches
                ]
            )
            repeated_bars = np.array(
                [
                    (bar.get_x(), bar.get_width(), bar.get_height())
                    for bar in repeated_axes[3].patches
                ]
            )
            np.testing.assert_allclose(weighted_bars, repeated_bars)

            metrics = model.metrics(X, y, sample_weight=weights)
            expected_df = max(float(np.sum(weights)) - metrics.effective_df, 1.0)
            expected_ratio = metrics.pearson_chi2 / expected_df
            assert f"χ²/df={expected_ratio:.2f}" in weighted_axes[3].get_title()
        finally:
            plt.close(weighted_figure)
            plt.close(repeated_figure)
