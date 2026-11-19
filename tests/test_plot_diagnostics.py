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
    Binomial,
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

        def gamma(self, shape, scale):
            self.gamma_shape = np.asarray(shape)
            self.gamma_scale = np.asarray(scale)
            return np.asarray(shape, dtype=np.float64) * np.asarray(scale, dtype=np.float64)

        def binomial(self, n, p):
            self.binomial_p = np.asarray(p)
            return np.zeros_like(self.binomial_p, dtype=np.int64)

    def test_non_tweedie_simulation_is_frequency_weight_invariant(self):
        """Copies of a draw are draws, so replication leaves the marginal alone."""
        mu = np.array([0.4, 1.2, 3.5])
        weights = np.array([1.0, 4.0, 9.0])
        rng = self._RecordingRng()

        diagnostics_module._simulate_response(
            Poisson(), mu, 1.0, weights, rng, weight_semantics="frequency"
        )
        np.testing.assert_array_equal(rng.poisson_lam, mu)

        diagnostics_module._simulate_response(
            Gaussian(), mu, 2.25, weights, rng, weight_semantics="frequency"
        )
        np.testing.assert_array_equal(rng.normal_loc, mu)
        assert rng.normal_scale == pytest.approx(1.5)

    def test_prior_weight_simulation_carries_the_row_marginal(self):
        """The crossed case the family rule could not express.

        Each parameter below is the one ``_quantile_residuals`` inverts for
        that family, so simulating any other way puts a correct fit outside its
        own envelope.
        """
        mu = np.array([0.4, 1.2, 3.5])
        weights = np.array([0.5, 4.0, 9.0])
        phi = 2.25
        rng = self._RecordingRng()

        diagnostics_module._simulate_response(
            Poisson(), mu, 1.0, weights, rng, weight_semantics="prior"
        )
        np.testing.assert_allclose(rng.poisson_lam, weights * mu)

        diagnostics_module._simulate_response(
            Gaussian(), mu, phi, weights, rng, weight_semantics="prior"
        )
        np.testing.assert_allclose(rng.normal_scale, np.sqrt(phi / weights))

        diagnostics_module._simulate_response(
            Gamma(), mu, phi, weights, rng, weight_semantics="prior"
        )
        np.testing.assert_allclose(rng.gamma_shape, weights / phi)
        np.testing.assert_allclose(rng.gamma_scale, mu * phi / weights)

        # Binomial is the exception, and deliberately so: with y pinned to
        # {0, 1} the prior likelihood's two reachable masses do not sum to one
        # for w != 1, so no {0,1}-supported marginal reproduces it. Both this
        # helper and the residual stay contract-invariant, which is what keeps
        # the envelope a valid reference band.
        probability = np.array([0.2, 0.5, 0.7])
        diagnostics_module._simulate_response(
            Binomial(), probability, phi, weights, rng, weight_semantics="prior"
        )
        np.testing.assert_allclose(rng.binomial_p, probability)

    @pytest.mark.parametrize(
        ("semantics", "expected"),
        [("prior", True), ("frequency", False)],
    )
    def test_tweedie_simulation_follows_the_contract_not_the_family(
        self, monkeypatch, semantics, expected
    ):
        captured = {}

        def fake_generate(n, mu, phi, p, rng=None):
            captured.update(n=n, mu=np.asarray(mu), phi=np.asarray(phi), p=p, rng=rng)
            return np.zeros(n)

        monkeypatch.setattr(
            "superglm._tweedie.generate_tweedie_cpg",
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
            weight_semantics=semantics,
        )

        np.testing.assert_array_equal(simulated, np.zeros(len(mu)))
        want = 0.8 / weights if expected else np.full_like(weights, 0.8)
        np.testing.assert_allclose(captured["phi"], want)
        assert captured["n"] == len(mu)
        assert captured["p"] == 1.5
        assert captured["rng"] is rng

    def test_row_replication_follows_the_contract_not_the_family(self):
        """Expanding rows is what "frequency" means, so only it expands."""
        rng = np.random.default_rng(3)
        n = 40
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = 1.0 + 2.0 * frame["x"].to_numpy() + rng.normal(0.0, 0.2, n)
        model = SuperGLM(family=Gaussian(), features={"x": Numeric()})
        model.fit(frame, y)
        metrics = model.metrics(frame, y)
        mu, eta = metrics._mu, metrics.eta
        qresid = metrics.residuals("quantile", seed=5)

        # A zero weight leaves under both readings; only frequency expands the rest.
        weights = np.full(n, 2.0)
        weights[:3] = 0.0

        prior_rows = diagnostics_module._diagnostic_rows(
            model, y, mu, eta, weights, qresid, 5, rng, weight_semantics="prior"
        )
        assert len(prior_rows[0]) == n - 3
        np.testing.assert_allclose(prior_rows[3], weights[weights > 0.0])

        frequency_rows = diagnostics_module._diagnostic_rows(
            model, y, mu, eta, weights, qresid, 5, rng, weight_semantics="frequency"
        )
        assert len(frequency_rows[0]) == 2 * (n - 3)
        np.testing.assert_array_equal(frequency_rows[3], np.ones(2 * (n - 3)))

    def test_capped_frequency_rows_read_each_occurrence_at_most_once(self):
        """Above the expansion cap the panels read a subset of the occurrences, none twice."""
        rng = np.random.default_rng(3)
        n = 1_001
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = 1.0 + 2.0 * frame["x"].to_numpy() + rng.normal(0.0, 0.2, n)
        model = SuperGLM(family=Gaussian(), features={"x": Numeric()})
        model.fit(frame, y)
        metrics = model.metrics(frame, y)
        # 100 occurrences per row is 100,100 in all, above the 100,000 cap; eta
        # carries each row's index so the multiplicities can be read back.
        weights = np.full(n, 100.0)
        index = np.arange(n, dtype=np.float64)
        rows = diagnostics_module._diagnostic_rows(
            model,
            y,
            metrics._mu,
            index,
            weights,
            metrics.residuals("quantile", seed=5),
            5,
            rng,
            weight_semantics="frequency",
        )
        multiplicity = np.bincount(rows[2].astype(np.intp), minlength=n)
        assert multiplicity.sum() == 100_000
        assert np.all(multiplicity <= weights)

    class _OmitFirstOccurrence:
        """A legal draw of ``size`` of ``a`` that omits the first value."""

        def choice(self, a, size, **kwargs):
            return np.arange(a - size, a)

    def test_capped_rows_that_hold_each_row_once_still_read_unit_weights(self, monkeypatch):
        """A capped draw that reads every row once still reads unit weights."""
        # Row 0 holds two occurrences, so 11 exceed a cap of 10 and the forced
        # draw omits one copy of row 0: every row is read exactly once.
        n = 10
        monkeypatch.setattr(diagnostics_module, "_FREQUENCY_EXPANSION_CAP", n)
        weights = np.ones(n)
        weights[0] = 2.0
        x = np.linspace(0.0, 1.0, 20)
        model = SuperGLM(family=Gaussian(), features={"x": Numeric()})
        model.fit(pd.DataFrame({"x": x}), 1.0 + x + 0.1 * np.sin(10.0 * x))
        rows = diagnostics_module._diagnostic_rows(
            model,
            np.zeros(n),
            np.zeros(n),
            np.arange(n, dtype=np.float64),
            weights,
            np.zeros(n),
            0,
            self._OmitFirstOccurrence(),
            weight_semantics="frequency",
        )
        multiplicity = np.bincount(rows[2].astype(np.intp), minlength=n)
        assert multiplicity.sum() == n
        assert np.all(multiplicity <= weights)
        np.testing.assert_array_equal(rows[3], np.ones(n))

    @staticmethod
    def _balanced_binary_model():
        """A Binomial fit whose prediction at ``x = 0`` is exactly 0.5."""
        model = SuperGLM(family=Binomial(), features={"x": Numeric()}, weight_semantics="frequency")
        model.fit(pd.DataFrame({"x": np.tile([-1.0, 1.0], 10)}), np.tile([0.0, 0.0, 1.0, 1.0], 5))
        return model

    def test_binary_qq_envelope_holds_a_calibrated_fit(self):
        """A simulated Bernoulli response and its jitter use independent uniforms."""
        # Calibrated outcomes: each row predicts 0.5, and half the outcomes are 1.
        # A jitter that reuses the response's uniforms puts no simulated PIT in
        # (0.25, 0.75), pushing the envelope to +-0.94 at theoretical +-0.4.
        model = self._balanced_binary_model()
        n = 2_000
        frame = pd.DataFrame({"x": np.zeros(n)})
        fig = model.plot_diagnostics(frame, np.tile([0.0, 1.0], n // 2), n_sim=20, seed=42)
        ax = fig.axes[0]
        theoretical, observed = ax.lines[0].get_data()
        envelope = ax.collections[0].get_paths()[0].vertices
        for target in (-0.4, 0.4):
            k = int(np.argmin(np.abs(theoretical - target)))
            band = envelope[envelope[:, 0] == theoretical[k], 1]
            assert band.min() <= observed[k] <= band.max(), (target, observed[k], band)
        plt.close(fig)

    def test_capped_selection_is_independent_of_the_observed_jitter(self, monkeypatch):
        """Which occurrences the cap selects does not share uniforms with their jitter."""
        # Two rows predicting 0.5 with outcomes 0 and 1, 2**34 occurrences each.
        # A zero's PIT is u/2 and a one's (1 + u)/2, so the middle quartile counts
        # differ by (zeros with u > 1/2) - (ones with u < 1/2), whose SD is
        # sqrt(m/2), about 224, for independent streams. Selection and jitter on
        # one stream leave it within 1 of 0.
        captured = []
        rows = diagnostics_module._diagnostic_rows

        def recording_rows(*args, **kwargs):
            out = rows(*args, **kwargs)
            captured.append(out[4])
            return out

        monkeypatch.setattr(diagnostics_module, "_diagnostic_rows", recording_rows)
        model = self._balanced_binary_model()
        frame = pd.DataFrame({"x": [0.0, 0.0]})
        weights = np.full(2, 2.0**34)
        for seed in (0, 1, 2):
            fig = model.plot_diagnostics(
                frame, np.array([0.0, 1.0]), sample_weight=weights, n_sim=1, seed=seed
            )
            plt.close(fig)
        quartile = 0.6744897501960817  # the standard normal's upper quartile
        differences = [
            int(np.sum((q > -quartile) & (q < 0.0)) - np.sum((q > 0.0) & (q < quartile)))
            for q in captured
        ]
        assert max(abs(d) for d in differences) > 3, differences

    @pytest.mark.parametrize(
        "family",
        [Gaussian(), Gamma(), Poisson(), NegativeBinomial(theta=2.0), Binomial()],
        ids=["gaussian", "gamma", "poisson", "negative_binomial", "binomial"],
    )
    def test_simulated_rows_are_standard_normal_under_their_own_contract(self, family):
        """The envelope's defining property, stated without reference to the code.

        A Q-Q envelope is only a reference band if data simulated from the
        fitted model produces quantile residuals that are standard normal.
        That holds exactly when ``_simulate_response`` draws from the same
        marginal ``_quantile_residuals`` inverts -- so this fails for any
        mismatch between the two, whichever side moved.
        """
        from scipy import stats as scipy_stats

        from superglm.inference.metrics import ModelMetrics

        rng = np.random.default_rng(404)
        n = 4000
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        linear = 0.5 + 0.8 * frame["x"].to_numpy()
        mu_true = np.exp(linear)
        weights = rng.uniform(0.4, 6.0, n)
        if isinstance(family, Binomial):
            y = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear))).astype(float)
        elif isinstance(family, Poisson | NegativeBinomial):
            y = rng.poisson(mu_true).astype(float)
        elif isinstance(family, Gaussian):
            y = linear + rng.normal(0.0, 0.4, n)
        else:
            y = rng.gamma(6.0, mu_true / 6.0)

        model = SuperGLM(
            family=family,
            features={"x": Numeric()},
            weight_semantics="prior",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame, y, sample_weight=weights)
        metrics = model.metrics(frame, y, weights)
        mu, phi = metrics._mu, metrics.phi

        matched = diagnostics_module._simulate_response(
            family, mu, phi, weights, np.random.default_rng(7), weight_semantics="prior"
        )
        residuals = ModelMetrics(model, y=matched, sample_weight=weights, _mu=mu).residuals(
            "quantile", seed=11
        )
        assert scipy_stats.kstest(residuals, "norm").pvalue > 0.01

        # Mutation check: the pre-fix behaviour simulated every non-Tweedie
        # family at unit weight, which is the frequency marginal.
        mismatched = diagnostics_module._simulate_response(
            family, mu, phi, weights, np.random.default_rng(7), weight_semantics="frequency"
        )
        wrong = ModelMetrics(model, y=mismatched, sample_weight=weights, _mu=mu).residuals(
            "quantile", seed=11
        )
        if isinstance(family, Binomial):
            # Binomial has no mismatch to detect, and that is the point: its
            # prior likelihood's two reachable masses at y in {0, 1} do not sum
            # to one for w != 1, so no {0,1}-supported marginal reproduces it.
            # Both the simulator and the residual therefore stay
            # contract-invariant, and the two arms are the same draw.
            np.testing.assert_array_equal(mismatched, matched)
        else:
            assert scipy_stats.kstest(wrong, "norm").pvalue < 1e-6

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


class TestTiedPredictions:
    """Rows sharing a prediction share every bin and neighbourhood, whatever their order."""

    def test_calibration_puts_one_prediction_in_one_bin(self):
        """Two rows predicting one rate, 2,000 copies each, one without an event and one with."""
        rng = np.random.default_rng(5)
        X, y = _make_binomial_data(rng, n=200)
        model = SuperGLM(
            family="binomial",
            features={"x": Numeric()},
            selection_penalty=0.0,
            weight_semantics="frequency",
        ).fit(X, y)
        frame = pd.DataFrame({"x": [0.3, 0.3]})
        fig = model.plot_diagnostics(
            frame, np.array([0.0, 1.0]), sample_weight=np.full(2, 2000.0), n_sim=1
        )
        try:
            points = fig.get_axes()[1].collections[0].get_offsets()
            np.testing.assert_allclose(points, [[model.predict(frame)[0], 0.5]], rtol=1e-12)
        finally:
            plt.close(fig)

    def test_median_trend_reads_whole_tiers(self):
        x = np.repeat([0.0, 1.0, 2.0], [30, 50, 20])
        y = np.random.default_rng(6).normal(size=100)
        tiers = [y[:30], y[30:80], y[80:]]
        for rows in (np.arange(100), np.random.default_rng(7).permutation(100)):
            fig, ax = plt.subplots()
            diagnostics_module._binned_median_line(ax, x[rows], y[rows], n_bins=20)
            line = ax.get_lines()[0]
            np.testing.assert_array_equal(line.get_xdata(), [0.0, 1.0, 2.0])
            np.testing.assert_allclose(line.get_ydata(), [np.median(t) for t in tiers])
            plt.close(fig)

    def test_lowess_neighbourhood_holds_every_tied_row(self):
        """At one shared x every row is at distance zero, so all of them enter the local fit."""
        y = np.random.default_rng(8).standard_t(3, size=60)
        weights = np.ones(60)
        for _ in range(3):  # Cleveland (1979): a fit, then two bisquare reweightings
            fit = np.sum(weights * y) / np.sum(weights)
            u = (y - fit) / (6.0 * np.median(np.abs(y - fit)))
            weights = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        rows = np.random.default_rng(9).permutation(60)
        for values in (y, y[rows]):
            smooth = diagnostics_module._lowess(np.zeros(60), values)
            np.testing.assert_allclose(smooth, fit, rtol=1e-13)
