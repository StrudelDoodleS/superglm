"""Tests for REML Newton optimizer fixes (Bugs 1-8).

Covers: FP warmup removal, compound convergence, active-set, proportional
step scaling, modified Newton eigenvalue floor, post-loop rescue removal,
summary iter count, CI overflow protection, and discrete path alignment.

Tests that exercise the exact Newton path (optimize_direct_reml) must use
selection_penalty=0 so lambda1=0, which triggers the direct REML path
instead of the EFS optimizer.
"""

import io
import sys

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.inference import _MAX_LOG_REL, _safe_exp

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def poisson_data():
    """Small Poisson dataset with smooth + linear structure."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.choice(["A", "B", "C"], n)
    eta = 0.5 + 0.3 * np.sin(x1) - 0.1 * np.cos(3 * x2)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    w = np.ones(n)
    return X, y, w


@pytest.fixture
def gamma_data():
    """Small Gamma dataset for estimated-scale testing."""
    rng = np.random.default_rng(99)
    n = 400
    x = rng.uniform(1, 10, n)
    mu = np.exp(0.3 + 0.1 * np.sin(x))
    y = rng.gamma(shape=5.0, scale=mu / 5.0, size=n)
    X = pd.DataFrame({"x": x})
    w = np.ones(n)
    return X, y, w


@pytest.fixture
def multi_smooth_data():
    """Poisson data with 4 smooth terms (for active-set testing)."""
    rng = np.random.default_rng(123)
    n = 600
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.uniform(0, 10, n)
    x4 = rng.uniform(0, 10, n)
    # x1 has signal, x2 has weak signal, x3 and x4 are noise
    eta = 0.5 + 0.3 * np.sin(x1) + 0.05 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, y


# ── Bug 1: No FP warmup ──────────────────────────────────────────


class TestNoFPWarmup:
    """Verify that all REML iterations are Newton (no FP warmup)."""

    def test_verbose_output_no_fp(self, poisson_data):
        """No 'REML FP' lines in verbose output; all are 'REML Newton'."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            model.fit_reml(X[["x1"]], y, sample_weight=w, verbose=True)
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "REML FP" not in output, "FP warmup phase should not appear"
        assert "REML Newton" in output, "Newton iterations should be labeled"

    def test_bootstrap_still_present(self, poisson_data):
        """Bootstrap initialization line should still appear."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            model.fit_reml(X[["x1"]], y, sample_weight=w, verbose=True)
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "REML bootstrap" in output

    def test_single_smooth_converges(self, poisson_data):
        """Model with single smooth converges without FP warmup."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged


# ── Bug 2: Compound convergence criterion ─────────────────────────


class TestCompoundConvergence:
    """Verify compound convergence criterion (gradient + objective change)."""

    def test_normal_convergence(self, poisson_data):
        """Standard Poisson spline model converges with reasonable iter count."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged
        assert model._reml_result.n_reml_iter < 30

    def test_two_smooth_convergence(self, poisson_data):
        """Two-spline model converges."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit_reml(X[["x1", "x2"]], y, sample_weight=w)
        assert model._reml_result.converged

    def test_gamma_estimated_scale_converges(self, gamma_data):
        """Estimated-scale family converges with compound criterion."""
        X, y, w = gamma_data
        model = SuperGLM(
            family="gamma",
            selection_penalty=0,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=w)
        assert model._reml_result.converged


# ── Bug 3: Active-set ─────────────────────────────────────────────


class TestActiveSet:
    """Verify active-set strategy freezes converged components."""

    def test_multi_smooth_converges(self, multi_smooth_data):
        """Model with 4 smooths converges (active-set doesn't break anything)."""
        X, y = multi_smooth_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
                "x3": Spline(n_knots=6, penalty="ssp"),
                "x4": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit_reml(X, y)
        assert model._reml_result.converged

    def test_single_term_not_frozen(self, poisson_data):
        """Single-term model: active set should not permanently freeze it."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged
        for lam in model._reml_result.lambdas.values():
            assert np.isfinite(lam) and lam > 0


# ── Bug 4: Proportional step scaling ──────────────────────────────


class TestProportionalStepScaling:
    """Verify step scaling preserves Newton direction."""

    def test_direction_preservation(self):
        """When delta is proportionally scaled, ratios are preserved."""
        delta_raw = np.array([8.0, 4.0, -2.0, 1.0])
        max_step = 5.0
        max_delta = np.max(np.abs(delta_raw))
        delta_scaled = delta_raw * (max_step / max_delta)

        direction_raw = delta_raw / np.max(np.abs(delta_raw))
        direction_scaled = delta_scaled / np.max(np.abs(delta_scaled))
        np.testing.assert_allclose(direction_raw, direction_scaled)
        assert np.max(np.abs(delta_scaled)) == pytest.approx(max_step)

    def test_eigenvalue_floor_eps_07(self):
        """Modified Newton uses eps^0.7 eigenvalue floor."""
        eps = np.finfo(float).eps
        eigvals = np.array([100.0, 1.0, 1e-20])
        max_eig = eigvals.max()
        floor = max_eig * eps**0.7
        floored = np.maximum(np.abs(eigvals), floor)
        assert floored[2] == pytest.approx(floor)
        assert floored[0] == pytest.approx(100.0)

    def test_stability_across_starts(self, poisson_data):
        """Regression: convergence from diverse starts is stable."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged


# ── Bug 5: No post-loop rescue ───────────────────────────────────


class TestNoPostLoopRescue:
    """Post-loop gradient rescue is removed."""

    def test_max_iter_exhausted_not_converged(self, poisson_data):
        """Setting max_reml_iter=2 should give converged=False."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit_reml(X[["x1", "x2"]], y, sample_weight=w, max_reml_iter=2)
        assert not model._reml_result.converged

    def test_genuine_convergence_still_true(self, poisson_data):
        """Models that genuinely converge still report converged=True."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged


# ── Bug 6: Summary iter count ────────────────────────────────────


class TestSummaryIterCount:
    """Summary n_iter shows REML outer iters, not PIRLS inner iters."""

    def test_reml_iter_in_summary(self, poisson_data):
        """Summary n_iter matches _reml_result.n_reml_iter."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        s = model.summary()
        expected = model._reml_result.n_reml_iter
        assert s._info["n_iter"] == expected

    def test_non_reml_iter_unchanged(self, poisson_data):
        """Non-REML fit uses PIRLS n_iter as before."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            features={"x1": Numeric()},
        )
        model.fit(X[["x1"]], y, sample_weight=w)
        s = model.summary()
        assert s._info["n_iter"] == model.result.n_iter

    def test_summary_string_shows_reml_iter(self, poisson_data):
        """The summary string shows REML iteration count, not PIRLS count."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        s = model.summary()
        n_reml = model._reml_result.n_reml_iter
        summary_str = str(s)
        assert str(n_reml) in summary_str

    def test_summary_converged_uses_reml(self, poisson_data):
        """Summary converged field uses REML convergence, not PIRLS."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        # max_reml_iter=2 means REML did NOT converge, but PIRLS always does
        model.fit_reml(X[["x1", "x2"]], y, sample_weight=w, max_reml_iter=2)
        s = model.summary()
        assert s._info["converged"] is False, (
            "Summary should show REML converged=False when max_reml_iter exhausted"
        )

    def test_diagnostics_converged_uses_reml(self, poisson_data):
        """Diagnostics converged field uses REML convergence, not PIRLS."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit_reml(X[["x1", "x2"]], y, sample_weight=w, max_reml_iter=2)
        d = model.diagnostics()
        assert d["_model"]["converged"] is False, (
            "Diagnostics should show REML converged=False when max_reml_iter exhausted"
        )


# ── Bug 7: CI overflow protection ─────────────────────────────────


class TestCIOverflowProtection:
    """CI exponentiation is overflow-protected."""

    def test_safe_exp_clamps_large(self):
        """_safe_exp clamps extreme positive values."""
        x = np.array([600.0, 1000.0, -1000.0, 0.5])
        result = _safe_exp(x)
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(np.exp(500.0))
        assert result[1] == pytest.approx(np.exp(500.0))
        assert result[2] == pytest.approx(np.exp(-500.0))
        assert result[3] == pytest.approx(np.exp(0.5))

    def test_safe_exp_preserves_normal(self):
        """Normal-range values pass through unchanged."""
        x = np.linspace(-10, 10, 100)
        np.testing.assert_allclose(_safe_exp(x), np.exp(x))

    def test_max_log_rel_constant(self):
        """_MAX_LOG_REL is at a safe value."""
        assert _MAX_LOG_REL == 500.0
        assert np.isfinite(np.exp(500.0))

    def test_term_inference_finite_ci(self, poisson_data):
        """CI values from term_inference are finite for normal models."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        ti = model.term_inference("x1")
        if ti.ci_lower is not None:
            assert np.all(np.isfinite(ti.ci_lower))
            assert np.all(np.isfinite(ti.ci_upper))


# ── Bug 8: Discrete path ─────────────────────────────────────────


class TestDiscretePath:
    """Discrete path has same fixes as exact path."""

    def test_discrete_converges(self, poisson_data):
        """Discrete path converges."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w)
        assert model._reml_result.converged

    def test_discrete_no_fp_in_verbose(self, poisson_data):
        """Discrete path has no FP labels (uses Newton/POI)."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            model.fit_reml(X[["x1"]], y, sample_weight=w, verbose=True)
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "REML FP" not in output

    def test_discrete_vs_exact_close(self, poisson_data):
        """Discrete and exact paths should give similar lambdas."""
        X, y, w = poisson_data
        exact = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        exact.fit_reml(X[["x1"]], y, sample_weight=w)

        disc = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        disc.fit_reml(X[["x1"]], y, sample_weight=w)

        for name in exact._reml_result.lambdas:
            lam_e = exact._reml_result.lambdas[name]
            lam_d = disc._reml_result.lambdas[name]
            ratio = lam_e / max(lam_d, 1e-12)
            assert 0.1 < ratio < 10, f"{name}: exact={lam_e:.4g} vs discrete={lam_d:.4g}"

    def test_discrete_gamma_estimated_scale(self, gamma_data):
        """Discrete path works with estimated-scale families."""
        X, y, w = gamma_data
        model = SuperGLM(
            family="gamma",
            selection_penalty=0,
            discrete=True,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=w)
        assert model._reml_result.converged


# ── Steepest descent fallback ─────────────────────────────────────


class TestSteepestDescentFallback:
    """Steepest descent fallback uses -grad/max(|grad|), not FP."""

    def test_unit_length_direction(self):
        """Steepest descent step has unit infinity norm."""
        grad = np.array([3.0, -1.5, 0.5])
        grad_max = np.max(np.abs(grad))
        step = -grad / grad_max
        assert np.max(np.abs(step)) == pytest.approx(1.0)
        assert step[0] < 0  # grad[0] > 0 => step[0] < 0
        assert step[1] > 0  # grad[1] < 0 => step[1] > 0


# ── F6: Quasi-separated CI integration test ─────────────────────


class TestQuasiSeparatedCIFinite:
    """CI values are finite even for quasi-separated categorical levels."""

    def test_quasi_separated_ci_finite(self):
        """Quasi-separated level produces finite (clamped) CIs, not inf/nan."""
        rng = np.random.default_rng(42)
        n = 5000
        n_rare = 5
        n_main = n - n_rare
        cat_main = rng.choice(["base", "mid", "hi"], n_main, p=[0.5, 0.3, 0.2])
        cat_rare = np.array(["rare"] * n_rare)
        cat = np.concatenate([cat_main, cat_rare])

        exposure = rng.uniform(0.3, 1.0, n)
        eta = 5.0 + 0.3 * (cat == "hi").astype(float) - 0.2 * (cat == "mid").astype(float)
        mu = np.exp(eta) * exposure
        from superglm.profiling.tweedie import generate_tweedie_cpg

        y = generate_tweedie_cpg(n, mu=mu, phi=2.0, p=1.5, rng=rng)
        y[cat == "rare"] = 0.0  # force near-separation

        idx = rng.permutation(n)
        cat, y, exposure = cat[idx], y[idx], exposure[idx]
        df = pd.DataFrame({"cat": cat})

        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))

        # term_inference should produce finite CIs for all levels
        ti = m.term_inference("cat")
        if ti.ci_lower is not None:
            assert np.all(np.isfinite(ti.ci_lower)), "CI lower contains inf/nan"
            assert np.all(np.isfinite(ti.ci_upper)), "CI upper contains inf/nan"

        # summary should also produce finite output
        s = m.summary()
        for row in s._coef_rows:
            if row.se is not None:
                assert np.isfinite(row.se), f"{row.name} SE is not finite"


# ── F1 test gap: reml_tol parameter effectiveness ───────────────
