"""Tests for monotone repair functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.constraints import MonotoneRepairer, MonotoneRepairResult, monotonicity_violation

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def increasing_poisson_data():
    """Synthetic increasing Poisson signal with noise."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(0, 10, n)
    # True signal: monotone increasing (log scale)
    log_rate = 0.3 * x - 1.0 + 0.3 * rng.normal(size=n)
    # Add a small dip to create violations without monotone constraint
    log_rate += 0.5 * np.sin(x)
    y = rng.poisson(np.exp(log_rate))
    y = np.maximum(y, 0)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"signal": x})
    return X, y, sample_weight


@pytest.fixture
def decreasing_poisson_data():
    """Synthetic decreasing Poisson signal."""
    rng = np.random.default_rng(99)
    n = 2000
    x = rng.uniform(0, 10, n)
    log_rate = -0.3 * x + 2.0 + 0.3 * rng.normal(size=n)
    log_rate += 0.5 * np.sin(x)
    y = rng.poisson(np.exp(log_rate))
    y = np.maximum(y, 0)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"signal": x})
    return X, y, sample_weight


# ── Phase 1: Spline API tests ──────────────────────────────────


class TestSplineAPIParams:
    def test_monotone_none_default(self):
        s = Spline()
        assert s.monotone is None
        assert s.monotone_mode == "postfit"

    def test_monotone_increasing(self):
        s = Spline(monotone="increasing")
        assert s.monotone == "increasing"

    def test_monotone_decreasing(self):
        s = Spline(monotone="decreasing")
        assert s.monotone == "decreasing"

    def test_monotone_invalid(self):
        with pytest.raises(ValueError, match="monotone must be"):
            Spline(monotone="flat")

    def test_monotone_mode_invalid(self):
        with pytest.raises(ValueError, match="monotone_mode must be"):
            Spline(monotone="increasing", monotone_mode="invalid")

    def test_monotone_fit_mode_builds_scop(self):
        """PSpline monotone_mode='fit' builds SCOP reparameterization."""
        from superglm.features.spline import PSpline

        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization is not None
        assert info.monotone_engine == "scop"

    def test_monotone_ns_rejected(self):
        with pytest.raises(NotImplementedError, match="monotone is not supported for kind='ns'"):
            Spline(kind="ns", monotone="increasing")

    def test_monotone_bs(self):
        s = Spline(kind="bs", monotone="increasing")
        assert s.monotone == "increasing"

    def test_monotone_cr(self):
        s = Spline(kind="cr", monotone="decreasing")
        assert s.monotone == "decreasing"


# ── Phase 2: MonotoneRepairer tests ────────────────────────────


class TestMonotoneRepairer:
    def test_monotonicity_violation_increasing(self):
        vals = np.array([1.0, 2.0, 1.5, 3.0, 2.8])
        viol = monotonicity_violation(vals, "increasing")
        assert viol == pytest.approx(0.5)  # 2.0 -> 1.5

    def test_monotonicity_violation_decreasing(self):
        vals = np.array([3.0, 2.0, 2.5, 1.0])
        viol = monotonicity_violation(vals, "decreasing")
        assert viol == pytest.approx(0.5)  # 2.0 -> 2.5

    def test_monotonicity_violation_zero_for_monotone(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        assert monotonicity_violation(vals, "increasing") == 0.0

    def test_repairer_direction_validation(self):
        with pytest.raises(ValueError, match="direction must be"):
            MonotoneRepairer(direction="flat")


# ── Phase 3-4: End-to-end monotone repair ──────────────────────


class TestApplyMonotonePostfit:
    def test_monotonize_alias(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)

        out = m.monotonize(X, sample_weight=sample_weight)

        assert out is m
        assert "signal" in m._monotone_repairs
        repair = m._monotone_repairs["signal"]
        assert isinstance(repair, MonotoneRepairResult)
        assert repair.max_violation_after < 0.01

    def test_increasing_repair(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)

        # Apply repair
        m.apply_monotone_postfit(X, sample_weight=sample_weight)

        assert "signal" in m._monotone_repairs
        repair = m._monotone_repairs["signal"]
        assert isinstance(repair, MonotoneRepairResult)
        assert repair.direction == "increasing"
        assert repair.max_violation_after < 0.01

    def test_decreasing_repair(self, decreasing_poisson_data):
        X, y, sample_weight = decreasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="decreasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)

        assert "signal" in m._monotone_repairs
        repair = m._monotone_repairs["signal"]
        assert repair.direction == "decreasing"
        assert repair.max_violation_after < 0.01

    def test_prediction_monotone_after_repair(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)

        # Predict on sorted grid
        x_sorted = np.linspace(0.5, 9.5, 100)
        X_pred = pd.DataFrame({"signal": x_sorted})
        mu = m.predict(X_pred)

        # Predictions should be largely monotone increasing
        # Note: prediction monotonicity is approximate since the isotonic
        # repair works on a grid and projection back to coefficients
        # introduces small violations
        diffs = np.diff(mu)
        violations = np.sum(diffs < -0.01 * np.mean(mu))
        assert violations < 15

    def test_idempotency(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)
        beta_after_first = m.result.beta.copy()

        # Second call should be no-op
        m.apply_monotone_postfit(X, sample_weight=sample_weight)
        np.testing.assert_array_equal(m.result.beta, beta_after_first)

    def test_no_monotone_specs_noop(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10)},  # no monotone
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)
        assert len(m._monotone_repairs) == 0

    def test_not_fitted_raises(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(monotone="increasing")},
            selection_penalty=0.0,
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.monotonize(X, sample_weight=sample_weight)

    def test_reconstruct_after_repair(self, increasing_poisson_data):
        """Reconstruct should use repaired beta."""
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=15, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)

        recon = m.reconstruct_feature("signal")
        log_rel = recon["log_relativity"]
        # Reconstructed curve should be monotone
        viol = monotonicity_violation(log_rel, "increasing")
        assert viol < 0.05  # small tolerance for projection error


# ── Phase 5: Summary integration ──────────────────────────────


class TestSummaryMonotoneIntegration:
    def test_summary_shows_monotone_annotation(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, monotone="increasing")},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        m.apply_monotone_postfit(X, sample_weight=sample_weight)

        summary = m.summary()
        text = str(summary)
        assert "mono=increasing" in text
        assert "repaired" in text

    def test_summary_no_monotone_no_annotation(self, increasing_poisson_data):
        X, y, sample_weight = increasing_poisson_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10)},
            selection_penalty=0.0,
        )
        m.fit(X, y, sample_weight=sample_weight)

        summary = m.summary()
        text = str(summary)
        assert "mono=" not in text


# ── Phase 6: Fit-time scaffold ──────────────────────────────────


class TestFitTimeScaffold:
    def test_derivative_grid_raises(self):
        from superglm.constraints import derivative_grid_matrix

        with pytest.raises(NotImplementedError, match="Fit-time monotone"):
            derivative_grid_matrix(None)
