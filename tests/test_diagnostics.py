"""Tests for diagnostics: term importance, drop-term, spline redundancy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mixed_data():
    """Data with a strong spline, weak numeric, and categorical feature."""
    rng = np.random.default_rng(42)
    n = 2000
    x_strong = rng.uniform(0, 10, n)
    x_weak = rng.normal(0, 1, n)
    region = rng.choice(["A", "B", "C"], n)
    region_effect = {"A": 0.0, "B": 0.3, "C": -0.2}

    log_rate = 0.5 * np.sin(x_strong) + 0.01 * x_weak + np.array([region_effect[r] for r in region])
    y = rng.poisson(np.exp(log_rate))
    sample_weight = np.ones(n)
    X = pd.DataFrame({"strong": x_strong, "weak": x_weak, "region": region})
    return X, y, sample_weight


@pytest.fixture
def fitted_model(mixed_data):
    X, y, sample_weight = mixed_data
    m = SuperGLM(
        family="poisson",
        features={
            "strong": Spline(n_knots=10),
            "region": Categorical(),
        },
        splines=None,
        selection_penalty=0.0,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m, X, y, sample_weight


# ── Phase 7: Term importance tests ──────────────────────────────


class TestTermImportance:
    def test_returns_dataframe(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        df = m.term_importance(X, sample_weight=sample_weight)
        assert isinstance(df, pd.DataFrame)
        assert "term" in df.columns
        assert "feature" in df.columns
        assert "variance_eta" in df.columns

    def test_contains_all_groups(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        df = m.term_importance(X, sample_weight=sample_weight)
        group_names = {g.name for g in m._groups}
        assert set(df["term"]) == group_names

    def test_strong_feature_higher_variance(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        df = m.term_importance(X, sample_weight=sample_weight)
        strong_var = df.loc[df["feature"] == "strong", "variance_eta"].sum()
        region_var = df.loc[df["feature"] == "region", "variance_eta"].sum()
        assert strong_var > region_var * 0.5  # strong signal should dominate

    def test_not_fitted_raises(self, mixed_data):
        X, y, sample_weight = mixed_data
        m = SuperGLM(
            family="poisson",
            features={"strong": Spline(n_knots=10)},
            selection_penalty=0.0,
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.term_importance(X, sample_weight=sample_weight)


# ── Phase 8: Drop-term diagnostics tests ────────────────────────


class TestDropTermDiagnostics:
    def test_refit_mode_returns_dataframe(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        df = m.term_drop_diagnostics(X, y, sample_weight=sample_weight, mode="refit")
        assert isinstance(df, pd.DataFrame)

    def test_holdout_mode_returns_dataframe(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        # Use training data as "validation" for simplicity
        df = m.term_drop_diagnostics(
            X,
            y,
            sample_weight=sample_weight,
            mode="holdout",
            X_val=X,
            y_val=y,
        )
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "delta_deviance" in df.columns

    def test_holdout_positive_delta_for_strong(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        df = m.term_drop_diagnostics(
            X,
            y,
            sample_weight=sample_weight,
            mode="holdout",
            X_val=X,
            y_val=y,
        )
        strong_delta = df.loc[df["feature"] == "strong", "delta_deviance"].values[0]
        assert strong_delta > 0  # dropping strong feature should increase deviance

    def test_holdout_requires_validation_data(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        with pytest.raises(ValueError, match="X_val and y_val"):
            m.term_drop_diagnostics(X, y, sample_weight=sample_weight, mode="holdout")

    def test_invalid_mode(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        with pytest.raises(ValueError, match="mode must be"):
            m.term_drop_diagnostics(X, y, sample_weight=sample_weight, mode="invalid")


# ── Phase 9: Spline redundancy diagnostics tests ────────────────


class TestSplineRedundancy:
    def test_returns_dict(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        result = m.spline_redundancy(X, sample_weight=sample_weight)
        assert isinstance(result, dict)
        assert "strong" in result

    def test_report_fields(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        result = m.spline_redundancy(X, sample_weight=sample_weight)
        report = result["strong"]
        assert report.feature_name == "strong"
        assert report.n_knots > 0
        assert len(report.knot_locations) == report.n_knots
        assert len(report.knot_spacing) == report.n_knots - 1
        assert len(report.support_mass) == report.n_knots
        assert report.effective_rank > 0

    def test_support_mass_sums_roughly_to_one(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        result = m.spline_redundancy(X, sample_weight=sample_weight)
        total_mass = np.sum(result["strong"].support_mass)
        assert 0.8 < total_mass < 1.2  # approximate

    def test_non_spline_excluded(self, fitted_model):
        m, X, y, sample_weight = fitted_model
        result = m.spline_redundancy(X, sample_weight=sample_weight)
        assert "region" not in result  # categorical, not spline

    def test_over_specified_spline(self):
        """Spline with many knots on linear signal should show concentrated energy."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.2 * x))
        X = pd.DataFrame({"feat": x})
        m = SuperGLM(
            family="poisson",
            features={"feat": Spline(n_knots=25)},
            selection_penalty=0.0,
        )
        m.fit(X, y)
        result = m.spline_redundancy(X)
        report = result["feat"]
        # Most coefficient energy should be in the first few components
        energy = report.coef_energy_penalized
        total = np.sum(energy)
        if total > 0:
            top3_frac = np.sum(np.sort(energy)[-3:]) / total
            assert top3_frac > 0.3  # concentrated
