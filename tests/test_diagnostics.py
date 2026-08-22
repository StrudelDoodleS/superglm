"""Tests for diagnostics: term importance, drop-term, spline redundancy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm.distributions import Tweedie, clip_mu
from superglm.links import stabilize_eta

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


def _fit_offset_diagnostic_model():
    rng = np.random.default_rng(20260729)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    exposure = rng.uniform(0.2, 2.5, size=n)
    offset = np.log(exposure)
    weights = rng.uniform(0.5, 2.0, size=n)
    y = rng.poisson(np.exp(0.15 + 0.45 * x - 0.2 * z + offset))
    X = pd.DataFrame({"x": x, "z": z})
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "z": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=weights, offset=offset)
    return model, X, y, weights, offset


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

    def test_refit_mode_forwards_nonuniform_sample_weight(self):
        rng = np.random.default_rng(20260727)
        n = 220
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = 0.4 + 0.8 * x - 0.35 * z + rng.normal(scale=0.25, size=n)
        y[x > 1.0] += 1.5
        sample_weight = np.where(x > 1.0, 0.08, 2.5)
        X = pd.DataFrame({"x": x, "z": z})
        model = SuperGLM(
            family="gaussian",
            features={"x": Numeric(), "z": Numeric()},
            selection_penalty=0.0,
        ).fit(X, y, sample_weight=sample_weight)

        expected = model.drop1(X, y, sample_weight=sample_weight)
        unweighted = model.drop1(X, y)
        actual = model.term_drop_diagnostics(
            X,
            y,
            sample_weight=sample_weight,
            mode="refit",
        )

        columns = [
            "feature",
            "deviance_reduced",
            "delta_deviance",
            "statistic",
            "p_value",
        ]
        pd.testing.assert_frame_equal(
            actual[columns].reset_index(drop=True),
            expected[columns].reset_index(drop=True),
            rtol=1e-10,
            atol=1e-10,
        )
        assert not np.allclose(
            actual["deviance_reduced"],
            unweighted["deviance_reduced"],
            rtol=1e-6,
            atol=1e-6,
        )

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

    def test_holdout_uses_validation_weights_and_offset(self):
        rng = np.random.default_rng(20260728)
        n_train, n_val = 240, 80
        x_train = rng.normal(size=n_train)
        z_train = rng.normal(size=n_train)
        x_val = rng.normal(size=n_val)
        z_val = rng.normal(size=n_val)
        exposure_train = rng.uniform(0.2, 2.5, size=n_train)
        exposure_val = rng.uniform(0.1, 3.0, size=n_val)
        offset_train = np.log(exposure_train)
        offset_val = np.log(exposure_val)
        weights_train = rng.uniform(0.5, 2.0, size=n_train)
        weights_val = rng.uniform(0.4, 2.3, size=n_val)
        y_train = rng.poisson(np.exp(0.2 + 0.55 * x_train - 0.3 * z_train + offset_train))
        y_val = rng.poisson(np.exp(0.2 + 0.55 * x_val - 0.3 * z_val + offset_val))
        X_train = pd.DataFrame({"x": x_train, "z": z_train})
        X_val = pd.DataFrame({"x": x_val, "z": z_val})
        model = SuperGLM(
            family="poisson",
            features={"x": Numeric(), "z": Numeric()},
            selection_penalty=0.0,
        ).fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            offset=offset_train,
        )

        actual = model.term_drop_diagnostics(
            X_train,
            y_train,
            sample_weight=weights_train,
            offset=offset_train,
            mode="holdout",
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=weights_val,
            offset_val=offset_val,
        )

        eta = np.full(n_val, model.result.intercept, dtype=np.float64) + offset_val
        contributions = {}
        for name in model._feature_order:
            group = model._feature_groups(name)[0]
            contribution = X_val[name].to_numpy() * model.result.beta[group.sl][0]
            contributions[name] = contribution
            eta += contribution
        mu_full = clip_mu(
            model._link.inverse(stabilize_eta(eta, model._link)),
            model._distribution,
        )
        dev_full = np.sum(weights_val * model._distribution.deviance_unit(y_val, mu_full))
        expected = []
        for name, contribution in contributions.items():
            mu_drop = clip_mu(
                model._link.inverse(stabilize_eta(eta - contribution, model._link)),
                model._distribution,
            )
            dev_drop = np.sum(weights_val * model._distribution.deviance_unit(y_val, mu_drop))
            expected.append(
                {
                    "feature": name,
                    "delta_deviance": dev_drop - dev_full,
                }
            )

        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            pd.DataFrame(expected).reset_index(drop=True),
            rtol=2e-10,
            atol=2e-10,
        )

    def test_holdout_same_objects_reuse_training_weight_and_offset(self):
        model, X, y, weights, offset = _fit_offset_diagnostic_model()
        actual = model.term_drop_diagnostics(
            X,
            y,
            sample_weight=weights,
            offset=offset,
            mode="holdout",
            X_val=X,
            y_val=y,
        )
        explicit = model.term_drop_diagnostics(
            X,
            y,
            sample_weight=weights,
            offset=offset,
            mode="holdout",
            X_val=X,
            y_val=y,
            sample_weight_val=weights,
            offset_val=offset,
        )
        pd.testing.assert_frame_equal(actual, explicit)

    def test_holdout_separate_rows_reject_training_weight_fallback(self, fitted_model):
        model, X, y, weights = fitted_model
        with pytest.raises(ValueError, match="sample_weight_val"):
            model.term_drop_diagnostics(
                X,
                y,
                sample_weight=weights,
                mode="holdout",
                X_val=X.copy(),
                y_val=y.copy(),
            )

    def test_holdout_offset_fit_requires_validation_offset(self):
        model, X, y, weights, offset = _fit_offset_diagnostic_model()
        with pytest.raises(ValueError, match="offset_val"):
            model.term_drop_diagnostics(
                X,
                y,
                sample_weight=weights,
                offset=offset,
                mode="holdout",
                X_val=X.copy(),
                y_val=y.copy(),
                sample_weight_val=weights.copy(),
            )

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.ones((4, 1)), "one-dimensional"),
            (np.ones(3), "length 4"),
            (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
            (np.array([1.0, -1.0, 1.0, 1.0]), "nonnegative"),
            (np.zeros(4), "all zero"),
        ],
    )
    def test_holdout_validates_sample_weight_val(self, fitted_model, weights, message):
        model, X, y, _ = fitted_model
        with pytest.raises(ValueError, match=message):
            model.term_drop_diagnostics(
                X,
                y,
                mode="holdout",
                X_val=X.iloc[:4].copy(),
                y_val=np.asarray(y[:4]),
                sample_weight_val=weights,
            )

    @pytest.mark.parametrize(
        ("offset_val", "message"),
        [
            (np.ones((4, 1)), "one-dimensional"),
            (np.ones(3), "length 4"),
            (np.array([0.0, np.inf, 0.0, 0.0]), "finite"),
        ],
    )
    def test_holdout_validates_offset_val(self, fitted_model, offset_val, message):
        model, X, y, _ = fitted_model
        with pytest.raises(ValueError, match=message):
            model.term_drop_diagnostics(
                X,
                y,
                mode="holdout",
                X_val=X.iloc[:4].copy(),
                y_val=np.asarray(y[:4]),
                offset_val=offset_val,
            )

    def test_holdout_validates_y_val_length(self, fitted_model):
        model, X, y, _ = fitted_model
        with pytest.raises(ValueError, match="y_val must have length 4"):
            model.term_drop_diagnostics(
                X,
                y,
                mode="holdout",
                X_val=X.iloc[:4].copy(),
                y_val=np.asarray(y[:3]),
            )

    @pytest.mark.parametrize(
        ("y_val", "message"),
        [
            (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
            (np.array([1.0, -1.0, 1.0, 1.0]), "nonnegative"),
        ],
    )
    def test_holdout_validates_y_val_domain(self, fitted_model, y_val, message):
        model, X, y, _ = fitted_model
        with pytest.raises(ValueError, match=message):
            model.term_drop_diagnostics(
                X,
                y,
                mode="holdout",
                X_val=X.iloc[:4].copy(),
                y_val=y_val,
            )

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

    def test_frequency_weights_match_literal_row_replication(self, fitted_model):
        model, X, _, _ = fitted_model
        model._weight_semantics = "frequency"
        weights = (1 + np.arange(len(X)) % 4).astype(np.float64)
        rows = np.repeat(np.arange(len(X)), weights.astype(np.int64))

        weighted = model.spline_redundancy(X, sample_weight=weights)["strong"]
        repeated = model.spline_redundancy(X.iloc[rows].reset_index(drop=True))["strong"]

        np.testing.assert_allclose(weighted.support_mass, repeated.support_mass, atol=2e-15)
        np.testing.assert_allclose(
            weighted.adjacent_basis_corr,
            repeated.adjacent_basis_corr,
            atol=2e-14,
        )
        assert weighted.effective_rank == repeated.effective_rank

    def test_prior_weights_do_not_change_physical_geometry(self, fitted_model):
        """The rule is now the declared contract's, not the family's: a prior
        weight leaves learned geometry a function of physical rows whatever the
        family, and Tweedie is simply the family that used to imply it."""
        model, X, _, _ = fitted_model
        model._distribution = Tweedie(p=1.5)
        model._weight_semantics = "prior"
        weights = np.linspace(0.2, 3.0, len(X))

        unit = model.spline_redundancy(X)["strong"]
        weighted = model.spline_redundancy(X, sample_weight=weights)["strong"]

        np.testing.assert_array_equal(weighted.support_mass, unit.support_mass)
        np.testing.assert_array_equal(weighted.adjacent_basis_corr, unit.adjacent_basis_corr)
        assert weighted.effective_rank == unit.effective_rank

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.ones(3), "length"),
            (np.array([1.0, np.nan]), "finite"),
            (np.array([1.0, -1.0]), "nonnegative"),
            (np.zeros(2), "all zero"),
        ],
    )
    def test_validates_frequency_weights(self, fitted_model, weights, message):
        model, X, _, _ = fitted_model
        model._weight_semantics = "frequency"
        with pytest.raises(ValueError, match=message):
            model.spline_redundancy(X.iloc[:2], sample_weight=weights)

    def test_tweedie_requires_strictly_positive_prior_weights(self, fitted_model):
        model, X, _, _ = fitted_model
        model._distribution = Tweedie(p=1.5)
        weights = np.ones(len(X))
        weights[0] = 0.0

        with pytest.raises(ValueError, match="strictly positive"):
            model.spline_redundancy(X, sample_weight=weights)

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
