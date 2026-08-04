"""Tests for drop1() likelihood ratio test."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from scipy.stats import f as f_dist

from superglm import LambdaPolicy, RandomEffect, SuperGLM, Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@pytest.fixture
def poisson_data():
    """Poisson data with one strong and one noise feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x_strong = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.5 * x_strong)
    sample_weight = np.ones(n)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"strong": x_strong, "noise": x_noise})
    return X, y, sample_weight


class TestDrop1Basic:
    def test_returns_dataframe(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "feature" in result.columns
        assert "p_value" in result.columns
        assert "delta_deviance" in result.columns

    def test_strong_feature_significant(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        strong_row = result[result["feature"] == "strong"].iloc[0]
        noise_row = result[result["feature"] == "noise"].iloc[0]

        # Strong feature should have large deviance change and small p-value
        assert strong_row["delta_deviance"] > noise_row["delta_deviance"]
        assert strong_row["p_value"] < 0.01
        # Noise feature should have small deviance change
        assert noise_row["p_value"] > 0.01

    def test_sorted_by_p_value(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        p_values = result["p_value"].values
        assert np.all(p_values[:-1] <= p_values[1:])

    def test_unfitted_raises(self):
        model = SuperGLM(
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.drop1(X, np.array([1, 2, 3]))

    def test_drop1_rejects_variance_component_models_before_refitting(self):
        X = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 20)})
        y = np.tile([0.2, 0.7, 1.1], 20)
        model = SuperGLM(
            family="gaussian",
            features={"group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
            selection_penalty=0.0,
        ).fit_reml(X, y, runtime_validation="skip")

        with pytest.raises(NotImplementedError, match="drop1.*variance-component.*REML"):
            model.drop1(X, y)

    @pytest.mark.parametrize("test", ["f", "ChiSq", "bogus"])
    def test_rejects_unknown_test_dispatch(self, poisson_data, test):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"strong": Numeric(), "noise": Numeric()},
        ).fit(X, y, sample_weight=sample_weight)

        with pytest.raises(ValueError, match="test must be 'Chisq' or 'F'"):
            model.drop1(X, y, sample_weight=sample_weight, test=test)

    def test_preserves_arbitrary_hashable_pandas_column_labels(self):
        rng = np.random.default_rng(223)
        n = 140
        x0 = rng.normal(size=n)
        x_none = rng.normal(size=n)
        x_empty = rng.normal(size=n)
        X = pd.DataFrame(
            {
                0: x0,
                None: x_none,
                "": x_empty,
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.2 * x0 - 0.15 * x_none)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                0: Numeric(),
                None: Numeric(),
                "": Numeric(),
            },
        ).fit(X, y)

        result = model.drop1(X, y)

        assert set(result["feature"].tolist()) == {0, None, ""}
        assert np.all(np.isfinite(result["p_value"]))


class TestDrop1Spline:
    def test_spline_delta_df_gt_1(self):
        """Spline features should have delta_df > 1."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        result = model.drop1(X, y)

        assert result.iloc[0]["delta_df"] > 1.5


class TestDrop1FTest:
    def test_f_test_gamma(self):
        """F-test should work for Gamma (estimated scale)."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        shape = 5.0
        y = rng.gamma(shape, scale=mu / shape, size=n)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="gamma",
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        result = model.drop1(X, y, test="F")

        assert result.iloc[0]["p_value"] < 0.05


class TestDrop1DispersionScaling:
    @pytest.mark.parametrize("test", ["Chisq", "F"])
    def test_zero_dispersion_and_zero_deviance_change_is_null_result(self, test):
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
        y = np.full(len(X), 3.5)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y)

        row = model.drop1(X, y, test=test).iloc[0]

        assert model.result.phi == 0.0
        assert row["delta_deviance"] == 0.0
        assert row["statistic"] == 0.0
        assert row["p_value"] == 1.0

    @pytest.mark.parametrize("test", ["Chisq", "F"])
    def test_zero_dispersion_with_nonzero_deviance_change_is_explicitly_undefined(self, test):
        x = np.arange(20, dtype=float)
        X = pd.DataFrame({"x": x})
        y = 2.0 + x
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y)

        assert model.result.phi == 0.0
        with pytest.raises(ValueError, match="undefined.*dispersion phi is zero"):
            model.drop1(X, y, test=test)

    def test_known_scale_chisq_uses_unit_not_pearson_dispersion(self):
        """Poisson's deviance LRT keeps phi=1 even under observed overdispersion."""
        rng = np.random.default_rng(2180)
        n = 500
        x = rng.normal(size=n)
        mu = np.exp(0.2 + 0.3 * x)
        theta = 0.8
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y)

        row = model.drop1(X, y).loc[lambda table: table["feature"] == "x"].iloc[0]
        metrics = model.metrics(X, y)
        pearson_dispersion = metrics.pearson_chi2 / (metrics.n_obs - metrics.effective_df)
        assert model.result.phi == 1.0
        assert pearson_dispersion > 2.0
        assert row["statistic"] == pytest.approx(row["delta_deviance"])
        assert row["p_value"] == pytest.approx(chi2.sf(row["delta_deviance"], row["delta_df"]))

    def test_known_scale_f_uses_unit_scale_and_frequency_residual_df(self):
        rng = np.random.default_rng(2182)
        n = 180
        x = rng.normal(size=n)
        y = rng.poisson(np.exp(0.1 + 0.25 * x)).astype(float)
        weights = rng.integers(1, 5, size=n).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y, sample_weight=weights)

        row = model.drop1(X, y, sample_weight=weights, test="F").iloc[0]
        expected_stat = row["delta_deviance"] / row["delta_df"]
        expected_residual_df = np.sum(weights) - model.result.effective_df

        assert model.result.phi == 1.0
        assert row["statistic"] == pytest.approx(expected_stat)
        assert row["p_value"] == pytest.approx(
            f_dist.sf(expected_stat, row["delta_df"], expected_residual_df)
        )

    def test_gaussian_chisq_test_has_near_nominal_seeded_type_i_error(self):
        """The default chi-square statistic must remove estimated dispersion."""
        rng = np.random.default_rng(218)
        reported_p = []
        unscaled_p = []

        for _ in range(60):
            x = rng.normal(size=60)
            y = rng.normal(scale=3.0, size=60)
            X = pd.DataFrame({"noise": x})
            model = SuperGLM(
                family="gaussian",
                selection_penalty=0.0,
                features={"noise": Numeric()},
            ).fit(X, y)

            table = model.drop1(X, y)
            row = table.loc[table["feature"] == "noise"].iloc[0]
            expected_stat = row["delta_deviance"] / model.result.phi
            assert row["statistic"] == pytest.approx(expected_stat)
            assert row["p_value"] == pytest.approx(chi2.sf(expected_stat, row["delta_df"]))
            reported_p.append(row["p_value"])
            unscaled_p.append(chi2.sf(row["delta_deviance"], row["delta_df"]))

        # This seeded ensemble has a nominal-scale rejection count, while the
        # unscaled mutation rejects almost half of the null covariates.
        assert 1 <= np.count_nonzero(np.asarray(reported_p) < 0.05) <= 8
        assert np.count_nonzero(np.asarray(unscaled_p) < 0.05) >= 20
        assert 0.25 < np.median(reported_p) < 0.75

    @pytest.mark.parametrize("test", ["Chisq", "F"])
    def test_estimated_scale_weighted_tests_match_frequency_row_replication(self, test):
        rng = np.random.default_rng(2183)
        n = 70
        x = rng.normal(size=n)
        y = 0.4 * x + rng.normal(scale=2.0, size=n)
        weights = rng.integers(1, 5, size=n).astype(float)
        X = pd.DataFrame({"x": x})

        weighted_model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y, sample_weight=weights)
        weighted = weighted_model.drop1(
            X,
            y,
            sample_weight=weights,
            test=test,
        ).iloc[0]

        repeated_rows = np.repeat(np.arange(n), weights.astype(int))
        repeated_X = X.iloc[repeated_rows].reset_index(drop=True)
        repeated_y = y[repeated_rows]
        repeated_model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(repeated_X, repeated_y)
        repeated = repeated_model.drop1(repeated_X, repeated_y, test=test).iloc[0]

        assert weighted["statistic"] == pytest.approx(repeated["statistic"], rel=2e-12)
        assert weighted["p_value"] == pytest.approx(repeated["p_value"], rel=2e-12)

    def test_single_feature_offset_reduction_is_an_exact_intercept_only_refit(self):
        rng = np.random.default_rng(2184)
        n = 240
        x = rng.normal(size=n)
        offset = np.linspace(-1.5, 1.2, n)
        y = rng.poisson(np.exp(0.2 + 0.35 * x + offset)).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y, offset=offset)

        row = model.drop1(X, y, offset=offset).iloc[0]
        intercept_only = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={},
        ).fit(X, y, offset=offset)
        expected_delta = intercept_only.result.deviance - model.result.deviance

        assert row["deviance_reduced"] == pytest.approx(intercept_only.result.deviance)
        assert row["delta_deviance"] == pytest.approx(expected_delta)
        assert row["statistic"] == pytest.approx(expected_delta)

    def test_tweedie_drop1_rejects_nonpositive_prior_weights(self):
        rng = np.random.default_rng(2185)
        n = 80
        X = pd.DataFrame({"x": rng.normal(size=n)})
        y = np.maximum(rng.gamma(shape=1.5, scale=1.0, size=n), 1e-6)
        valid_weights = np.ones(n)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y, sample_weight=valid_weights)
        invalid_weights = valid_weights.copy()
        invalid_weights[0] = 0.0

        with pytest.raises(ValueError, match="strictly positive"):
            model.drop1(X, y, sample_weight=invalid_weights)


class TestDrop1Interactions:
    def test_drops_dependent_interaction(self):
        """Dropping a main effect should also drop its interaction."""
        rng = np.random.default_rng(42)
        n = 500
        age = rng.uniform(18, 85, n)
        region = rng.choice(["A", "B"], n)
        mu = np.exp(-1.0 + 0.01 * age)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"age": age, "region": region})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={
                "age": Spline(n_knots=5, penalty="ssp"),
                "region": Categorical(base="first"),
            },
            interactions=[("age", "region")],
        )
        model.fit(X, y)

        # Should not error — interaction is dropped with parent
        result = model.drop1(X, y)
        assert len(result) == 2  # age, region — not the interaction
        assert set(result["feature"]) == {"age", "region"}


class TestDrop1FractionalEdf:
    def test_fractional_delta_df_preserved(self):
        """delta_df should preserve fractional values from penalized edf, not floor to 1."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.standard_normal(n)
        # x2 is a weak spline effect — will have small edf under penalty
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + 0.3 * x1 + 0.05 * np.sin(2 * np.pi * x2)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            spline_penalty=10.0,  # strong smoothing → fractional edf
            features={"x1": Numeric(), "x2": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        result = model.drop1(X, y)

        x2_row = result[result["feature"] == "x2"].iloc[0]
        # With strong smoothing (spline_penalty=10) on a weak effect, delta_df should be
        # fractional (between 0 and n_basis), reflecting the effective degrees of
        # freedom consumed by the smoothed spline.
        assert x2_row["delta_df"] > 0, "delta_df must be positive"
        # With heavy smoothing, edf should be well below the nominal basis size (10 knots → ~14 cols)
        assert x2_row["delta_df"] < 14, f"delta_df={x2_row['delta_df']:.2f} exceeds basis dimension"
        # p-value and deviance change should be present and finite
        assert np.isfinite(x2_row["delta_deviance"])
        assert np.isfinite(x2_row["p_value"])
        assert x2_row["delta_deviance"] >= 0
