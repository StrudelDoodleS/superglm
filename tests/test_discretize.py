"""Tests for spline discretization impact analysis."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    DiscretizationResult,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
    discretization_impact,
)
from superglm.distributions import Poisson, Tweedie
from superglm.penalties.group_lasso import GroupLasso

_EPS = float(np.finfo(np.float64).eps)
# A grid cell and the exact contribution are the same bilinear form reached by
# two associations -- a matrix chain in ``reconstruct``, an einsum in ``score``.
# Charging each the standard p-term inner-product bound at p = 49 coefficients,
# twice, and one ulp for the ``exp``, gives ~100 u; 128 u is that rounded up.
# The same constant and the same derivation as ``_GRID_CELL_RTOL`` in
# ``test_rating_table_prediction_equivalence``.
_NODE_EXACT_RTOL = 128 * _EPS


@pytest.fixture
def fitted_model():
    """Fit a model with spline, categorical, numeric, and polynomial features."""
    rng = np.random.default_rng(42)
    n = 3000
    x_spline = rng.uniform(18, 80, n)
    x_cat = rng.choice(["A", "B", "C"], n)
    x_num = rng.normal(0, 1, n)
    x_poly = rng.uniform(0, 10, n)
    eta = 0.02 * (x_spline - 40) ** 2 / 400 - 0.3 * (x_cat == "B") + 0.1 * x_num
    y = rng.poisson(np.exp(eta)).astype(float)
    sample_weight = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame(
        {
            "age": x_spline,
            "region": x_cat,
            "score": x_num,
            "density": x_poly,
        }
    )

    m = SuperGLM(
        family=Poisson(),
        penalty=GroupLasso(lambda1=0.01),
        features={
            "age": Spline(n_knots=10, penalty="ssp"),
            "region": Categorical(),
            "score": Numeric(),
            "density": Polynomial(degree=3),
        },
    )
    m.fit(df, y, sample_weight=sample_weight)
    return m, df, y, sample_weight


class TestRatingTableSchema:
    def test_correct_columns(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        expected_cols = {
            "bin_from",
            "bin_to",
            "relativity",
            "log_relativity",
            "n_obs",
            "sample_weight",
        }
        for name, table in result.tables.items():
            assert set(table.columns) == expected_cols, f"Wrong columns for {name}"

    def test_n_bins_rows(self, fitted_model):
        m, df, y, w = fitted_model
        for n_bins in [5, 10, 20]:
            result = m.discretization_impact(df, y, sample_weight=w, n_bins=n_bins)
            for name, table in result.tables.items():
                assert len(table) <= n_bins, (
                    f"Feature {name}: got {len(table)} rows for n_bins={n_bins}"
                )
                assert len(table) >= 1


class TestFeatureFiltering:
    def test_non_spline_features_excluded(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        # Only spline and polynomial features should appear
        assert "region" not in result.tables
        assert "score" not in result.tables
        # Spline and polynomial should be present
        assert "age" in result.tables
        assert "density" in result.tables

    def test_features_param_restricts(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, features=["age"])
        assert list(result.tables.keys()) == ["age"]

    def test_invalid_feature_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="Unknown feature"):
            m.discretization_impact(df, y, sample_weight=w, features=["nonexistent"])

    def test_non_continuous_feature_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="not a spline or polynomial"):
            m.discretization_impact(df, y, sample_weight=w, features=["region"])


class TestBinCoverage:
    def test_bins_span_data_range(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        for name, table in result.tables.items():
            x_vals = df[name].values
            assert table["bin_from"].iloc[0] <= x_vals.min() + 1e-10
            assert table["bin_to"].iloc[-1] >= x_vals.max() - 1e-10

    def test_no_gaps_between_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        for name, table in result.tables.items():
            if len(table) > 1:
                # Each bin_from should equal the previous bin_to
                for i in range(1, len(table)):
                    assert table["bin_to"].iloc[i - 1] == table["bin_from"].iloc[i], (
                        f"Gap between bins {i - 1} and {i} for {name}"
                    )

    def test_all_observations_assigned(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        n = len(df)
        for name, table in result.tables.items():
            assert table["n_obs"].sum() == n, (
                f"Total n_obs={table['n_obs'].sum()} != {n} for {name}"
            )


class TestPredictions:
    def test_predictions_shape(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        assert result.predictions.shape == (len(df),)
        assert result.original_predictions.shape == (len(df),)

    def test_predictions_positive(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        assert np.all(result.predictions > 0)
        assert np.all(result.original_predictions > 0)

    def test_original_matches_model_predict(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        expected = m.predict(df)
        np.testing.assert_allclose(result.original_predictions, expected, rtol=1e-10)

    def test_works_when_interactions_are_present(self):
        rng = np.random.default_rng(1234)
        n = 400
        df = pd.DataFrame(
            {
                "age": rng.uniform(18, 80, n),
                "region": rng.choice(["A", "B", "C"], n),
                "segment": rng.choice(["X", "Y"], n),
            }
        )
        eta = (
            -1.0
            + 0.15 * np.sin(df["age"].to_numpy() / 8.0)
            + 0.2 * (df["region"].to_numpy() == "B")
            + 0.1 * ((df["region"].to_numpy() == "C") & (df["segment"].to_numpy() == "Y"))
        )
        y = rng.poisson(np.exp(eta)).astype(float)
        w = rng.uniform(0.5, 2.0, n)
        model = SuperGLM(
            family=Poisson(),
            penalty=GroupLasso(lambda1=0.0),
            features={
                "age": Spline(n_knots=8),
                "region": Categorical(base="first"),
                "segment": Categorical(base="first"),
            },
            interactions=[("region", "segment")],
        )
        model.fit(df, y, sample_weight=w)

        result = model.discretization_impact(df, y, sample_weight=w, n_bins=20)

        assert list(result.tables) == ["age"]
        np.testing.assert_allclose(result.original_predictions, model.predict(df), rtol=1e-10)


class TestMetrics:
    def test_all_metric_keys_present(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        expected_keys = {
            "deviance_original",
            "deviance_discretized",
            "deviance_change",
            "deviance_change_pct",
            "max_abs_prediction_change_pct",
            "mean_abs_prediction_change_pct",
            "prediction_correlation",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_deviance_change_consistent(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        expected_change = (
            result.metrics["deviance_discretized"] - result.metrics["deviance_original"]
        )
        assert abs(result.metrics["deviance_change"] - expected_change) < 1e-10

    def test_high_correlation(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=20)
        assert result.metrics["prediction_correlation"] > 0.95


class TestSmallImpactAtHighBinCount:
    def test_tiny_deviance_change_at_50_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=50)
        assert abs(result.metrics["deviance_change_pct"]) < 1.0, (
            f"Deviance change {result.metrics['deviance_change_pct']:.4f}% too large at 50 bins"
        )

    def test_high_correlation_at_50_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=50)
        assert result.metrics["prediction_correlation"] > 0.99


class TestConvenienceMethod:
    def test_model_method_matches_function(self, fitted_model):
        m, df, y, w = fitted_model
        r1 = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        r2 = discretization_impact(m, df, y, sample_weight=w, n_bins=10)
        np.testing.assert_array_equal(r1.predictions, r2.predictions)
        assert r1.metrics == r2.metrics

    def test_returns_discretization_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w)
        assert isinstance(result, DiscretizationResult)


class TestBinStrategy:
    def test_uniform_equal_width(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10, bin_strategy="uniform")
        for name, table in result.tables.items():
            widths = (table["bin_to"] - table["bin_from"]).values
            # All bins should have the same width
            np.testing.assert_allclose(
                widths, widths[0], rtol=1e-10, err_msg=f"Unequal bin widths for {name}"
            )

    def test_uniform_produces_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, sample_weight=w, n_bins=10, bin_strategy="uniform")
        assert isinstance(result, DiscretizationResult)
        assert result.predictions.shape == (len(df),)

    def test_winsorized_structure(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(
            df, y, sample_weight=w, n_bins=10, bin_strategy="winsorized"
        )
        for name, table in result.tables.items():
            x_vals = df[name].values
            # First bin starts at data min, last ends at data max
            assert table["bin_from"].iloc[0] <= x_vals.min() + 1e-10
            assert table["bin_to"].iloc[-1] >= x_vals.max() - 1e-10
            # Should produce at least 3 bins (left tail, interior, right tail)
            assert len(table) >= 3, f"Expected >= 3 bins for {name}"
            # Bins should be contiguous
            if len(table) > 1:
                for i in range(1, len(table)):
                    assert table["bin_to"].iloc[i - 1] == table["bin_from"].iloc[i]

    def test_winsorized_produces_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(
            df, y, sample_weight=w, n_bins=10, bin_strategy="winsorized"
        )
        assert isinstance(result, DiscretizationResult)

    def test_invalid_strategy_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="Unknown bin_strategy"):
            m.discretization_impact(df, y, sample_weight=w, bin_strategy="bogus")

    def test_default_is_exposure_quantile(self, fitted_model):
        m, df, y, w = fitted_model
        r_default = m.discretization_impact(df, y, sample_weight=w, n_bins=10)
        r_explicit = m.discretization_impact(
            df, y, sample_weight=w, n_bins=10, bin_strategy="exposure_quantile"
        )
        np.testing.assert_array_equal(r_default.predictions, r_explicit.predictions)

    def test_all_strategies_cover_all_obs(self, fitted_model):
        m, df, y, w = fitted_model
        n = len(df)
        for strategy in ["exposure_quantile", "uniform", "winsorized"]:
            result = m.discretization_impact(
                df, y, sample_weight=w, n_bins=10, bin_strategy=strategy
            )
            for name, table in result.tables.items():
                assert table["n_obs"].sum() == n, f"strategy={strategy}, {name}: n_obs sum != {n}"


class TestDefaultExposure:
    def test_works_without_exposure(self, fitted_model):
        m, df, y, _ = fitted_model
        result = m.discretization_impact(df, y, n_bins=10)
        assert isinstance(result, DiscretizationResult)
        assert result.predictions.shape == (len(df),)


class TestWeightContract:
    @pytest.mark.parametrize("bin_strategy", ["exposure_quantile", "uniform", "winsorized"])
    def test_frequency_weights_match_literal_row_replication(
        self,
        fitted_model,
        bin_strategy,
    ):
        model, frame, y, _ = fitted_model
        weights = (1 + np.arange(len(frame)) % 4).astype(np.float64)
        rows = np.repeat(np.arange(len(frame)), weights.astype(np.int64))

        weighted = model.discretization_impact(
            frame,
            y,
            sample_weight=weights,
            n_bins=12,
            bin_strategy=bin_strategy,
            features=["age"],
        )
        repeated = model.discretization_impact(
            frame.iloc[rows].reset_index(drop=True),
            y[rows],
            n_bins=12,
            bin_strategy=bin_strategy,
            features=["age"],
        )

        weighted_table = weighted.tables["age"]
        repeated_table = repeated.tables["age"]
        pd.testing.assert_frame_equal(
            weighted_table.drop(columns="n_obs"),
            repeated_table.drop(columns="n_obs"),
            check_exact=False,
            rtol=2e-14,
            atol=2e-14,
        )
        assert weighted_table["n_obs"].sum() == len(frame)
        assert repeated_table["n_obs"].sum() == len(rows)
        np.testing.assert_allclose(
            np.repeat(weighted.predictions, weights.astype(np.int64)),
            repeated.predictions,
            rtol=2e-14,
            atol=2e-14,
        )
        for result in (weighted, repeated):
            assert result.metrics["deviance_change"] == (
                result.metrics["deviance_discretized"] - result.metrics["deviance_original"]
            )
            assert result.metrics["deviance_change_pct"] == (
                100.0 * result.metrics["deviance_change"] / result.metrics["deviance_original"]
            )

        deviance_scale = max(
            abs(weighted.metrics["deviance_original"]),
            abs(weighted.metrics["deviance_discretized"]),
            abs(repeated.metrics["deviance_original"]),
            abs(repeated.metrics["deviance_discretized"]),
        )
        change_atol = 64.0 * np.finfo(np.float64).eps * deviance_scale
        assert weighted.metrics["deviance_change"] == pytest.approx(
            repeated.metrics["deviance_change"],
            rel=2e-13,
            abs=change_atol,
        )
        pct_atol = (
            100.0
            * change_atol
            / min(
                weighted.metrics["deviance_original"],
                repeated.metrics["deviance_original"],
            )
        )
        assert weighted.metrics["deviance_change_pct"] == pytest.approx(
            repeated.metrics["deviance_change_pct"],
            rel=2e-13,
            abs=pct_atol,
        )

        for key in weighted.metrics.keys() - {"deviance_change", "deviance_change_pct"}:
            assert weighted.metrics[key] == pytest.approx(
                repeated.metrics[key],
                rel=2e-13,
                abs=2e-13,
            )

    def test_zero_frequency_rows_do_not_widen_uniform_geometry(self, fitted_model):
        model, frame, y, _ = fitted_model
        frame = frame.iloc[:80].copy()
        y = y[:80]
        frame.loc[frame.index[-1], "age"] = 1.0e6
        weights = np.ones(len(frame))
        weights[-1] = 0.0

        weighted = model.discretization_impact(
            frame,
            y,
            sample_weight=weights,
            n_bins=8,
            bin_strategy="uniform",
            features=["age"],
        )
        filtered = model.discretization_impact(
            frame.iloc[:-1],
            y[:-1],
            n_bins=8,
            bin_strategy="uniform",
            features=["age"],
        )

        assert weighted.tables["age"]["bin_to"].iloc[-1] == pytest.approx(
            filtered.tables["age"]["bin_to"].iloc[-1]
        )
        np.testing.assert_allclose(weighted.predictions[:-1], filtered.predictions)
        for key in weighted.metrics:
            assert weighted.metrics[key] == pytest.approx(filtered.metrics[key])

    def test_tweedie_prior_weights_leave_geometry_and_prediction_summaries_physical(
        self,
        fitted_model,
    ):
        model, frame, y, _ = fitted_model
        model._distribution = Tweedie(p=1.5)
        weights = np.linspace(0.2, 3.0, len(frame))

        unit = model.discretization_impact(
            frame,
            y,
            n_bins=12,
            features=["age"],
        )
        weighted = model.discretization_impact(
            frame,
            y,
            sample_weight=weights,
            n_bins=12,
            features=["age"],
        )

        columns = ["bin_from", "bin_to", "relativity", "log_relativity", "n_obs"]
        pd.testing.assert_frame_equal(
            weighted.tables["age"][columns],
            unit.tables["age"][columns],
        )
        for key in (
            "max_abs_prediction_change_pct",
            "mean_abs_prediction_change_pct",
            "prediction_correlation",
        ):
            assert weighted.metrics[key] == unit.metrics[key]
        assert weighted.metrics["deviance_original"] != unit.metrics["deviance_original"]

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.ones(4), "length"),
            (np.array([1.0, np.nan, 1.0]), "finite"),
            (np.array([1.0, -1.0, 1.0]), "nonnegative"),
            (np.zeros(3), "all zero"),
            (np.ones((3, 1)), "one-dimensional"),
        ],
    )
    def test_validates_frequency_weights(self, fitted_model, weights, message):
        model, frame, y, _ = fitted_model
        with pytest.raises(ValueError, match=message):
            model.discretization_impact(
                frame.iloc[:3],
                y[:3],
                sample_weight=weights,
                features=["age"],
            )

    def test_tweedie_requires_strictly_positive_prior_weights(self, fitted_model):
        model, frame, y, _ = fitted_model
        model._distribution = Tweedie(p=1.5)
        weights = np.ones(len(frame))
        weights[0] = 0.0

        with pytest.raises(ValueError, match="strictly positive"):
            model.discretization_impact(
                frame,
                y,
                sample_weight=weights,
                features=["age"],
            )


@pytest.fixture
def gridded_interaction_model():
    """Two spline parents and the tensor interaction between them.

    The interaction is the term the rating-table export ships as a sampled
    surface rather than as a lookup, so it is the one main effects do not
    stand in for.
    """
    rng = np.random.default_rng(5)
    n = 600
    df = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "density": rng.uniform(0.0, 10.0, n)})
    mu = np.exp(-1.0 + 0.02 * df["age"] + 0.05 * df["density"] + 0.004 * df["age"] * df["density"])
    y = rng.poisson(mu).astype(float)
    model = SuperGLM(
        family=Poisson(),
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "density": Spline(n_knots=6)},
        interactions=[("age", "density")],
    )
    model.fit(df, y)
    return model, df, y


class TestContinuousInteractionIsDiscretized:
    """A gridded interaction is an approximation, so the default answer counts it.

    The impact analysis answers "how far does the discretized table sit from the
    model".  Reporting only the binned MAIN EFFECTS of a fit that also carries a
    gridded interaction answers a question about a table nobody exports
    (issue #287).
    """

    def test_the_default_call_covers_the_interaction(self, gridded_interaction_model):
        model, df, y = gridded_interaction_model
        result = model.discretization_impact(df, y, n_bins=20)

        assert set(result.tables) == {"age", "density"}
        assert set(result.interaction_tables) == {"age:density"}

    def test_the_grid_carries_a_main_effect_tables_information(self, gridded_interaction_model):
        model, df, y = gridded_interaction_model
        grid = model.discretization_impact(df, y, n_bins=20).interaction_tables["age:density"]

        assert set(grid.columns) == {
            "age",
            "density",
            "relativity",
            "log_relativity",
            "n_obs",
            "sample_weight",
        }
        # One row per cell of the 20-per-axis grid, and every observation lands
        # on exactly one of them -- the same completeness a binned table has.
        assert len(grid) == 400
        assert int(grid["n_obs"].sum()) == len(df)
        assert float(grid["sample_weight"].sum()) == pytest.approx(float(len(df)))
        np.testing.assert_allclose(
            grid["relativity"].to_numpy(),
            np.exp(grid["log_relativity"].to_numpy()),
            rtol=0.0,
            atol=0.0,
        )

    def test_a_risk_sitting_on_a_grid_node_is_approximated_exactly(self, gridded_interaction_model):
        """The lookup rule is pinned by the case where it must cost nothing.

        A row whose two parent values ARE grid nodes gets the surface at its own
        location, so the replacement is the exact contribution and the
        discretized prediction must equal the smooth one.  Exact rather than
        ordered: a modulus-of-continuity argument shrinks the error BOUND with
        the node spacing but does not force the realised error over a fixed set
        of rows to fall, so comparing two fitted error measurements would be a
        constant fitted to this fit rather than a derived one.

        It also discriminates.  Any of the failures that a symmetric fixture
        hides -- reading the surface transposed, keying the axes on the wrong
        parent, an off-by-one in the nearest-node search -- moves a node row off
        its own value and breaks the equality, and the midpoint case below
        confirms the identity is not vacuous.
        """
        model, df, y = gridded_interaction_model
        n_bins = 12
        grid = model.discretization_impact(
            df, y, n_bins=n_bins, features=["age:density"]
        ).interaction_tables["age:density"]
        axis_age = np.array(sorted(set(grid["age"])), dtype=float)
        axis_density = np.array(sorted(set(grid["density"])), dtype=float)

        on_nodes = pd.DataFrame(
            {
                "age": np.repeat(axis_age, len(axis_density)),
                "density": np.tile(axis_density, len(axis_age)),
            }
        )
        y_nodes = np.ones(len(on_nodes))
        at_nodes = model.discretization_impact(
            on_nodes, y_nodes, n_bins=n_bins, features=["age:density"]
        )
        np.testing.assert_allclose(
            at_nodes.predictions,
            at_nodes.original_predictions,
            rtol=_NODE_EXACT_RTOL,
            atol=0.0,
        )
        assert at_nodes.metrics["max_abs_prediction_change_pct"] == pytest.approx(
            0.0, abs=100.0 * _NODE_EXACT_RTOL
        )

        # Halfway between nodes is the worst case for the same rule, and it is
        # not zero -- so the equality above is a property of sitting on a node.
        between = pd.DataFrame(
            {
                "age": np.repeat((axis_age[:-1] + axis_age[1:]) / 2.0, len(axis_density) - 1),
                "density": np.tile((axis_density[:-1] + axis_density[1:]) / 2.0, len(axis_age) - 1),
            }
        )
        off_nodes = model.discretization_impact(
            between, np.ones(len(between)), n_bins=n_bins, features=["age:density"]
        )
        assert off_nodes.metrics["max_abs_prediction_change_pct"] > 0.0

    def test_an_interaction_can_be_asked_for_by_name(self, gridded_interaction_model):
        model, df, y = gridded_interaction_model
        result = model.discretization_impact(df, y, n_bins=20, features=["age:density"])

        assert result.tables == {}
        assert set(result.interaction_tables) == {"age:density"}

    def test_a_repeated_name_is_not_counted_twice(self, gridded_interaction_model):
        """One block ships once, so its error is counted once.

        A repeated name used to add its replacement delta once per occurrence
        while writing a single table, so the sheet reported twice the
        discretisation error of a block the workbook carries once.
        """
        model, df, y = gridded_interaction_model
        once = model.discretization_impact(df, y, n_bins=20, features=["age:density"])
        twice = model.discretization_impact(
            df, y, n_bins=20, features=["age:density", "age:density"]
        )

        np.testing.assert_allclose(twice.predictions, once.predictions, rtol=0.0, atol=0.0)
        assert twice.metrics == once.metrics
        assert set(twice.interaction_tables) == {"age:density"}

    def test_an_unknown_name_is_still_an_unknown_feature(self, gridded_interaction_model):
        model, df, y = gridded_interaction_model
        with pytest.raises(ValueError, match="Unknown feature"):
            model.discretization_impact(df, y, features=["age:nonexistent"])


class TestGridParentsAreReadTheWayTheFitReadsThem:
    """The columns the sweep keys on must be the ones the grid is built over."""

    @staticmethod
    def _ordered_parent_model():
        bands = ["18-25", "26-35", "36-50", "51-65", "66+"]
        rng = np.random.default_rng(3)
        n = 900
        df = pd.DataFrame(
            {
                "band": rng.choice(bands, n),
                "density": rng.uniform(0.0, 10.0, n),
            }
        )
        eta = (
            -1.0
            + 0.15 * np.array([bands.index(b) for b in df["band"]])
            + 0.05 * df["density"].to_numpy()
        )
        y = rng.poisson(np.exp(eta)).astype(float)
        model = SuperGLM(
            family=Poisson(),
            selection_penalty=0.0,
            features={
                "band": OrderedCategorical(order=bands, basis=Spline(n_knots=4)),
                "density": Spline(n_knots=5),
            },
            interactions=[("band", "density")],
        )
        model.fit(df, y)
        return model, df, y

    def test_an_ordered_categorical_parent_is_resolved_to_its_scores(self):
        """The frame holds labels; the grid axis is in mapped-score space.

        A spline-mode ``OrderedCategorical`` parent contributes its inner spline
        on mapped numeric scores, so ``"66+"`` is what the frame carries and a
        number is what the axis carries.  Reading the column as float64 raised
        ``could not convert string to float`` and took the whole export down --
        a model that exported cleanly on master.

        Pinned by the space rather than by the absence of an exception.  Any
        numeric resolution -- level codes, positions, anything -- would clear a
        "does not raise" check, because ``_nearest_grid_index`` clamps an
        out-of-range value onto an end node and every row still lands in some
        cell while being silently mis-rated.  A row whose band's mapped score
        IS a grid node can only be approximated exactly if the sweep read the
        same space the grid was built over.
        """
        model, df, y = self._ordered_parent_model()
        # ``n_bins=5`` is load-bearing: ``OrderedCategorical`` maps its five
        # levels to ``linspace(0, 1, 5)`` and the grid axis is the training
        # range of those scores at five nodes, so the two arrays are identical
        # and EVERY band sits on a node. At any other resolution only the two
        # extreme bands do, and a wrong-but-numeric resolution -- level codes
        # 0..4 -- would clamp those same two onto the same end nodes and clear
        # the exactness assertion below while silently mis-rating the interior.
        result = model.discretization_impact(df, y, n_bins=5)

        assert set(result.interaction_tables) == {"band:density"}
        grid = result.interaction_tables["band:density"]
        assert int(grid["n_obs"].sum()) == len(df)
        assert np.isfinite(result.metrics["max_abs_prediction_change_pct"])

        # The band axis really is score space, not label or code space: every
        # node is one of the parent's own mapped scores.
        spec = model._specs["band"]
        scores = np.array(sorted(spec._level_to_value.values()), dtype=float)
        band_axis = np.array(sorted(set(grid[grid.columns[0]])), dtype=float)
        assert band_axis.min() == pytest.approx(scores.min())
        assert band_axis.max() == pytest.approx(scores.max())

        # And a frame sitting on nodes of both axes is approximated exactly,
        # which resolution to any other numeric space would break.
        density_axis = np.array(sorted(set(grid[grid.columns[1]])), dtype=float)
        on_node_bands = [
            label
            for label, score in spec._level_to_value.items()
            if np.min(np.abs(band_axis - float(score))) < 1e-12
        ]
        on_nodes = pd.DataFrame(
            {
                "band": np.repeat(on_node_bands, len(density_axis)),
                "density": np.tile(density_axis, len(on_node_bands)),
            }
        )
        assert len(on_node_bands) == 5, "every band must sit on a node for this to bite"
        at_nodes = model.discretization_impact(
            on_nodes, np.ones(len(on_nodes)), n_bins=5, features=["band:density"]
        )
        np.testing.assert_allclose(
            at_nodes.predictions,
            at_nodes.original_predictions,
            rtol=_NODE_EXACT_RTOL,
            atol=0.0,
        )

    def test_the_rating_table_export_of_such_a_model_still_builds(self):
        from superglm.export.rating_tables import build_rating_table_payload

        model, df, y = self._ordered_parent_model()
        payload = build_rating_table_payload(model, df, y, n_bins=5, impact_bins=(5,))

        assert "band:density" in set(payload.discretization_impact["feature"])


class TestBothGridAxesSurviveIntoTheTable:
    """A cell is two axis values, so the table needs two axis columns."""

    def test_a_same_feature_interaction_keeps_both_axes(self):
        """``interactions=[("age", "age")]`` fits and exports, so it must tabulate.

        Both parent names are the same string, and a later key in a DataFrame
        dict literal overwrites an earlier one, so the table silently carried
        one axis for a two-dimensional grid.
        """
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"age": rng.uniform(18.0, 80.0, 400)})
        y = rng.poisson(np.exp(-1.0 + 0.02 * df["age"])).astype(float)
        model = SuperGLM(
            family=Poisson(),
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6)},
            interactions=[("age", "age")],
        )
        model.fit(df, y)

        grid = model.discretization_impact(df, y, n_bins=5).interaction_tables["age:age"]

        assert len(grid) == 25
        axis_columns = [c for c in grid.columns if c.startswith("age")]
        assert axis_columns == ["age (axis 1)", "age (axis 2)"]
        # Two axes really are present: each takes every node value.
        for column in axis_columns:
            assert len(set(grid[column])) == 5

    def test_a_parent_named_for_a_value_column_keeps_its_axis(self):
        """A feature may legitimately be called ``relativity``."""
        from superglm.diagnostics.discretize import _axis_column_labels

        assert _axis_column_labels("relativity", "density") == (
            "relativity (axis 1)",
            "density (axis 2)",
        )
        assert _axis_column_labels("age", "n_obs") == ("age (axis 1)", "n_obs (axis 2)")
        assert _axis_column_labels("age", "density") == ("age", "density")


class TestTheSweepAndTheExporterAgreeOnWhatAGridIs:
    """One rule, applied in one place, for both the block and the sheet.

    ``_interaction_blocks`` ships a grid on the reconstruction's KEYS, calling
    ``reconstruct`` with ``n_points`` only when the signature takes one and
    accepting either axis orientation.  Any second rule -- a class check, a
    signature pre-filter, a stricter shape test -- makes the exporter and the
    sweep disagree about the same interaction, which is the shape issue #287
    took.  These pin the two shapes where a second rule diverged.
    """

    class _GridWithoutNPoints:
        """A grid reconstructor whose signature takes no ``n_points``."""

        parent_names = ("a", "b")

        def reconstruct(self, beta):
            x1 = np.linspace(0.0, 1.0, 4)
            x2 = np.linspace(0.0, 1.0, 4)
            surface = np.outer(x1, x2)
            return {
                "x1": x1,
                "x2": x2,
                "log_relativity": surface,
                "relativity": np.exp(surface),
                "interaction": True,
            }

    class _GridInNaturalOrder:
        """A non-square grid already shaped ``(len(x1), len(x2))``."""

        parent_names = ("a", "b")

        def reconstruct(self, beta, n_points=50):
            x1 = np.linspace(0.0, 1.0, 3)
            x2 = np.linspace(0.0, 1.0, 5)
            surface = np.outer(x1, x2 + 1.0)
            return {
                "x1": x1,
                "x2": x2,
                "relativity": np.exp(surface),
                "interaction": True,
            }

    class _Stub:
        """A grid reconstructor with a surface that depends on axis 1 alone.

        ``score`` returns zero, so the per-row replacement delta IS the surface
        value at the node a risk lands on -- which makes the discretized
        predictions read the lookup out loud.
        """

        parent_names = ("a", "b")

        def __init__(self, *, descending=False, inconsistent=False):
            self.descending = descending
            self.inconsistent = inconsistent

        def score(self, x1, x2, beta):
            return np.zeros(len(np.asarray(x1).ravel()))

        def reconstruct(self, beta, n_points=50):
            axis = np.array([3.0, 2.0, 1.0]) if self.descending else np.array([1.0, 2.0, 3.0])
            other = (
                np.array([30.0, 20.0, 10.0]) if self.descending else np.array([10.0, 20.0, 30.0])
            )
            # ``surface[j, i] = f(x1[i], x2[j])``, the convention both built-ins
            # return and ``orient_grid_surface`` normalises.  It depends on
            # BOTH axes -- a surface constant along axis 2 cannot witness that
            # axis's order, and ``_ascending_grid`` sorts both.
            surface = axis[None, :] + 0.01 * other[:, None]
            return {
                "x1": axis,
                "x2": other,
                "relativity": np.exp(surface),
                # Zero where the exported ``relativity`` says otherwise, when
                # asked for -- the two disagree only for a custom spec.
                "log_relativity": np.zeros((3, 3)) if self.inconsistent else surface,
                "interaction": True,
            }

    @staticmethod
    def _model_with(stub):
        """A fitted model whose interaction spec is the stub.

        Substituted after the fit and the prediction plan reset, so the sweep
        reaches the stub through ``discretization_impact`` itself. Calling the
        helpers directly would leave both fixes revertible with the tests still
        green, which is the failure mode AGENTS.md's mutation requirement is
        about.
        """
        rng = np.random.default_rng(2)
        n = 300
        df = pd.DataFrame({"a": rng.uniform(1.0, 3.0, n), "b": rng.uniform(10.0, 30.0, n)})
        y = rng.poisson(np.exp(-1.0 + 0.1 * df["a"])).astype(float)
        model = SuperGLM(
            family=Poisson(),
            selection_penalty=0.0,
            features={"a": Numeric(), "b": Numeric()},
            interactions=[("a", "b")],
        )
        model.fit(df, y)
        model._interaction_specs["a:b"] = stub
        model._prediction_plan = None
        return model, df, y

    def test_a_descending_axis_is_sorted_before_the_binary_search(self):
        """``_nearest_grid_index`` bisects, so axis order is load-bearing.

        The exporter preserves whatever order a reconstruction supplies and
        applies no monotonicity gate. With the axes descending, a binary search
        maps a risk onto a non-nearest node while every cell count and metric
        keeps claiming the documented nearest-node lookup.

        Read through the predictions: the stub's surface is its own axis-1 node
        value and its ``score`` is zero, so each row's delta must be the node
        nearest its ``a``. Sorted, that is 1, 2 or 3.
        """
        model, df, y = self._model_with(self._Stub(descending=True))
        result = model.discretization_impact(df, y, n_bins=3, features=["a:b"])

        # BOTH axis columns are stated in order, not as a set: sorting one and
        # leaving the other reversed ships a table whose ``n_obs`` hangs on the
        # wrong cells, and the surface below is what detects it.
        table = result.interaction_tables["a:b"]
        assert list(table[table.columns[0]])[:3] == [1.0, 1.0, 1.0]
        assert list(table[table.columns[1]])[:3] == [10.0, 20.0, 30.0]
        assert sorted(set(table[table.columns[0]])) == [1.0, 2.0, 3.0]
        assert int(table["n_obs"].sum()) == len(df)

        nodes_a = np.array([1.0, 2.0, 3.0])
        nodes_b = np.array([10.0, 20.0, 30.0])
        delta = np.log(result.predictions / result.original_predictions)
        expected = (
            nodes_a[np.abs(df["a"].to_numpy()[:, None] - nodes_a).argmin(axis=1)]
            + 0.01 * nodes_b[np.abs(df["b"].to_numpy()[:, None] - nodes_b).argmin(axis=1)]
        )
        np.testing.assert_allclose(delta, expected, rtol=_NODE_EXACT_RTOL, atol=0.0)

    def test_the_swept_surface_is_the_one_the_workbook_prints(self):
        """``_continuous_interaction_block`` emits ``relativity``, so measure that.

        For the two built-ins the pair agrees to an ulp, but a custom
        reconstructor can return two surfaces that disagree -- and then
        measuring the one the workbook does not ship reports error for a table
        nobody holds. The stub's ``log_relativity`` is all zeros, so preferring
        it would make every delta zero and the impact vanish.
        """
        model, df, y = self._model_with(self._Stub(inconsistent=True))
        result = model.discretization_impact(df, y, n_bins=3, features=["a:b"])

        delta = np.log(result.predictions / result.original_predictions)
        nodes_a = np.array([1.0, 2.0, 3.0])
        nodes_b = np.array([10.0, 20.0, 30.0])
        expected = (
            nodes_a[np.abs(df["a"].to_numpy()[:, None] - nodes_a).argmin(axis=1)]
            + 0.01 * nodes_b[np.abs(df["b"].to_numpy()[:, None] - nodes_b).argmin(axis=1)]
        )
        np.testing.assert_allclose(delta, expected, rtol=_NODE_EXACT_RTOL, atol=0.0)
        # Preferring ``log_relativity`` would have made every delta zero.
        assert np.abs(delta).min() > 0.0

        table = result.interaction_tables["a:b"]
        np.testing.assert_allclose(
            table["log_relativity"].to_numpy(),
            np.log(table["relativity"].to_numpy()),
            rtol=_NODE_EXACT_RTOL,
            atol=0.0,
        )

    class _MappingGrid:
        """A reconstruction returned as a mapping that is not a ``dict``."""

        parent_names = ("a", "b")

        def score(self, x1, x2, beta):
            return np.zeros(len(np.asarray(x1).ravel()))

        def reconstruct(self, beta, n_points=50):
            import collections

            axis = np.array([1.0, 2.0, 3.0])
            surface = np.repeat(axis[None, :], 3, axis=0)
            return collections.UserDict(
                {
                    "x1": axis,
                    "x2": np.array([10.0, 20.0, 30.0]),
                    "relativity": np.exp(surface),
                    "interaction": True,
                }
            )

    def test_a_mapping_that_is_not_a_dict_is_still_a_grid(self):
        """The exporter's test is a key subset, so the sweep's must be too.

        ``_interaction_blocks`` only iterates keys and subscripts values, so a
        ``UserDict`` ships as a grid block; an ``isinstance(raw, dict)`` in the
        sweep would refuse the same reconstruction and take the whole payload
        down with it.
        """
        model, df, y = self._model_with(self._MappingGrid())
        result = model.discretization_impact(df, y, n_bins=3, features=["a:b"])

        assert set(result.interaction_tables) == {"a:b"}
        delta = np.log(result.predictions / result.original_predictions)
        nodes = np.array([1.0, 2.0, 3.0])
        expected = nodes[np.abs(df["a"].to_numpy()[:, None] - nodes).argmin(axis=1)]
        np.testing.assert_allclose(delta, expected, rtol=_NODE_EXACT_RTOL, atol=0.0)

    def test_the_exporter_and_the_sweep_use_one_predicate_object(self):
        """The last un-shared copy of the grid rule, now shared.

        Four review rounds found the same failure -- the exporter routes a grid
        on one rule and the sweep re-decides with a second (a class check, a
        signature pre-filter, a shape acceptance set, an ``isinstance``), and
        where they disagree the payload dies refusing a block that shipped.
        Asserting the two literals against each other would repeat the mistake:
        value-identical is what every one of those four also was. This asserts
        they are the SAME OBJECT, so a fifth divergence is not expressible.
        """
        from superglm.diagnostics.discretize import _GRID_RECONSTRUCTION_KEYS
        from superglm.export import rating_tables

        assert rating_tables._grid_reconstruction_keys() is _GRID_RECONSTRUCTION_KEYS

    def test_a_grid_reconstructor_without_n_points_is_still_a_grid(self):
        from superglm.diagnostics.discretize import _grid_reconstruction

        raw = _grid_reconstruction(self._GridWithoutNPoints(), np.zeros(1), 7)
        assert raw is not None
        assert {"x1", "x2", "relativity"} <= set(raw)

    def test_both_orientations_the_exporter_accepts_are_normalised(self):
        from superglm.diagnostics.discretize import orient_grid_surface

        spec = self._GridInNaturalOrder()
        raw = spec.reconstruct(np.zeros(1))
        axis1 = np.asarray(raw["x1"])
        axis2 = np.asarray(raw["x2"])
        natural = np.log(np.asarray(raw["relativity"]))

        # Already `(len(x1), len(x2))`: returned untouched.
        oriented = orient_grid_surface("a:b", axis1, axis2, natural)
        np.testing.assert_allclose(oriented, natural, rtol=0.0, atol=0.0)
        # The meshgrid convention `(len(x2), len(x1))`: transposed to match.
        np.testing.assert_allclose(
            orient_grid_surface("a:b", axis1, axis2, natural.T), natural, rtol=0.0, atol=0.0
        )
        # Anything else is refused rather than silently reshaped.
        with pytest.raises(ValueError, match="expected"):
            orient_grid_surface("a:b", axis1, axis2, np.zeros((2, 2)))


class TestExactlyTabulatedInteractionIsNotDiscretized:
    """An interaction the export tabulates exactly has no error to report."""

    @staticmethod
    def _model():
        rng = np.random.default_rng(11)
        n = 600
        df = pd.DataFrame(
            {
                "region": rng.choice(["A", "B", "C"], n),
                "band": rng.choice(["lo", "hi"], n),
                "age": rng.uniform(18.0, 80.0, n),
            }
        )
        y = rng.poisson(np.exp(-1.0 + 0.02 * df["age"])).astype(float)
        model = SuperGLM(
            family=Poisson(),
            selection_penalty=0.0,
            features={
                "region": Categorical(),
                "band": Categorical(),
                "age": Spline(n_knots=6),
            },
            interactions=[("region", "band")],
        )
        model.fit(df, y)
        return model, df, y

    def test_a_categorical_interaction_contributes_no_grid(self):
        model, df, y = self._model()
        result = model.discretization_impact(df, y, n_bins=20)

        assert set(result.tables) == {"age"}
        assert result.interaction_tables == {}

    def test_asking_for_it_by_name_says_why_not(self):
        model, df, y = self._model()
        with pytest.raises(ValueError, match="not continuous-by-continuous"):
            model.discretization_impact(df, y, features=["region:band"])
