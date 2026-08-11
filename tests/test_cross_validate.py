"""Tests for cross_validate() free function."""

import inspect
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from superglm import (
    Categorical,
    CrossValidationResult,
    GroupElasticNet,
    GroupLasso,
    Numeric,
    OrderedCategorical,
    Spline,
    SuperGLM,
    cross_validate,
)
from superglm._frame import as_eager_frame
from superglm.distributions import Gaussian, Poisson, Tweedie
from superglm.model_selection import (
    _clone_model,
    _pooled_deviance_parts,
    _pooled_nll_parts,
    _resolve_level_bindings,
    _score_deviance,
    _score_gini,
    _score_nll,
)

# ── Helpers ───────────────────────────────────────────────────────


class SimpleKFold:
    """Minimal splitter for tests (no sklearn dependency)."""

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for size in fold_sizes:
            test = indices[current : current + size]
            train = np.concatenate([indices[:current], indices[current + size :]])
            yield train, test
            current += size


class SimpleGroupKFold:
    """Minimal group-aware splitter (splits on unique group values)."""

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups required")
        unique_groups = np.unique(groups)
        fold_sizes = np.full(self.n_splits, len(unique_groups) // self.n_splits, dtype=int)
        fold_sizes[: len(unique_groups) % self.n_splits] += 1
        current = 0
        for size in fold_sizes:
            test_groups = unique_groups[current : current + size]
            test_mask = np.isin(groups, test_groups)
            yield np.where(~test_mask)[0], np.where(test_mask)[0]
            current += size


class RecordingTwoFold:
    """Deterministic splitter recording the native frame it receives."""

    def __init__(self) -> None:
        self.frame_types: list[type] = []

    def split(self, X, y=None, groups=None):
        del y, groups
        self.frame_types.append(type(X))
        midpoint = len(X) // 2
        yield np.arange(midpoint, len(X)), np.arange(midpoint)
        yield np.arange(midpoint), np.arange(midpoint, len(X))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def poisson_data():
    """Synthetic Poisson data with one spline feature."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0, 10, n)
    mu = np.exp(0.5 + 0.1 * x)
    y = rng.poisson(mu).astype(float)
    sw = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame({"x": x})
    return df, y, sw


@pytest.fixture
def base_model():
    """Unfitted SuperGLM with spline on 'x'."""
    return SuperGLM(
        family="poisson",
        penalty=GroupLasso(lambda1=0.0),
        features={"x": Spline(n_knots=5)},
    )


@pytest.fixture
def categorical_data():
    """Synthetic Poisson data with one categorical feature."""
    rng = np.random.default_rng(17)
    n = 360
    band = rng.choice(["A", "B", "C"], n, p=[0.4, 0.35, 0.25])
    effect = {"A": 0.0, "B": 0.25, "C": -0.15}
    sample_weight = rng.uniform(0.5, 2.0, n)
    mu = np.exp(0.4 + np.array([effect[b] for b in band]))
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"band": band})
    return df, y, sample_weight


@pytest.fixture
def categorical_model():
    """Unfitted SuperGLM with one categorical feature."""
    return SuperGLM(
        family="poisson",
        penalty=GroupLasso(lambda1=0.0),
        features={"band": Categorical(base="first")},
    )


# ── Core functionality ───────────────────────────────────────────


class TestCrossValidateBasic:
    def test_smoke(self, poisson_data, base_model):
        """cross_validate runs and returns correct structure."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        assert isinstance(result, CrossValidationResult)
        assert isinstance(result.fold_scores, pd.DataFrame)
        assert len(result.fold_scores) == 3
        assert "deviance" in result.mean_scores
        assert "deviance" in result.std_scores
        assert result.oof_predictions is None
        assert result.estimators is None

    def test_fold_scores_columns(self, poisson_data, base_model):
        """fold_scores DataFrame has all required columns."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        required = {
            "fold",
            "n_train",
            "n_test",
            "fit_time_s",
            "score_time_s",
            "converged",
            "n_iter",
            "effective_df",
            "deviance",
        }
        assert required.issubset(set(result.fold_scores.columns))

    def test_fold_metadata_values(self, poisson_data, base_model):
        """Fold metadata (n_train, n_test, converged) has sensible values."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        fs = result.fold_scores
        assert all(fs["n_train"] > 0)
        assert all(fs["n_test"] > 0)
        assert all(fs["converged"])
        assert all(fs["n_iter"] > 0)
        assert all(fs["fit_time_s"] > 0)
        assert all(np.isfinite(fs["effective_df"]))

    def test_deviance_finite_positive(self, poisson_data, base_model):
        """All deviance scores are finite and positive."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        deviances = result.fold_scores["deviance"]
        assert all(np.isfinite(deviances))
        assert all(deviances > 0)


# ── Splitter variants ────────────────────────────────────────────


class TestSplitters:
    def test_group_kfold(self, poisson_data, base_model):
        """GroupKFold splitter with groups parameter works."""
        df, y, sw = poisson_data
        groups = np.repeat(np.arange(50), 10)
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleGroupKFold(3),
            sample_weight=sw,
            groups=groups,
        )
        assert len(result.fold_scores) == 3
        assert all(np.isfinite(result.fold_scores["deviance"]))

    def test_custom_splitter(self, poisson_data, base_model):
        """Any object with .split() works as a splitter."""
        df, y, sw = poisson_data

        class TwoFold:
            def split(self, X, y=None, groups=None):
                n = len(X)
                mid = n // 2
                yield np.arange(mid), np.arange(mid, n)
                yield np.arange(mid, n), np.arange(mid)

        result = cross_validate(
            base_model,
            df,
            y,
            cv=TwoFold(),
            sample_weight=sw,
        )
        assert len(result.fold_scores) == 2


# ── Data forwarding ──────────────────────────────────────────────


class TestDataForwarding:
    def test_polars_splitter_and_scorer_receive_native_frames(self):
        x = np.linspace(-1.0, 1.0, 60)
        X = pl.DataFrame({"x": x})
        y = 0.5 + 0.3 * x
        splitter = RecordingTwoFold()
        scorer_types: list[type] = []

        def native_scorer(model, X_val, y_val, *, sample_weight=None, offset=None):
            del sample_weight
            scorer_types.append(type(X_val))
            return float(np.mean(np.abs(y_val - model.predict(X_val, offset=offset))))

        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        result = cross_validate(model, X, y, cv=splitter, scoring=native_scorer)

        assert splitter.frame_types == [pl.DataFrame]
        assert scorer_types == [pl.DataFrame, pl.DataFrame]
        assert len(result.fold_scores) == 2

    def test_polars_and_pandas_cross_validation_outputs_are_numerically_equal(self):
        rng = np.random.default_rng(20260725)
        x = np.linspace(-1.0, 1.0, 90)
        z = rng.normal(size=len(x))
        y = 0.5 + 0.3 * x - 0.15 * z
        weights = rng.uniform(0.5, 1.5, len(x))
        pandas_X = pd.DataFrame({"x": x, "z": z})
        polars_X = pl.DataFrame({"x": x, "z": z})

        def evaluated(X):
            model = SuperGLM(
                family="gaussian",
                selection_penalty=0.0,
                features={"x": Numeric(), "z": Numeric()},
            )
            return cross_validate(
                model,
                X,
                y,
                cv=SimpleKFold(3),
                sample_weight=weights,
                scoring=("deviance", "nll"),
                return_estimators=True,
                return_oof=True,
            )

        pandas_result = evaluated(pandas_X)
        polars_result = evaluated(polars_X)

        columns = [
            column
            for column in pandas_result.fold_scores.columns
            if column not in {"fit_time_s", "score_time_s"}
        ]
        pd.testing.assert_frame_equal(
            polars_result.fold_scores[columns],
            pandas_result.fold_scores[columns],
            check_exact=True,
        )
        assert polars_result.mean_scores == pandas_result.mean_scores
        assert polars_result.pooled_scores == pandas_result.pooled_scores
        assert polars_result.std_scores == pandas_result.std_scores
        np.testing.assert_allclose(
            polars_result.oof_predictions,
            pandas_result.oof_predictions,
            rtol=0.0,
            atol=0.0,
        )
        for polars_indices, pandas_indices in zip(
            polars_result.fold_indices,
            pandas_result.fold_indices,
            strict=True,
        ):
            np.testing.assert_array_equal(polars_indices[0], pandas_indices[0])
            np.testing.assert_array_equal(polars_indices[1], pandas_indices[1])
        assert all(
            isinstance(estimator._fit_X_ref, pl.DataFrame) for estimator in polars_result.estimators
        )

    def test_sample_weight_affects_score(self, poisson_data, base_model):
        """Weighted deviance differs from unweighted."""
        df, y, sw = poisson_data
        r_weighted = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
            sample_weight=sw,
        )
        r_unweighted = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
        )
        # Scores should differ (not necessarily by much, but not identical)
        assert r_weighted.mean_scores["deviance"] != r_unweighted.mean_scores["deviance"]

    def test_offset_forwarding(self, poisson_data, base_model):
        """Offset is passed to fit and predict."""
        df, y, sw = poisson_data
        offset = np.log(sw)
        r_with = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
            offset=offset,
        )
        r_without = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
        )
        assert r_with.mean_scores["deviance"] != r_without.mean_scores["deviance"]

    def test_pooled_scores_forward_offset(self, poisson_data, base_model):
        """Pooled deviance/NLL use the validation offset just like fold scoring."""
        df, y, sw = poisson_data
        offset = np.log(sw)
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
            sample_weight=sw,
            offset=offset,
            scoring=("deviance", "nll"),
            return_estimators=True,
        )

        dev_num = 0.0
        dev_den = 0.0
        nll_num = 0.0
        nll_den = 0.0
        assert result.estimators is not None
        assert result.fold_indices is not None
        for est, (_train_idx, test_idx) in zip(result.estimators, result.fold_indices, strict=True):
            X_test = df.iloc[test_idx]
            y_test = y[test_idx]
            sw_test = sw[test_idx]
            off_test = offset[test_idx]
            mu = est.predict(X_test, offset=off_test)
            dev = est._distribution.deviance_unit(y_test, mu)
            ll = est._distribution.log_likelihood(y_test, mu, sw_test, phi=est.result.phi)
            dev_num += float(np.sum(sw_test * dev))
            dev_den += float(np.sum(sw_test))
            nll_num += float(-ll)
            nll_den += float(np.sum(sw_test))

        assert result.pooled_scores["deviance"] == pytest.approx(dev_num / dev_den)
        assert result.pooled_scores["nll"] == pytest.approx(nll_num / nll_den)


# ── Fit modes ─────────────────────────────────────────────────────


class TestFitModes:
    def test_fit_reml(self, poisson_data):
        """fit_mode='fit_reml' calls fit_reml on each fold."""
        df, y, sw = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=5)},
        )
        result = cross_validate(
            model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            fit_mode="fit_reml",
        )
        assert len(result.fold_scores) == 3
        assert all(result.fold_scores["converged"])

    def test_invalid_fit_mode(self, poisson_data, base_model):
        """Invalid fit_mode raises ValueError."""
        df, y, sw = poisson_data
        with pytest.raises(ValueError, match="fit_mode"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                fit_mode="fit_path",
            )


# ── Scoring ───────────────────────────────────────────────────────


class TestScoring:
    def test_multiple_string_scorers(self, poisson_data, base_model):
        """Multiple string scorers produce multiple columns."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            scoring=("deviance", "nll"),
        )
        assert "deviance" in result.fold_scores.columns
        assert "nll" in result.fold_scores.columns
        assert "deviance" in result.mean_scores
        assert "nll" in result.mean_scores

    def test_callable_scorer_scalar(self, poisson_data, base_model):
        """Callable scorer returning a float works."""
        df, y, sw = poisson_data

        def mae(model, X, y, *, sample_weight=None, offset=None):
            mu = model.predict(X, offset=offset)
            return float(np.mean(np.abs(y - mu)))

        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            scoring=mae,
        )
        assert "mae" in result.mean_scores
        assert result.mean_scores["mae"] > 0

    def test_callable_scorer_dict(self, poisson_data, base_model):
        """Callable scorer returning a dict produces multiple columns."""
        df, y, sw = poisson_data

        def multi_score(model, X, y, *, sample_weight=None, offset=None):
            mu = model.predict(X, offset=offset)
            return {
                "mae": float(np.mean(np.abs(y - mu))),
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
            }

        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            scoring=multi_score,
        )
        assert "mae" in result.mean_scores
        assert "rmse" in result.mean_scores

    def test_mixed_scorers(self, poisson_data, base_model):
        """Mix of string and callable scorers works."""
        df, y, sw = poisson_data

        def mae(model, X, y, *, sample_weight=None, offset=None):
            return float(np.mean(np.abs(y - model.predict(X, offset=offset))))

        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            scoring=("deviance", mae),
        )
        assert "deviance" in result.mean_scores
        assert "mae" in result.mean_scores

    def test_unknown_scorer_raises(self, poisson_data, base_model):
        """Unknown string scorer raises ValueError."""
        df, y, sw = poisson_data
        with pytest.raises(ValueError, match="Unknown scorer"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                scoring="bad_metric",
            )

    def test_single_string_scorer(self, poisson_data, base_model):
        """Single string (not tuple) works."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            scoring="deviance",
        )
        assert "deviance" in result.mean_scores

    def test_pooled_scores_match_ratio_of_sums_for_deviance_and_nll(self, poisson_data, base_model):
        """Pooled ratio metrics aggregate numerator and denominator before dividing."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3, shuffle=True, random_state=0),
            sample_weight=sw,
            scoring=("deviance", "nll"),
            return_estimators=True,
        )

        total_dev = 0.0
        total_nll = 0.0
        total_weight = 0.0
        for est, (_, test_idx) in zip(result.estimators, result.fold_indices, strict=True):
            X_test = df.iloc[test_idx]
            y_test = y[test_idx]
            sw_test = sw[test_idx]
            mu = est.predict(X_test)
            total_dev += float(np.sum(sw_test * est._distribution.deviance_unit(y_test, mu)))
            total_nll += float(
                -est._distribution.log_likelihood(y_test, mu, sw_test, phi=est.result.phi)
            )
            total_weight += float(np.sum(sw_test))

        assert result.pooled_scores["deviance"] == pytest.approx(total_dev / total_weight)
        assert result.pooled_scores["nll"] == pytest.approx(total_nll / total_weight)

    def test_pooled_scores_can_differ_from_equal_fold_mean(self, poisson_data, base_model):
        """Mean-of-fold ratios and ratio-of-sums are different quantities."""
        df, y, _ = poisson_data
        sw = np.ones_like(y)
        sw[:50] = 100.0

        class UnevenTwoFold:
            def split(self, X, y=None, groups=None):
                yield np.arange(50, len(X)), np.arange(50)
                yield np.arange(50), np.arange(50, len(X))

        result = cross_validate(
            base_model,
            df,
            y,
            cv=UnevenTwoFold(),
            sample_weight=sw,
            scoring=("deviance", "nll"),
        )

        assert result.mean_scores["deviance"] != pytest.approx(result.pooled_scores["deviance"])
        assert result.mean_scores["nll"] != pytest.approx(result.pooled_scores["nll"])


class TestWeightContractScoring:
    class _FixedPredictionModel:
        def __init__(self, distribution, mu, phi):
            self._distribution = distribution
            self._mu = np.asarray(mu, dtype=np.float64)
            self.result = SimpleNamespace(phi=phi)

        def predict(self, X, offset=None):
            del X, offset
            return self._mu

    @pytest.mark.parametrize(
        ("distribution", "y", "mu", "phi"),
        [
            (
                Gaussian(),
                np.array([-0.4, 0.2, 1.8, 2.7]),
                np.array([-0.1, 0.4, 1.5, 2.2]),
                1.3,
            ),
            (
                Poisson(),
                np.array([0.0, 1.0, 4.0, 2.0]),
                np.array([0.4, 1.2, 3.3, 2.4]),
                1.0,
            ),
        ],
    )
    def test_frequency_weighted_scores_match_literal_rows(self, distribution, y, mu, phi):
        weights = np.array([3, 1, 4, 2], dtype=np.float64)
        repeated = np.repeat(np.arange(len(y)), weights.astype(np.intp))
        weighted_model = self._FixedPredictionModel(distribution, mu, phi)
        repeated_model = self._FixedPredictionModel(distribution, mu[repeated], phi)

        weighted_deviance = _score_deviance(
            weighted_model,
            None,
            y,
            sample_weight=weights,
        )
        repeated_deviance = _score_deviance(
            repeated_model,
            None,
            y[repeated],
        )
        weighted_nll = _score_nll(
            weighted_model,
            None,
            y,
            sample_weight=weights,
        )
        repeated_nll = _score_nll(
            repeated_model,
            None,
            y[repeated],
        )

        assert weighted_deviance == pytest.approx(repeated_deviance)
        assert weighted_nll == pytest.approx(repeated_nll)

    def test_tweedie_scores_use_physical_observation_count(self):
        distribution = Tweedie(p=1.5)
        y = np.array([0.0, 0.7, 1.8, 3.2])
        mu = np.array([0.4, 0.9, 1.6, 2.7])
        weights = np.array([0.25, 0.8, 1.7, 3.4])
        model = self._FixedPredictionModel(distribution, mu, phi=0.9)

        deviance_total = float(np.sum(weights * distribution.deviance_unit(y, mu)))
        nll_total = -distribution.log_likelihood(y, mu, weights, phi=0.9)
        deviance_parts = _pooled_deviance_parts(
            model,
            None,
            y,
            sample_weight=weights,
        )
        nll_parts = _pooled_nll_parts(
            model,
            None,
            y,
            sample_weight=weights,
        )

        assert _score_deviance(model, None, y, sample_weight=weights) == pytest.approx(
            deviance_total / len(y)
        )
        assert _score_nll(model, None, y, sample_weight=weights) == pytest.approx(
            nll_total / len(y)
        )
        assert deviance_parts == pytest.approx((deviance_total, len(y)))
        assert nll_parts == pytest.approx((nll_total, len(y)))


# ── Return options ────────────────────────────────────────────────


class TestReturnOptions:
    def test_polars_plot_terms_by_fold_keeps_fold_support_native(self, monkeypatch):
        x = np.linspace(0.0, 1.0, 60)
        X = pl.DataFrame({"x": x})
        y = 0.5 + np.sin(3.0 * x)
        weights = np.linspace(0.5, 1.5, len(x))
        result = cross_validate(
            SuperGLM(
                family="gaussian",
                selection_penalty=0.0,
                features={"x": Spline(n_knots=5)},
            ),
            X,
            y,
            cv=SimpleKFold(3),
            sample_weight=weights,
            return_estimators=True,
        )
        captured = {}

        def fake_plot_term_comparison(**kwargs):
            captured.update(kwargs)
            return "figure"

        monkeypatch.setattr(
            "superglm.plotting.comparison.plot_term_comparison",
            fake_plot_term_comparison,
        )

        figure = result.plot_terms_by_fold(X, sample_weight=weights, terms=["x"])

        assert figure == "figure"
        assert captured["X"] is X
        assert all(
            isinstance(payload["X"], pl.DataFrame)
            for payload in captured["support_by_label"].values()
        )
        assert sorted(len(payload["X"]) for payload in captured["support_by_label"].values()) == [
            40,
            40,
            40,
        ]

    def test_return_oof(self, poisson_data, base_model):
        """return_oof=True fills correct indices."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_oof=True,
        )
        assert result.oof_predictions is not None
        assert result.oof_predictions.shape == (len(y),)
        # All observations should have been in exactly one test fold
        assert not np.any(np.isnan(result.oof_predictions))
        assert np.all(result.oof_predictions > 0)  # Poisson predictions > 0

    def test_return_estimators(self, poisson_data, base_model):
        """return_estimators=True returns fitted models."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            return_estimators=True,
        )
        assert result.estimators is not None
        assert len(result.estimators) == 3
        for est in result.estimators:
            assert isinstance(est, SuperGLM)
            # Each should be fitted
            assert est._result is not None

    def test_return_estimators_stores_fold_indices(self, poisson_data, base_model):
        """Fold train/test indices are retained for fold-aware tooling."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )
        assert result.fold_indices is not None
        assert len(result.fold_indices) == 3
        train_idx, test_idx = result.fold_indices[0]
        assert len(train_idx) > 0
        assert len(test_idx) > 0

    def test_plot_terms_by_fold_uses_labeled_fold_estimators(self, poisson_data, base_model):
        """The fold wrapper labels traces by fold index."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )

        fig = result.plot_terms_by_fold(
            df,
            sample_weight=sw,
            terms=["x"],
            engine="plotly",
            n_points=31,
        )
        assert isinstance(fig, go.Figure)
        names = {t.name for t in fig.data if t.name and t.name.startswith("fold_")}
        assert names == {"fold_0", "fold_1", "fold_2"}

    def test_plot_terms_by_fold_shows_per_fold_continuous_support(self, poisson_data, base_model):
        """Continuous fold plots include one support trace per fold in the lower panel."""
        pytest.importorskip("plotly")

        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )

        fig = result.plot_terms_by_fold(
            df, sample_weight=sw, terms=["x"], engine="plotly", n_points=31
        )
        support_traces = [
            t
            for t in fig.data
            if getattr(t, "xaxis", None) == "x2" and t.name == "Exposure density"
        ]
        assert len(support_traces) == 3

    def test_plot_terms_by_fold_shows_grouped_fold_exposure_bars(
        self, categorical_data, categorical_model
    ):
        """Categorical fold plots include grouped exposure bars by fold."""
        pytest.importorskip("plotly")

        df, y, sw = categorical_data
        result = cross_validate(
            categorical_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )

        fig = result.plot_terms_by_fold(
            df,
            sample_weight=sw,
            terms=["band"],
            engine="plotly",
            n_points=31,
        )
        bar_traces = [
            t
            for t in fig.data
            if getattr(t, "type", None) == "bar" and getattr(t, "xaxis", None) == "x2"
        ]
        assert len(bar_traces) == 3
        assert fig.layout.barmode == "group"

    def test_cross_validate_always_returns_curve_similarity(self, poisson_data, base_model):
        """Curve similarity is computed automatically when fold estimators exist."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )

        assert result.curve_similarity is not None
        assert "x" in result.curve_similarity
        assert set(result.curve_similarity["x"]["pairwise"]) == {"response", "link"}

    def test_curve_similarity_contains_pairwise_metric_frames(self, poisson_data, base_model):
        """Stored similarity includes pairwise metric matrices on response scale."""
        df, y, sw = poisson_data
        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
            return_estimators=True,
        )

        response = result.curve_similarity["x"]["pairwise"]["response"]
        assert set(response) == {"rmse", "max_abs_diff", "correlation"}
        assert response["rmse"].shape == (3, 3)
        np.testing.assert_allclose(np.diag(response["rmse"]), 0.0)


# ── Error handling ────────────────────────────────────────────────


class TestErrorHandling:
    def test_polars_fold_failure_honours_error_score_without_mutating_model(self):
        X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = np.linspace(0.0, 1.0, 20)

        class EmptyTrainingFold:
            @staticmethod
            def split(X, y=None, groups=None):
                del y, groups
                yield np.array([], dtype=int), np.arange(len(X))

        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        state_before = model.__dict__.copy()

        result = cross_validate(
            model,
            X,
            y,
            cv=EmptyTrainingFold(),
            error_score=np.nan,
        )

        assert np.isnan(result.fold_scores.loc[0, "deviance"])
        assert model.__dict__.keys() == state_before.keys()
        assert all(model.__dict__[name] is value for name, value in state_before.items())

    def test_error_score_nan(self, poisson_data):
        """Fold failure fills scores with NaN when error_score=np.nan."""
        df, y, sw = poisson_data

        class BadSplitter:
            def split(self, X, y=None, groups=None):
                # First fold: fine
                yield np.arange(250), np.arange(250, 500)
                # Second fold: empty train set → will fail
                yield np.array([], dtype=int), np.arange(500)

        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Spline(n_knots=5)},
        )
        result = cross_validate(
            model,
            df,
            y,
            cv=BadSplitter(),
            sample_weight=sw,
            error_score=np.nan,
        )
        assert len(result.fold_scores) == 2
        # First fold should be fine
        assert np.isfinite(result.fold_scores.iloc[0]["deviance"])
        # Second fold should be NaN
        assert np.isnan(result.fold_scores.iloc[1]["deviance"])

    def test_error_score_raise(self, poisson_data):
        """error_score='raise' propagates exceptions."""
        df, y, sw = poisson_data

        class BadSplitter:
            def split(self, X, y=None, groups=None):
                yield np.array([], dtype=int), np.arange(len(X))

        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Spline(n_knots=5)},
        )
        with pytest.raises(Exception):
            cross_validate(
                model,
                df,
                y,
                cv=BadSplitter(),
                sample_weight=sw,
                error_score="raise",
            )


# ── Input safety ──────────────────────────────────────────────────


class TestInputSafety:
    def test_input_model_not_mutated(self, poisson_data, base_model):
        """The input model is not modified by cross_validate."""
        df, y, sw = poisson_data
        assert base_model._result is None
        cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        assert base_model._result is None  # Still unfitted

    def test_fitted_model_not_mutated(self, poisson_data):
        """A pre-fitted model keeps its original coefficients."""
        df, y, sw = poisson_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Spline(n_knots=5)},
        )
        model.fit(df, y, sample_weight=sw)
        orig_beta = model._result.beta.copy()

        cross_validate(
            model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        np.testing.assert_array_equal(model._result.beta, orig_beta)


# ── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_weight_doc_scopes_literal_replication_to_fixed_cv_partition(self):
        doc = inspect.getdoc(cross_validate)

        assert doc is not None
        assert "splitters operate on the physical compact rows" in doc
        assert "within a fixed train/validation partition and fixed feature geometry" in doc

    def test_no_split_method(self, poisson_data, base_model):
        """Raise if cv has no .split() method."""
        df, y, _ = poisson_data
        with pytest.raises(TypeError, match="split"):
            cross_validate(base_model, df, y, cv=42)

    def test_sample_weight_length_mismatch(self, poisson_data, base_model):
        """Raise if sample_weight length doesn't match y."""
        df, y, _ = poisson_data
        with pytest.raises(ValueError, match="sample_weight"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                sample_weight=np.ones(10),
            )

    def test_offset_length_mismatch(self, poisson_data, base_model):
        """Raise if offset length doesn't match y."""
        df, y, _ = poisson_data
        with pytest.raises(ValueError, match="offset"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                offset=np.ones(10),
            )

    def test_groups_length_mismatch(self, poisson_data, base_model):
        """Raise if groups length doesn't match y."""
        df, y, _ = poisson_data
        with pytest.raises(ValueError, match="groups"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                groups=np.ones(10),
            )


# ── Auto-detect cloning ──────────────────────────────────────────


class TestCloneContract:
    @staticmethod
    def _configured_model(*, features=None, splines=None):
        with pytest.warns(UserWarning, match="convergence='coefficients' is experimental"):
            return SuperGLM(
                family="gaussian",
                link="identity",
                penalty=GroupElasticNet(
                    lambda1=0.37,
                    alpha=0.25,
                    features=["x"],
                ),
                spline_penalty=2.75,
                features=features,
                splines=splines,
                n_knots=[4, 5],
                degree=2,
                categorical_base="first",
                interactions=[("x", "z")],
                active_set=True,
                direct_solve="qr",
                discrete=True,
                n_bins={"x": 17, "z": 19},
                tol=1e-3,
                max_iter=7,
                convergence="coefficients",
                retain_fit_state=False,
            )

    @staticmethod
    def _assert_constructor_parity(source, clone):
        assert clone._family_config == source._family_config == "gaussian"
        assert clone._link_config == source._link_config == "identity"
        assert isinstance(clone._penalty_config, GroupElasticNet)
        assert clone.selection_penalty == pytest.approx(source.selection_penalty)
        assert clone._penalty_config.alpha == pytest.approx(source._penalty_config.alpha)
        assert clone._penalty_config.features == source._penalty_config.features == frozenset({"x"})
        assert clone.lambda2 == pytest.approx(source.lambda2)
        assert clone._splines == source._splines
        assert clone._n_knots == source._n_knots == [4, 5]
        assert clone._degree == source._degree == 2
        assert clone._categorical_base == source._categorical_base == "first"
        assert clone._config.interactions == source._config.interactions == (("x", "z"),)
        assert clone._active_set is source._active_set is True
        assert clone._direct_solve == source._direct_solve == "qr"
        assert clone._discrete is source._discrete is True
        assert clone._n_bins == source._n_bins == {"x": 17, "z": 19}
        assert clone._tol == pytest.approx(source._tol)
        assert clone._max_iter == source._max_iter == 7
        assert clone._convergence == source._convergence == "coefficients"
        assert clone._retain_fit_state is source._retain_fit_state is False

    def test_constructor_config_contract_covers_every_superglm_parameter(self):
        model = self._configured_model(splines=["x", "z"])

        constructor_kwargs = model._config.constructor_kwargs()

        assert set(constructor_kwargs) == set(inspect.signature(SuperGLM).parameters)

    def test_constructor_kwargs_prefer_resolved_features_to_splines_shorthand(self):
        model = self._configured_model(features={"x": Numeric(), "z": Numeric()})
        config = model._config.with_value(splines=("x", "z"))

        constructor_kwargs = config.constructor_kwargs()

        assert constructor_kwargs["features"] is not None
        assert constructor_kwargs["splines"] is None

    def test_clone_model_preserves_every_constructor_setting(self):
        model = self._configured_model(splines=["x", "z"])

        cloned = _clone_model(model)

        self._assert_constructor_parity(model, cloned)
        assert cloned._specs == {}
        assert cloned._feature_order == []

    def test_clone_unfitted_owns_mutable_configuration_and_has_no_fit_state(self):
        model = self._configured_model(
            features={"x": Spline(n_knots=4), "z": Numeric()},
        )

        cloned = model.clone_unfitted()

        assert cloned._config is not model._config
        assert cloned._config.penalty is not model._config.penalty
        assert cloned._config.feature_templates[0][1] is not model._config.feature_templates[0][1]
        assert cloned._penalty_config is not model._penalty_config
        assert cloned._specs["x"] is not model._specs["x"]
        assert cloned._n_bins is not model._n_bins
        assert cloned._result is None
        assert cloned._solver_result is None
        assert cloned._fit_state is None
        assert cloned._dm is None
        assert cloned._fit_revision == 0

        cloned._penalty_config.lambda1 = 9.0
        cloned._config.penalty.lambda1 = 8.0
        cloned._specs["x"].n_knots = 9
        cloned._n_bins["x"] = 99

        assert model._penalty_config.lambda1 == pytest.approx(0.37)
        assert model._config.penalty.lambda1 == pytest.approx(0.37)
        assert model._specs["x"].n_knots == 4
        assert model._n_bins["x"] == 17

    def test_clone_of_fitted_model_does_not_copy_learned_state(self, poisson_data, base_model):
        df, y, sample_weight = poisson_data
        base_model.fit(df, y, sample_weight=sample_weight)

        cloned = _clone_model(base_model)

        assert cloned._result is None
        assert cloned._solver_result is None
        assert cloned._fit_state is None
        assert cloned._dm is None
        assert cloned._groups == []
        assert cloned._fit_revision == 0

    def test_fit_workspace_does_not_reinvoke_subclass_constructor(self):
        constructor_tags = []

        class TaggedSuperGLM(SuperGLM):
            def __init__(self, tag, **kwargs):
                constructor_tags.append(tag)
                self.tag = tag
                super().__init__(**kwargs)

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = TaggedSuperGLM(
            "audit",
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(X, y)

        assert constructor_tags == ["audit"]
        assert model.tag == "audit"
        assert model._result is not None

    def test_fit_workspace_preserves_subclass_state_aliases(self):
        class StatefulSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                shared = []
                self.primary_events = shared
                self.aliased_events = shared

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = StatefulSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(X, y)

        assert model.primary_events is model.aliased_events

    def test_fit_workspace_preserves_subclass_aliases_to_base_configuration(self):
        class StatefulSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.config_alias = self._config
                self.penalty_alias = self._penalty_config

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = StatefulSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(X, y)

        assert model.config_alias is model._config
        assert model.penalty_alias is model._penalty_config

        model.fit(X, y)

        assert model.config_alias is model._config
        assert model.penalty_alias is model._penalty_config

    def test_fit_workspace_rebinds_subclass_self_references_on_publication(self):
        class StatefulSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.events = []
                self.owner = self
                self.callback = self.record_event
                self.fail_from_workspace = False

            def record_event(self, event):
                self.events.append(event)

            def _solver_pirls_result(self):
                self.callback("solver_result")
                if self.fail_from_workspace:
                    raise RuntimeError("injected workspace failure")
                return super()._solver_pirls_result()

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = StatefulSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(X, y)

        assert model.owner is model
        assert model.callback.__self__ is model
        assert model.events

        first_events = tuple(model.events)
        installed_events = model.events
        first_revision = model._fit_revision
        model.fail_from_workspace = True

        with pytest.raises(RuntimeError, match="injected workspace failure"):
            model.fit(X, y)

        assert model.owner is model
        assert model.callback.__self__ is model
        assert model.events is installed_events
        assert tuple(model.events) == first_events
        assert model._fit_revision == first_revision

        model.fail_from_workspace = False
        model.fit(X, y)

        assert model.owner is model
        assert model.callback.__self__ is model
        assert len(model.events) > len(first_events)
        assert model._fit_revision == first_revision + 1

    def test_mutable_subclass_state_is_transactional_across_refits(self):
        class StatefulSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.events = []
                self.fail_from_workspace = False

            def _solver_pirls_result(self):
                self.events.append("solver_result")
                if self.fail_from_workspace:
                    raise RuntimeError("injected workspace failure")
                return super()._solver_pirls_result()

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = StatefulSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(X, y)
        first_events = tuple(model.events)
        first_revision = model._fit_revision
        first_result = model._result
        installed_events = model.events
        installed_config = model._config

        model.fail_from_workspace = True
        with pytest.raises(RuntimeError, match="injected workspace failure"):
            model.fit(X, y)

        assert model.events is installed_events
        assert tuple(model.events) == first_events
        assert model._fit_revision == first_revision
        assert model._result is first_result
        assert model._config is installed_config

        model.fail_from_workspace = False
        model.fit(X, y)

        assert model._fit_revision == first_revision + 1
        assert len(model.events) > len(first_events)

    def test_failed_first_fit_does_not_mutate_subclass_state(self):
        class FailingSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.events = []

            def _solver_pirls_result(self):
                self.events.append("solver_result")
                raise RuntimeError("injected workspace failure")

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = FailingSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        original_dict = model.__dict__
        original_events = model.events

        with pytest.raises(RuntimeError, match="injected workspace failure"):
            model.fit(X, y)

        assert model.__dict__ is original_dict
        assert model.events is original_events
        assert model.events == []
        assert model._fit_revision == 0
        assert model._result is None

    def test_failed_subclass_state_publication_keeps_model_unfitted(self):
        class PublishOnlyFailure:
            def __init__(self, *, fail=False):
                self.fail = fail

            def __deepcopy__(self, memo):
                if self.fail:
                    raise ValueError("cannot publish extension")
                copied = type(self)(fail=True)
                memo[id(self)] = copied
                return copied

        class StatefulSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.extension = PublishOnlyFailure()

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model = StatefulSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        original_dict = model.__dict__
        original_extension = model.extension

        with pytest.raises(TypeError, match="subclass fit state 'extension'.*deepcopy"):
            model.fit(X, y)

        assert model.__dict__ is original_dict
        assert model.extension is original_extension
        assert model.extension.fail is False
        assert model._fit_revision == 0
        assert model._result is None

    def test_clone_unfitted_reconstructs_base_compatible_subclass(self):
        constructor_calls = []

        class CompatibleSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                constructor_calls.append(tuple(sorted(kwargs)))
                super().__init__(**kwargs)
                self.subclass_initialized = True
                self.events = []

        model = CompatibleSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = 1.0 + 0.25 * X["x"].to_numpy()
        model.fit(X, y)
        model.events.append("source-only")

        cloned = model.clone_unfitted()

        assert isinstance(cloned, CompatibleSuperGLM)
        assert cloned.subclass_initialized is True
        assert cloned.events == []
        assert cloned.events is not model.events
        assert len(constructor_calls) == 2
        assert cloned._result is None
        assert cloned._fit_revision == 0

    def test_clone_unfitted_reconstructs_transparent_variadic_subclass(self):
        class WrapperSuperGLM(SuperGLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        model = WrapperSuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        cloned = model.clone_unfitted()

        assert isinstance(cloned, WrapperSuperGLM)
        assert cloned._result is None
        assert cloned._fit_revision == 0

    def test_subclass_clone_does_not_repeat_splines_deprecation_warning(self):
        class WrapperSuperGLM(SuperGLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        with pytest.warns(FutureWarning, match=r"splines=.*features=.*Spline") as caught:
            model = WrapperSuperGLM(splines=["x"])
        assert len(caught) == 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cloned = model.clone_unfitted()

        assert isinstance(cloned, WrapperSuperGLM)

    def test_clone_unfitted_requires_override_for_required_subclass_configuration(self):
        class TaggedSuperGLM(SuperGLM):
            def __init__(self, tag, **kwargs):
                self.tag = tag
                super().__init__(**kwargs)

        model = TaggedSuperGLM(
            "audit",
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        with pytest.raises(TypeError, match="override clone_unfitted"):
            model.clone_unfitted()

    def test_cross_validate_estimators_preserve_constructor_configuration(self):
        X = pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, 60),
                "z": np.linspace(1.0, -1.0, 60),
            }
        )
        y = 1.5 + 0.4 * X["x"].to_numpy() - 0.2 * X["z"].to_numpy()
        model = self._configured_model(features={"x": Numeric(), "z": Numeric()})

        result = cross_validate(
            model,
            X,
            y,
            cv=SimpleKFold(2),
            return_estimators=True,
            error_score="raise",
        )

        assert result.estimators is not None
        assert len(result.estimators) == 2
        for estimator in result.estimators:
            self._assert_constructor_parity(model, estimator)
            assert estimator._result is not None
            assert estimator._config is not model._config
            assert estimator._config.penalty is not model._config.penalty
        assert result.estimators[0]._config is not result.estimators[1]._config
        assert result.estimators[0]._config.penalty is not result.estimators[1]._config.penalty


class TestAutoDetectClone:
    def test_autodetect_cross_validation_and_pickle_do_not_repeat_warning(
        self,
        poisson_data,
    ):
        df, y, sw = poisson_data
        with pytest.warns(FutureWarning, match=r"splines=.*features=.*Spline") as caught:
            model = SuperGLM(
                family="poisson",
                penalty=GroupLasso(lambda1=0.0),
                splines=["x"],
            )
        assert len(caught) == 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            restored = pickle.loads(pickle.dumps(model))
            result = cross_validate(
                restored,
                df,
                y,
                cv=SimpleKFold(2),
                sample_weight=sw,
            )

        assert all(np.isfinite(result.fold_scores["deviance"]))

    def test_unfitted_autodetect_model(self, poisson_data):
        """Unfitted auto-detect (splines=) model clones correctly."""
        df, y, sw = poisson_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.0),
            splines=["x"],
        )
        result = cross_validate(
            model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        assert all(np.isfinite(result.fold_scores["deviance"]))
        assert all(result.fold_scores["converged"])

    def test_fitted_autodetect_model(self, poisson_data):
        """Fitted auto-detect model clones without ValueError."""
        df, y, sw = poisson_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.0),
            splines=["x"],
        )
        model.fit(df, y, sample_weight=sw)

        result = cross_validate(
            model,
            df,
            y,
            cv=SimpleKFold(3),
            sample_weight=sw,
        )
        assert all(np.isfinite(result.fold_scores["deviance"]))
        assert all(result.fold_scores["converged"])


# ── Scorer edge cases ────────────────────────────────────────────


class TestGiniScorer:
    class _PredictionOnlyModel:
        def __init__(self, predictions):
            self.predictions = np.asarray(predictions, dtype=float)

        def predict(self, X, offset=None):
            return self.predictions

    def test_gini_perfect_ranking_is_normalized_to_one(self):
        model = self._PredictionOnlyModel([1.0, 0.0])

        score = _score_gini(model, None, np.array([1.0, 0.0]))

        assert score == pytest.approx(1.0)

    def test_gini_constant_predictions_have_no_ranking_signal(self):
        model = self._PredictionOnlyModel([1.0, 1.0])

        score = _score_gini(
            model,
            None,
            np.array([1.0, 0.0]),
            sample_weight=np.array([2.0, 1.0]),
        )

        assert score == pytest.approx(0.0)

    def test_gini_tied_predictions_are_permutation_invariant(self):
        y = np.array([10.0, 1.0, 8.0, 2.0, 6.0, 3.0])
        predictions = np.array([0.2, 0.2, 0.5, 0.5, 0.9, 0.9])
        sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 3.0])
        permutation = np.array([1, 0, 3, 2, 5, 4])

        score = _score_gini(
            self._PredictionOnlyModel(predictions),
            None,
            y,
            sample_weight=sample_weight,
        )
        permuted_score = _score_gini(
            self._PredictionOnlyModel(predictions[permutation]),
            None,
            y[permutation],
            sample_weight=sample_weight[permutation],
        )

        assert score == pytest.approx(permuted_score)

    def test_gini_perfect_weighted_frequency_ranking_is_normalized_to_one(self):
        y = np.array([2.0, 0.0, 1.0, 0.0])
        sample_weight = np.array([3.0, 0.5, 1.0, 4.0])
        model = self._PredictionOnlyModel(y)

        score = _score_gini(model, None, y, sample_weight=sample_weight)

        assert score == pytest.approx(1.0)

    def test_gini_weighted_constant_positive_target_is_zero(self):
        y = np.ones(3)
        sample_weight = np.array([0.1, 0.2, 10.1])
        model = self._PredictionOnlyModel(y)

        score = _score_gini(model, None, y, sample_weight=sample_weight)

        assert score == 0.0

    def test_gini_near_constant_target_is_numerically_stable(self):
        y = np.array([1.0, 1.0, np.nextafter(1.0, 2.0)])
        sample_weight = np.array([0.1, 0.2, 10.1])

        constant = _score_gini(
            self._PredictionOnlyModel(np.ones(3)),
            None,
            y,
            sample_weight=sample_weight,
        )
        perfect = _score_gini(
            self._PredictionOnlyModel(y),
            None,
            y,
            sample_weight=sample_weight,
        )
        reverse = _score_gini(
            self._PredictionOnlyModel(-y),
            None,
            y,
            sample_weight=sample_weight,
        )

        assert constant == 0.0
        assert perfect == pytest.approx(1.0)
        assert reverse == pytest.approx(-1.0)

    def test_gini_all_zero_response(self, poisson_data, base_model):
        """Gini scorer returns 0.0 when all y=0 (no division by zero)."""
        df, y, _ = poisson_data
        y_zero = np.zeros_like(y)
        result = cross_validate(
            base_model,
            df,
            y_zero,
            cv=SimpleKFold(3),
            scoring="gini",
        )
        assert all(np.isfinite(result.fold_scores["gini"]))


class TestScorerEdgeCases:
    def test_dict_scorer_reserved_key_raises(self, poisson_data, base_model):
        """Dict scorer returning a reserved column name raises ValueError."""
        df, y, _ = poisson_data

        def bad_scorer(model, X, y, *, sample_weight=None, offset=None):
            return {"fit_time_s": 999.0}

        with pytest.raises(ValueError, match="reserved"):
            cross_validate(
                base_model,
                df,
                y,
                cv=SimpleKFold(3),
                scoring=bad_scorer,
                error_score="raise",
            )


# ── Full-frame level binding ─────────────────────────────────────


class TestFullFrameLevelBinding:
    """One level universe per column, resolved before the split (spec §3.5)."""

    @staticmethod
    def _rare_level_frame(n=120, seed=0):
        rng = np.random.default_rng(seed)
        g = rng.choice(["a", "b"], size=n).astype(object)
        g[0] = "rare"  # exactly one row: pigeonhole-guaranteed to miss a fold
        X = pd.DataFrame({"g": g, "x": rng.normal(size=n)})
        y = rng.poisson(1.2, size=n).astype(float)
        return X, y

    def test_rare_level_no_longer_kills_folds(self):
        X, y = self._rare_level_frame()
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # expected pin warnings
            res = cross_validate(
                model,
                X,
                y,
                cv=SimpleKFold(5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert len(res.fold_scores) == 5
        assert all(np.isfinite(res.fold_scores["deviance"]))

    def test_folds_share_universe_and_base(self):
        X, y = self._rare_level_frame()
        w = np.ones(len(y))
        model = SuperGLM(family="poisson", features={"g": Categorical()})  # most_exposed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model,
                X,
                y,
                sample_weight=w,
                cv=SimpleKFold(4, shuffle=True, random_state=1),
                return_estimators=True,
                error_score="raise",
            )
        specs = [est._specs["g"] for est in res.estimators]
        universes = {tuple(s._levels) for s in specs}
        bases = {s._base_level for s in specs}
        assert len(universes) == 1 and len(bases) == 1
        assert set(next(iter(universes))) == {"a", "b", "rare"}

    def test_auto_detect_path_binds_too(self):
        X, y = self._rare_level_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = SuperGLM(family="poisson", splines=[])  # features=None auto-detect
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model,
                X,
                y,
                cv=SimpleKFold(5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert all(np.isfinite(res.fold_scores["deviance"]))

    def test_user_model_not_mutated(self):
        X, y = self._rare_level_frame()
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cross_validate(model, X, y, cv=SimpleKFold(3, shuffle=True, random_state=0))
        assert model._config.level_bindings is None
        assert model._specs["g"]._levels == []

    def test_numeric_only_model_binds_nothing(self, poisson_data, base_model):
        """No cat-family term means no binding and no configuration rewrite."""
        df, y, sw = poisson_data

        assert _resolve_level_bindings(base_model, as_eager_frame(df), sw) == {}

        result = cross_validate(
            base_model,
            df,
            y,
            cv=SimpleKFold(2),
            sample_weight=sw,
            return_estimators=True,
            error_score="raise",
        )
        assert all(est._config.level_bindings is None for est in result.estimators)

    def test_binding_universe_includes_dtype_declared_levels(self):
        """The dtype declaration is a superset the binding must not narrow."""
        X, y = self._rare_level_frame()
        X = X.assign(g=pd.Categorical(X["g"], categories=["a", "b", "rare", "never"]))
        model = SuperGLM(family="poisson", features={"g": Categorical()})

        binding = _resolve_level_bindings(model, as_eager_frame(X), np.ones(len(y)))["g"]

        assert list(binding.levels) == ["a", "b", "rare", "never"]
        assert binding.base == X["g"].value_counts().idxmax()

    def test_thin_special_survives_cv(self):
        """One special row: the folds whose training misses it pin instead of dying."""
        rng = np.random.default_rng(3)
        band = rng.choice(["b1", "b2", "b3"], size=100).astype(object)
        band[0] = "MISSING"  # exactly one row: pigeonhole-guaranteed to miss a fold
        X = pd.DataFrame({"band": band})
        y = rng.poisson(1.0, size=100).astype(float)
        model = SuperGLM(
            family="poisson",
            features={
                "band": OrderedCategorical(
                    order=["b1", "b2", "b3"], specials=["MISSING"], basis=Spline(n_knots=1)
                )
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # expected pin warnings
            res = cross_validate(
                model,
                X,
                y,
                cv=SimpleKFold(5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert len(res.fold_scores) == 5
        assert all(np.isfinite(res.fold_scores["deviance"]))

    def test_ordered_categorical_is_not_bound(self):
        """OrderedCategorical declares through order=; the CV bind stays out."""
        X, y = self._rare_level_frame()
        X = X.assign(band=np.where(np.arange(len(y)) % 2 == 0, "low", "high"))
        model = SuperGLM(
            family="poisson",
            features={
                "g": Categorical(base="first"),
                "band": OrderedCategorical(order=["low", "high"], basis=Spline(n_knots=1)),
            },
        )

        bindings = _resolve_level_bindings(model, as_eager_frame(X), None)

        assert set(bindings) == {"g"}
