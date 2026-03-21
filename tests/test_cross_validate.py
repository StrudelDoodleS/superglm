"""Tests for cross_validate() free function."""

import numpy as np
import pandas as pd
import pytest

from superglm import CrossValidationResult, GroupLasso, Spline, SuperGLM, cross_validate

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


# ── Return options ────────────────────────────────────────────────


class TestReturnOptions:
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


# ── Error handling ────────────────────────────────────────────────


class TestErrorHandling:
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


class TestAutoDetectClone:
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

    def test_gini_all_zero_response(self, poisson_data, base_model):
        """Gini scorer returns 0.0 when all y=0 (no division by zero)."""
        df, y, sw = poisson_data
        y_zero = np.zeros_like(y)
        result = cross_validate(
            base_model,
            df,
            y_zero,
            cv=SimpleKFold(3),
            scoring="gini",
        )
        assert all(np.isfinite(result.fold_scores["gini"]))
