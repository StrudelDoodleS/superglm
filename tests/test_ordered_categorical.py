"""Tests for OrderedCategorical feature type."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, SuperGLM

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def age_band_data():
    """Synthetic data with ordered age bands."""
    rng = np.random.default_rng(42)
    n = 2000
    bands = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    x = rng.choice(bands, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
    exposure = rng.uniform(0.3, 1.0, n)
    midpoints = {
        "18-25": 21.5,
        "26-35": 30.5,
        "36-45": 40.5,
        "46-55": 50.5,
        "56-65": 60.5,
        "65+": 70.0,
    }
    x_numeric = np.array([midpoints[v] for v in x])
    mu = np.exp(-2.0 + 0.01 * (x_numeric - 45) ** 2 / 100)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"age_band": x})
    return X, y, exposure, midpoints, bands


@pytest.fixture
def ordinal_data():
    """Synthetic data with generic ordered levels."""
    rng = np.random.default_rng(123)
    n = 1000
    levels = ["Low", "Medium", "High", "Very High"]
    x = rng.choice(levels, n, p=[0.3, 0.3, 0.25, 0.15])
    exposure = rng.uniform(0.5, 1.0, n)
    # True effect: monotone increasing
    effect = {"Low": 0.0, "Medium": 0.2, "High": 0.5, "Very High": 0.8}
    mu = np.exp(-1.5 + np.array([effect[v] for v in x]))
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"risk": x})
    return X, y, exposure, levels


# ── Constructor Tests ─────────────────────────────────────────────


class TestConstructor:
    def test_values_derive_ordering(self):
        spec = OrderedCategorical(values={"C": 3.0, "A": 1.0, "B": 2.0}, basis="spline")
        assert spec._ordered_levels == ["A", "B", "C"]

    def test_order_generates_linspace(self):
        spec = OrderedCategorical(order=["X", "Y", "Z"], basis="spline")
        assert spec._level_to_value == {"X": 0.0, "Y": 0.5, "Z": 1.0}

    def test_mutual_exclusion_both(self):
        with pytest.raises(ValueError, match="exactly one"):
            OrderedCategorical(values={"A": 1.0}, order=["A"])

    def test_mutual_exclusion_neither(self):
        with pytest.raises(ValueError, match="Must specify"):
            OrderedCategorical()

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="basis must be"):
            OrderedCategorical(order=["A", "B"], basis="cubic")

    def test_step_select_raises(self):
        with pytest.raises(ValueError, match="select=True is not supported"):
            OrderedCategorical(order=["A", "B", "C"], basis="step", select=True)


# ── Spline Mode: Build / Transform / Reconstruct ─────────────────


class TestSplineMode:
    def test_build_returns_groupinfo(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        result = spec.build(X["age_band"].values, exposure=exposure)
        # Should return GroupInfo (not a list when select=False)
        from superglm.types import GroupInfo

        assert isinstance(result, GroupInfo)
        assert result.n_cols > 0
        assert result.penalty_matrix is not None

    def test_build_select_returns_list(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3, select=True)
        result = spec.build(X["age_band"].values, exposure=exposure)
        assert isinstance(result, list)
        assert len(result) == 2  # linear + spline subgroups

    def test_transform_shape(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        spec.build(X["age_band"].values, exposure=exposure)
        T = spec.transform(X["age_band"].values)
        assert T.shape[0] == len(X)

    def test_reconstruct_has_level_annotations(self, age_band_data):
        X, y, exposure, midpoints, bands = age_band_data
        # Use full model pipeline so R_inv is set correctly
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, exposure=exposure)
        spec = model._specs["age_band"]
        beta_combined = model._result.beta[model._groups[0].sl]
        raw = spec.reconstruct(beta_combined)
        # Should have standard spline keys
        assert "x" in raw
        assert "log_relativity" in raw
        assert "relativity" in raw
        # Plus per-level annotations
        assert "levels" in raw
        assert "level_values" in raw
        assert "level_log_relativities" in raw
        assert "level_relativities" in raw
        assert set(raw["levels"]) == set(bands)

    def test_unseen_level_raises(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline")
        spec.build(X["age_band"].values, exposure=exposure)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))

    def test_n_knots_clamping(self):
        """n_knots larger than n_levels-1 should be clamped with a warning."""
        with pytest.warns(UserWarning, match="clamped"):
            spec = OrderedCategorical(order=["A", "B", "C"], n_knots=10)
        assert spec._spline.n_knots == 2  # min(10, 3-1) = 2

    def test_spline_matches_manual(self, age_band_data):
        """Spline mode should produce the same result as manual Spline on midpoints."""
        from superglm.features.spline import BasisSpline

        X, y, exposure, midpoints, _ = age_band_data
        x_vals = X["age_band"].values
        x_numeric = np.array([midpoints[v] for v in x_vals])

        # Manual spline
        manual_spline = BasisSpline(n_knots=3, degree=3, penalty="ssp")
        manual_info = manual_spline.build(x_numeric, exposure=exposure)

        # OrderedCategorical
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        ocat_info = spec.build(x_vals, exposure=exposure)

        # Same penalty matrix
        np.testing.assert_allclose(ocat_info.penalty_matrix, manual_info.penalty_matrix)
        # Same number of columns
        assert ocat_info.n_cols == manual_info.n_cols


# ── Step Mode: Build / Transform / Reconstruct ───────────────────


class TestStepMode:
    def test_build_returns_groupinfo(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        from superglm.types import GroupInfo

        assert isinstance(info, GroupInfo)

    def test_penalty_shape(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        # K=4, base excluded → K-1=3 columns
        n_cols = len(levels) - 1
        assert info.n_cols == n_cols
        assert info.penalty_matrix.shape == (n_cols, n_cols)

    def test_penalty_is_d1td1(self, ordinal_data):
        """Penalty should be D1'D1 (first-difference penalty)."""
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        n_cols = info.n_cols
        D1 = np.diff(np.eye(n_cols), n=1, axis=0)
        expected = D1.T @ D1
        np.testing.assert_allclose(info.penalty_matrix, expected)

    def test_penalty_rank(self, ordinal_data):
        """D1'D1 for K-1 columns should have rank K-2."""
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        rank = np.linalg.matrix_rank(info.penalty_matrix)
        assert rank == len(levels) - 2  # K-2

    def test_reparametrize_flag(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        assert info.reparametrize is True
        assert info.penalized is True

    def test_transform_shape(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        spec.build(X["risk"].values, exposure=exposure)
        T = spec.transform(X["risk"].values)
        assert T.shape == (len(X), len(levels) - 1)

    def test_reconstruct_format(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, exposure=exposure)
        beta = np.zeros(info.n_cols)
        raw = spec.reconstruct(beta)
        assert "base_level" in raw
        assert "levels" in raw
        assert "log_relativities" in raw
        assert "relativities" in raw
        assert set(raw["levels"]) == set(levels)

    def test_base_level_most_exposed(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step", base="most_exposed")
        spec.build(X["risk"].values, exposure=exposure)
        # Should pick the level with highest total exposure
        assert spec._base_level in levels

    def test_base_level_explicit(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step", base="High")
        spec.build(X["risk"].values, exposure=exposure)
        assert spec._base_level == "High"


# ── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_k2_step_unpenalized(self):
        """K=2 step mode: D1 is empty, should fall back to unpenalized."""
        rng = np.random.default_rng(42)
        x = rng.choice(["A", "B"], 100)
        spec = OrderedCategorical(order=["A", "B"], basis="step")
        info = spec.build(x)
        # Only 1 non-base column
        assert info.n_cols == 1
        # Should be unpenalized (no difference penalty possible)
        assert info.penalty_matrix is None
        assert info.reparametrize is False

    def test_k3_step_rank_deficient(self):
        """K=3 step mode: D1 is (1,2), omega is (2,2) rank 1."""
        rng = np.random.default_rng(42)
        x = rng.choice(["A", "B", "C"], 200)
        spec = OrderedCategorical(order=["A", "B", "C"], basis="step")
        info = spec.build(x)
        assert info.n_cols == 2
        assert info.penalty_matrix.shape == (2, 2)
        assert np.linalg.matrix_rank(info.penalty_matrix) == 1

    def test_order_single_value_linspace(self):
        """Single level should produce value=0.0."""
        # This is degenerate but shouldn't crash
        spec = OrderedCategorical(order=["Only"], basis="step")
        assert spec._level_to_value == {"Only": 0.0}

    def test_unseen_level_at_predict(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        spec.build(X["risk"].values, exposure=exposure)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))


# ── Integration Tests ─────────────────────────────────────────────


class TestIntegrationSpline:
    def test_fit_predict(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_summary(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, exposure=exposure)
        s = model.summary()
        text = str(s)
        assert "age_band" in text

    def test_relativities(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "age_band" in rels
        df = rels["age_band"]
        assert "relativity" in df.columns

    def test_term_inference(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, exposure=exposure)
        ti = model.term_inference("age_band")
        assert ti.kind == "spline"
        assert ti.x is not None
        assert ti.relativity is not None


class TestIntegrationStep:
    def test_fit_predict(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_summary(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, exposure=exposure)
        s = model.summary()
        text = str(s)
        assert "risk" in text

    def test_relativities(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "risk" in rels

    def test_term_inference(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, exposure=exposure)
        ti = model.term_inference("risk")
        assert ti.kind == "categorical"
        assert ti.levels is not None
        assert ti.relativity is not None


class TestIntegrationReml:
    def test_fit_reml_spline(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit_reml(X, y, exposure=exposure)
        assert model._reml_result is not None
        assert model._reml_result.converged
        assert len(model._reml_result.lambdas) > 0
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_fit_reml_step(self, ordinal_data):
        X, y, exposure, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit_reml(X, y, exposure=exposure)
        assert model._reml_result is not None
        assert model._reml_result.converged
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_fit_reml_select(self, age_band_data):
        X, y, exposure, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, select=True
                )
            },
        )
        model.fit_reml(X, y, exposure=exposure)
        assert model._reml_result is not None
        assert model._reml_result.converged


class TestIntegrationMixed:
    def test_ocat_with_other_features(self, age_band_data):
        """OrderedCategorical works alongside other feature types."""
        X, y, exposure, midpoints, _ = age_band_data
        rng = np.random.default_rng(42)
        X = X.copy()
        X["region"] = rng.choice(["A", "B", "C"], len(X))
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)
