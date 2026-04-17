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
    sample_weight = rng.uniform(0.3, 1.0, n)
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
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age_band": x})
    return X, y, sample_weight, midpoints, bands


@pytest.fixture
def ordinal_data():
    """Synthetic data with generic ordered levels."""
    rng = np.random.default_rng(123)
    n = 1000
    levels = ["Low", "Medium", "High", "Very High"]
    x = rng.choice(levels, n, p=[0.3, 0.3, 0.25, 0.15])
    sample_weight = rng.uniform(0.5, 1.0, n)
    # True effect: monotone increasing
    effect = {"Low": 0.0, "Medium": 0.2, "High": 0.5, "Very High": 0.8}
    mu = np.exp(-1.5 + np.array([effect[v] for v in x]))
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"risk": x})
    return X, y, sample_weight, levels


# ── Constructor Tests ─────────────────────────────────────────────


class TestConstructor:
    def test_values_derive_ordering(self):
        spec = OrderedCategorical(values={"C": 3.0, "A": 1.0, "B": 2.0}, basis="spline", n_knots=2)
        assert spec._ordered_levels == ["A", "B", "C"]

    def test_order_generates_linspace(self):
        spec = OrderedCategorical(order=["X", "Y", "Z"], basis="spline", n_knots=2)
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
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        result = spec.build(X["age_band"].values, sample_weight=sample_weight)
        # Should return GroupInfo (not a list when select=False)
        from superglm.types import GroupInfo

        assert isinstance(result, GroupInfo)
        assert result.n_cols > 0
        assert result.penalty_matrix is not None

    def test_build_select_returns_single_group_with_components(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3, select=True)
        result = spec.build(X["age_band"].values, sample_weight=sample_weight)
        assert not isinstance(result, list)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2  # null + wiggle

    def test_transform_shape(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        spec.build(X["age_band"].values, sample_weight=sample_weight)
        T = spec.transform(X["age_band"].values)
        assert T.shape[0] == len(X)

    def test_reconstruct_has_level_annotations(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        # Use full model pipeline so R_inv is set correctly
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
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
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis="spline")
        spec.build(X["age_band"].values, sample_weight=sample_weight)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))

    def test_n_knots_clamping(self):
        """n_knots larger than n_levels-1 should be clamped with a warning."""
        with pytest.warns(UserWarning, match="clamped"):
            spec = OrderedCategorical(order=["A", "B", "C"], n_knots=10)
        assert spec._spline.n_knots == 2  # min(10, 3-1) = 2

    def test_spline_matches_manual(self, age_band_data):
        """Spline mode should produce the same result as manual Spline on midpoints."""
        from superglm.features.spline import PSpline

        X, y, sample_weight, midpoints, _ = age_band_data
        x_vals = X["age_band"].values
        x_numeric = np.array([midpoints[v] for v in x_vals])

        # Manual spline
        manual_spline = PSpline(n_knots=3, degree=3, penalty="ssp")
        manual_info = manual_spline.build(x_numeric, sample_weight=sample_weight)

        # OrderedCategorical
        spec = OrderedCategorical(values=midpoints, basis="spline", n_knots=3)
        ocat_info = spec.build(x_vals, sample_weight=sample_weight)

        # Same penalty matrix
        np.testing.assert_allclose(ocat_info.penalty_matrix, manual_info.penalty_matrix)
        # Same number of columns
        assert ocat_info.n_cols == manual_info.n_cols


# ── Step Mode: Build / Transform / Reconstruct ───────────────────


class TestStepMode:
    def test_build_returns_groupinfo(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        from superglm.types import GroupInfo

        assert isinstance(info, GroupInfo)

    def test_penalty_shape(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        # K=4, base excluded → K-1=3 columns
        n_cols = len(levels) - 1
        assert info.n_cols == n_cols
        assert info.penalty_matrix.shape == (n_cols, n_cols)

    def test_penalty_is_projected_d1td1(self, ordinal_data):
        """Penalty should be Z'D1'D1Z (projected first-difference)."""
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        K = len(levels)
        base_idx = spec._ordered_levels.index(spec._base_level)
        D1 = np.diff(np.eye(K), n=1, axis=0)
        Z = np.zeros((K, K - 1))
        j = 0
        for i in range(K):
            if i != base_idx:
                Z[i, j] = 1.0
                j += 1
        expected = Z.T @ D1.T @ D1 @ Z
        np.testing.assert_allclose(info.penalty_matrix, expected)

    def test_penalty_rank(self, ordinal_data):
        """Projected D1 penalty on K-1 columns is full rank (K-1).

        The base-to-neighbor difference makes the projected penalty full rank,
        unlike naive D1 on K-1 columns which has rank K-2.
        """
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        rank = np.linalg.matrix_rank(info.penalty_matrix)
        assert rank == len(levels) - 1  # full rank in (K-1) space

    def test_reparametrize_flag(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        assert info.reparametrize is True
        assert info.penalized is True

    def test_transform_shape(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        spec.build(X["risk"].values, sample_weight=sample_weight)
        T = spec.transform(X["risk"].values)
        assert T.shape == (len(X), len(levels) - 1)

    def test_reconstruct_format(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        info = spec.build(X["risk"].values, sample_weight=sample_weight)
        beta = np.zeros(info.n_cols)
        raw = spec.reconstruct(beta)
        assert "base_level" in raw
        assert "levels" in raw
        assert "log_relativities" in raw
        assert "relativities" in raw
        assert set(raw["levels"]) == set(levels)

    def test_base_level_most_exposed(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step", base="most_exposed")
        spec.build(X["risk"].values, sample_weight=sample_weight)
        # Should pick the level with highest total sample_weight
        assert spec._base_level in levels

    def test_base_level_explicit(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step", base="High")
        spec.build(X["risk"].values, sample_weight=sample_weight)
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

    def test_k3_step_full_rank(self):
        """K=3 step mode: projected penalty is (2,2) full rank."""
        rng = np.random.default_rng(42)
        x = rng.choice(["A", "B", "C"], 200)
        spec = OrderedCategorical(order=["A", "B", "C"], basis="step")
        info = spec.build(x)
        assert info.n_cols == 2
        assert info.penalty_matrix.shape == (2, 2)
        assert np.linalg.matrix_rank(info.penalty_matrix) == 2  # full rank

    def test_middle_base_penalty_respects_adjacency(self):
        """D1 penalty with base in middle must still fuse original neighbours.

        With levels [A, B, C, D] and base=B, _non_base=[A, C, D].
        The penalty should fuse A↔B(=0), B(=0)↔C, C↔D — not A↔C directly.
        """
        rng = np.random.default_rng(42)
        x = rng.choice(["A", "B", "C", "D"], 400)
        spec = OrderedCategorical(order=["A", "B", "C", "D"], basis="step", base="B")
        info = spec.build(x)
        omega = info.penalty_matrix
        # Build expected: Z'D1'D1Z where Z removes row 1 (base=B at index 1)
        K = 4
        D1 = np.diff(np.eye(K), n=1, axis=0)
        Z = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected = Z.T @ D1.T @ D1 @ Z
        np.testing.assert_allclose(omega, expected)
        # The (0,0) entry should be 1 (A↔B penalty), not 2 (which you'd get
        # from naive D1 on [A,C,D] treating A↔C as adjacent)
        assert omega[0, 0] == pytest.approx(1.0)

    def test_endpoint_base_projected_penalty(self):
        """When base is the first level, projected penalty includes A↔B difference."""
        rng = np.random.default_rng(42)
        x = rng.choice(["A", "B", "C", "D"], 400)
        spec = OrderedCategorical(order=["A", "B", "C", "D"], basis="step", base="A")
        info = spec.build(x)
        # Z'D1'D1Z with base=A (index 0) should differ from naive:
        # (0,0) = 2 (A↔B + B↔C), not 1 (only B↔C in naive)
        K = 4
        D1 = np.diff(np.eye(K), n=1, axis=0)
        Z = np.zeros((K, K - 1))
        Z[1, 0] = Z[2, 1] = Z[3, 2] = 1.0
        expected = Z.T @ D1.T @ D1 @ Z
        np.testing.assert_allclose(info.penalty_matrix, expected)

    def test_order_single_value_linspace(self):
        """Single level should produce value=0.0."""
        # This is degenerate but shouldn't crash
        spec = OrderedCategorical(order=["Only"], basis="step")
        assert spec._level_to_value == {"Only": 0.0}

    def test_unseen_level_at_predict(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis="step")
        spec.build(X["risk"].values, sample_weight=sample_weight)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))


# ── Integration Tests ─────────────────────────────────────────────


class TestIntegrationSpline:
    def test_fit_predict(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_summary(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        s = model.summary()
        text = str(s)
        assert "age_band[18-25]" in text
        assert f"age_band[{bands[-1]}]" in text
        level_rows = [r for r in s._coef_rows if r.group == "age_band" and not r.is_spline]
        assert len(level_rows) == len(bands)
        assert all(r.name.startswith("age_band[") for r in level_rows)
        assert not any(r.name == "age_band" and r.coef is not None for r in s._coef_rows)
        assert all(np.isfinite(r.se) and r.se >= 0 for r in level_rows if r.se is not None)

    def test_relativities(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities()
        assert "age_band" in rels
        df = rels["age_band"]
        assert "relativity" in df.columns

    def test_term_inference(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        # Primary output is categorical (K levels), not a continuous curve
        assert ti.kind == "categorical"
        assert ti.levels is not None
        assert set(ti.levels) == set(bands)
        assert len(ti.relativity) == len(bands)

    def test_spline_se_at_levels(self, age_band_data):
        """Spline mode SEs should be at the K category positions, not on a grid."""
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        se = ti.se_log_relativity
        assert se is not None
        # SE array should have K entries (one per level), not 200
        assert len(se) == len(bands)
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.any(se > 0)
        assert np.max(se) < 5.0

    def test_smooth_curve_for_plotting(self, age_band_data):
        """Spline mode should provide a smooth_curve for plotting."""
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        curve = ti.smooth_curve
        assert curve is not None
        # Continuous grid (default 200 points)
        assert len(curve.x) == 200
        assert len(curve.relativity) == 200
        assert curve.se_log_relativity is not None
        assert len(curve.se_log_relativity) == 200
        assert curve.ci_lower is not None
        assert curve.ci_upper is not None

    def test_relativities_per_level(self, age_band_data):
        """relativities() should return per-level output, not a continuous curve."""
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities(with_se=True)
        df = rels["age_band"]
        assert "level" in df.columns
        assert len(df) == len(bands)
        assert "se_log_relativity" in df.columns
        assert np.all(np.isfinite(df["se_log_relativity"].values))


class TestIntegrationStep:
    def test_fit_predict(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_summary(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        s = model.summary()
        text = str(s)
        base_level = model._specs["risk"]._base_level
        visible_levels = [lev for lev in levels if lev != base_level]
        assert f"risk[{visible_levels[0]}]" in text
        assert f"risk[{visible_levels[-1]}]" in text
        level_rows = [r for r in s._coef_rows if r.group == "risk" and not r.is_spline]
        assert len(level_rows) == len(levels) - 1
        assert all(r.name.startswith("risk[") for r in level_rows)
        assert not any(r.name == "risk" and r.coef is not None for r in s._coef_rows)
        assert all(np.isfinite(r.se) and r.se >= 0 for r in level_rows if r.se is not None)

    def test_relativities(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities()
        assert "risk" in rels

    def test_term_inference(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("risk")
        assert ti.kind == "categorical"
        assert ti.levels is not None
        assert ti.relativity is not None

    def test_step_se_numerically_reasonable(self, ordinal_data):
        """Step mode SEs should be finite, positive for non-base, and sensible size."""
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("risk")
        se = ti.se_log_relativity
        assert se is not None
        assert np.all(np.isfinite(se))
        # Base level SE should be 0, non-base should be > 0
        base_idx = levels.index(model._specs["risk"]._base_level)
        assert se[base_idx] == 0.0
        non_base_se = np.delete(se, base_idx)
        assert np.all(non_base_se > 0)
        # SEs should be O(0.01-1) for reasonable Poisson data, not huge
        assert np.all(non_base_se < 5.0)

    def test_step_middle_base_fit(self):
        """Step mode with base in middle of ordering should fit correctly."""
        rng = np.random.default_rng(42)
        n = 2000
        levels = ["A", "B", "C", "D", "E"]
        x = rng.choice(levels, n)
        sample_weight = rng.uniform(0.5, 1.0, n)
        effect = {"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.5}
        mu = np.exp(-1.5 + np.array([effect[v] for v in x]))
        y = rng.poisson(mu * sample_weight).astype(float)
        X = pd.DataFrame({"cat": x})
        model = SuperGLM(
            features={"cat": OrderedCategorical(order=levels, basis="step", base="C")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("cat")
        assert ti.active
        # Relativities should be monotonically non-decreasing (true effect is monotone)
        rels = np.array([ti.relativity[levels.index(lev)] for lev in levels])
        assert rels[0] < rels[-1]  # first < last at minimum


class TestIntegrationReml:
    def test_fit_reml_spline(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={"age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3)},
        )
        model.fit_reml(X, y, sample_weight=sample_weight)
        assert model._reml_result is not None
        assert model._reml_result.converged
        assert len(model._reml_result.lambdas) > 0
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_fit_reml_step(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        model = SuperGLM(
            features={"risk": OrderedCategorical(order=levels, basis="step")},
        )
        model.fit_reml(X, y, sample_weight=sample_weight)
        assert model._reml_result is not None
        assert model._reml_result.converged
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_fit_reml_select(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis="spline", n_knots=3, select=True
                )
            },
        )
        model.fit_reml(X, y, sample_weight=sample_weight)
        assert model._reml_result is not None
        assert model._reml_result.converged


class TestIntegrationMixed:
    def test_ocat_with_other_features(self, age_band_data):
        """OrderedCategorical works alongside other feature types."""
        X, y, sample_weight, midpoints, _ = age_band_data
        rng = np.random.default_rng(42)
        X = X.copy()
        X["region"] = rng.choice(["A", "B", "C"], len(X))
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis="spline", n_knots=3),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)


class TestSplineObjectBasis:
    """OrderedCategorical with a Spline() object as basis."""

    def test_spline_object_builds_and_fits(self, age_band_data):
        from superglm.features.spline import Spline

        X, y, sample_weight, _, _ = age_band_data
        spec = OrderedCategorical(
            order=sorted(X["age_band"].unique()),
            basis=Spline(n_knots=4, kind="bs"),
        )
        model = SuperGLM(
            family="poisson",
            features={"age_band": spec},
            selection_penalty=0.0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged

    def test_spline_object_with_monotone(self, age_band_data):
        from superglm.features.spline import Spline

        X, y, sample_weight, _, _ = age_band_data
        spec = OrderedCategorical(
            order=sorted(X["age_band"].unique()),
            basis=Spline(n_knots=4, monotone="increasing"),
        )
        model = SuperGLM(
            family="poisson",
            features={"age_band": spec},
            selection_penalty=0.0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged
        # Monotone attribute should be visible on the internal spline
        assert spec._spline.monotone == "increasing"

    def test_spline_object_overrides_string_params(self, age_band_data):
        """When Spline object is passed, kind/n_knots/etc are ignored."""
        from superglm.features.spline import Spline

        X, y, sample_weight, _, _ = age_band_data
        spec = OrderedCategorical(
            order=sorted(X["age_band"].unique()),
            basis=Spline(n_knots=3, kind="cr"),
            kind="bs",  # should be ignored
            n_knots=10,  # should be ignored
        )
        # Internal spline should use the Spline object's params
        assert spec._spline.n_knots <= 3  # may be clamped but not 10

    def test_repr_shows_spline_object(self):
        from superglm.features.spline import Spline

        spec = OrderedCategorical(
            order=["a", "b", "c", "d"],
            basis=Spline(n_knots=3),
        )
        r = repr(spec)
        assert "OrderedCategorical" in r
        assert "4 levels" in r

    def test_n_knots_clamped_for_spline_object(self):
        from superglm.features.spline import Spline

        with pytest.warns(UserWarning, match="clamped"):
            spec = OrderedCategorical(
                order=["a", "b", "c"],  # 3 levels → max 2 knots
                basis=Spline(n_knots=10),
            )
        assert spec._spline.n_knots == 2
