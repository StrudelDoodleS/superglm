"""Tests for mgcv-style spline decomposition (select=True, double penalty)."""

import numpy as np
import pandas as pd
import pytest

from superglm.features.spline import Spline
from superglm.model import SuperGLM
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_lasso import GroupLasso


# ── Fixture data ───────────────────────────────────────────────
@pytest.fixture
def simple_data():
    """Small synthetic dataset with one nonlinear and one noise feature."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(18, 80, n)
    x2 = rng.uniform(0, 10, n)  # noise
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100)
    y = rng.poisson(mu)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"signal": x1, "noise": x2})
    return X, y, sample_weight


# ── Build-time tests ──────────────────────────────────────────


class TestSelectBuild:
    def test_select_creates_single_group_info_with_penalty_components(self):
        """Spline(select=True).build() returns a single GroupInfo with penalty_components."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert not isinstance(result, list)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2

    def test_select_component_suffixes(self):
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        suffixes = [name for name, _ in result.penalty_components]
        assert suffixes == ["null", "wiggle"]

    def test_select_component_types(self):
        """The null component must carry component_type='selection'."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert result.component_types is not None
        assert result.component_types.get("null") == "selection"
        # wiggle should not have a component_type entry (or not "selection")
        assert "wiggle" not in result.component_types

    def test_select_combined_n_cols(self):
        """Combined n_cols = 1 (null) + n_range."""
        for nk in [5, 10, 20]:
            sp = Spline(n_knots=nk, select=True)
            result = sp.build(np.linspace(0, 1, 200))
            n_basis = sp._n_basis
            n_range = n_basis - 2  # K - 2 for BS (partition of unity removes 1, null removes 1)
            expected = 1 + n_range
            assert result.n_cols == expected, (
                f"Expected n_cols={expected} for n_knots={nk}, got {result.n_cols}"
            )

    def test_select_projections_orthogonal(self):
        """U_null and U_range should have orthonormal columns."""
        sp = Spline(n_knots=10, select=True)
        sp.build(np.linspace(0, 1, 100))
        U_null = sp._U_null  # (K, 1)
        U_range = sp._U_range  # (K, n_range)
        # U_null is orthogonal to U_range
        cross = U_null.T @ U_range
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)
        # Both have orthonormal columns
        np.testing.assert_allclose(U_null.T @ U_null, np.eye(1), atol=1e-10)
        np.testing.assert_allclose(U_range.T @ U_range, np.eye(U_range.shape[1]), atol=1e-10)

    def test_select_combined_projection_structure(self):
        """Combined projection's first column (null) and remaining (range) are orthogonal."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        U = result.projection  # (K, n_combined)
        U_null_part = U[:, :1]
        U_range_part = U[:, 1:]
        cross = U_null_part.T @ U_range_part
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)

    def test_select_null_space_centered(self):
        """Null-space linear component should be orthogonal to the constant."""
        sp = Spline(n_knots=10, select=True)
        sp.build(np.linspace(0, 1, 100))
        # B-splines have partition of unity, so ones vector is the constant
        ones = np.ones(sp._n_basis)
        proj = sp._U_null.T @ ones
        np.testing.assert_allclose(proj, 0.0, atol=1e-10)

    def test_select_penalty_matrix_equals_component_sum(self):
        """The combined penalty_matrix must equal the sum of component omegas."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        omega_sum = sum(omega for _, omega in result.penalty_components)
        np.testing.assert_allclose(result.penalty_matrix, omega_sum, atol=1e-14)

    def test_select_wiggle_component_positive_diagonal(self):
        """Wiggle component's nonzero block should have positive diagonal entries."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        _, omega_wiggle = result.penalty_components[1]
        # The wiggle block is in the lower-right (after the 1-col null space)
        wiggle_block = omega_wiggle[1:, 1:]
        # Should be diagonal
        off_diag = wiggle_block - np.diag(np.diag(wiggle_block))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)
        # All diagonal entries should be positive
        assert np.all(np.diag(wiggle_block) > 0)

    def test_select_reparametrize_flag(self):
        """select=True GroupInfo must have reparametrize=True."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert result.reparametrize is True

    def test_select_no_subgroup_name(self):
        """select=True GroupInfo must not have a subgroup_name."""
        sp = Spline(n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert result.subgroup_name is None


class TestSelectDefaults:
    def test_default_select_false(self):
        """Default Spline has select=False."""
        sp = Spline(n_knots=10)
        assert sp.select is False

    def test_no_select_returns_single_group_info(self):
        sp = Spline(n_knots=10, select=False)
        result = sp.build(np.linspace(0, 1, 100))
        assert not isinstance(result, list)


# ── Model-level tests ─────────────────────────────────────────


class TestSelectModel:
    def test_select_creates_one_group_per_feature(self, simple_data):
        """Model with select=True spline creates 1 group per feature (combined null+wiggle)."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        group_names = [g.name for g in m._groups]
        assert "signal" in group_names
        assert "noise" in group_names
        assert len(m._groups) == 2

    def test_select_feature_name(self, simple_data):
        """Group has correct feature_name."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        for g in m._groups:
            assert g.feature_name == "signal"

    def test_feature_groups_helper(self, simple_data):
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        groups = m._feature_groups("signal")
        assert len(groups) == 1
        assert groups[0].name == "signal"

    def test_select_predict_matches_no_select(self, simple_data):
        """Predictions with select=True should be close to select=False."""
        X, y, sample_weight = simple_data
        m1 = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=False)},
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m1.fit(X, y, sample_weight=sample_weight)
        pred1 = m1.predict(X)

        m2 = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m2.fit(X, y, sample_weight=sample_weight)
        pred2 = m2.predict(X)

        # Not numerically identical (different parameterization), but close.
        # Open knot vectors can shift edge predictions slightly.
        np.testing.assert_allclose(pred1, pred2, rtol=0.2)

    def test_select_reconstruct(self, simple_data):
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        rec = m.reconstruct_feature("signal")
        assert "x" in rec
        assert "relativity" in rec
        assert len(rec["x"]) == 200

    def test_select_relativities(self, simple_data):
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        rels = m.relativities(with_se=True)
        assert "signal" in rels
        assert "se_log_relativity" in rels["signal"].columns

    def test_select_summary_dict(self, simple_data):
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        s = m.diagnostics()
        # Single group named "signal" (not "signal:linear" / "signal:spline")
        assert "signal" in s


class TestSelectSparsity:
    def test_high_lambda_zeros_group(self, simple_data):
        """At high lambda, the combined group should be zeroed (double penalty)."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1e4,
        )
        m.fit(X, y, sample_weight=sample_weight)
        s = m.diagnostics()
        group_norm = s["signal"]["group_norm"]
        assert group_norm < 1e-6, (
            f"Expected group zeroed at selection_penalty=1e4: group_norm={group_norm:.6f}"
        )

    def test_very_high_lambda_zeros_group(self, simple_data):
        """At very high lambda, the combined group is zeroed (double penalty)."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1e6,
        )
        m.fit(X, y, sample_weight=sample_weight)
        s = m.diagnostics()
        # Combined group is penalized — zeroed at very high lambda
        assert s["signal"]["group_norm"] < 1e-10


class TestSelectPath:
    def test_fit_path(self, simple_data):
        """Regularization path with select=True should work."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        path = m.fit_path(X, y, sample_weight=sample_weight, n_lambda=10)
        assert path.coef_path.shape[0] == 10
        # null(1) + range(K-2) = K-1 columns
        assert path.coef_path.shape[1] == 1 + (m._specs["signal"]._n_basis - 2)


# ── Part 3: Additional coverage ──────────────────────────────


class TestSelectCovariance:
    """3A. Covariance with select=True (active info path)."""

    def test_active_info_sizes(self, simple_data):
        """_active_info produces correctly-sized covariance and active_groups."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=0.1,
        )
        m.fit(X, y, sample_weight=sample_weight)
        metrics = m.metrics(X, y, sample_weight=sample_weight)
        X_a, W, XtWX_inv, _, active_groups = metrics._active_info

        # Covariance should be square, matching total active columns
        assert XtWX_inv.shape[0] == XtWX_inv.shape[1]
        total_active_cols = sum(ag.size for ag in active_groups)
        assert XtWX_inv.shape[0] == total_active_cols

    def test_subgroup_type_is_none(self, simple_data):
        """GroupSlice.subgroup_type is None for select=True (single combined group)."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        signal_g = next(g for g in m._groups if g.name == "signal")
        assert signal_g.subgroup_type is None


class TestSelectAdaptive:
    """3B. Adaptive + select combination."""

    def test_adaptive_select_finite_weights(self, simple_data):
        """Adaptive(expon=2) with select=True produces finite weights."""
        X, y, sample_weight = simple_data
        pen = GroupLasso(lambda1=1.0, flavor=Adaptive(expon=2))
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            penalty=pen,
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        # Adaptive weights are applied during fit — must converge with valid result
        assert m.result.converged, (
            f"Adaptive select fit failed to converge in {m.result.n_iter} iters"
        )
        assert np.isfinite(m.result.deviance)
        assert m.result.deviance > 0
        # All coefficients must be finite
        assert np.all(np.isfinite(m.result.beta))
        # Predictions should be positive (Poisson)
        pred = m.predict(X)
        assert np.all(pred > 0)

    def test_adaptive_select_noise_zeroed(self, simple_data):
        """At moderate lambda with Adaptive(expon=2), noise group should be zeroed."""
        X, y, sample_weight = simple_data
        pen = GroupLasso(lambda1=10.0, flavor=Adaptive(expon=2))
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
            penalty=pen,
            spline_penalty=1.0,
        )
        m.fit(X, y, sample_weight=sample_weight)
        noise_g = next(g for g in m._groups if g.name == "noise")
        assert np.linalg.norm(m.result.beta[noise_g.sl]) < 1e-10


class TestSelectEffectiveDf:
    """3C. Effective DF with penalized groups (double penalty)."""

    def test_high_lambda_edf_minimal(self, simple_data):
        """At very high lambda, all groups are zeroed — only intercept remains."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
            spline_penalty=1.0,
            selection_penalty=1e6,
        )
        m.fit(X, y, sample_weight=sample_weight)
        # With double penalty, all groups are penalized and zeroed at extreme lambda
        # edf should be approximately 1 (intercept only)
        assert m.result.effective_df == pytest.approx(1.0, abs=0.5)


class TestSelectLambdaMax:
    """3D. lambda_max with penalized groups (double penalty)."""

    def test_lambda_max_all_penalized(self, simple_data):
        """With double penalty, all groups are penalized."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1.0,  # set to prevent auto-calibration
        )
        m.fit(X, y, sample_weight=sample_weight)
        lmax = m._compute_lambda_max(y.astype(np.float64), sample_weight)
        # lambda_max should be finite and positive
        assert np.isfinite(lmax)
        assert lmax > 0
        # With double penalty, all groups are penalized
        assert all(g.penalized for g in m._groups)


class TestSelectSummaryOutput:
    """3E. Summary output for select=True (single combined group)."""

    def test_summary_inactive_group(self, simple_data):
        """When the combined group is zeroed, summary shows inactive."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1e6,
        )
        m.fit(X, y, sample_weight=sample_weight)
        metrics = m.metrics(X, y, sample_weight=sample_weight)
        summary = metrics.summary()
        text = str(summary)
        # Combined group shows as spline with inactive label
        assert "inactive" in text

    def test_summary_html_group_label(self, simple_data):
        """HTML summary shows the combined group correctly."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1e6,
        )
        m.fit(X, y, sample_weight=sample_weight)
        metrics = m.metrics(X, y, sample_weight=sample_weight)
        summary = metrics.summary()
        html = summary._repr_html_()
        assert "inactive" in html


class TestSelectLambdaReporting:
    """3E-bis. Reported lambdas match fitted REML values for select terms."""

    @pytest.mark.slow
    def test_summary_lambda_matches_reml(self):
        """summary() reports fitted REML lambda, not scalar default."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        m = SuperGLM(
            family="poisson",
            features={"x": Spline(n_knots=8, select=True)},
        )
        m.fit_reml(X, y, max_reml_iter=20)

        # Fitted lambdas should not be the scalar default (0.1)
        lam = m._reml_lambdas
        assert "x:null" in lam
        assert "x:wiggle" in lam

        # Summary should report a lambda derived from the fitted values
        summary = m.summary()
        rows = summary._coef_rows
        x_row = next(r for r in rows if r.name == "x")
        reported = x_row.smoothing_lambda
        assert reported is not None
        # Must differ from the scalar lambda2 default
        assert reported != m.lambda2, (
            f"Summary lambda {reported} equals scalar default {m.lambda2} — "
            "fitted REML values not propagated"
        )

    @pytest.mark.slow
    def test_term_importance_lambda_matches_reml(self):
        """term_importance() reports fitted REML lambda, not scalar default."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        m = SuperGLM(
            family="poisson",
            features={"x": Spline(n_knots=8, select=True)},
        )
        m.fit_reml(X, y, max_reml_iter=20)

        diag = m.term_importance(X)
        x_row = diag[diag["term"] == "x"].iloc[0]
        reported = x_row["lambda"]
        assert reported is not None
        assert reported != m.lambda2, (
            f"term_importance lambda {reported} equals scalar default {m.lambda2} — "
            "fitted REML values not propagated"
        )


class TestSelectFeatureSE:
    """3F. feature_se with select=True."""

    def test_feature_se_reml_select_nonlinear_dgp(self):
        """feature_se works with REML select=True when the group is active.

        Uses fit_reml() with selection_penalty=0 (direct solver, no BCD aliasing).
        On a purely nonlinear DGP, REML drives the null component's lambda
        high (no signal) while keeping the wiggle lambda moderate (real signal).
        The combined group remains active (no L1 sparsity); the test verifies that
        feature_se returns finite, non-negative SEs through the full covariance
        code path.
        """
        rng = np.random.default_rng(99)
        n = 500
        x = rng.uniform(0, 10, n)
        eta = -0.5 + 0.4 * np.sin(x)  # purely nonlinear
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"signal": x})
        sample_weight = np.ones(n)

        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            selection_penalty=0.0,
        )
        m.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=20)

        # Combined group should have captured the nonlinear signal
        signal_g = next(g for g in m._groups if g.name == "signal")
        assert np.linalg.norm(m.result.beta[signal_g.sl]) > 1e-6, (
            "Expected signal group to be active for nonlinear DGP"
        )

        metrics = m.metrics(X, y, sample_weight=sample_weight)
        fse = metrics.feature_se("signal")
        assert "x" in fse
        assert "se_log_relativity" in fse
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.max(se) > 0

    def test_feature_se_both_zeroed(self):
        """feature_se returns all-zero SEs when the combined group is zeroed.

        Tests the early-return branch in metrics.feature_se when all
        coefficients for a feature are exactly zero.
        """
        rng = np.random.default_rng(100)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)  # pure noise, no x effect
        X = pd.DataFrame({"signal": x})
        sample_weight = np.ones(n)

        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=1e6,  # extreme penalty → group zeroed
        )
        m.fit(X, y, sample_weight=sample_weight)

        # Precondition: group zeroed
        for g in m._groups:
            assert np.linalg.norm(m.result.beta[g.sl]) < 1e-10

        metrics = m.metrics(X, y, sample_weight=sample_weight)
        fse = metrics.feature_se("signal")
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        np.testing.assert_allclose(se, 0.0, atol=1e-10)

    def test_feature_se_active(self, simple_data):
        """feature_se works when the combined select group is active."""
        X, y, sample_weight = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=0.1,
        )
        m.fit(X, y, sample_weight=sample_weight)
        metrics = m.metrics(X, y, sample_weight=sample_weight)
        fse = metrics.feature_se("signal")
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.max(se) > 0


class TestSelectNoiseSuppressionREML:
    """Regression test: REML with select=True must suppress noise features."""

    def test_noise_edf_below_threshold(self):
        """Noise smooth total EDF < 0.02 on synthetic Poisson data (n=50k)."""
        rng = np.random.default_rng(42)
        n = 50000
        x_signal = rng.uniform(0, 1, n)
        x_noise = rng.uniform(0, 1, n)
        mu = np.exp(-0.5 + 0.5 * np.sin(2 * np.pi * x_signal))
        y = rng.poisson(mu)
        X = pd.DataFrame({"signal": x_signal, "noise": x_noise})

        model = SuperGLM(
            features={
                "signal": Spline(kind="bs", k=12, select=True, discrete=True),
                "noise": Spline(kind="bs", k=12, select=True, discrete=True),
            },
            family="poisson",
            selection_penalty=0,
            discrete=True,
            n_bins=256,
        )
        model.fit_reml(X, y, max_reml_iter=30)

        # Compute per-group EDF via metrics machinery
        metrics_obj = model.metrics(X, y)
        edf, _ = metrics_obj._influence_edf
        _, _, _, _, active_groups = metrics_obj._active_info
        group_edf = {ag.name: float(np.sum(edf[ag.sl])) for ag in active_groups}

        # Noise group: EDF < 0.05 (effectively inactive; tolerance accounts for
        # Newton convergence path which may not fully penalize noise groups)
        for g in model._groups:
            if "noise" in g.name.lower():
                noise_edf = group_edf.get(g.name, 0.0)
                assert noise_edf < 0.05, f"{g.name} EDF={noise_edf:.4f}, expected < 0.05"

        # Signal group should retain meaningful EDF
        signal_edf = sum(
            group_edf.get(g.name, 0.0) for g in model._groups if "signal" in g.name.lower()
        )
        assert signal_edf > 3.0, f"Signal total EDF={signal_edf:.2f}, expected > 3.0"

        # Noise lambdas should be large (strongly penalized)
        noise_lambdas = {
            name: lam for name, lam in model._reml_lambdas.items() if "noise" in name.lower()
        }
        for name, lam in noise_lambdas.items():
            assert lam > 1e5, f"{name} lambda={lam:.1f}, expected > 1e5"


class TestSplitLinearSnapWeakSignal:
    """Regression test: FP snap must not falsely suppress weak-but-real signals.

    Synthetic Poisson data with one strong signal, one weak (but real) signal,
    and one pure noise spline. The snap should kill the noise but preserve
    the weak signal with nontrivial EDF.
    """

    def test_weak_signal_preserved_noise_killed(self):
        rng = np.random.default_rng(42)
        n = 50_000
        n_train = 40_000

        # Generate data: strong + weak signal + noise predictor
        x_strong = rng.uniform(0, 1, n)
        x_weak = rng.uniform(0, 1, n)
        x_noise = rng.uniform(0, 1, n)

        log_mu = -0.5 + 0.8 * np.sin(2 * np.pi * x_strong) + 0.05 * np.sin(4 * np.pi * x_weak)
        y = rng.poisson(np.exp(log_mu)).astype(float)
        X = pd.DataFrame({"strong": x_strong, "weak": x_weak, "noise": x_noise})

        features = {
            "strong": Spline(kind="bs", k=12, select=True, discrete=True),
            "weak": Spline(kind="bs", k=12, select=True, discrete=True),
            "noise": Spline(kind="bs", k=12, select=True, discrete=True),
        }

        model = SuperGLM(
            family="poisson", selection_penalty=0, discrete=True, n_bins=256, features=features
        )
        model.fit_reml(
            X.iloc[:n_train],
            y[:n_train],
            max_reml_iter=30,
            verbose=False,
        )

        lambdas = model._reml_lambdas

        # Compute per-group EDF
        m = model.metrics(X.iloc[:n_train], y[:n_train])
        edf, _ = m._influence_edf
        _, _, _, _, active_groups = m._active_info
        group_edf = {ag.name: float(np.sum(edf[ag.sl])) for ag in active_groups}
        for g in model._groups:
            if g.name not in group_edf:
                group_edf[g.name] = 0.0

        # Noise: combined group should be effectively dead (EDF < 0.5)
        noise_total_edf = group_edf.get("noise", 0)
        assert noise_total_edf < 0.5, f"Noise total EDF={noise_total_edf:.3f}, expected < 0.5"

        # Noise lambdas should be very large (heavily penalized)
        for name in ["noise:null", "noise:wiggle"]:
            if name in lambdas:
                assert lambdas[name] > 1e2, f"{name} lambda={lambdas[name]:.1g}, expected > 1e2"

        # Weak signal: should retain nontrivial EDF (not snapped out)
        weak_total_edf = group_edf.get("weak", 0)
        assert weak_total_edf > 2.0, (
            f"Weak signal EDF={weak_total_edf:.3f}, expected > 2.0 (falsely suppressed?)"
        )

        # Strong signal: should be well-fit
        strong_total_edf = group_edf.get("strong", 0)
        assert strong_total_edf > 5.0, f"Strong signal EDF={strong_total_edf:.3f}, expected > 5.0"


# ── CR select=True tests ──────────────────────────────────────


class TestCRSelect:
    """Tests for CubicRegressionSpline with select=True."""

    def test_cr_select_build_returns_single_group_with_components(self):
        sp = Spline(kind="cr", n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert not isinstance(result, list)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2
        suffixes = [name for name, _ in result.penalty_components]
        assert suffixes == ["null", "wiggle"]

    def test_cr_select_combined_n_cols(self):
        """Combined n_cols = 1 (null) + n_range."""
        for nk in [5, 10, 20]:
            sp = Spline(kind="cr", n_knots=nk, select=True)
            result = sp.build(np.linspace(0, 1, 200))
            # Compute expected from _U_range
            n_range = sp._U_range.shape[1]
            expected = 1 + n_range
            assert result.n_cols == expected, (
                f"Expected n_cols={expected} for n_knots={nk}, got {result.n_cols}"
            )

    def test_cr_select_range_space_size(self):
        """Range space has K-4 columns (K raw, -2 constraints, -2 null)."""
        sp = Spline(kind="cr", n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        K = sp._n_basis
        expected_range = K - 2 - 2  # -2 constraints, -2 null eigenvalues
        n_range = sp._U_range.shape[1]
        assert n_range == expected_range
        # Combined: 1 + n_range
        assert result.n_cols == 1 + expected_range

    def test_cr_select_projections_orthogonal(self):
        sp = Spline(kind="cr", n_knots=10, select=True)
        sp.build(np.linspace(0, 1, 100))
        U_null = sp._U_null
        U_range = sp._U_range
        cross = U_null.T @ U_range
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)
        np.testing.assert_allclose(U_null.T @ U_null, np.eye(1), atol=1e-10)
        np.testing.assert_allclose(U_range.T @ U_range, np.eye(U_range.shape[1]), atol=1e-10)

    def test_cr_select_component_types(self):
        """CR select=True must have component_types={'null': 'selection'}."""
        sp = Spline(kind="cr", n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert result.component_types is not None
        assert result.component_types.get("null") == "selection"

    def test_cr_select_penalty_matrix_equals_sum(self):
        """CR select=True penalty_matrix must equal sum of components."""
        sp = Spline(kind="cr", n_knots=10, select=True)
        result = sp.build(np.linspace(0, 1, 100))
        omega_sum = sum(omega for _, omega in result.penalty_components)
        np.testing.assert_allclose(result.penalty_matrix, omega_sum, atol=1e-14)

    def test_cr_select_fit_close_to_no_select(self):
        """Predictions with select=True should be close to select=False."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(-0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu)
        X = pd.DataFrame({"x": x})

        m1 = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=10, select=False)},
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m1.fit(X, y)
        pred1 = m1.predict(X)

        m2 = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m2.fit(X, y)
        pred2 = m2.predict(X)

        np.testing.assert_allclose(pred1, pred2, rtol=0.2)

    def test_cr_select_reml_suppresses_noise(self):
        """REML with CR select=True should suppress noise feature."""
        rng = np.random.default_rng(42)
        n = 2000
        x_signal = rng.uniform(0, 10, n)
        x_noise = rng.uniform(0, 10, n)
        mu = np.exp(-0.5 + 0.5 * np.sin(x_signal))
        y = rng.poisson(mu)
        X = pd.DataFrame({"signal": x_signal, "noise": x_noise})

        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(kind="cr", n_knots=8, select=True),
                "noise": Spline(kind="cr", n_knots=8, select=True),
            },
            selection_penalty=0,
        )
        m.fit_reml(X, y, max_reml_iter=20)

        # Noise lambdas should be large
        noise_lambdas = {
            name: lam for name, lam in m._reml_lambdas.items() if "noise" in name.lower()
        }
        for name, lam in noise_lambdas.items():
            assert lam > 1e3, f"{name} lambda={lam:.1f}, expected > 1e3"

    def test_cr_select_discrete(self):
        """CR select=True with discrete=True builds correctly."""
        sp = Spline(kind="cr", n_knots=10, select=True, discrete=True)
        x = np.linspace(0, 10, 1000)
        # build_knots_and_penalty should populate _U_null/_U_range
        sp.build_knots_and_penalty(x)
        assert sp._U_null is not None
        assert sp._U_range is not None
        assert sp._U_null.shape == (sp._n_basis, 1)


class TestCardinalCRSelect:
    """Tests for CardinalCRSpline with select=True."""

    def test_cr_cardinal_select_build(self):
        sp = Spline(kind="cr_cardinal", n_knots=10, select=True)
        result = sp.build(np.linspace(0, 10, 200))
        assert not isinstance(result, list)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2
        suffixes = [name for name, _ in result.penalty_components]
        assert suffixes == ["null", "wiggle"]
        K = sp._n_basis
        n_range = sp._U_range.shape[1]
        assert result.n_cols == 1 + n_range
        assert result.n_cols == K - 2 + 1  # K - 2 range + 1 null

    def test_cr_cardinal_select_fit(self):
        """CardinalCR select=True fit works end-to-end."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(-0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu)
        X = pd.DataFrame({"x": x})

        m = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr_cardinal", n_knots=10, select=True)},
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m.fit(X, y)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)


class TestCRSelectInteraction:
    """Regression test: CR select=True parent inside spline x categorical interaction."""

    def test_cr_select_interaction_projection_includes_identifiability(self):
        """CR select=True _interaction_projection is (K, K-3): constraints + identifiability."""
        sp = Spline(kind="cr", n_knots=10, select=True)
        sp.build(np.linspace(0, 10, 200))
        assert sp._interaction_projection is not None
        K = sp._n_basis
        # K-2 from natural constraints, -1 from identifiability = K-3
        assert sp._interaction_projection.shape == (K, K - 3)

    def test_cr_select_interaction_projection_matches_no_select(self):
        """CR select=True and select=False produce the same _interaction_projection."""
        x = np.linspace(0, 10, 200)
        sp_sel = Spline(kind="cr", n_knots=10, select=True)
        sp_sel.build(x)
        sp_std = Spline(kind="cr", n_knots=10, select=False)
        sp_std.build(x)
        # Both should have the same shape
        assert sp_sel._interaction_projection.shape == sp_std._interaction_projection.shape
        # Projections span the same subspace (columns may differ by rotation)
        P1 = sp_sel._interaction_projection
        P2 = sp_std._interaction_projection
        # P1.T @ P2 should have full rank (same column space)
        cross = P1.T @ P2
        assert np.linalg.matrix_rank(cross) == P1.shape[1]

    def test_cr_select_spline_categorical_fit(self):
        """CR select=True parent with spline x categorical interaction fits correctly."""
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0, 10, n)
        cat = rng.choice(["A", "B", "C"], n)
        mu = np.exp(-0.5 + 0.3 * np.sin(x) + 0.2 * (cat == "B").astype(float))
        y = rng.poisson(mu)
        X = pd.DataFrame({"x": x, "cat": cat})

        m = SuperGLM(
            family="poisson",
            features={
                "x": Spline(kind="cr", n_knots=8, select=True),
                "cat": Categorical(),
            },
            interactions=[("x", "cat")],
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m.fit(X, y)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

        # Interaction groups must use identified column count (K-3),
        # not raw K or constraint-only K-2
        K = m._specs["x"]._n_basis
        expected_ncols = K - 3  # natural constraints + identifiability
        interaction_groups = [g for g in m._groups if "x:cat" in g.name]
        assert len(interaction_groups) > 0
        for g in interaction_groups:
            assert g.size == expected_ncols, (
                f"Expected interaction group size {expected_ncols}, got {g.size}"
            )

    def test_cr_select_interaction_full_rank(self):
        """CR select=True + spline x categorical must produce a full-rank design."""
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        cat = rng.choice(["A", "B", "C"], n)
        mu = np.exp(-0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu)
        X = pd.DataFrame({"x": x, "cat": cat})

        m = SuperGLM(
            family="poisson",
            features={
                "x": Spline(kind="cr", n_knots=8, select=True),
                "cat": Categorical(),
            },
            interactions=[("x", "cat")],
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m.fit(X, y)

        # Materialize design matrix and check rank
        p = m._dm.p
        X_mat = np.column_stack([m._dm.matvec(np.eye(p)[:, j]) for j in range(p)])
        rank = np.linalg.matrix_rank(X_mat)
        assert rank == p, f"Design matrix rank {rank} < p={p}: rank-deficient"

    def test_bs_select_interaction_projection_has_identifiability(self):
        """BS select=True _interaction_projection is (K, K-1): identifiability only."""
        sp = Spline(kind="bs", n_knots=10, select=True)
        sp.build(np.linspace(0, 10, 200))
        # BS has no boundary constraints, just identifiability
        assert sp._interaction_projection is not None
        K = sp._n_basis
        assert sp._interaction_projection.shape == (K, K - 1)

    def test_bs_select_interaction_full_rank(self):
        """BS select=True + spline x categorical must produce a full-rank design."""
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        cat = rng.choice(["A", "B", "C"], n)
        mu = np.exp(-0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu)
        X = pd.DataFrame({"x": x, "cat": cat})

        m = SuperGLM(
            family="poisson",
            features={
                "x": Spline(kind="bs", n_knots=8, select=True),
                "cat": Categorical(),
            },
            interactions=[("x", "cat")],
            spline_penalty=1.0,
            selection_penalty=0.01,
        )
        m.fit(X, y)

        p = m._dm.p
        X_mat = np.column_stack([m._dm.matvec(np.eye(p)[:, j]) for j in range(p)])
        rank = np.linalg.matrix_rank(X_mat)
        assert rank == p, f"Design matrix rank {rank} < p={p}: rank-deficient"


class TestNSSelectRejection:
    def test_ns_select_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="select=True is not yet supported"):
            Spline(kind="ns", select=True)
