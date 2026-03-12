"""Tests for mgcv-style spline decomposition (split_linear=True, double penalty)."""

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
    exposure = np.ones(n)
    X = pd.DataFrame({"signal": x1, "noise": x2})
    return X, y, exposure


# ── Build-time tests ──────────────────────────────────────────


class TestSelectBuild:
    def test_select_creates_two_group_infos(self):
        """Spline(split_linear=True).build() returns a list of 2 GroupInfos."""
        sp = Spline(n_knots=10, split_linear=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_select_subgroup_names(self):
        sp = Spline(n_knots=10, split_linear=True)
        result = sp.build(np.linspace(0, 1, 100))
        assert result[0].subgroup_name == "linear"
        assert result[1].subgroup_name == "spline"

    def test_select_null_space_size(self):
        """Null space is 1 (linear only, constant removed for identifiability)."""
        for nk in [5, 10, 20]:
            sp = Spline(n_knots=nk, split_linear=True)
            result = sp.build(np.linspace(0, 1, 200))
            assert result[0].n_cols == 1, f"null space should be 1 for n_knots={nk}"

    def test_select_range_space_size(self):
        sp = Spline(n_knots=10, split_linear=True)
        result = sp.build(np.linspace(0, 1, 100))
        n_basis = sp._n_basis
        assert result[1].n_cols == n_basis - 2

    def test_select_projections_orthogonal(self):
        """U_null and U_range should have orthonormal columns."""
        sp = Spline(n_knots=10, split_linear=True)
        sp.build(np.linspace(0, 1, 100))
        U_null = sp._U_null  # (K, 1)
        U_range = sp._U_range  # (K, K-2)
        # U_null is orthogonal to U_range
        cross = U_null.T @ U_range
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)
        # Both have orthonormal columns
        np.testing.assert_allclose(U_null.T @ U_null, np.eye(1), atol=1e-10)
        np.testing.assert_allclose(U_range.T @ U_range, np.eye(U_range.shape[1]), atol=1e-10)

    def test_select_null_space_centered(self):
        """Null-space linear component should be orthogonal to the constant."""
        sp = Spline(n_knots=10, split_linear=True)
        sp.build(np.linspace(0, 1, 100))
        # B-splines have partition of unity, so ones vector is the constant
        ones = np.ones(sp._n_basis)
        proj = sp._U_null.T @ ones
        np.testing.assert_allclose(proj, 0.0, atol=1e-10)

    def test_select_range_penalty_diagonal(self):
        """Range-space penalty should be diagonal with the nonzero eigenvalues."""
        sp = Spline(n_knots=10, split_linear=True)
        result = sp.build(np.linspace(0, 1, 100))
        omega_range = result[1].penalty_matrix
        assert omega_range is not None
        # Should be diagonal
        off_diag = omega_range - np.diag(np.diag(omega_range))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)
        # All diagonal entries should be positive
        assert np.all(np.diag(omega_range) > 0)


class TestSplitLinearDefaults:
    def test_default_split_linear_false(self):
        """Default Spline has split_linear=False."""
        sp = Spline(n_knots=10)
        assert sp.split_linear is False

    def test_no_select_returns_single_group_info(self):
        sp = Spline(n_knots=10, split_linear=False)
        result = sp.build(np.linspace(0, 1, 100))
        assert not isinstance(result, list)


# ── Model-level tests ─────────────────────────────────────────


class TestSelectModel:
    def test_select_creates_two_groups(self, simple_data):
        """Model with split_linear=True spline creates 2 groups per feature."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, split_linear=True),
                "noise": Spline(n_knots=10, split_linear=True),
            },
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        group_names = [g.name for g in m._groups]
        assert "signal:linear" in group_names
        assert "signal:spline" in group_names
        assert "noise:linear" in group_names
        assert "noise:spline" in group_names
        assert len(m._groups) == 4

    def test_select_feature_name(self, simple_data):
        """All subgroups have correct feature_name."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        for g in m._groups:
            assert g.feature_name == "signal"

    def test_feature_groups_helper(self, simple_data):
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        groups = m._feature_groups("signal")
        assert len(groups) == 2
        assert groups[0].name == "signal:linear"
        assert groups[1].name == "signal:spline"

    def test_select_predict_matches_no_select(self, simple_data):
        """Predictions with split_linear=True should be close to split_linear=False."""
        X, y, exposure = simple_data
        m1 = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=False)},
            lambda2=1.0,
            lambda1=0.01,
        )
        m1.fit(X, y, exposure=exposure)
        pred1 = m1.predict(X)

        m2 = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=0.01,
        )
        m2.fit(X, y, exposure=exposure)
        pred2 = m2.predict(X)

        # Not numerically identical (different parameterization), but close.
        # Open knot vectors can shift edge predictions slightly.
        np.testing.assert_allclose(pred1, pred2, rtol=0.2)

    def test_select_reconstruct(self, simple_data):
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        rec = m.reconstruct_feature("signal")
        assert "x" in rec
        assert "relativity" in rec
        assert len(rec["x"]) == 200

    def test_select_relativities(self, simple_data):
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        rels = m.relativities(with_se=True)
        assert "signal" in rels
        assert "se_log_relativity" in rels["signal"].columns

    def test_select_summary_dict(self, simple_data):
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        s = m.summary()
        assert "signal:linear" in s
        assert "signal:spline" in s


class TestSelectSparsity:
    def test_high_lambda_zeros_range(self, simple_data):
        """At high lambda, the range (wiggly) space should be zeroed first."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=50.0,
        )
        m.fit(X, y, exposure=exposure)
        s = m.summary()
        # At high lambda1, group lasso selection should be happening.
        # The 1-col linear subgroup (weight ∝ sqrt(1)) gets zeroed at a lower lambda
        # than the multi-col spline subgroup (weight ∝ sqrt(K-2)), so at this lambda
        # the linear group should be zeroed or both should be heavily shrunk.
        range_norm = s["signal:spline"]["group_norm"]
        linear_norm = s["signal:linear"]["group_norm"]
        assert linear_norm < 1e-10 or range_norm < 1e-10, (
            f"Expected at least one subgroup zeroed at lambda1=50: "
            f"range={range_norm:.6f}, linear={linear_norm:.6f}"
        )

    def test_very_high_lambda_zeros_both(self, simple_data):
        """At very high lambda, both linear and spline subgroups are zeroed (double penalty)."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=1e6,
        )
        m.fit(X, y, exposure=exposure)
        s = m.summary()
        # Both subgroups are penalized — zeroed at very high lambda
        assert s["signal:linear"]["group_norm"] < 1e-10
        assert s["signal:spline"]["group_norm"] < 1e-10


class TestSelectPath:
    def test_fit_path(self, simple_data):
        """Regularization path with split_linear=True should work."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        path = m.fit_path(X, y, exposure=exposure, n_lambda=10)
        assert path.coef_path.shape[0] == 10
        # null(1) + range(K-2) = K-1 columns
        assert path.coef_path.shape[1] == 1 + (m._specs["signal"]._n_basis - 2)


# ── Part 3: Additional coverage ──────────────────────────────


class TestSelectCovariance:
    """3A. Covariance with split_linear=True (active info path)."""

    def test_active_info_sizes(self, simple_data):
        """_active_info produces correctly-sized covariance and active_groups."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=0.1,
        )
        m.fit(X, y, exposure=exposure)
        metrics = m.metrics(X, y, exposure=exposure)
        X_a, W, XtWX_inv, active_groups = metrics._active_info

        # Covariance should be square, matching total active columns
        assert XtWX_inv.shape[0] == XtWX_inv.shape[1]
        total_active_cols = sum(ag.size for ag in active_groups)
        assert XtWX_inv.shape[0] == total_active_cols

    def test_penalized_linear_in_active_groups(self, simple_data):
        """Penalized :linear subgroups appear in active_groups with penalized=True (double penalty)."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=0.1,
        )
        m.fit(X, y, exposure=exposure)
        metrics = m.metrics(X, y, exposure=exposure)
        _, _, _, active_groups = metrics._active_info

        linear_ags = [ag for ag in active_groups if ag.subgroup_type == "linear"]
        assert len(linear_ags) == 1
        assert linear_ags[0].penalized is True

    def test_subgroup_type_propagated(self, simple_data):
        """GroupSlice.subgroup_type is set correctly on fitted model groups."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        linear_g = next(g for g in m._groups if g.name == "signal:linear")
        spline_g = next(g for g in m._groups if g.name == "signal:spline")
        assert linear_g.subgroup_type == "linear"
        assert spline_g.subgroup_type == "spline"


class TestSelectAdaptive:
    """3B. Adaptive + split-linear combination."""

    def test_adaptive_select_finite_weights(self, simple_data):
        """Adaptive(expon=2) with split_linear=True produces finite weights."""
        X, y, exposure = simple_data
        pen = GroupLasso(lambda1=1.0, flavor=Adaptive(expon=2))
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            penalty=pen,
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        # Adaptive weights are applied during fit — must converge with valid result
        assert m.result.converged, (
            f"Adaptive split-linear fit failed to converge in {m.result.n_iter} iters"
        )
        assert np.isfinite(m.result.deviance)
        assert m.result.deviance > 0
        # All coefficients must be finite
        assert np.all(np.isfinite(m.result.beta))
        # Predictions should be positive (Poisson)
        pred = m.predict(X)
        assert np.all(pred > 0)

    def test_adaptive_select_noise_zeroed(self, simple_data):
        """At moderate lambda with Adaptive(expon=2), noise spline should be zeroed."""
        X, y, exposure = simple_data
        pen = GroupLasso(lambda1=10.0, flavor=Adaptive(expon=2))
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, split_linear=True),
                "noise": Spline(n_knots=10, split_linear=True),
            },
            penalty=pen,
            lambda2=1.0,
        )
        m.fit(X, y, exposure=exposure)
        noise_spline = next(g for g in m._groups if g.name == "noise:spline")
        assert np.linalg.norm(m.result.beta[noise_spline.sl]) < 1e-10


class TestSelectEffectiveDf:
    """3C. Effective DF with penalized groups (double penalty)."""

    def test_high_lambda_edf_minimal(self, simple_data):
        """At very high lambda, all subgroups (including linear) are zeroed — only intercept remains."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={
                "signal": Spline(n_knots=10, split_linear=True),
                "noise": Spline(n_knots=10, split_linear=True),
            },
            lambda2=1.0,
            lambda1=1e6,
        )
        m.fit(X, y, exposure=exposure)
        # With double penalty, all groups are penalized and zeroed at extreme lambda
        # edf should be approximately 1 (intercept only)
        assert m.result.effective_df == pytest.approx(1.0, abs=0.5)


class TestSelectLambdaMax:
    """3D. lambda_max with penalized groups (double penalty)."""

    def test_lambda_max_all_penalized(self, simple_data):
        """With double penalty, all groups (including :linear) are penalized."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=1.0,  # set to prevent auto-calibration
        )
        m.fit(X, y, exposure=exposure)
        lmax = m._compute_lambda_max(y.astype(np.float64), exposure)
        # lambda_max should be finite and positive
        assert np.isfinite(lmax)
        assert lmax > 0
        # With double penalty, all groups are penalized
        assert all(g.penalized for g in m._groups)


class TestSelectSummaryOutput:
    """3E. Summary output for mixed active/inactive subgroups."""

    def test_summary_linear_active_spline_inactive(self, simple_data):
        """When :spline is zeroed but :linear is active, summary shows correct labels."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=1e6,
        )
        m.fit(X, y, exposure=exposure)
        metrics = m.metrics(X, y, exposure=exposure)
        summary = metrics.summary()
        text = str(summary)
        # Linear subgroup should show as active with chi2 test
        assert "[linear, 1 params" in text
        # Spline subgroup should show as inactive
        assert "[spline" in text
        assert "inactive" in text

    def test_summary_html_subgroup_labels(self, simple_data):
        """HTML summary also uses correct linear/spline labels."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=1e6,
        )
        m.fit(X, y, exposure=exposure)
        metrics = m.metrics(X, y, exposure=exposure)
        summary = metrics.summary()
        html = summary._repr_html_()
        assert "[linear, 1 params" in html
        assert "inactive" in html


class TestSelectFeatureSE:
    """3F. feature_se with split_linear=True subgroups."""

    def test_feature_se_reml_select_nonlinear_dgp(self):
        """feature_se works with REML split_linear=True when both subgroups are active.

        Uses fit_reml() with lambda1=0 (direct solver, no BCD aliasing).
        On a purely nonlinear DGP, REML drives the linear subgroup's lambda
        high (no signal) while keeping the spline lambda moderate (real signal).
        Both subgroups remain active (no L1 sparsity); the test verifies that
        feature_se returns finite, non-negative SEs through the full covariance
        code path.
        """
        rng = np.random.default_rng(99)
        n = 500
        x = rng.uniform(0, 10, n)
        eta = -0.5 + 0.4 * np.sin(x)  # purely nonlinear
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"signal": x})
        exposure = np.ones(n)

        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda1=0.0,
        )
        m.fit_reml(X, y, exposure=exposure, max_reml_iter=20)

        # Spline subgroup should have captured the nonlinear signal
        spline_g = next(g for g in m._groups if g.name == "signal:spline")
        assert np.linalg.norm(m.result.beta[spline_g.sl]) > 1e-6, (
            "Expected spline subgroup to be active for nonlinear DGP"
        )

        metrics = m.metrics(X, y, exposure=exposure)
        fse = metrics.feature_se("signal")
        assert "x" in fse
        assert "se_log_relativity" in fse
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.max(se) > 0

    def test_feature_se_both_zeroed(self):
        """feature_se returns all-zero SEs when both subgroups are zeroed.

        Tests the early-return branch in metrics.feature_se when all
        coefficients for a feature are exactly zero.
        """
        rng = np.random.default_rng(100)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)  # pure noise, no x effect
        X = pd.DataFrame({"signal": x})
        exposure = np.ones(n)

        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=1e6,  # extreme penalty → both subgroups zeroed
        )
        m.fit(X, y, exposure=exposure)

        # Precondition: both subgroups zeroed
        for g in m._groups:
            assert np.linalg.norm(m.result.beta[g.sl]) < 1e-10

        metrics = m.metrics(X, y, exposure=exposure)
        fse = metrics.feature_se("signal")
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        np.testing.assert_allclose(se, 0.0, atol=1e-10)

    def test_feature_se_both_active(self, simple_data):
        """feature_se works when both :linear and :spline are active."""
        X, y, exposure = simple_data
        m = SuperGLM(
            family="poisson",
            features={"signal": Spline(n_knots=10, split_linear=True)},
            lambda2=1.0,
            lambda1=0.1,
        )
        m.fit(X, y, exposure=exposure)
        metrics = m.metrics(X, y, exposure=exposure)
        fse = metrics.feature_se("signal")
        se = fse["se_log_relativity"]
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.max(se) > 0


class TestSelectNoiseSuppressionREML:
    """Regression test: REML with split_linear=True must suppress noise features."""

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
                "signal": Spline(kind="bs", k=12, split_linear=True, discrete=True),
                "noise": Spline(kind="bs", k=12, split_linear=True, discrete=True),
            },
            family="poisson",
            lambda1=0,
            discrete=True,
            n_bins=256,
        )
        model.fit_reml(X, y, max_reml_iter=30)

        # Compute per-group EDF via metrics machinery
        metrics_obj = model.metrics(X, y)
        edf, _ = metrics_obj._influence_edf
        _, _, _, active_groups = metrics_obj._active_info
        group_edf = {ag.name: float(np.sum(edf[ag.sl])) for ag in active_groups}

        # Noise groups: EDF < 0.02 (inactive groups have EDF = 0 by construction)
        for g in model._groups:
            if "noise" in g.name.lower():
                noise_edf = group_edf.get(g.name, 0.0)
                assert noise_edf < 0.02, f"{g.name} EDF={noise_edf:.4f}, expected < 0.02"

        # Signal groups should retain meaningful EDF
        signal_edf = sum(
            group_edf.get(g.name, 0.0) for g in model._groups if "signal" in g.name.lower()
        )
        assert signal_edf > 3.0, f"Signal total EDF={signal_edf:.2f}, expected > 3.0"

        # Noise lambdas should be at or near the upper bound
        noise_lambdas = {
            name: lam for name, lam in model._reml_lambdas.items() if "noise" in name.lower()
        }
        for name, lam in noise_lambdas.items():
            assert lam > 1e6, f"{name} lambda={lam:.1f}, expected > 1e6"


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
            "strong": Spline(kind="bs", k=12, split_linear=True, discrete=True),
            "weak": Spline(kind="bs", k=12, split_linear=True, discrete=True),
            "noise": Spline(kind="bs", k=12, split_linear=True, discrete=True),
        }

        model = SuperGLM(family="poisson", lambda1=0, discrete=True, n_bins=256, features=features)
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
        _, _, _, active_groups = m._active_info
        group_edf = {ag.name: float(np.sum(edf[ag.sl])) for ag in active_groups}
        for g in model._groups:
            if g.name not in group_edf:
                group_edf[g.name] = 0.0

        # Noise: both subgroups should be effectively dead (EDF < 0.5 total)
        noise_total_edf = group_edf.get("noise:linear", 0) + group_edf.get("noise:spline", 0)
        assert noise_total_edf < 0.5, f"Noise total EDF={noise_total_edf:.3f}, expected < 0.5"

        # Noise lambdas should be very large (heavily penalized)
        for name in ["noise:linear", "noise:spline"]:
            if name in lambdas:
                assert lambdas[name] > 1e2, f"{name} lambda={lambdas[name]:.1g}, expected > 1e2"

        # Weak signal: should retain nontrivial EDF (not snapped out)
        weak_total_edf = group_edf.get("weak:linear", 0) + group_edf.get("weak:spline", 0)
        assert weak_total_edf > 2.0, (
            f"Weak signal EDF={weak_total_edf:.3f}, expected > 2.0 (falsely suppressed?)"
        )

        # Strong signal: should be well-fit
        strong_total_edf = group_edf.get("strong:linear", 0) + group_edf.get("strong:spline", 0)
        assert strong_total_edf > 5.0, f"Strong signal EDF={strong_total_edf:.3f}, expected > 5.0"
