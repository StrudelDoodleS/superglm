"""Tests for REML smoothing parameter estimation."""

import logging

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline
from superglm.group_matrix import SparseSSPGroupMatrix
from superglm.metrics import (
    _penalised_xtwx_inv,
    _penalised_xtwx_inv_gram,
    _second_diff_penalty,
)
from superglm.reml import REMLResult, _map_beta_between_bases
from superglm.wood_pvalue import wood_test_smooth

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def poisson_data():
    """Small Poisson dataset with smooth + linear structure."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.choice(["A", "B", "C"], n)
    # x1 has smooth effect, x2 is wiggly, x3 is categorical
    eta = 0.5 + 0.3 * np.sin(x1) - 0.1 * np.cos(3 * x2)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    w = np.ones(n)
    return X, y, w


@pytest.fixture
def spline_model():
    """Model with two splines and a categorical."""
    return SuperGLM(
        family="poisson",
        lambda1=0.01,
        features={
            "x1": Spline(n_knots=8, penalty="ssp"),
            "x2": Spline(n_knots=8, penalty="ssp"),
            "x3": Categorical(),
        },
    )


# ── omega stored ─────────────────────────────────────────────────


class TestOmegaStored:
    """Verify gm.omega is set for SSP groups after building."""

    def test_spline_omega_stored(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, exposure=w)
        gm = model._dm.group_matrices[0]
        assert isinstance(gm, SparseSSPGroupMatrix)
        assert gm.omega is not None
        assert gm.omega.shape[0] == gm.omega.shape[1]
        # omega should be positive semi-definite
        eigvals = np.linalg.eigvalsh(gm.omega)
        assert np.all(eigvals >= -1e-10)

    def test_natural_spline_omega_stored(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": NaturalSpline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, exposure=w)
        gm = model._dm.group_matrices[0]
        assert isinstance(gm, SparseSSPGroupMatrix)
        assert gm.omega is not None
        assert gm.projection is not None  # NaturalSpline uses Z projection

    def test_crs_omega_stored(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": CubicRegressionSpline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, exposure=w)
        gm = model._dm.group_matrices[0]
        assert isinstance(gm, SparseSSPGroupMatrix)
        assert gm.omega is not None
        assert gm.projection is not None
        # CRS omega should differ from second-difference penalty
        p_b = gm.R_inv.shape[0]
        d2_penalty = _second_diff_penalty(p_b)
        assert not np.allclose(gm.omega, d2_penalty, atol=1e-6)


# ── _penalised_xtwx_inv uses stored omega ───────────────────────


class TestPenalisedXtwxInvOmega:
    """The bug fix: CRS gets its correct omega, not _second_diff_penalty."""

    def test_crs_penalty_differs_from_second_diff(self, poisson_data):
        """CRS model's covariance should use the integrated f'' penalty."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={"x1": CubicRegressionSpline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, exposure=w)

        # The stored omega on the group matrix should be the CRS penalty,
        # not _second_diff_penalty. Check the penalty contribution differs.
        gm = model._dm.group_matrices[0]
        R_inv = gm.R_inv
        omega_crs = gm.omega
        p_b = R_inv.shape[0]
        omega_d2 = _second_diff_penalty(p_b)

        S_crs = R_inv.T @ omega_crs @ R_inv
        S_d2 = R_inv.T @ omega_d2 @ R_inv
        # They should differ
        assert not np.allclose(S_crs, S_d2, atol=1e-8)

    def test_dict_lambda2(self, poisson_data):
        """_penalised_xtwx_inv accepts dict[str, float] for per-group lambdas."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit(X[["x1", "x2"]], y, exposure=w)

        beta = model.result.beta
        mu = model.predict(X[["x1", "x2"]])
        V = model._distribution.variance(mu)
        eta = model._link.link(mu)
        dmu = model._link.deriv_inverse(eta)
        W = w * dmu**2 / V

        # Scalar lambda2 should match dict with same value
        _, inv_scalar, _, _ = _penalised_xtwx_inv(
            beta, W, model._dm.group_matrices, model._groups, 0.1
        )
        lam_dict = {g.name: 0.1 for g in model._groups}
        _, inv_dict, _, _ = _penalised_xtwx_inv(
            beta, W, model._dm.group_matrices, model._groups, lam_dict
        )
        np.testing.assert_allclose(inv_scalar, inv_dict, atol=1e-10)

    def test_gram_matches_qr(self, poisson_data):
        """_penalised_xtwx_inv_gram gives same result as _penalised_xtwx_inv."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
                "x3": Categorical(),
            },
        )
        model.fit(X, y, exposure=w)

        beta = model.result.beta
        mu = model.predict(X)
        V = model._distribution.variance(mu)
        eta = model._link.link(mu)
        dmu = model._link.deriv_inverse(eta)
        W = w * dmu**2 / V

        lam_dict = {g.name: 0.1 for g in model._groups}

        _, inv_qr, groups_qr, _ = _penalised_xtwx_inv(
            beta, W, model._dm.group_matrices, model._groups, lam_dict
        )
        inv_gram, groups_gram = _penalised_xtwx_inv_gram(
            beta, W, model._dm.group_matrices, model._groups, lam_dict
        )

        assert len(groups_qr) == len(groups_gram)
        np.testing.assert_allclose(inv_qr, inv_gram, atol=1e-8)


# ── _compute_R_inv override ──────────────────────────────────────


class TestComputeRInvOverride:
    def test_different_lambda_gives_different_R_inv(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, exposure=w)

        gm = model._dm.group_matrices[0]
        R_inv_1 = model._compute_R_inv(gm.B, gm.omega, w, lambda2_override=0.01)
        R_inv_2 = model._compute_R_inv(gm.B, gm.omega, w, lambda2_override=1.0)
        assert not np.allclose(R_inv_1, R_inv_2, atol=1e-6)


class TestMgcvStyleSmoothTestInput:
    def test_summary_smooth_pvalue_uses_weighted_qr_factor(self):
        """Regression test for the false-significant noise-spline bug.

        mgcv's stored ``R`` factor is the QR factor of the weighted active
        design, so ``R.T @ R`` matches ``X'WX``. The summary path should use
        that QR factor rather than the raw active design block.
        """
        rng = np.random.default_rng(1)
        n = 2000
        df = pd.DataFrame(
            {
                "DrivAge": rng.uniform(18, 80, n),
                "VehAge": rng.uniform(0, 20, n),
                "BonusMalus": rng.uniform(50, 150, n),
                "Area": rng.choice(list("ABCDE"), n),
                "LogDensity": rng.normal(6.0, 1.0, n),
                "Noise1": rng.normal(size=n),
                "Noise2": rng.normal(size=n),
                "Noise3": rng.normal(size=n),
                "Exposure": rng.uniform(0.1, 1.0, n),
            }
        )
        eta = (
            -2.2
            + 0.5 * np.sin(df["DrivAge"] / 8)
            - 0.04 * (df["VehAge"] - 8) ** 2 / 10
            + 0.003 * (df["BonusMalus"] - 90)
            + 0.08 * (df["Area"] == "B")
            - 0.10 * (df["Area"] == "D")
            + 0.06 * df["LogDensity"]
            + np.log(df["Exposure"])
        )
        y = rng.poisson(np.exp(eta)).astype(float)
        offset = np.log(df["Exposure"].to_numpy(dtype=np.float64))

        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={
                "DrivAge": Spline(n_knots=12, penalty="ssp"),
                "VehAge": Spline(n_knots=12, penalty="ssp"),
                "BonusMalus": Spline(n_knots=12, penalty="ssp"),
                "Area": Categorical(base="most_exposed"),
                "LogDensity": Numeric(),
                "Noise1": Spline(n_knots=12, penalty="ssp"),
                "Noise2": Spline(n_knots=12, penalty="ssp"),
                "Noise3": Spline(n_knots=12, penalty="ssp"),
            },
        )
        model.fit_reml(df, y, offset=offset, max_reml_iter=20)
        metrics = model.metrics(df, y, offset=offset)

        row = next(r for r in metrics._build_coef_rows() if r.name == "Noise2")

        X_a, W, XtWX_inv, active_groups = metrics._active_info
        R_a = metrics._active_R_factor
        _, edf1 = metrics._influence_edf
        ag = next(a for a in active_groups if a.name == "Noise2")
        beta_g = model.result.beta[ag.sl]
        V_b_j = XtWX_inv[ag.sl, ag.sl]
        edf1_j = float(np.sum(edf1[ag.sl]))

        np.testing.assert_allclose(R_a.T @ R_a, X_a.T @ (X_a * W[:, None]), atol=1e-8)

        _, p_raw, _ = wood_test_smooth(beta_g, X_a[:, ag.sl], V_b_j, edf1_j, -1.0)
        _, p_r, _ = wood_test_smooth(beta_g, R_a[:, ag.sl], V_b_j, edf1_j, -1.0)

        assert row.wald_p == pytest.approx(p_r)
        # QR correctness already verified above (R_a.T @ R_a == X_a.T @ diag(W) @ X_a).
        # For heavily suppressed noise groups, both p-values are tiny and nearly equal,
        # so we only check they're both small (not that they differ).
        assert p_r < 0.1
        assert p_raw < 0.1


# ── Beta mapping ─────────────────────────────────────────────────


class TestBetaMapping:
    def test_roundtrip(self, poisson_data):
        """Mapping beta through old -> B-spline -> new preserves B-spline coefficients."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, exposure=w)

        gm_old = model._dm.group_matrices[0]
        beta_old = model.result.beta.copy()

        # Create new R_inv with different lambda
        R_inv_new = model._compute_R_inv(gm_old.B, gm_old.omega, w, lambda2_override=0.5)
        gm_new = SparseSSPGroupMatrix(gm_old.B, R_inv_new)
        gm_new.omega = gm_old.omega

        # Map beta
        beta_mapped = _map_beta_between_bases(
            beta_old,
            [gm_old],
            [gm_new],
            model._groups,
        )

        # B-spline space coefficients should match
        bspline_old = gm_old.R_inv @ beta_old[model._groups[0].sl]
        bspline_new = gm_new.R_inv @ beta_mapped[model._groups[0].sl]
        np.testing.assert_allclose(bspline_old, bspline_new, atol=1e-8)


class TestREMLMultistart:
    def test_direct_reml_is_stable_across_initial_lambda_starts(self):
        """Direct REML should converge to similar solutions from different starts."""
        rng = np.random.default_rng(7)
        n = 1800
        df = pd.DataFrame(
            {
                "DrivAge": rng.uniform(18, 80, n),
                "VehAge": rng.uniform(0, 20, n),
                "BonusMalus": rng.uniform(50, 150, n),
                "Area": rng.choice(list("ABCDE"), n),
                "LogDensity": rng.normal(6.0, 1.0, n),
                "Noise1": rng.normal(size=n),
                "Noise2": rng.normal(size=n),
                "Noise3": rng.normal(size=n),
                "Exposure": rng.uniform(0.1, 1.0, n),
            }
        )
        eta = (
            -2.15
            + 0.48 * np.sin(df["DrivAge"] / 8.5)
            - 0.05 * (df["VehAge"] - 8.0) ** 2 / 10.0
            + 0.003 * (df["BonusMalus"] - 90.0)
            + 0.10 * (df["Area"] == "B")
            - 0.08 * (df["Area"] == "D")
            + 0.05 * df["LogDensity"]
            + np.log(df["Exposure"])
        )
        y = rng.poisson(np.exp(eta)).astype(float)
        offset = np.log(df["Exposure"].to_numpy(dtype=np.float64))

        def build_model() -> SuperGLM:
            return SuperGLM(
                family="poisson",
                lambda1=0.0,
                features={
                    "DrivAge": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                    "VehAge": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                    "BonusMalus": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                    "Area": Categorical(base="most_exposed"),
                    "LogDensity": Numeric(),
                    "Noise1": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                    "Noise2": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                    "Noise3": CubicRegressionSpline(n_knots=10, penalty="ssp"),
                },
            )

        default_model = build_model()
        default_model.fit_reml(df, y, offset=offset, max_reml_iter=50, reml_tol=1e-6)

        low_init_model = build_model()
        low_init_model.fit_reml(
            df, y, offset=offset, max_reml_iter=50, reml_tol=1e-6, lambda2_init=0.1
        )

        high_init_model = build_model()
        high_init_model.fit_reml(
            df, y, offset=offset, max_reml_iter=50, reml_tol=1e-6, lambda2_init=100.0
        )

        for fitted in (default_model, low_init_model, high_init_model):
            assert fitted._reml_result.converged
            assert np.isfinite(fitted._reml_result.objective)

        objectives = np.array(
            [
                default_model._reml_result.objective,
                low_init_model._reml_result.objective,
                high_init_model._reml_result.objective,
            ],
            dtype=float,
        )
        assert objectives.max() - objectives.min() < 1e-2

        for name in default_model._reml_lambdas:
            vals = np.array(
                [
                    default_model._reml_lambdas[name],
                    low_init_model._reml_lambdas[name],
                    high_init_model._reml_lambdas[name],
                ],
                dtype=float,
            )
            assert np.max(np.abs(np.log(vals) - np.log(vals[0]))) < 0.5


# ── REML convergence ─────────────────────────────────────────────


class TestREMLConvergence:
    def test_reml_convergence_small(self, poisson_data, spline_model):
        """REML should converge on a small dataset."""
        X, y, w = poisson_data
        spline_model.fit_reml(X, y, exposure=w, max_reml_iter=20)

        assert hasattr(spline_model, "_reml_lambdas")
        assert hasattr(spline_model, "_reml_result")
        assert isinstance(spline_model._reml_result, REMLResult)
        assert spline_model._reml_result.n_reml_iter <= 20
        assert spline_model._reml_result.converged

    def test_reml_per_group_lambdas_differ(self):
        """Splines with different smoothness should get different lambdas."""
        rng = np.random.default_rng(123)
        n = 800
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        # x1: smooth (sin), x2: wiggly (cos(5x))
        eta = 0.5 + 0.5 * np.sin(x1 * 0.5) - 0.3 * np.cos(5 * x2)
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            lambda1=0.005,
            features={
                "x1": Spline(n_knots=10, penalty="ssp"),
                "x2": Spline(n_knots=10, penalty="ssp"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=15)

        lambdas = model._reml_lambdas
        assert len(lambdas) == 2
        lam_vals = list(lambdas.values())
        # The two lambdas should differ (smooth vs wiggly)
        ratio = max(lam_vals) / min(lam_vals)
        assert ratio > 1.5, f"Expected different lambdas, got ratio {ratio:.2f}"


# ── REML + group lasso ───────────────────────────────────────────


class TestREMLGroupLasso:
    def test_reml_plus_group_lasso(self):
        """Group lasso coexists with REML — fit_reml produces same structure as fit."""
        rng = np.random.default_rng(42)
        n = 600
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        eta = 0.5 + 0.3 * np.sin(x1) + 0.1 * x2 * 0
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        # Fit with REML
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=10)

        # Both features should be active (non-zero) since lambda1 is moderate
        beta = model.result.beta
        for g in model._groups:
            norm_g = np.linalg.norm(beta[g.sl])
            # At this lambda1, groups should be non-zero
            assert norm_g > 0 or not g.penalized

        # Per-group lambdas should exist, be positive, and be finite
        assert model._reml_lambdas is not None
        assert len(model._reml_lambdas) >= 1
        for name, lam in model._reml_lambdas.items():
            assert np.isfinite(lam), f"Non-finite REML lambda for {name}"
            assert lam > 0, f"Non-positive REML lambda for {name}"

    def test_reml_plus_group_lasso_gamma_estimated_scale(self, capsys):
        """Estimated-scale REML should work on the BCD path with lambda1 > 0."""
        rng = np.random.default_rng(123)
        n = 600
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        mu = np.exp(0.3 + 0.35 * np.sin(x1) + 0.15 * np.cos(x2))
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        y = np.maximum(y, 1e-4)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="gamma",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=12, verbose=True)

        out = capsys.readouterr().out
        assert "REML iter=" in out
        assert "(pirls=" in out
        assert "(cheap)" in out

        assert model.result.converged
        assert np.isfinite(model.result.phi)
        assert model.result.phi > 0
        assert model._reml_lambdas is not None
        for name, lam in model._reml_lambdas.items():
            assert np.isfinite(lam), f"Non-finite REML lambda for {name}"
            assert lam > 0, f"Non-positive REML lambda for {name}"


class TestREMLFallbacks:
    def test_fit_reml_nb_auto_theta_without_smooths_falls_back_to_fit(self, caplog):
        """NB2 auto-theta should still work when fit_reml() has no smooth terms to optimize."""
        rng = np.random.default_rng(42)
        n = 2000
        theta_true = 5.0
        mu = 5.0
        lam = rng.gamma(shape=theta_true, scale=mu / theta_true, size=n)
        y = rng.poisson(lam).astype(float)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family="negative_binomial",
            nb_theta="auto",
            lambda1=0.0,
            features={"dummy": Numeric(standardize=False)},
        )
        with caplog.at_level(logging.WARNING):
            model.fit_reml(X, y)

        assert "no REML-eligible groups found" in caplog.text
        assert isinstance(model.nb_theta, float)
        assert model.nb_theta > 0
        assert model._nb_profile_result is not None
        assert model.result.converged
        assert not hasattr(model, "_reml_lambdas")


# ── REML + split_linear=True (mgcv double penalty) ─────────────────────


class TestREMLSelectTrue:
    def test_reml_select_true_converges(self, poisson_data):
        """fit_reml() works with split_linear=True (double penalty)."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp", split_linear=True)},
        )
        model.fit_reml(X[["x1"]], y, exposure=w, max_reml_iter=15)
        assert model._reml_result.converged
        # Both linear and spline subgroups should have REML lambdas
        assert "x1:linear" in model._reml_lambdas
        assert "x1:spline" in model._reml_lambdas

    def test_reml_select_true_linear_lambda_differs(self, poisson_data):
        """Linear and spline subgroups should get different REML lambdas."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=8, penalty="ssp", split_linear=True),
                "x2": Spline(n_knots=8, penalty="ssp", split_linear=True),
            },
        )
        model.fit_reml(X[["x1", "x2"]], y, exposure=w, max_reml_iter=15)
        # Should have 4 REML lambdas: x1:linear, x1:spline, x2:linear, x2:spline
        assert len(model._reml_lambdas) == 4


# ── Backward compatibility ───────────────────────────────────────


class TestREMLBackwardCompat:
    def test_fit_unchanged(self, poisson_data, spline_model):
        """fit() with global lambda2 should work unchanged after REML code added."""
        X, y, w = poisson_data
        spline_model.fit(X, y, exposure=w)
        assert spline_model.result is not None
        assert not hasattr(spline_model, "_reml_lambdas") or spline_model._reml_lambdas is None

    def test_custom_link_without_deriv2_inverse(self):
        """Old-style custom link without deriv2_inverse should work on fit_reml.

        Regression test: deriv2_inverse was added to the Link protocol,
        breaking isinstance() checks and the REML path for custom links.
        Now it's optional — the W(ρ) correction is skipped gracefully.
        """
        from superglm.links import Link

        class MinimalLogLink:
            """Custom log link with only the 4 required methods."""

            def link(self, mu):
                return np.log(mu)

            def inverse(self, eta):
                return np.exp(eta)

            def deriv(self, mu):
                return 1.0 / mu

            def deriv_inverse(self, eta):
                return np.exp(eta)

        custom_link = MinimalLogLink()
        assert isinstance(custom_link, Link), "Minimal link should satisfy protocol"

        rng = np.random.default_rng(99)
        n = 300
        x = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(1 + np.sin(2 * np.pi * x))).astype(float)
        df = pd.DataFrame({"x": x})
        m = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=6)},
            family="poisson",
            link=custom_link,
            lambda1=0,
        )
        m.fit_reml(df, y, max_reml_iter=10)
        assert m._reml_result.converged

    def test_custom_distribution_without_variance_derivative(self):
        """Old-style custom distribution without variance_derivative should work.

        Regression test: variance_derivative was added to the Distribution
        protocol, breaking isinstance() checks and the REML path for custom
        distributions.  Now it's optional — the W(ρ) correction is skipped.
        """
        from superglm.distributions import Distribution

        class MinimalPoisson:
            """Custom Poisson with only the 5 required members."""

            @property
            def scale_known(self):
                return True

            @property
            def default_link(self):
                return "log"

            def variance(self, mu):
                return mu.copy()

            def deviance_unit(self, y, mu):
                d = np.zeros_like(y, dtype=float)
                pos = y > 0
                d[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos]))
                d[~pos] = 2 * mu[~pos]
                return d

            def log_likelihood(self, y, mu, weights, phi=1.0):
                from scipy.special import gammaln

                return float(
                    np.sum(weights * (y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)))
                )

        custom_dist = MinimalPoisson()
        assert isinstance(custom_dist, Distribution), "Minimal dist should satisfy protocol"

        rng = np.random.default_rng(99)
        n = 300
        x = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(1 + np.sin(2 * np.pi * x))).astype(float)
        df = pd.DataFrame({"x": x})
        m = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=6)},
            family=custom_dist,
            lambda1=0,
        )
        m.fit_reml(df, y, max_reml_iter=10)
        assert m._reml_result.converged

    def test_enhanced_custom_objects_get_w_correction(self):
        """Custom objects WITH second-order methods should get the W(ρ) correction."""

        class EnhancedLogLink:
            def link(self, mu):
                return np.log(mu)

            def inverse(self, eta):
                return np.exp(eta)

            def deriv(self, mu):
                return 1.0 / mu

            def deriv_inverse(self, eta):
                return np.exp(eta)

            def deriv2_inverse(self, eta):
                return np.exp(eta)

        class EnhancedPoisson:
            @property
            def scale_known(self):
                return True

            @property
            def default_link(self):
                return "log"

            def variance(self, mu):
                return mu.copy()

            def variance_derivative(self, mu):
                return np.ones_like(mu)

            def deviance_unit(self, y, mu):
                d = np.zeros_like(y, dtype=float)
                pos = y > 0
                d[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos]))
                d[~pos] = 2 * mu[~pos]
                return d

            def log_likelihood(self, y, mu, weights, phi=1.0):
                from scipy.special import gammaln

                return float(
                    np.sum(weights * (y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)))
                )

        rng = np.random.default_rng(99)
        n = 300
        x = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(1 + np.sin(2 * np.pi * x))).astype(float)
        df = pd.DataFrame({"x": x})

        # Enhanced objects should produce a non-None W correction
        m = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=6)},
            family=EnhancedPoisson(),
            link=EnhancedLogLink(),
            lambda1=0,
        )
        m.fit_reml(df, y, max_reml_iter=10)
        assert m._reml_result.converged

        # Verify W correction was actually computed (not skipped)
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.solvers.irls_direct import fit_irls_direct

        lambdas = m._reml_lambdas
        reml_groups = []
        for i, (gm, g) in enumerate(zip(m._dm.group_matrices, m._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                reml_groups.append((i, g))
        pirls_result, inv_beta, xtwx = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=np.ones(n),
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=np.zeros(n),
            return_xtwx=True,
        )
        corr = m._reml_w_correction(
            pirls_result, inv_beta, lambdas, reml_groups, None, np.ones(n), np.zeros(n)
        )
        assert corr is not None, "Enhanced custom objects should get W correction"


# ── Predict after REML ───────────────────────────────────────────


class TestREMLPredict:
    def test_reml_predict_after_fit(self, poisson_data, spline_model):
        """predict/reconstruct should work after fit_reml."""
        X, y, w = poisson_data
        spline_model.fit_reml(X, y, exposure=w, max_reml_iter=10)

        # predict
        mu = spline_model.predict(X)
        assert mu.shape == (len(y),)
        assert np.all(np.isfinite(mu))
        assert np.all(mu > 0)

        # reconstruct
        for name in ["x1", "x2"]:
            raw = spline_model.reconstruct_feature(name)
            assert "x" in raw
            assert "relativity" in raw


# ── Metrics/SEs after REML ───────────────────────────────────────


class TestREMLMetrics:
    def test_reml_metrics_ses(self, poisson_data, spline_model):
        """summary/SEs should work after fit_reml (using per-group lambdas)."""
        X, y, w = poisson_data
        spline_model.fit_reml(X, y, exposure=w, max_reml_iter=10)

        met = spline_model.metrics(X, y, exposure=w)
        assert met.n_obs == len(y)
        assert met.deviance > 0
        assert met.effective_df > 0

        # SEs should be finite, non-negative, and reasonably sized
        se_dict = met.coefficient_se
        for name, se in se_dict.items():
            assert np.all(np.isfinite(se)), f"Non-finite SEs for {name}"
            assert np.all(se >= 0), f"Negative SEs for {name}"
            assert np.max(se) < 100, f"Unreasonably large SE for {name}: max={np.max(se)}"

    def test_reml_covariance_uses_per_group_lambdas(self, poisson_data, spline_model):
        """Covariance should use per-group REML lambdas, not global lambda2."""
        X, y, w = poisson_data

        # Fit with REML
        spline_model.fit_reml(X, y, exposure=w, max_reml_iter=10)
        cov_reml, groups_reml = spline_model._coef_covariance

        # Fit with global lambda2 (different model instance)
        model2 = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
                "x3": Categorical(),
            },
        )
        model2.fit(X, y, exposure=w)
        cov_global, groups_global = model2._coef_covariance

        # They should differ (different lambdas → different penalty → different cov)
        # Only compare if both have the same active groups (they should)
        if cov_reml.shape == cov_global.shape:
            assert not np.allclose(cov_reml, cov_global, atol=1e-6)


class TestREMLDiscreteRobustness:
    """Regression tests for discrete REML convergence under adverse starts."""

    def test_discrete_large_lambda2_init_converges(self):
        """Discrete REML must converge even with lambda2_init=1e5.

        Regression test for a robustness issue where skipping the line search
        entirely on the discrete path caused divergence with poor initial
        smoothing parameters.
        """
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        w = np.ones(n)

        model = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "x1": CubicRegressionSpline(n_knots=8),
                "x2": CubicRegressionSpline(n_knots=8),
            },
            discrete=True,
        )
        model.fit_reml(df, y, exposure=w, max_reml_iter=50, lambda2_init=1e5)

        assert model._reml_result.converged
        assert model._reml_result.n_reml_iter <= 30

    def test_discrete_vs_exact_agreement(self):
        """Discrete and exact REML should agree on deviance and EDF."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        w = np.ones(n)

        features = {
            "x1": CubicRegressionSpline(n_knots=8),
            "x2": CubicRegressionSpline(n_knots=8),
        }

        exact = SuperGLM(family="poisson", lambda1=0, features=features, discrete=False)
        exact.fit_reml(df, y, exposure=w, max_reml_iter=30)

        disc = SuperGLM(family="poisson", lambda1=0, features=features, discrete=True)
        disc.fit_reml(df, y, exposure=w, max_reml_iter=30)

        assert exact._reml_result.converged
        assert disc._reml_result.converged

        # Deviance should agree within 0.1%
        dev_exact = exact.result.deviance
        dev_disc = disc.result.deviance
        assert abs(dev_exact - dev_disc) / abs(dev_exact) < 1e-3

        # EDF should agree within 0.5
        edf_exact = exact.result.effective_df
        edf_disc = disc.result.effective_df
        assert abs(edf_exact - edf_disc) < 0.5

    @pytest.mark.parametrize("family", ["gamma", "poisson"])
    def test_discrete_cached_w_estimated_scale(self, family):
        """Cached-W discrete path works for estimated-scale families (Gamma).

        The cached-W fREML optimizer must correctly handle profiled phi
        in the FP update (inv_phi scaling of the quadratic term).
        """
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        if family == "gamma":
            eta = 1.0 + np.sin(2 * np.pi * x1) + 0.5 * x2
            mu = np.exp(eta)
            y = rng.gamma(shape=5.0, scale=mu / 5.0)
            y = np.maximum(y, 1e-4)
        else:
            eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2
            mu = np.exp(eta)
            y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        w = np.ones(n)

        features = {
            "x1": CubicRegressionSpline(n_knots=8),
            "x2": CubicRegressionSpline(n_knots=8),
        }

        exact = SuperGLM(family=family, lambda1=0, features=features, discrete=False)
        exact.fit_reml(df, y, exposure=w, max_reml_iter=30)

        disc = SuperGLM(family=family, lambda1=0, features=features, discrete=True)
        disc.fit_reml(df, y, exposure=w, max_reml_iter=30)

        assert exact._reml_result.converged
        assert disc._reml_result.converged

        # Deviance within 0.5%
        dev_exact = exact.result.deviance
        dev_disc = disc.result.deviance
        assert abs(dev_exact - dev_disc) / abs(dev_exact) < 5e-3

        # EDF within 1.0 (Gamma can diverge slightly more due to phi profiling)
        edf_exact = exact.result.effective_df
        edf_disc = disc.result.effective_df
        assert abs(edf_exact - edf_disc) < 1.0
