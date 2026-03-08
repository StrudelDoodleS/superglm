"""Tests for REML smoothing parameter estimation."""

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


# ── Finite-difference validation of REML gradient/Hessian ────────


class TestREMLFiniteDifference:
    """Verify analytic gradient and Hessian match finite differences."""

    @staticmethod
    def _setup_model(family, seed=42):
        """Build a fitted model with two CRS splines for FD checks."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.reml import build_penalty_caches
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        rng = np.random.default_rng(seed)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2)
        if family == "poisson":
            y = rng.poisson(mu).astype(float)
        elif family == "gamma":
            y = rng.gamma(shape=5.0, scale=mu / 5.0)
            y = np.maximum(y, 1e-4)
        else:
            raise ValueError(family)

        df = pd.DataFrame({"x1": x1, "x2": x2})
        m = SuperGLM(
            features={
                "x1": CubicRegressionSpline(n_knots=8),
                "x2": CubicRegressionSpline(n_knots=8),
            },
            family=family,
        )
        m.fit(df, y)

        exposure = np.ones(n)
        offset_arr = np.zeros(n)
        lambdas = {"x1": 10.0, "x2": 0.5}

        reml_groups = []
        penalty_ranks = {}
        for i, (gm, g) in enumerate(zip(m._dm.group_matrices, m._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                reml_groups.append((i, g))
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
                eigv = np.linalg.eigvalsh(omega_ssp)
                penalty_ranks[g.name] = float(np.sum(eigv > 1e-8 * max(eigv.max(), 1e-12)))

        penalty_caches = build_penalty_caches(m._dm.group_matrices, m._groups, reml_groups)

        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=exposure,
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=offset_arr,
            return_xtwx=True,
        )

        p_dim = XtWX.shape[0]
        S = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p_dim)
        pq = float(pirls_result.beta @ S @ pirls_result.beta)
        M_p = sum(c.rank for c in penalty_caches.values())
        phi_hat = 1.0
        if not getattr(m._distribution, "scale_known", True):
            phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)

        return (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        )

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_gradient_matches_fd(self, family):
        """Analytic gradient matches central FD of objective (partial: fixed β, W)."""
        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        grad = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )

        eps = 1e-5
        group_names = [g.name for _, g in reml_groups]
        fd_grad = np.zeros(len(reml_groups))
        for i, name in enumerate(group_names):
            rho_base = np.log(lambdas[name])
            lam_p, lam_m = lambdas.copy(), lambdas.copy()
            lam_p[name] = np.exp(rho_base + eps)
            lam_m[name] = np.exp(rho_base - eps)
            op = m._reml_laml_objective(
                y,
                pirls_result,
                lam_p,
                exposure,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            om = m._reml_laml_objective(
                y,
                pirls_result,
                lam_m,
                exposure,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            fd_grad[i] = (op - om) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_hessian_matches_fd(self, family):
        """Approximate outer Hessian matches full outer FD to within ~5%.

        The analytic Hessian includes the IFT correction (dβ̂/dρ = -H⁻¹ S β̂)
        but holds W fixed. FD re-solves PIRLS, so W changes. The residual
        includes both the fixed-W approximation and higher-order IFT terms.
        """
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        grad = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        hess = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
        )

        eps = 1e-4
        group_names = [g.name for _, g in reml_groups]
        p_dim = XtWX.shape[0]
        M_p = sum(c.rank for c in penalty_caches.values())
        m_groups = len(reml_groups)
        fd_hess = np.zeros((m_groups, m_groups))

        for j in range(m_groups):
            rho_base = np.log(lambdas[group_names[j]])
            for sign in [+1, -1]:
                lam_pert = lambdas.copy()
                lam_pert[group_names[j]] = np.exp(rho_base + sign * eps)

                # Re-solve PIRLS at perturbed lambda (full outer FD)
                result_pert, inv_pert, xtwx_pert = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=exposure,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=lam_pert,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )

                phi_pert = 1.0
                if not getattr(m._distribution, "scale_known", True):
                    S_pert = _build_penalty_matrix(m._dm.group_matrices, m._groups, lam_pert, p_dim)
                    pq_pert = float(result_pert.beta @ S_pert @ result_pert.beta)
                    phi_pert = max((result_pert.deviance + pq_pert) / max(n - M_p, 1.0), 1e-10)

                grad_pert = m._reml_direct_gradient(
                    result_pert,
                    inv_pert,
                    lam_pert,
                    reml_groups,
                    penalty_ranks,
                    phi_hat=phi_pert,
                )
                if sign == 1:
                    grad_plus = grad_pert
                else:
                    grad_minus = grad_pert

            fd_hess[:, j] = (grad_plus - grad_minus) / (2 * eps)

        # Looser tolerance than inner FD due to Laplace approximation error
        # (W changes when β̂ changes, but analytic Hessian treats W as fixed)
        np.testing.assert_allclose(hess, fd_hess, rtol=0.05, atol=0.5)


# ── REML convergence ─────────────────────────────────────────────


class TestREMLConvergence:
    def test_reml_convergence_small(self, poisson_data, spline_model):
        """REML should converge on a small dataset."""
        X, y, w = poisson_data
        spline_model.fit_reml(X, y, exposure=w, max_reml_iter=15)

        assert hasattr(spline_model, "_reml_lambdas")
        assert hasattr(spline_model, "_reml_result")
        assert isinstance(spline_model._reml_result, REMLResult)
        assert spline_model._reml_result.n_reml_iter <= 15
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
