"""Comprehensive tests for opt-in robust (sandwich) covariance on SuperGLM (#695 S1)."""

from __future__ import annotations

import dataclasses
import math
import pickle
import shutil
import subprocess
import tracemalloc

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import (
    Binomial,
    Categorical,
    CloglogLink,
    Constraint,
    Gamma,
    Gaussian,
    GroupElasticNet,
    GroupLasso,
    InverseLink,
    LogitLink,
    LogLink,
    NegativeBinomial,
    Numeric,
    OrderedCategorical,
    Piecewise,
    Poisson,
    ProbitLink,
    RandomEffect,
    Ridge,
    SandwichCovariance,
    Spline,
    SuperGLM,
)
from superglm.inference._sandwich_core import (
    SANDWICH_LIFECYCLE,
    UNAVAILABLE_REASONS,
    ClusterScoreAccumulator,
    build_unavailable_covariance,
)

# statsmodels comes with the bench extra, which CI installs; without it the module is skipped.
sm = pytest.importorskip("statsmodels.api")


def _cov_tol(cov_ref: np.ndarray, *, extra: float = 1e-7) -> float:
    """Conditioning-scaled matrix tolerance based on float64 unit roundoff."""
    u = float(np.finfo(np.float64).eps) / 2.0
    norm = float(np.linalg.norm(cov_ref, ord=2))
    cond = float(np.linalg.cond(cov_ref)) if cov_ref.size else 1.0
    if not math.isfinite(cond):
        cond = 1e6
    return max(extra * max(norm, 1.0), 64.0 * cond * u * max(norm, 1.0))


def _sm_glm_cov(
    X_design: np.ndarray,
    y: np.ndarray,
    family,
    *,
    offset: np.ndarray | None = None,
    var_weights: np.ndarray | None = None,
    freq_weights: np.ndarray | None = None,
    cov_type: str = "HC0",
    cluster: np.ndarray | None = None,
    use_correction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit statsmodels GLM and return (params, observed-Hessian sandwich covariance)."""
    import statsmodels.stats.sandwich_covariance as sw

    glm = sm.GLM(
        y,
        X_design,
        family=family,
        offset=offset,
        var_weights=var_weights,
        freq_weights=freq_weights,
    )
    if cov_type == "cluster":
        fit = glm.fit(
            tol=1e-14,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": cluster, "use_correction": use_correction},
        )
        cov = np.asarray(fit.cov_params(), dtype=np.float64)
    elif cov_type == "HC1":
        # statsmodels 0.14 GLMResults lacks cov_HC1 and falls back to cov_white_simple(use_correction=False);
        # invoke statsmodels' own cov_white_simple(..., use_correction=True) for true HC1.
        fit = glm.fit(tol=1e-14, maxiter=200, cov_type="HC0")
        cov = np.asarray(sw.cov_white_simple(fit, use_correction=True), dtype=np.float64)
    else:
        fit = glm.fit(tol=1e-14, maxiter=200, cov_type=cov_type)
        cov = np.asarray(fit.cov_params(), dtype=np.float64)
    return np.asarray(fit.params, dtype=np.float64), cov


def _independent_penalized_sandwich(
    model: SuperGLM,
    y: np.ndarray,
    w: np.ndarray,
    *,
    cluster: np.ndarray | None = None,
    centered: bool = True,
    weight_semantics: str = "prior",
    ridge_lam: float = 0.0,
) -> np.ndarray:
    """Independent reference computing A^-1 B A^-1 directly from model._dm, y, w, S."""
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.model.fit_state import fitted_lambda2
    from superglm.model.state_ops import _public_intercept_shift
    from superglm.reml.observed_geometry import compute_observed_information_weights
    from superglm.solvers.mode_score import linear_predictor

    solver = model._solver_pirls_result()
    dm = model._dm
    X_s = dm.toarray()
    n, p = X_s.shape
    X_aug = np.column_stack([np.ones(n, dtype=np.float64), X_s])

    eta = linear_predictor(dm, solver, model._fit_offset)
    mu = model._link.inverse(eta)
    dmu = model._link.deriv_inverse(eta)
    var = model._distribution.variance(mu)
    s_unit = (y - mu) * dmu / var
    h_obs = compute_observed_information_weights(model._distribution, model._link, y, mu, eta, w)

    S_smooth = _active_penalty_matrix(
        list(dm.group_matrices),
        model._groups,
        model._groups,
        fitted_lambda2(model),
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    S_total = S_smooth.copy()
    if ridge_lam > 0.0:
        S_total[np.diag_indices_from(S_total)] += ridge_lam

    A = (X_aug.T * h_obs) @ X_aug
    A[1:, 1:] += S_total
    A_inv = np.linalg.pinv(A, rcond=1e-12)

    pos = w > 0.0
    if cluster is not None:
        uniq = np.unique(cluster[pos])
        G_mat = np.zeros((len(uniq), 1 + p), dtype=np.float64)
        for idx, cid in enumerate(uniq):
            mask = pos & (cluster == cid)
            G_mat[idx] = np.sum(X_aug[mask] * (w[mask] * s_unit[mask])[:, None], axis=0)
        if centered:
            G_mat = G_mat - np.mean(G_mat, axis=0, keepdims=True)
        B = G_mat.T @ G_mat
    elif weight_semantics == "frequency":
        U = X_aug[pos] * s_unit[pos, None]
        w_pos = w[pos]
        if centered:
            mean_u = np.sum(U * w_pos[:, None], axis=0) / np.sum(w_pos)
            U = U - mean_u[None, :]
        B = (U.T * w_pos) @ U
    else:
        U = X_aug[pos] * (w[pos] * s_unit[pos])[:, None]
        if centered:
            U = U - np.mean(U, axis=0, keepdims=True)
        B = U.T @ U

    V_solver = A_inv @ B @ A_inv
    m_shift = _public_intercept_shift(model, model._groups, p)
    T = np.eye(1 + p, dtype=np.float64)
    T[0, 1:] = m_shift
    return T @ V_solver @ T.T


class TestLifecycleAndContract:
    def test_lifecycle_table_walk_and_frozen_outputs(self):
        assert SANDWICH_LIFECYCLE == {
            ("init", "build_available"): "available",
            ("init", "build_unavailable"): "unavailable",
        }
        for reason in sorted(UNAVAILABLE_REASONS):
            res = build_unavailable_covariance(
                unavailable_reason=reason,  # type: ignore[arg-type]
                message=f"reason={reason}",
            )
            assert res.status == "unavailable"
            assert res.available is False
            assert res.unavailable_reason == reason
            assert reason in repr(res)
            with pytest.raises(RuntimeError, match=reason):
                res.require_available()

        with pytest.raises(ValueError, match="Unknown unavailable_reason"):
            build_unavailable_covariance(
                unavailable_reason="not_a_real_reason",  # type: ignore[arg-type]
                message="bad",
            )

        rng = np.random.default_rng(1)
        X = pd.DataFrame({"x": rng.normal(size=40)})
        y = 1.0 + 0.5 * X["x"].to_numpy() + rng.normal(scale=0.5, size=40)
        unfitted = SuperGLM(family="gaussian", features={"x": Numeric()})
        with pytest.raises(RuntimeError, match="fitted"):
            unfitted.sandwich_covariance()

        model = unfitted.fit(X, y)
        cov = model.sandwich_covariance()
        assert isinstance(cov, SandwichCovariance)
        assert cov.available is True
        assert cov.status == "available"
        assert cov.covariance is not None and not cov.covariance.flags.writeable
        assert cov.se is not None and not cov.se.flags.writeable
        assert cov.telemetry is not None
        assert cov.telemetry.scored_rows == 40

        with pytest.raises(ValueError, match="correction"):
            model.sandwich_covariance(correction="HC1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="center"):
            model.sandwich_covariance(center="median")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SandwichCovariance"):
            model.summary(covariance="HC0")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SandwichCovariance"):
            model.relativities(with_se=True, covariance="HC0")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SandwichCovariance"):
            model.term_inference("x", covariance="HC0")  # type: ignore[arg-type]

    def test_exported_from_root_and_inference(self):
        assert superglm.SandwichCovariance is SandwichCovariance
        from superglm.inference import SandwichCovariance as InfSandwich

        assert InfSandwich is SandwichCovariance


class TestStatsmodelsParity:
    def test_scope_probe_gaussian_offset_and_clusters(self):
        """Matches SCOPE.md probe fixture (HC0=0.155085, CR0=0.175914) and statsmodels 0.14.6."""
        rng = np.random.default_rng(7)
        n = 120
        x = rng.normal(size=n)
        offset = 0.3 * rng.normal(size=n)
        cluster = np.repeat(np.arange(30), 4)
        cl_err = rng.normal(scale=0.8, size=30)[cluster]
        sigma = 0.4 + 1.2 * (x**2)
        y = 1.5 + 0.7 * x + offset + cl_err + rng.normal(scale=sigma)

        X_df = pd.DataFrame({"x": x})
        X_sm = np.column_stack([np.ones(n), x])
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X_df, y, offset=offset
        )

        for cov_kwargs, sm_type, sm_corr in [
            ({}, "HC0", False),
            ({"correction": True}, "HC1", False),
            ({"cluster": cluster, "correction": False}, "cluster", False),
            ({"cluster": cluster, "correction": True}, "cluster", True),
        ]:
            cov = model.sandwich_covariance(**cov_kwargs)
            assert cov.available
            assert (
                cov.bread_reused is True
                if hasattr(cov, "bread_reused")
                else cov.telemetry.bread_reused
            )
            _, cov_sm = _sm_glm_cov(
                X_sm,
                y,
                sm.families.Gaussian(),
                offset=offset,
                cov_type=sm_type,
                cluster=cluster if sm_type == "cluster" else None,
                use_correction=sm_corr,
            )
            np.testing.assert_allclose(
                cov.covariance, cov_sm, atol=_cov_tol(cov_sm, extra=1e-11), rtol=1e-11
            )

        # Constant offset shift leaves slope covariance and fitted-value SEs invariant.
        model_shifted = SuperGLM(
            family="gaussian", selection_penalty=0.0, features={"x": Numeric()}
        ).fit(X_df, y, offset=offset + 12.5)
        cov_base = model.sandwich_covariance(cluster=cluster)
        cov_shift = model_shifted.sandwich_covariance(cluster=cluster)
        np.testing.assert_allclose(
            cov_shift.covariance, cov_base.covariance, rtol=1e-11, atol=1e-12
        )

    @pytest.mark.parametrize(
        ("family_name", "sg_family", "sg_link", "sm_family", "expected_curvature"),
        [
            (
                "poisson_log",
                Poisson(),
                LogLink(),
                sm.families.Poisson(sm.families.links.Log()),
                "fisher",
            ),
            (
                "binomial_logit",
                Binomial(),
                LogitLink(),
                sm.families.Binomial(sm.families.links.Logit()),
                "fisher",
            ),
            (
                "gamma_inv",
                Gamma(),
                InverseLink(),
                sm.families.Gamma(sm.families.links.InversePower()),
                "fisher",
            ),
            (
                "gaussian_log",
                Gaussian(),
                LogLink(),
                sm.families.Gaussian(sm.families.links.Log()),
                "observed",
            ),
            (
                "gamma_log",
                Gamma(),
                LogLink(),
                sm.families.Gamma(sm.families.links.Log()),
                "observed",
            ),
            (
                "binomial_probit",
                Binomial(),
                ProbitLink(),
                sm.families.Binomial(sm.families.links.Probit()),
                "observed",
            ),
            (
                "binomial_cloglog",
                Binomial(),
                CloglogLink(),
                sm.families.Binomial(sm.families.links.CLogLog()),
                "observed",
            ),
            (
                "nb_log",
                NegativeBinomial(theta=2.0),
                LogLink(),
                sm.families.NegativeBinomial(link=sm.families.links.Log(), alpha=0.5),
                "observed",
            ),
        ],
    )
    def test_canonical_and_noncanonical_statsmodels_parity(
        self, family_name, sg_family, sg_link, sm_family, expected_curvature
    ):
        rng = np.random.default_rng(101)
        n = 160
        x1 = rng.normal(scale=0.6, size=n)
        x2 = rng.uniform(-1.0, 1.0, size=n)
        w = rng.uniform(0.5, 2.5, size=n)
        cluster = np.repeat(np.arange(40), 4)
        offset = np.zeros(n)

        if family_name == "poisson_log":
            offset = rng.uniform(-0.3, 0.3, size=n)
            mu_true = np.exp(0.4 + 0.5 * x1 - 0.3 * x2 + offset)
            y = rng.poisson(mu_true).astype(float)
        elif family_name.startswith("binomial"):
            p_true = 1.0 / (1.0 + np.exp(-(0.1 + 0.8 * x1 - 0.5 * x2)))
            y = rng.binomial(1, p_true).astype(float)
        elif family_name == "gamma_inv":
            eta_true = 1.5 + 0.3 * x1 + 0.2 * x2
            mu_true = 1.0 / eta_true
            y = rng.gamma(shape=4.0, scale=mu_true / 4.0)
        elif family_name == "gaussian_log":
            mu_true = np.exp(1.2 + 0.35 * x1 - 0.25 * x2)
            # Mild heteroscedasticity, positive responses keeping observed Hessian PD.
            y = mu_true + rng.normal(scale=0.25 * (1.0 + np.abs(x1)), size=n)
        elif family_name == "gamma_log":
            offset = rng.uniform(-0.2, 0.2, size=n)
            mu_true = np.exp(0.8 + 0.4 * x1 - 0.3 * x2 + offset)
            y = rng.gamma(shape=3.0, scale=mu_true / 3.0)
        elif family_name == "nb_log":
            mu_true = np.exp(0.5 + 0.5 * x1 - 0.3 * x2)
            p_nb = 2.0 / (2.0 + mu_true)
            y = rng.negative_binomial(2.0, p_nb).astype(float)
        else:
            raise AssertionError(family_name)

        X_df = pd.DataFrame({"x1": x1, "x2": x2})
        X_sm = np.column_stack([np.ones(n), x1, x2])
        model = SuperGLM(
            family=sg_family,
            link=sg_link,
            selection_penalty=0.0,
            tol=1e-12,
            features={"x1": Numeric(), "x2": Numeric()},
        ).fit(X_df, y, sample_weight=w, offset=offset)

        for cov_kwargs, sm_type, sm_corr in [
            ({}, "HC0", False),
            ({"correction": True}, "HC1", False),
            ({"cluster": cluster, "correction": False}, "cluster", False),
            ({"cluster": cluster, "correction": True}, "cluster", True),
        ]:
            cov = model.sandwich_covariance(**cov_kwargs)
            assert cov.available, cov
            assert cov.bread_curvature == expected_curvature
            if expected_curvature == "fisher":
                assert cov.telemetry is not None and cov.telemetry.bread_reused is True
                assert cov.telemetry.n_factorizations == 0
            else:
                assert cov.telemetry is not None and cov.telemetry.bread_reused is False
                assert cov.telemetry.n_factorizations == 1

            params_sm, cov_sm = _sm_glm_cov(
                X_sm,
                y,
                sm_family,
                offset=offset if np.any(offset) else None,
                var_weights=w,
                cov_type=sm_type,
                cluster=cluster if sm_type == "cluster" else None,
                use_correction=sm_corr,
            )
            params_sg = np.concatenate([[model.result.intercept], model.result.beta])
            np.testing.assert_allclose(params_sg, params_sm, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                cov.covariance, cov_sm, atol=_cov_tol(cov_sm, extra=1e-6), rtol=1e-6
            )


class TestFrequencyWeightsReplication:
    @pytest.mark.parametrize("family_kind", ["poisson_log", "gamma_log"])
    def test_frequency_weights_match_explicit_row_expansion(self, family_kind):
        rng = np.random.default_rng(202)
        n = 50
        x = rng.normal(scale=0.7, size=n)
        cat = rng.choice(["A", "B", "C"], size=n)
        counts = rng.integers(1, 5, size=n)
        # Include zero-weight rows to verify alignment and exclusion.
        counts[3] = 0
        counts[17] = 0
        w = counts.astype(float)
        cluster = np.repeat(np.arange(10), 5)

        if family_kind == "poisson_log":
            mu = np.exp(0.2 + 0.5 * x + (cat == "B") * 0.4)
            y = rng.poisson(mu).astype(float)
            family, link = Poisson(), LogLink()
        else:
            mu = np.exp(0.5 + 0.4 * x - (cat == "C") * 0.3)
            y = rng.gamma(shape=3.5, scale=mu / 3.5)
            family, link = Gamma(), LogLink()

        X_compact = pd.DataFrame({"x": x, "cat": cat})
        X_exp = pd.DataFrame({"x": np.repeat(x, counts), "cat": np.repeat(cat, counts)})
        y_exp = np.repeat(y, counts)
        cl_exp = np.repeat(cluster, counts)

        # Unpenalised: HC0, HC1, CR0, CR1
        m_w = SuperGLM(
            family=family,
            link=link,
            selection_penalty=0.0,
            weight_semantics="frequency",
            tol=1e-12,
            features={"x": Numeric(), "cat": Categorical(base="A")},
        ).fit(X_compact, y, sample_weight=w)
        m_e = SuperGLM(
            family=family,
            link=link,
            selection_penalty=0.0,
            weight_semantics="frequency",
            tol=1e-12,
            features={"x": Numeric(), "cat": Categorical(base="A")},
        ).fit(X_exp, y_exp)

        for kwargs_w, kwargs_e in [
            ({}, {}),
            ({"correction": True}, {"correction": True}),
            ({"cluster": cluster, "correction": False}, {"cluster": cl_exp, "correction": False}),
            ({"cluster": cluster, "correction": True}, {"cluster": cl_exp, "correction": True}),
            ({"center": "mean"}, {"center": "mean"}),
        ]:
            cw = m_w.sandwich_covariance(**kwargs_w)
            ce = m_e.sandwich_covariance(**kwargs_e)
            assert cw.available and ce.available
            assert cw.n_units == pytest.approx(ce.n_units)
            np.testing.assert_allclose(
                cw.covariance, ce.covariance, atol=_cov_tol(ce.covariance, extra=1e-9), rtol=1e-9
            )

        # Penalised (Ridge): centered and uncentered
        m_wr = SuperGLM(
            family=family,
            link=link,
            penalty=Ridge(lambda1=2.0),
            weight_semantics="frequency",
            tol=1e-12,
            features={"x": Numeric(), "cat": Categorical(base="A")},
        ).fit(X_compact, y, sample_weight=w)
        m_er = SuperGLM(
            family=family,
            link=link,
            penalty=Ridge(lambda1=2.0),
            weight_semantics="frequency",
            tol=1e-12,
            features={"x": Numeric(), "cat": Categorical(base="A")},
        ).fit(X_exp, y_exp)

        for center_opt in ("auto", "none"):
            for use_cl in (False, True):
                cw = m_wr.sandwich_covariance(
                    cluster=cluster if use_cl else None, center=center_opt
                )
                ce = m_er.sandwich_covariance(cluster=cl_exp if use_cl else None, center=center_opt)
                assert cw.available and ce.available
                np.testing.assert_allclose(
                    cw.covariance,
                    ce.covariance,
                    atol=_cov_tol(ce.covariance, extra=1e-8),
                    rtol=1e-8,
                )


class TestPenalizedAndREMLReferences:
    def test_weighted_ridge_matches_independent_reference(self):
        rng = np.random.default_rng(303)
        n = 140
        X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        w = rng.uniform(0.5, 3.0, size=n)
        mu = np.exp(0.3 + 0.4 * X["x1"].to_numpy() - 0.3 * X["x2"].to_numpy())
        y = rng.gamma(shape=3.0, scale=mu / 3.0)
        cluster = np.repeat(np.arange(28), 5)

        model = SuperGLM(
            family="gamma",
            link="log",
            penalty=Ridge(lambda1=3.5),
            tol=1e-12,
            features={"x1": Numeric(), "x2": Numeric()},
        ).fit(X, y, sample_weight=w)

        for cl, cen in [(None, True), (None, False), (cluster, True), (cluster, False)]:
            cov = model.sandwich_covariance(cluster=cl, center="mean" if cen else "none")
            assert cov.available and cov.penalized is True
            assert cov.centered is cen
            ref = _independent_penalized_sandwich(
                model, y, w, cluster=cl, centered=cen, ridge_lam=3.5
            )
            np.testing.assert_allclose(
                cov.covariance, ref, atol=_cov_tol(ref, extra=1e-8), rtol=1e-8
            )

        # Default center="auto" centres penalised fits.
        assert model.sandwich_covariance().centered is True
        # correction=True rejected on penalised fits.
        with pytest.raises(ValueError, match="unpenalised"):
            model.sandwich_covariance(correction=True)

    @pytest.mark.parametrize("discrete", [False, True])
    def test_frozen_lambda_reml_exact_and_binned_match_independent_reference(self, discrete):
        rng = np.random.default_rng(404)
        n = 260
        x = rng.uniform(0.0, 10.0, size=n)
        z = rng.normal(size=n)
        w = rng.uniform(0.5, 2.0, size=n)
        eta = -0.3 + 0.4 * np.sin(x) + 0.25 * z
        y = rng.poisson(np.exp(eta) * w).astype(float)
        cluster = np.repeat(np.arange(26), 10)
        X = pd.DataFrame({"x": x, "z": z})

        model = SuperGLM(
            family="poisson",
            link="log",
            discrete=discrete,
            n_bins=32,
            features={"x": Spline(kind="cr", k=7), "z": Numeric()},
        ).fit_reml(X, y, sample_weight=w)

        for cl, cen in [(None, True), (None, False), (cluster, True)]:
            cov = model.sandwich_covariance(cluster=cl, center="mean" if cen else "none")
            assert cov.available
            assert cov.penalized is True
            ref = _independent_penalized_sandwich(model, y, w, cluster=cl, centered=cen)
            np.testing.assert_allclose(
                cov.covariance, ref, atol=_cov_tol(ref, extra=1e-8), rtol=1e-8
            )

    def test_intercept_only_model(self):
        rng = np.random.default_rng(405)
        n = 60
        X = pd.DataFrame(index=np.arange(n))
        y = rng.poisson(2.5, size=n).astype(float)
        cluster = np.repeat(np.arange(15), 4)
        model = SuperGLM(family="poisson", features={}).fit(X, y)

        cov_hc0 = model.sandwich_covariance()
        cov_cr0 = model.sandwich_covariance(cluster=cluster)
        assert cov_hc0.available and cov_hc0.covariance.shape == (1, 1)
        assert cov_cr0.available and cov_cr0.covariance.shape == (1, 1)
        _, sm_hc0 = _sm_glm_cov(np.ones((n, 1)), y, sm.families.Poisson(), cov_type="HC0")
        _, sm_cr0 = _sm_glm_cov(
            np.ones((n, 1)),
            y,
            sm.families.Poisson(),
            cov_type="cluster",
            cluster=cluster,
            use_correction=False,
        )
        np.testing.assert_allclose(cov_hc0.covariance, sm_hc0, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(cov_cr0.covariance, sm_cr0, rtol=1e-11, atol=1e-12)


class TestMgcvBlackBoxParity:
    def test_mgcv_black_box_fixed_penalty_raw_and_centered_figures(self):
        """Reproduce SCOPE.md's mgcv 1.9-3 black-box figures on the exact same basis & penalty.

        mgcv is GPL: invoked strictly as an external black-box Rscript command.
        """
        rscript = shutil.which("Rscript")
        if rscript is None:
            pytest.skip("Rscript not installed")

        r_code = r"""
        suppressPackageStartupMessages(library(mgcv))
        d <- data.frame(x=c(-3,-2,-1,0,1,2,3,4), y=c(-6,-3,-2,2,5,6,10,12))
        m <- gam(y ~ s(x, bs="cr", k=5), data=d, scale=1, sp=30, method="REML")
        X <- model.matrix(m)
        inv <- vcov(m)
        s <- X * as.numeric(d$y - fitted(m))
        u <- sweep(s, 2, colMeans(s))
        raw <- inv %*% crossprod(s) %*% inv
        cen <- inv %*% crossprod(u) %*% inv
        cat("COEF_RAW:", sprintf("%.15g", sqrt(diag(raw))), "\n")
        cat("COEF_CEN:", sprintf("%.15g", sqrt(diag(cen))), "\n")
        cat("FIT_RAW:", sprintf("%.15g", sqrt(diag(X %*% raw %*% t(X)))), "\n")
        cat("FIT_CEN:", sprintf("%.15g", sqrt(diag(X %*% cen %*% t(X)))), "\n")
        cat("S_PEN:", sprintf("%.15g", as.numeric(m$smooth[[1]]$S[[1]])), "\n")
        cat("X_FLAT:", sprintf("%.15g", as.numeric(X[, -1])), "\n")
        """
        proc = subprocess.run([rscript, "-e", r_code], capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            pytest.skip(f"R/mgcv unavailable: {proc.stderr.strip()}")

        parsed: dict[str, np.ndarray] = {}
        for line in proc.stdout.splitlines():
            for key in ("COEF_RAW:", "COEF_CEN:", "FIT_RAW:", "FIT_CEN:", "S_PEN:", "X_FLAT:"):
                if line.startswith(key):
                    parsed[key[:-1]] = np.fromstring(line[len(key) :], sep=" ")

        # Confirm SCOPE.md black-box smooth coefficient SEs (~0.111920 raw, ~0.111042 centered).
        assert np.any(np.isclose(parsed["COEF_RAW"], 0.111920, atol=2e-6))
        assert np.any(np.isclose(parsed["COEF_CEN"], 0.111042, atol=2e-6))

        # Transform mgcv's penalty S into diagonal Ridge coordinates (S = Q diag(d) Q')
        # so SuperGLM solves the identical fixed-penalty Gaussian problem on the exact basis.
        n = 8
        y = np.array([-6, -3, -2, 2, 5, 6, 10, 12], dtype=float)
        X_slopes = parsed["X_FLAT"].reshape((4, n)).T  # column-major from R
        S_pen = 30.0 * parsed["S_PEN"].reshape((4, 4))
        eigvals, Q = np.linalg.eigh(0.5 * (S_pen + S_pen.T))
        # Penalty rank is 3 (1 linear null direction + 3 positive eigenvalues).
        pos = eigvals > 1e-10
        scale = np.ones(4)
        scale[pos] = np.sqrt(eigvals[pos])
        Z = (X_slopes @ Q) / scale[None, :]
        cols = [f"z{j}" for j in range(4)]
        X_df = pd.DataFrame(Z, columns=cols)
        penalized_cols = [cols[j] for j in range(4) if pos[j]]

        model = SuperGLM(
            family="gaussian",
            penalty=Ridge(lambda1=1.0, features=penalized_cols),
            tol=1e-12,
            features={c: Numeric() for c in cols},
        ).fit(X_df, y)

        cov_raw = model.sandwich_covariance(center="none")
        cov_cen = model.sandwich_covariance(center="mean")
        assert cov_raw.available and cov_cen.available

        X_aug_z = np.column_stack([np.ones(n), Z])
        fit_raw_sg = np.sqrt(np.diag(X_aug_z @ cov_raw.covariance @ X_aug_z.T))
        fit_cen_sg = np.sqrt(np.diag(X_aug_z @ cov_cen.covariance @ X_aug_z.T))
        np.testing.assert_allclose(fit_raw_sg, parsed["FIT_RAW"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(fit_cen_sg, parsed["FIT_CEN"], rtol=1e-6, atol=1e-8)

        # Transform slope covariance back to mgcv's coordinates: b_mgcv = Q diag(1/scale) b_z.
        M = Q / scale[None, :]
        se_raw_mgcv_coords = np.sqrt(np.diag(M @ cov_raw.slope_covariance @ M.T))
        se_cen_mgcv_coords = np.sqrt(np.diag(M @ cov_cen.slope_covariance @ M.T))
        np.testing.assert_allclose(se_raw_mgcv_coords, parsed["COEF_RAW"][1:], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(se_cen_mgcv_coords, parsed["COEF_CEN"][1:], rtol=1e-6, atol=1e-8)


class TestNumericalStabilityInvariances:
    def test_cluster_score_accumulator_exact_cancellation_within_and_across_chunks(self):
        """Cancellation [+1e16, +1.0, -1e16] inside one chunk or across chunks sums to 1.0."""
        # Across three chunks: row 0 = +1e16, row 1 = +0.5, row 2 = +0.5, row 3 = -1e16
        scores = np.array(
            [
                [1e16, -2.0],
                [0.5, 3.0],
                [0.5, 1.0],
                [-1e16, 4.0],
                [2.0, -1.0],
                [3.0, 2.0],
            ],
            dtype=np.float64,
        )
        codes = np.array([0, 0, 0, 0, 1, 1], dtype=np.intp)
        acc = ClusterScoreAccumulator(2, 2, codes)
        acc.add_chunk(0, 2, scores[0:2])
        acc.add_chunk(2, 4, scores[2:4])
        acc.add_chunk(4, 6, scores[4:6])
        cancelled = acc.cancelled_clusters()
        assert 0 in cancelled
        acc.refine_cancelled(cancelled, lambda idx: scores[idx])
        np.testing.assert_allclose(acc.totals[0], [1.0, 6.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(acc.totals[1], [5.0, 1.0], rtol=0.0, atol=0.0)

    def test_end_to_end_clustered_cancellation_matches_exact_fsum(self):
        """End-to-end Gaussian clustered sandwich preserves digits under large opposing rows."""
        # 4 clusters, 4 rows each; cluster 0 has massive opposing residuals that cancel to +1.
        y = np.array(
            [
                1e12 + 1.0,
                -1e12 + 1.0,
                1.5,
                0.5,
                2.0,
                0.0,
                1.0,
                1.0,
                0.5,
                1.5,
                2.0,
                0.0,
                1.2,
                0.8,
                0.4,
                1.6,
            ],
            dtype=np.float64,
        )
        cluster = np.repeat(np.arange(4), 4)
        X = pd.DataFrame(index=np.arange(len(y)))
        model = SuperGLM(family="gaussian", features={}).fit(X, y)
        cov = model.sandwich_covariance(cluster=cluster)
        assert cov.available
        # Exact reference via math.fsum per cluster on residual y - mean(y)
        mu = float(model.result.intercept)
        g_exact = np.array(
            [math.fsum((y[cluster == c] - mu).tolist()) for c in range(4)],
            dtype=np.float64,
        )
        expected_var = float(np.sum(g_exact**2)) / (len(y) ** 2)
        np.testing.assert_allclose(cov.covariance[0, 0], expected_var, rtol=1e-12, atol=1e-14)

    def test_large_offset_centering_preserves_slope_precision(self):
        rng = np.random.default_rng(505)
        n = 120
        u_raw = rng.normal(size=n)
        # Subtract 1e8 back from the shifted column so both inputs have identical float64 spread
        # without conflating float64 input quantization with covariance centering precision.
        x_shift = 1e8 + u_raw
        u_exact = x_shift - 1e8
        y = 1.0 + 0.8 * u_exact + rng.normal(scale=0.5 + np.abs(u_exact), size=n)
        cluster = np.repeat(np.arange(20), 6)

        m_plain = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            pd.DataFrame({"x": u_exact}), y
        )
        m_shift = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            pd.DataFrame({"x": x_shift}), y
        )
        for kwargs in ({}, {"cluster": cluster}):
            c_plain = m_plain.sandwich_covariance(**kwargs)
            c_shift = m_shift.sandwich_covariance(**kwargs)
            assert c_plain.available and c_shift.available
            np.testing.assert_allclose(
                c_shift.covariance[1, 1], c_plain.covariance[1, 1], rtol=1e-11, atol=1e-14
            )

    def test_categorical_releveling_leaves_contrasts_and_fitted_se_invariant(self):
        rng = np.random.default_rng(606)
        n = 180
        cat = rng.choice(["A", "B", "C", "D"], size=n)
        x = rng.normal(size=n)
        cluster = np.repeat(np.arange(30), 6)
        eff = pd.Series(cat).map({"A": 0.0, "B": 0.3, "C": -0.2, "D": 0.5}).to_numpy()
        y = rng.poisson(np.exp(0.1 + eff + 0.4 * x)).astype(float)
        X = pd.DataFrame({"cat": cat, "x": x})

        m_a = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"cat": Categorical(base="A"), "x": Numeric()},
        ).fit(X, y)
        m_c = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"cat": Categorical(base="C"), "x": Numeric()},
        ).fit(X, y)

        for kwargs in ({}, {"cluster": cluster}, {"cluster": cluster, "correction": True}):
            cov_a = m_a.sandwich_covariance(**kwargs)
            cov_c = m_c.sandwich_covariance(**kwargs)
            assert cov_a.available and cov_c.available
            # Slope SE for numeric x is invariant.
            np.testing.assert_allclose(
                cov_a.covariance[-1, -1], cov_c.covariance[-1, -1], rtol=1e-10, atol=1e-12
            )
            # Fitted-value SEs across all training rows are invariant.
            Xa = np.column_stack([np.ones(n), m_a._dm.toarray()])
            Xc = np.column_stack([np.ones(n), m_c._dm.toarray()])
            se_fit_a = np.sqrt(np.einsum("ij,jk,ik->i", Xa, cov_a.covariance, Xa))
            se_fit_c = np.sqrt(np.einsum("ij,jk,ik->i", Xc, cov_c.covariance, Xc))
            np.testing.assert_allclose(se_fit_a, se_fit_c, rtol=1e-10, atol=1e-12)
            # Mean-centered categorical relativity SEs are invariant.
            rel_a = m_a.relativities(with_se=True, centering="mean", covariance=cov_a)["cat"]
            rel_c = m_c.relativities(with_se=True, centering="mean", covariance=cov_c)["cat"]
            rel_a = rel_a.set_index("level").sort_index()
            rel_c = rel_c.set_index("level").sort_index()
            np.testing.assert_allclose(
                rel_a["se_log_relativity"].to_numpy(),
                rel_c["se_log_relativity"].to_numpy(),
                rtol=1e-10,
                atol=1e-12,
            )

    def test_tiny_weights_weight_exponent_invariance(self):
        """Prior weights scaled by 1e-200 trigger weight_exponent > 0 with invariant sandwich."""
        rng = np.random.default_rng(607)
        n = 80
        X = pd.DataFrame({"x": rng.normal(size=n)})
        y = 1.0 + 0.5 * X["x"].to_numpy() + rng.normal(scale=0.5, size=n)
        cluster = np.repeat(np.arange(16), 5)

        m_unit = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X, y, sample_weight=np.ones(n)
        )
        m_tiny = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X, y, sample_weight=np.full(n, 1e-200)
        )
        assert m_tiny._solver_pirls_result().weight_exponent > 0

        for kwargs in ({}, {"cluster": cluster}, {"correction": True}):
            c_unit = m_unit.sandwich_covariance(**kwargs)
            c_tiny = m_tiny.sandwich_covariance(**kwargs)
            assert c_tiny.available
            np.testing.assert_allclose(c_tiny.covariance, c_unit.covariance, rtol=1e-11, atol=1e-12)


class TestClusterAlignmentAndValidation:
    def test_zero_weight_rows_ignored_even_with_missing_cluster_labels(self):
        rng = np.random.default_rng(707)
        n = 60
        x = rng.normal(size=n)
        y = 0.5 + 0.6 * x + rng.normal(scale=0.4, size=n)
        w = np.ones(n)
        w[[2, 19, 44]] = 0.0
        cl_obj = np.array([f"g{i // 4}" for i in range(n)], dtype=object)
        cl_obj[2] = None
        cl_obj[19] = np.nan
        cl_obj[44] = pd.NA

        X = pd.DataFrame({"x": x})
        m_full = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X, y, sample_weight=w
        )
        keep = w > 0.0
        m_pos = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X.loc[keep].reset_index(drop=True), y[keep]
        )

        cov_full = m_full.sandwich_covariance(cluster=cl_obj, correction=True)
        cov_pos = m_pos.sandwich_covariance(cluster=cl_obj[keep], correction=True)
        assert cov_full.available and cov_pos.available
        assert cov_full.n_clusters == cov_pos.n_clusters
        assert cov_full.n_units == cov_pos.n_units == 57.0
        np.testing.assert_allclose(cov_full.covariance, cov_pos.covariance, rtol=1e-12, atol=1e-13)

        # Also works with pandas Series / Categorical
        cov_cat = m_full.sandwich_covariance(cluster=pd.Series(cl_obj).astype("category"))
        assert cov_cat.available
        np.testing.assert_allclose(
            cov_cat.covariance,
            m_pos.sandwich_covariance(cluster=cl_obj[keep]).covariance,
            rtol=1e-12,
            atol=1e-13,
        )

    def test_invalid_clusters_rejected(self):
        rng = np.random.default_rng(708)
        n = 20
        X = pd.DataFrame({"x": rng.normal(size=n)})
        y = rng.normal(size=n)
        w = np.ones(n)
        w[0] = 0.0
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            X, y, sample_weight=w
        )

        # Wrong length
        with pytest.raises(ValueError, match="cluster length"):
            model.sandwich_covariance(cluster=np.arange(n - 1))
        # 2-D array
        with pytest.raises(ValueError, match="one-dimensional"):
            model.sandwich_covariance(cluster=np.zeros((n, 2)))
        # Missing label on positive-weight row
        bad_nan = np.arange(n, dtype=float)
        bad_nan[5] = np.nan
        with pytest.raises(ValueError, match="NaN|missing"):
            model.sandwich_covariance(cluster=bad_nan)
        bad_none = np.array([f"c{i % 4}" for i in range(n)], dtype=object)
        bad_none[5] = None
        with pytest.raises(ValueError, match="None|missing"):
            model.sandwich_covariance(cluster=bad_none)
        # Unhashable label
        bad_list = np.empty(n, dtype=object)
        for i in range(n):
            bad_list[i] = [i % 3]
        with pytest.raises(ValueError, match="hashable"):
            model.sandwich_covariance(cluster=bad_list)
        # Only 1 positive-weight cluster (second label only on zero-weight row 0)
        one_cluster = np.array(["only_zero"] + ["main"] * (n - 1), dtype=object)
        with pytest.raises(ValueError, match="at least 2"):
            model.sandwich_covariance(cluster=one_cluster)


class TestFitBindingAndRoundTrip:
    def test_roundtrip_and_refit_cannot_silently_reuse_covariance(self):
        rng = np.random.default_rng(808)
        n = 60
        X = pd.DataFrame({"x": rng.normal(size=n), "s": rng.uniform(0, 5, size=n)})
        y = rng.poisson(np.exp(0.2 + 0.3 * X["x"].to_numpy())).astype(float)

        m1 = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric(), "s": Spline(kind="cr", k=5)},
        ).fit(X, y)
        cov1 = m1.sandwich_covariance()
        assert cov1.available

        # Pickle round-trip of both model and covariance stays valid together.
        m1_rt = pickle.loads(pickle.dumps(m1))
        cov1_rt = pickle.loads(pickle.dumps(cov1))
        s_rt = m1_rt.summary(covariance=cov1_rt)
        assert s_rt["standard_errors"]["coefficient_se_scale"] == "sandwich"

        # A model fitted to another response is another fit and rejects cov1.
        y_other = y.copy()
        y_other[0] += 1.0
        m2 = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric(), "s": Spline(kind="cr", k=5)},
        ).fit(X, y_other)
        for call in (
            lambda: m2.summary(covariance=cov1),
            lambda: m2.relativities(with_se=True, covariance=cov1),
            lambda: m2.relativities(with_se=False, covariance=cov1),
            lambda: m2.term_inference("x", covariance=cov1),
            lambda: m2.simultaneous_bands("s", covariance=cov1),
        ):
            with pytest.raises(ValueError, match="different fit"):
                call()

        # Refitting m1 invalidates cov1 and cov1_rt.
        m1.fit(X, y)
        with pytest.raises(ValueError, match="different fit"):
            m1.summary(covariance=cov1)
        with pytest.raises(ValueError, match="different fit"):
            m1.summary(covariance=cov1_rt)

    def test_the_same_fit_on_the_same_rows_is_one_identity(self):
        """The binding is the fit's own record, not an identity allocated per fit.

        A rerun of one specification on the same rows publishes the same numbers and
        reads a covariance computed on the first run.
        """
        rng = np.random.default_rng(811)
        X = pd.DataFrame({"x": rng.normal(size=50)})
        y = rng.poisson(np.exp(0.2 + 0.3 * X["x"].to_numpy())).astype(float)
        first = SuperGLM(family="poisson", features={"x": Numeric()}).fit(X, y)
        rerun = SuperGLM(family="poisson", features={"x": Numeric()}).fit(X.copy(), y.copy())
        cov = first.sandwich_covariance()
        assert cov.binding == rerun.sandwich_covariance().binding
        np.testing.assert_array_equal(
            rerun.term_inference("x", covariance=cov).se_log_relativity,
            first.term_inference("x", covariance=cov).se_log_relativity,
        )

    def test_saved_fits_with_equal_coefficients_do_not_share_a_covariance(self):
        """Two intercept-only fits at intercept 0 on different responses are different fits."""
        frame = pd.DataFrame(index=range(8))
        models = []
        for amplitude in (1.0, 2.0):
            model = SuperGLM(family="gaussian", features={}).fit(
                frame, np.tile([-amplitude, amplitude], 4)
            )
            model = pickle.loads(pickle.dumps(model))
            # A model saved before fits could be bound stores no identity beyond its rows.
            vars(model._fit_state).pop("fit_uid", None)
            models.append(model)
        small, large = models
        assert small.result.intercept == large.result.intercept == 0.0
        cov_small, cov_large = small.sandwich_covariance(), large.sandwich_covariance()
        np.testing.assert_allclose(cov_small.se, [1.0 / math.sqrt(8.0)], rtol=1e-12)
        np.testing.assert_allclose(cov_large.se, [2.0 / math.sqrt(8.0)], rtol=1e-12)
        with pytest.raises(ValueError, match="different fit"):
            large.summary(covariance=cov_small)

    def test_a_weight_contract_is_part_of_the_fit(self):
        """All weights 4 on the same rows: the same intercept, two covariances.

        Frequency weights make each row four units, ``sqrt(sum w r^2) / sum w``;
        prior weights keep it one, ``sqrt(sum (w r)^2) / sum w``, twice that here.
        Neither reader accepts the other's covariance.
        """
        y = np.tile([-1.0, 1.0], 16)
        w = np.full(len(y), 4.0)
        frame = pd.DataFrame(index=range(len(y)))
        fits = {
            semantics: SuperGLM(family="gaussian", features={}, weight_semantics=semantics).fit(
                frame, y, sample_weight=w
            )
            for semantics in ("frequency", "prior")
        }
        assert fits["frequency"].result.intercept == fits["prior"].result.intercept == 0.0
        covs = {semantics: model.sandwich_covariance() for semantics, model in fits.items()}
        np.testing.assert_allclose(
            covs["frequency"].se, [math.sqrt(w @ y**2) / w.sum()], rtol=1e-12
        )
        np.testing.assert_allclose(
            covs["prior"].se, [math.sqrt((w * y) @ (w * y)) / w.sum()], rtol=1e-12
        )
        for own, other in (("prior", "frequency"), ("frequency", "prior")):
            intercept = fits[own].summary(covariance=covs[own])._coef_rows[0]
            np.testing.assert_allclose(intercept.se, covs[own].se[0], rtol=1e-12)
            with pytest.raises(ValueError, match="different fit"):
                fits[own].summary(covariance=covs[other])

    def test_a_model_rescored_on_other_rows_replays_no_scores(self):
        """``to_model(X=, y=)`` keeps the fitted design beside the supplied response."""
        from superglm.editor import EditorSession

        x = np.tile([-1.0, 1.0], 32)
        noise = np.tile([-1.0, -1.0, 1.0, 1.0], 16)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(family="gaussian", features={"x": Numeric()}).fit(X, x + 0.1 * noise)
        fitted = model.sandwich_covariance()
        np.testing.assert_allclose(fitted.se, [0.0125, 0.0125], rtol=1e-10)

        session = EditorSession.from_model(model, terms=["x"])
        rescored = session.to_model(X=pd.DataFrame({"x": -x}), y=-x + 10.0 * noise)
        np.testing.assert_array_equal(rescored.result.beta, model.result.beta)
        own = rescored.sandwich_covariance()
        assert not own.available and own.unavailable_reason == "missing_fit_state"
        assert "not the rows it was fitted on" in own.message


class TestUnavailableConditionsAndRecovery:
    def test_every_scope_unavailable_condition_and_refit_unpenalised_recovery(self, monkeypatch):
        rng = np.random.default_rng(909)
        n = 80
        x = rng.uniform(0.0, 10.0, size=n)
        z = rng.normal(size=n)
        X = pd.DataFrame({"x": x, "z": z})
        y_pois = rng.poisson(np.exp(0.1 + 0.1 * x + 0.3 * z)).astype(float)

        # 1. retain_fit_state=False -> missing_fit_state
        m_retired = SuperGLM(
            family="poisson", retain_fit_state=False, features={"z": Numeric()}
        ).fit(X, y_pois)
        c_ret = m_retired.sandwich_covariance()
        assert not c_ret.available and c_ret.unavailable_reason == "missing_fit_state"
        with pytest.raises(RuntimeError, match="missing_fit_state"):
            m_retired.summary(covariance=c_ret)

        # 2. Editor stale -> editor_stale
        m_ed = SuperGLM(family="poisson", features={"z": Numeric()}).fit(X, y_pois)
        m_ed._editor_inference_stale = True
        c_ed = m_ed.sandwich_covariance()
        assert not c_ed.available and c_ed.unavailable_reason == "editor_stale"

        # 3. Shape repaired -> shape_repaired
        y_nonmono = np.exp(0.2 - 0.1 * (x - 5.0) ** 2) + rng.uniform(0.1, 0.5, size=n)
        m_rep = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="ps", k=6, constraint=Constraint.postfit.increasing)},
        ).fit(X, y_nonmono)
        m_rep.monotonize(X)
        c_rep = m_rep.sandwich_covariance()
        assert not c_rep.available and c_rep.unavailable_reason == "shape_repaired"

        # 4. Unconverged fit -> not_converged
        y_gam = rng.gamma(shape=2.0, scale=np.exp(0.5 + 0.6 * z) / 2.0)
        m_unconv = SuperGLM(
            family="gamma", link="log", max_iter=1, tol=1e-14, features={"z": Numeric()}
        ).fit(X, y_gam)
        assert not m_unconv.result.converged
        c_unconv = m_unconv.sandwich_covariance()
        assert not c_unconv.available and c_unconv.unavailable_reason == "not_converged"

        # 5. Selection penalties (GroupLasso & GroupElasticNet alpha > 0) -> selection_penalty
        #    and verified reduced model via refit_unpenalised succeeds!
        m_gl = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.01),
            features={"x": Numeric(), "z": Numeric()},
        ).fit(X, y_pois)
        c_gl = m_gl.sandwich_covariance()
        assert not c_gl.available and c_gl.unavailable_reason == "selection_penalty"

        m_gen = SuperGLM(
            family="poisson",
            penalty=GroupElasticNet(lambda1=0.01, alpha=0.5),
            features={"x": Numeric(), "z": Numeric()},
        ).fit(X, y_pois)
        assert m_gen.sandwich_covariance().unavailable_reason == "selection_penalty"

        reduced = m_gl.refit_unpenalised(X, y_pois)
        c_red = reduced.sandwich_covariance()
        assert c_red.available

        # 6. Active constraints (Constraint.fit.increasing) -> active_constraints
        m_con = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="ps", k=6, constraint=Constraint.fit.increasing)},
        ).fit(X, y_nonmono)
        c_con = m_con.sandwich_covariance()
        assert not c_con.available and c_con.unavailable_reason == "active_constraints"

        # 7. Unsupported custom link curvature -> unsupported_curvature
        class _CustomLogLink:
            def link(self, mu):
                return np.log(mu)

            def inverse(self, eta):
                return np.exp(np.clip(eta, -20.0, 20.0))

            def deriv(self, mu):
                return 1.0 / mu

            def deriv_inverse(self, eta):
                return np.exp(np.clip(eta, -20.0, 20.0))

        m_cust = SuperGLM(
            family="gamma",
            link=_CustomLogLink(),
            selection_penalty=0.0,
            features={"z": Numeric()},
        ).fit(X, y_gam)
        c_cust = m_cust.sandwich_covariance()
        assert not c_cust.available and c_cust.unavailable_reason == "unsupported_curvature"

        # 8. Indefinite observed bread -> indefinite_bread (forced deterministically across BLAS builds)
        import superglm.inference.sandwich as sandwich_mod

        m_ind = SuperGLM(
            family="gamma", link="log", selection_penalty=0.0, features={"z": Numeric()}
        ).fit(X, y_gam)
        orig_score_curv = sandwich_mod._observed_score_and_curvature

        def _indefinite_curvature(dist, link, y_arr, mu_arr, eta_arr):
            s_unit, _h_unit, is_canon, curv = orig_score_curv(dist, link, y_arr, mu_arr, eta_arr)
            # Positive total intercept curvature, strongly negative curvature on tail |z| rows.
            h_bad = np.where(np.abs(z) > 0.8, -50.0, 2.0)
            return s_unit, h_bad, is_canon, curv

        with monkeypatch.context() as mp:
            mp.setattr(sandwich_mod, "_observed_score_and_curvature", _indefinite_curvature)
            c_ind = m_ind.sandwich_covariance()
            assert not c_ind.available and c_ind.unavailable_reason == "indefinite_bread"

        # 9. Workspace budget exceeded -> workspace_budget_exceeded
        m_ok = SuperGLM(family="poisson", selection_penalty=0.0, features={"z": Numeric()}).fit(
            X, y_pois
        )
        c_mem = m_ok.sandwich_covariance(max_memory=16)
        assert not c_mem.available and c_mem.unavailable_reason == "workspace_budget_exceeded"

    def test_collinear_rank_deficient_fit_available_uncorrected_rejects_correction(self):
        rng = np.random.default_rng(910)
        n = 60
        x = rng.normal(size=n)
        X = pd.DataFrame({"x1": x, "x2": 2.0 * x})
        y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=n)

        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x1": Numeric(), "x2": Numeric()},
        ).fit(X, y)
        cov = model.sandwich_covariance()
        assert cov.available
        assert cov.bread_rank < cov.n_params
        assert cov.se is not None and np.any(np.isnan(cov.se[1:]))

        with pytest.raises(ValueError, match="full-rank"):
            model.sandwich_covariance(correction=True)


class TestReadersAndStructuredFits:
    def test_summary_relativities_term_inference_and_simultaneous_bands(self):
        rng = np.random.default_rng(1001)
        n = 240
        age = rng.uniform(20.0, 70.0, size=n)
        region = rng.choice(["N", "S", "E"], size=n)
        z = rng.normal(size=n)
        cluster = np.repeat(np.arange(30), 8)
        eta = 0.2 + 0.35 * np.sin((age - 20.0) / 10.0) + (region == "S") * 0.4 + 0.2 * z
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"age": age, "region": region, "z": z})

        model = SuperGLM(
            family="poisson",
            features={
                "age": Spline(kind="cr", k=6),
                "region": Categorical(base="N"),
                "z": Numeric(),
            },
        ).fit_reml(X, y)

        cov = model.sandwich_covariance(cluster=cluster)
        assert cov.available and cov.clustered and cov.centered

        # Summary ASCII + HTML + dict (both compact and full)
        s_default = model.summary()
        s_robust = model.summary(detail="full", covariance=cov)
        s_default_after = model.summary()
        assert s_default_after is s_default  # default summary cache untouched

        txt = str(s_robust)
        html = s_robust._repr_html_()
        assert "Cluster CR0 (sandwich, centered)" in txt
        assert "No. Clusters" in txt and "30" in txt
        assert "Wood 2013" in txt and "omitted" in txt
        assert "curve SE:" in txt
        assert "Cluster CR0 (sandwich, centered)" in html
        assert "curve SE:" in html
        assert s_robust["standard_errors"]["coefficient_se_scale"] == "sandwich"
        assert s_robust["standard_errors"]["covariance_n_clusters"] == 30

        # Smooth Wood p-value must be blank while curve_se_min/max stay populated
        age_row = next(r for r in s_robust._coef_rows if r.name == "age")
        assert age_row.wald_chi2 is None and age_row.wald_p is None and age_row.ref_df is None
        assert age_row.curve_se_min is not None and np.isfinite(age_row.curve_se_min)
        # Penalised basis detail rows have blank z/p and valid finite SE/CI
        for br in s_robust._basis_detail["age"]:
            assert br.z is None and br.p is None
            assert np.isfinite(br.se) and br.se >= 0.0

        # relativities, term_inference, simultaneous_bands
        rel_def = model.relativities(with_se=True)
        rel_rob = model.relativities(with_se=True, covariance=cov)
        assert not np.allclose(
            rel_def["z"]["se_log_relativity"].to_numpy(),
            rel_rob["z"]["se_log_relativity"].to_numpy(),
        )
        np.testing.assert_allclose(
            rel_rob["z"]["se_log_relativity"].to_numpy(),
            [cov.se[-1]],
            rtol=1e-12,
        )

        ti = model.term_inference("age", simultaneous=True, covariance=cov)
        assert ti.covariance_method == cov.method_label
        assert ti.covariance_n_clusters == 30
        assert ti.covariance_n_units == float(n)
        assert ti.ci_lower_simultaneous is not None

        bands = model.simultaneous_bands("age", covariance=cov)
        np.testing.assert_allclose(bands["se"].to_numpy(), ti.se_log_relativity, rtol=1e-12)
        np.testing.assert_allclose(
            bands["ci_lower_simultaneous"].to_numpy(), ti.ci_lower_simultaneous, rtol=1e-12
        )

    def test_low_cluster_wald_blanked_and_separation_blanks_preserved(self):
        rng = np.random.default_rng(1002)
        n = 90
        x = rng.uniform(0.0, 10.0, size=n)
        cat = np.array(["A"] * 40 + ["B"] * 40 + ["SEP"] * 10)
        cluster = np.tile(np.arange(3), n // 3)
        y = rng.poisson(np.exp(0.2 + 0.1 * x + (cat == "B") * 0.4)).astype(float)
        # Complete separation on level SEP (all zeros)
        y[cat == "SEP"] = 0.0
        X = pd.DataFrame({"x": x, "cat": cat})

        with pytest.warns(superglm.SeparationWarning):
            model = SuperGLM(
                family="poisson",
                selection_penalty=0.0,
                features={
                    "x": Piecewise(breaks=[2.5, 5.0, 7.5]),  # 4-df block > G-1 = 2
                    "cat": Categorical(base="A"),
                },
            ).fit(X, y)

        cov = model.sandwich_covariance(cluster=cluster)
        assert cov.available
        s = model.summary(covariance=cov)

        # Piecewise 4-df whole-term Wald test blanked because G=3 cannot span 4 df.
        pw_row = next(r for r in s._coef_rows if r.name == "x" and r.is_spline)
        assert pw_row.wald_chi2 is None and pw_row.wald_p is None

        # Separated categorical level retains blank SE/z/p/CI and separated=True (#685).
        sep_row = next(r for r in s._coef_rows if r.name == "cat[SEP]")
        assert sep_row.separated is True
        assert sep_row.se is None and sep_row.z is None and sep_row.p is None
        assert "SEP" in str(s)

        ti_cat = model.term_inference("cat", covariance=cov)
        sep_idx = ti_cat.levels.index("SEP")
        assert ti_cat.separated is not None and bool(ti_cat.separated[sep_idx]) is True
        assert np.isnan(ti_cat.se_log_relativity[sep_idx])

    def test_ridge_summary_withholds_shrunk_z_p_keeps_se_ci(self):
        rng = np.random.default_rng(1003)
        n = 80
        X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        y = 1.0 + 0.5 * X["x"].to_numpy() - 0.3 * X["z"].to_numpy() + rng.normal(size=n)
        model = SuperGLM(
            family="gaussian",
            penalty=Ridge(lambda1=2.0, features=["x"]),
            features={"x": Numeric(), "z": Numeric()},
        ).fit(X, y)
        cov = model.sandwich_covariance()
        s = model.summary(covariance=cov)
        x_row = next(r for r in s._coef_rows if r.name == "x")
        z_row = next(r for r in s._coef_rows if r.name == "z")
        assert x_row.se is not None and x_row.se > 0.0
        assert x_row.z is None and x_row.p is None
        assert x_row.ci_low is not None and np.isfinite(x_row.ci_low)
        assert z_row.z is not None and z_row.p is not None
        assert "---" in str(s) and s._repr_html_()

    @pytest.mark.parametrize("family_pair", [("poisson", "log"), ("gamma", "log")])
    def test_structured_random_effect_matches_independent_reference(self, family_pair):
        family_name, link_name = family_pair
        rng = np.random.default_rng(1004)
        n_groups = 60
        n = 300
        grp = np.array([f"L{i % n_groups:02d}" for i in range(n)])
        x = rng.normal(scale=0.6, size=n)
        re_eff = rng.normal(scale=0.3, size=n_groups)
        eta = 0.4 + 0.4 * x + re_eff[np.arange(n) % n_groups]
        if family_name == "poisson":
            y = rng.poisson(np.exp(eta)).astype(float)
        else:
            y = rng.gamma(shape=3.5, scale=np.exp(eta) / 3.5)
        X = pd.DataFrame({"grp": grp, "x": x})

        model = SuperGLM(
            family=family_name,
            link=link_name,
            features={"grp": RandomEffect(), "x": Numeric()},
        ).fit_reml(X, y)

        cov = model.sandwich_covariance(cluster=grp)
        assert cov.available
        assert cov.telemetry is not None
        assert cov.telemetry.backend == "structured"
        assert cov.telemetry.bread_reused is (family_name == "poisson")
        ref = _independent_penalized_sandwich(model, y, np.ones(n), cluster=grp, centered=True)
        np.testing.assert_allclose(cov.covariance, ref, atol=_cov_tol(ref, extra=1e-7), rtol=1e-7)

        # Structured summary and term_inference work with robust covariance
        s = model.summary(covariance=cov)
        assert s["standard_errors"]["coefficient_se_scale"] == "sandwich"
        ti_re = model.term_inference("grp", covariance=cov)
        assert ti_re.covariance_n_clusters == n_groups
        assert ti_re.se_log_relativity is not None and np.all(np.isfinite(ti_re.se_log_relativity))


class TestRowScoresInTheTails:
    @pytest.mark.parametrize("clustered", [False, True])
    def test_binomial_logit_rows_past_the_mean_clip_use_one_likelihood(self, clustered):
        """Offsets of +-19 put every fitted probability past the solver's 1e-7 clip.

        The reference is the likelihood's own sandwich at the fitted intercept,
        ``sum g^2 / (sum p (1 - p))^2`` with ``p = expit(eta)`` and ``g`` the row or
        cluster sums of ``y - p``; unclustered it is ``1 / n``.
        """
        from scipy.special import expit

        n = 40
        y = np.tile([0.0, 1.0], n // 2)
        offset = np.tile([-19.0, 19.0], n // 2)
        model = SuperGLM(family="binomial", link="logit", features={}).fit(
            pd.DataFrame(index=range(n)), y, offset=offset
        )
        assert model.result.converged
        eta = model.result.intercept + offset
        assert np.all(np.abs(eta) > 16.2)  # every row is past logit(1e-7)
        score = np.where(y == 1.0, expit(-eta), -expit(eta))
        cluster = np.arange(n) // 3 if clustered else None
        units = score if cluster is None else np.bincount(cluster, weights=score)
        reference = float(units @ units) / float(np.sum(expit(eta) * expit(-eta))) ** 2

        cov = model.sandwich_covariance(cluster=cluster)
        assert cov.available
        np.testing.assert_allclose(cov.covariance[0, 0], reference, rtol=1e-10)
        if not clustered:
            np.testing.assert_allclose(cov.se[0], 1.0 / math.sqrt(n), rtol=1e-6)

    @pytest.mark.parametrize("clustered", [False, True])
    def test_binomial_logit_rows_past_the_predictor_clip_read_the_fitted_predictor(self, clustered):
        """Offsets of +-35 sit past the solver's logit band of +-20.

        The estimate is the likelihood's root (its score at the fit is rounding), so the
        reference is the likelihood's sandwich at the fitted predictor, unclipped:
        ``sum g^2 / (sum p (1 - p))^2`` with ``p = expit(eta)``.  Read at the band's edge
        the intercept's error would be 0.174180 instead of 0.223607.
        """
        from scipy.special import expit

        offset = np.tile([-19.0, 19.0, -35.0, 35.0], 10)
        y = (offset > 0.0).astype(float)
        model = SuperGLM(family="binomial", link="logit", features={}).fit(
            pd.DataFrame(index=range(len(y))), y, offset=offset
        )
        assert model.result.converged
        eta = model.result.intercept + offset
        assert np.sum(np.abs(eta) > 20.0) == 20
        score = np.where(y == 1.0, expit(-eta), -expit(eta))
        assert abs(score.sum()) < 1e-12
        cluster = np.arange(len(y)) // 3 if clustered else None
        units = score if cluster is None else np.bincount(cluster, weights=score)
        reference = float(units @ units) / float(np.sum(expit(eta) * expit(-eta))) ** 2

        cov = model.sandwich_covariance(cluster=cluster)
        assert cov.available
        np.testing.assert_allclose(cov.covariance[0, 0], reference, rtol=1e-10)
        if not clustered:
            # the twenty rows at +-19 carry the information, those at +-35 1e-7 of it
            np.testing.assert_allclose(cov.se[0], 1.0 / math.sqrt(20.0), rtol=1e-6)

    def test_negative_binomial_log_at_a_huge_theta_is_the_poisson_sandwich(self):
        """``theta**2`` overflows at 1e200; the rows are Poisson's ``y - mu`` and ``mu``."""
        X = pd.DataFrame({"x": np.tile([-1.0, 1.0, 1.0, -1.0, 0.5], 8)})
        y = np.tile([0.0, 1.0, 2.0, 3.0], 10)
        nb = SuperGLM(
            family=NegativeBinomial(theta=1e200), link="log", features={"x": Numeric()}
        ).fit(X, y)
        poisson = SuperGLM(family="poisson", features={"x": Numeric()}).fit(X, y)
        cov_nb = nb.sandwich_covariance()
        assert cov_nb.available, cov_nb.message
        np.testing.assert_allclose(
            cov_nb.covariance, poisson.sandwich_covariance().covariance, rtol=1e-6, atol=1e-12
        )


class TestCoefficientsTheFitDoesNotDetermine:
    def test_aliased_intercept_and_slopes_are_blank_in_every_robust_reader(self):
        """``b = a + 1`` leaves the direction (intercept, a, b) = (1, 1, -1) free.

        No coefficient of the three is determined, so the result, the summary
        table and dictionary, and the term readers give none an error or a p-value.
        """
        x = np.tile([-1.0, 1.0], 12)
        y = 1.0 + 2.0 * x + np.repeat(np.tile([-1.0, 1.0], 6), 2)
        model = SuperGLM(family="gaussian", features={"a": Numeric(), "b": Numeric()}).fit(
            pd.DataFrame({"a": x, "b": x + 1.0}), y
        )
        cov = model.sandwich_covariance()
        assert cov.available
        assert np.all(np.isnan(cov.se))
        assert not cov.intercept_estimable and not cov.coefficient_estimable.any()

        summary = model.summary(covariance=cov)
        rows = {row.name: row for row in summary._coef_rows}
        assert set(rows) == {"Intercept", "a", "b"}
        for row in rows.values():
            assert row.se is None or np.isnan(row.se)
            assert row.p is None or np.isnan(row.p)
        errors = summary["standard_errors"]["coefficient_se"]
        assert np.all(np.isnan(errors["a"])) and np.all(np.isnan(errors["b"]))
        assert "Intercept" in str(summary) and summary._repr_html_()

        with pytest.warns(UserWarning, match="does not determine every coefficient of 'a'"):
            term = model.term_inference("a", covariance=cov)
        assert term.se_log_relativity is None
        with pytest.warns(UserWarning, match="does not determine every coefficient"):
            frames = model.relativities(with_se=True, covariance=cov)
        assert np.all(np.isnan(frames["a"]["se_log_relativity"]))

    def test_an_alias_between_slopes_leaves_the_intercept_its_error(self):
        """``x2 = 2 x1`` is orthogonal to the intercept's contrast: only the slopes are blank.

        A control on the intercept's mask: its error and p-value stay.
        """
        rng = np.random.default_rng(910)
        x = rng.normal(size=60)
        y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=60)
        model = SuperGLM(family="gaussian", features={"x1": Numeric(), "x2": Numeric()}).fit(
            pd.DataFrame({"x1": x, "x2": 2.0 * x}), y
        )
        cov = model.sandwich_covariance()
        assert np.isfinite(cov.se[0]) and np.all(np.isnan(cov.se[1:]))
        intercept = next(
            r for r in model.summary(covariance=cov)._coef_rows if r.name == "Intercept"
        )
        np.testing.assert_allclose(intercept.se, cov.se[0], rtol=1e-12)
        assert np.isfinite(intercept.p)
        assert cov.intercept_estimable and not cov.coefficient_estimable.any()

    def test_separated_level_is_blank_in_the_result_and_the_summary_dictionary(self):
        x = np.repeat(["A", "B", "C"], 30)
        y = np.tile([0.0, 1.0, 2.0], 30)
        y[x == "C"] = 0.0
        with pytest.warns(superglm.SeparationWarning):
            model = SuperGLM(family="poisson", features={"x": Categorical(base="A")}).fit(
                pd.DataFrame({"x": x}), y
            )
        cov = model.sandwich_covariance()
        assert cov.available
        assert np.all(np.isfinite(cov.se[:2])) and np.isnan(cov.se[2])
        summary = model.summary(covariance=cov)
        errors = summary["standard_errors"]["coefficient_se"]["x"]
        np.testing.assert_allclose(errors[0], cov.se[1], rtol=1e-12)
        assert np.isnan(errors[1])
        assert next(r for r in summary._coef_rows if r.name == "x[C]").se is None


class TestWaldTestsACovarianceCanCarry:
    def test_two_clusters_carry_no_two_df_test_and_three_carry_it(self):
        """A clustered Wald test takes at most G - 1 restrictions (Cameron & Miller 2015)."""
        rng = np.random.default_rng(719)
        n = 150
        x = rng.uniform(-1.0, 1.0, n)
        y = rng.gamma(1.0, np.exp(0.7 + 1.5 * x))
        model = SuperGLM(family="gamma", link="log", features={"x": Piecewise(breaks=[0.0])}).fit(
            pd.DataFrame({"x": x}), y
        )

        def term_row(cov):
            return next(r for r in model.summary(covariance=cov)._coef_rows if r.name == "x")

        two = model.sandwich_covariance(cluster=np.arange(n) % 2)
        row = term_row(two)
        assert row.n_params == 2
        assert row.wald_chi2 is None and row.wald_p is None and row.ref_df is None
        assert two.wald_df_limit == 1

        three = model.sandwich_covariance(cluster=np.arange(n) % 3)
        assert three.wald_df_limit == 2
        row = term_row(three)
        beta = model.result.beta
        expected = float(beta @ np.linalg.solve(three.slope_covariance, beta))
        np.testing.assert_allclose(row.wald_chi2, expected, rtol=1e-8)
        assert row.ref_df == 2.0

    def test_a_penalty_on_another_term_leaves_an_unpenalized_block_its_test(self):
        rng = np.random.default_rng(72)
        x = rng.uniform(-1.0, 1.0, 150)
        z = rng.normal(size=150)
        model = SuperGLM(
            family="gaussian",
            features={"x": Piecewise(breaks=[-0.3, 0.3]), "z": Numeric()},
            penalty=Ridge(lambda1=3.0, features=["z"]),
        ).fit(pd.DataFrame({"x": x, "z": z}), 2.0 * x + z + rng.normal(size=150))
        cov = model.sandwich_covariance()
        assert cov.penalized_group_names == frozenset({"z"})
        rows = {row.name: row for row in model.summary(covariance=cov)._coef_rows}
        block = next(g for g in cov.active_groups if g.name == "x")
        beta = model.result.beta[block.sl]
        expected = float(beta @ np.linalg.solve(cov.slope_covariance[block.sl, block.sl], beta))
        assert rows["x"].wald_chi2 is not None
        np.testing.assert_allclose(rows["x"].wald_chi2, expected, rtol=1e-8)
        assert rows["x"].ref_df == 3.0
        # the ridge-shrunk coefficient itself still carries no z or p
        assert rows["z"].z is None and rows["z"].p is None and np.isfinite(rows["z"].se)


class TestUnitsAndWorkspace:
    def test_a_change_of_units_moves_exponents_and_no_digits(self):
        """Predictors rescaled by 2**332 and 2**-332 rescale their errors and nothing else.

        One exponent for the whole Gram squared the small column's influence
        (about 1e-200 of the large one) to zero.
        """
        rng = np.random.default_rng(19)
        x = np.tile([-1.0, 1.0], 32)
        z = np.tile([-1.0, -1.0, 1.0, 1.0], 16)
        y = 0.5 * x + 0.5 * z + rng.normal(size=64)

        def fitted(frame):
            model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()})
            return model.fit(frame, y).sandwich_covariance()

        base = fitted(pd.DataFrame({"x": x, "z": z}))
        scale = 2.0**332
        scaled = fitted(pd.DataFrame({"x": x * scale, "z": z / scale}))
        assert scaled.available
        np.testing.assert_allclose(scaled.se, base.se * [1.0, 1.0 / scale, scale], rtol=1e-9)
        np.testing.assert_allclose(scaled.covariance[1, 2], base.covariance[1, 2], rtol=1e-9)

    def test_the_cap_covers_every_dense_buffer_and_sizes_the_chunks(self):
        """``max_memory`` bounds the cluster buffers, the covariance and the replay chunk together."""
        n, width, n_clusters = 4096, 48, 2048
        rng = np.random.default_rng(823)
        frame = pd.DataFrame(rng.normal(size=(n, width)), columns=[f"c{i}" for i in range(width)])
        y = 0.1 * frame.to_numpy() @ rng.normal(size=width) + rng.normal(size=n)
        model = SuperGLM(family="gaussian", features={c: Numeric() for c in frame}).fit(frame, y)
        cluster = np.arange(n) % n_clusters
        free = model.sandwich_covariance(cluster=cluster, center=True)

        cap = 2_000_000
        tracemalloc.start()
        try:
            capped = model.sandwich_covariance(cluster=cluster, center=True, max_memory=cap)
            traced = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert capped.available
        assert capped.telemetry.peak_bytes <= cap
        assert capped.telemetry.n_chunks > free.telemetry.n_chunks
        np.testing.assert_allclose(capped.covariance, free.covariance, rtol=1e-10, atol=1e-18)
        # the fit's own per-row vectors are outside the cap: 32 of them is a wide envelope
        assert traced <= cap + 32 * 8 * n

        # both G x q cluster buffers alone fill this cap: nothing is left for the rest
        q = 1 + width
        refused = model.sandwich_covariance(
            cluster=cluster, center=True, max_memory=16 * n_clusters * q + 8 * q * q
        )
        assert not refused.available
        assert refused.unavailable_reason == "workspace_budget_exceeded"

    def test_the_result_shares_no_mutable_descriptor_with_the_model(self):
        rng = np.random.default_rng(31)
        X = pd.DataFrame({"x": rng.normal(size=40)})
        model = SuperGLM(family="gaussian", features={"x": Numeric()}).fit(
            X, X["x"].to_numpy() + rng.normal(size=40)
        )
        cov = model.sandwich_covariance()
        block = cov.active_groups[0]
        assert (block.name, block.start, block.end, block.size) == ("x", 0, 1, 1)
        assert not any(block is group for group in model._groups)
        with pytest.raises(dataclasses.FrozenInstanceError):
            block.end = block.start  # type: ignore[misc]


class TestFitsThatAreRegularAfterAll:
    def test_a_shape_rule_that_binds_nowhere_is_the_unconstrained_fit(self):
        x = np.linspace(-1.0, 1.0, 120)
        X = pd.DataFrame({"x": x})
        y = 3.0 * x + 0.06 * np.sin(27.0 * x)

        def fitted(constraint):
            return SuperGLM(
                family="gaussian", features={"x": Spline(kind="ps", k=6, constraint=constraint)}
            ).fit(X, y)

        ruled, free = fitted(Constraint.fit.increasing), fitted(None)
        assert ruled._solver_pirls_result().rank_info.face is None
        cov = ruled.sandwich_covariance()
        assert cov.available, cov.message
        np.testing.assert_allclose(
            ruled.term_inference("x", covariance=cov).se_log_relativity,
            free.term_inference("x", covariance=free.sandwich_covariance()).se_log_relativity,
            rtol=1e-8,
        )

    def test_a_selection_penalty_with_no_target_is_the_unpenalized_fit(self):
        X = pd.DataFrame({"band": list("abcdefgh") * 20})
        y = np.tile(np.arange(8.0), 20) + np.sin(np.arange(160.0))

        def fitted(lambda1):
            return SuperGLM(
                family="gaussian",
                features={
                    "band": OrderedCategorical(
                        order=list("abcdefgh"), basis=Piecewise(breaks=["d"])
                    )
                },
                penalty=GroupLasso(lambda1=lambda1),
            ).fit(X, y)

        lasso, free = fitted(1.0), fitted(0.0)
        assert not any(group.penalized for group in lasso._groups)
        cov = lasso.sandwich_covariance()
        assert cov.available, cov.message
        np.testing.assert_allclose(
            cov.covariance, free.sandwich_covariance().covariance, rtol=1e-10
        )
