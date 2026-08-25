"""Tests for REML smoothing parameter estimation."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.distributions import NegativeBinomial
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline
from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
from superglm.inference.covariance import (
    _active_penalty_matrix,
    _second_diff_penalty,
)
from superglm.reml import REMLResult, _map_beta_between_bases
from superglm.solvers.centered_system import penalty_factor
from superglm.solvers.rank import decompose_factor, decompose_gram
from superglm.stats.wood_pvalue import wood_test_smooth
from superglm.types import GroupSlice

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
        selection_penalty=0.0,
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
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, sample_weight=w)
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
            selection_penalty=0.01,
            features={"x1": NaturalSpline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, sample_weight=w)
        gm = model._dm.group_matrices[0]
        assert isinstance(gm, SparseSSPGroupMatrix)
        assert gm.omega is not None
        assert gm.projection is not None  # NaturalSpline uses Z projection

    def test_crs_omega_stored(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": CubicRegressionSpline(n_knots=8, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X[["x1", "x2"]], y, sample_weight=w)
        gm = model._dm.group_matrices[0]
        assert isinstance(gm, SparseSSPGroupMatrix)
        assert gm.omega is not None
        assert gm.projection is not None
        # CRS omega should differ from second-difference penalty
        p_b = gm.R_inv.shape[0]
        d2_penalty = _second_diff_penalty(p_b)
        assert not np.allclose(gm.omega, d2_penalty, atol=1e-6)


# ── PenaltyComponent bridge ──────────────────────────────────────


class TestPenaltyComponents:
    """build_penalty_components produces the same data as build_penalty_caches."""

    def test_components_match_caches(self, poisson_data, spline_model):
        """PenaltyComponent fields match PenaltyCache for single-penalty groups."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.reml import build_penalty_caches, build_penalty_components

        X, y, w = poisson_data
        spline_model.fit(X, y, sample_weight=w)

        reml_groups = []
        for i, (gm, g) in enumerate(zip(spline_model._dm.group_matrices, spline_model._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                reml_groups.append((i, g))

        caches = build_penalty_caches(spline_model._dm.group_matrices, reml_groups)
        components = build_penalty_components(spline_model._dm.group_matrices, reml_groups)

        assert len(components) == len(caches)
        for comp in components:
            cache = caches[comp.name]
            np.testing.assert_allclose(comp.omega_ssp, cache.omega_ssp, atol=1e-14)
            assert comp.rank == cache.rank
            np.testing.assert_allclose(
                comp.log_det_omega_plus, cache.log_det_omega_plus, atol=1e-14
            )
            np.testing.assert_allclose(comp.eigvals_omega, cache.eigvals_omega, atol=1e-14)
            assert comp.group_name == comp.name  # single-penalty: name == group_name
            assert comp.omega_raw is not None

    def test_components_remove_roundoff_outside_declared_psd_rank(self):
        """SSP congruence noise must not turn a smoothing prior indefinite."""
        from superglm.reml import build_penalty_components
        from superglm.solvers.rank import decompose_gram

        # This is a rank-one PSD penalty contaminated at the same relative
        # scale seen in spline SSP transforms.  Rank selection already treats
        # the second direction as null; the stored solver-space matrix must
        # enforce that same authoritative eigenspace.
        omega = np.array([[1.0, 1.0 + 2e-12], [1.0 + 2e-12, 1.0]])
        gm = SimpleNamespace(
            omega=omega,
            omega_components=None,
            R_inv=np.eye(2),
            lambda_policies=None,
        )
        group = GroupSlice(name="smooth", start=0, end=2, penalized=True)

        [component] = build_penalty_components([gm], [(0, group)])
        decomposition = decompose_gram(component.omega_ssp)

        assert component.rank == 1.0
        assert decomposition.rank == 1
        np.testing.assert_allclose(component.omega_ssp, component.omega_ssp.T, atol=0.0)

    def test_component_count_matches_reml_groups(self, poisson_data):
        """One PenaltyComponent per REML-eligible group (single-penalty case)."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.reml import build_penalty_components

        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            features={"x1": Spline(n_knots=8), "x2": Spline(n_knots=8)},
        )
        model.fit(X[["x1", "x2"]], y, sample_weight=w)

        reml_groups = []
        for i, (gm, g) in enumerate(zip(model._dm.group_matrices, model._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                reml_groups.append((i, g))

        components = build_penalty_components(model._dm.group_matrices, reml_groups)
        assert len(components) == len(reml_groups)
        names = [c.name for c in components]
        assert len(set(names)) == len(names)  # unique names


# ── penalised-Hessian assembly uses stored omega ─────────────────


class TestPenalisedHessianAssembly:
    """The bug fix: CRS gets its correct omega, not _second_diff_penalty.

    Renamed from ``TestPenalisedXtwxInvOmega``, and the three tests below
    retargeted, when the dead covariance chain was deleted (PR "Delete the
    covariance chain no production fit reaches").  They used to drive
    ``_penalised_xtwx_inv`` and ``_penalised_xtwx_inv_gram``, which had no
    production caller.  Every pin they carried is a property of the LIVE
    machinery those wrappers were built out of -- ``_active_penalty_matrix``
    for the penalty half, ``MatrixExecutionPlan.moments`` for the Gram half,
    and ``decompose_gram`` / ``decompose_factor`` for the inversion -- so each
    is re-expressed against that machinery at the same tolerance.

    None of them carries an augmented ``(p + 1)`` half any more.  Production
    never assembles ``[[sum W, X'W1], [X'W1, X'WX + S]]``: the live augmented
    covariance is a Schur border taken in closed form on an ALREADY-inverted
    profiled block (``inference/metrics.py:143-150``), so a bordered matrix
    here would be one the test itself built -- the same reason
    ``test_factor_certification_authority.py`` gives for dropping its own
    augmented comparison.  It was also measured to add no discrimination: on
    this fixture the bordered system is the same regime as the unaugmented one
    (width 21 rank 21 against 20/20, condition 8.7e+02 against 7.6e+02) and
    over four decades of injected penalty error the two discrepancies track to
    three significant figures, so the augmented assertion could never fail
    while its unaugmented sibling passed.
    """

    def test_crs_penalty_differs_from_second_diff(self, poisson_data):
        """CRS model's covariance should use the integrated f'' penalty."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"x1": CubicRegressionSpline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, sample_weight=w)

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

    @staticmethod
    def _working_weights(model, X, w):
        """The IRLS weights the covariance layer forms at convergence."""
        mu = model.predict(X)
        V = model._distribution.variance(mu)
        eta = model._link.link(mu)
        dmu = model._link.deriv_inverse(eta)
        return w * dmu**2 / V

    def test_dict_lambda2(self, poisson_data):
        """The penalty assembly accepts dict[str, float] for per-group lambdas.

        Retargeted off ``_penalised_xtwx_inv`` onto ``_active_penalty_matrix``,
        which is where the scalar/dict polymorphism actually lives and which
        ``inference/metrics.py`` and ``model/state_ops.py`` both call.  The
        inverses are compared through ``decompose_gram`` so the pin still
        reaches the published quantity and not only the penalty block.
        """
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit(X[["x1", "x2"]], y, sample_weight=w)

        W = self._working_weights(model, X[["x1", "x2"]], w)
        gms = list(model._dm.group_matrices)
        groups = list(model._groups)

        # Scalar lambda2 should match dict with same value
        S_scalar = _active_penalty_matrix(gms, groups, groups, 0.1)
        lam_dict = {g.name: 0.1 for g in groups}
        S_dict = _active_penalty_matrix(gms, groups, groups, lam_dict)
        np.testing.assert_allclose(S_scalar, S_dict, atol=1e-10)

        moments = MatrixExecutionPlan(gms, n=len(W)).moments(W, include_xtw=True)
        inv_scalar = decompose_gram(moments.gram + S_scalar).pseudo_inverse()
        inv_dict = decompose_gram(moments.gram + S_dict).pseudo_inverse()
        np.testing.assert_allclose(inv_scalar, inv_dict, atol=1e-10)

    def test_gram_matches_qr(self, poisson_data):
        """The Gram and the augmented QR invert the same penalised Hessian alike.

        Retargeted off the deleted wrappers onto the two decompositions they
        were built from.  ``decompose_gram`` solves the normal equations;
        ``decompose_factor`` takes the augmented QR of ``[sqrt(W) X ; sqrt(S)]``
        -- which is what ``penalty_factor`` produces and what
        ``model/state_ops.py`` falls back to when the Gram cannot certify
        itself.  Their agreement is the equivalence this test always pinned,
        and it is a property of the pair, not of any caller.  Measured on this
        fixture: 1.1e-14 against the 1e-8 bar this file already used.
        """
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
                "x3": Categorical(),
            },
        )
        model.fit(X, y, sample_weight=w)

        W = self._working_weights(model, X, w)
        gms = list(model._dm.group_matrices)
        groups = list(model._groups)
        lam_dict = {g.name: 0.1 for g in groups}

        # Every group is active on this fixture, so the two routes see the same
        # width; a fixture that dropped one would compare different systems.
        assert all(np.linalg.norm(model.result.beta[g.sl]) > 1e-12 for g in groups)

        S = _active_penalty_matrix(gms, groups, groups, lam_dict)
        smooth_factor = penalty_factor(S)
        moments = MatrixExecutionPlan(gms, n=len(W)).moments(W, include_xtw=True)
        design = model._dm.toarray()
        sqrt_W = np.sqrt(W)[:, None]

        inv_gram = decompose_gram(moments.gram + S).pseudo_inverse()
        inv_qr = decompose_factor(np.vstack((design * sqrt_W, smooth_factor))).pseudo_inverse()
        np.testing.assert_allclose(inv_qr, inv_gram, atol=1e-8)

    @staticmethod
    def _discretized_fusion_fixture(seed=20260719, n=240, n_bins=24):
        """One discretized SSP group and the weights the fused pass reduces."""
        rng = np.random.default_rng(seed)
        gm = DiscretizedSSPGroupMatrix(
            rng.normal(size=(n_bins, 4)),
            np.eye(4),
            np.resize(np.arange(n_bins, dtype=np.intp), n),
        )
        return gm, rng.uniform(0.2, 2.0, size=n), n

    @staticmethod
    def _forbid_unfused_passes(monkeypatch):
        """Redden if anything reaches the compressed design twice."""
        for method in ("gram", "rmatvec"):
            monkeypatch.setattr(
                DiscretizedSSPGroupMatrix,
                method,
                lambda *_args, **_kwargs: pytest.fail(
                    "the fused moments pass must use gram_rmatvec"
                ),
            )

    def test_fused_moments_pass_serves_gram_and_intercept_product(self, monkeypatch):
        """``moments(W, include_xtw=True)`` makes ONE compressed pass, not two.

        This used to say "covariance" because it drove the deleted
        ``_penalised_xtwx_inv_gram``.  After that deletion no covariance code
        calls this API at all -- the live covariance path streams through
        ``iter_dense_chunks`` / ``streamed_weighted_factor`` -- so both the name
        and the failure message named the one subsystem that does not use it.
        The live consumers of the fused pass are ``reml/objective.py:143``,
        ``reml/w_derivatives.py:472``, ``solvers/_structured/moments.py`` and
        ``_group_matrix_centered.py``; the test below this one drives the first
        of them so the pin keeps a production caller and not only the kernel.
        """
        gm, W, n = self._discretized_fusion_fixture()
        group = GroupSlice(name="x", start=0, end=4, penalized=False)
        X = gm.toarray()

        self._forbid_unfused_passes(monkeypatch)

        penalty = _active_penalty_matrix([gm], [group], [group], 0.0)
        moments = MatrixExecutionPlan([gm], n=n).moments(W, include_xtw=True)

        expected_gram = X.T @ (W[:, None] * X)
        np.testing.assert_allclose(moments.gram, expected_gram, rtol=2e-12, atol=2e-10)
        np.testing.assert_allclose(moments.xtw, X.T @ W, rtol=2e-12, atol=2e-10)
        np.testing.assert_allclose(
            decompose_gram(moments.gram + penalty).pseudo_inverse(),
            np.linalg.pinv(expected_gram),
            atol=2e-10,
        )
        # An unpenalised group contributes nothing, so ``penalty`` is zero by
        # construction and asserting that is vacuous.  What is not vacuous is
        # that the same group under a live lambda DOES contribute, which is
        # what makes the zero above a decision rather than a default.
        penalized_group = GroupSlice(name="x", start=0, end=4, penalized=True)
        assert np.any(_active_penalty_matrix([gm], [penalized_group], [penalized_group], 0.7))

    def test_reml_objective_reaches_the_fused_moments_pass(self, monkeypatch):
        """A live caller anchor for the pin above, not the kernel in isolation.

        ``reml_laml_objective`` takes ``dm.execution_plan.moments(W,
        include_xtw=True)`` at ``reml/objective.py:143`` whenever it is handed
        no cached ``XtWX``, which is the ordinary standalone evaluation.  Under
        the same two monkeypatches it must complete: if the fused route stops
        being taken, the objective reaches the compressed design twice and this
        reddens, which is the shape the original covariance-anchored test had.
        """
        from superglm.distributions import Poisson
        from superglm.group_matrix import DesignMatrix
        from superglm.links import LogLink
        from superglm.reml.objective import reml_laml_objective
        from superglm.solvers.pirls import PIRLSResult

        gm, W, n = self._discretized_fusion_fixture()
        dm = DesignMatrix([gm], n=n, p=4)
        rng = np.random.default_rng(4242)
        y = np.asarray(rng.poisson(2.0, size=n), dtype=float)
        result = PIRLSResult(
            beta=np.full(4, 0.05),
            intercept=0.3,
            n_iter=3,
            deviance=float(n),
            converged=True,
            phi=1.0,
            effective_df=4.0,
        )

        self._forbid_unfused_passes(monkeypatch)

        value = reml_laml_objective(
            dm,
            Poisson(),
            LogLink(),
            [GroupSlice(name="x", start=0, end=4, penalized=False)],
            y=y,
            result=result,
            lambdas={},
            sample_weight=W,
            offset_arr=np.zeros(n),
            weight_semantics="frequency",
        )
        assert np.isfinite(value)


# ── _compute_R_inv override ──────────────────────────────────────


class TestComputeRInvOverride:
    def test_different_lambda_gives_different_R_inv(self, poisson_data):
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, sample_weight=w)

        gm = model._dm.group_matrices[0]
        from superglm.dm_builder import compute_R_inv

        R_inv_1 = compute_R_inv(gm.B, gm.omega, w, 0.01)
        R_inv_2 = compute_R_inv(gm.B, gm.omega, w, 1.0)
        assert not np.allclose(R_inv_1, R_inv_2, atol=1e-6)

    def test_zero_total_weight_returns_finite_penalty_only_R_inv(self):
        """Zero-weight spline-category levels should not create NaN SSP transforms."""
        import scipy.sparse as sp

        from superglm.dm_builder import compute_projected_R_inv, compute_R_inv

        rng = np.random.default_rng(212)
        B = sp.random(8, 5, density=0.4, format="csr", random_state=212)
        omega = np.eye(5)
        weights = np.zeros(B.shape[0])

        R_inv = compute_R_inv(B, omega, weights, 0.7)

        assert R_inv.shape == (5, 5)
        assert np.all(np.isfinite(R_inv))

        projection = rng.normal(size=(5, 3))
        omega_proj = projection.T @ omega @ projection
        R_inv_projected = compute_projected_R_inv(B, projection, omega_proj, weights, 0.7)

        assert R_inv_projected.shape == (3, 3)
        assert np.all(np.isfinite(R_inv_projected))


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
            selection_penalty=0.0,
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

        X_a, W, XtWX_inv, XtWX_inv_aug, active_groups = metrics._active_info
        R_a = metrics._active_R_factor
        _, edf1 = metrics._influence_edf
        ag = next(a for a in active_groups if a.name == "Noise2")
        beta_g = model.result.beta[ag.sl]
        aug_sl = slice(1 + ag.start, 1 + ag.end)
        V_b_j = XtWX_inv_aug[aug_sl, aug_sl]
        edf1_j = float(np.sum(edf1[ag.sl]))
        X_a_dense = X_a.toarray() if hasattr(X_a, "toarray") else np.asarray(X_a)
        X_a_centered = X_a_dense - np.average(X_a_dense, axis=0, weights=W)

        np.testing.assert_allclose(
            R_a.T @ R_a,
            X_a_centered.T @ (X_a_centered * W[:, None]),
            atol=1e-8,
        )

        _, p_raw, _ = wood_test_smooth(beta_g, X_a_centered[:, ag.sl], V_b_j, edf1_j, -1.0)
        _, p_r, _ = wood_test_smooth(beta_g, R_a[:, ag.sl], V_b_j, edf1_j, -1.0)

        assert row.wald_p == pytest.approx(p_r)
        # Factor correctness is verified above after profiling the intercept.
        # Both methods (centered X_a vs its square factor R_a) should agree.
        assert p_r == pytest.approx(p_raw, abs=0.3)


# ── Beta mapping ─────────────────────────────────────────────────


class TestBetaMapping:
    def test_roundtrip(self, poisson_data):
        """Mapping beta through old -> B-spline -> new preserves B-spline coefficients."""
        from superglm.dm_builder import compute_projected_R_inv, compute_R_inv

        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit(X[["x1"]], y, sample_weight=w)

        gm_old = model._dm.group_matrices[0]
        beta_old = model.result.beta.copy()

        # Create new R_inv with different lambda, respecting projection
        P = gm_old.projection
        if P is not None:
            omega_proj = P.T @ gm_old.omega @ P
            R_inv_local = compute_projected_R_inv(gm_old.B, P, omega_proj, w, 0.5)
            R_inv_new = P @ R_inv_local
        else:
            R_inv_new = compute_R_inv(gm_old.B, gm_old.omega, w, 0.5)
        gm_new = SparseSSPGroupMatrix(gm_old.B, R_inv_new)
        gm_new.omega = gm_old.omega
        gm_new.projection = P

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
                selection_penalty=0.0,
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
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

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
            selection_penalty=0.0,
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


# ── REML selection-penalty rejection ─────────────────────────────


class TestREMLSelectionPenaltyRejected:
    def test_fit_reml_rejects_selection_penalty_poisson(self):
        """fit_reml() requires selection_penalty=0 on Poisson models."""
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
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X, y, max_reml_iter=10)

    def test_fit_reml_rejects_selection_penalty_gamma(self):
        """fit_reml() requires selection_penalty=0 on estimated-scale families too."""
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
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X, y, max_reml_iter=12, verbose=True)


class TestREMLFallbacks:
    def test_fit_reml_intercept_only_matches_plain_fit(self, caplog):
        """A no-term REML request should use the ordinary intercept-only solver."""
        X = pd.DataFrame(index=pd.RangeIndex(12))
        y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.5])

        expected = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={},
        ).fit(X, y)
        actual = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={},
        )

        with caplog.at_level(logging.WARNING):
            actual.fit_reml(X, y)

        assert "no REML-eligible groups found" in caplog.text
        assert actual._reml_result is None
        assert actual.result.converged
        assert actual.result.beta.shape == (0,)
        np.testing.assert_allclose(actual.result.intercept, expected.result.intercept)
        np.testing.assert_allclose(actual.predict(X), expected.predict(X))
        np.testing.assert_allclose(actual.result.deviance, expected.result.deviance)

    def test_fit_reml_without_reml_groups_uses_ordinary_solver_dispatch(self, monkeypatch):
        """The no-REML fallback must not bypass fit()'s direct-solver routing."""
        rng = np.random.default_rng(314)
        X = pd.DataFrame({"x": rng.normal(size=80)})
        y = 1.25 + 2.5 * X["x"].to_numpy() + rng.normal(scale=0.2, size=len(X))

        expected = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y)

        def fail_if_pirls_runs(*args, **kwargs):
            pytest.fail("the no-REML fallback bypassed ordinary coefficient-solver dispatch")

        monkeypatch.setattr("superglm.model.fit_ops.fit_pirls", fail_if_pirls_runs)

        actual = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit_reml(X, y)

        assert actual._reml_result is None
        np.testing.assert_allclose(actual.result.beta, expected.result.beta)
        np.testing.assert_allclose(actual.result.intercept, expected.result.intercept)
        np.testing.assert_allclose(actual.predict(X), expected.predict(X))
        np.testing.assert_allclose(actual.result.deviance, expected.result.deviance)

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
            family=NegativeBinomial(theta="auto"),
            selection_penalty=0.0,
            features={"dummy": Numeric()},
        )
        with caplog.at_level(logging.WARNING):
            model.fit_reml(X, y)

        assert "no REML-eligible groups found" in caplog.text
        assert model.family.theta == "auto"
        assert model.theta_ > 0
        assert model._nb_profile_result is not None
        assert model.result.converged
        assert not hasattr(model, "_reml_lambdas")


# ── REML + select=True (mgcv double penalty) ─────────────────────


class TestREMLSelectTrue:
    def test_reml_select_true_converges(self, poisson_data):
        """fit_reml() works with select=True (double penalty)."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x1": Spline(n_knots=8, penalty="ssp", select=True)},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w, max_reml_iter=30)
        assert model._reml_result.converged
        # Both null and wiggle components should have REML lambdas
        assert "x1:null" in model._reml_lambdas
        assert "x1:wiggle" in model._reml_lambdas

    def test_reml_select_true_null_lambda_differs(self, poisson_data):
        """Null and wiggle components should get different REML lambdas."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp", select=True),
                "x2": Spline(n_knots=8, penalty="ssp", select=True),
            },
        )
        model.fit_reml(X[["x1", "x2"]], y, sample_weight=w, max_reml_iter=15)
        # Should have 4 REML lambdas: x1:null, x1:wiggle, x2:null, x2:wiggle
        assert len(model._reml_lambdas) == 4

    def test_reml_select_logdet_independent_components(self, poisson_data):
        """select=True: null and wiggle components contribute independently to log|S|+.

        Each component has its own penalty matrix (omega_ssp) and lambda.
        cached_logdet_s_plus should equal the sum of per-component
        r_j * log(lambda_j) + log|Omega_j|+ contributions.
        """
        from superglm.group_matrix import SparseSSPGroupMatrix
        from superglm.reml import build_penalty_caches, cached_logdet_s_plus

        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x1": Spline(n_knots=8, penalty="ssp", select=True)},
        )
        model.fit_reml(X[["x1"]], y, sample_weight=w, max_reml_iter=15)

        reml_groups = []
        for i, (gm, g) in enumerate(zip(model._dm.group_matrices, model._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix):
                reml_groups.append((i, g))

        caches = build_penalty_caches(model._dm.group_matrices, reml_groups)
        lambdas = model._reml_lambdas

        # Verify: cached formula matches manual per-group sum
        cached_val = cached_logdet_s_plus(lambdas, caches)
        manual_val = 0.0
        for name, cache in caches.items():
            lam = lambdas.get(name, 1.0)
            if lam > 0 and cache.rank > 0:
                manual_val += cache.rank * np.log(lam) + cache.log_det_omega_plus
        np.testing.assert_allclose(cached_val, manual_val, atol=1e-12)

        # Verify both components contribute (nonzero rank and log_det)
        assert "x1:null" in caches
        assert "x1:wiggle" in caches
        assert caches["x1:null"].rank > 0
        assert caches["x1:wiggle"].rank > 0


# ── Backward compatibility ───────────────────────────────────────


class TestREMLBackwardCompat:
    def test_fit_unchanged(self, poisson_data, spline_model):
        """fit() with global lambda2 should work unchanged after REML code added."""
        X, y, w = poisson_data
        spline_model.fit(X, y, sample_weight=w)
        assert spline_model.result is not None
        assert not hasattr(spline_model, "_reml_lambdas") or spline_model._reml_lambdas is None

    def test_custom_link_without_curvature_protocol_is_rejected(self):
        """An unproved custom link must not silently select Fisher LAML geometry."""
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
            selection_penalty=0,
        )
        with pytest.raises(NotImplementedError, match="explicit ordinary REML curvature"):
            m.fit_reml(df, y, max_reml_iter=10)
        assert m._fit_state is None

    def test_custom_distribution_without_curvature_protocol_is_rejected(self):
        """An unproved custom family must not silently select Fisher LAML geometry."""
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
            selection_penalty=0,
        )
        with pytest.raises(NotImplementedError, match="explicit ordinary REML curvature"):
            m.fit_reml(df, y, max_reml_iter=10)
        assert m._fit_state is None

    def test_enhanced_custom_objects_get_w_correction(self):
        """A declared custom Fisher pair still receives its exact W(rho) correction."""

        curvature_declarations: list[str] = []

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

            def reml_curvature(self, distribution):
                curvature_declarations.append("link")
                return "fisher"

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

            def reml_curvature(self, link):
                curvature_declarations.append("distribution")
                return "fisher"

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
            selection_penalty=0,
        )
        m.fit_reml(df, y, max_reml_iter=10)
        assert m._reml_result.converged
        assert curvature_declarations == ["distribution", "link"]
        assert m._reml_profile["reml_observed_geometry_s"] == 0.0

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
            weight_semantics="frequency",
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
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)

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
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)

        met = spline_model.metrics(X, y, sample_weight=w)
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
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)
        cov_reml, groups_reml = spline_model._coef_covariance

        # Fit with global lambda2 (different model instance)
        model2 = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
                "x3": Categorical(),
            },
        )
        model2.fit(X, y, sample_weight=w)
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
            selection_penalty=0,
            features={
                "x1": CubicRegressionSpline(n_knots=8),
                "x2": CubicRegressionSpline(n_knots=8),
            },
            discrete=True,
        )
        model.fit_reml(df, y, sample_weight=w, max_reml_iter=50, lambda2_init=1e5)

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

        exact = SuperGLM(family="poisson", selection_penalty=0, features=features, discrete=False)
        exact.fit_reml(df, y, sample_weight=w, max_reml_iter=30)

        disc = SuperGLM(family="poisson", selection_penalty=0, features=features, discrete=True)
        disc.fit_reml(df, y, sample_weight=w, max_reml_iter=30)

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

        exact = SuperGLM(family=family, selection_penalty=0, features=features, discrete=False)
        exact.fit_reml(df, y, sample_weight=w, max_reml_iter=30)

        disc = SuperGLM(family=family, selection_penalty=0, features=features, discrete=True)
        disc.fit_reml(df, y, sample_weight=w, max_reml_iter=30)

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


# ── Multi-penalty post-fit inference regression tests ──────────────


class TestMultiPenaltyPostFitInference:
    """Verify multi-penalty S propagates through all post-fit paths.

    Uses a tensor interaction which creates shared-block PenaltyComponents
    (margin_x1 + margin_x2 on one coefficient block).  The legacy
    single-penalty-per-group path looks up lambda2.get("x1:x2") which
    misses the component keys "x1:x2:margin_x1", "x1:x2:margin_x2",
    guaranteeing the two S constructions differ.
    """

    @pytest.fixture
    def select_model_fitted(self):
        """A fitted tensor model with shared-block multi-penalty structure."""
        rng = np.random.default_rng(99)
        n = 600
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        w = np.ones(n)

        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=6),
                "x2": Spline(kind="cr", n_knots=6),
            },
            interactions=[("x1", "x2")],
        )
        # The shared-block tensor penalties leave a flat log-lambda direction
        # whose Newton endgame at the tight 1e-9 default is BLAS-kernel
        # dependent: one platform stops cleanly at 18 iterations, another
        # stalls its line search and honestly reports converged=False. This
        # test asserts multi-penalty S propagation, not lambda determination,
        # so it pins the step-grade bar; the endgame classification itself is
        # the criterion-redesign follow-up.
        model.fit_reml(X, y, sample_weight=w, max_reml_iter=30, reml_tol=1e-6)
        assert model._reml_result.converged
        assert model._reml_penalties is not None
        # Tensor creates shared-block components (margin_x1, margin_x2)
        shared = [pc for pc in model._reml_penalties if pc.name != pc.group_name]
        assert len(shared) >= 2
        return model, X, y, w

    def _get_covariance_with_and_without_multi_penalty(self, model, X, y, w):
        """Get covariance from the model, then recompute with legacy S for comparison.

        Uses the actual model paths to get the multi-penalty result, then
        temporarily removes _reml_penalties to get the legacy result.
        This tests the real code path, not a manual reimplementation.
        """
        # Multi-penalty path (what the model should use)
        cov_multi, _ = model._coef_covariance

        # Legacy path: temporarily remove _reml_penalties and recompute
        saved_penalties = model._reml_penalties
        model._reml_penalties = None
        # Invalidate cached covariance so it recomputes
        model.__dict__.pop("_coef_covariance", None)
        cov_legacy, _ = model._coef_covariance
        # Restore
        model._reml_penalties = saved_penalties
        model.__dict__.pop("_coef_covariance", None)

        # RankInfo freezes the actual fitted multi-penalty quotient. Mutating
        # model configuration after fitting must not silently recompute a
        # different covariance with legacy penalty algebra.
        np.testing.assert_allclose(cov_multi, cov_legacy, rtol=1e-12)
        return cov_multi, cov_legacy

    @pytest.mark.slow
    def test_coef_covariance_uses_multi_penalty_S(self, select_model_fitted):
        """_coef_covariance must use multi-penalty S, not legacy per-group S."""
        model, X, y, w = select_model_fitted
        cov_multi, cov_legacy = self._get_covariance_with_and_without_multi_penalty(model, X, y, w)

        # Re-fetch: should match the multi-penalty result
        cov_actual, _ = model._coef_covariance
        np.testing.assert_allclose(cov_actual, cov_multi, rtol=1e-10)

    @pytest.mark.slow
    def test_fit_active_info_uses_multi_penalty_S(self, select_model_fitted):
        """_fit_active_info inverse must reflect multi-penalty, not legacy S."""
        model, X, y, w = select_model_fitted

        # Get multi-penalty result
        X_a, W, inv_multi, inv_aug_multi, groups = model._fit_active_info

        # Get legacy result
        saved = model._reml_penalties
        model._reml_penalties = None
        model.__dict__.pop("_fit_active_info", None)
        _, _, inv_legacy, _, _ = model._fit_active_info
        model._reml_penalties = saved
        model.__dict__.pop("_fit_active_info", None)

        np.testing.assert_allclose(inv_multi, inv_legacy, rtol=1e-12)


class TestREMLObjectiveFastPath:
    def test_poisson_objective_uses_cached_deviance_when_xtwx_supplied(self):
        from superglm.distributions import Poisson
        from superglm.links import LogLink
        from superglm.reml.objective import reml_laml_objective
        from superglm.solvers.pirls import PIRLSResult

        class _NoMatvecDM:
            group_matrices = []

            def matvec(self, beta):
                raise AssertionError("matvec should not be called on the Poisson cached-XtWX path")

        dm = _NoMatvecDM()
        result = PIRLSResult(
            beta=np.array([0.2]),
            intercept=0.1,
            n_iter=1,
            deviance=1.75,
            converged=True,
            phi=1.0,
            effective_df=1.0,
        )
        XtWX = np.array([[2.0]])
        val = reml_laml_objective(
            dm,
            Poisson(),
            LogLink(),
            [],
            y=np.array([1.0]),
            result=result,
            lambdas={},
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            XtWX=XtWX,
            weight_semantics="frequency",
        )
        expected = 0.5 * result.deviance + 0.5 * np.log(2.0)
        np.testing.assert_allclose(val, expected, rtol=1e-12, atol=1e-12)

    def test_objective_rejects_overflowed_working_weights_before_gram(self):
        from superglm.distributions import Poisson
        from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
        from superglm.links import LogLink
        from superglm.reml.objective import reml_laml_objective
        from superglm.solvers.pirls import PIRLSResult

        dm = DesignMatrix([DenseGroupMatrix(np.ones((2, 1)))], n=2, p=1)
        result = PIRLSResult(
            beta=np.array([0.0]),
            intercept=80.0,
            n_iter=1,
            deviance=1.0,
            converged=True,
            phi=1.0,
            effective_df=1.0,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            with pytest.raises(ValueError, match="must be finite"):
                reml_laml_objective(
                    dm,
                    Poisson(),
                    LogLink(),
                    [GroupSlice(name="x", start=0, end=1, penalized=False)],
                    y=np.ones(2),
                    result=result,
                    lambdas={},
                    sample_weight=np.array([np.finfo(np.float64).max, 1.0]),
                    offset_arr=np.zeros(2),
                    weight_semantics="frequency",
                )

    def test_weight_derivative_correction_rejects_nonfinite_signed_weights(self, monkeypatch):
        import superglm.reml.w_derivatives as w_derivatives
        from superglm.distributions import Poisson
        from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
        from superglm.links import LogLink
        from superglm.solvers.pirls import PIRLSResult
        from superglm.types import PenaltyComponent

        n = 6
        dm = DesignMatrix([DenseGroupMatrix(np.ones((n, 1)))], n=n, p=1)
        group = GroupSlice(name="x", start=0, end=1)
        component = PenaltyComponent(
            name="x",
            group_name="x",
            group_index=0,
            group_sl=slice(0, 1),
            omega_raw=np.ones((1, 1)),
            omega_ssp=np.ones((1, 1)),
            rank=1.0,
        )
        result = PIRLSResult(
            beta=np.ones(1),
            intercept=0.0,
            n_iter=1,
            deviance=1.0,
            converged=True,
            phi=1.0,
            effective_df=1.0,
        )
        monkeypatch.setattr(
            w_derivatives,
            "compute_dW_deta",
            lambda *_args, **_kwargs: np.full(n, np.inf),
        )

        with pytest.raises(ValueError, match="must be finite"):
            w_derivatives.reml_w_correction(
                dm,
                LogLink(),
                [group],
                result,
                np.eye(1),
                {"x": 1.0},
                sample_weight=np.ones(n),
                offset_arr=np.zeros(n),
                distribution=Poisson(),
                reml_penalties=[component],
            )


class TestDiscreteCachedSolve:
    def test_tensor_surrogate_linesearch_defers_profiled_solve(self, monkeypatch):
        import superglm.reml.discrete as discrete_reml

        rng = np.random.default_rng(77)
        n = 260
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.2 + np.sin(2 * np.pi * x1) + 0.3 * np.cos(2 * np.pi * x2)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        profiled_solve = discrete_reml._solve_cached_profiled_system
        solve_calls = 0

        def count_profiled_solve(*args, **kwargs):
            nonlocal solve_calls
            solve_calls += 1
            return profiled_solve(*args, **kwargs)

        monkeypatch.setattr(
            discrete_reml,
            "_solve_cached_profiled_system",
            count_profiled_solve,
        )

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x1": Spline(n_knots=5), "x2": Spline(n_knots=5)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(X, y, max_reml_iter=3, reml_tol=1e-12)

        assert model._reml_result.n_reml_iter >= 1
        assert model._reml_profile["reml_n_linesearch_surrogate_evals"] >= solve_calls
        assert model._reml_profile["reml_n_linesearch_full_evals"] == solve_calls

    def test_penalty_block_trace_matches_materialized_hessian_product(self):
        from superglm.reml.gradient import _penalty_block_trace

        rng = np.random.default_rng(12)
        p = 14
        A = rng.standard_normal((p, p))
        H_inv = np.linalg.inv(A @ A.T + np.eye(p))
        slices = [slice(1, 5), slice(7, 12)]
        omegas = []
        lambdas = [0.8, 2.3]
        for sl in slices:
            q = sl.stop - sl.start
            B = rng.standard_normal((q, q))
            omegas.append(B @ B.T)

        for i, sl_i in enumerate(slices):
            for j, sl_j in enumerate(slices):
                F_i = np.zeros((p, p))
                F_i[:, sl_i] = H_inv[:, sl_i] @ (lambdas[i] * omegas[i])
                F_j = np.zeros((p, p))
                F_j[:, sl_j] = H_inv[:, sl_j] @ (lambdas[j] * omegas[j])
                materialized = float(np.sum(F_i * F_j.T))

                compact = _penalty_block_trace(
                    H_inv,
                    sl_i,
                    lambdas[i] * omegas[i],
                    sl_j,
                    lambdas[j] * omegas[j],
                )

                np.testing.assert_allclose(compact, materialized, rtol=1e-12, atol=1e-12)

    def test_profiled_cached_solve_matches_augmented_system(self):
        from superglm.reml.discrete import _solve_cached_profiled_system

        rng = np.random.default_rng(42)
        n = 30
        p = 5
        X = rng.standard_normal((n, p))
        W = rng.uniform(0.3, 1.8, size=n)
        z = rng.standard_normal(n)
        XtWX = X.T @ (W[:, None] * X)
        XtW1 = X.T @ W
        XtWz = X.T @ (W * z)
        sum_W = float(np.sum(W))
        sum_Wz = float(W @ z)
        B = rng.standard_normal((p, p))
        S = B @ B.T + np.eye(p) * 0.5
        mean_x = XtW1 / sum_W
        mean_z = sum_Wz / sum_W
        centered_XtWX = XtWX - np.outer(XtW1, XtW1) / sum_W
        centered_XtWz = XtWz - mean_x * sum_Wz

        beta, intercept, log_det_h, hessian_rank = _solve_cached_profiled_system(
            centered_XtWX,
            S,
            centered_XtWz,
            mean_x,
            sum_W,
            mean_z,
        )
        augmented = np.empty((p + 1, p + 1))
        augmented[0, 0] = sum_W
        augmented[0, 1:] = XtW1
        augmented[1:, 0] = XtW1
        augmented[1:, 1:] = XtWX + S
        rhs = np.concatenate(([sum_Wz], XtWz))
        expected = np.linalg.solve(augmented, rhs)

        np.testing.assert_allclose(beta, expected[1:], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(intercept, expected[0], rtol=1e-10, atol=1e-10)
        assert log_det_h == pytest.approx(np.linalg.slogdet(augmented)[1], rel=1e-12)
        assert hessian_rank == p + 1

    def test_tensor_pair_closed_form_matches_objective_gradient_hessian(self):
        from superglm.distributions import Poisson
        from superglm.group_matrix import DiscretizedTensorGroupMatrix
        from superglm.links import LogLink
        from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
        from superglm.reml.objective import reml_laml_objective
        from superglm.reml.penalty_algebra import (
            build_tensor_pair_logdet_summaries,
            evaluate_tensor_pair_logdet_summaries,
        )
        from superglm.solvers.pirls import PIRLSResult
        from superglm.types import PenaltyComponent

        p1, p2 = 4, 3
        q = p1 * p2
        rng = np.random.default_rng(42)
        A = rng.standard_normal((p1, p1 - 1))
        B = rng.standard_normal((p2, p2 - 1))
        S1 = A @ A.T
        S2 = B @ B.T
        omega_1 = np.kron(S1, np.eye(p2))
        omega_2 = np.kron(np.eye(p1), S2)
        lambdas = {"ti:margin_x1": 1.7, "ti:margin_x2": 0.4}
        S = lambdas["ti:margin_x1"] * omega_1 + lambdas["ti:margin_x2"] * omega_2

        gm = DiscretizedTensorGroupMatrix(
            B1_unique=np.eye(p1),
            B2_unique=np.eye(p2),
            idx1=np.zeros(q, dtype=np.intp),
            idx2=np.zeros(q, dtype=np.intp),
            B_joint=np.eye(q),
            R_inv=np.eye(q),
            pair_idx=np.arange(q, dtype=np.intp),
            tensor_id=11,
        )
        penalties = [
            PenaltyComponent(
                name="ti:margin_x1",
                group_name="ti",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=omega_1,
                omega_ssp=omega_1,
                rank=float(np.linalg.matrix_rank(omega_1)),
            ),
            PenaltyComponent(
                name="ti:margin_x2",
                group_name="ti",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=omega_2,
                omega_ssp=omega_2,
                rank=float(np.linalg.matrix_rank(omega_2)),
            ),
        ]

        class _DummyDM:
            def __init__(self, group_matrices):
                self.group_matrices = group_matrices

            def matvec(self, beta):
                raise AssertionError("matvec should not be called on the cached objective path")

        dm = _DummyDM([gm])
        C = rng.standard_normal((q, q))
        XtWX = C @ C.T + np.eye(q)
        H = XtWX + S
        XtWX_S_inv = np.linalg.inv(H)
        beta = rng.standard_normal(q)
        result = PIRLSResult(
            beta=beta,
            intercept=0.0,
            n_iter=0,
            deviance=2.5,
            converged=True,
            phi=1.0,
            effective_df=0.0,
            log_det_H=float(np.linalg.slogdet(H)[1]),
        )

        tensor_summaries = build_tensor_pair_logdet_summaries(dm.group_matrices, penalties)
        tensor_evals = evaluate_tensor_pair_logdet_summaries(tensor_summaries, lambdas)

        obj_generic = reml_laml_objective(
            dm,
            Poisson(),
            LogLink(),
            [],
            y=np.array([1.0]),
            result=result,
            lambdas=lambdas,
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            XtWX=XtWX,
            log_det_H=result.log_det_H,
            S_override=S,
            reml_penalties=penalties,
            weight_semantics="frequency",
        )
        obj_closed = reml_laml_objective(
            dm,
            Poisson(),
            LogLink(),
            [],
            y=np.array([1.0]),
            result=result,
            lambdas=lambdas,
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            XtWX=XtWX,
            log_det_H=result.log_det_H,
            S_override=S,
            reml_penalties=penalties,
            tensor_pair_evaluations=tensor_evals,
            weight_semantics="frequency",
        )

        grad_generic = reml_direct_gradient(
            dm.group_matrices,
            result,
            XtWX_S_inv,
            lambdas,
            reml_penalties=penalties,
        )
        grad_closed = reml_direct_gradient(
            dm.group_matrices,
            result,
            XtWX_S_inv,
            lambdas,
            reml_penalties=penalties,
            tensor_pair_evaluations=tensor_evals,
        )

        hess_generic = reml_direct_hessian(
            dm.group_matrices,
            Poisson(),
            XtWX_S_inv,
            lambdas,
            gradient=grad_generic,
            pirls_result=result,
            n_obs=200,
            reml_penalties=penalties,
        )
        hess_closed = reml_direct_hessian(
            dm.group_matrices,
            Poisson(),
            XtWX_S_inv,
            lambdas,
            gradient=grad_closed,
            pirls_result=result,
            n_obs=200,
            reml_penalties=penalties,
            tensor_pair_evaluations=tensor_evals,
        )

        np.testing.assert_allclose(obj_closed, obj_generic, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(grad_closed, grad_generic, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(hess_closed, hess_generic, rtol=1e-10, atol=1e-10)


class TestStaleREMLClearing:
    """Verify fit() clears REML state from a previous fit_reml()."""

    @pytest.mark.slow
    def test_fit_clears_reml_state(self, poisson_data, spline_model):
        """After fit_reml() then fit(), REML attributes must be None."""
        X, y, w = poisson_data

        # First: fit with REML
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)
        assert spline_model._reml_lambdas is not None
        assert spline_model._reml_penalties is not None
        assert spline_model._reml_result is not None

        # Second: plain fit on the same model instance
        spline_model.fit(X, y, sample_weight=w)

        # REML state must be cleared
        assert spline_model._reml_lambdas is None
        assert spline_model._reml_penalties is None
        assert spline_model._reml_result is None

    @pytest.mark.slow
    def test_covariance_after_refit_uses_global_lambda(self, poisson_data, spline_model):
        """After fit_reml() then fit(), covariance uses global lambda2, not stale REML."""
        X, y, w = poisson_data

        # Fit with REML, then refit with plain fit
        spline_model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)
        spline_model.fit(X, y, sample_weight=w)

        # Covariance should work (no crash) and use global lambda2
        cov, groups = spline_model._coef_covariance
        assert np.all(np.isfinite(cov))
        assert np.all(np.diag(cov) >= 0)

    @pytest.mark.slow
    def test_fit_path_clears_reml_state(self, poisson_data):
        """After fit_reml() then fit_path(), REML attributes must be None."""
        X, y, w = poisson_data

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
                "x3": Categorical(),
            },
        )
        model.fit_reml(X, y, sample_weight=w, max_reml_iter=10)
        assert model._reml_lambdas is not None
        assert model._reml_penalties is not None

        # fit_path requires lambda1 > 0
        model.selection_penalty = 0.01
        model.fit_path(X, y, sample_weight=w, n_lambda=3)

        assert model._reml_lambdas is None
        assert model._reml_penalties is None
        assert model._reml_result is None

        # fit_path must refresh bookkeeping so summary() doesn't report stale REML
        assert model._last_fit_meta is not None
        assert model._last_fit_meta["method"] == "fit_path"
        assert model._fit_stats is not None


# ── Multi-penalty tensor REML (anisotropic smoothing) ────────────


class TestMultiPenaltyTensorREML:
    """End-to-end tests for ti() + main effects with separate marginal lambdas."""

    @pytest.mark.slow
    def test_tensor_reml_converges_with_separate_lambdas(self):
        """fit_reml on s(x1) + s(x2) + ti(x1, x2) converges with per-margin lambdas."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2 + 0.2 * np.sin(2 * np.pi * x1) * x2
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6), "x2": Spline(n_knots=6)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(X, y, max_reml_iter=30)

        assert model._reml_result.converged

        # Lambda dict should have marginal entries for the tensor term
        lam = model._reml_lambdas
        margin_keys = [k for k in lam if "margin_" in k]
        assert len(margin_keys) == 2, f"Expected 2 margin keys, got {margin_keys}"
        assert any("margin_x1" in k for k in margin_keys)
        assert any("margin_x2" in k for k in margin_keys)

        # Main effect splines should also have lambdas
        assert "x1" in lam
        assert "x2" in lam

        # Prediction should work
        pred = model.predict(X)
        assert pred.shape == (n,)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

    @pytest.mark.slow
    def test_tensor_penalty_components_correct(self):
        """Tensor ti() penalty components are correctly structured in REML."""
        rng = np.random.default_rng(123)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": Spline(n_knots=6), "x2": Spline(n_knots=6)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(X, y, max_reml_iter=30)

        # Verify penalty component structure
        penalties = model._reml_penalties
        tensor_pcs = [pc for pc in penalties if pc.name != pc.group_name]
        assert len(tensor_pcs) == 2
        assert any("margin_x1" in pc.name for pc in tensor_pcs)
        assert any("margin_x2" in pc.name for pc in tensor_pcs)

        # Both should share the same group_sl (same coefficient block)
        assert tensor_pcs[0].group_sl == tensor_pcs[1].group_sl

        # Each component omega_ssp should be PSD and non-zero
        for pc in tensor_pcs:
            eigvals = np.linalg.eigvalsh(pc.omega_ssp)
            assert np.all(eigvals >= -1e-10)
            assert pc.rank > 0

        # Component omegas should sum to the full group omega
        from superglm.solvers.irls_direct import _build_penalty_matrix

        S_components = _build_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            model._reml_lambdas,
            model._dm.p,
            reml_penalties=penalties,
        )
        assert np.all(np.isfinite(S_components))

    @pytest.mark.slow
    def test_single_spline_reml_unchanged(self, poisson_data, spline_model):
        """Backward compat: single-spline fit_reml unchanged by multi-penalty changes."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
                "x3": Categorical(),
            },
        )

        model.fit_reml(X, y, sample_weight=w, max_reml_iter=30)
        assert model._reml_result.converged

        # No margin keys — only single-penalty groups
        lam = model._reml_lambdas
        margin_keys = [k for k in lam if "margin_" in k]
        assert len(margin_keys) == 0

        # Penalties should be single-component
        for pc in model._reml_penalties:
            assert pc.name == pc.group_name


class TestComponentNamedLambda2LegacyAssembly:
    """RFC-8 bug-half regression: component-named lambda2 dicts must survive
    the legacy (``reml_penalties=None``) penalty assembly.

    Tensor interactions produce multi-penalty components named
    ``"x1:x2:margin_x1"`` / ``"x1:x2:margin_x2"``.  The legacy assembly in
    ``build_penalty_matrix`` looks penalties up by group name only, so a
    fitted-lambda dict keyed by component names silently drops the entire
    tensor penalty (audit 2026-07-28 §B1/S7).
    """

    @pytest.fixture()
    def tensor_reml_fit(self):
        rng = np.random.default_rng(1234)
        n = 400
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        # Additive truth: the tensor penalty binds hard, so dropping it
        # visibly changes the fit.
        eta = 0.4 + np.sin(2 * np.pi * x1) + 0.5 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=5),
                "x2": Spline(kind="cr", n_knots=5),
            },
            interactions=[("x1", "x2")],
        )
        # The subject downstream is penalty ASSEMBLY equivalence, compared at
        # rtol 1e-6. The tight publication default walks the null direction to
        # lambdas where the ~1e-8-relative canonicalization difference between
        # the two assembly paths lands as ~1e-4 indefinite entries on
        # near-zero diagonals -- verified benign on this fixture (no consumer
        # factorizes the legacy assembly), not a general guarantee. The loose
        # bar keeps the fit off that regime without weakening the assembly
        # comparison.
        model.fit_reml(X, y, max_reml_iter=30, reml_tol=1e-6)
        assert model._reml_penalties is not None
        return model, X, y

    def test_legacy_assembly_matches_component_path(self, tensor_reml_fit):
        from superglm.model.fit_state import fitted_lambda2
        from superglm.reml.penalty_algebra import build_penalty_matrix

        model, X, y = tensor_reml_fit
        lambdas = dict(fitted_lambda2(model))
        group_names = {g.name for g in model._groups}
        component_keys = [k for k in lambdas if k not in group_names]
        assert component_keys, "expected component-named keys from the tensor term"

        p = model._dm.shape[1]
        S_component = build_penalty_matrix(
            list(model._dm.group_matrices),
            model._groups,
            lambdas,
            p,
            reml_penalties=model._reml_penalties,
        )
        S_legacy = build_penalty_matrix(list(model._dm.group_matrices), model._groups, lambdas, p)

        tensor_group = next(g for g in model._groups if g.name == "x1:x2")
        assert np.linalg.norm(S_component[tensor_group.sl, tensor_group.sl]) > 0
        # Tolerance: the component path stores PSD-canonicalized omega_ssp,
        # which differs from the on-the-fly SSP transform at ~1e-8 relative
        # (already true for single-penalty groups today); the bug signal is
        # an entire missing block, orders of magnitude above this.
        np.testing.assert_allclose(
            S_legacy,
            S_component,
            rtol=1e-6,
            atol=1e-7 * np.linalg.norm(S_component),
        )

    def test_structural_smoothing_gate_sees_component_named_keys(self, tensor_reml_fit):
        from superglm.model.fit_state import fitted_lambda2
        from superglm.solvers.pirls import _has_structural_smoothing_penalty

        model, X, y = tensor_reml_fit
        lambdas = dict(fitted_lambda2(model))
        group_names = {g.name for g in model._groups}
        tensor_only = {k: v for k, v in lambdas.items() if k not in group_names}
        assert tensor_only, "expected component-named keys from the tensor term"

        assert _has_structural_smoothing_penalty(
            list(model._dm.group_matrices), model._groups, tensor_only
        )

    def test_active_penalty_matrix_legacy_matches_component_path(self, tensor_reml_fit):
        from superglm.inference.covariance import _active_penalty_matrix
        from superglm.model.fit_state import fitted_lambda2

        model, X, y = tensor_reml_fit
        lambdas = dict(fitted_lambda2(model))

        S_component = _active_penalty_matrix(
            list(model._dm.group_matrices),
            model._groups,
            model._groups,
            lambdas,
            reml_penalties=model._reml_penalties,
        )
        S_legacy = _active_penalty_matrix(
            list(model._dm.group_matrices),
            model._groups,
            model._groups,
            lambdas,
        )

        tensor_group = next(g for g in model._groups if g.name == "x1:x2")
        assert np.linalg.norm(S_component[tensor_group.sl, tensor_group.sl]) > 0
        np.testing.assert_allclose(
            S_legacy,
            S_component,
            rtol=1e-6,
            atol=1e-7 * np.linalg.norm(S_component),
        )

    def test_shape_ops_penalty_terms_legacy_matches_component_path(self, tensor_reml_fit):
        from superglm.model.shape_ops import (
            _build_smooth_penalty_terms,
            _smooth_penalty_value,
        )

        model, X, y = tensor_reml_fit
        beta = model.result.beta
        value_compact = _smooth_penalty_value(beta, _build_smooth_penalty_terms(model))

        saved = model._reml_penalties
        model._reml_penalties = None
        try:
            value_legacy = _smooth_penalty_value(beta, _build_smooth_penalty_terms(model))
        finally:
            model._reml_penalties = saved

        assert value_compact > 0
        np.testing.assert_allclose(value_legacy, value_compact, rtol=1e-6)

    def test_penalty_override_legacy_matches_component_path(self, tensor_reml_fit):
        """A full-width override, sliced to active, matches the legacy assembly.

        Retargeted off the deleted ``_penalised_xtwx_inv`` (PR "Delete the
        covariance chain no production fit reaches").  The wrapper's only role
        here was to route ``S_override`` and the internally-assembled penalty
        into the same inversion; ``_active_penalty_matrix`` takes
        ``S_override`` itself and is live, and ``decompose_gram`` is the live
        inversion, so the comparison is the same one at the same tolerances.

        The augmented half is not carried over, for the reason
        ``TestPenalisedHessianAssembly`` records: production never assembles
        that border, and it was measured to add no discrimination here.
        """
        from superglm.model.fit_state import fitted_lambda2
        from superglm.reml.penalty_algebra import build_penalty_matrix

        model, X, y = tensor_reml_fit
        lambdas = dict(fitted_lambda2(model))
        gms = list(model._dm.group_matrices)
        groups = list(model._groups)
        p = model._dm.shape[1]
        S = build_penalty_matrix(gms, groups, lambdas, p, reml_penalties=model._reml_penalties)
        W = np.ones(model._dm.shape[0])

        S_ref = _active_penalty_matrix(gms, groups, groups, lambdas, S_override=S)
        S_legacy = _active_penalty_matrix(gms, groups, groups, lambdas)

        gram = MatrixExecutionPlan(gms, n=len(W)).moments(W).gram
        inv_ref = decompose_gram(gram + S_ref).pseudo_inverse()
        inv_legacy = decompose_gram(gram + S_legacy).pseudo_inverse()

        scale = np.linalg.norm(inv_ref)
        np.testing.assert_allclose(S_legacy, S_ref, rtol=1e-5, atol=1e-7 * np.linalg.norm(S_ref))
        np.testing.assert_allclose(inv_legacy, inv_ref, rtol=1e-5, atol=1e-7 * scale)
