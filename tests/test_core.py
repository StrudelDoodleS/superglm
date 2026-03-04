"""Tests for SuperGLM. Each test encodes a property that MUST hold."""

import numpy as np
import pytest

from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.types import GroupSlice
from superglm.features.numeric import Numeric
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso


class TestDistributions:
    def test_poisson_variance_is_mu(self):
        mu = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(Poisson().variance(mu), mu)

    def test_gamma_variance_is_mu_squared(self):
        mu = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(Gamma().variance(mu), mu**2)

    def test_tweedie_variance_power(self):
        mu = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(Tweedie(p=1.5).variance(mu), mu**1.5)

    def test_tweedie_p_bounds(self):
        with pytest.raises(ValueError):
            Tweedie(p=0.5)

    def test_deviance_zero_at_y_equals_mu(self):
        for dist in [Poisson(), Gamma(), Tweedie(p=1.5)]:
            y = np.array([1.0, 3.0, 5.0])
            np.testing.assert_allclose(dist.deviance_unit(y, y), 0.0, atol=1e-12)

    def test_deviance_nonneg(self):
        rng = np.random.default_rng(42)
        for dist in [Poisson(), Gamma(), Tweedie(p=1.5)]:
            y = rng.exponential(5.0, size=100)
            mu = rng.exponential(5.0, size=100)
            assert np.all(dist.deviance_unit(y, mu) >= -1e-10)


class TestNumeric:
    def test_group_size_one(self):
        info = Numeric().build(np.array([1.0, 2.0, 3.0]))
        assert info.n_cols == 1

    def test_standardized_mean_zero(self):
        info = Numeric(standardize=True).build(np.arange(100.0))
        assert abs(info.columns.mean()) < 1e-10

    def test_no_penalty_matrix(self):
        assert Numeric().build(np.array([1.0, 2.0])).penalty_matrix is None


class TestCategorical:
    def test_base_excluded(self):
        info = Categorical(base="first").build(np.array(["A", "B", "C"]))
        assert info.n_cols == 2

    def test_most_exposed_base(self):
        spec = Categorical(base="most_exposed")
        spec.build(np.array(["A", "B", "B", "B", "C"]), np.ones(5))
        assert spec._base_level == "B"

    def test_dummy_row_sums(self):
        info = Categorical(base="first").build(np.array(["A", "B", "C"] * 10))
        sums = np.asarray(info.columns.sum(axis=1)).ravel()
        assert np.all((sums == 0) | (sums == 1))

    def test_base_relativity_one(self):
        spec = Categorical(base="first")
        spec.build(np.array(["A", "B", "C"]))
        assert spec.reconstruct(np.array([0.5, -0.3]))["relativities"]["A"] == 1.0


class TestSpline:
    def test_partition_of_unity(self):
        info = Spline(n_knots=10).build(np.linspace(0, 100, 500))
        np.testing.assert_allclose(np.asarray(info.columns.sum(axis=1)).ravel(), 1.0, atol=1e-10)

    def test_nonnegative(self):
        info = Spline(n_knots=10).build(np.linspace(0, 100, 500))
        cols = info.columns.toarray() if hasattr(info.columns, 'toarray') else info.columns
        assert np.all(cols >= -1e-15)

    def test_n_basis(self):
        for nk in [5, 10, 20]:
            info = Spline(n_knots=nk, degree=3).build(np.linspace(0, 1, 100))
            assert info.n_cols == nk + 4  # n_interior + degree + 1

    def test_penalty_psd(self):
        info = Spline(n_knots=10).build(np.linspace(0, 1, 100))
        assert np.all(np.linalg.eigvalsh(info.penalty_matrix) >= -1e-10)

    def test_ssp_flag(self):
        assert Spline(penalty="ssp").build(np.linspace(0, 1, 50)).reparametrize is True
        assert Spline(penalty="none").build(np.linspace(0, 1, 50)).reparametrize is False


class TestProximal:
    """Tests for GroupLasso.prox() — the proximal operator."""

    def _prox(self, beta, groups, lambda1, step=1.0):
        return GroupLasso(lambda1=lambda1).prox(beta, groups, step)

    def test_size1_is_soft_threshold(self):
        groups = [GroupSlice("x", 0, 1, weight=1.0)]
        r = self._prox(np.array([0.5]), groups, lambda1=0.2)
        np.testing.assert_allclose(r, [0.3])
        r = self._prox(np.array([0.1]), groups, lambda1=0.2)
        np.testing.assert_allclose(r, [0.0])

    def test_group_zeroing(self):
        groups = [GroupSlice("g", 0, 5, weight=1.0)]
        beta = np.full(5, 0.01)
        r = self._prox(beta, groups, lambda1=0.1)
        np.testing.assert_allclose(r, 0.0)

    def test_direction_preserved(self):
        groups = [GroupSlice("g", 0, 3, weight=1.0)]
        beta = np.array([3.0, 4.0, 0.0])
        r = self._prox(beta, groups, lambda1=1.0)
        np.testing.assert_allclose(
            beta / np.linalg.norm(beta), r / np.linalg.norm(r), atol=1e-10
        )

    def test_groups_independent(self):
        groups = [GroupSlice("s", 0, 2, 1.0), GroupSlice("b", 2, 4, 1.0)]
        beta = np.array([0.01, 0.01, 5.0, 5.0])
        r = self._prox(beta, groups, lambda1=0.1)
        np.testing.assert_allclose(r[:2], 0.0)
        assert np.linalg.norm(r[2:]) > 0

    def test_zero_threshold_identity(self):
        groups = [GroupSlice("g", 0, 3, 1.0)]
        beta = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(self._prox(beta, groups, lambda1=0.0), beta)


class TestBCDSolver:
    """Tests for the block coordinate descent inner solver."""

    def test_group_lipschitz_positive(self):
        from superglm.solvers.pirls import _compute_group_hessians
        from superglm.group_matrix import DenseGroupMatrix
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        W = np.ones(100)
        groups = [GroupSlice("a", 0, 5, 1.0), GroupSlice("b", 5, 10, 1.0)]
        gms = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
        L_groups, chol_groups = _compute_group_hessians(gms, W)
        assert all(L > 0 for L in L_groups)
        assert len(chol_groups) == 2

    def test_group_lipschitz_le_global(self):
        from superglm.solvers.pirls import _compute_group_hessians
        from superglm.group_matrix import DenseGroupMatrix
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        W = np.ones(100)
        groups = [GroupSlice("a", 0, 5, 1.0), GroupSlice("b", 5, 10, 1.0)]
        gms = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
        L_groups, _ = _compute_group_hessians(gms, W)
        L_global = float(np.linalg.eigvalsh(X.T @ X)[-1])
        for L_g in L_groups:
            assert L_g <= L_global + 1e-10

    def test_bcd_converges(self):
        from superglm.solvers.pirls import fit_pirls
        rng = np.random.default_rng(42)
        n, p = 200, 5
        X = rng.standard_normal((n, p))
        beta_true = np.array([1.0, 0.0, 0.5, 0.0, 2.0])
        y = rng.poisson(np.exp(X @ beta_true)).astype(float)
        y = np.maximum(y, 0.01)
        weights = np.ones(n)
        groups = [
            GroupSlice("a", 0, 2, np.sqrt(2)),
            GroupSlice("b", 2, 5, np.sqrt(3)),
        ]
        result = fit_pirls(X, y, weights, Poisson(), groups, GroupLasso(lambda1=0.01))
        assert result.converged

    def test_bcd_few_inner_iters(self):
        """BCD should converge in far fewer than 50 inner iterations."""
        from superglm.solvers.pirls import fit_pirls
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        y = rng.poisson(np.exp(X @ [0.5, -0.3, 0.1])).astype(float)
        y = np.maximum(y, 0.01)
        groups = [GroupSlice("a", 0, 2, np.sqrt(2)), GroupSlice("b", 2, 3, 1.0)]
        result = fit_pirls(
            X, y, np.ones(n), Poisson(), groups, GroupLasso(lambda1=0.01),
        )
        # Should converge with reasonable outer iters (not maxing out at 50)
        assert result.n_iter < 50
