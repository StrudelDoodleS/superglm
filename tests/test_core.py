"""Tests for SuperGLM. Each test encodes a property that MUST hold."""

import numpy as np
import pytest

from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import (
    CubicRegressionSpline,
    NaturalSpline,
    Spline,
    _SplineBase,
)
from superglm.penalties.group_lasso import GroupLasso
from superglm.types import GroupSlice


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
        cols = info.columns.toarray() if hasattr(info.columns, "toarray") else info.columns
        assert np.all(cols >= -1e-15)

    def test_n_basis(self):
        for nk in [5, 10, 20]:
            info = Spline(n_knots=nk, degree=3).build(np.linspace(0, 1, 100))
            assert info.n_cols == nk + 3  # K - 1 = n_interior + degree (identifiability)

    def test_penalty_psd(self):
        info = Spline(n_knots=10).build(np.linspace(0, 1, 100))
        assert np.all(np.linalg.eigvalsh(info.penalty_matrix) >= -1e-10)

    def test_ssp_flag(self):
        assert Spline(penalty="ssp").build(np.linspace(0, 1, 50)).reparametrize is True
        assert Spline(penalty="none").build(np.linspace(0, 1, 50)).reparametrize is False


class TestNaturalSpline:
    """Tests for NaturalSpline boundary constraints."""

    def test_natural_n_basis(self):
        """K-3 columns (natural constraints remove 2, identifiability removes 1)."""
        info = NaturalSpline(n_knots=10).build(np.linspace(0, 1, 200))
        assert info.n_cols == 10 + 4 - 3  # K - 3

    def test_natural_penalty_psd(self):
        """Projected penalty is positive semi-definite."""
        info = NaturalSpline(n_knots=10).build(np.linspace(0, 1, 200))
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.all(eigvals >= -1e-10)

    def test_natural_zero_second_deriv_at_boundaries(self):
        """f''(lo) ~ 0 and f''(hi) ~ 0 for any coefficient in the constrained space."""
        from scipy.interpolate import BSpline as BSpl

        sp = NaturalSpline(n_knots=10)
        sp.build(np.linspace(0, 100, 500))
        Z = sp._Z
        rng = np.random.default_rng(42)
        for _ in range(10):
            alpha = rng.standard_normal(Z.shape[1])
            beta_orig = Z @ alpha  # (K,) in original B-spline space
            spl = BSpl(sp._knots, beta_orig, sp.degree)
            np.testing.assert_allclose(spl(sp._lo, nu=2), 0.0, atol=1e-10)
            np.testing.assert_allclose(spl(sp._hi, nu=2), 0.0, atol=1e-10)

    def test_natural_penalty_null_space_0d(self):
        """After identifiability, omega has full rank (no null space)."""
        info = NaturalSpline(n_knots=10).build(np.linspace(0, 1, 200))
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        n_null = np.sum(eigvals < 1e-10)
        assert n_null == 0  # constant removed by identifiability

    def test_natural_projection_shape(self):
        """GroupInfo.projection is (K, K-3) after natural + identifiability."""
        sp = NaturalSpline(n_knots=10)
        info = sp.build(np.linspace(0, 1, 200))
        K = sp._n_basis
        assert info.projection is not None
        assert info.projection.shape == (K, K - 3)


class TestSplineBaseHierarchy:
    """Tests for _SplineBase class hierarchy and knot features."""

    def test_spline_base_not_directly_usable(self):
        """_SplineBase._build_penalty raises NotImplementedError."""
        base = _SplineBase(n_knots=5)
        with pytest.raises(NotImplementedError):
            base.build(np.linspace(0, 1, 100))

    def test_isinstance_base(self):
        """Both Spline and NaturalSpline are instances of _SplineBase."""
        assert isinstance(Spline(n_knots=5), _SplineBase)
        assert isinstance(NaturalSpline(n_knots=5), _SplineBase)

    def test_explicit_knots(self):
        """Spline(knots=[...]) places those exact interior knots."""
        knots = np.array([20.0, 40.0, 60.0, 80.0])
        sp = Spline(knots=knots)
        sp.build(np.linspace(0, 100, 500))
        interior = sp._knots[sp.degree + 1 : -(sp.degree + 1)]
        np.testing.assert_array_equal(interior, knots)
        assert sp.n_knots == 4

    def test_explicit_knots_frozen_on_refit(self):
        """Same knots used regardless of data distribution."""
        knots = np.array([20.0, 40.0, 60.0, 80.0])
        sp = Spline(knots=knots)
        # Fit on uniform data
        sp.build(np.linspace(0, 100, 500))
        interior1 = sp._knots[sp.degree + 1 : -(sp.degree + 1)].copy()
        # Refit on skewed data
        sp.build(np.concatenate([np.linspace(0, 10, 400), np.linspace(90, 100, 100)]))
        interior2 = sp._knots[sp.degree + 1 : -(sp.degree + 1)]
        np.testing.assert_array_equal(interior1, interior2)

    def test_explicit_knots_natural_spline(self):
        """NaturalSpline also accepts explicit knots."""
        knots = np.array([25.0, 50.0, 75.0])
        sp = NaturalSpline(knots=knots)
        info = sp.build(np.linspace(0, 100, 300))
        interior = sp._knots[sp.degree + 1 : -(sp.degree + 1)]
        np.testing.assert_array_equal(interior, knots)
        # NaturalSpline with 3 interior knots: K = 3+3+1 = 7, K-3 = 4
        assert info.n_cols == sp._n_basis - 3

    def test_uniform_default(self):
        """Spline() uses uniform strategy by default."""
        sp = Spline(n_knots=5)
        assert sp.knot_strategy == "uniform"
        sp.build(np.linspace(0, 100, 500))
        interior = sp._knots[sp.degree + 1 : -(sp.degree + 1)]
        expected = np.linspace(0, 100, 7)[1:-1]
        np.testing.assert_allclose(interior, expected, atol=1e-10)

    def test_invalid_knots_raises(self):
        """Empty knots array raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Spline(knots=np.array([]))

    def test_invalid_extrapolation_raises(self):
        """Unknown extrapolation policy should fail at construction time."""
        with pytest.raises(ValueError, match="extrapolation must be one of"):
            Spline(extrapolation="banana")


class TestSplineExtrapolation:
    @pytest.mark.parametrize("spec_cls", [Spline, NaturalSpline, CubicRegressionSpline])
    def test_clip_freezes_at_boundary(self, spec_cls):
        """Default clipping should reuse the boundary basis outside fit range."""
        x_train = np.linspace(0.0, 1.0, 200)
        spec = spec_cls(n_knots=8, extrapolation="clip")
        spec.build(x_train)

        below = spec.transform(np.array([-0.5]))
        at_lo = spec.transform(np.array([0.0]))
        above = spec.transform(np.array([1.5]))
        at_hi = spec.transform(np.array([1.0]))

        np.testing.assert_allclose(below, at_lo, atol=1e-12)
        np.testing.assert_allclose(above, at_hi, atol=1e-12)

    @pytest.mark.parametrize("spec_cls", [Spline, NaturalSpline, CubicRegressionSpline])
    def test_error_mode_rejects_out_of_range(self, spec_cls):
        """extrapolation='error' should fail on out-of-range prediction."""
        x_train = np.linspace(0.0, 1.0, 200)
        spec = spec_cls(n_knots=8, extrapolation="error")
        spec.build(x_train)

        with pytest.raises(ValueError, match="outside training range"):
            spec.transform(np.array([-0.1, 0.4]))

    @pytest.mark.parametrize("spec_cls", [NaturalSpline, CubicRegressionSpline])
    def test_extend_exposes_tail_behavior(self, spec_cls):
        """extend should not collapse to the boundary basis for constrained splines."""
        x_train = np.linspace(0.0, 1.0, 200)
        spec = spec_cls(n_knots=8, extrapolation="extend")
        spec.build(x_train)

        below = spec.transform(np.array([-0.5]))
        at_lo = spec.transform(np.array([0.0]))
        above = spec.transform(np.array([1.5]))
        at_hi = spec.transform(np.array([1.0]))

        assert not np.allclose(below, at_lo)
        assert not np.allclose(above, at_hi)


class TestCubicRegressionSpline:
    """Tests for CubicRegressionSpline (integrated f'' squared penalty + natural constraints)."""

    def test_cr_n_basis(self):
        """K-3 columns after natural constraints plus intercept absorption."""
        sp = CubicRegressionSpline(n_knots=10)
        info = sp.build(np.linspace(0, 1, 200))
        assert info.n_cols == 10 + 4 - 3  # K - 2, then drop the constant direction

    def test_cr_penalty_psd(self):
        """Integrated penalty is positive semi-definite."""
        info = CubicRegressionSpline(n_knots=10).build(np.linspace(0, 1, 200))
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.all(eigvals >= -1e-10)

    def test_cr_natural_boundary(self):
        """f''(lo) = f''(hi) = 0 for random coefficients in the constrained space."""
        from scipy.interpolate import BSpline as BSpl

        sp = CubicRegressionSpline(n_knots=10)
        sp.build(np.linspace(0, 100, 500))
        Z = sp._Z
        rng = np.random.default_rng(42)
        for _ in range(10):
            alpha = rng.standard_normal(Z.shape[1])
            beta_orig = Z @ alpha
            spl = BSpl(sp._knots, beta_orig, sp.degree)
            np.testing.assert_allclose(spl(sp._lo, nu=2), 0.0, atol=1e-10)
            np.testing.assert_allclose(spl(sp._hi, nu=2), 0.0, atol=1e-10)

    def test_cr_penalty_null_space_1d(self):
        """Identified CR penalty leaves only the linear null direction."""
        info = CubicRegressionSpline(n_knots=10).build(np.linspace(0, 1, 200))
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        n_null = np.sum(eigvals < 1e-10)
        assert n_null == 1

    def test_cr_penalty_differs_from_pspline(self):
        """Integrated penalty stays distinct from the natural-spline penalty."""
        x = np.linspace(0, 1, 200)
        cr = CubicRegressionSpline(n_knots=10)
        ns = NaturalSpline(n_knots=10)
        cr_info = cr.build(x)
        ns_info = ns.build(x)
        assert cr_info.penalty_matrix.shape[0] == ns_info.penalty_matrix.shape[0]
        assert not np.isclose(
            np.trace(cr_info.penalty_matrix),
            np.trace(ns_info.penalty_matrix),
            atol=1e-6,
        )

    def test_cr_projection_avoids_intercept_duplication(self):
        """The identified CR basis should be linearly independent of the intercept."""
        x = np.linspace(0, 1, 200)
        sp = CubicRegressionSpline(n_knots=10)
        info = sp.build(x)
        basis = info.columns.toarray() @ info.projection
        assert np.linalg.matrix_rank(basis) == info.n_cols
        assert np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), basis])) == info.n_cols + 1

    def test_cr_isinstance_base(self):
        """CubicRegressionSpline is an instance of _SplineBase."""
        assert isinstance(CubicRegressionSpline(n_knots=5), _SplineBase)

    def test_cr_degree_always_3(self):
        """.degree == 3 regardless."""
        assert CubicRegressionSpline(n_knots=5).degree == 3
        assert CubicRegressionSpline(n_knots=20).degree == 3

    def test_cr_explicit_knots(self):
        """knots param works."""
        knots = np.array([20.0, 40.0, 60.0, 80.0])
        sp = CubicRegressionSpline(knots=knots)
        sp.build(np.linspace(0, 100, 500))
        interior = sp._knots[sp.degree + 1 : -(sp.degree + 1)]
        np.testing.assert_array_equal(interior, knots)
        assert sp.n_knots == 4


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
        np.testing.assert_allclose(beta / np.linalg.norm(beta), r / np.linalg.norm(r), atol=1e-10)

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
        from superglm.group_matrix import DenseGroupMatrix
        from superglm.solvers.pirls import _compute_group_hessians

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        W = np.ones(100)
        groups = [GroupSlice("a", 0, 5, 1.0), GroupSlice("b", 5, 10, 1.0)]
        gms = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
        L_groups, chol_groups = _compute_group_hessians(gms, W)
        assert all(L > 0 for L in L_groups)
        assert len(chol_groups) == 2

    def test_group_lipschitz_le_global(self):
        from superglm.group_matrix import DenseGroupMatrix
        from superglm.solvers.pirls import _compute_group_hessians

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
        from superglm.links import LogLink

        result = fit_pirls(X, y, weights, Poisson(), LogLink(), groups, GroupLasso(lambda1=0.01))
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
        from superglm.links import LogLink

        result = fit_pirls(
            X,
            y,
            np.ones(n),
            Poisson(),
            LogLink(),
            groups,
            GroupLasso(lambda1=0.01),
        )
        # Should converge with reasonable outer iters (not maxing out at 50)
        assert result.n_iter < 50


class TestAndersonAcceleration:
    """Tests for Anderson acceleration in the PIRLS solver."""

    @pytest.fixture
    def poisson_problem(self):
        from superglm.links import LogLink

        rng = np.random.default_rng(42)
        n, p = 300, 8
        X = rng.standard_normal((n, p))
        beta_true = np.array([1.0, 0.5, 0.0, 0.0, -0.3, 0.0, 0.0, 0.2])
        y = rng.poisson(np.exp(X @ beta_true)).astype(float)
        y = np.maximum(y, 0.01)
        weights = np.ones(n)
        groups = [
            GroupSlice("a", 0, 3, np.sqrt(3)),
            GroupSlice("b", 3, 5, np.sqrt(2)),
            GroupSlice("c", 5, 8, np.sqrt(3)),
        ]
        return X, y, weights, groups, Poisson(), LogLink()

    def test_anderson_converges(self, poisson_problem):
        """Anderson acceleration should converge to a solution."""
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = poisson_problem
        result = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.05),
            anderson_memory=5,
        )
        assert result.converged

    def test_anderson_matches_baseline(self, poisson_problem):
        """Anderson and non-Anderson should converge to the same solution."""
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = poisson_problem
        pen_args = dict(lambda1=0.05)

        baseline = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(**pen_args),
        )
        anderson = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(**pen_args),
            anderson_memory=5,
        )
        # Same solution (deviance should match closely)
        assert anderson.deviance == pytest.approx(baseline.deviance, rel=1e-4)
        np.testing.assert_allclose(anderson.beta, baseline.beta, atol=1e-3)

    def test_anderson_memory_sizes(self, poisson_problem):
        """Different memory sizes should all converge."""
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = poisson_problem
        for m in [1, 3, 5, 10]:
            result = fit_pirls(
                X,
                y,
                w,
                family,
                link,
                groups,
                GroupLasso(lambda1=0.05),
                anderson_memory=m,
            )
            assert result.converged, f"anderson_memory={m} did not converge"


class TestActiveSet:
    """Tests for active-set BCD optimization."""

    @pytest.fixture
    def sparse_problem(self):
        """Problem where many groups should be zeroed (high lambda)."""
        from superglm.links import LogLink

        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, 10))
        # Only first group matters
        y = rng.poisson(np.exp(0.5 * X[:, 0])).astype(float)
        y = np.maximum(y, 0.01)
        weights = np.ones(n)
        groups = [
            GroupSlice("a", 0, 2, np.sqrt(2)),
            GroupSlice("b", 2, 4, np.sqrt(2)),
            GroupSlice("c", 4, 6, np.sqrt(2)),
            GroupSlice("d", 6, 8, np.sqrt(2)),
            GroupSlice("e", 8, 10, np.sqrt(2)),
        ]
        return X, y, weights, groups, Poisson(), LogLink()

    def test_active_set_converges(self, sparse_problem):
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = sparse_problem
        result = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.1),
            active_set=True,
        )
        assert result.converged

    def test_active_set_matches_baseline(self, sparse_problem):
        """Active-set should produce the same solution as baseline."""
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = sparse_problem
        baseline = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.1),
        )
        active = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.1),
            active_set=True,
        )
        assert active.deviance == pytest.approx(baseline.deviance, rel=1e-4)
        np.testing.assert_allclose(active.beta, baseline.beta, atol=1e-3)

    def test_active_set_with_anderson(self, sparse_problem):
        """Active-set + Anderson should converge and match baseline."""
        from superglm.solvers.pirls import fit_pirls

        X, y, w, groups, family, link = sparse_problem
        baseline = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.1),
        )
        combined = fit_pirls(
            X,
            y,
            w,
            family,
            link,
            groups,
            GroupLasso(lambda1=0.1),
            anderson_memory=5,
            active_set=True,
        )
        assert combined.converged
        assert combined.deviance == pytest.approx(baseline.deviance, rel=1e-4)


# ── Knot governance ──────────────────────────────────────────────


class TestFittedKnots:
    """Public accessors for fitted knot locations."""

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr", "cr_cardinal"])
    def test_fitted_knots_returns_interior(self, kind):
        sp = Spline(kind=kind, n_knots=6)
        assert sp.fitted_knots is None  # before fit

        x = np.linspace(0, 10, 300)
        sp.build(x)
        knots = sp.fitted_knots
        assert knots is not None
        assert len(knots) == 6
        assert knots[0] > 0.0  # strictly inside boundaries
        assert knots[-1] < 10.0

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr", "cr_cardinal"])
    def test_fitted_knots_is_copy(self, kind):
        """Mutating the returned array must not alter the spec."""
        sp = Spline(kind=kind, n_knots=4)
        sp.build(np.linspace(0, 10, 200))
        k1 = sp.fitted_knots
        k1[:] = 999.0
        k2 = sp.fitted_knots
        assert not np.any(k2 == 999.0)

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr", "cr_cardinal"])
    def test_fitted_boundary(self, kind):
        sp = Spline(kind=kind, n_knots=4)
        assert sp.fitted_boundary is None  # before fit
        x = np.linspace(2.0, 8.0, 200)
        sp.build(x)
        lo, hi = sp.fitted_boundary
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(8.0)

    def test_explicit_knots_round_trip(self):
        """fitted_knots from one fit can freeze placement on a second fit."""
        sp1 = Spline(kind="cr", n_knots=8)
        x_uniform = np.linspace(0, 100, 500)
        sp1.build(x_uniform)
        frozen = sp1.fitted_knots

        sp2 = Spline(kind="cr", knots=frozen)
        x_skewed = np.concatenate([np.linspace(0, 10, 400), np.linspace(90, 100, 100)])
        sp2.build(x_skewed)
        np.testing.assert_array_equal(sp2.fitted_knots, frozen)


class TestQuantileKnotGovernance:
    """Quantile knot placement stores and freezes exact positions."""

    def test_quantile_of_unique(self):
        """Quantile strategy uses unique(x), producing strictly increasing knots."""
        # Data with heavy ties: 80% at value 50, rest spread
        rng = np.random.default_rng(0)
        x = np.concatenate([np.full(800, 50.0), rng.uniform(0, 100, 200)])
        sp = Spline(kind="bs", n_knots=8, knot_strategy="quantile")
        sp.build(x)
        knots = sp.fitted_knots
        assert len(knots) == 8
        # Strictly increasing (no duplicates from ties)
        assert np.all(np.diff(knots) > 0)

    def test_quantile_knots_frozen_on_refit(self):
        """Quantile knots from first fit can freeze placement on refit."""
        rng = np.random.default_rng(42)
        x1 = rng.exponential(10, 500)
        sp = Spline(kind="cr_cardinal", n_knots=6, knot_strategy="quantile")
        sp.build(x1)
        knots_fit1 = sp.fitted_knots.copy()

        # Refit on totally different distribution
        sp2 = Spline(kind="cr_cardinal", knots=knots_fit1)
        x2 = rng.uniform(0, 100, 500)
        sp2.build(x2)
        np.testing.assert_array_equal(sp2.fitted_knots, knots_fit1)


class TestKnotSummary:
    """SuperGLM.knot_summary() exposes fitted knot metadata."""

    def test_knot_summary_content(self):
        import pandas as pd

        from superglm import SuperGLM

        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({"x": rng.uniform(0, 10, n), "cat": rng.choice(["a", "b"], n)})
        y = rng.poisson(2.0, n).astype(float)

        model = SuperGLM(
            features={"x": Spline(kind="cr", n_knots=6), "cat": Categorical()},
            family="poisson",
        )
        model.fit(X=df, y=y)
        ks = model.knot_summary()

        # Only spline features appear
        assert "x" in ks
        assert "cat" not in ks

        info = ks["x"]
        assert info["kind"] == "CubicRegressionSpline"
        assert info["knot_strategy"] == "uniform"
        assert len(info["interior_knots"]) == 6
        assert info["boundary"][0] == pytest.approx(df["x"].min())
        assert info["boundary"][1] == pytest.approx(df["x"].max())

    def test_knot_summary_explicit_strategy_label(self):
        import pandas as pd

        from superglm import SuperGLM

        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        y = rng.poisson(1.0, n).astype(float)

        model = SuperGLM(
            features={"x": Spline(knots=np.array([2.0, 5.0, 8.0]))},
            family="poisson",
        )
        model.fit(X=df, y=y)
        assert model.knot_summary()["x"]["knot_strategy"] == "explicit"


class TestKnotPickleRoundTrip:
    """Pickle round-trip preserves fitted knots exactly."""

    @pytest.mark.parametrize("kind", ["bs", "cr", "cr_cardinal"])
    def test_pickle_preserves_knots(self, kind):
        import pickle

        import pandas as pd

        from superglm import SuperGLM

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        y = rng.poisson(2.0, n).astype(float)

        model = SuperGLM(
            features={"x": Spline(kind=kind, n_knots=6)},
            family="poisson",
        )
        model.fit(X=df, y=y)
        pred_before = model.predict(df)
        knots_before = model.knot_summary()["x"]["interior_knots"].copy()

        # Round-trip through pickle
        blob = pickle.dumps(model)
        model2 = pickle.loads(blob)

        knots_after = model2.knot_summary()["x"]["interior_knots"]
        pred_after = model2.predict(df)
        np.testing.assert_array_equal(knots_before, knots_after)
        np.testing.assert_array_equal(pred_before, pred_after)
