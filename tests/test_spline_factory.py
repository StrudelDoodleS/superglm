"""Tests for the Spline(kind=..., k=...) factory API."""

import warnings

import numpy as np
import pytest

from superglm import Constraint
from superglm.features.spline import (
    BSplineSmooth,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
    Spline,
    _SplineBase,
    n_knots_from_k,
)

# ── Factory dispatch ─────────────────────────────────────────────


class TestSplineFactoryDispatch:
    """Spline() should dispatch to the correct concrete class."""

    def test_spline_function_stays_on_public_module(self):
        assert Spline.__module__ == "superglm.features.spline"

    def test_n_knots_function_stays_on_public_module(self):
        assert n_knots_from_k.__module__ == "superglm.features.spline"

    def test_default_dispatch(self):
        s = Spline(n_knots=8)
        assert isinstance(s, CubicRegressionSpline)

    def test_bs_explicit(self):
        s = Spline(kind="bs", n_knots=8, penalty="ssp")
        assert isinstance(s, BSplineSmooth)
        assert s.n_knots == 8
        assert s.penalty == "ssp"

    def test_ns(self):
        s = Spline(kind="ns", n_knots=8)
        assert isinstance(s, NaturalSpline)
        assert s.n_knots == 8

    def test_cr(self):
        s = Spline(kind="cr", n_knots=8)
        assert isinstance(s, CubicRegressionSpline)
        assert s.n_knots == 8
        assert s.degree == 3  # always cubic

    def test_all_kinds_are_spline_base(self):
        for kind in ["bs", "ns", "cr"]:
            s = Spline(kind=kind, n_knots=5)
            assert isinstance(s, _SplineBase)

    def test_bs_select(self):
        s = Spline(kind="bs", n_knots=8, select=True)
        assert isinstance(s, BSplineSmooth)
        assert s.select is True

    def test_params_forwarded(self):
        s = Spline(
            kind="bs",
            n_knots=12,
            degree=2,
            knot_strategy="quantile",
            penalty="none",
            extrapolation="extend",
        )
        assert s.n_knots == 12
        assert s.degree == 2
        assert s.knot_strategy == "quantile"
        assert s.penalty == "none"
        assert s.extrapolation == "extend"

    def test_cr_is_cubic(self):
        """CR is always cubic; a degree other than 3 is refused (TestDefaultKindIsCr)."""
        s = Spline(kind="cr", n_knots=8)
        assert s.degree == 3

    def test_ns_accepts_degree(self):
        s = Spline(kind="ns", n_knots=8, degree=2)
        assert s.degree == 2

    def test_discrete_and_nbins_forwarded(self):
        s = Spline(kind="bs", n_knots=8, discrete=True, n_bins=128)
        assert s.discrete is True
        assert s.n_bins == 128


# ── k mapping ────────────────────────────────────────────────────


class TestKMapping:
    """Test k → n_knots conversion and resulting basis dimensions."""

    def test_n_knots_from_k_bs(self):
        # bs: n_knots = k - degree - 1
        assert n_knots_from_k("bs", 14, degree=3) == 10  # 14 - 3 - 1 = 10
        assert n_knots_from_k("bs", 20, degree=3) == 16  # 20 - 3 - 1 = 16
        assert n_knots_from_k("bs", 7, degree=2) == 4  # 7 - 2 - 1 = 4

    def test_n_knots_from_k_ns(self):
        # ns: n_knots = k - degree + 1
        assert n_knots_from_k("ns", 10, degree=3) == 8  # 10 - 3 + 1 = 8
        assert n_knots_from_k("ns", 20, degree=3) == 18  # 20 - 3 + 1 = 18

    def test_n_knots_from_k_cr(self):
        # cr: same as ns (n_knots = k - degree + 1), mgcv-aligned
        assert n_knots_from_k("cr", 10, degree=3) == 8  # 10 - 3 + 1 = 8
        assert n_knots_from_k("cr", 20, degree=3) == 18  # 20 - 3 + 1 = 18

    def test_factory_with_k_bs(self):
        """Spline(kind='bs', k=14) should produce n_knots=10 for degree=3."""
        s = Spline(kind="bs", k=14)
        assert isinstance(s, BSplineSmooth)
        assert s.n_knots == 10

    def test_factory_with_k_ns(self):
        s = Spline(kind="ns", k=10)
        assert isinstance(s, NaturalSpline)
        assert s.n_knots == 8

    def test_factory_with_k_cr(self):
        s = Spline(kind="cr", k=10)
        assert isinstance(s, CubicRegressionSpline)
        assert s.n_knots == 8

    def test_k_produces_correct_ncols_bs(self):
        """For bs, built column count is k-1 (identifiability removes 1)."""
        k = 14
        s = Spline(kind="bs", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k - 1

    def test_k_produces_correct_ncols_ns(self):
        """For ns, built column count is k-1 (identifiability removes 1)."""
        k = 10
        s = Spline(kind="ns", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k - 1

    def test_k_produces_correct_ncols_cr(self):
        """For cr, built column count is k-1 (identifiability removes 1)."""
        k = 10
        s = Spline(kind="cr", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k - 1


# ── Validation ───────────────────────────────────────────────────


class TestSplineValidation:
    """Test error handling for bad inputs."""

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown spline kind"):
            Spline(kind="tp")

    def test_k_and_n_knots_both_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both k and n_knots"):
            Spline(kind="bs", k=14, n_knots=10)

    def test_k_too_small_bs(self):
        with pytest.raises(ValueError, match="too small"):
            Spline(kind="bs", k=3)  # min is degree+2=5 for degree=3

    def test_k_too_small_ns(self):
        with pytest.raises(ValueError, match="too small"):
            Spline(kind="ns", k=2)  # min is degree=3

    def test_k_too_small_cr(self):
        with pytest.raises(ValueError, match="too small"):
            Spline(kind="cr", k=2)

    def test_select_ns_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="select=True is not supported"):
            Spline(kind="ns", n_knots=8, select=True)

    def test_select_cr_succeeds(self):
        s = Spline(kind="cr", n_knots=8, select=True)
        assert isinstance(s, CubicRegressionSpline)
        assert s.select is True

    def test_select_cr_cardinal_succeeds(self):
        s = Spline(kind="cr_cardinal", n_knots=8, select=True)
        assert s.select is True

    def test_n_knots_from_k_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown spline kind"):
            n_knots_from_k("xyz", 10)

    def test_n_knots_from_k_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            n_knots_from_k("bs", 4, degree=3)  # min is 5


# ── Direct Class Usage ───────────────────────────────────────────


class TestDirectClasses:
    """Concrete spline classes still build directly."""

    def test_bspline_smooth_direct(self):
        s = BSplineSmooth(n_knots=8, penalty="ssp")
        assert isinstance(s, _SplineBase)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_natural_spline_direct(self):
        s = NaturalSpline(n_knots=8)
        assert isinstance(s, _SplineBase)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_crs_direct(self):
        s = CubicRegressionSpline(n_knots=8)
        assert isinstance(s, _SplineBase)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_old_spline_syntax(self):
        """Spline(n_knots=8, penalty='ssp') still works (defaults to kind='cr')."""
        s = Spline(n_knots=8, penalty="ssp")
        assert isinstance(s, CubicRegressionSpline)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_default_n_knots(self):
        """Spline() with no size arg uses n_knots=10 default."""
        s = Spline()
        assert isinstance(s, CubicRegressionSpline)
        assert s.n_knots == 10


# ── PSpline factory dispatch ────────────────────────────────────


class TestPSplineFactory:
    """Tests for the new kind='ps' dispatch path."""

    def test_ps_dispatch(self):
        """Spline(kind='ps') should dispatch to PSpline."""
        s = Spline(kind="ps", n_knots=8)
        assert isinstance(s, PSpline)

    def test_ps_isinstance_splinebase(self):
        s = Spline(kind="ps", n_knots=8)
        assert isinstance(s, _SplineBase)

    def test_ps_params_forwarded(self):
        s = Spline(
            kind="ps",
            n_knots=12,
            degree=2,
            knot_strategy="quantile",
            penalty="none",
            extrapolation="extend",
        )
        assert s.n_knots == 12
        assert s.degree == 2
        assert s.knot_strategy == "quantile"
        assert s.penalty == "none"
        assert s.extrapolation == "extend"

    def test_ps_select(self):
        s = Spline(kind="ps", n_knots=8, select=True)
        assert isinstance(s, PSpline)
        assert s.select is True

    def test_ps_constraint_token_normalizes_to_internal_monotone_fields(self):
        s = Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing)

        assert isinstance(s, PSpline)
        assert s.monotone == "increasing"
        assert s.monotone_mode == "fit"

    def test_ps_builds_different_penalty_from_bs(self):
        """kind='ps' and kind='bs' share basis geometry but not penalty semantics."""
        s_bs = Spline(kind="bs", n_knots=8)
        s_ps = Spline(kind="ps", n_knots=8)
        assert type(s_bs) is BSplineSmooth
        assert type(s_ps) is PSpline
        x = np.linspace(0, 1, 100)
        info_bs = s_bs.build(x)
        info_ps = s_ps.build(x)
        import scipy.sparse as sp

        cols_bs = info_bs.columns
        cols_ps = info_ps.columns
        if sp.issparse(cols_bs):
            cols_bs = cols_bs.toarray()
        if sp.issparse(cols_ps):
            cols_ps = cols_ps.toarray()
        np.testing.assert_array_equal(cols_bs, cols_ps)
        assert not np.allclose(info_bs.penalty_matrix, info_ps.penalty_matrix)

    def test_ps_k_mapping(self):
        """k parameter works with kind='ps'."""
        s = Spline(kind="ps", k=14)
        assert s.n_knots == 10  # 14 - 3 - 1

    def test_n_knots_from_k_ps(self):
        """n_knots_from_k accepts 'ps' kind."""
        assert n_knots_from_k("ps", 14, degree=3) == 10
        assert n_knots_from_k("ps", 20, degree=3) == 16


# ── Default kind is "cr" ────────────────────────────────────────


class TestDefaultKindIsCr:
    """``Spline()`` and ``s()`` default to kind='cr' from 0.40 (it was 'ps').

    Every test here fails on the 0.39 factory: its default built a PSpline,
    and it accepted any ``degree`` for a cubic regression spline and ignored
    it, so a ``degree=2`` request silently fitted a cubic.
    """

    def test_default_is_cubic_regression_spline(self):
        s = Spline(n_knots=8)
        assert isinstance(s, CubicRegressionSpline)
        assert not isinstance(s, PSpline)

    def test_default_builds_exactly_the_explicit_cr(self):
        x = np.random.default_rng(0).uniform(0.0, 10.0, 300)
        default = Spline(n_knots=8).build(x)
        explicit = Spline(kind="cr", n_knots=8).build(x)

        def dense(columns):
            return columns.toarray() if hasattr(columns, "toarray") else np.asarray(columns)

        assert default.n_cols == explicit.n_cols == 9  # 8 interior + 2 boundary knots, centred
        np.testing.assert_array_equal(dense(default.columns), dense(explicit.columns))
        np.testing.assert_array_equal(default.penalty_matrix, explicit.penalty_matrix)

    def test_s_default_is_cubic_regression_spline(self):
        from superglm.terms import s

        assert isinstance(s("age", n_knots=8).spec, CubicRegressionSpline)

    def test_default_no_warning(self):
        """The default kind should not emit a FutureWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Spline(n_knots=8)  # should not raise

    @pytest.mark.parametrize("kind", ["cr", "cr_cardinal"])
    @pytest.mark.parametrize("degree", [1, 2, 4])
    def test_cubic_regression_kinds_refuse_another_degree(self, kind, degree):
        with pytest.raises(ValueError, match="always cubic") as excinfo:
            Spline(kind=kind, n_knots=8, degree=degree)
        message = str(excinfo.value)
        assert f"degree={degree} cannot apply" in message
        assert "kind='ps' or kind='bs'" in message

    def test_default_kind_refuses_a_degree_rather_than_fitting_a_cubic(self):
        """Under the new default, ``Spline(degree=2)`` (a quadratic P-spline in
        0.39) must not silently become a cubic regression spline."""
        from superglm.terms import s

        with pytest.raises(ValueError, match="kind='cr'.*always cubic"):
            Spline(n_knots=8, degree=2)
        with pytest.raises(ValueError, match="kind='cr'.*always cubic"):
            s("age", n_knots=8, degree=2)
        assert Spline(kind="ps", n_knots=8, degree=2).degree == 2

    @pytest.mark.parametrize("m", [4, (2, 4)], ids=["int", "tuple"])
    def test_default_kind_refuses_a_penalty_order_above_its_maximum(self, m):
        """``Spline(m=4)`` was a valid P-spline in 0.39. The default now names itself,
        its maximum and the remedy, not the class the caller never wrote."""
        with pytest.raises(ValueError, match="kind='cr'.*takes penalty orders up to 3") as excinfo:
            Spline(m=m)
        message = str(excinfo.value)
        assert "so m=4 cannot apply" in message
        assert "Pass kind='ps' for a penalty of order 4" in message
        assert "CubicRegressionSpline" not in message

    def test_s_default_refuses_a_penalty_order_above_its_maximum(self):
        from superglm.terms import s

        with pytest.raises(ValueError, match="kind='cr'.*takes penalty orders up to 3"):
            s("age", m=4)

    @pytest.mark.parametrize("kind, cap", [("cr", 3), ("cr_cardinal", 2)])
    def test_cubic_regression_kinds_refuse_a_penalty_order_above_their_maximum(self, kind, cap):
        with pytest.raises(ValueError, match="takes penalty orders up to") as excinfo:
            Spline(kind=kind, n_knots=8, m=cap + 1)
        message = str(excinfo.value)
        assert f"up to {cap}, so m={cap + 1} cannot apply" in message
        assert f"Pass kind='ps' for a penalty of order {cap + 1}" in message

    def test_penalty_order_maximum_is_read_from_the_class(self, monkeypatch):
        """The refusal takes its cap from ``CubicRegressionSpline._max_penalty_order``,
        so raising that cap admits m=4; a hard-coded 3 in the factory would not."""
        monkeypatch.setattr(CubicRegressionSpline, "_max_penalty_order", 5)
        assert isinstance(Spline(m=4), CubicRegressionSpline)

    def test_penalty_order_at_the_maximum_builds_and_the_remedy_is_a_p_spline(self):
        assert isinstance(Spline(m=3), CubicRegressionSpline)
        assert isinstance(Spline(kind="ps", m=4), PSpline)


# ── kind="bs" is real BSplineSmooth ─────────────────────────────


class TestBsFactory:
    """kind='bs' dispatches directly to the integrated-derivative B-spline smooth."""

    def test_bs_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Spline(kind="bs", n_knots=8)

    def test_ps_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Spline(kind="ps", n_knots=8)  # should not raise

    def test_bs_creates_bspline_smooth(self):
        s = Spline(kind="bs", n_knots=8)
        assert isinstance(s, BSplineSmooth)


# ── A fixed penalty is measured in knot intervals ───────────────


class TestPenaltyUnits:
    """``cr``, ``cr_cardinal`` and ``bs`` penalties are integrated over ``x / hbar``.

    ``hbar`` is the mean knot interval. Over ``x`` the curvature integral
    scales as ``units**-3``, so on d4e8b4c9 the kindless default fitted at the
    default ``spline_penalty`` smoothed the same column a thousandfold less
    when it was measured in units a tenth the size.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"kind": "cr_cardinal"},
            {"kind": "bs"},
            {"kind": "bs", "m": (1, 2)},
            {"kind": "ps"},
        ],
        ids=["default", "cr_cardinal", "bs", "bs_m12", "ps"],
    )
    @pytest.mark.parametrize("units", [2.0**10, 2.0**-10])
    def test_a_fixed_penalty_fit_ignores_the_columns_units(self, kwargs, units):
        """A power-of-two change of units is exact (Higham, *Accuracy and
        Stability of Numerical Algorithms*, 2nd ed., section 2.1, absent
        underflow and overflow): knots, intervals and the mapped knots scale
        or stay bitwise, so the basis and the penalty, and so the fit, are
        bitwise those of the original units. Over ``x`` the penalty scaled by
        ``units**-3``. ``ps``, whose difference penalty never saw the units,
        is the control."""
        import pandas as pd

        from superglm import SuperGLM

        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 10.0, 2_000)
        y = rng.poisson(np.exp(0.3 + 0.5 * np.sin(x / 1.5))).astype(float)

        def fitted(scale):
            model = SuperGLM(family="poisson", features={"x": Spline(n_knots=8, **kwargs)})
            frame = pd.DataFrame({"x": scale * x})
            return model.fit(frame, y).predict(frame)

        np.testing.assert_array_equal(fitted(units), fitted(1.0))

    def test_a_shaped_range_fit_ignores_the_columns_units(self):
        """The range's real penalty and its structural one, in knot intervals too."""
        import pandas as pd

        from superglm import PolynomialRange, SuperGLM

        rng = np.random.default_rng(4)
        x = rng.uniform(0.0, 10.0, 2_000)
        y = rng.poisson(np.exp(0.3 + 0.5 * np.sin(x / 1.5))).astype(float)

        def fitted(scale):
            ranges = [PolynomialRange(3.0 * scale, 6.0 * scale, 1)]
            spec = Spline(kind="cr", n_knots=8, polynomial_ranges=ranges)
            frame = pd.DataFrame({"x": scale * x})
            return SuperGLM(family="poisson", features={"x": spec}).fit(frame, y).predict(frame)

        np.testing.assert_array_equal(fitted(2.0**10), fitted(1.0))

    def test_reml_bounds_a_near_linear_term_in_knot_intervals(self):
        """REML clips lambda to [1e-6, 1e10]. In 0.39 the bounds applied in the
        column's units: at 2**20 times larger units the cap, 1e10, is about 7e-9 in
        knot intervals, far under this term's optimum of 1.5e4, so the scaled fit
        stopped there and differed by up to 6.8%. The bounds now apply in knot
        intervals: the optimum is the same at both scales and well inside them,
        and since scaling by a power of two is exact, the fits agree bitwise."""
        import pandas as pd

        from superglm import SuperGLM

        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, 10.0, 4_000)
        y = rng.poisson(np.exp(0.2 + 0.08 * x)).astype(float)

        def fitted(scale):
            frame = pd.DataFrame({"x": scale * x})
            model = SuperGLM(family="poisson", features={"x": Spline(n_knots=8)})
            model.fit_reml(frame, y)
            return model._reml_lambdas["x"], model.predict(frame)

        (large, scaled), (unit, plain) = fitted(2.0**20), fitted(1.0)
        assert large == unit
        assert 1e-6 * 1e2 < unit < 1e10 / 1e2
        np.testing.assert_array_equal(scaled, plain)

    @pytest.mark.parametrize("kind", ["bs", "cr"])
    def test_a_range_leaves_the_penalty_away_from_it_unchanged(self, kind):
        """hbar counted a range's edges: PolynomialRange(3, 6) on eight knots over
        [0, 10] turns nine knot intervals into eight, which scaled the rest of the
        curve's penalty by (9/8)**3 at a fixed spline_penalty. The last three basis
        functions' support, from the knot at 6.67, is clear of the range, so their
        block is the same with the range as without it."""
        from superglm import PolynomialRange

        x = np.random.default_rng(4).uniform(0.0, 10.0, 2_000)

        def block(ranges):
            spec = Spline(kind=kind, n_knots=8, polynomial_ranges=ranges)
            spec.build(x)
            return spec._build_penalty_for_order(2)[-3:, -3:]

        np.testing.assert_array_equal(block([PolynomialRange(3.0, 6.0, 1)]), block([]))

    def test_on_even_knots_the_curvature_penalty_is_a_sandwiched_difference_penalty(self):
        """``hbar**3 integral f''**2 = (D2 beta)' G (D2 beta)`` on evenly spaced
        knots (Li and Cao, arXiv:2201.06808, section 2.4), ``G`` the Gram of the
        linear B-splines on unit spacing, ``tridiag(1/6, 2/3, 1/6)``: the
        standard second-difference penalty weighted by a matrix whose
        eigenvalues lie in (1/3, 1), which is why a fixed ``spline_penalty``
        smooths these kinds about as a P-spline. Gauss-Legendre is exact on
        the piecewise quadratic integrand, so the two agree to the round-off
        of the quadrature's sums. The comparison is on the basis functions
        four or more from either end, whose entries see only knot intervals
        inside ``[t[3], t[n]]``: the builder also integrates the intervals
        outside it, where the evaluator extends the end polynomials."""
        from superglm.features._spline_penalties import build_integrated_derivative_penalty

        knots = 0.37 * np.arange(-3.0, 16.0)  # open, evenly spaced, cubic
        penalty = build_integrated_derivative_penalty(knots, 3, 2)
        n_basis = penalty.shape[0]
        D2 = np.diff(np.eye(n_basis), n=2, axis=0)
        G = (
            np.diag(np.full(n_basis - 2, 2.0 / 3.0))
            + np.diag(np.full(n_basis - 3, 1.0 / 6.0), 1)
            + np.diag(np.full(n_basis - 3, 1.0 / 6.0), -1)
        )
        inside = slice(4, n_basis - 4)
        expected = (D2.T @ G @ D2)[inside, inside]
        penalty = penalty[inside, inside]
        u = np.finfo(np.float64).eps / 2
        # Each entry sums at most 4 intervals' blocks of 3 nodes, each a product
        # of B-spline derivative values below 4 in magnitude, over unit knots
        # within u of their places: 64 n u of the largest entry covers both.
        tolerance = 64 * n_basis * u * np.abs(expected).max()
        assert np.abs(penalty - expected).max() <= tolerance
