"""Tests for the Spline(kind=..., k=...) factory API."""

import warnings

import numpy as np
import pytest

from superglm.features.spline import (
    BasisSpline,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
    Spline,
    _SplineBase,
    n_knots_from_k,
)

# ── Factory dispatch ─────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestSplineFactoryDispatch:
    """Spline() should dispatch to the correct concrete class."""

    def test_bs_default(self):
        s = Spline(n_knots=8)
        assert isinstance(s, BasisSpline)

    def test_bs_explicit(self):
        s = Spline(kind="bs", n_knots=8, penalty="ssp")
        assert isinstance(s, BasisSpline)
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
        assert isinstance(s, BasisSpline)
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

    def test_cr_ignores_degree(self):
        """CR is always cubic regardless of degree param."""
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


@pytest.mark.filterwarnings("ignore::FutureWarning")
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
        assert isinstance(s, BasisSpline)
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


@pytest.mark.filterwarnings("ignore::FutureWarning")
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


# ── Backward compatibility ───────────────────────────────────────


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestBackwardCompat:
    """Existing code using concrete classes or old Spline(...) syntax still works."""

    def test_basis_spline_direct(self):
        s = BasisSpline(n_knots=8, penalty="ssp")
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
        """Spline(n_knots=8, penalty='ssp') still works (defaults to kind='ps')."""
        s = Spline(n_knots=8, penalty="ssp")
        assert isinstance(s, PSpline)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_default_n_knots(self):
        """Spline() with no size arg uses n_knots=10 default."""
        s = Spline()
        assert isinstance(s, PSpline)
        assert s.n_knots == 10


# ── PSpline factory dispatch ────────────────────────────────────


class TestPSplineFactory:
    """Tests for the new kind='ps' dispatch path."""

    def test_ps_dispatch(self):
        """Spline(kind='ps') should dispatch to PSpline."""
        s = Spline(kind="ps", n_knots=8)
        assert isinstance(s, PSpline)

    def test_ps_isinstance_basisspline(self):
        """PSpline instances are also BasisSpline (alias)."""
        s = Spline(kind="ps", n_knots=8)
        assert isinstance(s, BasisSpline)

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

    def test_ps_builds_same_as_old_bs(self):
        """kind='ps' and kind='bs' produce identical spline objects."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            s_bs = Spline(kind="bs", n_knots=8)
        s_ps = Spline(kind="ps", n_knots=8)
        assert type(s_bs) is type(s_ps)
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

    def test_basisspline_is_pspline_alias(self):
        """BasisSpline is PSpline (backward-compatible alias)."""
        assert BasisSpline is PSpline

    def test_ps_k_mapping(self):
        """k parameter works with kind='ps'."""
        s = Spline(kind="ps", k=14)
        assert s.n_knots == 10  # 14 - 3 - 1

    def test_n_knots_from_k_ps(self):
        """n_knots_from_k accepts 'ps' kind."""
        assert n_knots_from_k("ps", 14, degree=3) == 10
        assert n_knots_from_k("ps", 20, degree=3) == 16


# ── Default kind is "ps" ────────────────────────────────────────


class TestDefaultKindIsPs:
    """The Spline() factory now defaults to kind='ps'."""

    def test_default_is_pspline(self):
        s = Spline(n_knots=8)
        assert isinstance(s, PSpline)

    def test_default_no_warning(self):
        """Default kind='ps' should not emit a FutureWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Spline(n_knots=8)  # should not raise


# ── kind="bs" deprecation warning ───────────────────────────────


class TestBsDeprecation:
    """kind='bs' emits a FutureWarning; kind='ps' does not."""

    def test_bs_emits_future_warning(self):
        with pytest.warns(FutureWarning, match="kind='bs'"):
            Spline(kind="bs", n_knots=8)

    def test_ps_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Spline(kind="ps", n_knots=8)  # should not raise

    def test_bs_warning_message_content(self):
        with pytest.warns(FutureWarning, match="integrated-derivative"):
            Spline(kind="bs", n_knots=8)

    def test_bs_still_creates_pspline(self):
        """Even with the warning, kind='bs' still creates a PSpline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            s = Spline(kind="bs", n_knots=8)
        assert isinstance(s, PSpline)
