"""Tests for the Spline(kind=..., k=...) factory API."""

import numpy as np
import pytest

from superglm.features.spline import (
    BasisSpline,
    CubicRegressionSpline,
    NaturalSpline,
    Spline,
    _SplineBase,
    n_knots_from_k,
)

# ── Factory dispatch ─────────────────────────────────────────────


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

    def test_bs_split_linear(self):
        s = Spline(kind="bs", n_knots=8, split_linear=True)
        assert isinstance(s, BasisSpline)
        assert s.split_linear is True

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
        # cr: n_knots = k - degree + 2 (2 natural + 1 identifiability = 3 removed)
        assert n_knots_from_k("cr", 10, degree=3) == 9  # 10 - 3 + 2 = 9
        assert n_knots_from_k("cr", 20, degree=3) == 19  # 20 - 3 + 2 = 19

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
        assert s.n_knots == 9

    def test_k_produces_correct_ncols_bs(self):
        """For bs, k should equal the number of basis columns."""
        k = 14
        s = Spline(kind="bs", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k

    def test_k_produces_correct_ncols_ns(self):
        """For ns, k should equal the post-constraint column count."""
        k = 10
        s = Spline(kind="ns", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k

    def test_k_produces_correct_ncols_cr(self):
        """For cr, k should equal the post-constraint column count."""
        k = 10
        s = Spline(kind="cr", k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == k


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

    def test_split_linear_non_bs_raises(self):
        with pytest.raises(ValueError, match="split_linear is only supported"):
            Spline(kind="ns", n_knots=8, split_linear=True)

    def test_split_linear_cr_raises(self):
        with pytest.raises(ValueError, match="split_linear is only supported"):
            Spline(kind="cr", n_knots=8, split_linear=True)

    def test_n_knots_from_k_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown spline kind"):
            n_knots_from_k("xyz", 10)

    def test_n_knots_from_k_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            n_knots_from_k("bs", 4, degree=3)  # min is 5


# ── Backward compatibility ───────────────────────────────────────


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
        """Spline(n_knots=8, penalty='ssp') still works (defaults to kind='bs')."""
        s = Spline(n_knots=8, penalty="ssp")
        assert isinstance(s, BasisSpline)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols > 0

    def test_default_n_knots(self):
        """Spline() with no size arg uses n_knots=10 default."""
        s = Spline()
        assert isinstance(s, BasisSpline)
        assert s.n_knots == 10
