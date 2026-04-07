"""Tests for BSplineSmooth — B-spline smooth with integrated-derivative penalty."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from superglm.features.spline import (
    BSplineSmooth,
    PSpline,
    _SplineBase,
    n_knots_from_k,
)
from superglm.types import GroupInfo

RNG = np.random.default_rng(42)
X = RNG.uniform(0, 1, 500)


# ---------------------------------------------------------------------------
# Basic class properties
# ---------------------------------------------------------------------------


class TestBSplineSmoothClass:
    def test_is_spline_base_instance(self):
        spec = BSplineSmooth(n_knots=10, degree=3, penalty="none")
        assert isinstance(spec, _SplineBase)

    def test_penalty_semantics(self):
        spec = BSplineSmooth()
        assert spec._penalty_semantics == "integrated_derivative"

    def test_default_degree_is_3(self):
        spec = BSplineSmooth()
        assert spec.degree == 3

    def test_general_degree_allowed(self):
        spec = BSplineSmooth(n_knots=10, degree=4, penalty="none")
        info = spec.build(X)
        assert isinstance(info, GroupInfo)
        assert info.n_cols > 0


# ---------------------------------------------------------------------------
# Penalty matrix properties
# ---------------------------------------------------------------------------


class TestPenaltyProperties:
    @pytest.fixture()
    def info(self):
        spec = BSplineSmooth(n_knots=10, degree=3, penalty="none")
        return spec.build(X)

    def test_penalty_shape(self, info):
        S = info.penalty_matrix
        assert S.shape[0] == S.shape[1] == info.n_cols

    def test_penalty_symmetry(self, info):
        S = info.penalty_matrix
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_penalty_positive_semidefinite(self, info):
        S = info.penalty_matrix
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-10)

    def test_penalty_differs_from_pspline(self):
        """BSplineSmooth and PSpline share the same basis but different penalties."""
        bss = BSplineSmooth(n_knots=10, degree=3, penalty="none")
        ps = PSpline(n_knots=10, degree=3, penalty="none")
        info_bss = bss.build(X)
        info_ps = ps.build(X)
        # Same number of columns
        assert info_bss.n_cols == info_ps.n_cols
        # But different penalty values
        assert not np.allclose(info_bss.penalty_matrix, info_ps.penalty_matrix)


# ---------------------------------------------------------------------------
# Analytic penalty tests (penalty="none" to avoid SSP reparametrisation)
# ---------------------------------------------------------------------------


def _fit_raw_and_penalise(spec, x, y, order):
    """Fit y = f(x) on the raw (unprojected) basis, compute beta^T S_raw beta.

    Uses the raw penalty from ``_build_penalty_for_order`` so that the
    identifiability projection does not alter the penalty integral.
    Returns the scalar penalty value.
    """
    info = spec.build(x)
    B_raw = (
        np.asarray(info.columns.todense())
        if hasattr(info.columns, "todense")
        else np.asarray(info.columns)
    )
    beta, _, _, _ = np.linalg.lstsq(B_raw, y, rcond=None)
    S_raw = spec._build_penalty_for_order(order)
    return float(beta @ S_raw @ beta)


class TestAnalyticPenalties:
    """Verify integral penalty values against known analytic results.

    Uses the raw (unprojected) penalty so that the identifiability
    constraint does not change the integral domain.  Expected values
    are computed analytically over the full knot span [a, b] (which
    extends beyond [0, 1] due to open-knot padding).
    """

    def test_linear_has_zero_second_deriv_penalty(self):
        """f(x)=x, f''=0, so int (f'')^2 dx = 0 everywhere."""
        x = np.linspace(0, 1, 500)
        y = x.copy()
        spec = BSplineSmooth(n_knots=20, degree=3, penalty="none", m=2)
        penalty_val = _fit_raw_and_penalise(spec, x, y, order=2)
        np.testing.assert_allclose(penalty_val, 0.0, atol=1e-8)

    def test_quadratic_second_deriv_penalty(self):
        """f(x)=x^2, f''=2, int_a^b 4 dx = 4*(b-a)."""
        x = np.linspace(0, 1, 500)
        y = x**2
        spec = BSplineSmooth(n_knots=20, degree=3, penalty="none", m=2)
        penalty_val = _fit_raw_and_penalise(spec, x, y, order=2)
        a, b = spec._knots[0], spec._knots[-1]
        expected = 4.0 * (b - a)
        np.testing.assert_allclose(penalty_val, expected, rtol=1e-3)

    def test_cubic_second_deriv_penalty(self):
        """f(x)=x^3, f''=6x, int_a^b (6x)^2 dx = 36*(b^3 - a^3)/3."""
        x = np.linspace(0, 1, 500)
        y = x**3
        spec = BSplineSmooth(n_knots=20, degree=3, penalty="none", m=2)
        penalty_val = _fit_raw_and_penalise(spec, x, y, order=2)
        a, b = spec._knots[0], spec._knots[-1]
        expected = 36.0 * (b**3 - a**3) / 3.0
        np.testing.assert_allclose(penalty_val, expected, rtol=1e-3)

    def test_quadratic_first_deriv_penalty(self):
        """f(x)=x^2, f'=2x, int_a^b (2x)^2 dx = 4*(b^3 - a^3)/3."""
        x = np.linspace(0, 1, 500)
        y = x**2
        spec = BSplineSmooth(n_knots=20, degree=3, penalty="none", m=1)
        penalty_val = _fit_raw_and_penalise(spec, x, y, order=1)
        a, b = spec._knots[0], spec._knots[-1]
        expected = 4.0 * (b**3 - a**3) / 3.0
        np.testing.assert_allclose(penalty_val, expected, rtol=1e-3)


class TestDifferentMOrders:
    def test_m1_and_m2_produce_different_penalties(self):
        spec1 = BSplineSmooth(n_knots=10, degree=3, penalty="none", m=1)
        spec2 = BSplineSmooth(n_knots=10, degree=3, penalty="none", m=2)
        info1 = spec1.build(X)
        info2 = spec2.build(X)
        assert not np.allclose(info1.penalty_matrix, info2.penalty_matrix)


# ---------------------------------------------------------------------------
# select=True
# ---------------------------------------------------------------------------


class TestSelectMode:
    def test_select_returns_group_info_with_two_components(self):
        spec = BSplineSmooth(n_knots=10, degree=3, select=True)
        result = spec.build(X)
        assert isinstance(result, GroupInfo)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2
        names = [name for name, _ in result.penalty_components]
        assert "null" in names
        assert "wiggle" in names
        assert result.component_types == {"null": "selection"}


# ---------------------------------------------------------------------------
# Multi-m
# ---------------------------------------------------------------------------


class TestMultiM:
    def test_multi_m_produces_penalty_components(self):
        spec = BSplineSmooth(n_knots=10, degree=3, penalty="none", m=(1, 2))
        info = spec.build(X)
        assert isinstance(info, GroupInfo)
        assert info.penalty_components is not None
        assert len(info.penalty_components) == 2


# ---------------------------------------------------------------------------
# n_knots_from_k
# ---------------------------------------------------------------------------


class TestNKnotsFromK:
    def test_ps_kind_in_n_knots_from_k(self):
        """kind='ps' should work in n_knots_from_k and give same result as 'bs'."""
        assert n_knots_from_k("ps", 14, degree=3) == n_knots_from_k("bs", 14, degree=3)
        assert n_knots_from_k("ps", 14, degree=3) == 10


# ---------------------------------------------------------------------------
# SSP reparametrisation
# ---------------------------------------------------------------------------


class TestSSP:
    def test_ssp_reparametrisation_works(self):
        spec = BSplineSmooth(n_knots=10, degree=3, penalty="ssp")
        info = spec.build(X)
        assert isinstance(info, GroupInfo)
        assert info.reparametrize is True
        assert info.penalty_matrix is not None


# ---------------------------------------------------------------------------
# Pickle roundtrip
# ---------------------------------------------------------------------------


class TestPickle:
    def test_pickle_roundtrip(self):
        spec = BSplineSmooth(n_knots=10, degree=3, penalty="none")
        info_before = spec.build(X)
        data = pickle.dumps(spec)
        spec2 = pickle.loads(data)
        info_after = spec2.build(X)
        np.testing.assert_allclose(
            np.asarray(info_before.columns.todense()),
            np.asarray(info_after.columns.todense()),
        )
        np.testing.assert_allclose(info_before.penalty_matrix, info_after.penalty_matrix)


# ---------------------------------------------------------------------------
# m parameter semantics
# ---------------------------------------------------------------------------


class TestMParameterSemantics:
    """m has different meanings for PSpline vs BSplineSmooth.

    After the identifiability constraint (one column removed), the
    null-space dimension of a 2nd-order penalty drops from 2 to 1.
    Both PSpline (difference) and BSplineSmooth (derivative) share this
    property but differ in their nonzero eigenvalue spectra.
    """

    def test_pspline_m_is_difference_order(self):
        """PSpline m=2: second-difference penalty D2^T D2.

        After identifiability constraint, null space has 1 zero eigenvalue.
        """
        s = PSpline(n_knots=8, m=2, penalty="none")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        S = info.penalty_matrix
        eigvals = np.linalg.eigvalsh(S)
        n_zero = np.sum(np.abs(eigvals) < 1e-10)
        assert n_zero == 1, f"Expected 1 zero eigenvalue for m=2 difference, got {n_zero}"

    def test_bsplinesmooth_m_is_derivative_order(self):
        """BSplineSmooth m=2: integrated second-derivative penalty.

        After identifiability constraint, null space has 1 zero eigenvalue.
        """
        s = BSplineSmooth(n_knots=8, m=2, penalty="none")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        S = info.penalty_matrix
        eigvals = np.linalg.eigvalsh(S)
        n_zero = np.sum(np.abs(eigvals) < 1e-10)
        assert n_zero == 1, f"Expected 1 zero eigenvalue for m=2 derivative, got {n_zero}"

    def test_nonzero_eigenvalues_differ(self):
        """Nonzero eigenvalues should differ between PSpline and BSplineSmooth."""
        x = np.linspace(0, 1, 200)
        ps = PSpline(n_knots=8, m=2, penalty="none")
        bs = BSplineSmooth(n_knots=8, m=2, penalty="none")
        info_ps = ps.build(x)
        info_bs = bs.build(x)
        eig_ps = np.sort(np.linalg.eigvalsh(info_ps.penalty_matrix))
        eig_bs = np.sort(np.linalg.eigvalsh(info_bs.penalty_matrix))
        nonzero_ps = eig_ps[np.abs(eig_ps) > 1e-10]
        nonzero_bs = eig_bs[np.abs(eig_bs) > 1e-10]
        assert not np.allclose(nonzero_ps, nonzero_bs, rtol=0.1)


class TestBSplineSmoothMValidation:
    """BSplineSmooth rejects m > degree (derivative order too high)."""

    def test_m_equals_degree_ok(self):
        """m=3 on degree=3 B-spline: 3rd derivative is piecewise constant,
        so int (f''')^2 dx is well-defined."""
        s = BSplineSmooth(n_knots=8, degree=3, m=3)
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.n_cols > 0
        S = info.penalty_matrix
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(S) >= -1e-10)

    def test_m_exceeds_degree_raises(self):
        s = BSplineSmooth(n_knots=8, degree=3, m=4)
        x = np.linspace(0, 1, 200)
        with pytest.raises(ValueError, match="Derivative order.*> spline degree"):
            s.build(x)

    def test_m_less_than_degree_ok(self):
        s = BSplineSmooth(n_knots=8, degree=3, m=2)
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.n_cols > 0
