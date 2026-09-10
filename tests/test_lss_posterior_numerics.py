"""Analytic posterior controls; no fitting or empirical covariance thresholds."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import hadamard

import superglm.distributional.posterior as posterior
from superglm.types import PenaltyComponent
from tests.test_lss_posterior import _corrected_shim, _covariance_shim


def _diagonal_hessian_fit(exponent: int):
    names = ("location:x#first", "location:x#second")
    components = tuple(
        PenaltyComponent(
            name=name,
            group_name="location:x",
            group_index=0,
            group_sl=slice(0, 2),
            omega_raw=np.diag([float(index == 0), float(index == 1)]),
            omega_ssp=np.diag([float(index == 0), float(index == 1)]),
            rank=1.0,
        )
        for index, name in enumerate(names)
    )
    fitted = _corrected_shim(
        lambdas=dict.fromkeys(names, 1.0),
        gradient=dict.fromkeys(names, 0.0),
        hessian=np.diag([1.0, np.ldexp(1.0, exponent)]),
        penalties=components,
    )
    # J = -diag(0, 2**(exponent/2)), so J H^-1 J.T = diag(0, 1).
    fitted.result.coefficients[:] = [0.0, np.ldexp(1.0, exponent // 2)]
    return fitted


@pytest.mark.parametrize("exponent", [-40, -54, -1060])
def test_corrected_covariance_keeps_certified_positive_hessian_direction(exponent):
    fitted = _diagonal_hessian_fit(exponent)
    result = posterior.posterior_covariance(fitted, kind="corrected")
    np.testing.assert_array_equal(result, np.diag([1.0, 2.0]))


@pytest.mark.parametrize("powers", [(0, 0), (200, -300)])
def test_corrected_covariance_applies_correlated_hessian_in_the_right_coordinates(powers):
    fitted = _diagonal_hessian_fit(-40)
    powers = np.array(powers)
    fitted.smoothing.smoothing_hessian = np.ldexp(
        np.array([[2.0, 1.0], [1.0, 2.0]]), powers[:, None] + powers[None, :]
    )
    fitted.result.coefficients[:] = np.ldexp(np.array([1.0, 2.0]), powers)
    result = posterior.posterior_covariance(fitted, kind="corrected")
    # H^-1 = [[2,-1],[-1,2]]/3 before the congruence; J = -diag(1,2) D.
    expected = np.array([[5.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 11.0 / 3.0]])
    bound = 16 * len(result) * np.finfo(float).eps * np.linalg.norm(expected)
    assert np.linalg.norm(result - expected) <= bound


@pytest.mark.parametrize("replayed", [False, True])
def test_hessian_certificate_must_exclude_an_indefinite_enclosed_matrix(replayed):
    gap = 2.0**-30
    hessian = np.array([[1.0, 1.0 - gap], [1.0 - gap, 1.0]])
    bound = np.array([[0.0, 2.0 * gap], [2.0 * gap, 0.0]])
    # H+B has exact eigenvalue -gap although nominal H has eigenvalue +gap.
    with pytest.raises(RuntimeError, match="positive|certificate|invertib"):
        posterior._trusted_smoothing_hessian(
            hessian, bound, count=2, certificate_fraction=0.1, replayed=replayed
        )


def test_hessian_enclosure_with_resolved_positive_lower_bound_is_accepted():
    gap = 2.0**-30
    hessian = np.array([[1.0, 1.0 - gap], [1.0 - gap, 1.0]])
    bound = np.array([[0.0, gap / 16.0], [gap / 16.0, 0.0]])
    result = posterior._trusted_smoothing_hessian(
        hessian, bound, count=2, certificate_fraction=0.1, replayed=False
    )
    np.testing.assert_array_equal(result, hessian)


@pytest.mark.parametrize("exponent", [-30, -1060])
def test_sampling_retains_variance_under_power_of_two_coefficient_units(exponent):
    ordinary = posterior.posterior_draws(_covariance_shim(np.eye(2)), 16, seed=713)
    scaled = posterior.posterior_draws(
        _covariance_shim(np.diag([1.0, np.ldexp(1.0, exponent)])), 16, seed=713
    )
    recovered = scaled.coefficients.copy()
    recovered[:, 1] = np.ldexp(recovered[:, 1], -exponent // 2)
    np.testing.assert_array_equal(recovered, ordinary.coefficients)


def test_correlated_sampling_is_invariant_to_exact_diagonal_unit_changes():
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    powers = np.array([200, -300])
    scaled_covariance = np.ldexp(covariance, powers[:, None] + powers[None, :])
    ordinary = posterior.posterior_draws(_covariance_shim(covariance), 16, seed=911)
    scaled = posterior.posterior_draws(_covariance_shim(scaled_covariance), 16, seed=911)
    np.testing.assert_array_equal(
        np.ldexp(scaled.coefficients, -powers[None, :]), ordinary.coefficients
    )


def test_sampling_keeps_a_resolved_positive_correlated_mode():
    gap = 2.0**-30
    covariance = np.array([[1.0, 1.0 - gap], [1.0 - gap, 1.0]])
    root = posterior._posterior_covariance_factor(covariance)
    contrast = root.T @ np.array([1.0, -1.0])
    variance = float(contrast @ contrast)
    # The exact contrast variance is 2*gap; use an absolute reconstruction
    # bound for this ill-conditioned covariance, not small-eigenvalue accuracy.
    bound = 64 * np.finfo(float).eps * np.linalg.norm(covariance)
    assert abs(variance - 2 * gap) <= bound
    assert variance > 0.0


@pytest.mark.parametrize("value", [np.nextafter(0.0, 1.0), np.ldexp(1.0, 1023)])
def test_sampling_preserves_finite_covariance_at_both_exponent_limits(value):
    fitted = _covariance_shim(np.array([[value]]))
    draws = posterior.posterior_draws(fitted, 16, seed=51)
    expected = np.random.default_rng(51).standard_normal((16, 1)) * np.sqrt(value)
    np.testing.assert_allclose(draws.coefficients, expected, rtol=4 * np.finfo(float).eps, atol=0.0)


def test_sampling_refuses_negative_variance_even_if_edf_tolerance_is_large():
    fitted = _covariance_shim(np.diag([1.0, -1.0e-9]))
    with pytest.raises(ValueError, match="negative|semidefinite"):
        posterior.posterior_draws(fitted, 16)


def test_sampling_clips_only_arithmetic_size_psd_boundary_defect():
    rounded = np.array([[1.0, 1.0 + np.finfo(float).eps], [1.0 + np.finfo(float).eps, 1.0]])
    draws = posterior.posterior_draws(_covariance_shim(rounded), 16, seed=113)
    np.testing.assert_allclose(
        draws.coefficients[:, 0] - draws.coefficients[:, 1],
        0.0,
        atol=4 * np.finfo(float).eps * np.max(np.abs(draws.coefficients)),
        rtol=0.0,
    )
    assert np.any(draws.coefficients[:, 0] != 0.0)
    indefinite = np.array([[1.0, 1.0 + 2.0**-30], [1.0 + 2.0**-30, 1.0]])
    with pytest.raises(ValueError, match="negative|semidefinite|reconstruct"):
        posterior.posterior_draws(_covariance_shim(indefinite), 16, seed=113)


def test_sampling_keeps_exact_zero_variance_and_zero_covariance():
    draws = posterior.posterior_draws(_covariance_shim(np.diag([1.0, 0.0])), 16, seed=43)
    np.testing.assert_array_equal(draws.coefficients[:, 1], 0.0)
    assert np.any(draws.coefficients[:, 0] != 0.0)
    zero = posterior.posterior_draws(_covariance_shim(np.zeros((2, 2))), 16, seed=43)
    np.testing.assert_array_equal(zero.coefficients, 0.0)


def test_sampling_checks_reconstruction_instead_of_trusting_eigensolver(monkeypatch):
    eigen = np.linalg.eigh

    def missing_direction(matrix):
        values, vectors = eigen(matrix)
        values[0] = 0.0
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", missing_direction)
    with pytest.raises(ValueError, match="reconstruct|semidefinite|numerical"):
        posterior.posterior_draws(_covariance_shim(np.eye(2)), 16)


def test_binary_congruence_encloses_off_diagonal_scaling_underflow():
    matrix = np.array([[2.0**1000, 2.0**-100], [2.0**-100, 2.0**1000]])
    scaled, error, powers = posterior._binary_congruence(matrix)
    for i in range(2):
        for j in range(2):
            exact = Fraction.from_float(matrix[i, j]) / (Fraction(2) ** int(powers[i] + powers[j]))
            assert abs(Fraction.from_float(scaled[i, j]) - exact) <= Fraction.from_float(
                error[i, j]
            )
    assert scaled[0, 1] == 0.0
    assert error[0, 1] > 0.0


def test_hessian_positivity_does_not_trust_a_corrupted_triangular_inverse(monkeypatch):
    monkeypatch.setattr(
        posterior.linalg, "solve_triangular", lambda matrix, rhs, **kwargs: rhs * 0.0
    )
    with pytest.raises(RuntimeError, match="positive invertibility"):
        posterior._trusted_smoothing_hessian(
            np.eye(2), np.zeros((2, 2)), count=2, certificate_fraction=0.1, replayed=False
        )


def test_posterior_controls_with_binary64_working_arithmetic(monkeypatch):
    import superglm.reml.multi_penalty as kernel

    for module in (kernel, posterior):
        monkeypatch.setattr(module, "_LD", np.float64)
        monkeypatch.setattr(module, "_U_LD", np.finfo(float).eps / 2)
    monkeypatch.setattr(kernel, "_TINY_LD", np.nextafter(0.0, 1.0))
    result = posterior.posterior_covariance(_diagonal_hessian_fit(-1060), kind="corrected")
    np.testing.assert_array_equal(result, np.diag([1.0, 2.0]))
    test_hessian_certificate_must_exclude_an_indefinite_enclosed_matrix(False)
    test_sampling_retains_variance_under_power_of_two_coefficient_units(-1060)
    test_sampling_clips_only_arithmetic_size_psd_boundary_defect()
    test_binary_congruence_encloses_off_diagonal_scaling_underflow()


@pytest.mark.parametrize("binary64", [False, True])
def test_sampling_accepts_exact_spd_hadamard_covariance_on_both_working_precisions(
    monkeypatch, binary64
):
    import superglm.reml.multi_penalty as kernel

    if binary64:
        for module in (kernel, posterior):
            monkeypatch.setattr(module, "_LD", np.float64)
            monkeypatch.setattr(module, "_U_LD", np.finfo(float).eps / 2)
        monkeypatch.setattr(kernel, "_TINY_LD", np.nextafter(0.0, 1.0))
    signs = hadamard(8).astype(float)
    values = 1.0 + np.arange(8) * 2.0**-20
    covariance = (signs * values) @ signs.T / 8
    # The integer orthogonality identity proves the exact dyadic matrix is
    # SPD with eigenvalues 1+k*2^-20, independently of an eigensolver.
    np.testing.assert_array_equal(signs @ signs.T, 8 * np.eye(8))
    root = posterior._posterior_covariance_factor(covariance)
    magnitude = np.linalg.norm(np.abs(root) @ np.abs(root.T))
    assert np.linalg.norm(root @ root.T - covariance) <= (
        16 * len(covariance) * np.finfo(float).eps * magnitude
    )


@pytest.mark.parametrize("terms", [2, 4])
def test_corrected_covariance_accumulates_subnormal_terms_before_materialization(terms):
    names = tuple(f"location:x#{i}" for i in range(terms))
    components = tuple(
        PenaltyComponent(
            name=name,
            group_name="location:x",
            group_index=0,
            group_sl=slice(0, 1),
            omega_raw=np.ones((1, 1)),
            omega_ssp=np.ones((1, 1)),
            rank=1.0,
        )
        for name in names
    )
    fitted = _corrected_shim(
        lambdas=dict.fromkeys(names, 1.0),
        gradient=dict.fromkeys(names, 0.0),
        hessian=np.eye(terms),
        penalties=components,
    )
    tiny = np.nextafter(0.0, 1.0)
    fitted.inference.covariance = np.diag([tiny, 1.0])
    fitted.result.coefficients[:] = [2.0**536, 0.0]
    fitted.result.solve_terminal = lambda rhs: fitted.inference.covariance @ rhs
    exact = Fraction.from_float(tiny) + terms * Fraction(2) ** -1076
    # Two terms test a half-minsub correction before adding fixed variance;
    # four terms test products lost individually despite a representable sum.
    assert float(exact) == 2 * tiny
    actual = posterior.posterior_covariance(fitted, kind="corrected")
    np.testing.assert_array_equal(actual, np.diag([float(exact), 1.0]))


def test_corrected_covariance_preserves_small_cross_terms_in_large_rows():
    fitted = _diagonal_hessian_fit(-40)
    covariance = np.array([[2.0**500, 2.0**-1000], [2.0**-1000, 2.0**500]])
    fitted.inference.covariance = covariance
    fitted.result.coefficients[:] = 1.0
    fitted.result.solve_terminal = lambda rhs: covariance @ rhs
    fitted.smoothing.smoothing_hessian = np.eye(2)
    actual = posterior.posterior_covariance(fitted, kind="corrected")
    expected = float(Fraction(2) ** -1000 + 2 * Fraction(2) ** -500)
    assert actual[0, 1] == actual[1, 0] == expected
    assert np.all(np.isfinite(actual))
