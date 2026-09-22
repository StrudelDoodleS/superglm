"""Portable binary64 penalty geometry against independent exact models."""

import math
from dataclasses import replace

import numpy as np
import pytest

import superglm.reml.multi_penalty as multi
import superglm.reml.penalty_support as support


@pytest.mark.parametrize("n", [24, 64])
def test_normal_diagonal_geometry_is_certified_in_float64(n):
    # This exposed the dimension-dependent extended-arithmetic enclosure:
    # normal diagonal geometry was refused when its working type was binary64.
    components = [np.eye(n), 2 * np.eye(n)]
    weights = np.array([2.0, 3.0])
    result = multi.similarity_transform_logdet(components, weights)
    gradient = multi.logdet_s_gradient(result, components, weights)
    hessian = multi.logdet_s_hessian(result, components, weights)
    certificate = result._certificate
    assert result.rank == n
    assert certificate is not None
    eps = np.finfo(float).eps
    assert abs(result.logdet_s_plus - n * np.log(8)) <= certificate.logdet_error + n * eps
    assert np.all(np.abs(gradient - n * np.array([0.25, 0.75])) <= certificate.gradient_error)
    expected_hessian = n * 0.1875 * np.array([[1, -1], [-1, 1]])
    assert np.all(np.abs(hessian - expected_hessian) <= certificate.hessian_error)
    assert np.all(np.abs(result.S_pinv_plus - np.eye(n) / 8) <= certificate.inverse_error)
    np.testing.assert_allclose(gradient, n * np.array([0.25, 0.75]), rtol=32 * eps, atol=0)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=64 * eps, atol=0)
    assert result.S_pinv_plus.dtype == np.float64


@pytest.mark.parametrize("n", [24, 64, 128])
def test_normal_dense_geometry_has_independent_spectral_derivatives(n):
    rng = np.random.default_rng(602 + n)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    diagonal = np.linspace(1.0, 2.0, n)
    first = (basis * diagonal) @ basis.T
    second = (basis * diagonal[::-1]) @ basis.T
    weights = np.array([2.0, 3.0])
    result = multi.similarity_transform_logdet([first, second], weights)
    eigenvalues = 2 * diagonal + 3 * diagonal[::-1]
    fractions = 2 * diagonal / eigenvalues
    expected_gradient = np.array([fractions.sum(), n - fractions.sum()])
    cross = np.sum(fractions * (1 - fractions))
    expected_hessian = cross * np.array([[1.0, -1.0], [-1.0, 1.0]])
    # The spectral fixture has condition <= 2. QR, Gram extraction, and the
    # independent eigenvalue formula contribute O(n*eps) backward error.
    allowance = 64 * n * np.finfo(float).eps
    np.testing.assert_allclose(result._gradient, expected_gradient, rtol=allowance, atol=0)
    np.testing.assert_allclose(result._hessian, expected_hessian, rtol=allowance, atol=0)
    assert abs(result.logdet_s_plus - np.log(eigenvalues).sum()) <= allowance * n
    np.testing.assert_allclose(
        result.E_sqrt.T @ result.E_sqrt, 2 * first + 3 * second, rtol=0, atol=allowance * 8
    )


@pytest.mark.parametrize("width", [5, 8])
def test_tensor_difference_penalty_retains_analytic_null_space(width):
    difference = np.diff(np.eye(width), axis=0)
    penalty = difference.T @ difference
    components = [np.kron(penalty, np.eye(width)), np.kron(np.eye(width), penalty)]
    result = multi.similarity_transform_logdet(components, np.array([2.0, 3.0]))
    eigenvalues = 2 - 2 * np.cos(np.arange(width) * np.pi / width)
    total = (2 * eigenvalues[:, None] + 3 * eigenvalues[None, :]).ravel()[1:]
    fractions = (2 * eigenvalues[:, None] * np.ones((1, width))).ravel()[1:] / total
    cross = np.sum(fractions * (1 - fractions))
    assert result.rank == width**2 - 1
    allowance = 64 * width**2 * np.finfo(float).eps
    np.testing.assert_allclose(
        result.Q_zero @ result.Q_zero.T,
        np.ones((width**2, width**2)) / width**2,
        rtol=0,
        atol=allowance,
    )
    np.testing.assert_allclose(
        result._gradient, [fractions.sum(), len(total) - fractions.sum()], rtol=allowance, atol=0
    )
    np.testing.assert_allclose(
        result._hessian, cross * np.array([[1, -1], [-1, 1]]), rtol=allowance, atol=0
    )
    assert abs(result.logdet_s_plus - np.log(total).sum()) <= allowance * len(total)


def test_public_scalar_tensor_reml_completes_at_the_analytic_constant_fit():
    import pandas as pd

    from superglm import Spline, SuperGLM

    first, second = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    frame = pd.DataFrame({"first": first.ravel(), "second": second.ravel()})
    target = np.full(len(frame), 5.0)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={"first": Spline(n_knots=4), "second": Spline(n_knots=4)},
        interactions=[("first", "second")],
    ).fit_reml(frame, target, max_reml_iter=8)
    prediction = model.predict(frame)
    # Constant counts have the exact penalized solution beta=0, intercept=log(5),
    # for every positive smoothing weight. No optimizer reference is needed.
    np.testing.assert_allclose(
        prediction, target, rtol=64 * len(frame) * np.finfo(float).eps, atol=0
    )
    assert prediction.dtype == np.float64


def test_support_does_not_silently_drop_a_column_when_balancing_underflows():
    root = np.diag([1e200, 1e-200])
    with pytest.raises(support.PenaltyNumericalError, match="support|balanc"):
        support._penalty_support_from_roots(
            [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
        )


@pytest.mark.parametrize("width", [2, 4])
def test_source_root_support_does_not_materialize_overflowing_frobenius_scale(width):
    from decimal import Decimal, localcontext
    from fractions import Fraction

    magnitude, weight = 1e308, 1e-308
    root = magnitude * np.eye(width)
    selected = support._penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    result = multi._evaluate_penalty_support(selected, np.array([weight]))
    assert result.rank == width
    assert np.all(np.isfinite(result.E_sqrt))
    assert result.E_sqrt.dtype == np.float64
    exact_inverse = 1 / (Fraction.from_float(magnitude) ** 2 * Fraction.from_float(weight))
    expected = np.eye(width) * float(exact_inverse)
    assert np.all(np.abs(result.S_pinv_plus - expected) <= result._certificate.inverse_error)
    np.testing.assert_array_equal(result._gradient, [width])
    with localcontext() as context:
        context.prec = 80
        penalty = Decimal.from_float(magnitude) ** 2 * Decimal.from_float(weight)
        exact_logdet = width * penalty.ln()
    assert abs(result.logdet_s_plus - float(exact_logdet)) <= result._certificate.logdet_error


def test_direct_candidate_basis_volume_has_one_analytic_logdet_charge():
    # E0 = C B.T has logdet(E0 E0.T) = 2 log|det C| + logdet(B.T B).
    # Dropping/doubling that last charge, or retaining the rank-multiplied
    # norm bound, violates this analytic near-identity volume enclosure.
    rank, delta = 4, 2.0**-10
    root = np.column_stack([np.eye(rank), np.zeros(rank)])
    selected = support._penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    selected = replace(selected, Q_plus=(1 + delta) * root.T)
    candidate = multi._direct_candidate(selected, np.ones(1), np.finfo(float).eps ** (1 / 3))
    assert candidate is not None
    _, _, logdet, bound, _ = candidate
    volume = 2 * rank * math.log1p(delta)
    arithmetic = 64 * rank * np.finfo(float).eps
    assert abs(logdet - volume) <= arithmetic
    assert volume <= bound
    # D=t I, t=2*delta+delta**2, and ||D||_F < 1/2. The trace-series
    # remainder plus r*(t-log1p(t)) is at most 2*r*t**2; arithmetic is O(r*u).
    t = 2 * delta + delta**2
    assert bound <= volume + 2 * rank * t**2 + arithmetic


def test_direct_candidate_basis_volume_retains_off_diagonal_error(monkeypatch):
    # This exact stored basis has det(B.T B)=0.8**2. Supply a valid but
    # uncertain Gram witness so off-diagonal error cannot be mistaken for zero.
    root = np.column_stack([np.eye(2), np.zeros(2)])
    selected = support._penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    basis = np.array([[1.0, 0.6], [0.0, 0.8], [0.0, 0.0]])
    selected = replace(selected, Q_plus=basis)
    gram = np.array([[1.0, 0.3], [0.3, 1.0]])
    error = np.array([[0.0, 0.3], [0.3, 4 * np.finfo(float).eps]])
    monkeypatch.setattr(multi, "_basis_gram", lambda *_: (gram, error))
    candidate = multi._direct_candidate(selected, np.ones(1), np.finfo(float).eps ** (1 / 3))
    assert candidate is not None
    assert abs(2 * math.log(0.8)) <= candidate[3]


@pytest.mark.parametrize("uncertainty", [0.25, 1.0])
def test_direct_candidate_basis_volume_refuses_uncertified_gram(monkeypatch, uncertainty):
    # The observed product is nonsingular I. At uncertainty=1 its enclosure
    # also permits a zero eigenvalue, so a finite candidate is not certified.
    root = np.column_stack([np.eye(2), np.zeros(2)])
    selected = support._penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    gram, error = np.eye(2), np.diag([uncertainty, 0.0])
    monkeypatch.setattr(multi, "_basis_gram", lambda *_: (gram, error))
    candidate = multi._direct_candidate(selected, np.ones(1), np.finfo(float).eps ** (1 / 3))
    assert (candidate is None) == (uncertainty == 1.0)


def test_direct_candidate_basis_volume_reuses_gram_without_duplicate_norms(monkeypatch):
    # Dispatch/work is separate from the analytic volume assertions above.
    root = np.column_stack([np.eye(2), np.zeros(2)])
    selected = support._penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    original_gram, original_bound = multi._basis_gram, multi._logdet_defect_bound
    original_norm, original_materialization = multi._norm_upper, multi._materialization_logdet_bound
    pairs, consumed, norm_calls, before_materialization = [], [], [], []

    def gram(*args):
        pair = original_gram(*args)
        pairs.append(pair)
        return pair

    def bound(product, error):
        consumed.append((product, error))
        return original_bound(product, error)

    def norm(value):
        norm_calls.append(1)
        return original_norm(value)

    def materialization(*args, **kwargs):
        before_materialization.append(len(norm_calls))
        return original_materialization(*args, **kwargs)

    monkeypatch.setattr(multi, "_basis_gram", gram)
    monkeypatch.setattr(multi, "_logdet_defect_bound", bound)
    monkeypatch.setattr(multi, "_norm_upper", norm)
    monkeypatch.setattr(multi, "_materialization_logdet_bound", materialization)
    assert multi._direct_candidate(selected, np.ones(1), np.finfo(float).eps ** (1 / 3)) is not None
    assert len(pairs) == len(consumed) == 1
    assert all(a is b for a, b in zip(pairs[0], consumed[0], strict=True))
    assert before_materialization == [2]
