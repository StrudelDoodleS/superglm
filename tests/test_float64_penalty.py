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


def _dense_component(omega: np.ndarray, rank: int):
    from superglm.types import PenaltyComponent

    width = omega.shape[0]
    return PenaltyComponent(
        name="x",
        group_name="x",
        group_index=0,
        group_sl=slice(0, width),
        omega_raw=None,
        omega_ssp=omega,
        rank=float(rank),
    )


def test_the_range_product_stays_in_range_where_the_dense_product_does():
    """A penalty of ``1e-310`` against coefficients of ``1.5e308``: ``Omega beta`` is about 0.03.

    The product over the penalty's range formed ``V' beta`` (``2.1e308``)
    first, which overflows before the eigenvalue brings it back into range,
    so the PIRLS score and its rounding floor read ``inf`` where the dense
    product was finite.  Scaling ``beta`` and ``Lambda`` by powers of two
    keeps every intermediate normal.  Against the exact ``Omega beta`` of the
    stored values the error is the eigensolver's backward error, ``w u
    ||Omega||_2`` for the operator and as much for the dropped eigenvalue
    (the file's ``p u`` convention), the product's ``gamma_{2w+1}`` of its
    magnitude, and the subnormal eigenvalue's own rounding, ``2**-1075``
    times ``|V| |V'| |beta| <= ||beta||_1``.  Mutation: a19d2fe4 returned
    ``inf`` for both the product and its magnitude.
    """
    from fractions import Fraction

    from superglm.reml.penalty_algebra import (
        penalty_component_magnitude_matvec,
        penalty_component_matvec,
    )

    omega = np.full((2, 2), 1e-310)
    beta = np.full(2, 1.5e308)
    component = _dense_component(omega, 1)
    product = penalty_component_matvec(component, beta)
    magnitude = penalty_component_magnitude_matvec(component, np.abs(beta))
    assert np.all(np.isfinite(product)) and np.all(np.isfinite(magnitude))
    exact = np.array(
        [
            float(
                sum(Fraction(float(o)) * Fraction(float(b)) for o, b in zip(row, beta, strict=True))
            )
            for row in omega
        ]
    )
    u = 2.0**-53
    width = omega.shape[0]
    gamma = (2 * width + 1) * u / (1 - (2 * width + 1) * u)
    # products of norms formed by powers of two that keep each factor normal
    shift = 2.0**600
    operator = float(np.linalg.norm(omega * shift, 2)) * float(np.linalg.norm(beta / shift))
    subnormal = float(np.ldexp(float(np.sum(np.abs(beta) / 2.0)), -1074))
    tolerance = 2.0 * width * u * operator + gamma * magnitude + subnormal
    assert np.all(np.abs(product - exact) <= tolerance)
    assert np.all(magnitude >= (1.0 - 2.0 * gamma) * np.abs(product))


def test_the_range_product_scaling_changes_no_bit_in_the_normal_range():
    """Powers of two are exact: a normal-range product is the unscaled one bit for bit."""
    from superglm.reml.penalty_algebra import (
        _penalty_range_basis,
        penalty_component_magnitude_matvec,
        penalty_component_matvec,
    )

    rng = np.random.default_rng(31)
    root = rng.normal(size=(4, 7))
    omega = root.T @ root * 3.7
    beta = rng.normal(size=7) * 250.0
    component = _dense_component(omega, 4)
    values, vectors = _penalty_range_basis(component, omega, 4)
    unscaled = vectors @ (values * (vectors.T @ beta))
    absolute = np.abs(vectors)
    unscaled_size = absolute @ (np.abs(values) * (absolute.T @ np.abs(beta)))
    assert np.array_equal(penalty_component_matvec(component, beta), unscaled)
    assert np.array_equal(
        penalty_component_magnitude_matvec(component, np.abs(beta)), unscaled_size
    )


def test_the_penalty_product_rounding_counts_the_range_chain():
    """``penalty_product_rounding`` counts every rounding of an entry of a penalty product.

    Two rank-deficient dense penalties on the same 12 coefficients, each formed
    over its range: ``2 w`` roundings per component, one for its ``lambda``,
    one to add the second component and one for the magnitude's own:
    ``2 w + m + 1 = 27``.  The count the product's readers used, ``p + 2 =
    14``, was the dense product's.  Against exact rational arithmetic on the
    same eigenpairs, the formed sum is within that ``gamma_k`` of its formed
    magnitude.  Mutation check: a19d2fe4 had no count but ``p + 2``.
    """
    from fractions import Fraction

    from superglm.reml.penalty_algebra import (
        _penalty_range_basis,
        penalty_component_magnitude_matvec,
        penalty_component_matvec,
        penalty_product_rounding,
    )

    rng = np.random.default_rng(7)
    width = 12
    components = []
    for name, rank in (("x", 9), ("y", 10)):
        root = rng.normal(size=(rank, width))
        component = _dense_component(root.T @ root, rank)
        components.append(replace(component, name=name))
    assert penalty_product_rounding(width, components[:1]) == 2 * width + 1 + 1
    assert penalty_product_rounding(width, components) == 2 * width + 2 + 1
    assert penalty_product_rounding(width, None) == width + 2
    lambdas = {"x": 3.0, "y": 0.25}
    beta = rng.normal(size=width) * 40.0
    product = np.zeros(width)
    magnitude = np.zeros(width)
    exact = [Fraction(0)] * width
    for component in components:
        lam = lambdas[component.name]
        product += lam * penalty_component_matvec(component, beta)
        magnitude += lam * penalty_component_magnitude_matvec(component, np.abs(beta))
        values, vectors = _penalty_range_basis(component, component.omega_ssp, int(component.rank))
        projection = [
            sum(Fraction(float(v)) * Fraction(float(b)) for v, b in zip(column, beta, strict=True))
            for column in vectors.T
        ]
        for row in range(width):
            exact[row] += Fraction(lam) * sum(
                Fraction(float(vectors[row, k])) * Fraction(float(values[k])) * projection[k]
                for k in range(len(values))
            )
    k = penalty_product_rounding(width, components)
    u = 2.0**-53
    gamma = k * u / (1 - k * u)
    error = np.abs(product - np.array([float(value) for value in exact]))
    assert np.all(error <= gamma * magnitude)


def test_the_solver_passes_the_penalty_product_rounding_to_its_mode_residual(monkeypatch):
    """Every mode residual the structured solver forms reads the derived count.

    A spline (9 coefficients, over its range) beside a 40-level random effect
    (an identity): ``2 * 9 + 1 + 1``.  Mutation check: a19d2fe4 passed none,
    so the residual's floor used the dense product's ``p + 2``.
    """
    import pandas as pd

    import superglm.reml.penalty_algebra as penalty_algebra
    import superglm.solvers.irls_direct as irls_direct
    from superglm import RandomEffect, Spline, SuperGLM

    counts = []
    real_count = penalty_algebra.penalty_product_rounding

    def count(width, components):
        counts.append((width, real_count(width, components)))
        return counts[-1][1]

    received = []
    real_residual = irls_direct.penalized_mode_residual

    def residual(**kwargs):
        received.append(kwargs.get("penalty_rounding"))
        return real_residual(**kwargs)

    monkeypatch.setattr(penalty_algebra, "penalty_product_rounding", count)
    monkeypatch.setattr(irls_direct, "penalized_mode_residual", residual)
    rng = np.random.default_rng(3)
    n = 600
    levels = [f"g{i:02d}" for i in range(40)]
    frame = pd.DataFrame({"a": rng.uniform(size=n), "g": rng.choice(levels, size=n)})
    effect = dict(zip(levels, rng.normal(0.0, 0.3, 40), strict=True))
    y = rng.poisson(np.exp(0.3 + np.sin(4 * frame["a"]) + frame["g"].map(effect))).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"a": Spline(kind="ps", k=10), "g": RandomEffect()},
    )
    model.fit_reml(frame, y)
    assert model._reml_profile["direct_backend"] == "structured"
    assert received and None not in received
    assert set(received) <= {k for _, k in counts}
    # the spline's range chain and the random effect's identity, not the dense p + 2
    assert all(k == 2 * 9 + 1 + 1 != width + 2 for width, k in counts)


def test_the_truncated_direction_pull_is_charged_the_penalty_product_rounding():
    """The pull ``D' S beta`` reads the penalty product, so it carries ``gamma_{k + 2p + 2}``.

    The direction lies in the penalty's null space, so its bend is exactly 0
    at any count and ``penalty_rounding`` reaches the floor only through the
    pull's charge.  With a pull size of 1e-2 per coefficient, ``k = 2**40``
    puts that charge above the step (the rows sit at their own maximum,
    ratio 0), where the dense default leaves the step refused.  Mutation
    check: 71c697b0 charged the pull ``gamma_{p + 2}`` whatever the count,
    and read 1.4e9 at both.
    """
    import math

    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import MODE_CERTIFICATION_BAR, truncated_direction_ratio

    codes = np.array([-1, -1, -1, -1, 0, 0, 1, 1, 2, 2])
    response = np.array([0.0, 1.0, 0.0, 1.0] + [1.0] * 6)
    weight = np.where(codes < 0, 1e8, 1e-8)
    odds = math.exp(-20.0) / -math.expm1(-20.0)
    second = np.array([[1.0, -2.0, 1.0]])
    penalty = 1e6 * (second.T @ second)
    direction = np.array([[1.0], [2.0], [3.0]]) / 4.0
    assert np.all(penalty @ direction == 0.0)

    def ratio(rounding):
        value, _ = truncated_direction_ratio(
            dm=DesignMatrix([CategoricalGroupMatrix(codes, 3)], n=10, p=3),
            null_basis=direction,
            angle=1e-12,
            mean_x=np.zeros(3),
            row_score=np.where(codes < 0, weight * (2.0 * response - 1.0), weight),
            fisher_weights=np.where(codes < 0, weight, weight * odds),
            response=response,
            positive_prior=np.ones(10, dtype=bool),
            penalty_gradient=np.zeros(3),
            penalty_size=np.full(3, 1e-2),
            penalty_apply=lambda v: penalty @ np.asarray(v),
            penalty_size_apply=lambda v: np.abs(penalty) @ np.abs(np.asarray(v)),
            bar=MODE_CERTIFICATION_BAR,
            underflow=0.0,
            penalty_rounding=rounding,
        )
        return value

    assert ratio(None) > 1.0
    assert ratio(2**40) == 0.0
