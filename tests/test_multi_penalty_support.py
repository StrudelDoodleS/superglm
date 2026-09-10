"""Analytic finite-weight geometry and adversarial reference-root checks."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from superglm.reml.multi_penalty import (
    logdet_s_gradient,
    logdet_s_hessian,
    similarity_transform_logdet,
)


def _check_derivatives(result, components, weights, gradient, hessian):
    actual_gradient = logdet_s_gradient(result, components, weights)
    actual_hessian = logdet_s_hessian(result, components, weights)
    eps = np.finfo(float).eps
    np.testing.assert_allclose(actual_gradient, gradient, rtol=0, atol=32 * eps * result.rank)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=0, atol=64 * eps * result.rank)
    certificate = result._certificate
    assert certificate is not None
    assert np.all(np.abs(actual_gradient - gradient) <= certificate.gradient_error + 8 * eps)
    assert np.all(np.abs(actual_hessian - hessian) <= certificate.hessian_error + 8 * eps)


def test_positive_weight_ratios_do_not_change_rank_or_derivatives():
    components = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    weights = np.array([1e12, 3.0])
    result = similarity_transform_logdet(components, weights)
    assert result.rank == 2
    assert abs(result.logdet_s_plus - np.log(3e12)) <= 32 * np.finfo(float).eps * 30
    _check_derivatives(result, components, weights, np.ones(2), np.zeros((2, 2)))
    np.testing.assert_allclose(result.E_sqrt.T @ result.E_sqrt, np.diag(weights), rtol=8e-15)
    np.testing.assert_allclose(result.S_pinv_plus * weights[None, :], np.eye(2), atol=8e-15)


def test_three_scales_and_common_null_space():
    components = [np.diag(np.eye(5)[i]) for i in range(3)]
    weights = np.array([1e300, 1.0, 1e-300])
    result = similarity_transform_logdet(components, weights)
    assert result.rank == 3
    assert result.Q_zero.shape == (5, 2)
    assert abs(result.logdet_s_plus) < 64 * np.finfo(float).eps * np.log(1e300)
    _check_derivatives(result, components, weights, np.ones(3), np.zeros((3, 3)))
    np.testing.assert_allclose(np.diag(result.S_pinv_plus)[:3] * weights, np.ones(3), rtol=1e-13)


def test_coordinate_volume_correction_and_overlapping_derivatives():
    vector = np.array([1.0, 2.0, 3.0, 0.0])
    components = [np.outer(vector, vector), 4.0 * np.outer(vector, vector)]
    weights = np.array([2.0, 3.0])
    result = similarity_transform_logdet(components, weights)
    assert result.rank == 1
    assert abs(result.logdet_s_plus - np.log(196.0)) < 64 * np.finfo(float).eps
    _check_derivatives(
        result,
        components,
        weights,
        np.array([1 / 7, 6 / 7]),
        6 / 49 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )


def test_ratio_51_deflation_is_corrected_against_reference_roots():
    vector = np.array([1.0, 1e-14])
    components = [np.diag([1.0, 0.0]), np.outer(vector, vector), np.diag([0.0, 1.0])]
    weights = np.array([1e30, 1e30, 1.0])
    result = similarity_transform_logdet(components, weights)
    with localcontext() as context:
        context.prec = 80
        expected = float((Decimal(102) * Decimal("1e30")).ln())
    assert result.rank == 2
    assert abs(result.logdet_s_plus - expected) <= 32 * np.finfo(float).eps * abs(expected)
    _check_derivatives(
        result,
        components,
        weights,
        np.array([101.0, 101.0, 2.0]) / 102,
        np.array([[101.0, -1.0, -100.0], [-1.0, 101.0, -100.0], [-100.0, -100.0, 200.0]]) / 10404,
    )


def test_reference_correction_repairs_a_full_rank_wrong_candidate():
    from superglm.reml.multi_penalty import _reference_correct_once

    roots = (np.array([[1.0, 0.0]]), np.array([[1.0, 1e-14]]), np.array([[0.0, 1.0]]))
    weights = np.array([1e30, 1e30, 1.0])
    E = np.diag([np.sqrt(2e30), 1.0])
    J = np.diag([1 / np.sqrt(2e30), 1.0])
    ell = np.log(2e30)
    updated_E, updated_J, updated_ell, factors = _reference_correct_once(roots, weights, E, J, ell)
    assert abs(updated_ell - ell - np.log(51.0)) < 64 * np.finfo(float).eps
    np.testing.assert_allclose(updated_E @ updated_J, np.eye(2), atol=32 * np.finfo(float).eps)
    np.testing.assert_allclose(
        sum(W.T @ W for W in factors), np.eye(2), atol=32 * np.finfo(float).eps
    )


@pytest.mark.parametrize("weights", [[-1.0, 1.0], [np.inf, 1.0], [np.nan, 1.0]])
def test_invalid_weights_are_refused(weights):
    with pytest.raises(ValueError, match="finite and non-negative"):
        similarity_transform_logdet([np.eye(2), np.eye(2)], np.array(weights))


def test_zero_weights_are_inactive():
    components = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    weights = np.array([2.0, 0.0])
    result = similarity_transform_logdet(components, weights)
    assert result.rank == 1
    _check_derivatives(result, components, weights, np.array([1.0, 0.0]), np.zeros((2, 2)))


def test_unrepresentable_required_inverse_is_refused():
    from superglm.reml.multi_penalty import PenaltyNumericalError

    with pytest.raises(PenaltyNumericalError, match="inverse"):
        similarity_transform_logdet([np.array([[1e-310]])], np.ones(1))


@pytest.mark.parametrize("multiple", [1, 3, 5])
def test_symmetric_subnormal_component_keeps_its_weighted_geometry(multiple):
    penalty = multiple * np.nextafter(0.0, 1.0)
    weight = 1e308
    result = similarity_transform_logdet([np.array([[penalty]])], np.array([weight]))
    with localcontext() as context:
        context.prec = 100
        exact_sum = Decimal.from_float(penalty) * Decimal.from_float(weight)
        exact_log = exact_sum.ln()
        # The kernel certificate concerns its frozen component root; the
        # original scalar's square-root rounding is extraction evidence.
        frozen_root = Decimal.from_float(result._support.component_roots[0][0, 0])
        exact_inverse = 1 / (Decimal.from_float(weight) * frozen_root**2)
        log_error = abs(Decimal.from_float(result.logdet_s_plus) - exact_log)
        inverse_error = abs(Decimal.from_float(result.S_pinv_plus[0, 0]) - exact_inverse)
    assert result.rank == 1
    assert log_error <= Decimal.from_float(result._certificate.logdet_error)
    assert inverse_error <= Decimal.from_float(result._certificate.inverse_error[0, 0])
    _check_derivatives(
        result, [np.array([[penalty]])], np.array([weight]), np.ones(1), np.zeros((1, 1))
    )


def test_subnormal_dense_inverse_survives_output_symmetrization():
    penalty, weight = 1e308, 2e15
    result = similarity_transform_logdet([np.array([[penalty]])], np.array([weight]))
    with localcontext() as context:
        context.prec = 100
        expected = 1 / (Decimal.from_float(penalty) * Decimal.from_float(weight))
    assert result.rank == 1
    assert result.S_pinv_plus[0, 0] == float(expected) > 0
    assert abs(Decimal.from_float(result.S_pinv_plus[0, 0]) - expected) <= Decimal.from_float(
        result._certificate.inverse_error[0, 0]
    )


def test_active_support_preserves_extraction_and_projection_evidence():
    from dataclasses import replace

    from superglm.reml.multi_penalty import _evaluate_penalty_support
    from superglm.reml.penalty_support import _penalty_support

    components = [np.ones((2, 2)), np.diag([1.0, 0.0])]
    support = _penalty_support(components)
    # A conservative upstream projection ledger must survive activation changes.
    projection = tuple(np.full_like(root, np.finfo(float).eps) for root in support.component_roots)
    support = replace(support, support_projection_bounds=projection)
    result = _evaluate_penalty_support(support, np.array([1.0, 0.0]))
    assert np.any(support.component_reconstruction_bounds[0])
    np.testing.assert_array_equal(
        result._support.component_reconstruction_bounds[0],
        support.component_reconstruction_bounds[0],
    )
    assert np.all(result._support.support_projection_bounds[0] >= projection[0])


def test_root_scaling_underflow_is_propagated_at_the_actual_working_dtype(monkeypatch):
    from superglm.reml import multi_penalty as module

    monkeypatch.setattr(module, "_LD", np.float64)
    monkeypatch.setattr(module, "_U_LD", np.finfo(float).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(0.0, 1.0))
    root, weight, inverse = 1e-170, 1e-308, 1e154
    actions, bounds = module._reference_root_actions(
        (np.array([[root]]),), np.array([weight]), np.array([[inverse]])
    )
    with localcontext() as context:
        context.prec = 100
        exact = (
            Decimal.from_float(weight).sqrt()
            * Decimal.from_float(root)
            * Decimal.from_float(inverse)
        )
        error = abs(Decimal.from_float(actions[0][0, 0]) - exact)
    assert error <= Decimal.from_float(bounds[0][0, 0])


def test_three_overlapping_components_have_analytic_cross_terms():
    components = [np.diag([1.0, 1.0, 0.0]), np.diag([0.0, 1.0, 1.0]), np.diag([1.0, 0.0, 1.0])]
    weights = np.array([2.0, 3.0, 5.0])
    result = similarity_transform_logdet(components, weights)
    contributions = weights[:, None] * np.stack([np.diag(p) for p in components])
    fractions = contributions / contributions.sum(axis=0)
    gradient = fractions.sum(axis=1)
    hessian = np.diag(gradient) - fractions @ fractions.T
    _check_derivatives(result, components, weights, gradient, hessian)
    assert abs(result.logdet_s_plus - np.log(7 * 5 * 8)) <= result._certificate.logdet_error


def test_separation_parameter_cannot_drop_a_positive_weight_direction():
    components = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    weights = np.array([1e24, 1e-12])
    for separation in (1e-2, 1e-5, 1e-10):
        result = similarity_transform_logdet(components, weights, eps_rank=separation)
        assert result.rank == 2
        _check_derivatives(result, components, weights, np.ones(2), np.zeros((2, 2)))


def test_large_action_uncertainty_refuses_even_when_measured_whitening_is_good(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._reference_root_actions

    def uncertain(*args, **kwargs):
        factors, _ = original(*args, **kwargs)
        return factors, tuple(np.full_like(factor, 1e-3) for factor in factors)

    monkeypatch.setattr(module, "_reference_root_actions", uncertain)
    with pytest.raises(module.PenaltyNumericalError, match="accuracy contract"):
        module.similarity_transform_logdet([np.eye(2)], np.ones(1))


@pytest.mark.parametrize("mutation", ["root", "derivative_factor"])
def test_reference_certificate_rejects_inconsistent_geometry(monkeypatch, mutation):
    from superglm.reml import multi_penalty as module

    original = module._reference_correct_once

    def corrupted(*args, **kwargs):
        E, J, ell, factors = original(*args, **kwargs)
        if mutation == "root":
            E = E * 1.01
        else:
            factors = (factors[0] * 1.01, *factors[1:])
        return E, J, ell, factors

    monkeypatch.setattr(module, "_reference_correct_once", corrupted)
    with pytest.raises(module.PenaltyNumericalError):
        module.similarity_transform_logdet([np.eye(2)], np.ones(1))


def test_omitting_reference_update_is_detected_for_ratio_51(monkeypatch):
    from superglm.reml import multi_penalty as module

    vector = np.array([1.0, 1e-14])
    components = [np.diag([1.0, 0.0]), np.outer(vector, vector), np.diag([0.0, 1.0])]
    weights = np.array([1e30, 1e30, 1.0])

    def candidate(*_):
        return (
            np.diag([np.sqrt(2e30), 1.0]),
            np.diag([1 / np.sqrt(2e30), 1.0]),
            np.log(2e30),
            0.0,
            100.0,
        )

    def no_correction(roots, values, E, J, ell, *, _evidence, _refine=True):
        _evidence.append((np.zeros_like(E), 0.0, 0.0))
        return E, J, ell, module._reference_root_actions(roots, values, J)[0]

    monkeypatch.setattr(module, "_separated_candidate", candidate)
    monkeypatch.setattr(module, "_reference_correct_once", no_correction)
    with pytest.raises(module.PenaltyNumericalError, match="accuracy contract"):
        module.similarity_transform_logdet(components, weights)


def test_derivatives_refuse_stale_components_and_weights():
    components = [np.eye(2), np.diag([1.0, 2.0])]
    weights = np.ones(2)
    result = similarity_transform_logdet(components, weights)
    with pytest.raises(ValueError, match="weights do not match"):
        logdet_s_gradient(result, components, weights * 2)
    components[0][0, 0] = 2.0
    with pytest.raises(ValueError, match="components do not match"):
        logdet_s_hessian(result, components, weights)


def test_all_inactive_components_have_exact_zero_geometry():
    components = [np.eye(3), np.diag([1.0, 0.0, 2.0])]
    result = similarity_transform_logdet(components, np.zeros(2))
    assert result.rank == 0
    assert result.logdet_s_plus == 0
    assert not np.any(result.E_sqrt)
    assert not np.any(result.S_pinv_plus)
    _check_derivatives(result, components, np.zeros(2), np.zeros(2), np.zeros((2, 2)))


@pytest.mark.parametrize("without_fma", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_compensated_dot_encloses_a_cancelling_exact_rational_dot(
    monkeypatch, without_fma, compiled
):
    import math

    from superglm.reml import multi_penalty as module
    from superglm.reml.multi_penalty import _compensated_dot

    if not compiled:
        monkeypatch.setattr(module, "_dot2_value", lambda *_: (0.0, False), raising=False)
    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)
    left = np.array([1e6, 1e6 + 1, 1e-6, -3.0])
    right = np.array([1.0, -1.0, 3.0, 1e-6], dtype=np.longdouble)
    right[0] += np.longdouble(2) ** -60
    value, error = _compensated_dot(left, right)
    exact = sum(
        Fraction.from_float(float(x)) * Fraction(*y.as_integer_ratio())
        for x, y in zip(left, right, strict=True)
    )
    assert abs(Fraction.from_float(value) - exact) <= Fraction.from_float(error)
    assert error < 8 * np.finfo(float).eps


def test_compensated_dot_uses_the_compiled_scalar_recurrence(monkeypatch):
    from superglm.reml import multi_penalty as module
    from superglm.reml._compensated import _dot2_value

    calls = []

    def tracked(left, right):
        value, success = _dot2_value(left, right)
        calls.append((len(left), success))
        return value, success

    monkeypatch.setattr(module, "_dot2_value", tracked, raising=False)
    left = np.array([1e6, 1e6 + 1, 1e-6, -3.0])
    right = np.array([1.0, -1.0, 3.0, 1e-6], dtype=np.longdouble)
    value, error = module._compensated_dot(left, right)
    exact = sum(
        Fraction.from_float(float(x)) * Fraction(*y.as_integer_ratio())
        for x, y in zip(left, right, strict=True)
    )
    assert calls == [(8, True)]
    assert abs(Fraction.from_float(value) - exact) <= Fraction.from_float(error)


def test_compensated_dot_refuses_unrepresentable_intermediate_magnitudes():
    from superglm.reml.multi_penalty import PenaltyNumericalError, _compensated_dot

    with pytest.raises(PenaltyNumericalError, match="arithmetic error bound"):
        _compensated_dot(np.array([1e308, -1e308]), np.array([2.0, 2.0], dtype=np.longdouble))


def test_cancelling_gram_inner_product_uses_root_contraction(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._cross_value
    calls = []

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_cross_value", tracked)
    components = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0]), np.diag([0.0, 1.0])]
    result = module.similarity_transform_logdet(components, np.ones(3))
    assert calls
    _check_derivatives(
        result,
        components,
        np.ones(3),
        np.array([1.0, 0.5, 0.5]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.25, -0.25], [0.0, -0.25, 0.25]]),
    )


@pytest.mark.parametrize("inactive", [False, True])
def test_independent_frozen_rows_have_exact_affine_log_weight_derivatives(inactive):
    from superglm.reml.multi_penalty import _evaluate_penalty_support
    from superglm.reml.penalty_support import _penalty_support_from_roots

    # Frozen 1+5 component roots from the ordered-category select fit. Their
    # stacked unweighted matrix is well conditioned; weighting causes the
    # cancellation that made the generic derivative enclosure refuse.
    roots = [
        np.array(
            [
                [
                    2.5760699381293324,
                    0.15096686598725334,
                    0.835702708932025,
                    0.13771294273144452,
                    -0.039838480774300876,
                    0.00024398701292805788,
                ]
            ]
        ),
        np.array(
            [
                [
                    -2.1323665117400314e-17,
                    -0.8840818180145362,
                    -1.123568504788146,
                    1.2043316440424927,
                    -1.1816163803089845,
                    0.5216713112738603,
                ],
                [
                    4.474243542928916e-17,
                    0.2201898740773631,
                    -1.535744009414525,
                    -1.5437246744691284,
                    -1.129070160466224,
                    -1.4788443867273622,
                ],
                [
                    1.329817110422435e-17,
                    -0.4334419505781178,
                    -1.185947912116825,
                    -0.8078265960960161,
                    2.501547491151326,
                    0.3516766908453225,
                ],
                [
                    -3.7649238162753865e-18,
                    0.8827125330559674,
                    -1.3688386288956256,
                    1.953170658416632,
                    0.5400154482362695,
                    0.07917753792307254,
                ],
                [
                    1.4027077878102064e-15,
                    0.3241226984270676,
                    -0.4014181648250418,
                    -0.9835306293159186,
                    -0.7193447037200487,
                    2.7040469505080575,
                ],
            ]
        ),
    ]
    weights = [1e10, 0.0004952032033797616]
    if inactive:
        roots.append(np.eye(6)[:2])
        weights.append(0.0)
    support = _penalty_support_from_roots(
        roots,
        resolution_limited=[False] * len(roots),
        input_error_bounds=[np.zeros_like(root) for root in roots],
    )
    result = _evaluate_penalty_support(support, np.array(weights))
    expected_gradient = np.array([1.0, 5.0, 0.0] if inactive else [1.0, 5.0])
    np.testing.assert_array_equal(result._gradient, expected_gradient)
    assert not np.any(result._hessian)
    assert not np.any(result._certificate.gradient_error)
    assert not np.any(result._certificate.hessian_error)
    assert result.rank == 6
    assert result._correction_count >= 1
    assert result._certificate.whitening_error < 1
    assert result._certificate.duality_error < 1
    with localcontext() as context:
        context.prec = 100
        matrix = [[Decimal.from_float(value) for value in row] for row in np.vstack(roots[:2])]
        unweighted_logdet = Decimal(0)
        for column in range(6):
            pivot = max(range(column, 6), key=lambda row: abs(matrix[row][column]))
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            diagonal = matrix[column][column]
            unweighted_logdet += abs(diagonal).ln()
            for row in range(column + 1, 6):
                multiplier = matrix[row][column] / diagonal
                for index in range(column + 1, 6):
                    matrix[row][index] -= multiplier * matrix[column][index]
        expected_logdet = (
            2 * unweighted_logdet
            + Decimal.from_float(weights[0]).ln()
            + 5 * Decimal.from_float(weights[1]).ln()
        )
        log_error = abs(Decimal.from_float(result.logdet_s_plus) - expected_logdet)
    assert log_error <= Decimal.from_float(result._certificate.logdet_error)


def test_redundant_component_rows_retain_generic_derivatives(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._component_gram
    calls = []

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_component_gram", tracked)
    components = [np.eye(6), np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    weights = np.array([3.0, 5.0])
    result = module.similarity_transform_logdet(components, weights)
    assert result.rank == 6
    assert len(calls) == 2
    _check_derivatives(
        result,
        components,
        weights,
        np.array([5 + 3 / 8, 5 / 8]),
        15 / 64 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )


def test_ordinary_cross_terms_use_the_certified_gram_contraction(monkeypatch):
    from superglm.reml import multi_penalty as module

    calls = []
    original = module._gram_cross

    def tracked(*args):
        result = original(*args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(module, "_gram_cross", tracked)
    result = module.similarity_transform_logdet([np.eye(4), 2 * np.eye(4)], np.ones(2))
    assert calls and all(calls)
    _check_derivatives(
        result,
        [np.eye(4), 2 * np.eye(4)],
        np.ones(2),
        np.array([4 / 3, 8 / 3]),
        8 / 9 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )


def test_coordinate_volume_oracle_detects_omitted_map_term(monkeypatch):
    from superglm.reml import multi_penalty as module

    vector = np.array([1.0, 2.0, 3.0, 0.0])
    components = [np.outer(vector, vector), 4 * np.outer(vector, vector)]
    weights = np.array([2.0, 3.0])
    monkeypatch.setattr(module, "_direct_candidate", lambda *_: None)
    original = module._separated_candidate
    baseline = module.similarity_transform_logdet(components, weights)
    np.testing.assert_allclose(
        baseline.logdet_s_plus, np.log(196.0), rtol=0, atol=64 * np.finfo(float).eps
    )

    def without_map(support, values, separation):
        E, J, ell, error, volume = original(support, values, separation)
        ell -= 2 * np.log(np.abs(np.diag(support.coordinate_triangular))).sum()
        return E, J, ell, error, volume

    monkeypatch.setattr(module, "_separated_candidate", without_map)
    mutated = module.similarity_transform_logdet(components, weights)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            mutated.logdet_s_plus, np.log(196.0), rtol=0, atol=64 * np.finfo(float).eps
        )


def test_ratio_51_oracle_detects_checking_deflated_reference_roots(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._reference_root_actions
    vector = np.array([1.0, 1e-14])
    components = [np.diag([1.0, 0.0]), np.outer(vector, vector), np.diag([0.0, 1.0])]

    def deflated(roots, weights, inverse, **kwargs):
        changed = [root.copy() for root in roots]
        changed[1][:, 1] = 0.0
        return original(changed, weights, inverse, **kwargs)

    monkeypatch.setattr(module, "_reference_root_actions", deflated)
    mutated = module.similarity_transform_logdet(components, np.array([1e30, 1e30, 1.0]))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            mutated.logdet_s_plus, np.log(102e30), rtol=0, atol=64 * np.finfo(float).eps * 74
        )


def test_normal_inverse_uses_native_products_with_a_retained_operand_enclosure(monkeypatch):
    from superglm.reml import multi_penalty as module

    rng = np.random.default_rng(301)
    inverse_root = rng.normal(size=(7, 5)).astype(np.longdouble)
    inverse_root += np.ldexp(np.longdouble(1), -60)

    def unexpected_wide_product(*_):
        pytest.fail("normal inverse materialization used a wide matrix product")

    monkeypatch.setattr(module, "_matmul_enclosed", unexpected_wide_product)
    inverse, error = module._inverse_gram_enclosed(inverse_root, 0.0)
    for row, column in np.ndindex(inverse.shape):
        exact = sum(
            (
                Fraction(*left.as_integer_ratio()) * Fraction(*right.as_integer_ratio())
                for left, right in zip(inverse_root[row], inverse_root[column], strict=True)
            ),
            Fraction(0),
        )
        observed = Fraction.from_float(inverse[row, column])
        assert abs(observed - exact) <= Fraction.from_float(error[row, column])


def test_native_inverse_encloses_signed_cancellation_and_metric_uncertainty():
    from superglm.reml.multi_penalty import _inverse_gram_enclosed

    inverse_root = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, 2.0**-50], [2.0**40, 1.0, -(2.0**40)]],
        dtype=np.longdouble,
    )
    eta = 1 / 16
    inverse, error = _inverse_gram_enclosed(inverse_root, eta)
    for row, column in np.ndindex(inverse.shape):
        gram = sum(
            (
                Fraction(*left.as_integer_ratio()) * Fraction(*right.as_integer_ratio())
                for left, right in zip(inverse_root[row], inverse_root[column], strict=True)
            ),
            Fraction(0),
        )
        for metric in (Fraction(1), 1 - Fraction.from_float(eta)):
            assert abs(Fraction.from_float(inverse[row, column]) - gram / metric) <= (
                Fraction.from_float(error[row, column])
            )


@pytest.mark.parametrize("first_entry", [np.longdouble(1), np.longdouble("1e-160")])
def test_inverse_metric_enclosure_covers_cross_row_whitening_defects(first_entry):
    from superglm.reml.multi_penalty import _inverse_gram_enclosed

    inverse_root = np.diag(np.array([first_entry, 1], dtype=np.longdouble))
    eta = Fraction(1, 16)
    off_diagonal = Fraction(1, 32)
    assert 2 * off_diagonal**2 < eta**2
    # G = [[1, 1/32], [1/32, 1]] has ||G-I||_F < eta. A dense
    # G^-1-I can couple disjoint rows of J despite |J| @ |J.T| being diagonal.
    denominator = 1 - off_diagonal**2
    first = Fraction(*first_entry.as_integer_ratio())
    exact = [
        [first**2 / denominator, -first * off_diagonal / denominator],
        [-first * off_diagonal / denominator, 1 / denominator],
    ]
    inverse, error = _inverse_gram_enclosed(inverse_root, float(eta))
    for row, column in np.ndindex(inverse.shape):
        assert abs(Fraction.from_float(inverse[row, column]) - exact[row][column]) <= (
            Fraction.from_float(error[row, column])
        )


def test_frozen_near_orthogonal_root_inverse_has_an_entrywise_enclosure():
    from superglm.reml.multi_penalty import _evaluate_penalty_support
    from superglm.reml.penalty_support import _penalty_support_from_roots

    root = np.array(
        [
            [float.fromhex("-0x1.0a5e252d288eep-3"), float.fromhex("0x1.fba6a661622eap-1")],
            [float.fromhex("-0x1.fba6a661622eap-1"), float.fromhex("-0x1.0a5e252d288edp-3")],
        ]
    )
    support = _penalty_support_from_roots(
        [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
    )
    result = _evaluate_penalty_support(support, np.ones(1))
    frozen = [[Fraction.from_float(value) for value in row] for row in support.component_roots[0]]
    a = frozen[0][0] ** 2 + frozen[1][0] ** 2
    b = frozen[0][0] * frozen[0][1] + frozen[1][0] * frozen[1][1]
    c = frozen[0][1] ** 2 + frozen[1][1] ** 2
    determinant = a * c - b**2
    expected = [[c / determinant, -b / determinant], [-b / determinant, a / determinant]]
    for row, column in np.ndindex((2, 2)):
        actual = Fraction.from_float(result.S_pinv_plus[row, column])
        bound = Fraction.from_float(result._certificate.inverse_error[row, column])
        assert abs(actual - expected[row][column]) <= bound


@pytest.mark.parametrize("value", [np.sqrt(np.longdouble(5e-324)), np.longdouble(1e154)])
def test_inverse_exponent_boundaries_keep_the_wide_fallback(monkeypatch, value):
    from superglm.reml import multi_penalty as module

    calls = []
    original = module._matmul_enclosed

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_matmul_enclosed", tracked)
    inverse, error = module._inverse_gram_enclosed(np.array([[value]]), 0.0)
    assert calls == [1]
    assert inverse[0, 0] != 0
    exact = Fraction(*value.as_integer_ratio()) ** 2
    assert abs(Fraction.from_float(inverse[0, 0]) - exact) <= Fraction.from_float(error[0, 0])


def test_dense_inverse_native_dispatch_preserves_preceding_admission(monkeypatch):
    from superglm.reml import multi_penalty as module

    components = [np.eye(4), 2 * np.eye(4)]
    weights = np.array([3.0, 5.0])
    baseline = module.similarity_transform_logdet(components, weights)
    original = module._matmul_enclosed

    def forbid_wide_inverse(left, right, **kwargs):
        if (
            left.dtype == np.dtype(np.longdouble)
            and right.dtype == np.dtype(np.longdouble)
            and np.shares_memory(left, right)
            and np.array_equal(right, left.T)
        ):
            pytest.fail("dense inverse used the wide matrix product")
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "_matmul_enclosed", forbid_wide_inverse)
    result = module.similarity_transform_logdet(components, weights)
    assert result.logdet_s_plus == baseline.logdet_s_plus
    assert result._correction_count == baseline._correction_count
    np.testing.assert_array_equal(result._gradient, baseline._gradient)
    np.testing.assert_array_equal(result._hessian, baseline._hessian)
    assert result._certificate.logdet_error == baseline._certificate.logdet_error


def test_aggregate_admission_precedes_scalar_compensation(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._reference_root_actions
    modes = []

    def tracked(*args, **kwargs):
        modes.append(kwargs.get("_refine", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_reference_root_actions", tracked)
    result = module.similarity_transform_logdet([np.eye(2), np.eye(2)], np.ones(2))
    assert result._correction_count == 1
    assert modes == [False, False]
    _check_derivatives(
        result,
        [np.eye(2), np.eye(2)],
        np.ones(2),
        np.ones(2),
        np.array([[0.5, -0.5], [-0.5, 0.5]]),
    )


@pytest.mark.parametrize("wide_bound", [1e-3, 10.0])
def test_refined_certificate_reuses_geometry_and_charges_its_evidence_once(monkeypatch, wide_bound):
    from superglm.reml import multi_penalty as module

    components = [np.eye(2), np.eye(2)]
    weights = np.ones(2)
    baseline = module.similarity_transform_logdet(components, weights)
    original_actions = module._reference_root_actions
    original_correction = module._reference_correct_once
    modes, corrections, inverse_roots = [], [], []

    def uncertain_wide(*args, **kwargs):
        mode = kwargs.get("_refine", True)
        modes.append(mode)
        inverse_roots.append(args[2].copy())
        factors, bounds = original_actions(*args, **kwargs)
        if not mode:
            bounds = tuple(np.full_like(factor, wide_bound) for factor in factors)
        return factors, bounds

    def counted(*args, **kwargs):
        corrections.append(kwargs.get("_refine", True))
        return original_correction(*args, **kwargs)

    monkeypatch.setattr(module, "_reference_root_actions", uncertain_wide)
    monkeypatch.setattr(module, "_reference_correct_once", counted)
    result = module.similarity_transform_logdet(components, weights)
    assert corrections == [False]
    assert modes == [False, False, True]
    np.testing.assert_array_equal(inverse_roots[1], inverse_roots[2])
    assert result._correction_count == 1
    assert result._certificate.logdet_error == baseline._certificate.logdet_error
    assert result.logdet_s_plus == baseline.logdet_s_plus


def test_wide_cache_disagreement_uses_the_existing_second_refined_correction(monkeypatch):
    from superglm.reml import multi_penalty as module

    original = module._reference_correct_once
    modes = []

    def uncertain_cache(*args, **kwargs):
        mode = kwargs.get("_refine", True)
        modes.append(mode)
        E, J, ell, cached = original(*args, **kwargs)
        if not mode:
            cached = (cached[0] * 1.01, *cached[1:])
        return E, J, ell, cached

    monkeypatch.setattr(module, "_reference_correct_once", uncertain_cache)
    result = module.similarity_transform_logdet([np.eye(2), np.eye(2)], np.ones(2))
    assert modes == [False, True]
    assert result._correction_count == 2
    _check_derivatives(
        result,
        [np.eye(2), np.eye(2)],
        np.ones(2),
        np.ones(2),
        np.array([[0.5, -0.5], [-0.5, 0.5]]),
    )


def _assert_positive_product_enclosed(left, right, bound):
    for row, column in np.ndindex(bound.shape):
        exact = sum(
            (
                Fraction(*a.as_integer_ratio()) * Fraction(*b.as_integer_ratio())
                for a, b in zip(left[row], right[:, column], strict=True)
            ),
            Fraction(0),
        )
        assert exact <= Fraction.from_float(bound[row, column])


@pytest.mark.parametrize("working_dtype", [np.float64, np.longdouble])
def test_positive_bound_uses_outward_native_operands_without_mutating_inputs(
    monkeypatch, working_dtype
):
    from superglm.reml import multi_penalty as module

    monkeypatch.setattr(module, "_LD", working_dtype)
    monkeypatch.setattr(module, "_U_LD", np.finfo(working_dtype).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(working_dtype(0), working_dtype(1)))
    step = working_dtype(2) ** -60
    left = np.array([[1 + step, 0], [3 - step, 2 + step]], dtype=working_dtype)
    right = np.array([[2 - step, 1 + step], [0, 4 - step]], dtype=working_dtype)
    left.flags.writeable = right.flags.writeable = False
    saved_left, saved_right = left.copy(), right.copy()
    original = module._positive_native_product
    calls = []

    def tracked(a, b):
        calls.append(1)
        assert a.dtype == b.dtype == np.dtype(float)
        assert np.all(a.astype(working_dtype) >= left)
        assert np.all(b.astype(working_dtype) >= right)
        assert np.all(a[left == 0] == 0)
        assert np.all(b[right == 0] == 0)
        return original(a, b)

    monkeypatch.setattr(module, "_positive_native_product", tracked)
    bound = module._positive_product(left, right)
    assert calls == [1]
    _assert_positive_product_enclosed(left, right, bound)
    np.testing.assert_array_equal(left, saved_left)
    np.testing.assert_array_equal(right, saved_right)


def test_native_positive_bound_encloses_exact_longdouble_products():
    from superglm.reml.multi_penalty import _positive_product

    rng = np.random.default_rng(921)
    left = np.abs(rng.normal(size=(9, 17))).astype(np.longdouble)
    right = np.abs(rng.normal(size=(17, 7))).astype(np.longdouble)
    left += np.longdouble(2) ** -60
    right += np.longdouble(2) ** -60
    _assert_positive_product_enclosed(left, right, _positive_product(left, right))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.longdouble("1e-400"), np.longdouble("1e300")),
        (np.longdouble(np.nextafter(0.0, 1.0)), np.longdouble("1e300")),
        (np.longdouble("1e-200"), np.longdouble("1e-200")),
        (np.longdouble("1e154"), np.longdouble("1e154")),
        (
            np.nextafter(np.longdouble(np.finfo(float).max), np.longdouble(np.inf)),
            np.longdouble("1e-308"),
        ),
    ],
)
def test_positive_bound_exponent_edges_stay_outside_the_native_range(monkeypatch, left, right):
    from superglm.reml import multi_penalty as module

    if not np.isfinite(left) or left == 0:
        pytest.skip("this source value requires a wider exponent range or significand")

    def forbidden(*_):
        pytest.fail("unsupported exponent range reached the native positive product")

    monkeypatch.setattr(module, "_positive_native_product", forbidden)
    a, b = np.array([[left]]), np.array([[right]])
    _assert_positive_product_enclosed(a, b, module._positive_product(a, b))


def test_positive_bound_zero_and_overflow_controls(monkeypatch):
    from superglm.reml import multi_penalty as module

    zero = module._positive_product(np.zeros((3, 2)), np.ones((2, 4)))
    np.testing.assert_array_equal(zero, np.zeros((3, 4)))

    def forbidden(*_):
        pytest.fail("an overflowing absolute sum reached the native positive product")

    monkeypatch.setattr(module, "_positive_native_product", forbidden)
    with pytest.raises(module.PenaltyNumericalError):
        module._positive_product(np.full((1, 2), 1e154), np.full((2, 1), 1e154))
    with np.errstate(invalid="ignore"), pytest.raises(module.PenaltyNumericalError):
        module._positive_product(np.zeros((1, 1)), np.full((1, 1), np.inf))
    with pytest.raises(ValueError):
        module._positive_product(np.zeros((1, 2)), np.zeros((3, 1)))


@pytest.mark.parametrize("working_dtype", [np.float64, np.longdouble])
def test_strictly_sub_minimum_positive_bound_uses_an_exact_fill(monkeypatch, working_dtype):
    from superglm.reml import multi_penalty as module

    monkeypatch.setattr(module, "_LD", working_dtype)
    monkeypatch.setattr(module, "_U_LD", np.finfo(working_dtype).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(working_dtype(0), working_dtype(1)))
    left = np.full((3, 16), np.ldexp(working_dtype(0.75), -550))[:, ::2]
    right = np.full((16, 4), np.ldexp(working_dtype(0.75), -527))[::2, :]
    left.flags.writeable = right.flags.writeable = False
    saved_left, saved_right = left.copy(), right.copy()

    def forbidden(*_):
        pytest.fail("a provably sub-minimum bound reached matrix-product arithmetic")

    monkeypatch.setattr(module, "_positive_native_product", forbidden)
    monkeypatch.setattr(module, "_upper", forbidden)
    bound = module._positive_product(left, right)
    np.testing.assert_array_equal(bound, np.full((3, 4), np.nextafter(0.0, 1.0)))
    _assert_positive_product_enclosed(left, right, bound)
    np.testing.assert_array_equal(left, saved_left)
    np.testing.assert_array_equal(right, saved_right)


def test_strictly_sub_minimum_positive_bound_uses_original_wide_exponents(monkeypatch):
    from superglm.reml import multi_penalty as module

    if np.finfo(np.longdouble).minexp >= np.finfo(float).minexp:
        pytest.skip("this source value requires a wider exponent range")
    left = np.full((2, 3), np.nextafter(np.longdouble(0), np.longdouble(1)))
    right = np.full((3, 2), np.ldexp(np.longdouble(0.75), 15000))

    def forbidden(*_):
        pytest.fail("a sub-minimum bound was lost through a float64 operand cast")

    monkeypatch.setattr(module, "_positive_native_product", forbidden)
    monkeypatch.setattr(module, "_upper", forbidden)
    bound = module._positive_product(left, right)
    np.testing.assert_array_equal(bound, np.full((2, 2), np.nextafter(0.0, 1.0)))
    _assert_positive_product_enclosed(left, right, bound)


def test_adjacent_sub_minimum_exponent_keeps_the_wide_bound(monkeypatch):
    from superglm.reml import multi_penalty as module

    left = np.full((1, 8), np.ldexp(np.longdouble(0.75), -550))
    right = np.full((8, 1), np.ldexp(np.longdouble(0.75), -526))
    calls = []
    original_upper = module._upper

    def tracked(value):
        calls.append(1)
        return original_upper(value)

    def forbidden(*_):
        pytest.fail("the subnormal product reached the normal native range")

    monkeypatch.setattr(module, "_upper", tracked)
    monkeypatch.setattr(module, "_positive_native_product", forbidden)
    bound = module._positive_product(left, right)
    assert calls
    assert bound[0, 0] > np.nextafter(0.0, 1.0)
    _assert_positive_product_enclosed(left, right, bound)


def test_positive_bound_empty_inner_dimension_is_exact_zero():
    from superglm.reml.multi_penalty import _positive_product

    np.testing.assert_array_equal(
        _positive_product(np.empty((3, 0)), np.empty((0, 4))), np.zeros((3, 4))
    )


@pytest.mark.parametrize("weights", [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
def test_summary_omits_only_unrequested_inverse_outputs(monkeypatch, weights):
    from superglm.reml import multi_penalty as module
    from superglm.reml.penalty_support import _penalty_support

    scale = np.ldexp(1.0, -1060)
    components = [np.diag([scale, 0.0]), np.diag([0.0, scale])]
    values = np.asarray(weights)
    rank = int(np.count_nonzero(values))
    if rank:
        with pytest.raises(module.PenaltyNumericalError, match="required dense penalty inverse"):
            module.similarity_transform_logdet(components, values)
    else:
        assert module.similarity_transform_logdet(components, values).rank == 0

    def forbidden(*_):
        pytest.fail("a determinant summary materialized an unused dense inverse")

    monkeypatch.setattr(module, "_inverse_gram_enclosed", forbidden)
    result = module._evaluate_penalty_summary(_penalty_support(components), values)
    assert isinstance(result, module._PenaltySummary)
    assert not hasattr(result, "S_pinv_plus")
    assert not hasattr(result, "E_sqrt")
    assert not hasattr(result._certificate, "inverse_error")
    assert result.rank == result._support.rank == rank
    np.testing.assert_array_equal(result.gradient, values)
    np.testing.assert_array_equal(result.hessian, np.zeros((2, 2)))
    assert not result.gradient.flags.writeable
    assert not result.hessian.flags.writeable
    assert not result._input_lambdas.flags.writeable
    for root, value in zip(result._support.component_roots, values, strict=True):
        assert len(root) == int(value > 0)
    with localcontext() as context:
        context.prec = 100
        exact = -1060 * rank * Decimal(2).ln()
        assert abs(Decimal(result.logdet_s_plus) - exact) <= Decimal(
            result._certificate.logdet_error
        )


def test_summary_keeps_generic_derivatives_and_the_full_admission_certificate():
    from superglm.reml import multi_penalty as module
    from superglm.reml.penalty_support import _penalty_support

    components = [np.eye(2), 4 * np.eye(2)]
    values = np.array([3.0, 4.0])
    support = _penalty_support(components)
    full = module._evaluate_penalty_support(support, values)
    result = module._evaluate_penalty_summary(support, values)
    assert result.rank == 2
    assert result.logdet_s_plus == full.logdet_s_plus
    assert result._correction_count == full._correction_count
    np.testing.assert_array_equal(result.gradient, full._gradient)
    np.testing.assert_array_equal(result.hessian, full._hessian)
    for name in (
        "whitening_error",
        "duality_error",
        "logdet_error",
        "gradient_error",
        "hessian_error",
        "resolution_limited",
    ):
        np.testing.assert_array_equal(
            getattr(result._certificate, name), getattr(full._certificate, name)
        )
    for index, exact in enumerate([Fraction(6, 19), Fraction(32, 19)]):
        assert abs(Fraction.from_float(result.gradient[index]) - exact) <= Fraction.from_float(
            result._certificate.gradient_error[index]
        )
    for row, column in np.ndindex(2, 2):
        exact = Fraction(96 if row == column else -96, 361)
        assert abs(Fraction.from_float(result.hessian[row, column]) - exact) <= Fraction.from_float(
            result._certificate.hessian_error[row, column]
        )


def test_summary_propagates_failure_before_admission(monkeypatch):
    from superglm.reml import multi_penalty as module
    from superglm.reml.penalty_support import _penalty_support

    original = module._derivative_values
    calls = []

    def uncertain(*args):
        gradient, hessian, gradient_error, hessian_error = original(*args)
        calls.append(1)
        return gradient, hessian, np.ones_like(gradient_error), hessian_error

    def forbidden(*_):
        pytest.fail("an unadmitted summary reached output materialization")

    monkeypatch.setattr(module, "_derivative_values", uncertain)
    monkeypatch.setattr(module, "_inverse_gram_enclosed", forbidden)
    with pytest.raises(module.PenaltyNumericalError, match="accuracy contract"):
        module._evaluate_penalty_summary(_penalty_support([np.eye(2)]), np.ones(1))
    assert len(calls) == 3


def test_materialization_reuses_the_existing_wide_product_and_magnitude(monkeypatch):
    from superglm.reml import multi_penalty as module

    left = np.array([[1.25, 0.5], [-0.25, 2.0]])
    right = np.array([[1.0, 0.125, -0.5], [0.25, 1.5, 0.75]])
    evidence = []
    product, _ = module._matmul_enclosed(left, right, _evidence=evidence)
    inverse = np.linalg.pinv(product)
    expected = module._materialization_logdet_bound(left, right, product, inverse)
    original = module._positive_product

    def no_duplicate_magnitude(a, b):
        if np.array_equal(a, np.abs(left)) and np.array_equal(b, np.abs(right)):
            pytest.fail("the materialization bound recomputed an existing product magnitude")
        return original(a, b)

    monkeypatch.setattr(module, "_positive_product", no_duplicate_magnitude)
    actual = module._materialization_logdet_bound(
        left, right, product, inverse, _product_evidence=evidence[0]
    )
    assert actual == expected


@pytest.mark.parametrize("mutation", ["left", "right", "output"])
@pytest.mark.parametrize("working_dtype", [np.float64, np.longdouble])
def test_materialization_evidence_preserves_operand_and_output_changes(
    monkeypatch, mutation, working_dtype
):
    from superglm.reml import multi_penalty as module

    monkeypatch.setattr(module, "_LD", working_dtype)
    monkeypatch.setattr(module, "_U_LD", np.finfo(working_dtype).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(working_dtype(0), working_dtype(1)))
    left = np.array([[1.25, 0.5], [-0.25, 2.0]])
    right = np.array([[1.0, 0.125, -0.5], [0.25, 1.5, 0.75]])
    evidence = []
    product, _ = module._matmul_enclosed(left, right, _evidence=evidence)
    inverse = np.linalg.pinv(product)
    original_bound = module._materialization_logdet_bound(left, right, product, inverse)
    {"left": left, "right": right, "output": product}[mutation][0, 0] += 1e-3
    expected = module._materialization_logdet_bound(left, right, product, inverse)
    original = module._positive_product
    calls = []

    def tracked(a, b):
        if np.array_equal(a, np.abs(left)) and np.array_equal(b, np.abs(right)):
            calls.append(1)
        return original(a, b)

    monkeypatch.setattr(module, "_positive_product", tracked)
    actual = module._materialization_logdet_bound(
        left, right, product, inverse, _product_evidence=evidence[0]
    )
    assert actual == expected
    assert actual > original_bound
    assert bool(calls) == (mutation != "output")


@pytest.mark.parametrize("mutation", [None, "root", "inverse_root"])
def test_corrected_dual_product_is_reused_only_for_unchanged_operands(monkeypatch, mutation):
    from superglm.reml import multi_penalty as module
    from superglm.reml.penalty_support import _penalty_support

    original_correction, original_product = module._reference_correct_once, module._matmul_enclosed
    returned = []
    repeated = []

    def corrected(*args, **kwargs):
        E, J, logdet, actions = original_correction(*args, **kwargs)
        if mutation == "root":
            E *= 1.01
        elif mutation == "inverse_root":
            J *= 1.01
        returned.append((E, J))
        return E, J, logdet, actions

    def tracked(left, right, **kwargs):
        if returned and left is returned[-1][0] and right is returned[-1][1]:
            repeated.append(1)
        return original_product(left, right, **kwargs)

    monkeypatch.setattr(module, "_reference_correct_once", corrected)
    monkeypatch.setattr(module, "_matmul_enclosed", tracked)
    support = _penalty_support([np.diag([1.0, 1.0, 0.0]), np.diag([4.0, 4.0, 0.0])])
    if mutation is None:
        result = module._evaluate_penalty_summary(support, np.array([3.0, 4.0]))
        assert result.rank == 2
        assert result._correction_count == 1
        assert repeated == []
    else:
        with pytest.raises(module.PenaltyNumericalError):
            module._evaluate_penalty_summary(support, np.array([3.0, 4.0]))
        assert repeated
