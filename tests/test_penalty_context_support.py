"""A fixed raw penalty support survives an invertible SSP coordinate change."""

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.reml import multi_penalty
from superglm.reml import penalty_algebra as algebra
from superglm.reml.penalty_support import PenaltyNumericalError


def _context(raw, coordinate_map):
    width = coordinate_map.shape[1]
    group = SimpleNamespace(name="shared", sl=slice(0, width), size=width)
    matrix = SimpleNamespace(
        R_inv=coordinate_map,
        omega=sum(raw),
        omega_components=[(chr(ord("a") + index), matrix) for index, matrix in enumerate(raw)],
    )
    components, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    return components, matrix, group


def _exact_geometry():
    raw = [np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 1.0, 0.0])]
    coordinate_map = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
    return raw, coordinate_map


def _decimal_inverse(matrix):
    width = len(matrix)
    augmented = [row + [Decimal(i == j) for j in range(width)] for i, row in enumerate(matrix)]
    for column in range(width):
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(width):
            if row != column:
                multiple = augmented[row][column]
                augmented[row] = [
                    value - multiple * other
                    for value, other in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[width:] for row in augmented]


def _decimal_fixed_root(geometry, component_index, coordinate_map):
    def cast(value):
        return Decimal.from_float(float(value))

    basis = [[cast(value) for value in row] for row in geometry.support.Q_plus]
    width, rank = len(basis), geometry.support.rank
    gram = [[sum(row[i] * row[j] for row in basis) for j in range(rank)] for i in range(rank)]
    inverse = _decimal_inverse(gram)
    projector = [
        [
            sum(basis[i][a] * inverse[a][b] * basis[j][b] for a in range(rank) for b in range(rank))
            for j in range(width)
        ]
        for i in range(width)
    ]
    return [
        [
            sum(
                cast(row[k]) * projector[k][a] * cast(coordinate_map[a, j])
                for k in range(width)
                for a in range(width)
            )
            for j in range(width)
        ]
        for row in geometry.support.component_roots[component_index]
    ]


def _decimal_determinant(matrix):
    work = [row.copy() for row in matrix]
    determinant = Decimal(1)
    for column in range(len(work)):
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, len(work)):
            multiplier = work[row][column] / pivot
            for j in range(column + 1, len(work)):
                work[row][j] -= multiplier * work[column][j]
    return determinant


@pytest.mark.parametrize("active", [0, 1])
def test_zero_activity_keeps_the_fixed_ssp_target_under_null_shear(active):
    rng = np.random.default_rng(817)
    basis = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    roots = [rng.normal(size=(3, 4)) @ basis[:, :4].T for _ in range(2)]
    raw = [root.T @ root for root in roots]
    shear = np.eye(5)
    shear[4, 0] = 1e5
    coordinate_map = basis @ shear @ basis.T
    components, _, _ = _context(raw, coordinate_map)
    weights = {
        component.name: 2.0 * (index == active) for index, component in enumerate(components)
    }
    result = algebra._compute_penalty_logdet_evaluation(weights, components)
    with localcontext() as context:
        context.prec = 100
        root = _decimal_fixed_root(algebra._context_geometry(components), active, coordinate_map)
        gram = [
            [sum(a * b for a, b in zip(left, right, strict=True)) for right in root]
            for left in root
        ]
        expected = _decimal_determinant(gram).ln() + len(root) * Decimal(2).ln()
        assert abs(Decimal.from_float(result.logdet) - expected) <= Decimal.from_float(
            float(result.logdet_error)
        )


def test_zero_activity_keeps_generic_gradient_and_mixed_hessian():
    roots = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    raw = [np.outer(root, root) for root in roots]
    _, coordinate_map = _exact_geometry()
    components, _, _ = _context(raw, coordinate_map)
    values = np.array([2.0, 3.0, 5.0, 0.0])
    weights = dict(zip((component.name for component in components), values, strict=True))
    result = algebra._compute_penalty_logdet_evaluation(weights, components)
    expected_gradient = np.array([16.0, 21.0, 25.0, 0.0]) / 31
    assert result.rank == 2
    assert abs(result.logdet - np.log(89 * 31)) <= result.logdet_error
    for i, left in enumerate(components):
        assert (
            abs(result.gradient[left.name] - expected_gradient[i])
            <= result.gradient_error[left.name]
        )
        for j, right in enumerate(components):
            second = expected_gradient[i] if i == j else values[i] * values[j] / 31
            expected = second - expected_gradient[i] * expected_gradient[j]
            key = (left.name, right.name)
            assert abs(result.hessian[key] - expected) <= result.hessian_error[key]


def test_zero_activity_cannot_ignore_perpendicular_root_uncertainty():
    raw = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    components, _, _ = _context(raw, np.eye(2))
    geometry = algebra._context_geometry(components)
    geometry.ssp_root_errors = (np.array([[0.0, 1e-5]]), np.zeros((1, 2)))
    geometry.ssp_refined = True
    # Both [1, 0] and [1, 1e-5] satisfy this root enclosure. Their rank-one
    # log determinants differ by log(1 + 1e-10), even though the uncertainty
    # times the selected inverse action [1, 0].T vanishes exactly.
    with pytest.raises(PenaltyNumericalError, match="active support volume"):
        algebra._compute_penalty_logdet_evaluation({"shared:a": 1.0, "shared:b": 0.0}, components)


def test_invertible_ssp_map_preserves_shared_raw_support():
    rng = np.random.default_rng(817)
    basis = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    roots = [rng.normal(size=(3, 4)) @ basis[:, :4].T for _ in range(2)]
    raw = [root.T @ root for root in roots]
    coordinate_map = basis @ np.diag([1.0, 1.0, 1.0, 1.0, 1e5]) @ basis.T
    original, _, _ = _context(raw, np.eye(5))
    transformed, _, _ = _context(raw, coordinate_map)

    # The map is invertible. Re-extracting each transformed Gram separately
    # rotated its null space enough to report rank 5 on the unfixed code.
    assert algebra.compute_total_penalty_rank(original) == 4
    assert algebra.compute_total_penalty_rank(transformed) == 4
    weights = {"shared:a": 2.0, "shared:b": 3.0}
    before = algebra._compute_penalty_logdet_evaluation(weights, original)
    after = algebra._compute_penalty_logdet_evaluation(weights, transformed)
    assert before.rank == after.rank == 4
    assert before.gradient == after.gradient
    assert before.hessian == after.hessian
    geometry = algebra._context_geometry(transformed)
    with localcontext() as context:
        context.prec = 80
        cast = Decimal.from_float
        q = [[cast(float(value)) for value in row] for row in geometry.support.Q_plus]
        gram = [[sum(row[i] * row[j] for row in q) for j in range(4)] for i in range(4)]
        inverse = _decimal_inverse(gram)
        projector = [
            [
                sum(q[i][a] * inverse[a][b] * q[j][b] for a in range(4) for b in range(4))
                for j in range(5)
            ]
            for i in range(5)
        ]
        for component, root, bound in zip(
            transformed,
            geometry.support.component_roots,
            geometry.matrix_error_bounds,
            strict=True,
        ):
            mapped = [
                [
                    sum(
                        cast(float(row[k])) * projector[k][a] * cast(float(coordinate_map[a, j]))
                        for k in range(5)
                        for a in range(5)
                    )
                    for j in range(5)
                ]
                for row in root
            ]
            for i, j in np.ndindex((5, 5)):
                expected = sum(row[i] * row[j] for row in mapped)
                assert abs(cast(float(component.omega_ssp[i, j])) - expected) <= cast(
                    float(bound[i, j])
                )


@pytest.mark.parametrize(
    ("weights", "rank", "determinant"),
    [
        ((2.0, 3.0), 2, 6 * 89),
        ((2.0, 0.0), 1, 2 * 5),
        ((0.0, 3.0), 1, 3 * 25),
        ((0.0, 0.0), 0, 1),
    ],
)
def test_ssp_volume_uses_the_exact_active_support(weights, rank, determinant):
    components, _, _ = _context(*_exact_geometry())
    lambdas = dict(zip((component.name for component in components), weights, strict=True))
    result = algebra._compute_penalty_logdet_evaluation(lambdas, components)
    assert result.rank == rank
    assert abs(result.logdet - np.log(determinant)) <= result.logdet_error
    assert list(result.gradient.values()) == [float(value > 0) for value in weights]
    assert max(map(abs, result.hessian.values())) == 0


def test_context_reuses_only_its_last_exact_weight_evaluation(monkeypatch):
    from superglm.reml import penalty_support

    calls = []
    constructions = []
    evaluate = multi_penalty._evaluate_penalty_summary
    construct = penalty_support._penalty_support

    def construct_once(matrices):
        constructions.append(tuple(matrix.shape for matrix in matrices))
        return construct(matrices)

    def counted(support, weights, *args, **kwargs):
        calls.append((support, tuple(weights)))
        return evaluate(support, weights, *args, **kwargs)

    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", counted)
    monkeypatch.setattr(penalty_support, "_penalty_support", construct_once)
    components, _, _ = _context(*_exact_geometry())
    weights = {"shared:a": 2.0, "shared:b": 3.0}
    algebra.compute_total_penalty_rank(components)
    algebra.compute_logdet_s_plus(weights, components)
    algebra.compute_logdet_s_derivatives(weights, components)
    assert len(calls) == 1
    changed = dict(weights, **{"shared:a": np.nextafter(2.0, np.inf)})
    algebra.compute_logdet_s_plus(changed, components)
    algebra.compute_logdet_s_plus(weights, components)
    assert len(calls) == 3
    assert all(call[0] is calls[0][0] for call in calls)
    assert constructions == [((3, 3), (3, 3))]


def test_context_volume_cache_tracks_zero_activity():
    components, _, _ = _context(*_exact_geometry())
    for values, determinant in [
        ((2.0, 3.0), 534),
        ((2.0, 0.0), 10),
        ((0.0, 3.0), 75),
        ((2.0, 3.0), 534),
    ]:
        weights = dict(zip(("shared:a", "shared:b"), values, strict=True))
        result = algebra._compute_penalty_logdet_evaluation(weights, components)
        assert abs(result.logdet - np.log(determinant)) <= result.logdet_error


def test_context_does_not_reuse_an_old_basis_or_replaced_component():
    raw, coordinate_map = _exact_geometry()
    components, matrix, group = _context(raw, coordinate_map)
    weights = {"shared:a": 2.0, "shared:b": 3.0}
    original = algebra._compute_penalty_logdet_evaluation(weights, components)
    coordinate_map *= 2
    raw[0][:] *= 3
    unchanged = algebra._compute_penalty_logdet_evaluation(weights, components)
    assert unchanged.logdet == original.logdet

    rebuilt, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    new = algebra._compute_penalty_logdet_evaluation(weights, rebuilt)
    expected = original.logdet + np.log(3) + 4 * np.log(2)
    assert abs(new.logdet - expected) <= original.logdet_error + new.logdet_error

    replacement = replace(components[0], omega_ssp=np.eye(3))
    replaced = algebra._compute_penalty_logdet_evaluation(weights, [replacement, components[1]])
    assert replaced.rank == 3
    with pytest.raises(ValueError):
        components[0].omega_ssp[0, 0] = 9.0
    with pytest.raises(ValueError):
        components[0].omega_ssp.setflags(write=True)


def test_ssp_volume_accounts_for_the_stored_basis_gram():
    components, _, _ = _context(*_exact_geometry())
    geometry = algebra._context_geometry(components)
    support = geometry.get_support()
    scaled = replace(support, Q_plus=support.Q_plus * np.array([1 + 2e-8, 1 - 1e-8]))
    volume, error = algebra._support_coordinate_volume(scaled, geometry.coordinate_map)
    assert abs(volume - np.log(89)) <= error


def test_a_noninjective_square_map_cannot_expose_the_raw_rank():
    raw, _ = _exact_geometry()
    coordinate_map = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(PenaltyNumericalError, match="SSP coordinate"):
        _context(raw, coordinate_map)


@pytest.mark.parametrize(("raw_exponent", "map_exponent"), [(-1060, 530), (1000, -500)])
@pytest.mark.parametrize("active", [True, False])
def test_raw_summary_does_not_require_an_unused_dense_inverse(raw_exponent, map_exponent, active):
    raw = [
        np.ldexp(np.diag([1.0, 0.0]), raw_exponent),
        np.ldexp(np.diag([0.0, 1.0]), raw_exponent),
    ]
    components, _, _ = _context(raw, np.ldexp(np.eye(2), map_exponent))
    result = algebra._compute_penalty_logdet_evaluation(
        {"shared:a": 1.0, "shared:b": float(active)}, components
    )
    assert result.rank == 1 + active
    assert abs(result.logdet) <= result.logdet_error
    assert list(result.gradient.values()) == [1.0, float(active)]
    assert max(map(abs, result.hessian.values())) == 0.0


def test_failed_weight_evaluation_does_not_replace_the_last_valid_state(monkeypatch):
    components, _, _ = _context(*_exact_geometry())
    weights = {"shared:a": 2.0, "shared:b": 3.0}
    original = algebra._compute_penalty_logdet_evaluation(weights, components)
    evaluate = multi_penalty._evaluate_penalty_summary

    def refuse(support, values, *args, **kwargs):
        if values[0] == 4:
            raise PenaltyNumericalError("injected current-state refusal")
        pytest.fail("a failed trial discarded the retained valid evaluation")

    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", refuse)
    with pytest.raises(PenaltyNumericalError, match="injected current-state"):
        algebra.compute_logdet_s_plus(dict(weights, **{"shared:a": 4.0}), components)
    assert algebra.compute_logdet_s_plus(weights, components) == original.logdet
    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", evaluate)
