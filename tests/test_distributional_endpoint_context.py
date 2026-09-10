"""Endpoint determinant transport keeps the selected local penalty targets."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.smoothing import endpoint_laml as endpoint
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.group_matrix import DenseGroupMatrix
from superglm.reml import penalty_algebra as algebra
from superglm.types import GroupSlice, PenaltyComponent
from tests.test_distributional_penalty_context import _copy_problem, _predictor
from tests.test_penalty_context_support import _decimal_determinant, _decimal_fixed_root


def _retained_face_problem(magnitude=1e8):
    source, coordinate_map, predictors = _copy_problem("shear", magnitude)
    selected = PenaltyComponent(
        "x:face",
        "x",
        0,
        slice(0, 1),
        np.eye(1),
        np.eye(1),
        rank=1.0,
        eigvals_omega=np.ones(1),
    )
    first = replace(predictors[0], penalties=(selected,))
    layout = build_stacked_layout((first, predictors[1]))
    face = build_penalty_face(layout, ("first:x#face",))
    return source, coordinate_map, layout, face


@pytest.mark.parametrize("magnitude", [1.0, 1e5, 1e8])
@pytest.mark.parametrize("active", [0, 1])
def test_endpoint_retains_the_fixed_context_target_and_its_bound(magnitude, active):
    source, coordinate_map, layout, face = _retained_face_problem(magnitude)
    finite = layout.penalties[1:]
    weights = {layout.penalty_names[0]: 0.0}
    weights.update({item.name: 2.0 * (index == active) for index, item in enumerate(finite)})
    result = endpoint._projected_finite_penalty_evaluation(
        layout=layout, lambdas=weights, face=face
    )
    geometry = algebra._context_geometry(source)
    with localcontext() as context:
        context.prec = 100
        root = _decimal_fixed_root(geometry, active, coordinate_map)
        gram = [
            [sum(a * b for a, b in zip(left, right, strict=True)) for right in root]
            for left in root
        ]
        exact = _decimal_determinant(gram).ln() + len(root) * Decimal(2).ln()
        assert result.rank == 3
        assert abs(Decimal.from_float(result.logdet) - exact) <= Decimal.from_float(
            result.logdet_error
        )
    for index, item in enumerate(finite):
        assert result.gradient[item.name] == 3.0 * (index == active)
        for other in finite:
            assert result.hessian[item.name, other.name] == 0.0


def test_endpoint_context_guard_mutation_reproduces_the_old_target_loss(monkeypatch):
    source, coordinate_map, layout, face = _retained_face_problem()
    weights = {item.name: 0.0 for item in layout.penalties}
    weights[layout.penalties[1].name] = 2.0
    # The ordinary context route must not reselect the retained SSP Gram.
    from superglm.reml import penalty_support

    def discarded_target(*_args, **_kwargs):
        raise AssertionError("retained endpoint target was re-extracted from an SSP Gram")

    monkeypatch.setattr(penalty_support, "_component_root", discarded_target)
    result = endpoint._projected_finite_penalty_evaluation(
        layout=layout, lambdas=weights, face=face
    )
    assert result.rank == 3
    # Removing just the family attachment restores the old local extraction.
    copied = tuple(replace(item) for item in layout.penalties)
    changed = replace(layout, penalties=copied)
    with pytest.raises(AssertionError, match="re-extracted"):
        endpoint._projected_finite_penalty_evaluation(layout=changed, lambdas=weights, face=face)


@pytest.mark.parametrize("last_weight", [0.0, 7.0])
def test_mixed_endpoint_keeps_context_and_generic_manual_derivatives(last_weight):
    source, _, predictors = _copy_problem("shear", 1e8)
    selected = PenaltyComponent(
        "x:face",
        "x",
        0,
        slice(0, 1),
        np.eye(1),
        np.eye(1),
        rank=1.0,
        eigvals_omega=np.ones(1),
    )
    roots = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    manual = tuple(
        PenaltyComponent(
            f"manual:p{index}",
            "manual",
            0,
            slice(0, 3),
            np.outer(root, root),
            np.outer(root, root),
            rank=1.0,
            eigvals_omega=np.array([root @ root]),
        )
        for index, root in enumerate(roots)
    )
    third = _predictor(
        "third", 2, DenseGroupMatrix(np.eye(5, 3)), GroupSlice("manual", 0, 3), manual
    )
    layout = build_stacked_layout(
        (replace(predictors[0], penalties=(selected,)), predictors[1], third)
    )
    face = build_penalty_face(layout, ("first:x#face",))
    values = (0.0, 2.0, 0.0, 2.0, 3.0, 5.0, last_weight)
    weights = dict(zip(layout.penalty_names, values, strict=True))
    actual = endpoint._projected_finite_penalty_evaluation(
        layout=layout, lambdas=weights, face=face
    )
    local = algebra._compute_penalty_logdet_evaluation(
        dict(zip((item.name for item in source), (2.0, 0.0), strict=True)), source
    )
    assert actual.rank == 5 + (last_weight > 0)
    with localcontext() as context:
        context.prec = 100
        expected_extra = Decimal(31).ln()
        if last_weight:
            expected_extra += Decimal.from_float(last_weight).ln()
        error = abs(
            Decimal.from_float(actual.logdet) - Decimal.from_float(local.logdet) - expected_extra
        )
        assert error <= Decimal.from_float(actual.logdet_error + local.logdet_error)
    names = layout.penalty_names[3:]
    lambdas = (2, 3, 5)
    gradient = tuple(Fraction(value * (sum(lambdas) - value), 31) for value in lambdas)
    for i, name in enumerate(names[:3]):
        assert abs(Fraction(actual.gradient[name]) - gradient[i]) <= Fraction(
            actual.gradient_error[name]
        )
        for j, other in enumerate(names[:3]):
            direct = gradient[i] if i == j else Fraction(lambdas[i] * lambdas[j], 31)
            exact = direct - gradient[i] * gradient[j]
            assert abs(Fraction(actual.hessian[name, other]) - exact) <= Fraction(
                actual.hessian_error[name, other]
            )
    assert actual.gradient[names[3]] == float(last_weight > 0)
    for name in names:
        for retained in layout.penalty_names[1:3]:
            assert actual.hessian[name, retained] == actual.hessian[retained, name] == 0.0
            assert actual.hessian_error[name, retained] == 0.0


def test_joint_face_map_certificate_includes_cross_group_terms():
    from superglm.reml.penalty_support import PenaltyNumericalError

    off_diagonal = 2.0**-20
    row_map = np.array([[1.0, 0.0, 0.0], [off_diagonal, np.sqrt(1 - off_diagonal**2), 0.0]])
    # Both diagonal block volumes are one to floating precision, but the
    # common map has a resolvable cross-block defect and must be refused.
    assert np.max(np.abs(np.sum(row_map**2, axis=1) - 1)) <= np.finfo(float).eps
    with pytest.raises(PenaltyNumericalError, match="accuracy contract"):
        algebra._joint_near_isometry_volume_error(row_map, rank=2, root_rows=2, components=2)


def test_joint_face_map_bound_encloses_finite_nonorthogonality_and_zero_rank():
    from superglm.reml.penalty_support import PenaltyNumericalError

    row_map = np.array([[1.0 + 2.0**-48, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bound = algebra._joint_near_isometry_volume_error(row_map, rank=2, root_rows=3, components=2)
    with localcontext() as context:
        context.prec = 100
        exact = 2 * Decimal.from_float(row_map[0, 0]).ln()
        assert 0 < exact <= Decimal.from_float(bound)
    assert (
        algebra._joint_near_isometry_volume_error(row_map, rank=0, root_rows=3, components=2) == 0.0
    )
    with pytest.raises(PenaltyNumericalError, match="preserve penalty support"):
        algebra._joint_near_isometry_volume_error(
            np.zeros((1, 2)), rank=1, root_rows=1, components=1
        )
