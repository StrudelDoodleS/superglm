"""Layout relocation preserves the selected local penalty family and its evidence."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._predictor_compiler import CompiledPredictorDesign
from superglm.distributional import layout as layout_module
from superglm.distributional.predictor import CompiledPredictor
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, SparseSSPGroupMatrix
from superglm.links import IdentityLink
from superglm.reml import penalty_algebra as algebra
from superglm.types import GroupSlice
from tests.test_penalty_context_support import _decimal_determinant, _decimal_fixed_root


def _predictor(name, index, matrix, group, components):
    compiled = CompiledPredictorDesign(
        DesignMatrix([matrix], n=matrix.shape[0], p=group.size),
        (group,),
        {},
        (group.name,),
        {},
        (),
    )
    return CompiledPredictor(
        name, index, IdentityLink(), compiled, True, np.zeros(matrix.shape[0]), tuple(components)
    )


def _copy_problem(kind="shear", magnitude=1e8):
    rng = np.random.default_rng(817)
    basis = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    roots = [rng.normal(size=(3, 4)) @ basis[:, :4].T for _ in range(2)]
    raw = [root.T @ root for root in roots]
    middle = np.eye(5)
    middle[4, 0 if kind == "shear" else 4] = magnitude
    coordinate_map = basis @ middle @ basis.T
    matrix = SparseSSPGroupMatrix(sp.eye(5, format="csr"), coordinate_map)
    matrix.omega = sum(raw)
    matrix.omega_components = [("a", raw[0]), ("b", raw[1])]
    group = GroupSlice("shared", 0, 5)
    components, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    first = _predictor("first", 0, DenseGroupMatrix(np.ones((5, 1))), GroupSlice("x", 0, 1), ())
    second = _predictor("second", 1, matrix, group, components)
    return components, coordinate_map, (first, second)


def _evaluate(components, values):
    return algebra._compute_penalty_logdet_evaluation(
        dict(zip((component.name for component in components), values, strict=True)),
        list(components),
    )


@pytest.mark.parametrize("kind", ["scale", "shear"])
@pytest.mark.parametrize("active", [0, 1])
def test_layout_copies_preserve_the_fixed_target_logdet_certificate(kind, active):
    source, coordinate_map, predictors = _copy_problem(kind)
    layout = layout_module.build_stacked_layout(predictors)
    values = np.array([2.0 * (index == active) for index in range(2)])
    geometry = algebra._context_geometry(source)
    with localcontext() as context:
        context.prec = 100
        root = _decimal_fixed_root(geometry, active, coordinate_map)
        gram = [
            [sum(a * b for a, b in zip(left, right, strict=True)) for right in root]
            for left in root
        ]
        expected = _decimal_determinant(gram).ln() + len(root) * Decimal(2).ln()
        for components in (source, layout.predictors[1].penalties, layout.penalties):
            result = _evaluate(components, values)
            assert algebra.compute_total_penalty_rank(list(components)) == 4
            assert result.rank == 3
            assert abs(Decimal.from_float(result.logdet) - expected) <= Decimal.from_float(
                result.logdet_error
            )


def test_layout_copy_uses_new_owners_and_preserves_existing_immutable_evidence(monkeypatch):
    source, _, predictors = _copy_problem()
    original = algebra._context_geometry(source)
    _evaluate(source, [2.0, 3.0])

    def forbid_volume(*_args, **_kwargs):
        pytest.fail("unchanged local placement rebuilt its coordinate-volume certificate")

    monkeypatch.setattr(algebra, "_support_coordinate_volume", forbid_volume)
    layout = layout_module.build_stacked_layout(predictors)
    local = algebra._context_geometry(list(layout.predictors[1].penalties))
    embedded = algebra._context_geometry(list(layout.penalties))
    assert local is not None and embedded is not None
    assert local is not original and embedded is not original and local is not embedded
    for owner, components in (
        (local, layout.predictors[1].penalties),
        (embedded, layout.penalties),
    ):
        for field in (
            "support",
            "coordinate_map",
            "matrix_error_bounds",
            "ssp_roots",
            "ssp_root_errors",
            "volume",
            "volume_activity",
        ):
            assert getattr(owner, field) is getattr(original, field)
        assert owner.ssp_refined == original.ssp_refined
        assert owner.last_weights is owner.last_evaluation is owner.face_support is None
        for component, held in zip(components, owner.matrices, strict=True):
            assert component.omega_ssp is held
            with pytest.raises(ValueError):
                held.setflags(write=True)
    assert local.keys[0][2:4] == (0, slice(0, 5))
    assert embedded.keys[0][2:4] == (1, slice(3, 8))
    first = _evaluate(layout.penalties, [2.0, 3.0])
    cached = embedded.last_evaluation
    second = _evaluate(layout.penalties, [2.0, 3.0])
    assert embedded.last_evaluation is cached
    assert first == second
    _evaluate(layout.penalties, [3.0, 2.0])
    assert embedded.last_evaluation is not cached
    assert local.last_evaluation is None
    assert original.last_weights == (2.0, 3.0)


@pytest.mark.parametrize("mutation", ["matrix", "order", "kind", "width", "group", "negative"])
def test_layout_context_transfer_refuses_changed_copy_geometry(mutation):
    source, _, predictors = _copy_problem()
    predictor = predictors[1]
    copied = [layout_module._qualify_local_component(item, predictor=predictor) for item in source]
    if mutation == "matrix":
        copied[0] = replace(copied[0], omega_ssp=copied[0].omega_ssp + np.eye(5))
    elif mutation == "order":
        copied.reverse()
    elif mutation == "kind":
        copied[0] = replace(copied[0], penalty_kind="repeated", repeat_count=1, block_width=5)
    elif mutation == "width":
        copied[0] = replace(copied[0], group_sl=slice(0, 4))
    elif mutation == "group":
        copied[1] = replace(copied[1], group_index=1)
    else:
        copied = [replace(component, group_index=-1) for component in copied]
    with pytest.raises(ValueError, match="copy|copied"):
        algebra._rebind_penalty_context(source, copied)
    assert all(getattr(component, "_penalty_geometry", None) is None for component in copied)


def test_layout_context_transfer_needs_a_complete_valid_source_family():
    source, _, predictors = _copy_problem()
    predictor = predictors[1]
    copied = [layout_module._qualify_local_component(item, predictor=predictor) for item in source]
    algebra._rebind_penalty_context(source[:1], copied[:1])
    assert getattr(copied[0], "_penalty_geometry", None) is None
    invalidated = [replace(source[0], omega_ssp=source[0].omega_ssp.copy()), source[1]]
    algebra._rebind_penalty_context(invalidated, copied)
    assert all(getattr(component, "_penalty_geometry", None) is None for component in copied)


@pytest.mark.parametrize("collision", ["name", "index", "slice"])
def test_layout_context_transfer_cannot_merge_distinct_source_families(collision):
    _, _, predictors = _copy_problem()
    matrix = predictors[1].compiled.design.group_matrices[0]
    groups = (GroupSlice("left", 0, 5), GroupSlice("right", 5, 10))
    source, _, _ = algebra.build_penalty_context([matrix, matrix], list(enumerate(groups)))
    copied = [
        layout_module._copy_component(
            component,
            name=f"location:{component.group_name}#{component.name.rsplit(':', 1)[1]}",
            group_name=f"location:{component.group_name}",
            group_index=component.group_index,
            group_sl=component.group_sl,
        )
        for component in source
    ]
    for index in (2, 3):
        changes = {
            "name": {"group_name": copied[0].group_name},
            "index": {"group_index": copied[0].group_index},
            "slice": {"group_sl": copied[0].group_sl},
        }[collision]
        copied[index] = replace(copied[index], **changes)
    with pytest.raises(ValueError, match="copied"):
        algebra._rebind_penalty_context(source, copied)
    assert all(getattr(component, "_penalty_geometry", None) is None for component in copied)


def test_copied_context_invalidates_replaced_matrices_without_changing_other_owners():
    source, _, predictors = _copy_problem()
    layout = layout_module.build_stacked_layout(predictors)
    held = algebra._context_geometry(list(layout.penalties))
    local = algebra._context_geometry(list(layout.predictors[1].penalties))
    layout.penalties[0].omega_ssp = np.eye(5)
    assert algebra._context_geometry(list(layout.penalties)) is None
    assert algebra._context_geometry(list(layout.predictors[1].penalties)) is local
    assert algebra._context_geometry(source) is not held
    assert _evaluate(layout.penalties, [2.0, 3.0]).rank == 5
    assert _evaluate(source, [2.0, 3.0]).rank == 4


def test_layout_transfer_preserves_lazy_singleton_support(monkeypatch):
    matrix = SparseSSPGroupMatrix(sp.eye(3, format="csr"), np.eye(3))
    matrix.omega = np.diag([1.0, 2.0, 0.0])
    group = GroupSlice("single", 0, 3)
    source, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    owner = algebra._context_geometry(source)
    assert owner.support is None
    predictor = _predictor("location", 0, matrix, group, source)
    from superglm.reml import penalty_support

    def forbid_support(*_args, **_kwargs):
        pytest.fail("copy eagerly selected a previously lazy support")

    monkeypatch.setattr(penalty_support, "_penalty_support", forbid_support)
    layout = layout_module.build_stacked_layout([predictor])
    for components in (layout.penalties, layout.predictors[0].penalties):
        copied = algebra._context_geometry(list(components))
        assert copied is not None and copied is not owner
        assert copied.support is None
