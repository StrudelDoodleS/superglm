"""Weight-independent component support and original-coordinate roots."""

import numpy as np
import pytest


def test_symmetric_validation_preserves_the_represented_entries_at_both_range_limits():
    from superglm.reml.penalty_support import _validated_matrix

    tiny = np.nextafter(0.0, 1.0)
    matrix = np.array([[np.finfo(float).max, 3 * tiny], [3 * tiny, tiny]])
    np.testing.assert_array_equal(_validated_matrix(matrix), matrix)


def test_component_root_retains_representable_diagonal_range():
    from superglm.reml.penalty_support import _component_root

    root, limited, _ = _component_root(np.diag([1e308, 1e-308, 0.0]))
    assert root.shape == (2, 3)
    scaled = root / np.array([1e154, 1e-154, 1.0])
    np.testing.assert_allclose(scaled.T @ scaled, np.diag([1.0, 1.0, 0.0]), atol=8e-16)
    assert not limited


def test_component_root_covers_the_full_aliased_component():
    from superglm.reml.penalty_support import _component_root

    matrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    root, limited, _ = _component_root(matrix)
    assert root.shape == (1, 3)
    np.testing.assert_allclose(root.T @ root, matrix, atol=8e-16)
    assert limited


def test_support_preserves_component_units_and_freezes_owned_arrays():
    from superglm.reml.penalty_support import _penalty_support

    matrices = [np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 1.0, 0.0])]
    reference = _penalty_support(matrices)
    changed = _penalty_support([matrices[0] * 2.0**900, matrices[1] * 2.0**-900])
    assert reference.rank == changed.rank == 2
    projector = np.diag([1.0, 1.0, 0.0])
    for support in (reference, changed):
        np.testing.assert_allclose(support.Q_plus @ support.Q_plus.T, projector, atol=8e-16)
        np.testing.assert_allclose(
            support.coordinate_map, support.Q_plus @ support.coordinate_triangular, atol=8e-16
        )
        assert not support.component_roots[0].flags.writeable
        assert not support.Q_plus.flags.writeable
    matrices[0][0, 0] = 7.0
    assert reference.component_roots[0][0, 0] == 1.0


@pytest.mark.parametrize("matrix", [-np.eye(2) * 1e-300, np.array([[1.0, 2.0], [2.0, 1.0]])])
def test_material_indefiniteness_is_not_a_numerical_null(matrix):
    from superglm.reml.penalty_support import _component_root

    with pytest.raises(ValueError, match="positive semidefinite"):
        _component_root(matrix)


def test_factor_support_does_not_form_a_gram(monkeypatch):
    import superglm.reml.penalty_support as module

    original = module.decompose_gram
    inputs = []

    def component_only(matrix, **kwargs):
        inputs.append(matrix.copy())
        return original(matrix, **kwargs)

    monkeypatch.setattr(module, "decompose_gram", component_only)
    components = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    assert module._penalty_support(components).rank == 2
    assert len(inputs) == 2
    for actual, expected in zip(inputs, components, strict=True):
        np.testing.assert_array_equal(actual, expected)
