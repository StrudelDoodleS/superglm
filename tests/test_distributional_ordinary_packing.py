"""Complete ordinary-block writes into reusable scratch, with exact domain checks."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributional.solver import _ordinary_packing as packing


def _matrix(values, layout):
    if layout == "A":
        storage = np.empty((2 * values.shape[0], 2 * values.shape[1]))
        view = storage[::-2, ::-2]
        view[:] = values
        return view
    return np.array(values, dtype=np.float64, order=layout)


@pytest.mark.parametrize("source_layout", ["C", "F", "A"])
@pytest.mark.parametrize("destination_layout", ["C", "F", "A"])
@pytest.mark.parametrize("readonly", [False, True])
def test_dense_copy_covers_only_selected_block_preserving_exact_values(
    source_layout, destination_layout, readonly
):
    literal = np.array([[0.0, -0.0], [2.0**-128, -(2.0**128)], [0.5, -3.0]])
    values = _matrix(literal, source_layout)
    values.flags.writeable = not readonly
    panel = _matrix(np.full((5, 6), 17.0), destination_layout)
    expected = panel.copy()
    expected[:3, 2:4] = literal
    assert packing._copy_dense_checked(panel, values, 2)
    np.testing.assert_array_equal(panel, expected)
    np.testing.assert_array_equal(np.signbit(panel[:3, 2:4]), np.signbit(literal))
    np.testing.assert_array_equal(values, literal)


@pytest.mark.parametrize(
    "value,accepted",
    [
        (0.0, True),
        (-0.0, True),
        (2.0**-128, True),
        (-(2.0**-128), True),
        (2.0**128, True),
        (-(2.0**128), True),
        (np.nextafter(2.0**-128, 0.0), False),
        (np.nextafter(2.0**128, np.inf), False),
        (np.nextafter(0.0, 1.0), False),
        (np.nan, False),
        (np.inf, False),
        (-np.inf, False),
    ],
)
def test_dense_domain_matches_original_zero_or_bounded_finite_predicate(value, accepted):
    values = np.array([[value, 2.0], [3.0, -0.0]])
    panel = np.full((2, 4), 19.0)
    assert packing._copy_dense_checked(panel, values, 1) is accepted
    # Even refused operands fully prepare scratch; refusal must not skip later
    # source checks or leave a partially written block that could be published.
    np.testing.assert_array_equal(panel[:, 1:3], values)
    np.testing.assert_array_equal(np.signbit(panel[:, 1:3]), np.signbit(values))
    np.testing.assert_array_equal(panel[:, (0, 3)], 19.0)


@pytest.mark.parametrize("destination_layout", ["C", "F", "A"])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("strided_codes", [False, True])
def test_categorical_write_replaces_poison_and_base_rows_each_chunk(
    destination_layout, readonly, strided_codes
):
    panel = _matrix(np.full((5, 7), np.nan), destination_layout)
    for literal in (np.array([0, 2, 1], dtype=np.intp), np.array([2, 1], dtype=np.intp)):
        if strided_codes:
            storage = np.empty(len(literal) * 2, dtype=np.intp)
            codes = storage[::-2]
            codes[:] = literal
        else:
            codes = literal.copy()
        codes.flags.writeable = not readonly
        before = panel.copy()
        packing._write_categorical_block(panel, codes, 2, 2)
        expected = before.copy()
        expected[: len(codes), 2:4] = literal[:, None] == np.arange(2)
        np.testing.assert_array_equal(panel, expected)
        assert not np.any(np.signbit(panel[: len(codes), 2:4]))
        np.testing.assert_array_equal(codes, literal)


def test_poisoned_base_row_is_an_unfixed_scatter_counterexample():
    codes = np.array([0, 2, 1], dtype=np.intp)
    old_panel = np.full((3, 2), 7.0)
    # The old scatter only assigned one-hot entries, relying on an earlier
    # whole-panel fill. Removing that fill leaves poison in zero/base entries.
    for row, code in enumerate(codes):
        if code < 2:
            old_panel[row, code] = 1.0
    expected = (codes[:, None] == np.arange(2)).astype(np.float64)
    assert not np.array_equal(old_panel, expected)
    packing._write_categorical_block(old_panel, codes, 0, 2)
    np.testing.assert_array_equal(old_panel, expected)


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("rows", [0, 1, 4])
def test_empty_and_partial_block_shapes_keep_adjacent_scratch_untouched(width, rows):
    panel = np.full((5, 6), 23.0)
    values = np.ones((rows, width))
    expected = panel.copy()
    expected[:rows, 1 : 1 + width] = values
    assert packing._copy_dense_checked(panel, values, 1)
    np.testing.assert_array_equal(panel, expected)
    codes = np.full(rows, width, dtype=np.intp)
    packing._write_categorical_block(panel, codes, 1, width)
    expected[:rows, 1 : 1 + width] = 0.0
    np.testing.assert_array_equal(panel, expected)


def test_warmup_covers_actual_layout_and_readonly_signatures_without_fastmath():
    packing._warmup_ordinary_packing()
    kernels = (packing._copy_dense_checked, packing._write_categorical_block)
    compiled = tuple(tuple(kernel.nopython_signatures) for kernel in kernels)
    for source_layout in ("C", "F", "A"):
        for destination_layout in ("C", "F", "A"):
            for readonly in (False, True):
                values = _matrix(np.ones((3, 2)), source_layout)
                values.flags.writeable = not readonly
                panel = _matrix(np.zeros((3, 4)), destination_layout)
                assert packing._copy_dense_checked(panel, values, 1)
                for stride in (1, -2):
                    codes = np.zeros(3 if stride == 1 else 6, dtype=np.intp)[::stride]
                    codes.flags.writeable = not readonly
                    packing._write_categorical_block(panel, codes, 1, 2)
    assert tuple(tuple(kernel.nopython_signatures) for kernel in kernels) == compiled
    assert all(not kernel.targetoptions.get("fastmath", False) for kernel in kernels)
