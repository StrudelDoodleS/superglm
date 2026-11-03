"""The panel's categorical writer: complete one-hot blocks into row-block panels."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributional.solver import _ordinary_packing as packing


@pytest.fixture(autouse=True, scope="module")
def _closed_writers():
    """Close the writer before any test calls it, as ``superglm.warmup`` does.

    An open dispatcher compiles one specialisation per operand layout it sees; a
    worker that ran only some layouts then holds several equally good matches
    for another, and numba refuses the call as ambiguous.
    """
    packing._warmup_ordinary_packing()


def _matrix(values, layout):
    if layout == "A":
        storage = np.empty((2 * values.shape[0], 2 * values.shape[1]))
        view = storage[::-2, ::-2]
        view[:] = values
        return view
    return np.array(values, dtype=np.float64, order=layout)


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
    expected = panel.copy()
    codes = np.full(rows, width, dtype=np.intp)
    packing._write_categorical_block(panel, codes, 1, width)
    expected[:rows, 1 : 1 + width] = 0.0
    np.testing.assert_array_equal(panel, expected)


def test_warmup_covers_actual_layout_and_readonly_signatures_without_fastmath():
    packing._warmup_ordinary_packing()
    kernel = packing._write_categorical_block
    compiled = tuple(kernel.nopython_signatures)
    for destination_layout in ("C", "F", "A"):
        for readonly in (False, True):
            panel = _matrix(np.zeros((3, 4)), destination_layout)
            for stride in (1, -2):
                codes = np.zeros(3 if stride == 1 else 6, dtype=np.intp)[::stride]
                codes.flags.writeable = not readonly
                packing._write_categorical_block(panel, codes, 1, 2)
    assert tuple(kernel.nopython_signatures) == compiled
    assert not kernel.targetoptions.get("fastmath", False)
