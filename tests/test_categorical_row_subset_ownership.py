"""Categorical subsets preserve stored columns and own their mutable codes."""

import numpy as np
import pytest

from superglm.group_matrix import CategoricalGroupMatrix

# Column 3 is deliberately unobserved; -1 and the stored sink 4 both mean
# the all-zero base row. This oracle does not use categorical construction.
CODES = np.array([2, -1, 0, 4, 1, 2], dtype=np.intp)
STORED_COLUMNS = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=float,
)
SELECTORS = [
    pytest.param(np.array([5, 1, 5, 0, 3, 2]), id="repeated-reordered"),
    pytest.param(np.array([-1, -5, -3]), id="negative-indices"),
    pytest.param(np.array([False, True, True, True, False, True]), id="boolean"),
    pytest.param(slice(1, 6, 2), id="strided-slice"),
    pytest.param(slice(None, None, -1), id="reversed-slice"),
    pytest.param(np.array([3, 1, 3]), id="only-sink"),
    pytest.param(np.array([], dtype=np.intp), id="empty-indices"),
    pytest.param(np.zeros(6, dtype=bool), id="empty-boolean"),
    pytest.param(slice(2, 2), id="empty-slice"),
]


@pytest.mark.parametrize("selector", SELECTORS)
def test_categorical_subset_preserves_rows_sink_and_unobserved_columns(selector):
    parent = CategoricalGroupMatrix(CODES, n_levels=4)

    child = parent.row_subset(selector)

    expected = STORED_COLUMNS[selector]
    assert child.shape == expected.shape
    assert child.n_levels == 4
    np.testing.assert_array_equal(child.toarray(), expected)
    # Small integer values make every product and sum exact in float64.
    beta = np.array([2.0, -3.0, 5.0, 7.0])
    weights = np.arange(1, len(expected) + 1, dtype=float)
    np.testing.assert_array_equal(child.matvec(beta), expected @ beta)
    np.testing.assert_array_equal(child.rmatvec(weights), expected.T @ weights)
    np.testing.assert_array_equal(child.gram(weights), expected.T @ (weights[:, None] * expected))


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("mutate", ["parent", "child"])
def test_categorical_subset_code_mutation_is_isolated(selector, mutate):
    parent = CategoricalGroupMatrix(CODES, n_levels=4)
    child = parent.row_subset(selector)

    if mutate == "parent":
        parent.codes[:] = 0
        np.testing.assert_array_equal(child.toarray(), STORED_COLUMNS[selector])
    else:
        child.codes[:] = 0
        np.testing.assert_array_equal(parent.toarray(), STORED_COLUMNS)

    # Neither mutation may escape through the caller's original code array.
    np.testing.assert_array_equal(CODES, [2, -1, 0, 4, 1, 2])
