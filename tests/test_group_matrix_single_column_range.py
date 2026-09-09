"""One-column range dispatch and scalar-product correctness are separate."""

import linecache
import sys

import numpy as np
import pytest

from superglm._group_matrix._group_matrix_core import DenseGroupMatrix
from superglm._group_matrix._group_matrix_range import group_range_matvec


@pytest.mark.parametrize("width", [1, 2])
def test_single_column_range_avoids_matrix_vector_dispatch(width):
    executed_matmul = []

    def trace(frame, event, arg):
        if event == "line" and frame.f_code is group_range_matvec.__code__:
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            if " @ " in line:
                executed_matmul.append(line)
        return trace

    group = DenseGroupMatrix(np.arange(12 * width, dtype=float).reshape(12, width))
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        group_range_matvec(group, 2, 9, np.ones(width))
    finally:
        sys.settrace(previous)
    assert len(executed_matmul) == int(width != 1)


@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("start,stop", [(0, 9), (2, 7), (4, 4)])
def test_single_column_range_matches_live_scalar_products(order, start, stop):
    source = np.arange(18, dtype=float).reshape(9, 2)
    matrix = source[:, :1] if order == "strided" else np.array(source[:, :1], order=order)
    group = DenseGroupMatrix(matrix)
    # Exercise the actual stored stride even if construction normalizes storage.
    group.M = matrix
    beta = np.array([-0.375])
    for factor in (1.0, -2.0):
        group.M[:] *= factor
        actual = group_range_matvec(group, start, stop, beta)
        expected = np.array([float(value) * float(beta[0]) for value in matrix[start:stop, 0]])
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (stop - start,)


@pytest.mark.parametrize("coefficient", [0.0, -0.25, 0.75, 1.0])
def test_single_column_products_satisfy_rounding_bounds(coefficient):
    info = np.finfo(float)
    source = np.array([0.0, -0.0, info.smallest_subnormal, -info.tiny, 1e-200, -1e200, info.max])
    group = DenseGroupMatrix(source[:, None])
    result = group_range_matvec(group, 0, len(source), np.array([coefficient]))
    # One scalar multiplication: relative epsilon plus one subnormal quantum.
    exact = source.astype(np.longdouble) * np.longdouble(coefficient)
    error = np.abs(result.astype(np.longdouble) - exact)
    bound = np.abs(exact) * info.eps + np.longdouble(info.smallest_subnormal)
    assert np.all(error <= bound)


@pytest.mark.parametrize("width", [0, 2])
def test_malformed_single_column_beta_keeps_matmul_error(width):
    group = DenseGroupMatrix(np.ones((3, 1)))
    with pytest.raises(ValueError):
        group_range_matvec(group, 0, 3, np.ones(width))


def test_single_column_custom_buffers_remain_refused():
    class CustomArray(np.ndarray):
        pass

    group = DenseGroupMatrix(np.ones((3, 1)))
    group.M = group.M.view(CustomArray)
    assert group_range_matvec(group, 0, 3, np.ones(1)) is None
