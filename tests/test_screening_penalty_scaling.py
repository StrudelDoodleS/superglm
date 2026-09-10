"""Finite PSD penalty roots across the binary64 exponent range."""

import math

import numpy as np
import pytest

from superglm.screening._factor_kernels import _penalty_root
from superglm.screening._overlap import tensor_penalty_root


@pytest.mark.parametrize(
    "diagonal",
    [
        [1.0],
        [math.ldexp(1.0, 1023)],
        [math.ldexp(1.0, -1074)],
        [math.ldexp(1.0, 1023), math.ldexp(1.0, -1074)],
    ],
)
@pytest.mark.parametrize("tensor", [False, True])
def test_screening_positive_diagonal_root_preserves_each_represented_direction(diagonal, tensor):
    penalty = np.diag(diagonal)
    if tensor:
        root = tensor_penalty_root(penalty, np.zeros((1, 1)))
    else:
        root, dropped, bound = _penalty_root(penalty)
        assert dropped == 0.0
        assert np.isfinite(bound)
    assert np.all(np.isfinite(root))
    # Column scaling makes the exact identity observable without squaring an
    # extreme root or using an absolute tolerance that erases the tiny column.
    scaled = root / np.sqrt(np.asarray(diagonal))[None, :]
    np.testing.assert_allclose(
        scaled.T @ scaled,
        np.eye(len(diagonal)),
        rtol=0,
        atol=8 * len(diagonal) * np.finfo(float).eps,
    )
