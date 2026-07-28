"""Lossless row-support compression for factored SSP group matrices.

Compression here is deduplication of repeated design rows, never binning, so it
must leave every fitted quantity unchanged.  It is unrelated to ``discrete=True``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_support import detect_row_support


def test_detect_row_support_compresses_repeated_rows():
    base = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    rows = base[np.array([0, 1, 2, 0, 1, 0])]

    result = detect_row_support(sp.csr_matrix(rows))

    assert result is not None
    b_unique, row_index = result
    assert b_unique.shape == (3, 2)
    assert row_index.shape == (6,)
    np.testing.assert_allclose(b_unique[row_index], rows)


def test_detect_row_support_declines_when_rows_are_distinct():
    rows = np.arange(20.0).reshape(10, 2)

    assert detect_row_support(sp.csr_matrix(rows)) is None


def test_detect_row_support_declines_on_empty_basis():
    assert detect_row_support(sp.csr_matrix((0, 3))) is None


def test_detect_row_support_respects_max_ratio():
    base = np.array([[1.0, 0.0], [0.0, 2.0]])
    rows = base[np.array([0, 1, 0, 1])]

    assert detect_row_support(sp.csr_matrix(rows), max_ratio=0.9) is not None
    assert detect_row_support(sp.csr_matrix(rows), max_ratio=0.1) is None
