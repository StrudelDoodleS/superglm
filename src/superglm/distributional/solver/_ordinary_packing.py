"""Single-write ordinary blocks for caller-validated reusable row scratch."""

from __future__ import annotations

import numpy as np
from numba import njit

# Match the original global-moment operand predicate exactly. Zeros, including
# negative zero, are permitted; nonzero magnitudes must retain normal-range
# headroom for the separately bounded degree-three moment products.
_MIN_FACTOR = 2.0**-128
_MAX_FACTOR = 2.0**128


@njit(cache=True, fastmath=False)
def _copy_dense_checked(panel, values, start):
    """Copy every dense entry and report its unchanged numerical eligibility.

    The caller validates exact array types/dtypes and destination dimensions
    before native execution. A False return is a deferred numerical refusal:
    all entries are still copied, and later hard live-source checks must run
    before any persistent moment is updated. No owned allocation is introduced.
    """
    valid = True
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value != 0.0 and not _MIN_FACTOR <= abs(value) <= _MAX_FACTOR:
                valid = False
            panel[row, start + column] = value
    return valid


@njit(cache=True, fastmath=False)
def _write_categorical_block(panel, codes, start, width):
    """Write the complete zero/one block, including every sink/base row.

    The caller checks intp code bounds [0, width], panel bounds and row counts
    before this native writer. Columns outside the block and unused trailing
    rows remain untouched. Every active block entry is written exactly once.
    """
    for row in range(codes.shape[0]):
        code = codes[row]
        for column in range(width):
            panel[row, start + column] = 1.0 if code == column else 0.0


def _warmup_ordinary_packing():
    """Compile C/F/strided destinations and readonly/layout input variants."""

    def matrix(rows, columns, layout):
        if layout == "A":
            return np.ones((2 * rows, 2 * columns), dtype=np.float64)[::-2, ::-2]
        return np.ones((rows, columns), dtype=np.float64, order=layout)

    for destination_layout in ("C", "F", "A"):
        panel = matrix(3, 4, destination_layout)
        for source_layout in ("C", "F", "A"):
            for readonly in (False, True):
                values = matrix(3, 2, source_layout)
                values.flags.writeable = not readonly
                _copy_dense_checked(panel, values, 1)
        for stride in (1, -2):
            for readonly in (False, True):
                codes = np.zeros(3 if stride == 1 else 6, dtype=np.intp)[::stride]
                codes.flags.writeable = not readonly
                _write_categorical_block(panel, codes, 1, 2)
