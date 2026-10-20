"""Single-write ordinary blocks for caller-validated reusable row scratch."""

from __future__ import annotations

from numba import njit, types

from superglm._numba_compile import compile_closed

# Match the original global-moment operand predicate exactly. Zeros, including
# negative zero, are permitted. Nonzero magnitudes keep the separately bounded
# degree-three moment products within the normal exponent range.
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


# One signature per writer serves every operand form: a dense source is a row
# span of a stored design block in any layout or writeability, categorical
# codes a row span of stored codes, and the destination a row span of the
# plan's panel.
_ORDINARY_WRITERS = (
    (
        _copy_dense_checked,
        (
            types.Array(types.float64, 2, "A"),
            types.Array(types.float64, 2, "A", readonly=True),
            types.intp,
        ),
    ),
    (
        _write_categorical_block,
        (
            types.Array(types.float64, 2, "A"),
            types.Array(types.intp, 1, "A", readonly=True),
            types.intp,
            types.intp,
        ),
    ),
)


def _warmup_ordinary_packing():
    """Compile the two writers' signatures (``compile_closed``)."""
    compile_closed(*_ORDINARY_WRITERS)
