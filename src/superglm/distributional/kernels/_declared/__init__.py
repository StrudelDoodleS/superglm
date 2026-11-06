"""Generated row kernels of declared severity laws, and the array-level call into them.

Each module here is written by ``tools/generate_severity_families.py``; see its
docstring for the kernel signatures: one kernel per law, taking the order at
run time, releasing the GIL.  A kernel marks a row it cannot represent with a
non-finite entry instead of raising.  :func:`law_jets` returns the packed jet
``(J0, ..., Jp)`` (plan interface I1) and the rows whose channels are all
finite; :func:`density_rows` is the call of the families' ordinary rows.
"""

from __future__ import annotations

from importlib import import_module
from math import comb
from types import ModuleType

import numpy as np
from numpy.typing import NDArray


def _column(values: object, n_rows: int) -> NDArray[np.float64]:
    # One operand form, whatever the caller holds (read-only, strided): one compiled signature.
    column = np.require(values, dtype=np.float64, requirements=("C", "A", "W"))
    if column.shape != (n_rows,):
        raise ValueError(f"row arrays must have shape ({n_rows},)")
    return column


def _evaluate(
    module: ModuleType,
    law: str,
    order: int,
    response: object,
    parameters: tuple | NDArray,
    data: tuple,
    multiplier: object | None,
) -> tuple[NDArray[np.float64], ...]:
    """``law``'s packed jet to ``order``, as written: a non-finite entry marks a row the kernel cannot represent."""
    if law not in module.LAWS or not 0 <= order <= module.MAX_ORDER:
        raise ValueError(f"{module.__name__} has no {law} kernel of order {order}")
    n_rows, k = np.shape(response)[0], len(module.PARAMETERS)
    if len(parameters) != k or len(data) != len(module.DATA):
        raise ValueError(f"{module.__name__} takes {module.PARAMETERS} and {module.DATA}")
    arguments = [_column(response, n_rows), *(_column(parameters[j], n_rows) for j in range(k))]
    arguments += [_column(column, n_rows) for column in data]
    if law == "density":
        arguments.append(np.ones(n_rows) if multiplier is None else _column(multiplier, n_rows))
    # Levels above ``order`` are unread: empty rows keep the kernel's one signature.
    jets = (np.empty(n_rows),) + tuple(
        np.empty((n_rows if level <= order else 0, comb(k + level - 1, level)))
        for level in range(1, module.MAX_ORDER + 1)
    )
    getattr(module, law)(*arguments, order, *jets)
    return jets[: order + 1]


def law_jets(
    module: ModuleType,
    law: str,
    order: int,
    response: object,
    parameters: object,
    data: tuple = (),
    multiplier: object | None = None,
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.bool_]]:
    """Jet of ``law`` (``"density"``, ``"logsf"``, ``"logcdf"``) to ``order`` in the natural parameters.

    ``parameters`` is ``(n, K)``; ``data`` holds the declaration's per-row data
    columns; ``multiplier`` (density only, default ones) scales each row.  The
    second result marks the rows whose channels are all finite.
    """
    theta = np.asarray(parameters, dtype=np.float64)
    n_rows = np.shape(response)[0]
    if theta.shape != (n_rows, len(module.PARAMETERS)):
        raise ValueError(f"{module.__name__} takes ({n_rows}, {len(module.PARAMETERS)}) parameters")
    jets = _evaluate(module, law, order, response, theta.T, data, multiplier)
    valid = np.isfinite(jets[0])
    for channel in jets[1:]:
        valid &= np.isfinite(channel).all(axis=1)
    return jets, valid


def channels(jets: tuple[NDArray[np.float64], ...]) -> tuple:
    """``(value, score, hessian)`` of a jet taken to order 0, 1 or 2, ``None`` for a level it lacks."""
    return (*jets, None, None)[:3]


# The largest input the compiled rows take.  Below it every entry of the order-two jet at
# unit mass is at least ``2^-800`` (a product of at most four reciprocals of inputs), a
# normal number, so the row's mass multiplies it without losing digits; the array rows
# rescale an entry that would be subnormal, which only inputs past it need.
_MAX_INPUT = 2.0**200


def _within_range(columns: tuple) -> bool:
    """Every column is finite and at most ``_MAX_INPUT`` in magnitude."""
    return all(
        column.size == 0 or max(column.max(), -column.min()) <= _MAX_INPUT
        for column in map(np.asarray, columns)
    )


def density_rows(
    module: ModuleType,
    order: int,
    response: object,
    parameters: tuple,
    data: tuple = (),
    multiplier: object | None = None,
) -> tuple[NDArray[np.float64], ...] | None:
    """The density's packed jet to ``order`` over ordinary rows, or ``None``.

    ``parameters`` is the natural parameters' columns.  The kernel runs without
    the GIL, so row chunks on the worker pool run side by side.  ``None`` means
    an input is past ``_MAX_INPUT`` or some row has a non-finite channel, which a
    caller answers by evaluating the rows with its array route, whose domain is
    wider and whose refusals stay its own.  One sum per channel tests for the
    channels: the sum of finite entries is finite unless it overflows, and then
    the array route evaluates the rows.
    """
    columns = (response, *parameters, *data, *(() if multiplier is None else (multiplier,)))
    if not _within_range(columns):
        return None
    jets = _evaluate(module, "density", order, response, parameters, data, multiplier)
    with np.errstate(over="ignore", invalid="ignore"):
        finite = all(np.isfinite(channel.sum()) for channel in jets)
    return jets if finite else None


# The laws whose density kernel serves a family's ordinary rows (``kernels.gamma``,
# ``log_normal``, ``generalized_pareto`` and ``two_piece``).
_ROW_LAWS = (
    "gamma",
    "generalized_pareto",
    "log_normal_mean",
    "two_piece_location",
    "two_piece_mean",
)


def _warmup_rows() -> None:
    """Compile the density kernel of each law above, on no rows."""
    empty = np.zeros(0)
    for name in _ROW_LAWS:
        module = import_module(f"{__name__}.{name}")
        density_rows(
            module, 2, empty, (empty,) * len(module.PARAMETERS), (empty,) * len(module.DATA)
        )


__all__ = ["channels", "density_rows", "law_jets"]
