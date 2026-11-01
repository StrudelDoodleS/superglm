"""Generated row kernels of declared severity laws, and the array-level call into them.

Each module here is written by ``tools/generate_severity_families.py``; see its
docstring for the kernel signatures.  :func:`law_jets` returns the packed jet
``(J0, ..., Jp)`` (plan interface I1) and the rows whose channels are all
finite: a generated kernel marks a row it cannot represent with a non-finite
entry instead of raising.
"""

from __future__ import annotations

from math import comb
from types import ModuleType

import numpy as np
from numpy.typing import NDArray


def _column(values: object, n_rows: int) -> NDArray[np.float64]:
    column = np.ascontiguousarray(values, dtype=np.float64)
    if column.shape != (n_rows,):
        raise ValueError(f"row arrays must have shape ({n_rows},)")
    return column


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
    columns; ``multiplier`` (density only, default ones) scales each row.
    """
    if law not in module.LAWS or not 0 <= order <= module.MAX_ORDER:
        raise ValueError(f"{module.__name__} has no {law} kernel of order {order}")
    y = np.ascontiguousarray(response, dtype=np.float64)
    n_rows, k = len(y), len(module.PARAMETERS)
    theta = np.asarray(parameters, dtype=np.float64)
    if theta.shape != (n_rows, k) or len(data) != len(module.DATA):
        raise ValueError(f"{module.__name__} takes ({n_rows}, {k}) parameters and {module.DATA}")
    arguments = [y, *(_column(theta[:, j], n_rows) for j in range(k))]
    arguments += [_column(column, n_rows) for column in data]
    if law == "density":
        arguments.append(np.ones(n_rows) if multiplier is None else _column(multiplier, n_rows))
    jets = (np.empty(n_rows),) + tuple(
        np.empty((n_rows, comb(k + level - 1, level))) for level in range(1, order + 1)
    )
    getattr(module, f"{law}_{order}")(*arguments, *jets)
    valid = np.isfinite(jets[0])
    for channel in jets[1:]:
        valid &= np.isfinite(channel).all(axis=1)
    return jets, valid


__all__ = ["law_jets"]
