"""A call's own tracemalloc peak in a process whose other threads allocate too."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import Any


def own_traced_peak(call: Callable[..., Any], *args: Any) -> tuple[int, Any]:
    """``call(*args)``'s traced peak, the least of three calls, and the last call's result.

    tracemalloc's peak is the process's: another thread's allocations (an
    xdist worker's receiver thread answering a steal request builds a set of
    the stolen indices and a new queue) add to it and never subtract, so the
    least of three calls is the call's own. A call that changes state each
    measurement depends on is passed as a closure that sets that state first.
    """
    peaks = []
    for _ in range(3):
        tracemalloc.start()
        try:
            result = call(*args)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    return min(peaks), result
