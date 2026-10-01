"""Free a numba compile's reference cycles at the next ``fs`` leaf build.

Numba's type inference tries overloads and keeps the failures it meets
(``TypingError``, ``RequireLiteralValue``) with their tracebacks: reference
cycles of exceptions, tracebacks and frames.  A frame reaches its caller
through ``f_back``, and a frame object that outlives its call keeps the
locals it returned with, so every frame on the stack when a kernel compiles
keeps its locals alive until the cyclic collector reaches the oldest
generation.  CPython defers that collection while the long-lived objects
pending collection are fewer than a quarter of those already long-lived
(CPython's garbage-collector design notes, ``InternalDocs/garbage_collector.md``),
and the compiles of a fit leave on the order of 10^5 long-lived objects
behind.  On an ``fs`` term the pinned locals include ``K p^2`` stacks of
signed pseudo-rows: a first fit on an empty numba cache (400 levels beside a
120-level categorical, Tweedie) peaked at 1046 MiB, against 611 MiB once its
kernels were cached and 625 MiB for the cold master it was gated on (#432).

``_MarkCompile`` listens to numba's ``"numba:compile"`` event (the event API
of ``numba.core.event``) and only records that the outermost compile of a
superglm kernel has ended.  ``collect_after_compile``, called by the ``fs``
leaf build before it forms a new system, then runs one full collection,
unless the application has disabled the collector: the frames the cycles
reach have returned by then, and the dead stacks go before the next one is
allocated.  The event fires only on a disk-cache miss (a cached kernel loads
without compiling), so a warm cache never collects; ``superglm.warmup``,
which compiles outside any fit (and in every process an uncached inline
helper), drops its own mark (``forget_compiles``).  Measured on that cold
first fit: 1.4 s of collection and a 773 MiB peak (1.24x).  Collecting at
every compile's end instead reached 1.20x for 3.5 s per cold fit and about
9 s per CI worker; the owner chose the ``fs`` checkpoint (2026-10-01).  Other
packages' kernels are left alone.
"""

from __future__ import annotations

import gc
import threading

from numba.core import event as numba_event  # type: ignore[import-untyped]

_state = threading.local()
_compiled = [False]  # a superglm compile ended since the last collection (any thread)


def _is_superglm_kernel(data) -> bool:
    function = getattr(data.get("dispatcher"), "py_func", None) if isinstance(data, dict) else None
    module = getattr(function, "__module__", None) or ""
    return module == "superglm" or module.startswith("superglm.")


class _MarkCompile(numba_event.Listener):
    """Record the end of the outermost compile that includes a superglm kernel."""

    def on_start(self, event) -> None:
        _state.depth = getattr(_state, "depth", 0) + 1
        if _is_superglm_kernel(event.data):
            _state.ours = True

    def on_end(self, event) -> None:
        _state.depth = max(getattr(_state, "depth", 1) - 1, 0)
        if _state.depth == 0 and getattr(_state, "ours", False):
            _state.ours = False
            _compiled[0] = True


def forget_compiles() -> None:
    """Drop the mark of compiles that ran outside any fit (``superglm.warmup``).

    Their frames hold no fit's arrays, so the next ``fs`` leaf build need not
    collect for them: without this, a warm-cache process paid one full
    collection on its first fit for warmup's own compile of the uncached
    ``_add_raw_row`` (0.15 s on a 30,000-row ``sz`` fit).
    """
    _compiled[0] = False


def collect_after_compile() -> None:
    """One full collection if a superglm kernel compiled since the last one."""
    if _compiled[0]:
        _compiled[0] = False
        if gc.isenabled():
            gc.collect()


numba_event.register("numba:compile", _MarkCompile())
