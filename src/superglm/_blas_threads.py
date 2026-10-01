"""Scoped BLAS thread capping for solver work.

Threaded OpenBLAS LAPACK is measured 6-35x *slower* than single-threaded on
the p^3 kernels this library lives on (potrf/trtri/pocon/syevd at
p ~ 150-1000): 56 ms vs 1.6 ms per ``decompose_gram`` at p=203 on a 16-core
box, and whole fits 1.9-4.8x faster with the pool capped to one thread
(exact, discrete, wide-categorical and multi-tensor configurations alike).
The fan-out/synchronisation overhead of the threaded kernels dominates these
small factorizations, while the library's genuinely large row-space work runs
in numba kernels and bincount aggregations that never enter BLAS.

The cap is scoped to fit calls and restored afterwards, and it targets only
BLAS pools -- tabmat's OpenMP kernels and numba threading are untouched.

``SUPERGLM_BLAS_THREADS`` overrides the policy: unset or ``auto`` caps BLAS
to one thread during fits; an integer caps to that many; ``native`` disables
capping and leaves the user's BLAS configuration alone. Wide designs release
the automatic cap once they reach the measured p³ break-even.
"""

from __future__ import annotations

import os
import threading
import warnings
from contextlib import contextmanager

_ENV_VAR = "SUPERGLM_BLAS_THREADS"

# threadpool_limits mutates process-global BLAS state and restores whatever it
# observed at entry, so overlapping scopes in different threads would restore
# in the wrong order and leave the process pinned at the cap after all fits
# returned.  Every reason to cap -- a fit no wide fit has released, and a
# narrow kernel inside any fit -- is counted under one lock, and one
# registration serves them all (``_settle``): it is made on the transition
# into the cap, when the pools are at the process's own state, and undone on
# the transition out, so no scope or kernel restores a state another one
# recorded.  Wide designs are tracked per owning scope (thread-local stack)
# so that a wide fit's release of the cap ends WITH that fit: when the last
# wide scope exits while capped scopes remain, the cap is re-armed.
_scope_lock = threading.Lock()
_active_scopes = 0
_wide_scopes = 0
_narrow_kernels = 0
_registration = None
_tls = threading.local()


def _scope_stack() -> list:
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = _tls.stack = []
    return stack


# Break-even measured on the 16-core reference box (audit J.6 follow-up): the
# single-thread cap wins 7x at p=203 and 1.4x at p=834, breaks even near
# p=1500 and loses 1.6x by p=2500. Designs at or past break-even release the
# cap for the remainder of the fit.
_WIDE_DESIGN_THRESHOLD = 1_500


def _auto_policy() -> bool:
    """True when the automatic policy governs, including unparseable values.

    An unparseable environment value already falls back to the automatic cap
    in :func:`_resolve_limit`, so widening must stay active for it too —
    otherwise a typo would yield the cap without its wide-design release.
    """
    raw = os.environ.get(_ENV_VAR, "auto").strip().lower()
    if raw in ("", "auto"):
        return True
    if raw in ("native", "off", "none", "false"):
        return False
    try:
        int(raw)
    except ValueError:
        return True
    return False


def _settle(limit: int, *, narrow: bool = False) -> None:
    """Hold the cap exactly while something wants it; call with ``_scope_lock`` held.

    The cap is wanted while a fit runs and no wide fit has released it, or
    while a narrow kernel runs.  The registration is made only on the
    transition into the cap, when no other registration exists and the pools
    are at the process's own state, which it therefore records; it is undone
    only on the transition out.  A narrow kernel's transition goes through
    the cached controller (``_narrow_controller``), every other one through
    ``threadpool_limits``.
    """
    global _registration
    wanted = (_active_scopes > 0 and _wide_scopes == 0) or _narrow_kernels > 0
    if wanted and _registration is None:
        if narrow:
            _registration = _narrow_controller().limit(limits=limit, user_api="blas")
        else:
            from threadpoolctl import threadpool_limits

            _registration = threadpool_limits(limits=limit, user_api="blas")
    elif not wanted and _registration is not None:
        _registration.unregister()
        _registration = None


def _release_current_scope() -> None:
    global _wide_scopes
    stack = _scope_stack()
    scope = stack[-1] if stack else None
    if scope is None:
        return
    with _scope_lock:
        if not scope["wide"]:
            scope["wide"] = True
            _wide_scopes += 1
        _settle(1)


def allow_wide_design(p: int) -> None:
    """Release the automatic cap while a wide design's fit is active.

    Called once per fit as soon as the design width is known.  Only the
    automatic policy widens; an explicit integer cap from the environment is
    respected as given.  Under concurrent fits the widest active design wins
    for the overlap — but only for the overlap: the release is owned by the
    calling fit's scope, and when the last wide scope exits the cap is
    re-armed for any narrow fits still running.
    """
    if p < _WIDE_DESIGN_THRESHOLD or not _auto_policy():
        return
    _release_current_scope()


def keep_narrow_cap(width: int) -> None:
    """Re-arm the automatic cap in a wide fit whose dense kernels all act on ``width`` (perf F7).

    ``allow_wide_design`` releases the cap on the design width, before any
    route is chosen.  The fs leaf route never factorizes a matrix wider than
    its border, so below the break-even its fit keeps the single thread:
    faster for those kernels, and its result then does not depend on the BLAS
    thread count (design §14 T4).  A nested chain is not re-capped: its parent
    levels' blocks are wider than its border, and pg17_E_discrete ran 1.5x
    master's time on default threads with the cap re-armed.  Called once the
    route is known; the fit's scope then ends as a narrow one.  While another
    fit's wide scope stays open the pools stay released for the overlap, as
    ``allow_wide_design`` says, and this fit's narrow kernels still take one
    thread (``narrow_kernel_blas_threads``).
    """
    global _wide_scopes
    if width >= _WIDE_DESIGN_THRESHOLD or not _auto_policy():
        return
    stack = _scope_stack()
    scope = stack[-1] if stack else None
    limit = _resolve_limit()
    if scope is None or not scope["wide"] or limit is None:
        return
    with _scope_lock:
        scope["wide"] = False
        _wide_scopes -= 1
        _settle(limit)


@contextmanager
def narrow_kernel_blas_threads(width: int):
    """Re-cap BLAS to one thread around dense kernels of ``width`` below the break-even.

    A wide design releases the automatic cap for its whole fit
    (``allow_wide_design``), but the dense LAPACK kernels of a structured
    factor's border act on the border width, not the design's; below the
    same measured break-even the single thread wins for them too (a border
    of 1,045 columns inside a 16,851-column fit: threaded ``potrf`` 40-110
    ms against 10 ms).  Only the automatic policy caps, and only inside a
    fit; an explicit integer or ``native``, a kernel at or above the
    break-even, or a call outside a fit is left alone.  The kernels then run
    on one thread whatever the pool, so their results do not depend on the
    thread count, including while another thread's wide fit has released
    the pools.  Kernels are counted (``_settle``): overlapping kernels in
    different threads keep the cap until the last one exits, and the pools
    come back to the state before the first.
    """
    global _narrow_kernels
    if width >= _WIDE_DESIGN_THRESHOLD or not _scope_stack() or not _auto_policy():
        yield
        return
    with _scope_lock:
        _narrow_kernels += 1
        try:
            _settle(1, narrow=True)
        except BaseException:
            _narrow_kernels -= 1
            raise
    try:
        yield
    finally:
        with _scope_lock:
            _narrow_kernels -= 1
            _settle(1)


_NARROW_CONTROLLER = None


def _narrow_controller():
    """One ``ThreadpoolController`` for every narrow re-cap (perf scout F13).

    ``threadpool_limits`` scans the loaded libraries on every entry (0.4 ms);
    a controller created once, after numpy's and scipy's BLAS have loaded (a
    wide fit is running when this is first called), limits in microseconds.
    Owner: this module; lifetime: the process; invalidation: none (the pools it
    found stay loaded).
    """
    global _NARROW_CONTROLLER
    if _NARROW_CONTROLLER is None:
        from threadpoolctl import ThreadpoolController

        _NARROW_CONTROLLER = ThreadpoolController()
    return _NARROW_CONTROLLER


def _resolve_limit() -> int | None:
    """Thread cap for solver BLAS calls, or None to leave BLAS untouched."""
    raw = os.environ.get(_ENV_VAR, "auto").strip().lower()
    if raw in ("", "auto"):
        return 1
    if raw in ("native", "off", "none", "false"):
        return None
    try:
        value = int(raw)
    except ValueError:
        warnings.warn(
            f"{_ENV_VAR}={raw!r} is not an integer, 'auto', or 'native'; "
            "applying the default single-thread cap",
            stacklevel=3,
        )
        return 1
    return value if value > 0 else None


@contextmanager
def solver_blas_threads():
    """Cap BLAS pools for the duration of a fit, restoring them on exit.

    Safe under concurrent fits: nested or overlapping scopes share one
    registration, the native BLAS configuration is restored when the last
    scope exits, and a wide design's cap release (``allow_wide_design``)
    ends when its owning scope does — remaining narrow fits are re-capped.
    """
    global _active_scopes, _wide_scopes
    limit = _resolve_limit()
    if limit is None:
        yield
        return
    scope = {"wide": False}
    stack = _scope_stack()
    entered = False
    try:
        with _scope_lock:
            _active_scopes += 1
            entered = True
            _settle(limit)
        stack.append(scope)
        yield
    finally:
        if stack and stack[-1] is scope:
            stack.pop()
        if entered:
            with _scope_lock:
                _active_scopes -= 1
                if scope["wide"]:
                    _wide_scopes -= 1
                _settle(limit)
