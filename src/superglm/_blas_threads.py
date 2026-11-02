"""Scoped BLAS thread capping for solver work.

Threaded OpenBLAS LAPACK is much slower than single-threaded on the p^3 kernels
this library lives on (potrf/trtri/pocon/syevd at p ~ 150-1000): the
fan-out/synchronisation overhead dominates these small factorizations, while the
library's large row-space work runs in numba kernels and bincount aggregations
that never enter BLAS.  The cap is scoped to fit calls and restored afterwards,
and targets only BLAS pools (tabmat's OpenMP kernels and numba threading are
untouched).

``SUPERGLM_BLAS_THREADS`` overrides the policy: unset or ``auto`` caps BLAS to
one thread during fits; an integer caps to that many; ``native`` disables
capping.  Wide designs the structured backend can take (a RandomEffect or
FactorSmooth term) release the automatic cap once they reach the measured p³
break-even; every other design keeps it below 2,500 columns, its row passes
running on the worker pool (``_parallel``).
"""

from __future__ import annotations

import os
import threading
import warnings
from contextlib import contextmanager

_ENV_VAR = "SUPERGLM_BLAS_THREADS"

# threadpool_limits mutates process-global BLAS state and restores what it saw at
# entry, so overlapping scopes in different threads would restore in the wrong
# order.  Every reason to cap (a fit no wide fit has released, a narrow kernel) is
# counted under one lock and one registration serves them all (``_settle``), made
# on the transition into the cap and undone on the transition out.  Wide designs
# are tracked per owning scope so a wide fit's release ends with that fit.
_scope_lock = threading.Lock()
_active_scopes = 0
_wide_scopes = 0
_narrow_kernels = 0
# Pooled sections (``pooled_blas_threads``) force one thread under every
# policy, so the registration records the limit it was made at and is
# re-made when the wanted limit changes.
_pooled_sections = 0
_registration = None
_registration_limit: int | None = None
_tls = threading.local()


def _scope_stack() -> list:
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = _tls.stack = []
    return stack


# Break-even measured on the 16-core reference box: the single-thread cap wins
# at small p and loses by p=2500.  Designs at or past break-even release the cap
# for the rest of the fit if the structured backend can take them
# (``allow_wide_design``).
_WIDE_DESIGN_THRESHOLD = 1_500
# A dense design (no structured term) keeps the cap below this width; at or
# above it the release applies until a complete fit there shows otherwise.
_DENSE_WIDE_DESIGN_THRESHOLD = 2_500


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

    The cap is wanted while a fit runs and no wide fit has released it, or while a
    narrow kernel runs.  The registration is made only on the transition into the
    cap, when the pools are at the process's own state, and undone on the transition
    out.  A narrow kernel or a pooled section wants one thread; otherwise a fit
    wants ``limit``.  Under an explicit integer policy a pooled section can want a
    different limit from the one registered; the registration is then undone and
    made again at the wanted limit.
    """
    global _registration, _registration_limit
    if _pooled_sections > 0 or _narrow_kernels > 0:
        wanted: int | None = 1
    elif _active_scopes > 0 and _wide_scopes == 0:
        wanted = limit
    else:
        wanted = None
    if _registration is not None and _registration_limit != wanted:
        _registration.unregister()
        _registration = None
        _registration_limit = None
    if wanted is not None and _registration is None:
        if narrow:
            _registration = _narrow_controller().limit(limits=wanted, user_api="blas")
        else:
            from threadpoolctl import threadpool_limits

            _registration = threadpool_limits(limits=wanted, user_api="blas")
        _registration_limit = wanted


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


def allow_wide_design(p: int, *, structured: bool = True) -> None:
    """Release the automatic cap while a wide design's fit is active.

    Called once per fit as soon as the design width is known.  Only the automatic
    policy widens; an explicit integer cap from the environment is respected.  Under
    concurrent fits the widest active design wins for the overlap only: the release
    is owned by the calling fit's scope, and when the last wide scope exits the cap
    is re-armed for any narrow fits still running.

    ``structured`` is False for a design the structured backend cannot take: it
    keeps the cap below ``_DENSE_WIDE_DESIGN_THRESHOLD`` because its row passes run
    on the worker pool and its dense kernels at those widths are faster on one
    thread (a pass on the fit's own thread releases it,
    ``release_for_serial_row_passes``).
    """
    threshold = _WIDE_DESIGN_THRESHOLD if structured else _DENSE_WIDE_DESIGN_THRESHOLD
    if p < threshold or not _auto_policy():
        return
    _release_current_scope()


def release_for_serial_row_passes(p: int) -> None:
    """Release the automatic cap for the rest of a fit whose row passes run on one thread.

    A dense design keeps the cap below ``_DENSE_WIDE_DESIGN_THRESHOLD`` because its
    row passes run on the worker pool.  A pass on the fit's own thread (the chunked
    solved-row route in ``reml._leverage_rows``, the serial centred Grams) has
    nothing beside its n·p² products.  On a pool worker there is no fit scope and
    this does nothing.  From ``_WIDE_DESIGN_THRESHOLD`` such a fit is released for
    its remaining work.
    """
    if p < _WIDE_DESIGN_THRESHOLD or not _auto_policy():
        return
    _release_current_scope()


def keep_narrow_cap(width: int) -> None:
    """Re-arm the automatic cap in a wide fit whose dense kernels all act on ``width``.

    ``allow_wide_design`` releases the cap on the design width before any route is
    chosen.  The fs leaf route never factorizes a matrix wider than its border, so
    below the break-even its fit keeps the single thread (faster, and independent of
    the BLAS thread count).  A nested chain is not re-capped: its parent levels'
    blocks are wider than its border.  Called once the route is known.  While
    another fit's wide scope stays open the pools stay released, and this fit's
    narrow kernels still take one thread (``narrow_kernel_blas_threads``).
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

    A wide design releases the cap for its whole fit (``allow_wide_design``), but the
    dense LAPACK kernels of a structured factor's border act on the border width;
    below the same break-even the single thread wins for them too.  Only the
    automatic policy caps, and only inside a fit; an explicit integer or ``native``,
    a kernel at or above the break-even, or a call outside a fit is left alone.
    Results do not depend on the thread count.  Kernels are counted (``_settle``):
    overlapping kernels keep the cap until the last exits.
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


@contextmanager
def pooled_blas_threads():
    """Hold BLAS at one thread while a worker pool runs, under every policy.

    A pooled kernel (``_parallel``) is the one parallelism level: its
    workers call LAPACK concurrently, so BLAS's own threads would only
    oversubscribe the cores, and its result must not depend on the worker
    count, which needs every task to run the same single-threaded arithmetic
    whether one worker or many run it.  Unlike ``narrow_kernel_blas_threads``
    this caps inside or outside a fit, at any width, and under an explicit
    ``SUPERGLM_BLAS_THREADS`` or ``native`` policy too, for the duration of
    the section only; the pools return to the fit's (or the process's) state
    when the last overlapping section exits (``_settle``).
    """
    global _pooled_sections
    with _scope_lock:
        _pooled_sections += 1
        try:
            _settle(1, narrow=True)
        except BaseException:
            _pooled_sections -= 1
            raise
    try:
        yield
    finally:
        with _scope_lock:
            _pooled_sections -= 1
            _settle(_resolve_limit() or 1)


_NARROW_CONTROLLER = None


def _narrow_controller():
    """One ``ThreadpoolController`` for every narrow re-cap.

    ``threadpool_limits`` scans the loaded libraries on every entry (0.4 ms); a
    controller created once, after numpy's and scipy's BLAS have loaded, limits in
    microseconds.  Owner: this module; lifetime: the process; invalidation: none.
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
