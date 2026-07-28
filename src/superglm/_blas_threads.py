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
capping and leaves the user's BLAS configuration alone.
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
# returned.  A refcount keeps exactly one registration alive: the first
# entrant records the true native state and the last exit restores it.
_scope_lock = threading.Lock()
_active_scopes = 0
_registration = None


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


def allow_wide_design(p: int) -> None:
    """Release the automatic cap for the rest of the fit on a wide design.

    Called once per fit as soon as the design width is known.  Only the
    automatic policy widens; an explicit integer cap from the environment is
    respected as given.  Under concurrent fits the widest active design wins
    for the overlap, trading a transient small-p slowdown for never
    throttling a genuinely wide factorization.
    """
    global _registration
    if p < _WIDE_DESIGN_THRESHOLD or not _auto_policy():
        return
    with _scope_lock:
        if _registration is not None:
            _registration.unregister()
            _registration = None


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
    registration, and the native BLAS configuration is restored when the
    last scope exits.
    """
    global _active_scopes, _registration
    limit = _resolve_limit()
    if limit is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    with _scope_lock:
        _active_scopes += 1
        if _active_scopes == 1:
            _registration = threadpool_limits(limits=limit, user_api="blas")
    try:
        yield
    finally:
        with _scope_lock:
            _active_scopes -= 1
            if _active_scopes == 0 and _registration is not None:
                _registration.unregister()
                _registration = None
