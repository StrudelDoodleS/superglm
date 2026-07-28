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
from contextlib import contextmanager

_ENV_VAR = "SUPERGLM_BLAS_THREADS"


def _resolve_limit() -> int | None:
    """Thread cap for solver BLAS calls, or None to leave BLAS untouched."""
    raw = os.environ.get(_ENV_VAR, "auto").strip().lower()
    if raw in ("", "auto"):
        return 1
    if raw == "native":
        return None
    try:
        value = int(raw)
    except ValueError:
        return 1
    return value if value > 0 else None


@contextmanager
def solver_blas_threads():
    """Cap BLAS pools for the duration of a fit, restoring them on exit."""
    limit = _resolve_limit()
    if limit is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=limit, user_api="blas"):
        yield
