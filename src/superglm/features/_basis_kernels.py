"""Dense spline-basis rows for prediction, from compiled kernels that release the GIL.

A prediction design needs the basis rows themselves, dense and short (``K`` columns), so the
detour through a SciPy sparse matrix and back costs more than the arithmetic: SciPy builds
index arrays and a CSR for a row block that is then ``toarray``-ed. These kernels write the
rows straight into the output, one row at a time, so row blocks run on threads
(``_parallel.pooled_map``) without the interpreter serialising them.

Cox and de Boor's recurrence in Dierckx's order (1993, *Curve and Surface Fitting with Splines*,
routine ``fpbspl``) gives the ``degree + 1`` nonzero B-spline values of a row: the operations of
SciPy's ``BSpline.design_matrix`` in its order, so a row agrees with it to rounding (to the bit
where the platform's compiler does not fuse a multiply and an add, as clang does on ARM64).
The cardinal cubic regression basis is the natural-spline form of Wood (2017, *Generalized
Additive Models*, 2nd ed., section 5.3.1), agreeing with its NumPy evaluation to a few ulp.
``dense_basis`` serves every ``transform`` of a spline term, so a ``SuperGLM`` fit reads these
kernels too, from 8,192 rows (its canonicalization calls ``transform``): its designs and
a cardinal term's column means agree with the SciPy and NumPy ones to those few ulp. A
``SuperLSS`` fit builds its design another way and never reaches them.

A kernel compiles, or loads from Numba's cache, on the first call that reaches it, which costs
more than a call below a design block gains. ``dense_basis`` and ``dense_raw_basis`` therefore
choose by the rows of the block: up to ``SCALAR_MAX_ROWS`` rows (a quote) one row at a time in
Python, with the kernel's operations in its order; below ``KERNEL_MIN_ROWS`` the SciPy and NumPy
evaluation; a full design block the kernels. None of the first two needs a compile, and all three
give the same rows.
"""

from __future__ import annotations

from bisect import bisect_right
from math import isfinite
from typing import Any

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

#: Rows from which a block reads the compiled kernels: the prediction design block
#: (``DESIGN_BLOCK_ROWS``), where a call gains more than its first compile or cache load costs
#: within a few calls.
KERNEL_MIN_ROWS = 8192
#: Rows up to which a block is evaluated in Python, one row at a time: about 2 microseconds a row
#: against about 25 for the SciPy sparse construction of any size. A whole prediction costs the
#: same either way at 16 rows and more with the loop beyond.
SCALAR_MAX_ROWS = 16


@njit(cache=True, nogil=True, error_model="numpy")
def _bspline_rows(x, knots, degree, n_basis, out):
    h = np.empty(degree + 2)
    previous = np.empty(degree + 1)
    for i in range(x.shape[0]):
        xi = x[i]
        if not np.isfinite(xi):
            # A degree-0 row does no arithmetic with its point, so a NaN would read as a finite
            # one-hot row: a non-finite point is a row of NaN, which sends the call to SciPy.
            for k in range(n_basis):
                out[i, k] = np.nan
            continue
        # The span with knots[mu] <= x < knots[mu + 1], clamped to the first and last interior
        # spans: the right end point and any extrapolated point read the end polynomial.
        low = degree
        high = n_basis
        while low < high:
            middle = (low + high) >> 1
            if knots[middle] <= xi:
                low = middle + 1
            else:
                high = middle
        span = low - 1
        if span < degree:
            span = degree
        if span > n_basis - 1:
            span = n_basis - 1
        # Dierckx's ``fpbspl`` order of operations, the one SciPy's ``design_matrix`` performs,
        # so a row is the same bits; a repeated knot gives a zero term, never a division by zero.
        h[0] = 1.0
        for j in range(1, degree + 1):
            for n in range(j):
                previous[n] = h[n]
            h[0] = 0.0
            for n in range(1, j + 1):
                xb = knots[span + n]
                xa = knots[span + n - j]
                if xb == xa:
                    h[n] = 0.0
                    continue
                step = previous[n - 1] / (xb - xa)
                h[n - 1] = h[n - 1] + step * (xb - xi)
                h[n] = step * (xi - xa)
        for k in range(n_basis):
            out[i, k] = 0.0
        for k in range(degree + 1):
            out[i, span - degree + k] = h[k]


@njit(cache=True, nogil=True, error_model="numpy")
def _cardinal_rows(x, knots, widths, second_derivative_map, out):
    n_knots = knots.shape[0]
    for i in range(x.shape[0]):
        xi = x[i]
        low = 0
        high = n_knots
        while low < high:
            middle = (low + high) >> 1
            if knots[middle] <= xi:
                low = middle + 1
            else:
                high = middle
        j = low - 1
        if j < 0:
            j = 0
        if j > n_knots - 2:
            j = n_knots - 2
        h = widths[j]
        t = (xi - knots[j]) / h
        a = 1.0 - t
        c1 = h * h * (a * a * a - a) / 6.0
        c2 = h * h * (t * t * t - t) / 6.0
        for k in range(n_knots):
            out[i, k] = c1 * second_derivative_map[j, k] + c2 * second_derivative_map[j + 1, k]
        out[i, j] += a
        out[i, j + 1] += t


#: The kernels a pooled section runs on its helper threads (``nogil``, never ``prange``).
_POOLED_ROW_KERNELS = (_bspline_rows, _cardinal_rows)


def bspline_dense_python(x: NDArray, knots: NDArray, degree: int) -> NDArray[np.float64]:
    """``bspline_dense`` for a few rows, in Python: the kernel's operations in its order."""
    n_basis = len(knots) - degree - 1
    knot_list = knots.tolist()
    points = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros((len(points), n_basis))
    for i, xi in enumerate(points.tolist()):
        if not isfinite(xi):
            out[i] = np.nan
            continue
        span = min(max(bisect_right(knot_list, xi, degree, n_basis) - 1, degree), n_basis - 1)
        h = [1.0] + [0.0] * degree
        for j in range(1, degree + 1):
            previous = h[:j]
            h[0] = 0.0
            for n in range(1, j + 1):
                xb = knot_list[span + n]
                xa = knot_list[span + n - j]
                if xb == xa:
                    h[n] = 0.0
                    continue
                step = previous[n - 1] / (xb - xa)
                h[n - 1] = h[n - 1] + step * (xb - xi)
                h[n] = step * (xi - xa)
        out[i, span - degree : span + 1] = h
    return out


def cardinal_dense_python(x: NDArray, knots: NDArray, second_derivative_map: NDArray) -> NDArray:
    """``cardinal_dense`` for a few rows, in Python: the kernel's operations in its order."""
    n_knots = len(knots)
    knot_list = knots.tolist()
    points = np.asarray(x, dtype=np.float64).ravel()
    out = np.empty((len(points), n_knots))
    for i, xi in enumerate(points.tolist()):
        j = min(max(bisect_right(knot_list, xi) - 1, 0), n_knots - 2)
        h = knot_list[j + 1] - knot_list[j]
        t = (xi - knot_list[j]) / h
        a = 1.0 - t
        c1 = h * h * (a * a * a - a) / 6.0
        c2 = h * h * (t * t * t - t) / 6.0
        out[i] = c1 * second_derivative_map[j] + c2 * second_derivative_map[j + 1]
        out[i, j] += a
        out[i, j + 1] += t
    return out


def _operand(values: Any) -> NDArray[np.float64]:
    """``values`` as the read-only C float64 array the kernels compile one signature for.

    A covariate column is writable or a read-only view (pandas Copy-on-Write, Polars), and Numba
    compiles one version of a kernel for each; every operand is passed read-only so one serves both
    and ``superglm.warmup`` has compiled it.
    """
    view = np.ascontiguousarray(values, dtype=np.float64).view()
    view.flags.writeable = False
    return view


def bspline_dense(x: NDArray, knots: NDArray, degree: int) -> NDArray[np.float64]:
    """The ``(len(x), len(knots) - degree - 1)`` B-spline basis, extended past the end knots."""
    points = _operand(np.ravel(x))
    n_basis = len(knots) - degree - 1
    out = np.empty((len(points), n_basis), dtype=np.float64)
    _bspline_rows(points, _operand(knots), degree, n_basis, out)
    return out


def cardinal_dense(x: NDArray, knots: NDArray, second_derivative_map: NDArray) -> NDArray:
    """The ``(len(x), len(knots))`` cardinal cubic regression basis."""
    points = _operand(np.ravel(x))
    knot_values = _operand(knots)
    out = np.empty((len(points), len(knot_values)), dtype=np.float64)
    _cardinal_rows(
        points,
        knot_values,
        _operand(np.diff(knot_values)),
        _operand(second_derivative_map),
        out,
    )
    return out


def _evaluates_by_scipy(spec: Any, spline_base: type, *overrides: type) -> bool:
    """Whether ``spec._basis_matrix`` is the SciPy B-spline design (``extend`` aside)."""
    method = type(spec)._basis_matrix
    if method is spline_base._basis_matrix:
        return True
    return spec.extrapolation != "extend" and any(method is cls._basis_matrix for cls in overrides)


def _refused_by_scipy(spec: Any, points: NDArray) -> bool:
    """Whether ``BSpline.design_matrix``, which does not extrapolate for ``error``, refuses ``points``.

    It refuses a point outside the base interval ``[t_k, t_n]``. Prepared ``error`` points
    lie within the training range; a custom knot vector can still have a narrower base interval.
    The kernels clamp to the end span there, so retain SciPy's refusal instead.
    """
    if spec.extrapolation != "error" or not points.size:
        return False
    knots = spec._knots
    return bool(
        points.min() < knots[spec.degree] or points.max() > knots[len(knots) - spec.degree - 1]
    )


def _evaluations(n_rows: int) -> tuple[Any, Any] | None:
    """``(bspline, cardinal)`` for a block of ``n_rows`` rows, or ``None`` for the library's own."""
    if n_rows >= KERNEL_MIN_ROWS:
        return bspline_dense, cardinal_dense
    if n_rows <= SCALAR_MAX_ROWS:
        return bspline_dense_python, cardinal_dense_python
    return None


def dense_basis(spec: Any, x: NDArray) -> NDArray:
    """``spec._basis_matrix(x).toarray()``, from the kernels or the Python rows where the spec allows."""
    from superglm.features import _spline_runtime
    from superglm.features.spline import (
        CardinalCRSpline,
        CubicRegressionSpline,
        NaturalSpline,
        _SplineBase,
    )

    evaluations = _evaluations(np.size(x))
    if getattr(spec, "_scop_Sigma", None) is not None or evaluations is None:
        return spec._basis_matrix(x).toarray()  # a saved SCOP term keeps its exact route
    bspline, cardinal = evaluations
    if type(spec) is CardinalCRSpline and spec.extrapolation != "extend":
        x_eval, _ = _spline_runtime.prepare_eval_points(spec, x)
        rows = cardinal(x_eval, spec._cr_knots, spec._cr_M)
    elif isinstance(spec, _SplineBase) and _evaluates_by_scipy(
        spec, _SplineBase, CubicRegressionSpline, NaturalSpline
    ):
        x_eval, _ = _spline_runtime.prepare_eval_points(spec, x)
        if _refused_by_scipy(spec, x_eval):
            return spec._basis_matrix(x).toarray()
        rows = bspline(x_eval, spec._knots, spec.degree)
    else:
        return spec._basis_matrix(x).toarray()
    # A non-finite row (a NaN or infinite covariate, a degenerate cardinal knot vector) is SciPy's and
    # NumPy's to answer: SciPy raises for such a point, as it does at every call size.
    return rows if np.isfinite(rows).all() else spec._basis_matrix(x).toarray()


def dense_raw_basis(evaluator: Any, x: NDArray) -> NDArray:
    """``evaluator(x)``, read from the kernels only where it is the spec's stock raw basis."""
    from superglm.features.spline import CardinalCRSpline, _SplineBase

    evaluations = _evaluations(np.size(x))
    if evaluations is None:
        return np.asarray(evaluator(x), dtype=np.float64)
    bspline, cardinal = evaluations
    spec = getattr(evaluator, "__self__", None)
    method = getattr(evaluator, "__func__", None)
    rows = None
    if type(spec) is CardinalCRSpline and method is CardinalCRSpline._raw_basis_matrix:
        rows = cardinal(np.clip(x, spec._lo, spec._hi), spec._cr_knots, spec._cr_M)
    elif isinstance(spec, _SplineBase) and method is _SplineBase._raw_basis_matrix:
        rows = bspline(np.clip(x, spec._lo, spec._hi), spec._knots, spec.degree)
    if rows is not None and np.isfinite(rows).all():
        return rows
    return np.asarray(evaluator(x), dtype=np.float64)


def _warmup_basis_kernels() -> None:
    """Compile both kernels (``superglm.warmup``), so a service's first quote does not."""
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    bspline_dense(np.array([0.25, 0.75]), knots, 3)
    cardinal_dense(np.array([0.25, 0.75]), np.array([0.0, 0.5, 1.0]), np.zeros((3, 3)))
