"""FactorSmooth ``fs`` block leaves on the one engine (one-engine design §3.4, §3.6, §3.12).

An ``fs`` term is a depth-1 block tree under the super-root: level ``l`` is a
leaf with ``k`` coefficients (its basis rows ``z_r``), the intercept is the
super-root and every other column is the border.  Every border column that is
not one-hot is centred on the §3.2 prior-weighted centre ``c0``
(``factor_smooth_prior_statistics``); the factor works in the centred
coordinates ``[1, X - 1 c0']`` and maps its public results to raw ones with
``R = I - e_0 c'`` (as ``NestedSchurFactor``).

**Per PIRLS iterate** (``build_factor_smooth_leaf_system``), one pass over the
rows in level order, in fixed chunks and a fixed order (``_leaf_pass``):
each level's rows ``g_r = sqrt|w_r| [z_r, 1, x_r - c0, zeta_r]`` (``zeta`` the
working response, so the right-hand side travels inside the factorization)
are reduced by Householder QR, a level cut by a chunk edge merging its pieces
by one more QR of the stacked triangles (TSQR).  Fisher rows (``w >= 0`` by
type) keep the triangle ``R_l``'s leading ``k`` rows; its trailing rows, the
within-level residual of the border on the level basis, go straight into the
border Gram ``W_r``.  A level is closed into that compact form as the pass
leaves it, so no ``(K, p, p)`` stack of triangles is formed
(``FactorSmoothLeafData``).
Signed rows (observed curvature, by type) follow Chandrasekaran, Gu and Sayed
as Bojanczyk, Higham and Patel (2003, SIMAX 24) state it: the middle factor
``M_l = Q_l' J Q_l`` (a merge maps it by the merge's orthogonal factor), its
eigendecomposition ``M_l = V Lambda V'``, and the signed pseudo-rows ``B_l =
|Lambda|^1/2 V' R_l`` with signature ``sign Lambda``, so ``B_l' J B_l = R_l'
M_l R_l = G_l' J G_l`` with every error sitting between the triangular factors.

**Per lambda trial** (``FactorSmoothLeafFactor``), one small QR per level with
the penalty's square root inside, never a subtraction of data-sized moments:
Fisher ``QR([[R_l,top], [sqrt(P_l), 0]]) = [[U_l, V_l], [0, T_l]]``, so ``D_l =
U_l'U_l``, ``F_l = U_l^-1 V_l`` and the level's border piece is ``T_l'T_l``;
signed ``QR([[B_l], [sqrt(P_l), 0]])`` with the middle factor ``M~`` of the
stacked signature, ``U_l = chol(M~_zz)' R~_zz`` (a failed Cholesky is the tree
pivot certificate refusing the iterate: ``D_l`` is not positive definite), the
border piece ``R~_bb' (M~ / M~_zz) R~_bb``, a Schur complement inside the
bounded middle factor.  The intercept-and-border Gram ``Q_x = W_r + sum_l
T_l'T_l`` (Fisher) or ``sum_l R~_bb'(M~/M~_zz)R~_bb`` (signed) then loses the
super-root (``D_0 = Q_x[0, 0]``, ``c* = Q_x[1:, 0] / D_0``), and the
intercept-profiled rest goes to ``border.factor_border``: structural deflation
(the complete random-effect blocks' sums), complete pivoting, Rump's
verification and disclosure (§3.6).  ``log|H| = sum_l log|D_l| + log D_0 + log
pdet(Q_rest)``.

**The border bound** (``_border_bound``).  Every rounding of the route is of
one of two kinds.  Row-level: the rows of a QR perturbed columnwise,
``||dA_m|| <= g ||A_m||`` (Householder QR, Higham 2002 Theorem 19.4 with its
constant taken as 2; the rows' own formation; the pseudo-rows), so projected on
``v_j = [-F_j; e_j]`` it is at most ``eta_j = g |v_j|' a``; sandwiched: an
error of a middle factor or of a product between the triangular factors, at
most ``sigma ||R v_i|| ||R v_j||``.  With ``r_j`` the augmented unsigned
residual ``||A v_j||`` and the rows' weight-error scale ``e_r`` (``r^e``),

    |dQ_ij| <= sum_l [r_i eta_j + eta_i r_j + (1 + kappa) eta_i eta_j
                      + sigma rbar_i rbar_j + g_16 r^e_i r^e_j + second order]

to second order, ``kappa = ||M~_zz^-1||`` (1 for Fisher rows) and ``rbar = r +
eta``.  It is majorised by ``sqrt(U_i U_j)`` with ``U_j = sum_l [t r_j^2 +
eta_j^2 / t + ...]`` (Cauchy-Schwarz with one global ``t``, chosen as
``sqrt(nu / rho)`` on the Jacobi scale, where the Perron-Frobenius sum ``sum_j
U_j / Q_jj`` equals the normwise ``2 sqrt(rho nu)``), the form
``factor_border`` takes.  A Gram-level majorant ``eps a a'`` would charge a
border column in (or near) a level's span, such as the intercept or a
level-constant attribute, its whole mass against a Schur piece only the
penalty pins: measured on the stage-2 entry gate it gave ``u_s`` up to 1.6e5
where this bound gives 1.5e-3, with every measured error within it.

**Discrete terms** read their basis rows from the support table
(``B_unique @ natural_map``) in the same row pass.

Caches: ``FactorSmoothLeafFactor`` forms ``Q^+`` in raw coordinates on first
read (owner: the factor; lifetime: the factor; invalidation: none, since every
input is fixed at construction); its ``(K, k, q)``-sized products
(``_REBUILT_ON_USE``) are dropped from a pickle and formed again on first use.
The leaf data of an iterate belong to its system and are never modified.  The
lineage's leaf memo (``build_factor_smooth_leaf_system``) holds the last
system until the fit publishes its state (``release_leaf_memo``); the
published state holds copies of the terminal system and factor
(``published_leaf_system``), whose leaf keeps what a factor build reads and no
way back to the rows.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from superglm._blas_threads import narrow_kernel_blas_threads
from superglm._numba_compile import collect_after_compile
from superglm.solvers._structured.border import (
    BorderCertificate,
    BorderGenerators,
    factor_border,
)
from superglm.solvers._structured.factors import _block_derivative_cross_traces
from superglm.solvers._structured.layout import FactorSmoothLeafLayout
from superglm.solvers._structured.leaf_kernels import _shifted_sums
from superglm.solvers._structured.moments import _border_generators, _leaf_rows
from superglm.solvers._structured.nested import _super_root_inverse
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    _BlockDiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _multiply_symmetric_bdlr,
    _operator_bdlr,
    _trace_general_bdlr_product,
    _trace_symmetric_bdlr,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers._structured.factors import DerivativeDirection

# IEEE binary64 unit roundoff (Higham 2002, section 2.1) and gamma_n (section 3.1).
_UNIT = 2.0**-53
_EPS = float(np.finfo(np.float64).eps)
_CHUNK = 8192
# LLVM flags of the dense kernels: reassociation lets a dot product vectorize and
# contraction fuses multiply-adds.  Neither drops NaN or infinity semantics, the
# compiled code is fixed (so a result does not depend on a thread count), and any
# summation order keeps the dot product's gamma_n |a|'|b| bound (Higham 2002, §3.1).
_KERNEL_MATH = {"reassoc", "contract"}


def _gamma(count: float) -> float:
    return count * _UNIT / (1.0 - count * _UNIT)


def _gamma_array(count: NDArray) -> NDArray:
    return count * _UNIT / (1.0 - count * _UNIT)


# ---------------------------------------------------------------- kernels --
# LAPACK's Householder QR (dgeqrf) and its orthogonal factor (dorgqr), called from
# numba through scipy's shipped LAPACK (scipy.linalg.cython_lapack) by symbol name,
# so the compiled kernels cache.  No compiled code of our own.
def _register_lapack() -> None:
    import llvmlite.binding as llvm
    from numba.extending import get_cython_function_address

    for name in ("dgeqrf", "dorgqr"):
        llvm.add_symbol(
            f"superglm_{name}", get_cython_function_address("scipy.linalg.cython_lapack", name)
        )


_register_lapack()
_INT = numba.types.CPointer(numba.types.int32)
_DOUBLE = numba.types.CPointer(numba.types.float64)
_dgeqrf = numba.types.ExternalFunction(
    "superglm_dgeqrf", numba.types.void(_INT, _INT, _DOUBLE, _INT, _DOUBLE, _DOUBLE, _INT, _INT)
)
_dorgqr = numba.types.ExternalFunction(
    "superglm_dorgqr",
    numba.types.void(_INT, _INT, _INT, _DOUBLE, _INT, _DOUBLE, _DOUBLE, _INT, _INT),
)


@numba.njit(cache=True, fastmath=_KERNEL_MATH)
def _mm(A, B):  # pragma: no cover - compiled
    """``A @ B`` by fixed-order loops (no BLAS: the result does not depend on a thread count)."""
    m, inner = A.shape
    n = B.shape[1]
    out = np.zeros((m, n))
    for i in range(m):
        for t in range(inner):
            a = A[i, t]
            if a == 0.0:
                continue
            for j in range(n):
                out[i, j] += a * B[t, j]
    return out


@numba.njit(cache=True)
def _house_qr(G, tau, work, ints):  # pragma: no cover - compiled
    """LAPACK ``dgeqrf`` of the column-major ``G`` (m x p, an ``(p, m)`` C array's transpose).

    The triangle ends in the upper part and reflector ``j`` below the diagonal
    of column ``j``, as ``dgeqrf`` leaves them (Householder QR, Higham 2002
    Theorem 19.4).  ``work`` (at least ``p`` entries) and ``ints`` (5 int32)
    are the caller's workspace.  Called under one BLAS thread (``_leaf_pass``
    and the factor's construction), so the result does not depend on a thread
    count.
    """
    m, p = G.shape
    At = G.T
    ints[0] = m
    ints[1] = p
    ints[2] = At.shape[1]
    ints[3] = len(work)
    ints[4] = 0
    _dgeqrf(
        ints[0:1].ctypes,
        ints[1:2].ctypes,
        At.ctypes,
        ints[2:3].ctypes,
        tau.ctypes,
        work.ctypes,
        ints[3:4].ctypes,
        ints[4:5].ctypes,
    )


@numba.njit(cache=True)
def _house_q(G, tau, Q, work, ints):  # pragma: no cover - compiled
    """``Q = H_0 ... H_{r-1} [I_r; 0]`` (m x r) from ``_house_qr``'s reflectors, by LAPACK ``dorgqr``.

    ``Q`` is column-major (an ``(r, m)`` C array's transpose).
    """
    m, p = G.shape
    r = min(m, p)
    Qt = Q.T
    At = G.T
    for c in range(r):
        for i in range(m):
            Qt[c, i] = At[c, i]
    ints[0] = m
    ints[1] = r
    ints[2] = r
    ints[3] = Qt.shape[1]
    ints[4] = len(work)
    ints[5] = 0
    _dorgqr(
        ints[0:1].ctypes,
        ints[1:2].ctypes,
        ints[2:3].ctypes,
        Qt.ctypes,
        ints[3:4].ctypes,
        tau.ctypes,
        work.ctypes,
        ints[4:5].ctypes,
        ints[5:6].ctypes,
    )


@numba.njit(cache=True)
def _upper_of(G, R):  # pragma: no cover - compiled
    """Write the ``min(m, p) x p`` upper triangle of ``G`` into ``R`` (p x p), zeros elsewhere."""
    m, p = G.shape
    for i in range(p):
        for j in range(p):
            R[i, j] = G[i, j] if (i < m and j >= i) else 0.0


@numba.njit(cache=True)
def _merge_level(R_acc, M_acc, R_new, M_new, signed):  # pragma: no cover - compiled
    """TSQR merge of two pieces of one level: ``QR([R_acc; R_new])``, and ``M = Q_a'M_a Q_a + Q_b'M_b Q_b``."""
    p = R_acc.shape[0]
    # column-major work arrays: the Householder loops run down columns
    stacked = np.zeros((p, 2 * p)).T
    for i in range(p):
        for j in range(p):
            stacked[i, j] = R_acc[i, j]
            stacked[p + i, j] = R_new[i, j]
    tau = np.zeros(p)
    work = np.empty(64 * (p + 1))
    ints = np.empty(6, np.int32)
    _house_qr(stacked, tau, work, ints)
    if signed:
        Q = np.zeros((p, 2 * p)).T
        _house_q(stacked, tau, Q, work, ints)
        merged = np.zeros((p, p))
        for part in range(2):
            source = M_acc if part == 0 else M_new
            block = Q[part * p : (part + 1) * p, :]
            middle = _mm(source, block)
            merged += _mm(block.T, middle)
        for i in range(p):
            for j in range(p):
                M_acc[i, j] = 0.5 * (merged[i, j] + merged[j, i])
    _upper_of(stacked, R_acc)


@numba.njit(cache=True)
def _leaf_segments(
    rows, weights, error, levels, R_acc, M_acc, E_acc, started, counts, merges, signed
):  # pragma: no cover - compiled
    """Fold one chunk of level-ordered rows into the per-level triangles (``_leaf_pass``).

    ``rows`` (m, p) unweighted ``[z, 1, x - c0, zeta]``; a row of zero weight
    joins only the signed rows' error Gram.  Fixed order, no BLAS.
    """
    m, p = rows.shape
    work = np.empty(64 * (p + 1))
    ints = np.empty(6, np.int32)
    start = 0
    while start < m:
        level = levels[start]
        stop = start
        while stop < m and levels[stop] == level:
            stop += 1
        live = 0
        for r in range(start, stop):
            if weights[r] != 0.0:
                live += 1
        if signed:
            gram = E_acc[level]
            for r in range(start, stop):
                e = error[r]
                if e == 0.0:
                    continue
                for i in range(p - 1):
                    value = e * rows[r, i]
                    for j in range(p - 1):
                        gram[i, j] += value * rows[r, j]
        if live:
            # column-major (live x p): the Householder loops run down columns
            G = np.empty((p, live)).T
            sign = np.empty(live)
            position = 0
            for r in range(start, stop):
                w = weights[r]
                if w == 0.0:
                    continue
                root = math.sqrt(abs(w))
                for j in range(p):
                    G[position, j] = root * rows[r, j]
                sign[position] = 1.0 if w > 0.0 else -1.0
                position += 1
            tau = np.zeros(min(live, p))
            _house_qr(G, tau, work, ints)
            R_new = np.zeros((p, p))
            _upper_of(G, R_new)
            M_new = np.eye(p)
            if signed:
                width = min(live, p)
                Q = np.zeros((width, live)).T
                _house_q(G, tau, Q, work, ints)
                for i in range(width):
                    for j in range(width):
                        s = 0.0
                        for r in range(live):
                            s += Q[r, i] * sign[r] * Q[r, j]
                        M_new[i, j] = s
                for i in range(width):
                    for j in range(i):
                        value = 0.5 * (M_new[i, j] + M_new[j, i])
                        M_new[i, j] = value
                        M_new[j, i] = value
            if started[level]:
                # M_acc is a (1, 1, 1) placeholder for Fisher rows: never index it by level
                _merge_level(
                    R_acc[level], M_acc[level] if signed else M_acc[0], R_new, M_new, signed
                )
                merges[level] += 1
            else:
                for i in range(p):
                    for j in range(p):
                        R_acc[level, i, j] = R_new[i, j]
                        if signed:
                            M_acc[level, i, j] = M_new[i, j]
                started[level] = True
            counts[level] += live
        start = stop


@numba.njit(cache=True)
def _close_level(
    R, level, k, top, tail_norms2, tail_gram, with_gram, tail_root
):  # pragma: no cover
    """Fold one level's finished triangle ``R`` ``(p, p)`` into the compact leaf data.

    ``top[level]`` its leading ``k`` rows; ``tail_norms2[level]`` the squared
    column norms of its trailing rows ``T = R[k:, k:]`` (upper triangular);
    ``tail_gram += T'T`` when ``with_gram``, each entry once and mirrored, so
    it stays exactly symmetric.  A non-empty ``tail_root`` ``(p - k, p - k)``
    becomes ``R`` of ``QR([tail_root; T])`` (a TSQR merge, ``_merge_level``):
    over the pass, in level order, the triangular factor of every level's
    trailing rows stacked, which an ``sz`` term's thin-level aliases and data
    estimability read (#432; no level's triangle is kept for it).  Fixed
    order; the merge's LAPACK runs on the pass's own thread.
    """
    p = R.shape[0]
    w = p - k
    for i in range(k):
        for j in range(p):
            top[level, i, j] = R[i, j]
    for j in range(w):
        s = 0.0
        for a in range(j + 1):
            value = R[k + a, k + j]
            s += value * value
        tail_norms2[level, j] = s
    if with_gram:
        for i in range(w):
            for j in range(i, w):
                s = 0.0
                for a in range(i + 1):
                    s += R[k + a, k + i] * R[k + a, k + j]
                tail_gram[i, j] += s
                if j != i:
                    tail_gram[j, i] += s
    if tail_root.shape[0]:
        trailing = np.zeros((w, w))
        for i in range(w):
            for j in range(i, w):
                trailing[i, j] = R[k + i, k + j]
        no_middle = np.zeros((1, 1))
        _merge_level(tail_root, no_middle, trailing, no_middle, False)


@numba.njit(cache=True)
def _close_levels(R_acc, k, top, tail_norms2, tail_gram, with_gram):  # pragma: no cover
    """``_close_level`` over every level of ``R_acc`` ``(K, p, p)``, in level order."""
    no_root = np.zeros((0, 0))
    for level in range(R_acc.shape[0]):
        _close_level(R_acc[level], level, k, top, tail_norms2, tail_gram, with_gram, no_root)


@numba.njit(cache=True)
def _fisher_leaf_segments(
    rows,
    weights,
    levels,
    R_open,
    open_state,
    top,
    tail_norms2,
    tail_gram,
    counts,
    merges,
    k,
    tail_root,
):  # pragma: no cover - compiled
    """``_leaf_segments`` for Fisher rows, holding only the open level's triangle.

    The rows come in level order, so a level is finished once the pass
    leaves it: its triangle ``R_open`` is then closed into the compact leaf
    data (``_close_level``, merging ``tail_root``) and the next level opens.
    ``open_state`` is ``[level, started]`` across chunks (``level`` ``-1``
    before the first row); the caller closes the last level.  The triangle of
    each level is the one ``_leaf_segments`` forms, bit for bit (the same QR
    and merges).
    Returns ``-1``, or the row of a level out of order.
    """
    m, p = rows.shape
    work = np.empty(64 * (p + 1))
    ints = np.empty(6, np.int32)
    no_middle = np.zeros((1, 1))
    start = 0
    while start < m:
        level = levels[start]
        stop = start
        while stop < m and levels[stop] == level:
            stop += 1
        if level != open_state[0]:
            if level < open_state[0]:
                return start
            if open_state[0] >= 0:
                _close_level(R_open, open_state[0], k, top, tail_norms2, tail_gram, True, tail_root)
            open_state[0] = level
            open_state[1] = 0
            for i in range(p):
                for j in range(p):
                    R_open[i, j] = 0.0
        live = 0
        for r in range(start, stop):
            if weights[r] != 0.0:
                live += 1
        if live:
            G = np.empty((p, live)).T
            position = 0
            for r in range(start, stop):
                w = weights[r]
                if w == 0.0:
                    continue
                root = math.sqrt(abs(w))
                for j in range(p):
                    G[position, j] = root * rows[r, j]
                position += 1
            tau = np.zeros(min(live, p))
            _house_qr(G, tau, work, ints)
            R_new = np.zeros((p, p))
            _upper_of(G, R_new)
            if open_state[1]:
                _merge_level(R_open, no_middle, R_new, no_middle, False)
                merges[level] += 1
            else:
                for i in range(p):
                    for j in range(p):
                        R_open[i, j] = R_new[i, j]
                open_state[1] = 1
            counts[level] += live
        start = stop
    return -1


@numba.njit(cache=True)
def _exact_basis_rows(out, order, lo, hi, data, indices, indptr, natural_map):  # pragma: no cover
    """Rows ``order[lo:hi]`` of ``B @ natural_map`` from ``B``'s CSR arrays, stored order."""
    k = natural_map.shape[1]
    for i in range(hi - lo):
        row = order[lo + i]
        for j in range(k):
            out[i, j] = 0.0
        for t in range(indptr[row], indptr[row + 1]):
            value = data[t]
            column = indices[t]
            for j in range(k):
                out[i, j] += value * natural_map[column, j]


@numba.njit(cache=True)
def _table_basis_rows(out, order, lo, hi, bins, table):  # pragma: no cover - compiled
    """Rows ``order[lo:hi]`` of a discrete term's support table."""
    k = table.shape[1]
    for i in range(hi - lo):
        b = bins[order[lo + i]]
        for j in range(k):
            out[i, j] = table[b, j]


@numba.njit(cache=True)
def _upper_solve(U, B, out):  # pragma: no cover - compiled
    """``out = U^-1 B`` for an upper-triangular ``U`` (k x k) by back substitution."""
    k = U.shape[0]
    for c in range(B.shape[1]):
        for i in range(k - 1, -1, -1):
            s = B[i, c]
            for j in range(i + 1, k):
                s -= U[i, j] * out[j, c]
            out[i, c] = s / U[i, i]


@numba.njit(cache=True)
def _level_row_forms(U, F, levels, Z, tree_out, cross_out):  # pragma: no cover - compiled
    """Per (row, level) piece ``z``: ``tree_out = ||U_l^-T z||^2`` and ``cross_out = F_l' z``.

    ``U_l^-T z`` by forward substitution on ``U_l'`` (``D_l = U_l' U_l``), in
    a fixed order; ``F`` ``(K, k, w)``.
    """
    pieces, k = Z.shape
    width = F.shape[2]
    v = np.empty(k)
    for t in range(pieces):
        level = levels[t]
        total = 0.0
        for i in range(k):
            s = Z[t, i]
            for j in range(i):
                s -= U[level, j, i] * v[j]
            v[i] = s / U[level, i, i]
            total += v[i] * v[i]
        tree_out[t] = total
        for c in range(width):
            s = 0.0
            for i in range(k):
                s += F[level, i, c] * Z[t, i]
            cross_out[t, c] = s


@numba.njit(cache=True)
def _fisher_trials(R_acc, roots, k, Qx, U_out, F_out, Dinv_out, logdets):  # pragma: no cover
    """Per level ``QR([[R_top], [sqrt(P), 0]])``: ``U``, ``F = U^-1 V``, ``D^-1``, ``log|D|``;
    ``Qx += T'T``.  Returns ``-1`` or the first level whose pivot is not positive.
    ``R_acc`` ``(K, >= k, p)``: only each level's leading ``k`` rows are read.
    """
    K, p = R_acc.shape[0], R_acc.shape[2]
    width = p - k
    stacked = np.empty((p, 2 * k)).T
    tau = np.zeros(min(2 * k, p))
    work = np.empty(64 * (p + 1))
    ints = np.empty(6, np.int32)
    V = np.empty((k, width))
    identity = np.eye(k)
    Uinv = np.empty((k, k))
    for level in range(K):
        for i in range(k):
            for j in range(p):
                stacked[i, j] = R_acc[level, i, j]
                stacked[k + i, j] = roots[level, i, j] if j < k else 0.0
        _house_qr(stacked, tau, work, ints)
        total = 0.0
        for i in range(k):
            pivot = stacked[i, i]
            if not pivot != 0.0 or not math.isfinite(pivot):
                return level
            total += 2.0 * math.log(abs(pivot))
            for j in range(k):
                U_out[level, i, j] = stacked[i, j] if j >= i else 0.0
            for j in range(width):
                V[i, j] = stacked[i, k + j]
        logdets[level] = total
        _upper_solve(U_out[level], V, F_out[level])
        _upper_solve(U_out[level], identity, Uinv)
        Dinv_out[level] = _mm(Uinv, Uinv.T)
        rows = min(2 * k, p) - k
        for a in range(rows):
            for i in range(width):
                ti = stacked[k + a, k + i] if k + i >= k + a else 0.0
                if ti == 0.0:
                    continue
                for j in range(width):
                    if k + j >= k + a:
                        Qx[i, j] += ti * stacked[k + a, k + j]
    return -1


@numba.njit(cache=True)
def _signed_trials(
    B, signature, roots, k, Qx, U_out, F_out, Dinv_out, logdets, kappa, sigma_extra
):  # pragma: no cover - compiled
    """Per level the J-orthogonal step of §3.4 on ``[[B], [sqrt(P), 0]]``; ``Qx += R~_bb' S~ R~_bb``.

    Returns ``-1`` or the first level whose ``M~_zz`` is not positive
    definite (the tree pivot certificate).
    """
    K, p, _ = B.shape
    width = p - k
    m = p + k
    gk, gk1 = _UNIT * k / (1.0 - _UNIT * k), _UNIT * (k + 1) / (1.0 - _UNIT * (k + 1))
    g2p = _UNIT * 2 * p / (1.0 - _UNIT * 2 * p)
    stacked = np.empty((p, m)).T
    tau = np.zeros(p)
    Q = np.empty((p, m)).T
    J = np.empty(m)
    work = np.empty(64 * (p + 1))
    ints = np.empty(6, np.int32)
    for level in range(K):
        for i in range(p):
            for j in range(p):
                stacked[i, j] = B[level, i, j]
            J[i] = signature[level, i]
        for i in range(k):
            for j in range(p):
                stacked[p + i, j] = roots[level, i, j] if j < k else 0.0
            J[p + i] = 1.0
        _house_qr(stacked, tau, work, ints)
        _house_q(stacked, tau, Q, work, ints)
        M = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1):
                s = 0.0
                for r in range(m):
                    s += Q[r, i] * J[r] * Q[r, j]
                M[i, j] = s
                M[j, i] = s
        L = np.zeros((k, k))
        for j in range(k):
            s = M[j, j]
            for t in range(j):
                s -= L[j, t] * L[j, t]
            if not s > 0.0:
                return level
            L[j, j] = math.sqrt(s)
            for i in range(j + 1, k):
                s = M[i, j]
                for t in range(j):
                    s -= L[i, t] * L[j, t]
                L[i, j] = s / L[j, j]
        # N = L^-1 M_zb (forward substitution), Linv = L^-1
        N = np.zeros((k, width))
        for c in range(width):
            for i in range(k):
                s = M[i, k + c]
                for t in range(i):
                    s -= L[i, t] * N[t, c]
                N[i, c] = s / L[i, i]
        Linv = np.zeros((k, k))
        for c in range(k):
            for i in range(k):
                s = 1.0 if i == c else 0.0
                for t in range(i):
                    s -= L[i, t] * Linv[t, c]
                Linv[i, c] = s / L[i, i]
        schur = np.empty((width, width))
        for i in range(width):
            for j in range(width):
                s = M[k + i, k + j]
                for t in range(k):
                    s -= N[t, i] * N[t, j]
                schur[i, j] = s
        Rzz = np.zeros((k, k))
        Rzb = np.empty((k, width))
        Rbb = np.zeros((width, width))
        for i in range(k):
            for j in range(i, k):
                Rzz[i, j] = stacked[i, j]
            for j in range(width):
                Rzb[i, j] = stacked[i, k + j]
        for i in range(width):
            for j in range(i, width):
                Rbb[i, j] = stacked[k + i, k + j]
        piece = _mm(Rbb.T, _mm(schur, Rbb))
        for i in range(width):
            for j in range(width):
                Qx[i, j] += 0.5 * (piece[i, j] + piece[j, i])
        total = 0.0
        for i in range(k):
            if not Rzz[i, i] != 0.0 or not math.isfinite(Rzz[i, i]):
                return level
            total += 2.0 * math.log(L[i, i]) + 2.0 * math.log(abs(Rzz[i, i]))
        logdets[level] = total
        U = _mm(L.T, Rzz)
        for i in range(k):
            for j in range(k):
                U_out[level, i, j] = U[i, j] if j >= i else 0.0
        # F = R~zz^-1 (R~zb + M~zz^-1 M~zb R~bb), M~zz^-1 M~zb = L^-T N
        LtN = _mm(Linv.T, N)
        right = Rzb + _mm(LtN, Rbb)
        _upper_solve(Rzz, right, F_out[level])
        Uinv = np.empty((k, k))
        _upper_solve(U_out[level], np.eye(k), Uinv)
        Dinv_out[level] = _mm(Uinv, Uinv.T)
        # kappa >= ||M~zz^-1||_2 = ||L^-1||_2^2 <= ||L^-1||_F^2
        kf = 0.0
        for i in range(k):
            for j in range(k):
                kf += Linv[i, j] * Linv[i, j]
        kappa[level] = kf
        # the Schur step's own rounding (Higham Theorems 8.5 and 10.3), sandwiched
        absN = np.abs(N)
        dN = gk * _mm(np.abs(Linv), _mm(np.abs(L), absN))
        err = gk1 * (np.abs(M[k:, k:]) + _mm(absN.T, absN)) + 2.0 * _mm(absN.T, dN)
        ef = 0.0
        sf = 0.0
        for i in range(width):
            for j in range(width):
                ef += err[i, j] * err[i, j]
                sf += schur[i, j] * schur[i, j]
        sigma_extra[level] = math.sqrt(ef) + g2p * math.sqrt(sf) + k * gk1
    return -1


# ------------------------------------------------------------ leaf data ----
class _TrailingRowsSource:
    """An ``sz`` leaf's trailing rows' triangular factor, formed again by its own leaf pass.

    An ``sz`` leaf keeps no ``(K, p, p)`` triangles during the fit (Opus
    review P1): its data estimability reads the trailing rows only through
    their triangular factor.  A leaf whose pass did not merge it (no thin
    level: nothing reads it before publication) forms it here once, by the
    same compact pass with the merge (``_close_level``): the same rows,
    kernel and order, so the same level triangles bit for bit, and no
    ``(K, p, p)`` stack (#432; the pass formerly kept every level's triangle
    to read their trailing rows).  It holds the layout and the read-only rows
    the leaf was formed from (the lineage memo's copies).  Never pickled: a
    restored in-fit leaf has no source (``trailing_root`` then says so).
    """

    def __init__(self, layout, weights, weighted_rhs, center, error, signed: bool) -> None:
        self.layout = layout
        self.rows = (weights, weighted_rhs, center, error)
        self.signed = bool(signed)

    def root(self) -> NDArray | None:
        if self.layout is None:
            return None
        weights, weighted_rhs, center, error = self.rows
        assert weights is not None and weighted_rhs is not None and center is not None
        with narrow_kernel_blas_threads(self.layout.block_size + self.layout.width + 2):
            parts = _leaf_pass(
                self.layout,
                weights,
                weighted_rhs,
                center,
                error,
                signed=self.signed,
                compact=True,
                tail_root=True,
            )
        assert isinstance(parts, _LeafParts)
        return parts.tail_root

    def __getstate__(self) -> dict:
        return {}

    def __setstate__(self, state: dict) -> None:
        self.layout, self.rows, self.signed = None, (None, None, None, None), False


@dataclass(frozen=True, eq=False)
class FactorSmoothLeafData:
    """One iterate's leaf statistics of an ``fs`` term (module docstring).

    Each level's triangle ``R_l`` of ``sqrt|w| [z, 1, x - c0, zeta]`` (``p = k
    + q + 2``) is upper triangular, so every row below its leading ``k`` is
    zero on the level basis: ``top`` ``(K, k, p)`` holds those leading rows,
    all a per-lambda step reads; ``tail_norms2`` ``(K, q + 2)`` the squared
    column norms of the trailing rows (the border bound's residuals and
    column scales) and ``tail_gram`` ``(q + 2, q + 2)`` their Gram summed over
    the levels, which for Fisher rows is the within-level residual Gram
    ``within``.  For signed rows ``pseudo_rows`` and ``signature`` are ``B_l``
    and ``sign Lambda_l``, and of each level's error Gram ``sum_r e_r x_r
    x_r'`` (``x = [z, 1, x - c0]``) the border bound and the Levenberg scale
    read only its leading ``k`` rows ``error_rows`` ``(K, k, p - 1)`` and its
    diagonal ``error_diagonal`` ``(K, p - 1)``.  ``counts`` and ``merges`` are
    each level's live rows and TSQR merges (the bound's constants).
    ``center`` is ``c0`` ``(q,)`` and ``generators`` the structural nulls of
    the border (``factor_smooth_prior_statistics``).

    Memory (Opus review P1): nothing here is ``K p^2`` except the signed
    pseudo-rows a step reads.  An ``sz`` leaf's data estimability and
    thin-level aliases read the trailing rows through their triangular factor
    ``tail_root`` (``trailing_root``, ``(q + 1)^2``): a pass beside thin
    levels merges it level by level (``_close_level``); any other ``sz``
    leaf forms it by one more compact pass on first read
    (``trailing_source``), and the published leaf keeps it (``published``).
    No pass keeps a ``(K, p, p)`` stack for it (#432).  ``triangles`` holds
    the whole triangles only where a caller formed them (``_assemble_system``).
    """

    top: NDArray
    tail_norms2: NDArray
    tail_gram: NDArray | None
    pseudo_rows: NDArray | None
    signature: NDArray | None
    error_rows: NDArray | None
    error_diagonal: NDArray | None
    counts: NDArray
    merges: NDArray
    center: NDArray
    generators: BorderGenerators | None
    signed: bool
    block_size: int
    triangles: NDArray | None = None
    tail_root: NDArray | None = None
    trailing_source: _TrailingRowsSource | None = None

    @property
    def width(self) -> int:
        """``p = k + q + 2``: basis, intercept, border and right-hand side."""
        return int(self.top.shape[2])

    @property
    def within(self) -> NDArray | None:
        """The within-level residual Gram of Fisher rows (``None`` for signed rows)."""
        return None if self.signed else self.tail_gram

    def stacked_tail_rows(self) -> NDArray:
        """Rows ``(m, q + 1)`` whose Gram is the trailing rows' over ``[1, x - c0]``.

        The trailing rows themselves while a caller's triangles are held, else
        their triangular factor (``trailing_root``): the same singular values
        and right singular vectors on any product (an orthogonal map of the
        rows).
        """
        if self.tail_root is None and self.triangles is not None:
            k = self.block_size
            q1 = self.width - k - 1
            return np.ascontiguousarray(self.triangles[:, k:, k : k + q1]).reshape(-1, q1)
        return self.trailing_root()

    def trailing_root(self) -> NDArray:
        """The trailing rows' triangular factor over ``[1, x - c0]``: the same products' norms.

        ``tail_root``, the pass's merge (beside thin levels) or the published
        leaf's; else formed on first read and held, from a caller's triangles
        or by ``trailing_source``'s compact pass.  An ``sz`` term's
        thin-level aliases read it at every factor build
        (``SumToZeroTreeFactor._penalized_aliases``), its data estimability
        at publication.  Owner: this leaf; lifetime: the leaf's;
        invalidation: none (the leaf data are immutable).  ``(q + 1)^2``
        floats.
        """
        if self.tail_root is not None:
            return self.tail_root
        held = self.__dict__.get("_trailing_root")
        if held is None:
            if self.triangles is not None:
                # one BLAS thread, as the pass: its bits do not depend on the pool
                with narrow_kernel_blas_threads(self.width):
                    held = np.linalg.qr(self.stacked_tail_rows(), mode="r")
            elif self.trailing_source is not None:
                held = self.trailing_source.root()
            if held is None:
                raise ValueError(
                    "This leaf keeps neither its triangles nor their trailing factor, and "
                    "cannot form them again (a leaf restored from a pickle in the middle of "
                    "a fit)."
                )
            object.__setattr__(self, "_trailing_root", held)
        return held

    def trailing_gram(self) -> NDArray | None:
        """``sum_l T_l' T_l`` of the trailing rows ``T_l``: ``tail_gram``, or formed from the triangles."""
        if self.tail_gram is not None or self.triangles is None:
            return self.tail_gram
        k = self.block_size
        tail = self.triangles[:, k:, k:]
        gram = np.einsum("kai,kaj->ij", tail, tail, optimize=True)
        return 0.5 * (gram + gram.T)

    def published(self) -> FactorSmoothLeafData:
        """The leaf a published state keeps: what a factor build reads, without the whole triangles.

        A published system still serves a lambda-only trial
        (``assembly.solve_cached_structured``), so its leaf keeps the leading
        rows, the trailing rows' norms and Gram, and, on signed rows, the
        pseudo-rows and error Grams.  Only an ``sz`` leaf's way to its trailing
        rows goes (its triangles, or the pass that forms them): it keeps their
        triangular factor (``trailing_root``) instead.
        """
        if self.triangles is None and self.trailing_source is None:
            return self
        tail_root = self.trailing_root()
        return dataclasses.replace(
            self,
            triangles=None,
            trailing_source=None,
            tail_root=tail_root,
            tail_gram=self.trailing_gram(),
        )


def factor_smooth_prior_statistics(
    layout: FactorSmoothLeafLayout,
    prior_weights: NDArray | None,
    *,
    chunk_size: int = _CHUNK,
) -> tuple[NDArray, BorderGenerators | None]:
    """The border centre ``c0`` and the structural null generators for these prior weights.

    One-engine design §3.2 by type, as ``moments.nested_prior_statistics``:
    every border column outside a one-hot block is centred on its shifted
    prior-weighted mean ``c0_j = x_ref,j + sum_r omega_r (x_rj - x_ref,j) /
    sum_r omega_r`` (``x_ref`` the first row in level order with ``omega >
    0``), formed from the rows the leaf pass itself forms; one-hot columns
    keep 0.  Cache contract: the lineage slot ``_lineage_slot(layout,
    "prior")``, keyed by the layout's sources and the prior weights themselves
    (held as a read-only copy), at most two entries.
    """
    n = len(layout.leaf_order)
    weights = np.ones(n) if prior_weights is None else np.asarray(prior_weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError("prior_weights must match the design rows.")
    prior_cache = _lineage_slot(layout, "prior")
    sources = layout.lineage_sources
    for held_sources, source, held, center, generators in prior_cache:
        if _same_sources(held_sources, sources) and (
            source is weights or np.array_equal(held, weights)
        ):
            return center, generators
    held = np.array(weights, dtype=np.float64, copy=True)
    held.setflags(write=False)
    q = layout.width
    center = np.zeros(q)
    dense = np.flatnonzero(~layout.one_hot_columns)
    if dense.size:
        omega = held[layout.leaf_order]
        positive = np.flatnonzero(omega > 0.0)
        if not positive.size:
            raise ValueError("prior weights must contain a positive entry.")
        zero = np.zeros(q)
        reference = np.empty((1, q))
        first = int(positive[0])
        _leaf_rows(layout, reference, first, first + 1, zero)
        total = np.zeros(dense.size)
        buffer = np.empty((min(chunk_size, n), q))
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            rows = buffer[: hi - lo]
            _leaf_rows(layout, rows, lo, hi, zero)
            _shifted_sums(
                np.ascontiguousarray(rows[:, dense]), omega[lo:hi], reference[0, dense], total
            )
        center[dense] = reference[0, dense] + total / float(np.sum(omega))
    center.setflags(write=False)
    generators = _border_generators(layout, held)
    prior_cache.insert(0, (sources, weights, held, center, generators))
    del prior_cache[2:]
    return center, generators


def release_leaf_memo(layout_cache: dict | None) -> None:
    """Empty the lineage's leaf memo once a fit has published its state (Opus review P1).

    The memo (``build_factor_smooth_leaf_system``) serves a repeated build
    within a fit.  Its last system would otherwise stay alive with the
    design: on signed rows its pseudo-rows and error Grams, on an ``sz`` term
    its whole triangles, ``K p^2`` each.  A later fit on the lineage fills it
    again.
    """
    from superglm.solvers._structured.selection import shared_nesting_cache

    if layout_cache is None:
        return
    memo = shared_nesting_cache(layout_cache).get(("fs_slot", "leaf_memo"))
    if memo is not None:
        memo.clear()


def _lineage_slot(layout: FactorSmoothLeafLayout, name: str) -> list:
    """A bounded list in the layout's lineage cache, one per ``name`` (perf F15).

    Owner: the lineage's nesting cache (``selection.shared_nesting_cache``),
    shared by the layouts of every lambda rebuild of the design.  Each entry
    carries the ``lineage_sources`` it was formed from and matches only a
    layout whose sources are those same objects (``_same_sources``): a rebuild
    that passes the term's arrays and border matrices through shares it, and
    one that re-creates a border matrix misses and replaces it.  The callers
    keep at most two entries, so no sequence of rebuilds accumulates entries
    or keeps retired border matrices alive.
    """
    return layout.lineage_cache.setdefault(("fs_slot", name), [])


def _same_sources(held: tuple, sources: tuple) -> bool:
    return len(held) == len(sources) and all(a is b for a, b in zip(held, sources, strict=True))


def _leaf_pass(
    layout: FactorSmoothLeafLayout,
    W: NDArray,
    Wz: NDArray,
    center: NDArray,
    error: NDArray | None,
    *,
    signed: bool,
    compact: bool = False,
    tail_root: bool = False,
    chunk_size: int = _CHUNK,
) -> tuple:
    """The per-level triangles (and middle factors, error Grams) of one iterate's rows.

    Returns ``(R_acc, M_acc, E_acc, counts, merges)``.  ``compact`` closes
    each level into the compact leaf data as the pass leaves it and forms no
    ``(K, p, p)`` triangle or middle factor (Opus review P1): Fisher rows in
    the kernel itself (``_fisher_leaf_segments``), signed rows a few levels at
    a time (``_SignedWindow``); it returns their ``_LeafParts``, with the
    trailing rows' triangular factor over ``[1, x - c0]`` when ``tail_root``
    (``_close_level``).
    """
    dominant = layout.dominant
    k, q = layout.block_size, layout.width
    p = k + q + 2
    K = layout.leaf_count
    order = layout.leaf_order
    n = len(order)
    window = (
        _SignedWindow(K, k, p, with_gram=dominant.factor_basis == "sz", with_root=tail_root)
        if compact and signed
        else None
    )
    if compact and not signed:
        top = np.zeros((K, k, p))
        tail_norms2 = np.zeros((K, p - k))
        tail_gram = np.zeros((p - k, p - k))
        root = np.zeros((p - k, p - k) if tail_root else (0, 0))
        R_open = np.zeros((p, p))
        open_state = np.array([-1, 0], dtype=np.int64)
    elif window is None:
        R_acc = np.zeros((K, p, p))
        M_acc = np.zeros((K, p, p)) if signed else np.zeros((1, 1, 1))
        E_acc = np.zeros((K, p - 1, p - 1)) if signed else np.zeros((1, 1, 1))
    started = np.zeros(K, dtype=np.bool_)
    counts = np.zeros(K, dtype=np.int64)
    merges = np.zeros(K, dtype=np.int64)
    levels = layout.leaf_levels
    weights_all = np.asarray(W, dtype=np.float64)
    rhs_all = np.asarray(Wz, dtype=np.float64)
    error_all = None
    if signed:
        error_all = np.abs(weights_all) if error is None else np.asarray(error, dtype=np.float64)
    no_error = np.zeros(0)
    table = layout.basis_table
    basis_source = layout.sorted_basis
    natural = np.ascontiguousarray(dominant.natural_map, dtype=np.float64)
    identity = np.arange(n, dtype=np.intp)
    rows = np.empty((min(chunk_size, max(n, 1)), p))
    border = np.empty((rows.shape[0], q))
    basis = np.empty((rows.shape[0], k))
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        m = hi - lo
        index = order[lo:hi]
        if table is None:
            data, indices, indptr = basis_source
            _exact_basis_rows(basis, identity, lo, hi, data, indices, indptr, natural)
        else:
            _table_basis_rows(basis, identity, lo, hi, basis_source[0], table)
        if q:
            _leaf_rows(layout, border[:m], lo, hi, center)
        w = weights_all[index]
        with np.errstate(divide="ignore", invalid="ignore"):
            response = np.where(w != 0.0, rhs_all[index] / np.where(w != 0.0, w, 1.0), 0.0)
        chunk = rows[:m]
        chunk[:, :k] = basis[:m]
        chunk[:, k] = 1.0
        chunk[:, k + 1 : k + 1 + q] = border[:m]
        chunk[:, p - 1] = response
        if not np.all(np.isfinite(chunk)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {layout.dominant_group_name!r} has non-finite leaf rows."
            )
        if window is not None:
            assert error_all is not None
            window.fold(chunk, w, np.ascontiguousarray(error_all[index]), levels[lo:hi])
            continue
        if compact:
            if (
                _fisher_leaf_segments(
                    chunk,
                    w,
                    levels[lo:hi],
                    R_open,
                    open_state,
                    top,
                    tail_norms2,
                    tail_gram,
                    counts,
                    merges,
                    k,
                    root,
                )
                >= 0
            ):
                raise ValueError("The leaf pass needs its rows in level order.")
            continue
        _leaf_segments(
            chunk,
            w,
            no_error if error_all is None else np.ascontiguousarray(error_all[index]),
            levels[lo:hi],
            R_acc,
            M_acc,
            E_acc,
            started,
            counts,
            merges,
            signed,
        )
    if window is not None:
        return window.finish()
    if compact:
        if open_state[0] >= 0:
            _close_level(R_open, open_state[0], k, top, tail_norms2, tail_gram, True, root)
        return _LeafParts(
            top,
            tail_norms2,
            tail_gram,
            counts,
            merges,
            tail_root=_trailing_root_of(root) if tail_root else None,
        )
    return R_acc, (M_acc if signed else None), (E_acc if signed else None), counts, merges


@numba.njit(cache=True)
def _moment_segments(basis, border, weights, levels, D, C, A, zs, xs, ws):  # pragma: no cover
    """Raw moments of several signed weight vectors in one pass over level-ordered rows.

    Per weight vector ``t``: ``D[t, l] += a z z'``, ``C[t, l] += a z x'``, ``A[t]
    += a x x'`` (lower triangles), ``zs[t, l] += a z``, ``xs[t] += a x``, ``ws[t]
    += a``; zero border entries are skipped (one-hot rows).  Plain left-to-right
    sums, as the moment builder they replace.
    """
    m, k = basis.shape
    q = border.shape[1]
    nw = weights.shape[0]
    present = np.empty(q, np.int64)
    for r in range(m):
        level = levels[r]
        count = 0
        for j in range(q):
            if border[r, j] != 0.0:
                present[count] = j
                count += 1
        for t in range(nw):
            a = weights[t, r]
            if a == 0.0:
                continue
            ws[t] += a
            for i in range(k):
                zi = a * basis[r, i]
                zs[t, level, i] += zi
                for j in range(i + 1):
                    D[t, level, i, j] += zi * basis[r, j]
                for c in range(count):
                    j = present[c]
                    C[t, level, i, j] += zi * border[r, j]
            for c in range(count):
                i = present[c]
                xi = a * border[r, i]
                xs[t, i] += xi
                for d in range(c + 1):
                    j = present[d]
                    A[t, i, j] += xi * border[r, j]


@numba.njit(cache=True)
def _raw_moment_segments(
    data, indices, indptr, lo, border, weights, levels, G, C, A, zs, xs, ws
):  # pragma: no cover - compiled
    """``_moment_segments`` on the raw CSR basis rows ``lo + r`` (level order).

    ``G[t, l]`` (lower triangle), ``C[t, l]`` and ``zs[t, l]`` are in the raw
    basis columns; the caller maps them to the natural basis once
    (``M' G M``, ``M' C``, ``zs M``), so a row costs its few stored entries
    rather than ``k`` dense ones (perf finding F16).
    """
    m, q = border.shape
    nw = weights.shape[0]
    present = np.empty(q, np.int64)
    for r in range(m):
        level = levels[r]
        start, stop = indptr[lo + r], indptr[lo + r + 1]
        count = 0
        for j in range(q):
            if border[r, j] != 0.0:
                present[count] = j
                count += 1
        for t in range(nw):
            a = weights[t, r]
            if a == 0.0:
                continue
            ws[t] += a
            for s in range(start, stop):
                ci = indices[s]
                zi = a * data[s]
                zs[t, level, ci] += zi
                for u in range(start, s + 1):
                    cj = indices[u]
                    value = zi * data[u]
                    if u != s and cj == ci:
                        value += value  # a repeated column: both orders land on the diagonal
                    if cj <= ci:
                        G[t, level, ci, cj] += value
                    else:
                        G[t, level, cj, ci] += value
                for c in range(count):
                    j = present[c]
                    C[t, level, ci, j] += zi * border[r, j]
            for c in range(count):
                i = present[c]
                xi = a * border[r, i]
                xs[t, i] += xi
                for d in range(c + 1):
                    j = present[d]
                    A[t, i, j] += xi * border[r, j]


def factor_smooth_moment_operators(
    layout: FactorSmoothLeafLayout,
    weights: Sequence[NDArray],
    *,
    center: NDArray | None = None,
    chunk_size: int = _CHUNK,
    level_cross: bool = False,
) -> list[tuple]:
    """``(moment operator, X'a, sum a)`` for every signed weight vector, in one row pass.

    The REML weight-derivative operators (perf finding F5): they only enter
    traces, so they stay on moments (design §3.4), formed for all directions
    together from the rows the leaf pass forms (the level-sorted basis and the
    border rows).  With ``center`` (the border's ``c0``, ``(q,)``) the border
    rows are ``x - c0`` and the moments and ``X'a`` are those of the shifted
    rows, as the leaf pass forms them (design §3.2); without it, raw.  With
    ``level_cross`` each tuple also carries the per-level ``Z_l' a_l``
    ``(K, k)`` of an ``sz`` term (``None`` for ``fs``), whose public part is
    ``X'a``'s level block.
    """
    dominant = layout.dominant
    k, q, K = layout.block_size, layout.width, layout.leaf_count
    order = layout.leaf_order
    n = len(order)
    nw = len(weights)
    stacked = np.stack([np.asarray(w, dtype=np.float64) for w in weights])
    table = layout.basis_table
    natural = np.ascontiguousarray(dominant.natural_map, dtype=np.float64)
    width = k if table is not None else natural.shape[0]  # exact rows: raw CSR columns
    D = np.zeros((nw, K, width, width))
    C = np.zeros((nw, K, width, q))
    A = np.zeros((nw, q, q))
    zs = np.zeros((nw, K, width))
    xs = np.zeros((nw, q))
    ws = np.zeros(nw)
    levels = layout.leaf_levels
    source = layout.sorted_basis
    identity = np.arange(n, dtype=np.intp)
    shift = np.zeros(q) if center is None else np.asarray(center, dtype=np.float64)
    if shift.shape != (q,):
        raise ValueError("center must match the border width.")
    rows = min(chunk_size, max(n, 1))
    basis = np.empty((rows, k)) if table is not None else np.empty((0, k))
    border = np.empty((rows, q))
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        m = hi - lo
        if q:
            _leaf_rows(layout, border[:m], lo, hi, shift)
        chunk_weights = np.ascontiguousarray(stacked[:, order[lo:hi]])
        if table is None:
            data, indices, indptr = source
            _raw_moment_segments(
                data,
                indices,
                indptr,
                lo,
                border[:m],
                chunk_weights,
                levels[lo:hi],
                D,
                C,
                A,
                zs,
                xs,
                ws,
            )
            continue
        _table_basis_rows(basis, identity, lo, hi, source[0], table)
        _moment_segments(basis[:m], border[:m], chunk_weights, levels[lo:hi], D, C, A, zs, xs, ws)
    if table is None:  # the raw moments in the natural basis, once
        full = np.tril(D) + np.swapaxes(np.tril(D, -1), -1, -2)
        D = natural.T @ full @ natural
        C = natural.T @ C
        zs = zs @ natural
    results = []
    sum_to_zero = layout.dominant.factor_basis == "sz"
    p = len(layout.small_indices) + layout.structured_indices.size
    operator_type = SumToZeroBlockOperator if sum_to_zero else BlockSymmetricOperator
    for t in range(nw):
        # mirrored from one triangle, so exactly symmetric: the natural map's
        # matmul rounds the two triangles differently
        lower_D = np.tril(D[t]) + np.transpose(np.tril(D[t], -1), (0, 2, 1))
        lower_A = np.tril(A[t]) + np.tril(A[t], -1).T
        operator = operator_type(
            A=lower_A,
            C=C[t],
            D=lower_D,
            small_indices=layout.small_indices,
            structured_indices=layout.structured_indices,
        )
        cross = np.empty(p)
        cross[layout.small_indices] = xs[t]
        # sz: the public vector [I; -1']' of the level one (one-engine design §3.5)
        cross[layout.structured_indices] = zs[t][:-1] - zs[t][-1:] if sum_to_zero else zs[t]
        entry = (operator, cross, float(ws[t]))
        if level_cross:
            entry += (zs[t].copy() if sum_to_zero else None,)
        results.append(entry)
    return results


# --------------------------------------------------------------- system ----
@dataclass(frozen=True)
class FactorSmoothLeafSystem:
    """One iterate's ``fs`` system: leaf data, the raw moment operator and working statistics.

    ``operator`` is the raw-coordinate moment operator ``[Z X]' W [Z X]`` formed
    from the leaf data by products (it serves traces and inference, never a
    factorization); ``leaf`` is what ``FactorSmoothLeafFactor`` factors.
    """

    operator: BlockSymmetricOperator
    leaf: FactorSmoothLeafData
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    sum_w: float
    sum_wz: float
    dominant_group_index: int
    dominant_group_name: str
    # The border block and cross of the rows shifted by ``c0`` (``leaf.center``):
    # ``sum w (x - c0)(x - c0)'`` and ``sum w (x - c0)``, and the level-by-border
    # cross ``sum w z (x - c0)'`` ``(K, k, q)``, from the leaf Grams.
    shifted_gram_small: NDArray
    shifted_xtw_small: NDArray
    shifted_level_cross: NDArray
    # The last factor built on this system (``assembly.build_augmented_block_factor``),
    # reused when the next has bitwise the same penalty parts (perf F1).  Owner:
    # this system; lifetime: the system, one slot; key: ``penalty_small`` and
    # ``penalty_local`` compared exactly (a Levenberg shift enters them); the
    # leaf data are the system's own, so nothing else invalidates it.  The slot
    # holds the factor through a weak reference (perf F17): the factor holds
    # this system, so a strong slot made a cycle that kept every iterate's
    # system and factor alive until the cyclic collector ran.
    factor_memo: list = field(default_factory=list, repr=False, compare=False)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["factor_memo"] = []  # a weak reference has no state to save
        return state

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def published(self) -> FactorSmoothLeafSystem:
        """This system as a published state keeps it (``published_leaf_system``)."""
        return published_leaf_system(self)

    @cached_property
    def centred_data_operator(self) -> CenteredBlockOperator:
        """``X~' W X~`` centred on the working-weighted mean, formed on the rows shifted by ``c0``.

        Centring is shift-invariant: ``X - 1 m' = (X - 1 c0') - 1 (m - c0)'``,
        so the operator of the shifted moments, the shifted cross ``sum w (x -
        c0)`` and the centre ``m - c0`` is the same matrix as the raw one
        centred on ``m``.  The raw form subtracts ``sum w m m'`` from ``X' W X``
        and loses ``eps |c0|^2 / var`` of each large-offset column to
        cancellation (all of it at an offset near ``1e8``); the shifted form
        rounds at the columns' spread about ``c0`` (Chan, Golub & LeVeque
        1983, §3; one-engine design §3.2).  Cache owner: this system (it is
        immutable); lifetime: the system.
        """
        return _shifted_centred_operator(self, BlockSymmetricOperator, None)


def published_leaf_system(system):
    """A copy of an ``fs`` or ``sz`` leaf system for a published state (Opus review P1).

    The same moments, the same centred data operator (formed once, before the
    copy, so a published factor still recognises it as its own data) and the
    leaf without the whole triangles (``FactorSmoothLeafData.published``); no
    factor memo.  The fit's own system is left as it is: the lineage's memo
    may serve it to a later fit, which builds factors on it.
    """
    system.centred_data_operator  # noqa: B018 - formed once, carried by the copy
    state = system.__getstate__()
    state["leaf"] = system.leaf.published()
    clone = object.__new__(type(system))
    clone.__setstate__(state)
    return clone


def _is_own_centred_data(operator, system, xtw: NDArray, sum_w: float) -> bool:
    """Whether ``operator`` is ``system``'s centred data operator, in either form.

    The shifted form is ``system.centred_data_operator`` (matched on its
    moments object and vectors); the raw form centres ``system.operator`` on
    ``xtw / sum_w``.  The two are the same matrix.
    """
    if not isinstance(operator, CenteredBlockOperator) or operator.total != sum_w:
        return False
    if operator.raw is system.operator:
        return np.array_equal(operator.cross, xtw) and np.array_equal(operator.center, xtw / sum_w)
    own = system.__dict__.get("centred_data_operator")
    return (
        own is not None
        and operator.raw is own.raw
        and np.array_equal(operator.cross, own.cross)
        and np.array_equal(operator.center, own.center)
    )


def share_cross_block(operator, cross: NDArray):
    """``operator`` keeping ``cross`` itself as its cross block, not the copy it made (Opus review P1).

    The block operators copy every block they are given.  A penalty never
    touches the level-by-border cross block, so the penalized operators of a
    system (one per trial) and its centred operator would each keep a ``(K,
    k, q)`` copy of the system's.  ``cross`` is shared only when it holds the
    same values and is a read-only array that owns its data, so nothing can
    write through it; the copy is then dropped.
    """
    if (
        isinstance(cross, np.ndarray)
        and cross.dtype == np.float64
        and cross.base is None
        and not cross.flags.writeable
        and np.array_equal(operator.C, cross)
    ):
        object.__setattr__(operator, "C", cross)
    return operator


def _shifted_centred_operator(
    system, operator_type, raw_structured_cross: NDArray | None
) -> CenteredBlockOperator:
    """The centred data operator of an ``fs`` or ``sz`` leaf system on its ``c0``-shifted moments.

    The border block is ``system.shifted_gram_small``, the level-by-border cross
    ``system.shifted_level_cross`` (``C_c = sum w z (x - c0)'``), the level
    blocks the system's own; the cross vector is ``[sum w (x - c0) | X_t' w]`` (the level
    columns are not shifted) and the centre that over ``sum_w``.
    """
    raw = system.operator
    shifted = operator_type(
        A=system.shifted_gram_small,
        C=system.shifted_level_cross,
        D=raw.D,
        small_indices=raw.small_indices,
        structured_indices=raw.structured_indices,
    )
    share_cross_block(shifted, system.shifted_level_cross)
    cross = np.empty(raw.shape[0], dtype=np.float64)
    cross[raw.small_indices] = system.shifted_xtw_small
    cross[raw.structured_indices] = system.xtw_structured
    return CenteredBlockOperator(
        raw=shifted,
        cross=cross,
        total=system.sum_w,
        center=cross / system.sum_w,
        raw_structured_cross=raw_structured_cross,
    )


def build_factor_smooth_leaf_system(
    layout: FactorSmoothLeafLayout,
    W: NDArray,
    Wz: NDArray,
    *,
    prior_weights: NDArray | None = None,
    error: NDArray | None = None,
    signed: bool = False,
) -> FactorSmoothLeafSystem:
    """Build one iterate's ``fs`` leaf system (module docstring).

    ``signed`` is the rows' type (observed curvature), never read off their
    values: Fisher rows with a negative weight are a caller error.  ``error``
    is the rows' weight-error scale ``e_r >= |w_r|`` (``None``: ``|w|``).
    """
    weights = np.asarray(W, dtype=np.float64)
    weighted_rhs = np.asarray(Wz, dtype=np.float64)
    n = len(layout.leaf_order)
    if weights.shape != (n,) or weighted_rhs.shape != (n,):
        raise ValueError("W and Wz must be one-dimensional arrays with the design's rows.")
    if not signed and np.any(weights < 0.0):
        raise ValueError("Fisher rows must have non-negative weights; signed rows declare it.")
    if not (np.all(np.isfinite(weights)) and np.all(np.isfinite(weighted_rhs))):
        raise np.linalg.LinAlgError(
            f"FactorSmooth term {layout.dominant_group_name!r} has non-finite working rows."
        )
    center, generators = factor_smooth_prior_statistics(layout, prior_weights)
    held_error = None if error is None else np.asarray(error, dtype=np.float64)
    memo = _lineage_slot(layout, "leaf_memo")  # one entry for the whole lineage
    sources = layout.lineage_sources
    for held_sources, key, system in memo:
        held_w, held_wz, held_e, held_signed, held_center = key
        if (
            _same_sources(held_sources, sources)
            and held_signed == bool(signed)
            and held_center is center
            and np.array_equal(held_w, weights)
            and np.array_equal(held_wz, weighted_rhs)
            and (
                (held_e is None and held_error is None)
                or (
                    held_e is not None
                    and held_error is not None
                    and np.array_equal(held_e, held_error)
                )
            )
        ):
            return system
    # a kernel compiled since the last build left frames, and the dead systems
    # they reach, in reference cycles: free them before forming a new stack
    collect_after_compile()
    # LAPACK inside the pass runs on one BLAS thread under the automatic policy
    # (a wide fit re-capped at the leaf's width), so its bits do not depend on it
    basis = layout.dominant.factor_basis
    held = tuple(
        None if values is None else _read_only_copy(values)
        for values in (weights, weighted_rhs, held_error)
    )
    # no (K, p, p) triangle or middle factor is formed.  Beside thin levels an
    # sz leaf's aliases read its trailing rows' factor at every factor build,
    # so the pass merges it; any other sz leaf forms it by one more compact
    # pass when its estimability reads it (#432)
    thin_levels = layout.thin_levels(prior_weights) if basis == "sz" else ()
    with narrow_kernel_blas_threads(layout.block_size + layout.width + 2):
        parts = _leaf_pass(
            layout,
            weights,
            weighted_rhs,
            center,
            error,
            signed=signed,
            compact=True,
            tail_root=bool(thin_levels),
        )
    source = (
        _TrailingRowsSource(layout, held[0], held[1], center, held[2], signed)
        if basis == "sz"
        else None
    )
    assert isinstance(parts, _LeafParts)
    system = _assemble_compact(
        parts,
        trailing_source=source,
        center=center,
        generators=generators,
        signed=signed,
        block_size=layout.block_size,
        weights=weights,
        weighted_rhs=weighted_rhs,
        small_indices=layout.small_indices,
        structured_indices=layout.structured_indices,
        group_index=layout.dominant_group_index,
        group_name=layout.dominant_group_name,
        basis=basis,
        thin_levels=thin_levels,
        thin_counts=(layout.thin_level_counts(prior_weights) if basis == "sz" else None),
    )
    memo[:] = [(sources, (*held, bool(signed), center), system)]
    return system


def _read_only_copy(values: NDArray) -> NDArray:
    copy = np.array(values, dtype=np.float64, copy=True)
    copy.setflags(write=False)
    return copy


def _assemble_system(
    R_acc: NDArray,
    M_acc: NDArray | None,
    E_acc: NDArray | None,
    counts: NDArray,
    merges: NDArray,
    *,
    center: NDArray,
    generators: BorderGenerators | None,
    signed: bool,
    block_size: int,
    weights: NDArray,
    weighted_rhs: NDArray,
    small_indices: NDArray,
    structured_indices: NDArray,
    group_index: int,
    group_name: str,
    basis: str = "fs",
    thin_levels: tuple = (),
    thin_counts: tuple[NDArray, NDArray] | None = None,
):
    """The leaf data (pseudo-rows or within Gram) and the raw moment operator by products.

    The whole triangles ``R_acc`` are closed level by level into the compact
    leaf data (``_close_levels``, the arithmetic of the compact pass) and the
    system is assembled from it (``_assemble_compact``); signed rows read
    their triangles and middle factors once more there, and an ``sz`` leaf
    keeps the triangles.
    """
    k = block_size
    K, p = R_acc.shape[0], R_acc.shape[2]
    top = np.zeros((K, k, p))
    tail_norms2 = np.zeros((K, p - k))
    tail_gram = np.zeros((p - k, p - k))
    _close_levels(R_acc, k, top, tail_norms2, tail_gram, not signed)
    parts = _LeafParts(top, tail_norms2, None if signed else tail_gram, counts, merges)
    if signed:
        assert M_acc is not None
        pseudo_rows, signature, level_rows, border = _signed_parts(R_acc, M_acc, k)
        assert E_acc is not None
        parts = parts._replace(
            pseudo_rows=pseudo_rows,
            signature=signature,
            error_rows=np.ascontiguousarray(E_acc[:, :k, :]),
            error_diagonal=np.diagonal(E_acc, axis1=1, axis2=2).copy(),
            level_rows=level_rows,
            border=border,
        )
    return _assemble_compact(
        parts._replace(triangles=R_acc),
        center=center,
        generators=generators,
        signed=signed,
        block_size=block_size,
        weights=weights,
        weighted_rhs=weighted_rhs,
        small_indices=small_indices,
        structured_indices=structured_indices,
        group_index=group_index,
        group_name=group_name,
        basis=basis,
        thin_levels=thin_levels,
        thin_counts=thin_counts,
    )


class _LeafParts(NamedTuple):
    """One iterate's compact leaf data as a pass or ``_assemble_system`` forms it.

    ``top``, ``tail_norms2``, ``tail_gram`` (Fisher rows), ``counts`` and
    ``merges`` as ``FactorSmoothLeafData``; signed rows add the pseudo-rows,
    signatures and error Grams' leading rows and diagonals, and their level
    Grams' level-basis rows and
    border sum (``_signed_parts``); ``triangles`` the whole triangles where a
    caller formed them (an ``sz`` leaf keeps them).
    """

    top: NDArray
    tail_norms2: NDArray
    tail_gram: NDArray | None
    counts: NDArray
    merges: NDArray
    pseudo_rows: NDArray | None = None
    signature: NDArray | None = None
    error_rows: NDArray | None = None
    error_diagonal: NDArray | None = None
    level_rows: NDArray | None = None
    border: NDArray | None = None
    triangles: NDArray | None = None
    tail_root: NDArray | None = None


def _trailing_root_of(root: NDArray) -> NDArray:
    """The ``(q + 1, q + 1)`` factor over ``[1, x - c0]`` of a pass's trailing ``tail_root``.

    The merged triangle spans ``[1, x - c0, z]``; its leading block is the
    triangular factor of the stacked rows without the right-hand side.
    """
    width = root.shape[0] - 1
    held = np.array(root[:width, :width], copy=True)
    held.setflags(write=False)
    return held


def _assemble_compact(
    parts: _LeafParts,
    *,
    trailing_source: _TrailingRowsSource | None = None,
    center: NDArray,
    generators: BorderGenerators | None,
    signed: bool,
    block_size: int,
    weights: NDArray,
    weighted_rhs: NDArray,
    small_indices: NDArray,
    structured_indices: NDArray,
    group_index: int,
    group_name: str,
    basis: str = "fs",
    thin_levels: tuple = (),
    thin_counts: tuple[NDArray, NDArray] | None = None,
):
    """The system of one iterate from its compact leaf data (``FactorSmoothLeafData``).

    The level Grams ``G_l = R_l' M_l R_l`` (``M_l = I`` for Fisher rows) are
    never formed whole (Opus review P1: ``(K, p, p)`` beside a wide border):
    the moment operator reads only their level-basis rows ``G_l[:k, :]`` and
    the border block summed over the levels (``_fisher_moments``; signed
    rows bring theirs, ``_signed_parts``).  ``basis="sz"`` returns the
    ``SumToZeroLeafSystem`` of the balance tree: the same leaf data over all
    ``K`` levels, the level-space moment operator behind the public ``K - 1``
    coordinates and the public vectors ``[I; -1']' v`` of the level ones
    (one-engine design §3.5); its leaf reaches its trailing rows through
    ``trailing_source`` (or the triangles a caller formed) until it is
    published (``FactorSmoothLeafData.published``).
    """
    k = block_size
    top, tail_norms2, tail_gram = parts.top, parts.tail_norms2, parts.tail_gram
    if signed:
        assert parts.pseudo_rows is not None and parts.signature is not None
        assert parts.level_rows is not None and parts.border is not None
        level_rows, border = parts.level_rows, parts.border
    else:
        assert tail_gram is not None
        level_rows, border = _fisher_moments(top, tail_gram, k)
    leaf = FactorSmoothLeafData(
        top=top,
        tail_norms2=tail_norms2,
        tail_gram=tail_gram,
        pseudo_rows=parts.pseudo_rows,
        signature=parts.signature,
        error_rows=parts.error_rows,
        error_diagonal=parts.error_diagonal,
        counts=parts.counts,
        merges=parts.merges,
        center=center,
        generators=generators,
        signed=bool(signed),
        block_size=k,
        triangles=parts.triangles if basis == "sz" else None,
        tail_root=parts.tail_root if basis == "sz" else None,
        trailing_source=trailing_source,
    )
    return _system_from_moments(
        leaf,
        level_rows,
        border,
        weights=weights,
        weighted_rhs=weighted_rhs,
        small_indices=small_indices,
        structured_indices=structured_indices,
        group_index=group_index,
        group_name=group_name,
        basis=basis,
        thin_levels=thin_levels,
        thin_counts=thin_counts,
    )


def _fisher_moments(top: NDArray, tail_gram: NDArray, k: int) -> tuple[NDArray, NDArray]:
    """``G_l[:k, :]`` ``(K, k, p)`` and ``sum_l G_l[k:, k:]`` of Fisher rows, ``G_l = R_l'R_l``.

    ``R_l`` is zero below its leading ``k`` rows on the level basis, so the
    level-basis rows of ``G_l`` are products of ``top`` alone, and the border
    block is ``top``'s trailing columns' Gram plus the trailing rows' one.
    """
    level_rows = np.einsum("kai,kaj->kij", top[:, :, :k], top, optimize=True)
    right = top[:, :, k:]
    border = np.einsum("kai,kaj->ij", right, right, optimize=True) + tail_gram
    return level_rows, border


def _signed_parts(R: NDArray, M: NDArray, k: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Signed rows' pseudo-rows and signatures, ``G_l[:k, :]`` and ``sum_l G_l[k:, k:]``.

    ``_signed_block`` over blocks of about half a MiB of ``p x p`` matrices, so no
    temporary is another ``(K, p, p)`` array beside the ones returned (Opus
    review P1); each level's own numbers do not depend on the block.
    """
    K, p = R.shape[0], R.shape[2]
    pseudo_rows = np.empty((K, p, p))
    signature = np.empty((K, p))
    level_rows = np.empty((K, k, p))
    border = np.zeros((p - k, p - k))
    step = max(1, (1 << 16) // (p * p))
    for lo in range(0, K, step):
        hi = min(lo + step, K)
        pseudo_rows[lo:hi], signature[lo:hi], level_rows[lo:hi], piece = _signed_block(
            R[lo:hi], M[lo:hi], k
        )
        border += piece
    return pseudo_rows, signature, level_rows, border


def _signed_block(R: NDArray, M: NDArray, k: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """``_signed_parts`` of a block of levels ``R``, ``M`` ``(b, p, p)``.

    ``M_l = V Lambda V'`` gives ``B_l = |Lambda|^1/2 V' R_l`` with signature
    ``sign Lambda`` (module docstring), and ``G_l = R_l' M_l R_l``: ``R_l[:, :k]``
    is zero below row ``k``, so ``G_l[:k, :] = R_l[:k, :k]' (M_l R_l)[:k, :]``,
    and the border block runs over every row of the middle factor.
    """
    lam, vectors = np.linalg.eigh(M)
    pseudo_rows = np.sqrt(np.abs(lam))[:, :, None] * np.einsum(
        "kba,kbj->kaj", vectors, R, optimize=True
    )
    signature = np.where(lam < 0.0, -1.0, 1.0)
    level_rows = np.einsum("kai,kaj->kij", R[:, :k, :k], np.matmul(M[:, :k, :], R), optimize=True)
    right = R[:, :, k:]
    border = np.einsum("kai,kaj->ij", right, np.matmul(M, right), optimize=True)
    return pseudo_rows, signature, level_rows, border


class _SignedWindow:
    """The signed rows of an ``fs`` pass, a few levels at a time (Opus review P1).

    The rows come in level order, so a level is finished once the pass leaves
    it.  ``fold`` runs the leaf kernel (``_leaf_segments``) over at most
    ``size`` levels at a time in window slots, closes every finished level
    into its pseudo-rows, signature, level Gram rows, border sum, leading
    rows, trailing norms and error Gram rows and diagonal (``_signed_block``,
    ``_close_level``), and carries the open one in slot 0; ``finish`` closes
    the last.  Each level's triangle, middle factor and error Gram are the
    kernel's on the whole pass, bit for bit, and no ``(K, p, p)`` triangle,
    middle factor or error Gram is formed: the pseudo-rows the per-lambda
    step reads are all that is ``K p^2``.  ``with_gram`` (an ``sz`` term) also
    sums the trailing rows' Gram, which its data estimability's Fisher system
    reads.  A level with no rows keeps zeros and signature one, as the whole
    pass gives it.
    """

    def __init__(
        self, K: int, k: int, p: int, *, with_gram: bool = False, with_root: bool = False
    ) -> None:
        self.k, self.p = k, p
        self.with_gram = bool(with_gram)
        self.tail_gram = np.zeros((p - k, p - k)) if with_gram else np.zeros((0, 0))
        self.tail_root = np.zeros((p - k, p - k) if with_root else (0, 0))
        self.size = max(2, (1 << 16) // (p * p))
        self.top = np.zeros((K, k, p))
        self.tail_norms2 = np.zeros((K, p - k))
        self.error_rows = np.zeros((K, k, p - 1))
        self.error_diagonal = np.zeros((K, p - 1))
        self.pseudo_rows = np.zeros((K, p, p))
        self.signature = np.ones((K, p))
        self.level_rows = np.zeros((K, k, p))
        self.border = np.zeros((p - k, p - k))
        self.counts = np.zeros(K, dtype=np.int64)
        self.merges = np.zeros(K, dtype=np.int64)
        size = self.size
        self._R = np.zeros((size, p, p))
        self._M = np.zeros((size, p, p))
        self._E = np.zeros((size, p - 1, p - 1))
        self._started = np.zeros(size, dtype=np.bool_)
        self._counts = np.zeros(size, dtype=np.int64)
        self._merges = np.zeros(size, dtype=np.int64)
        self._levels = np.zeros(0, dtype=np.intp)  # the absolute level of each live slot

    def fold(self, rows: NDArray, weights: NDArray, error: NDArray, levels: NDArray) -> None:
        """Fold one chunk of level-ordered rows."""
        starts = np.flatnonzero(levels[1:] != levels[:-1]) + 1
        bounds = np.concatenate(([0], starts[self.size - 1 :: self.size], [len(levels)]))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            if hi > lo:
                self._fold(rows[lo:hi], weights[lo:hi], error[lo:hi], levels[lo:hi])

    def _fold(self, rows: NDArray, weights: NDArray, error: NDArray, levels: NDArray) -> None:
        first = int(levels[0])
        if self._levels.size and int(self._levels[0]) != first:
            self._close(1)
        if self._levels.size and int(levels[-1]) < int(self._levels[0]):
            raise ValueError("The leaf pass needs its rows in level order.")
        changes = np.flatnonzero(levels[1:] != levels[:-1]) + 1
        slots = np.zeros(len(levels), dtype=np.intp)
        slots[changes] = 1
        slots = np.cumsum(slots)
        self._levels = np.concatenate(([first], levels[changes])).astype(np.intp)
        _leaf_segments(
            rows,
            weights,
            error,
            slots,
            self._R,
            self._M,
            self._E,
            self._started,
            self._counts,
            self._merges,
            True,
        )
        used = len(self._levels)
        self._close(used - 1)  # leaves the open level's absolute index alone in _levels
        if used > 1:
            for buffer in (self._R, self._M, self._E, self._started, self._counts, self._merges):
                buffer[0] = buffer[used - 1]
                buffer[1:used] = 0

    def _close(self, count: int) -> None:
        """Close slots ``[0, count)``, whose levels are finished, into the leaf data."""
        if count <= 0:
            return
        levels = self._levels[:count]
        k = self.k
        for slot, level in enumerate(levels):
            _close_level(
                self._R[slot],
                level,
                k,
                self.top,
                self.tail_norms2,
                self.tail_gram,
                self.with_gram,
                self.tail_root,
            )
        pseudo_rows, signature, level_rows, border = _signed_block(
            self._R[:count], self._M[:count], k
        )
        self.pseudo_rows[levels] = pseudo_rows
        self.signature[levels] = signature
        self.level_rows[levels] = level_rows
        self.border += border
        self.error_rows[levels] = self._E[:count, :k, :]
        self.error_diagonal[levels] = np.diagonal(self._E[:count], axis1=1, axis2=2)
        self.counts[levels] = self._counts[:count]
        self.merges[levels] = self._merges[:count]
        for buffer in (self._R, self._M, self._E, self._started, self._counts, self._merges):
            buffer[:count] = 0
        self._levels = self._levels[count:]

    def finish(self) -> _LeafParts:
        """Close the open level; the pass's compact signed leaf data."""
        self._close(len(self._levels))
        return _LeafParts(
            self.top,
            self.tail_norms2,
            self.tail_gram if self.with_gram else None,
            self.counts,
            self.merges,
            pseudo_rows=self.pseudo_rows,
            signature=self.signature,
            error_rows=self.error_rows,
            error_diagonal=self.error_diagonal,
            level_rows=self.level_rows,
            border=self.border,
            tail_root=_trailing_root_of(self.tail_root) if self.tail_root.shape[0] else None,
        )


def _system_from_moments(
    leaf: FactorSmoothLeafData,
    level_rows: NDArray,
    border: NDArray,
    *,
    weights: NDArray,
    weighted_rhs: NDArray,
    small_indices: NDArray,
    structured_indices: NDArray,
    group_index: int,
    group_name: str,
    basis: str = "fs",
    thin_levels: tuple = (),
    thin_counts: tuple[NDArray, NDArray] | None = None,
):
    """The system of ``leaf`` from its level Grams' level-basis rows and border sum."""
    k = leaf.block_size
    center = leaf.center
    q = len(center)
    D = level_rows[:, :, :k]
    D = np.ascontiguousarray(0.5 * (D + D.transpose(0, 2, 1)))
    zw = np.ascontiguousarray(level_rows[:, :, k])
    C_centred = np.array(level_rows[:, :, k + 1 : k + 1 + q], copy=True)
    C_centred.setflags(write=False)  # shared by the centred operator (share_cross_block)
    zwz = np.ascontiguousarray(level_rows[:, :, k + q + 1])
    border = 0.5 * (border + border.T)
    moment = border[0, 1 : q + 1]
    shifted_A = border[1 : q + 1, 1 : q + 1]
    shifted_A = 0.5 * (shifted_A + shifted_A.T)
    A = border[1 : q + 1, 1 : q + 1] + np.outer(center, moment) + np.outer(moment, center)
    A = A + border[0, 0] * np.outer(center, center)
    A = 0.5 * (A + A.T)
    C = C_centred + zw[:, :, None] * center[None, None, :]
    if basis == "sz":
        from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
        from superglm.solvers._structured.balance_tree import SumToZeroLeafSystem

        return SumToZeroLeafSystem(
            operator=SumToZeroBlockOperator(
                A=A, C=C, D=D, small_indices=small_indices, structured_indices=structured_indices
            ),
            leaf=leaf,
            xtw_small=moment + border[0, 0] * center,
            xtw_structured=adjoint_sum_to_zero_blocks(zw),
            xtwz_small=border[1 : q + 1, q + 1] + border[0, q + 1] * center,
            xtwz_structured=adjoint_sum_to_zero_blocks(zwz),
            raw_xtw_structured=zw,
            sum_w=float(np.sum(weights)),
            sum_wz=float(np.sum(weighted_rhs)),
            dominant_group_index=group_index,
            dominant_group_name=group_name,
            thin_levels=tuple(thin_levels),
            thin_counts=thin_counts,
            shifted_gram_small=shifted_A,
            shifted_xtw_small=np.array(moment, copy=True),
            shifted_level_cross=C_centred,
        )
    operator = BlockSymmetricOperator(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    return FactorSmoothLeafSystem(
        operator=operator,
        leaf=leaf,
        xtw_small=moment + border[0, 0] * center,
        xtw_structured=zw,
        xtwz_small=border[1 : q + 1, q + 1] + border[0, q + 1] * center,
        xtwz_structured=zwz,
        sum_w=float(np.sum(weights)),
        sum_wz=float(np.sum(weighted_rhs)),
        dominant_group_index=group_index,
        dominant_group_name=group_name,
        shifted_gram_small=shifted_A,
        shifted_xtw_small=np.array(moment, copy=True),
        shifted_level_cross=C_centred,
    )


# ---------------------------------------------------- penalized operator ----
@dataclass(frozen=True)
class FactorSmoothPenalizedOperator(BlockSymmetricOperator):
    """The penalized raw moment operator of an ``fs`` system, with its penalty parts.

    ``penalty_small`` ``(q, q)`` and ``penalty_local`` ``(K, k, k)`` are the
    border and level penalties the factor takes as square roots; ``A`` and
    ``D`` already include them (the operator every trace reads).
    """

    penalty_small: NDArray | None = None
    penalty_local: NDArray | None = None

    @classmethod
    def with_penalties(
        cls, operator: BlockSymmetricOperator, penalty_small: NDArray, penalty_local: NDArray
    ) -> FactorSmoothPenalizedOperator:
        """The moments ``operator`` plus the penalty parts, which it also keeps apart."""
        small = np.asarray(penalty_small, dtype=np.float64)
        local = np.asarray(penalty_local, dtype=np.float64)
        penalized = cls(
            A=operator.A + small,
            C=operator.C,
            D=operator.D + local,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            penalty_small=0.5 * (small + small.T),
            penalty_local=0.5 * (local + local.transpose(0, 2, 1)),
        )
        return share_cross_block(penalized, operator.C)


# ---------------------------------------------------------------- factor ----
def _penalty_roots(P: NDArray) -> NDArray:
    """``(K, k, k)`` rows ``Lambda^1/2 V'`` of each level penalty ``P_l = V Lambda V'`` (``P_l >= 0``)."""
    symmetric = 0.5 * (P + P.transpose(0, 2, 1))
    if np.all(symmetric == symmetric[:1]):
        lam, vectors = np.linalg.eigh(symmetric[0])
        root = np.sqrt(np.maximum(lam, 0.0))[:, None] * vectors.T
        return np.ascontiguousarray(np.broadcast_to(root, symmetric.shape))
    lam, vectors = np.linalg.eigh(symmetric)
    return np.ascontiguousarray(
        np.sqrt(np.maximum(lam, 0.0))[:, :, None] * vectors.transpose(0, 2, 1)
    )


class FactorSmoothLeafFactor:
    """Factorization of an ``fs`` system in augmented raw coordinates ``[1, X]`` (module docstring).

    Public attributes and methods follow the retired ``BlockSchurFactor``:
    ``shape``, ``small_indices`` (intercept 0 first), ``structured_indices``
    ``(K, k)``, ``rank``, ``rank_truncated``, ``schur_condition_estimate``,
    ``logdet``, ``solve``,
    the selected inverses and every trace.  ``solve_data`` solves the
    normal equations whose right-hand side travelled inside the leaf
    factorization, in the centred coordinates when asked (the PIRLS state of
    §3.8).  ``border_certificate`` and ``weakly_identified_coefficients`` are
    the §3.6 disclosure.  ``excluded`` border columns (augmented global
    indices) are left out of the Laplace term as ``NestedSchurFactor`` does.
    """

    backend = "structured"

    def __init__(
        self,
        system: FactorSmoothLeafSystem,
        penalized: FactorSmoothPenalizedOperator,
        *,
        max_structured_inverse_block: int = 256,
        excluded: tuple[int, ...] = (),
    ):
        with narrow_kernel_blas_threads(system.leaf.width):
            self._construct(system, penalized, max_structured_inverse_block, excluded)

    def _construct(
        self,
        system: FactorSmoothLeafSystem,
        penalized: FactorSmoothPenalizedOperator,
        max_structured_inverse_block: int,
        excluded: tuple[int, ...],
    ) -> None:
        leaf = system.leaf
        operator = system.operator
        if penalized.penalty_local is None or penalized.penalty_small is None:
            raise ValueError("An fs factor needs the penalized operator's penalty parts.")
        self.system = system
        self.penalized = penalized
        self.term_name = system.dominant_group_name
        self.dominant_group_name = system.dominant_group_name
        self.max_structured_inverse_block = int(max_structured_inverse_block)
        K, k = operator.n_levels, operator.block_size
        q = len(operator.small_indices)
        p = leaf.width
        self.n_levels, self.block_size = K, k
        self.small_indices = np.concatenate(([0], operator.small_indices + 1)).astype(np.intp)
        self.structured_indices = (operator.structured_indices + 1).astype(np.intp)
        size = operator.shape[0] + 1
        self.shape = (size, size)
        self._small_position = np.full(size, -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q + 1)
        self._structured_position = np.full(size, -1, dtype=np.intp)
        self._structured_position[self.structured_indices.ravel()] = np.arange(K * k)
        self._center = leaf.center
        c_full = np.concatenate(([0.0], leaf.center))
        self._c_full = c_full

        # Per lambda: the level penalties' square roots inside one QR per level.
        P = np.asarray(penalized.penalty_local, dtype=np.float64)
        roots = _penalty_roots(P)
        width = p - k  # intercept, border, right-hand side
        Qx = np.zeros((width, width)) if leaf.within is None else np.array(leaf.within, copy=True)
        U = np.zeros((K, k, k))
        F = np.zeros((K, k, width))
        Dinv = np.zeros((K, k, k))
        logdets = np.zeros(K)
        kappa = np.ones(K)
        sigma_extra = np.zeros(K)
        if leaf.signed:
            assert leaf.pseudo_rows is not None and leaf.signature is not None
            failed = _signed_trials(
                np.ascontiguousarray(leaf.pseudo_rows),
                np.ascontiguousarray(leaf.signature),
                roots,
                k,
                Qx,
                U,
                F,
                Dinv,
                logdets,
                kappa,
                sigma_extra,
            )
        else:
            failed = _fisher_trials(leaf.top, roots, k, Qx, U, F, Dinv, logdets)
        if failed >= 0:
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} level {failed} has a pivot block that is "
                "not positive definite (the tree pivot certificate refuses the iterate)."
            )
        if not (np.all(np.isfinite(F)) and np.all(np.isfinite(Dinv)) and np.all(np.isfinite(Qx))):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} has a non-representable elimination."
            )
        Qx = 0.5 * (Qx + Qx.T)
        self.level_logdets = logdets
        self._U = U
        self._D_inv_cache = Dinv
        self._F_centred = F[:, :, : q + 1]
        self._level_rhs = F[:, :, q + 1]
        diagonal_U = np.abs(np.diagonal(U, axis1=1, axis2=2))
        self.minimum_local_diagonal = float(np.min(diagonal_U) ** 2) if K else 0.0
        Q_xx = Qx[: q + 1, : q + 1]
        self._q_xz = Qx[: q + 1, q + 1].copy()

        # The border bound (module docstring) on the intercept-and-border Gram.
        bound = self._border_bound(leaf, roots, P, F[:, :, : q + 1], Dinv, Q_xx, kappa, sigma_extra)

        # The super-root: the intercept, eliminated last (design §3.1).
        D0 = float(Q_xx[0, 0])
        super_floor = bound[0] + 4.0 * _EPS * abs(D0)
        if D0 < -super_floor:
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} has materially negative curvature along "
                f"the intercept: the super-root pivot {D0:.6g} is below minus its certified "
                f"uncertainty {super_floor:.3g}, so the Hessian is indefinite there."
            )
        if not D0 > super_floor:
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} is aliased with the fitted intercept: "
                "the super-root pivot is within its certified uncertainty."
            )
        center_star = Q_xx[1:, 0] / D0
        Q_rest = Q_xx[1:, 1:] - D0 * np.outer(center_star, center_star)
        Q_rest = 0.5 * (Q_rest + Q_rest.T)
        # |d(Q_ij - q_i q_j / D0)| <= gamma_3 (|Q_ij| + |q_i q_j| / D0) <= 2 gamma_3 sqrt(Q_ii Q_jj),
        # and the full Gram's own majorant carried through the profiling (first order).
        diag_full = np.abs(np.diag(Q_xx))[1:]
        rest_bound = (np.sqrt(bound[1:]) + np.abs(center_star) * math.sqrt(bound[0])) ** 2
        rest_bound = rest_bound + 2.0 * _gamma(3) * diag_full
        S_rest = np.asarray(penalized.penalty_small, dtype=np.float64)
        generators = leaf.generators
        self.excluded = tuple(int(index) for index in excluded)
        if self.excluded:
            rest = self._small_position[np.asarray(self.excluded, dtype=np.intp)] - 1
            if np.any(rest < 0):
                raise ValueError("Only border slope columns can be excluded from an fs factor.")
            if generators is not None and np.any(generators.matrix[rest]):
                raise ValueError("An excluded border column cannot carry a structural generator.")
            Q_rest, S_rest, rest_bound = Q_rest.copy(), S_rest.copy(), rest_bound.copy()
            Q_rest[rest, :] = Q_rest[:, rest] = 0.0
            S_rest[rest, :] = S_rest[:, rest] = 0.0
            rest_bound[rest] = 0.0
        border = factor_border(
            Q_rest,
            S_rest,
            rest_bound,
            generators,
            term_name=self.term_name,
            term_kind="FactorSmooth term",
        )
        self._border = border
        self._center_star = center_star
        self._intercept_pivot = D0
        self.border_certificate: BorderCertificate = border.certificate
        certificate = border.certificate
        self.weakly_identified_coefficients: tuple[int, ...] = tuple(
            sorted(
                {
                    int(self.small_indices[column + 1])
                    for direction, weak in zip(
                        certificate.directions, certificate.weak, strict=True
                    )
                    if weak
                    for column in direction
                }
            )
        )
        self._scaled_eigenvalues_cache: NDArray | None = None
        self.schur_condition_estimate = border.condition
        null = border.null
        self._logdet = float(np.sum(logdets) + math.log(D0) + border.logdet)
        self.rank = int(K * k + q + 1 - null.shape[1])
        self.rank_truncated = self.rank < size
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None

    def published(self, system: FactorSmoothLeafSystem) -> FactorSmoothLeafFactor:
        """This factor on ``system``, its published copy (``published_leaf_system``).

        Every number is this factor's own; only the system it was built on is
        replaced, which no solve, inverse or trace reads.
        """
        clone = copy.copy(self)
        clone.system = system
        return clone

    @cached_property
    def _F(self) -> NDArray:
        """``F`` in raw coordinates, ``F_raw[:, :, j] = F_c[:, :, j] + c0_j F_c[:, :, 0]`` (``C = C_c T``).

        Formed on first read (only raw data operators read it, ``_inverse_bdlr``);
        owner and lifetime the factor, never pickled (``__getstate__``).
        """
        return self._F_centred + self._F_centred[:, :, :1] * self._c_full[None, None, :]

    # The (K, k, q)-sized products formed on first read from the factor's own
    # numbers (Opus review P1): dropped from a pickle and formed again, by the
    # same arithmetic, on first use after loading.  Owner: the factor; lifetime:
    # the factor in memory; nothing invalidates them.  The small caches (the
    # border inverse, the retained projector, the scaled eigenvalues) are kept.
    _REBUILT_ON_USE = ("_F", "_inverse_bdlr_centred", "_penalty_total", "_penalty_product")

    def __getstate__(self) -> dict:
        state = {
            name: value for name, value in self.__dict__.items() if name not in self._REBUILT_ON_USE
        }
        state["_inverse_bdlr_cache"] = None
        return state

    @cached_property
    def minimum_local_eigenvalue(self) -> float:
        """``min_l lambda_min(D_l)`` (reporting only; formed on first read from the pivots)."""
        if not self.n_levels:
            return 0.0
        return float(np.min(np.linalg.eigvalsh(np.einsum("kai,kaj->kij", self._U, self._U))))

    # -- the bound ---------------------------------------------------------
    @staticmethod
    def _border_bound(
        leaf: FactorSmoothLeafData,
        roots: NDArray,
        P: NDArray,
        F: NDArray,
        Dinv: NDArray,
        Q_xx: NDArray,
        kappa: NDArray,
        sigma_extra: NDArray,
    ) -> NDArray:
        """``U`` ``(q + 1,)`` with ``|dQ_ij| <= sqrt(U_i U_j)`` on the intercept-and-border Gram."""
        top = leaf.top
        K, p = top.shape[0], top.shape[2]
        k = leaf.block_size
        q1 = F.shape[2]
        if K == 0:
            return np.zeros(q1)
        counts = leaf.counts.astype(np.float64)
        merges = leaf.merges.astype(np.float64)
        # augmented unsigned residuals ||A v_j|| of every column j, v_j = [-F_j; e_j]:
        # the triangle's rows below its leading k are zero on the level basis, so
        # theirs is the trailing rows' column norm whatever F is
        residual_rows = top[:, :, k : k + q1] - np.matmul(top[:, :, :k], F)
        penalty_part = np.matmul(roots, F)
        r2 = np.sum(residual_rows**2, axis=1) + leaf.tail_norms2[:, :q1]
        r2 = r2 + np.sum(penalty_part**2, axis=1)
        # column scales a~ of the error-weighted rows (Fisher: e = w, the triangle's columns)
        if leaf.signed and leaf.error_diagonal is not None:
            a2 = leaf.error_diagonal.copy()
        else:
            a2 = np.sum(top[:, :, : p - 1] ** 2, axis=1)
            a2[:, k:] += leaf.tail_norms2[:, : p - 1 - k]
        a2[:, :k] += np.diagonal(P, axis1=1, axis2=2)
        a = np.sqrt(np.maximum(a2, 0.0))
        v_tilde = a[:, k : k + q1] + np.matmul(a[:, None, :k], np.abs(F))[:, 0, :]
        m_trial = (p + k) if leaf.signed else 2 * k
        g_row = (
            2.0 * _gamma_array(2.0 * np.maximum(counts, 1.0) * p)
            + 2.0 * merges * _gamma(4.0 * p * p)
            + 2.0 * _gamma(2.0 * m_trial * p)
            + _gamma(2)
        )
        if leaf.signed:
            g_row = g_row + math.sqrt(p) * _gamma(p + 1)
        eta = g_row[:, None] * v_tilde
        if leaf.signed and leaf.error_rows is not None and leaf.error_diagonal is not None:
            # v_j' E v_j with v_j = [-F_j; e_j]: F_j' E_zz F_j - 2 F_j' E_zx e_j + E_jj,
            # from the error Gram's leading rows and diagonal alone, without forming v
            E = leaf.error_rows
            re2 = np.sum(F * np.matmul(E[:, :, :k], F), axis=1)
            re2 = re2 - 2.0 * np.sum(F * E[:, :, k : k + q1], axis=1)
            re2 = re2 + leaf.error_diagonal[:, k : k + q1]
            re2 = np.maximum(re2, 0.0) + np.sum(penalty_part**2, axis=1)
        else:
            re2 = r2
        # sandwiched errors, per level
        if leaf.signed:
            sigma = (
                2.0 * math.sqrt(p) * _gamma_array(2.0 * np.maximum(counts, 1.0) * p)
                + p * _gamma_array(np.maximum(counts, 1.0))
                + p * p * _UNIT
                + 2.0 * math.sqrt(p) * _gamma(2.0 * m_trial * p)
                + p * _gamma(m_trial)
                + merges * (2.0 * math.sqrt(p) * _gamma(4.0 * p * p) + p * _gamma(2.0 * p))
                + sigma_extra
            )
            sigma = sigma + sigma**2 * kappa
        else:
            sigma = np.full(K, _gamma(2 * k) + _gamma(2 * p))
        # second order through the Schur complement's inverse: kappa and omega
        az = a[:, :k]
        Dinv_abs = np.abs(Dinv)
        omega_norm = np.sqrt(np.sum((az[:, :, None] * Dinv_abs * az[:, None, :]) ** 2, axis=(1, 2)))
        omega = g_row**2 * k * omega_norm
        r = np.sqrt(np.maximum(r2, 0.0))
        rbar = r + eta
        d = np.abs(np.diag(Q_xx))
        live = d > 0.0
        rho = float(np.sum(r2[:, live] / d[live])) if np.any(live) else 0.0
        nu = float(np.sum(eta[:, live] ** 2 / d[live])) if np.any(live) else 0.0
        t = math.sqrt(nu / rho) if rho > 0.0 and nu > 0.0 else 1.0
        second = (np.sqrt(kappa)[:, None] * eta + np.sqrt(omega)[:, None] * rbar) ** 2
        U = np.sum(
            t * r2 + eta**2 / t + eta**2 + sigma[:, None] * rbar**2 + _gamma(16) * re2 + second,
            axis=0,
        )
        U = U + _gamma(K + 1) * np.sum(rbar**2, axis=0)
        return np.nextafter(U, np.inf)

    # -- Q^+ in raw coordinates ---------------------------------------------
    @cached_property
    def _Q_inverse_centred(self) -> NDArray:
        return _super_root_inverse(self._border.inverse, self._center_star, self._intercept_pivot)

    @cached_property
    def _Q_inverse_raw(self) -> NDArray:
        """``(I - e_0 c') Q^+ (I - c e_0')``: the centred border inverse in raw coordinates."""
        inverse = self._Q_inverse_centred
        c = self._c_full
        column = inverse @ c
        raw = inverse.copy()
        raw[0, :] -= column
        raw[:, 0] -= column
        raw[0, 0] += float(c @ column)
        return raw

    @property
    def _D_inv(self) -> NDArray:
        return self._D_inv_cache

    def _local_solve(self, rhs: NDArray) -> NDArray:
        return np.einsum("kij,kjm->kim", self._D_inv_cache, rhs, optimize=True)

    # -- solves ---------------------------------------------------------------
    def solve(
        self,
        rhs: NDArray,
        *,
        centred: bool = False,
        border_centred: NDArray | None = None,
    ) -> NDArray:
        """``H^-1 rhs`` in raw coordinates for ``(p,)`` or ``(p, m)``, through the centred factor.

        ``R' r`` in (the border rows lose ``c`` times the intercept row), the
        centred solve, then ``R x`` out (only the intercept entry moves; with
        ``centred`` it stays the centred ``alpha``), as ``NestedSchurFactor``.
        ``border_centred``, when given, is the border part of ``rhs`` already
        in centred coordinates, in place of that subtraction, as
        ``SumToZeroTreeFactor.solve``'s.
        """
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        rhs_small = values[self.small_indices]
        rhs_small = rhs_small - self._c_full[:, None] * rhs_small[:1]
        if border_centred is not None:
            rhs_small[1:] = np.asarray(border_centred, dtype=np.float64).reshape(
                len(self.small_indices) - 1, -1
            )
        rhs_structured = values[self.structured_indices]
        D_inv_rhs = self._local_solve(rhs_structured)
        # C' D^-1 r_t = F' r_t
        schur_rhs = rhs_small - np.einsum(
            "kiq,kim->qm", self._F_centred, rhs_structured, optimize=True
        )
        solution_small = self._Q_inverse_centred @ schur_rhs
        solution = np.empty_like(values)
        solution[self.structured_indices] = D_inv_rhs - np.einsum(
            "kiq,qm->kim", self._F_centred, solution_small, optimize=True
        )
        if not centred:
            solution_small[:1] -= self._c_full @ solution_small
        solution[self.small_indices] = solution_small
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} solve is not representable."
            )
        return solution[:, 0] if vector_rhs else solution

    def solve_data(self, *, centred: bool = False) -> NDArray:
        """``H^-1 [1 X]' W z`` from the right-hand side inside the leaf factorization.

        The border part is solved through the data-side border inverse
        (``BorderFactor.apply_data``), the super-root first; the level part is
        ``D_l^-1 c_zl - F_l x_b``.  With ``centred`` the intercept entry stays
        the centred ``alpha`` about ``c0`` (one-engine design §3.8).
        """
        q_xz = self._q_xz
        rest = q_xz[1:] - self._center_star * q_xz[0]
        x_rest = self._border.apply_data(rest)
        alpha = q_xz[0] / self._intercept_pivot - float(self._center_star @ x_rest)
        border = np.concatenate(([alpha], x_rest))
        solution = np.empty(self.shape[0])
        solution[self.structured_indices] = self._level_rhs - np.einsum(
            "kij,j->ki", self._F_centred, border, optimize=True
        )
        if not centred:
            border = border.copy()
            border[0] -= float(self._center @ x_rest)
        solution[self.small_indices] = border
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} solve is not representable."
            )
        return solution

    def logdet(self) -> float:
        return self._logdet

    def row_quadratic_forms(self, rows) -> NDArray:
        """``a_i' H^+ a_i`` per augmented data row ``a_i = [1, x_i]`` of ``rows`` ``(m, p)``.

        In the centred coordinates ``H^+ = [[D^-1 + F Q^+ F', -F Q^+], [-Q^+ F',
        Q^+]]`` (``D = blockdiag(D_l)``), so a row with level part ``z`` in its
        level ``l`` and border part ``a`` (``R'`` applied: ``[1, x - c0]``)
        gives ``||U_l^-T z||^2 + y' Q^+ y`` with ``y = a - F_l' z``, ``D_l = U_l'
        U_l``: the random-effect and fixed-effect halves of the hat diagonal of
        Bates et al. (2015, eqs. 63-65), as ``NestedSchurFactor``.  The level
        half is the squared norm of a row of the level's orthogonal factor, a
        sum of squares with no cancellation; ``y' Q^+ y`` takes the super-root,
        then the data side of the border (a data row has no component along
        the deflated structural nulls, ``BorderFactor.quadratic_data``).  No
        ``p x p`` block is formed: O(nnz k + m q^2).  Leverage (design §3.10,
        definition A) is ``w_i`` times this.
        """
        import scipy.sparse

        values = scipy.sparse.csr_array(rows, dtype=np.float64)
        if values.shape[1] != self.shape[0]:
            raise ValueError(f"rows must have shape (m, {self.shape[0]}), got {values.shape}.")
        m = values.shape[0]
        K, k = self.n_levels, self.block_size
        border = values[:, self.small_indices].toarray()
        # R' a: the centred border columns lose c0 times the intercept entry
        border[:, 1:] -= border[:, :1] * self._center[None, :]
        pieces = values[:, self.structured_indices.ravel()].tocoo()
        row, column = (np.asarray(index, dtype=np.intp) for index in pieces.coords)
        keys, slot = np.unique(row * K + column // k, return_inverse=True)
        Z = np.zeros((len(keys), k))
        np.add.at(Z, (slot, column % k), pieces.data)
        tree = np.empty(len(keys))
        cross = np.empty((len(keys), border.shape[1]))
        _level_row_forms(self._U, np.ascontiguousarray(self._F_centred), keys % K, Z, tree, cross)
        owner = keys // K
        np.subtract.at(border, owner, cross)
        rest = border[:, 1:] - border[:, :1] * self._center_star[None, :]
        result = np.bincount(owner, weights=tree, minlength=m)
        result += border[:, 0] ** 2 / self._intercept_pivot + self._border.quadratic_data(rest)
        if not np.all(np.isfinite(result)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} row quadratic forms are not representable."
            )
        return result

    def scaled_schur_eigenvalues(self) -> NDArray:
        """Ascending eigenvalues of the scaled deflated border matrix, cached (``NestedSchurFactor``)."""
        if self._scaled_eigenvalues_cache is None:
            self._scaled_eigenvalues_cache = np.linalg.eigvalsh(self._border.scaled_matrix)
        return self._scaled_eigenvalues_cache

    def _validate_selected_indices(self, indices: NDArray) -> NDArray[np.intp]:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def _centred_rows(self, selected: NDArray) -> NDArray:
        """Rows of the centred ``H_c^+`` at ``selected`` against every border coordinate ``(m, q + 1)``."""
        small = self._small_position[selected]
        rows = np.empty((len(selected), len(self.small_indices)))
        border = small >= 0
        rows[border] = self._Q_inverse_centred[small[border]]
        if np.any(~border):
            F_flat = self._F_centred.reshape(self.n_levels * self.block_size, -1)
            positions = self._structured_position[selected[~border]]
            rows[~border] = -F_flat[positions] @ self._Q_inverse_centred
        return rows

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """The raw-coordinate inverse on ``indices``, from the centred factor.

        Off the intercept every entry of ``H^-1`` is the same in both
        coordinates (``R = I - e_0 c'`` moves only the intercept row), so they
        are formed from the centred ``F`` and ``Q^+``, free of the columns'
        offsets; the intercept's row and column take the map.
        """
        selected = self._validate_selected_indices(indices)
        small_mask = self._small_position[selected] >= 0
        structured = np.flatnonzero(~small_mask)
        positions = self._structured_position[selected[structured]]
        if len(positions) > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {len(positions)} x {len(positions)} inverse block "
                f"for structured term {self.term_name!r}; request its diagonal instead."
            )
        rows = self._centred_rows(selected)
        columns = np.empty((len(self.small_indices), len(selected)))
        small = self._small_position[selected]
        columns[:, small_mask] = np.eye(len(self.small_indices))[:, small[small_mask]]
        F_flat = self._F_centred.reshape(self.n_levels * self.block_size, -1)
        columns[:, structured] = -F_flat[positions].T
        inverse = rows @ columns
        # the tree's own block: D_l^-1 within a level
        levels = positions // self.block_size
        coordinates = positions % self.block_size
        for row, (level, coordinate) in enumerate(zip(levels, coordinates, strict=True)):
            same = np.flatnonzero(levels == level)
            inverse[structured[row], structured[same]] += self._D_inv_cache[
                level, coordinate, coordinates[same]
            ]
        intercept = np.flatnonzero(selected == 0)
        if intercept.size:
            # R x out: row and column 0 lose c' times the border rows (centred)
            c = self._c_full
            border_rows = rows @ c  # (H_c^+ c) at the selected coordinates
            row0 = int(intercept[0])
            inverse[row0, :] -= border_rows
            inverse[:, row0] -= border_rows
            inverse[row0, row0] += float(c @ self._Q_inverse_centred @ c)
        inverse = 0.5 * (inverse + inverse.T)
        if not np.all(np.isfinite(inverse)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} selected inverse is not representable."
            )
        return inverse

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        diagonal = np.empty(len(selected), dtype=np.float64)
        small_mask = self._small_position[selected] >= 0
        if np.any(small_mask):
            small_position = self._small_position[selected[small_mask]]
            diagonal[small_mask] = np.diag(self._Q_inverse_raw)[small_position]
        if np.any(~small_mask):
            positions = self._structured_position[selected[~small_mask]]
            levels = positions // self.block_size
            coordinates = positions % self.block_size
            F_selected = self._F_centred.reshape(self.n_levels * self.block_size, -1)[positions]
            diagonal[~small_mask] = self._D_inv_cache[levels, coordinates, coordinates] + np.sum(
                (F_selected @ self._Q_inverse_centred) * F_selected, axis=1
            )
        if not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError(
                f"FactorSmooth term {self.term_name!r} selected inverse is not representable."
            )
        return diagonal

    def _bdlr(self, F: NDArray, core: NDArray) -> _BlockDiagonalLowRank:
        q = len(self.small_indices)
        basis = np.zeros((self.shape[0], q), dtype=np.float64)
        basis[self.small_indices] = np.eye(q)
        basis[self.structured_indices] = -F
        return _BlockDiagonalLowRank(
            blocks=self._D_inv,
            structured_indices=self.structured_indices,
            basis=basis,
            core=core,
            shape=self.shape,
        )

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        """``H^-1`` in raw coordinates, for raw data operators (moment operators)."""
        if self._inverse_bdlr_cache is None:
            self._inverse_bdlr_cache = self._bdlr(self._F, self._Q_inverse_raw)
        return self._inverse_bdlr_cache

    @cached_property
    def _inverse_bdlr_centred(self) -> _BlockDiagonalLowRank:
        """``H_c^+`` in the centred coordinates: penalty traces and the slope block ``M_ss``.

        A penalty operator is the same in both coordinates (``S e_0 = 0``) and so
        is every trace with it, and ``M_ss``; the centred form never carries a
        column's offset (cache owner: the factor; lifetime: the factor).
        """
        return self._bdlr(self._F_centred, self._Q_inverse_centred)

    @cached_property
    def _penalty_total(self) -> _BlockDiagonalLowRank:
        """The whole penalty ``S`` in augmented coordinates (border and level parts).

        The form ``_operator_bdlr`` gives a block operator with no cross block,
        built without that ``(K, k, q)`` block of zeros: the level penalties
        as the blocks, and the border penalty (when it is not zero) on the
        border's unit basis.
        """
        q = len(self.small_indices)
        small, local = self.penalized.penalty_small, self.penalized.penalty_local
        assert small is not None and local is not None  # checked at construction
        A = np.zeros((q, q))
        A[1:, 1:] = small
        blocks = np.array(local, dtype=np.float64, copy=True)
        blocks.setflags(write=False)
        if not np.any(A):
            return _BlockDiagonalLowRank(
                blocks=blocks,
                structured_indices=self.structured_indices,
                basis=np.empty((self.shape[0], 0)),
                core=np.empty((0, 0)),
                shape=self.shape,
            )
        basis = np.zeros((self.shape[0], q), dtype=np.float64)
        basis[self.small_indices] = np.eye(q)
        A.setflags(write=False)
        return _BlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=self.structured_indices,
            basis=basis,
            core=A,
            shape=self.shape,
        )

    @cached_property
    def _retained_own(self) -> NDArray:
        """``diag(H^+ H)`` on the border ``(q + 1,)``: 1 unless truncated (as ``NestedSchurFactor``)."""
        border = self._border
        return np.concatenate(([1.0], 1.0 - np.einsum("ij,ij->i", border.null, border.null_left)))

    @cached_property
    def _penalty_product(self):
        """``H_c^+ S`` as a general block-diagonal-plus-low-rank product (centred)."""
        return _multiply_symmetric_bdlr(self._inverse_bdlr_centred, self._penalty_total)

    def _identity_diagonal(self) -> NDArray:
        """``diag(H^+ (H - S)) = diag(H^+ H) - diag(H^+ S)``: the own data operator's route."""
        diagonal = -_general_bdlr_diagonal(self._penalty_product)
        diagonal[self.structured_indices] += 1.0
        diagonal[self.small_indices] += self._retained_own
        return diagonal

    def _identity_square_diagonal(self) -> NDArray:
        """``diag((H^+ (H - S))^2) = diag(P) - 2 diag(H^+ S) + diag((H^+ S)^2)``, ``P = H^+ H``."""
        product = self._penalty_product
        diagonal = _general_bdlr_square_diagonal(product) - 2.0 * _general_bdlr_diagonal(product)
        diagonal[self.structured_indices] += 1.0
        diagonal[self.small_indices] += self._retained_own
        return diagonal

    def _penalty_operator(
        self, component: PenaltyComponent, scale: float
    ) -> BlockSymmetricOperator:
        q = len(self.small_indices)
        indices = _component_indices(component, self.shape[0])
        local_small = self._small_position[indices]
        local_structured = self._structured_position[indices]
        A = np.zeros((q, q))
        C = np.zeros((self.n_levels, self.block_size, q))
        D = np.zeros((self.n_levels, self.block_size, self.block_size))
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
            elif np.all(local_structured >= 0):
                D[
                    local_structured // self.block_size,
                    local_structured % self.block_size,
                    local_structured % self.block_size,
                ] = scale
            else:
                raise ValueError("Identity penalty crosses structured partitions.")
        elif component.penalty_kind == "repeated":
            if not np.all(local_structured >= 0):
                raise ValueError("Repeated penalty must lie in the structured block.")
            if component.repeat_count != self.n_levels or component.block_width != self.block_size:
                raise ValueError("Repeated penalty geometry does not match the block factor.")
            if not np.array_equal(
                indices.reshape(self.n_levels, self.block_size), self.structured_indices
            ):
                raise ValueError("Repeated penalty ordering does not match the block factor.")
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (self.block_size, self.block_size):
                raise ValueError("Repeated penalty local matrix has the wrong shape.")
            D[:] = scale * omega
        else:
            if not np.all(local_small >= 0):
                raise ValueError("Dense penalties must lie in the block factor's small block.")
            A[np.ix_(local_small, local_small)] = scale * _component_omega(component, self.shape[0])
        return BlockSymmetricOperator(
            A=A,
            C=C,
            D=D,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return _trace_symmetric_bdlr(
            self._inverse_bdlr_centred,
            _operator_bdlr(self._penalty_operator(component, 1.0), self.structured_indices),
        )

    def penalty_cross_trace(
        self, left: PenaltyComponent, right: PenaltyComponent, left_scale: float, right_scale: float
    ) -> float:
        inverse = self._inverse_bdlr_centred
        return _trace_general_bdlr_product(
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(self._penalty_operator(left, left_scale), self.structured_indices),
            ),
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(self._penalty_operator(right, right_scale), self.structured_indices),
            ),
        )

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return ``trace(H^-1 O_i H^-1 O_j)`` for every pair, ``O = scale * Omega + dH``.

        Penalty-only directions take the centred inverse; a direction with a raw
        data operator takes the raw one it is expressed against.
        """
        if all(operator is None for _, _, operator in directions):
            return _block_derivative_cross_traces(_CentredView(self), directions)
        return _block_derivative_cross_traces(self, directions)

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
        )

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
            )
        )

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
            )
        )

    def operator_cross_trace(
        self, left: CompactSymmetricOperator, right: CompactSymmetricOperator
    ) -> float:
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_bdlr()
        return _trace_general_bdlr_product(
            _multiply_symmetric_bdlr(inverse, _operator_bdlr(left, self.structured_indices)),
            _multiply_symmetric_bdlr(inverse, _operator_bdlr(right, self.structured_indices)),
        )

    def penalty_operator_cross_trace(
        self, component: PenaltyComponent, scale: float, operator: CompactSymmetricOperator
    ) -> float:
        return self.operator_cross_trace(self._penalty_operator(component, scale), operator)


class _CentredView:
    """The two attributes ``_block_derivative_cross_traces`` reads, on the centred inverse."""

    def __init__(self, factor: FactorSmoothLeafFactor) -> None:
        self._factor = factor

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        return self._factor._inverse_bdlr_centred

    def _penalty_operator(self, component: PenaltyComponent, scale: float):
        return self._factor._penalty_operator(component, scale)


class ProfiledFactorSmoothLeafFactor:
    """Slope inverse ``H_c^-1 = M_ss`` of an ``fs`` fit with the intercept profiled out.

    ``M_ss``, the slope block of ``H_aug^-1``, is the same in the raw and the
    centred coordinates, so every product here reads the augmented factor's
    centred ``F`` and ``Q^+`` (``_inverse_bdlr``), free of the columns'
    offsets.  The factor's own centred data operator (the ``edf`` and
    ``edf1`` operator: ``CenteredBlockOperator(system.operator, xtw, sum_w,
    mean_x)``) takes the identity routes ``diag(H^+ (H - S))`` and its square,
    through the penalty alone (``FactorSmoothLeafFactor._identity_diagonal``):
    the moment form ``A - C'D^-1 C`` would carry ``eps`` times the raw mass
    into directions only the penalty pins.  ``logdet()`` is the augmented
    ``logdet - log(sum_w)``.
    """

    backend = "structured"

    def __init__(
        self,
        *,
        augmented_factor: FactorSmoothLeafFactor,
        sum_w: float,
        xtw: NDArray,
    ):
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = np.asarray(xtw, dtype=np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        if augmented_factor.shape != (len(self.xtw) + 1, len(self.xtw) + 1):
            raise ValueError("Augmented factor width does not match xtw.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.n_levels = augmented_factor.n_levels
        self.block_size = augmented_factor.block_size
        self.max_structured_inverse_block = augmented_factor.max_structured_inverse_block
        self.border_certificate = augmented_factor.border_certificate
        self.weakly_identified_slopes = tuple(
            index - 1 for index in augmented_factor.weakly_identified_coefficients if index > 0
        )
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None

    @property
    def minimum_local_eigenvalue(self) -> float:
        return self.augmented_factor.minimum_local_eigenvalue

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_inverse_bdlr_cache"] = None  # (p, q) basis, formed again on first use
        return state

    def published(self, system: FactorSmoothLeafSystem | None = None):
        """This factor with its augmented factor on the published ``system`` (its own, published, if ``None``)."""
        augmented = self.augmented_factor
        clone = copy.copy(self)
        clone.augmented_factor = augmented.published(
            augmented.system.published() if system is None else system
        )
        return clone

    @staticmethod
    def _shift_indices(indices: NDArray) -> NDArray[np.intp]:
        return np.asarray(indices, dtype=np.intp) + 1

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start = component.group_sl.start
        stop = component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return dataclasses.replace(
            component, group_sl=slice(start + 1, stop + 1, component.group_sl.step)
        )

    def _is_own_data(self, operator) -> bool:
        """Whether ``operator`` is the factor's centred data operator (the identity routes).

        Either form of it: ``system.centred_data_operator`` (on the ``c0``-shifted
        moments) or the raw moments centred on ``mean_x``; the identity routes
        read neither's numbers.
        """
        return _is_own_centred_data(operator, self.augmented_factor.system, self.xtw, self.sum_w)

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        augmented_rhs = np.zeros((self.shape[0] + 1, values.shape[1]))
        augmented_rhs[1:] = values
        solution = self.augmented_factor.solve(augmented_rhs, centred=True)[1:]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_block(self._shift_indices(indices))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_diagonal(self._shift_indices(indices))

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left),
            self._shift_component(right),
            left_scale,
            right_scale,
        )

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        """``M_ss`` from the centred factor (its slope rows; the intercept column stays in the core)."""
        cached = self._inverse_bdlr_cache
        if cached is not None:
            return cached
        augmented = self.augmented_factor
        q_augmented = len(augmented.small_indices)
        basis = np.zeros((self.shape[0], q_augmented), dtype=np.float64)
        if len(self.small_indices):
            basis[self.small_indices, 1:] = np.eye(len(self.small_indices))
        basis[self.structured_indices] = -augmented._F_centred
        cached = _BlockDiagonalLowRank(
            blocks=augmented._D_inv,
            structured_indices=self.structured_indices,
            basis=basis,
            core=augmented._Q_inverse_centred,
            shape=self.shape,
        )
        self._inverse_bdlr_cache = cached
        return cached

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if self._is_own_data(operator):
            return float(np.sum(self.augmented_factor._identity_diagonal()[1:]))
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
        )

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if self._is_own_data(operator):
            return self.augmented_factor._identity_diagonal()[1:]
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
            )
        )

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if self._is_own_data(operator):
            return self.augmented_factor._identity_square_diagonal()[1:]
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(), _operator_bdlr(operator, self.structured_indices)
            )
        )

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_bdlr()
        return _trace_general_bdlr_product(
            _multiply_symmetric_bdlr(inverse, _operator_bdlr(left, self.structured_indices)),
            _multiply_symmetric_bdlr(inverse, _operator_bdlr(right, self.structured_indices)),
        )

    def _penalty_operator(
        self, component: PenaltyComponent, scale: float
    ) -> BlockSymmetricOperator:
        penalty = self.augmented_factor._penalty_operator(self._shift_component(component), scale)
        return BlockSymmetricOperator(
            A=penalty.A[1:, 1:],
            C=penalty.C[:, :, 1:],
            D=penalty.D,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        return self.operator_cross_trace(self._penalty_operator(component, scale), operator)

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return ``trace(H^-1 O_i H^-1 O_j)`` for every pair, ``O = scale * Omega + dH``."""
        return _block_derivative_cross_traces(self, directions)
