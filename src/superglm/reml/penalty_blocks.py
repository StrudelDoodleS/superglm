"""Block-wise ``log|S_lambda|_+`` and its log-lambda derivatives on a fixed penalty support.

``S_lambda`` is block diagonal over terms, so its generalised determinant and
its derivatives in ``rho = log(lambda)`` are sums over blocks, with no second
derivatives between blocks (Wood, Pya and Saefken 2016, JASA 111, sec. 3.1.1):

- a single-penalty block contributes ``rank(S) log(lambda) + log|S|_+``; the
  constant is formed once, its gradient is the rank and its Hessian is zero;
- a multi-lambda block ``sum_j lambda_j S_j`` is evaluated for each proposal
  after the similarity transform of Wood (2011, JRSS-B 73(1), Appendix B),
  which concentrates the dominant terms into their own leading columns, so the
  rounding of a term with a large weight cannot leak into the eigenvalues the
  smaller terms set;
- a tensor block ``lambda_1 A (x) I + lambda_2 I (x) B`` has the eigenvalues
  ``lambda_1 a_i + lambda_2 b_j``.

Every block works on component roots ``R_j`` with ``R_j' R_j = S_j``, given in
the coordinates of the block's support (the range of the unweighted sum), so
``sum_j lambda_j S_j`` is nonsingular there for positive weights.  Appendix B
is carried out on those roots: the eigenvectors of ``sum_{alpha} lambda_i S_i``
are the right singular vectors of the stacked weighted roots, and Householder
QR of the transformed stacked root acts on columns without mixing them, as the
appendix's final QR of ``S`` does (Higham 2002, Theorem 19.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._roundoff import EPS
from superglm.reml.penalty_support import PenaltyNumericalError
from superglm.solvers.rank import SHARED_RANK_POLICY

#: Wood (2011), Appendix B step 2: a term dominates when its weighted norm is
#: at least this fraction of the largest remaining one.
DOMINANCE = EPS ** (1.0 / 3.0)


def _log_gram_norm(root: NDArray) -> float:
    """``log ||R' R||_F`` from the smaller Gram of ``R`` over its largest entry; ``-inf`` at zero.

    A finite root can have a Gram beyond binary64 (``1e155 I`` has norm
    ``1e310``); its logarithm stays finite.
    """
    scale = float(np.max(np.abs(root), initial=0.0))
    if scale == 0:
        return -math.inf
    unit = root / scale
    gram = unit @ unit.T if unit.shape[0] <= unit.shape[1] else unit.T @ unit
    return 2.0 * math.log(scale) + math.log(float(np.linalg.norm(gram)))


def similarity_transformed_roots(
    roots: Sequence[NDArray], weights: NDArray, *, dominance: float = DOMINANCE
) -> tuple[list[NDArray], list[float]]:
    """Weighted roots ``sqrt(lambda_j) R_j`` after Wood's (2011) Appendix B transform, and ``log Omega_j``.

    Each pass ranks the terms that still have content in the open trailing
    block by ``Omega_i = lambda_i ||S_i||_F`` there and takes the dominant set
    ``alpha``; it stops when every such term dominates.  Otherwise the block is
    rotated onto the right singular vectors of the stacked dominant roots (the
    eigenvectors of ``sum_alpha lambda_i S_i``), the leading columns whose
    dominant singular values are at least ``sqrt(eps)`` of the largest are
    closed, and the pass repeats on the rest.

    Every step is an orthogonal rotation of the trailing columns, so the
    transform discards no direction and the determinant is unchanged.  The
    appendix sets the dominant terms to zero beyond their formal rank because
    a formed ``lambda_i S_i`` carries rounding of order ``eps lambda_i ||S_i||``
    in every entry, which is first order in the eigenvalues the smaller terms
    set.  Stacked roots keep each term's rounding in that term's own rows: after
    the rotation a dominant root's trailing columns hold its true content there
    plus rounding of order ``u sqrt(lambda_i) ||R_i||``, which reaches the
    trailing Gram only squared and through the closed block's Schur complement.
    They are carried into the next pass, where they compete by their own norm,
    rather than cut at a fixed rank: a penalty whose spectrum runs below any such
    cut, as a column-equilibrated support keeps, still sets eigenvalues that the
    smaller terms' weighted ones can match.

    ``log Omega_j`` is taken on the whole support, in the first pass; the
    rotations leave it unchanged.
    """
    current = [
        math.sqrt(float(weight)) * np.array(root, dtype=np.float64)
        for root, weight in zip(roots, weights, strict=True)
    ]
    width = current[0].shape[1]
    log_dominance = math.log(dominance)
    start = 0
    magnitudes: dict[int, float] = {}
    while start < width:
        norms = {index: _log_gram_norm(root[:, start:]) for index, root in enumerate(current)}
        if start == 0:
            magnitudes = norms
        remaining = [index for index, value in norms.items() if value > -math.inf]
        if not remaining:
            break
        largest = max(norms[index] for index in remaining)
        alpha = [index for index in remaining if norms[index] >= largest + log_dominance]
        if len(alpha) == len(remaining):
            # Comparable terms need no further separation (step 4).
            break
        dominant = np.vstack([current[index][:, start:] for index in alpha])
        dominant /= np.max(np.abs(dominant))
        _, singular, vh = scipy.linalg.svd(dominant, full_matrices=True, check_finite=False)
        rotation = vh.T
        for index in remaining:
            current[index][:, start:] = current[index][:, start:] @ rotation
        start += int(np.count_nonzero(singular > SHARED_RANK_POLICY.factor_rcond * singular[0]))
    return current, [magnitudes.get(index, -math.inf) for index in range(len(current))]


def multi_penalty_logdet(
    roots: Sequence[NDArray], weights: NDArray
) -> tuple[float, NDArray, NDArray]:
    """``log|sum_j lambda_j S_j|`` on the support, and its gradient and Hessian in log weights.

    ``roots`` are the components' roots in support coordinates; every weight is
    positive.  When the roots' row counts add up to the support width, their
    ranges form a direct sum and ``log|S| = log|S(1)| + sum_j r_j log(lambda_j)``
    exactly, by multiplicativity of the determinant, with gradient ``r_j`` and
    a zero Hessian (Wood and Fasiolo 2017, sec. 2).  Otherwise, with ``R`` the
    QR factor of the transformed stacked root and ``Z_j = sqrt(lambda_j) R_j
    R^-1``, the derivatives are Wood's (2011) Appendix B traces
    ``lambda_j tr(S^-1 S_j) = ||Z_j||_F^2`` and, for ``i != j``,
    ``-lambda_i lambda_j tr(S^-1 S_i S^-1 S_j) = -||Z_i Z_j'||_F^2``; the
    diagonal is the sum of the off-diagonal traces, since ``sum_j Z_j' Z_j = I``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    count = len(roots)
    width = roots[0].shape[1]
    rows = np.array([len(root) for root in roots], dtype=np.float64)
    if rows.sum() == width:
        unit = np.vstack(roots)
        logdet = _root_logdet(unit) + math.fsum(
            float(r * math.log(w)) for r, w in zip(rows, weights, strict=True)
        )
        return logdet, rows, np.zeros((count, count))
    transformed, magnitudes = similarity_transformed_roots(roots, weights)
    # Householder QR is not row-wise backward stable without pivoting: a small
    # pivot row is reflected into the larger rows below it and lost at their
    # rounding.  With column pivoting, rows sorted by decreasing norm give the
    # row-wise bound of row pivoting (Cox and Higham 1998, Lemma 4.2; Powell and
    # Reid 1969).  The transform orders the columns by dominance, and the terms
    # are stacked by decreasing Omega_j, so the factor is independent of their order.
    order = sorted(range(count), key=lambda index: -magnitudes[index])
    stacked = np.vstack([transformed[index] for index in order])
    upper = scipy.linalg.qr(stacked, mode="r", check_finite=False)[0][:width]
    diagonal = np.abs(np.diag(upper))
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0):
        raise PenaltyNumericalError("weighted penalty roots do not span their support")
    logdet = 2.0 * math.fsum(np.log(diagonal).tolist())
    factors = [
        scipy.linalg.solve_triangular(upper, root.T, trans="T", check_finite=False).T
        for root in transformed
    ]
    gradient = np.array([float(np.sum(factor * factor)) for factor in factors])
    hessian = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            cross = factors[i] @ factors[j].T
            hessian[i, j] = hessian[j, i] = -float(np.sum(cross * cross))
    for i in range(count):
        hessian[i, i] = -math.fsum(hessian[i].tolist())
    return logdet, gradient, hessian


def _root_logdet(root: NDArray) -> float:
    """``log det(R' R)`` for a root ``R`` of full column rank, from its QR factor."""
    width = root.shape[1]
    if width == 0:
        return 0.0
    upper = scipy.linalg.qr(root, mode="r", check_finite=False)[0][:width]
    return 2.0 * math.fsum(np.log(np.abs(np.diag(upper))).tolist())


def unit_logdet(roots: Sequence[NDArray]) -> float:
    """``log|sum_j S_j|`` on the support from the stacked roots."""
    return _root_logdet(np.vstack(roots))


def map_volume(basis: NDArray, coordinate_map: NDArray) -> float | None:
    """``log det(B' B) - log det(Q' Q)`` for ``B = C' Q``: the change of ``log|S|_+`` under ``C``.

    For a penalty whose range is spanned by ``Q``, ``log|C' S C|_+ = log|S|_+ +
    log det(Q' C C' Q) - log det(Q' Q)`` whenever ``C' Q`` has full column rank
    (``AB`` and ``BA`` share their nonzero spectrum; Knill 2014, Cauchy-Binet
    for pseudo-determinants).  ``None`` when ``C' Q`` loses a column, which no
    determinant identity can survive.
    """
    rank = basis.shape[1]
    if rank == 0:
        return 0.0
    mapped = np.asarray(coordinate_map, dtype=np.float64).T @ np.asarray(basis, dtype=np.float64)
    upper = scipy.linalg.qr(mapped, mode="r", check_finite=False)[0][:rank]
    diagonal = np.abs(np.diag(upper))
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0):
        return None
    gram = scipy.linalg.cholesky(basis.T @ basis, lower=False, check_finite=False)
    return 2.0 * math.fsum(np.log(diagonal).tolist()) - 2.0 * math.fsum(
        np.log(np.diag(gram)).tolist()
    )


def kronecker_sum_logdet(
    left_log: NDArray, right_log: NDArray
) -> tuple[float, NDArray, NDArray, int]:
    """``log|lambda_1 A (x) I + lambda_2 I (x) B|_+`` from the margins' weighted log-eigenvalues.

    ``left_log`` holds ``log(lambda_1 a_i)``, ``-inf`` for a null ``a_i`` or a
    zero weight, and ``right_log`` likewise.  The Kronecker sum has the
    eigenvalues ``lambda_1 a_i + lambda_2 b_j`` (Higham, "What is the
    Kronecker product?", 2020: the margins' eigenvectors diagonalise both
    terms); a cell counts when either term is positive.  With ``f_ij = lambda_1 a_i /
    (lambda_1 a_i + lambda_2 b_j)``, the gradient is ``(sum f, sum (1 - f))``
    and the Hessian is ``c [[1, -1], [-1, 1]]`` with ``c = sum f (1 - f)``.
    Cells are summed in the log domain, so neither weight can overflow them.
    Returns the value, gradient, Hessian and cell count.
    """
    left_log = np.asarray(left_log, dtype=np.float64)
    right_log = np.asarray(right_log, dtype=np.float64)
    cell = np.logaddexp(left_log[:, None], right_log[None, :])
    active = np.isfinite(cell)
    retained = cell[active]
    left = np.broadcast_to(left_log[:, None], cell.shape)[active]
    right = np.broadcast_to(right_log[None, :], cell.shape)[active]
    left_fraction = np.exp(left - retained)
    right_fraction = np.exp(right - retained)
    # A one-sided cell contributes exactly one to its active derivative.
    left_fraction[~np.isfinite(right)] = 1.0
    right_fraction[~np.isfinite(left)] = 1.0
    cross = math.fsum((left_fraction * right_fraction).tolist())
    gradient = np.array([math.fsum(left_fraction.tolist()), math.fsum(right_fraction.tolist())])
    hessian = np.array([[cross, -cross], [-cross, cross]])
    return math.fsum(retained.tolist()), gradient, hessian, int(np.count_nonzero(active))


def kronecker_split(
    first: NDArray, second: NDArray
) -> tuple[int, int, NDArray, NDArray, bool] | None:
    """Margins ``(p1, p2, A, B, swapped)`` when the two penalties are exactly ``A (x) I`` and ``I (x) B``.

    The test is equality of every entry with the Kronecker product rebuilt from
    the extracted margin, so it decides structure, not a tolerance.  ``swapped``
    says the first penalty is the ``I (x) B`` one.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    width = first.shape[0]
    if first.shape != (width, width) or second.shape != (width, width):
        return None
    for p1 in range(2, width // 2 + 1):
        if width % p1:
            continue
        p2 = width // p1
        for swapped, (outer, inner) in enumerate(((first, second), (second, first))):
            left = outer.reshape(p1, p2, p1, p2)[:, 0, :, 0]
            right = inner.reshape(p1, p2, p1, p2)[0, :, 0, :]
            if np.array_equal(np.kron(left, np.eye(p2)), outer) and np.array_equal(
                np.kron(np.eye(p1), right), inner
            ):
                return p1, p2, left, right, bool(swapped)
    return None
