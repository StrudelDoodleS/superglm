"""The border factorization of a nested chain (one-engine design, section 3.6).

``NestedSchurFactor`` eliminates the tree and the intercept (the super-root)
and hands this module the intercept-profiled border Schur complement
``Q = Q_d + S`` in the rest coordinates: ``Q_d`` its data part, ``S`` the
border penalty, and ``U`` the running bound of every term that formed ``Q_d``,
so that ``|dQ_ij| <= sqrt(U_i U_j)`` componentwise.  Five steps, one path:

1. **Structural deflation, by term type.**  ``BorderGenerators`` holds the
   exact null basis ``N`` of the data part that the specification
   determines: a complete one-hot block sums to the intercept, which the
   super-root has already eliminated, and a nested pair of border blocks has
   a parent indicator equal to the sum of its children's.  ``N`` is reduced
   so that ``N[references] = I``, which makes ``B = [N, E_M]`` unimodular
   (``E_M`` the identity columns off the references).  In that basis the data
   rows and columns of ``N`` are set to exact zero, the penalty parts come
   from ``S`` directly (``a_NN = N'SN``, ``a_MN = E_M'SN``), and ``N`` is
   eliminated first: ``Q''' = Q_d[M, M] + S[M, M] - a_MN a_NN^-1 a_NM``.
2. **Pivoted Cholesky** (LAPACK ``dpstrf``, complete pivoting) of the
   Jacobi-scaled ``Q_s = D_s Q''' D_s`` with the stopping tolerance
   ``tau = c_R(n) + u_s``.  ``u_s = sum_j U_j / Q_jj`` bounds ``||dQ_s||_2``
   (Perron-Frobenius on the rank-one majorant ``sqrt(u_i u_j)``) and
   ``c_R(n)`` is Rump's constant (Rump 2006, BIT 46, Theorem 2.3 and bound
   I of section 3): ``gamma_{n+1} (1 - gamma_{n+1})^-1 tr(A) + n M eta`` with
   ``gamma_k = k u / (1 - k u)``, ``u = 2^-53``, ``M = 3 (2 n + max a_ii)``
   and ``eta = 2^-1074``.  A column whose pivot is within its own ``U_j`` is
   an exact null before the factorization, as before.
3. **Verification** of the retained block (Rump 2006, Corollaries 2.4 and
   2.7): the floating-point Cholesky of ``Q_s[P_r, P_r] - (c_R(r) + tau) I``,
   its diagonal rounded downwards by Rump's Lemma 2.5, must run to
   completion.  That proves every eigenvalue of the retained principal block
   exceeds ``tau``, so every matrix in the ``u_s`` ball is positive definite
   there and, by Cauchy interlacing, ``Q_s`` has ``r`` eigenvalues above
   ``tau``.  If it fails, ``r <- r - 1`` and the test repeats.  The test
   runs first on the whole live block: when it certifies ``lambda_min(Q_s) >
   tau`` there is nothing to truncate (every pivot of step 2 is at least
   ``lambda_min``), so the unpivoted Cholesky factors ``Q_s`` and step 2 is
   skipped; ``dpstrf`` runs only when some direction is not certified.
4. **The truncated subspace.**  ``Z`` from the pivoted factor,
   ``P [-L11^-T L21'; I]`` orthonormalized, gives the Ritz values ``Theta =
   eig(Z'Q_sZ)`` and the residual ``R = Q_sZ - Z(Z'Q_sZ)``.  With the
   certified gap ``delta = tau - max Theta > ||R||_2``, the Davis-Kahan sin
   theorem and the quadratic residual bound of Mathias (1998, SIMAX 19,
   541-550; as Zhu and Knyazev 2016, arXiv 1601.06146, Corollary 5.4, state
   it) bound every eigenvalue of ``Q_s`` in the trailing cluster by ``max
   Theta + ||R||^2 / (delta cos theta_max) + u_s`` with ``sin theta_max <=
   ||R|| / delta``; the true matrix adds ``u_s`` (Weyl).  The certificate
   records it; ``inf`` when the gap is not certified.
5. **Disclosure.**  Every truncated direction is reported with the rest
   columns it touches and its exact penalty curvature ``z'Sz`` (in the
   deflated coordinates).  With Fisher rows the data part is positive
   semidefinite, so a curvature above its own rounding certifies that the
   direction is not an exact null of ``H``: it is reported as weakly
   identified rather than aliased.

The generalized inverse is the Moore-Penrose inverse, in the scaled metric,
of ``Q_s`` compressed onto the retained subspace: ``A = P Q_s P`` with ``P =
I - ZZ'`` and ``Z`` the orthonormal scaled null basis (the Rayleigh-Ritz
compression onto ``span(Z)^perp``; Parlett, *The Symmetric Eigenvalue
Problem*, 1998, chapter 11).  A truncated direction is not an exact null of
``Q_s``: it keeps curvature up to its Ritz values ``Theta`` and residual
``R`` of step 4, and ``A = Q_s - (Z Theta Z' + Z R' + R Z')`` drops exactly
those, which the certificate bounds.  Because ``A Z = 0``, ``A^+ = P (A +
ZZ')^-1 P`` whenever ``A`` is positive definite on ``span(Z)^perp`` (the
null-space identity for a symmetric matrix and an orthonormal basis of its
null space; the Cholesky of ``A + ZZ'`` checks it and refuses otherwise, as
before).  The solve, the explicit inverse and the pseudo-determinant ``log
det(A + ZZ') = log pdet(A)`` all describe ``A``, so the inverse is positive
semidefinite whatever the truncated curvature; without the compression,
``(Q_s + ZZ')^-1 - ZZ'`` inverts nothing when ``Q_s Z != 0`` and can be
indefinite.  Every result is projected by ``P``, which leaves its truncated
component at the rounding of the projection rather than at ``eps kappa(A +
ZZ')``.  The cost is ``O(width^2 t)`` for ``t`` truncated directions, once
per factor, and ``O(width t)`` per solved column.  The inverse is mapped
back through ``D_s``, the elimination of ``N`` and ``B``.  The
pseudo-determinant is the unscaled Moore-Penrose one, ``log det a_NN + log pdet(Q''')``; with every deflated
block penalized a null of ``Q`` vanishes on the block, so neither ``B`` nor
the elimination moves it.

Every rank decision, bound and ``log pdet`` is formed at construction, from
the factorization alone.  The explicit inverses (``BorderFactor.inverse``,
``inverse_data``, the retained ``inverse_kept`` and the first-order
``logdet_bound``) are formed on first read, and a data-side solve
(``apply_data``) goes through the Cholesky factor itself, a backward-stable
solve (Higham 2002, chapter 10) instead of a product with the explicit
inverse: most factors a fit builds are only ever solved with (measured on
pg17: 43 of 62 builds).  Cache owner: the factor; lifetime: the factor;
invalidation: none, since every input is fixed at construction.  No rank
decision or bound reads a lazily formed quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._blas_threads import narrow_kernel_blas_threads

# LAPACK's double-precision Cholesky, pivoted Cholesky and Cholesky inverse
dpotrf, dpstrf, dpotri = scipy.linalg.get_lapack_funcs(
    ("potrf", "pstrf", "potri"), dtype=np.float64
)

# Rump (2006) works in the unit roundoff u = 2^-53 and the underflow unit
# eta = 2^-1074 of IEEE binary64 (section 1).
_UNIT_ROUNDOFF = 2.0**-53
_UNDERFLOW_UNIT = 2.0**-1074
_EPS = float(np.finfo(np.float64).eps)
# Lemma 2.5 of Rump (2006): fl(d - phi |d|) <= a - b for d = fl(a - b).
_PHI = _UNIT_ROUNDOFF * (1.0 + 2.0 * _UNIT_ROUNDOFF)


def _gamma(count: float) -> float:
    """Higham's ``gamma_n = n u / (1 - n u)`` (2002, Lemma 3.1)."""
    return count * _UNIT_ROUNDOFF / (1.0 - count * _UNIT_ROUNDOFF)


@dataclass(frozen=True, eq=False)
class BorderGenerators:
    """Exact null generators of the border's data part (design section 3.6, step 1).

    ``matrix`` ``(q, g)`` holds small integers: every column is an integer
    combination of the specification's data-null vectors (the block sums of
    complete one-hot border blocks and the parent-minus-children vectors of
    nested border blocks), reduced so that ``matrix[references] = I``.
    ``references`` ``(g,)`` holds each generator's reference column, the
    most prior-exposed column available when it was chosen.  ``labels``
    names each generator for disclosure.  Built once per layout and prior
    weights by ``nested.NestedStructuredLayout.prior_statistics``.
    """

    matrix: NDArray
    references: NDArray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        references = np.array(self.references, dtype=np.intp, copy=True)
        if matrix.ndim != 2 or references.shape != (matrix.shape[1],):
            raise ValueError("Border generators need a (q, g) matrix and g references.")
        if matrix.shape[1] and not np.array_equal(matrix[references], np.eye(matrix.shape[1])):
            raise ValueError("Border generators must be reduced to the identity on references.")
        matrix.setflags(write=False)
        references.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def count(self) -> int:
        return int(self.matrix.shape[1])

    def shifted(self, offset: int) -> BorderGenerators:
        """The same generators with ``offset`` leading zero rows (for example an intercept column)."""
        matrix = np.vstack((np.zeros((offset, self.count)), self.matrix))
        return BorderGenerators(matrix, self.references + offset, self.labels)

    def dropped(self, leading: int) -> BorderGenerators:
        """The same generators without ``leading`` rows, which must be zero."""
        if np.any(self.matrix[:leading]) or np.any(self.references < leading):
            raise ValueError("Only exactly zero leading generator rows can be dropped.")
        return BorderGenerators(self.matrix[leading:], self.references - leading, self.labels)


def reduce_generators(
    candidates: NDArray, exposure: NDArray, labels: tuple[str, ...]
) -> BorderGenerators:
    """Select independent generators from integer candidates and reduce them to ``N[refs] = I``.

    Gauss-Jordan elimination with unit pivots in a fixed order: each
    candidate, in column order, is reduced against the generators already
    chosen; a candidate that reduces to zero depends on them and is dropped;
    otherwise its reference is the column of largest ``exposure`` among its
    entries of magnitude one (lowest column on ties), and every earlier
    generator is cleared on it.  All arithmetic is on small integers, exact
    in float64.  Deterministic: a function of the candidates and exposures.
    """
    columns: list[NDArray] = []
    references: list[int] = []
    kept: list[str] = []
    for index in range(candidates.shape[1]):
        vector = np.array(candidates[:, index], dtype=np.float64)
        for column, reference in zip(columns, references, strict=True):
            if vector[reference] != 0.0:
                vector -= vector[reference] * column
        units = np.flatnonzero(np.abs(vector) == 1.0)
        if not units.size:
            if np.any(vector):
                raise ValueError("A border generator reduced to a non-unit integer vector.")
            continue
        reference = int(units[np.argmax(exposure[units])])
        vector = vector / vector[reference]
        for position, column in enumerate(columns):
            if column[reference] != 0.0:
                columns[position] = column - column[reference] * vector
        columns.append(vector)
        references.append(reference)
        kept.append(labels[index] if index < len(labels) else f"generator {index}")
    width = candidates.shape[0]
    matrix = np.column_stack(columns) if columns else np.zeros((width, 0))
    return BorderGenerators(matrix, np.array(references, dtype=np.intp), tuple(kept))


def rump_constant(size: int, trace: float, largest_diagonal: float) -> float:
    """Rump (2006) bound I: ``||Delta(A)||_2`` for an ``size``-square symmetric ``A``.

    ``gamma_{n+1} (1 - gamma_{n+1})^-1 tr(A) + n M eta`` with ``M = 3 (2 n +
    max a_ii)``, rounded upwards by a relative ``4 u`` so that the
    floating-point evaluation of this expression cannot undercut it.
    """
    if size <= 0:
        return 0.0
    gamma = (size + 1) * _UNIT_ROUNDOFF / (1.0 - (size + 1) * _UNIT_ROUNDOFF)
    if not gamma < 1.0:
        raise ValueError("Rump's constant needs gamma_{n+1} < 1.")
    bound = (
        gamma / (1.0 - gamma) * float(trace)
        + size * 3.0 * (2.0 * size + float(largest_diagonal)) * _UNDERFLOW_UNIT
    )
    return bound * (1.0 + 4.0 * _UNIT_ROUNDOFF)


def _shifted_down(block: NDArray, shift: float) -> NDArray:
    """``block`` with every diagonal entry replaced by a float at most ``a_ii - shift``.

    Rump (2006), Lemma 2.5 and (2.11): ``fl(d - phi |d|)`` with ``d =
    fl(a_ii - shift)`` and ``phi = u (1 + 2 u)``.
    """
    shifted = np.array(block, dtype=np.float64, copy=True)
    diagonal = np.diag(shifted) - shift
    np.fill_diagonal(shifted, diagonal - _PHI * np.abs(diagonal))
    return shifted


def verified_positive_definite(block: NDArray, shift: float) -> bool:
    """Whether the floating-point Cholesky of ``block - (c_R + shift) I`` runs to completion.

    By Rump (2006), Corollary 2.4 with Theorem 2.3's constant ``c_R``, success
    proves ``lambda_min(block) > shift``.
    """
    size = block.shape[0]
    if size == 0:
        return True
    diagonal = np.diag(block)
    if np.any(diagonal <= 0.0):
        return False
    constant = rump_constant(size, float(np.sum(diagonal)), float(np.max(diagonal)))
    _, info = dpotrf(_shifted_down(block, constant + shift), lower=1, clean=0)
    return info == 0


@dataclass(frozen=True, eq=False)
class BorderCertificate:
    """What the border factorization certified (design section 3.7), for disclosure.

    ``width`` columns entered the scaled matrix after deflation (``deflated``
    generators removed, ``exact_null`` columns zero by their own bound);
    ``tau = c_R(width) + u_s`` stopped ``dpstrf``; the verified retained rank
    is ``rank`` after ``decrements`` failed verifications.
    ``trailing_bound`` bounds every scaled eigenvalue of the truncated cluster
    (``nan`` when nothing was truncated by ``dpstrf``, ``inf`` when the gap
    was not certified).  ``logdet_bound`` is the first-order bound ``tau
    tr(Q_s,ret^-1)`` on the border part of ``log|H|``.  ``directions`` holds,
    per truncated direction, the rest columns it touches;
    ``penalty_curvature`` its ``z'Sz`` and ``weak`` whether that exceeds its
    rounding, which certifies the direction is not an exact null of ``H``.
    """

    width: int
    deflated: int
    exact_null: int
    tau: float
    u_s: float
    rank: int
    decrements: int
    trailing_bound: float
    _logdet_bound_source: _BorderInverse | float  # the bound, or the inverse that forms it
    directions: tuple[NDArray, ...]
    penalty_curvature: NDArray
    weak: NDArray

    @cached_property
    def logdet_bound(self) -> float:
        """``tau tr(Q_s,ret^-1)``, the first-order bound on the border part of ``log|H|``."""
        source = self._logdet_bound_source
        if isinstance(source, _BorderInverse):
            return source.logdet_bound()
        return float(source)

    def curvature_floor(self, norm: float) -> float:
        """``tau + 2 width eps norm``: a computed scaled eigenvalue below minus it is negative curvature.

        ``tau = c_R + u_s`` is what the certificate cannot tell apart from zero
        in the scaled matrix itself (the data's running bound ``u_s`` and the
        Cholesky constant ``c_R``); a backward-stable symmetric eigensolver
        moves each eigenvalue by at most ``p(n) eps ||Q_s||_2`` (LAPACK Users'
        Guide, 3rd ed., section 4.7), taken here as ``2 n eps`` times
        ``norm``, the computed ``||Q_s||_2``.  Step 4's refusal of a Ritz value
        is the same two terms with ``||Q_s||_2 <= n`` for the unit-diagonal
        live block, so an eigenvalue below minus this floor is one the factor's
        own certificate calls material, and one above it is within what the
        factor truncates as a null.
        """
        return float(self.tau + 2.0 * self.width * _EPS * float(norm))


def _projected_off(matrix: NDArray, basis: NDArray | None) -> NDArray:
    """``P matrix P`` with ``P = I - B B'`` for a symmetric ``matrix`` and orthonormal ``B``.

    The congruence keeps a positive semidefinite ``matrix`` semidefinite to
    the rounding of the products, and leaves ``B'`` of the result at the
    rounding of one projection whatever the conditioning of ``matrix``.
    ``O(width^2 t)`` for ``t`` columns; ``matrix`` itself when ``B`` is empty.
    """
    if basis is None or not basis.shape[1]:
        return matrix
    right = matrix @ basis
    projected = matrix - basis @ right.T - right @ basis.T + basis @ (basis.T @ right) @ basis.T
    return 0.5 * (projected + projected.T)


class _BorderInverse:
    """The inverses of one factored border, each formed on first read (module docstring).

    ``kind`` is ``"full"`` (``factor`` the Cholesky factor of the scaled
    ``Q_s[P, P]``, ``pivot`` ``P``) or ``"deflated"`` (``factor`` that of
    ``A + ZZ'``, ``A = P Q_s P`` the compression of the module docstring,
    ``null_scaled`` ``Z``, ``truncated`` its columns that are not exact
    nulls); ``scale`` is ``D_s``; ``deflation``
    ``(N, references)`` with ``M``, ``G`` and ``a_lower`` the structural
    elimination of step 1, or ``None``.  Every lazy body runs under the same
    one-thread BLAS cap as the construction (``narrow_kernel_blas_threads``):
    a lazily formed ``dpotri`` would otherwise run at the pool width a wide
    fit released, where two OpenBLAS pools contend (measured 3.8 s against
    0.5 s per build).
    """

    def __init__(
        self,
        *,
        kind,
        factor,
        pivot,
        null_scaled,
        scale,
        deflation,
        M,
        G,
        a_lower,
        m,
        tau,
        live,
        truncated=None,
    ):
        self.kind, self.factor, self.pivot, self.null_scaled = kind, factor, pivot, null_scaled
        self.scale, self.deflation, self.M, self.G = scale, deflation, M, G
        self.a_lower, self.m, self.tau, self.live = a_lower, m, tau, live
        # the truncated (not exact-null) columns of ``null_scaled``: every
        # result is projected off them (``_projected_off``)
        self.truncated = truncated

    @cached_property
    def inverse_scaled(self) -> NDArray:
        """``Q_s^+`` on the retained subspace (``P (A + ZZ')^-1 P`` when truncated)."""
        width = len(self.scale)
        with narrow_kernel_blas_threads(self.m):
            if self.kind == "full":
                # the (pivoted) factor itself inverts Q_s[P, P] (LAPACK dpotri,
                # which forms the lower triangle of L^-T L^-1)
                lower, info = dpotri(self.factor, lower=1)
                if info != 0:  # pragma: no cover - a completed Cholesky is nonsingular
                    raise np.linalg.LinAlgError("The border inverse failed.")
                permuted = np.tril(lower)
                permuted += np.tril(lower, -1).T
                if np.array_equal(self.pivot, np.arange(width)):
                    return permuted
                inverse_scaled = np.empty((width, width))
                inverse_scaled[np.ix_(self.pivot, self.pivot)] = permuted
                return inverse_scaled
            inverse_scaled = scipy.linalg.cho_solve(
                (self.factor, True), np.eye(width), check_finite=False
            )
            inverse_scaled = (
                0.5 * (inverse_scaled + inverse_scaled.T) - self.null_scaled @ self.null_scaled.T
            )
            return _projected_off(inverse_scaled, getattr(self, "truncated", None))

    @cached_property
    def inverse_kept(self) -> NDArray:
        """``D_s Q_s^+ D_s``, the retained inverse in the deflated coordinates."""
        return self.scale[:, None] * self.inverse_scaled * self.scale[None, :]

    def apply_kept(self, values: NDArray) -> NDArray:
        """``inverse_kept @ values`` through the factor, never forming the inverse."""
        scale = self.scale
        z = scale[:, None] * values if values.ndim == 2 else scale * values
        with narrow_kernel_blas_threads(self.m):
            if self.kind == "full":
                x = np.empty_like(z)
                x[self.pivot] = scipy.linalg.cho_solve(
                    (self.factor, True), z[self.pivot], check_finite=False
                )
            else:
                Z = self.null_scaled
                x = scipy.linalg.cho_solve((self.factor, True), z, check_finite=False)
                x = x - Z @ (Z.T @ z)
                truncated = getattr(self, "truncated", None)
                if truncated is not None and truncated.shape[1]:
                    x = x - truncated @ (truncated.T @ x)
        return scale[:, None] * x if x.ndim == 2 else scale * x

    def logdet_bound(self) -> float:
        return float(self.tau * np.sum(np.diag(self.inverse_scaled)[self.live]))

    @cached_property
    def inverse_data(self) -> NDArray:
        """The data-side inverse, ``inverse`` less ``N a_NN^-1 N'`` (``BorderFactor``)."""
        if self.deflation is None:
            return self.inverse_kept
        N, _ = self.deflation
        count, width, M = N.shape[1], len(self.M), self.M
        with narrow_kernel_blas_threads(self.m):
            inverse_kept, G = self.inverse_kept, self.G
            GQ = G.T @ inverse_kept  # (count, width)
            # K^g = L1^-T diag(a_NN^-1, Q'''^g) L1^-1 = a_NN^-1 on the N block
            # plus the rest, then Q^g = B K^g B' with B = [N, E_M], the N part
            # through small products; the data side takes K_NN = GQ G.
            right = np.zeros((count + width, self.m))  # K^g B'
            right[:count] = (GQ @ G) @ N.T
            right[:count][:, M] -= GQ
            right[count:] = -GQ.T @ N.T
            right[count:][:, M] += inverse_kept
            product = N @ right[:count]
            product[M] += right[count:]
            return 0.5 * (product + product.T)

    @cached_property
    def inverse(self) -> NDArray:
        """The generalized inverse: the data side plus the structural part ``N a_NN^-1 N'``."""
        if self.deflation is None:
            return self.inverse_data
        N, _ = self.deflation
        count = N.shape[1]
        with narrow_kernel_blas_threads(self.m):
            # the assembly is affine in K_NN through N K_NN N' alone: the full
            # inverse adds the structural part N a_NN^-1 N' to the data side
            a_inverse = scipy.linalg.cho_solve(
                (self.a_lower, True), np.eye(count), check_finite=False
            )
            structural = N @ (0.5 * (a_inverse + a_inverse.T)) @ N.T
            return self.inverse_data + 0.5 * (structural + structural.T)


@dataclass(frozen=True, eq=False)
class _DataSide:
    """The factored data-side inverse: ``x_M = inverse_kept y_M``, ``x_N = -G' x_M``, ``x = N x_N + E_M x_M``."""

    M: NDArray
    N: NDArray | None
    G: NDArray | None
    source: _BorderInverse | None

    @property
    def inverse_kept(self) -> NDArray:
        return np.zeros((0, 0)) if self.source is None else self.source.inverse_kept


@dataclass(frozen=True, eq=False)
class BorderFactor:
    """The factored intercept-profiled border Schur complement ``Q`` (rest coordinates).

    ``inverse`` is the generalized inverse described in the module docstring
    and ``logdet`` the unscaled Moore-Penrose ``log pdet(Q)``.
    ``inverse_data`` is ``inverse`` less ``N a_NN^-1 N'``, its part along the
    deflated structural nulls: equal to it on every vector ``y`` with ``N'y =
    0``, which every data-derived vector satisfies exactly (the data part of
    ``Q`` vanishes along ``N``), and free of the ``1 / a_NN`` entries whose
    products with such a vector cancel.  ``a_NN`` is the penalty curvature
    alone, so with a tiny penalty those entries dwarf the result: solves of a
    normal-equations right-hand side and every product with a data operator
    read ``inverse_data``; penalty traces and variances read ``inverse``.  ``null``
    ``(m, k)`` holds unscaled null vectors ``V`` and ``null_left`` the matching
    ``W`` with ``Q^+ Q = I - V W'``.  ``scaled_matrix`` is the Jacobi-scaled
    deflated matrix (exact-null rows and columns zero) whose eigenvalues the
    observed-geometry build checks; ``condition`` is the retained squared
    pivot ratio of ``dpstrf`` (``inf`` when anything was truncated).
    ``inverse`` and ``inverse_data`` are formed on first read, and
    ``apply_data`` solves through the factor (module docstring).
    """

    source: _BorderInverse | None
    logdet: float
    null: NDArray
    null_left: NDArray
    scaled_matrix: NDArray
    condition: float
    certificate: BorderCertificate
    data_side: _DataSide | None = None
    # rest columns null by their own bound (the diagonal test), scale 1 in
    # every scaled null basis, as in the rank decision
    exact_null_columns: NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))

    @cached_property
    def inverse(self) -> NDArray:
        return np.zeros((0, 0)) if self.source is None else self.source.inverse

    @cached_property
    def inverse_data(self) -> NDArray:
        return np.zeros((0, 0)) if self.source is None else self.source.inverse_data

    @property
    def deflated(self) -> bool:
        """Whether step 1 deflated a structural generator (``inverse_data`` differs from ``inverse``)."""
        return self.source is not None and self.source.deflation is not None

    def apply_data(self, values: NDArray) -> NDArray:
        """``inverse_data @ values`` for data-derived ``values`` ``(m,)`` or ``(m, r)``, factored.

        ``B' y = [N'y; y_M]`` with ``N'y = 0`` exactly for data: ``x_M =
        inverse_kept y_M``, ``x_N = -G' x_M``, with ``inverse_kept y_M`` a
        solve through the Cholesky factor (module docstring).  The explicit
        ``inverse_data`` would instead cancel ``N'y`` inside the product
        through entries as large as the weakest retained curvature allows.
        """
        side = self.data_side
        if side is None or side.source is None:
            return np.zeros_like(values)
        x_M = side.source.apply_kept(values[side.M])
        if side.N is None or side.G is None:
            return x_M
        result = side.N @ (-(side.G.T @ x_M))
        result[side.M] += x_M
        return result

    def quadratic_data(self, rows: NDArray) -> NDArray:
        """``y' inverse_data y`` per row of data-derived ``rows`` ``(r, m)``: ``y_M' inverse_kept y_M``."""
        side = self.data_side
        if side is None or side.N is None:
            return np.sum((rows @ self.inverse_data) * rows, axis=1)
        block = rows[:, side.M]
        return np.sum((block @ side.inverse_kept) * block, axis=1)


def _row_norms(matrix: NDArray) -> NDArray:
    """Each row's 2-norm, scaled by its largest entry so it neither overflows nor underflows."""
    if matrix.size == 0:
        return np.zeros(matrix.shape[0])
    largest = np.max(np.abs(matrix), axis=1)
    safe = np.where(largest > 0.0, largest, 1.0)
    return largest * np.sqrt(np.sum((matrix / safe[:, None]) ** 2, axis=1))


def _paired_majorant(e: NDArray, g: NDArray, diagonal: NDArray) -> NDArray:
    """``t e_i^2 + g_i^2 / t`` with ``t`` minimising its Jacobi-scaled sum, range-safe.

    Majorises ``|e_i g_j| + |g_i e_j| <= sqrt(b_i b_j)`` for every ``t > 0``
    (Cauchy-Schwarz).  With ``e = e_max e^`` and ``g = g_max g^`` the minimiser
    is ``t = (g_max / e_max) rho``, ``rho = sqrt(sum g^^2 / d / sum e^^2 / d)``,
    so ``b_i = e_max g_max (rho e^_i^2 + g^_i^2 / rho)``: no square of ``e``
    or ``g`` and no quotient of their sums is formed.  Zero where every
    product ``e_i g_j`` is.
    """
    e_max = float(np.max(e, initial=0.0))
    g_max = float(np.max(g, initial=0.0))
    if not (e_max > 0.0 and g_max > 0.0):
        return np.zeros_like(e)
    e_hat, g_hat = e / e_max, g / g_max
    resolved = diagonal > 0.0
    rho = 1.0
    if np.any(resolved):
        scale = diagonal[resolved] / float(np.max(diagonal[resolved]))
        spread = float(np.sum(e_hat[resolved] ** 2 / scale))
        reach = float(np.sum(g_hat[resolved] ** 2 / scale))
        if spread > 0.0 and reach > 0.0 and np.isfinite(spread) and np.isfinite(reach):
            rho = math.sqrt(reach / spread)
    return e_max * g_max * (rho * e_hat**2 + g_hat**2 / rho)


def _deflate(Q_d, S, U, generators):
    """Step 1: the kept generators, the deflated matrix ``Q'''``, its bound and the elimination."""
    m = Q_d.shape[0]
    if generators is None or not generators.count:
        return None, np.arange(m), Q_d + S, np.array(U, dtype=np.float64), S, 0.0, None, None
    N, references = generators.matrix, generators.references
    SN = S @ N
    a_NN = N.T @ SN
    # A generator with no penalty curvature (a user-fixed zero penalty on its
    # block) is itself an exact null; it is not deflated, and the pivoted
    # factorization below truncates it with the other aliases.
    penalized = np.diag(a_NN) > 0.0
    if not np.all(penalized):
        N, references, SN = N[:, penalized], references[penalized], SN[:, penalized]
        a_NN = a_NN[np.ix_(penalized, penalized)]
    if not N.shape[1]:
        return None, np.arange(m), Q_d + S, np.array(U, dtype=np.float64), S, 0.0, None, None
    a_NN = 0.5 * (a_NN + a_NN.T)
    # The generators' penalty products themselves round: |d a_MN| <= gamma_m
    # |S||N| and |d a_NN| <= gamma_(2m) |N|'|S||N| (Higham 2002, section 3.5).
    # A thin level's penalized alias lies almost in S's null space, so a_NN is
    # those products' cancellation (2e-10 against |N|'|S||N| near 1e-3 on an
    # sz term whose every level is thin), and the elimination multiplies their
    # rounding by its multiplier G = a_MN a_NN^-1 (below).  Unresolved when
    # ||a_NN^-1|| ||E_NN|| reaches 1/2: those generators then stay with the
    # pivoted factorization, as an exact null does.
    absolute_N = np.abs(N)
    S_N = np.abs(S) @ absolute_N
    E_MN = _gamma(m) * S_N
    E_NN = _gamma(2 * m) * (absolute_N.T @ S_N)
    a_values = np.linalg.eigvalsh(a_NN)
    E_norm = float(np.linalg.norm(E_NN, 2))
    ratio = E_norm / float(a_values[0]) if a_values[0] > 0.0 else float("inf")
    if not ratio < 0.5:
        return None, np.arange(m), Q_d + S, np.array(U, dtype=np.float64), S, 0.0, None, None
    keep = np.ones(m, dtype=bool)
    keep[references] = False
    M = np.flatnonzero(keep)
    lower = scipy.linalg.cholesky(a_NN, lower=True, check_finite=False)
    a_MN = SN[M]
    Y = scipy.linalg.solve_triangular(lower, a_MN.T, lower=True, check_finite=False)
    correction = Y.T @ Y
    S_M = S[np.ix_(M, M)] - correction
    S_M = 0.5 * (S_M + S_M.T)
    Q = Q_d[np.ix_(M, M)] + S_M
    count = N.shape[1]
    # the elimination's own rounding, componentwise on the diagonal
    bound = U[M] + (count + 3) * _EPS * (np.abs(np.diag(S)[M]) + np.diag(correction))
    logdet_N = float(2.0 * np.sum(np.log(np.diag(lower))))
    # G = a_MN a_NN^-1, the elimination multiplier of L1 = [[I, 0], [G, I]]
    G = scipy.linalg.solve_triangular(lower, Y, lower=True, trans="T", check_finite=False).T
    # The products' rounding through the elimination: at first order S_M moves
    # by d a_MN G' + G d a_NM - G d a_NN G', and the inverse's remainder is a
    # factor 1 / (1 - ratio).  With e_i = ||E_MN[i]|| and g_i = ||G[i]||,
    # |e_i g_j| + |g_i e_j| <= sqrt((t e_i^2 + g_i^2 / t)(t e_j^2 + g_j^2 / t))
    # for every t > 0 (Cauchy-Schwarz) and |g_i' d a_NN g_j| <= ||E_NN|| g_i g_j,
    # so the column bound below keeps factor_border's |dQ_ij| <= sqrt(b_i b_j);
    # t minimises the Jacobi-scaled sum u_s, as the border majorant's own does.
    # Range-safe (Sol review of 0ccab297): norms by max-abs scaling and t through
    # ratios of scaled sums, so a penalty near 1e-140 neither overflows t nor
    # underflows e_i^2 into a spurious t = 1.
    e = _row_norms(E_MN[M])
    g = _row_norms(G)
    bound = bound + _paired_majorant(e, g, np.diag(Q)) + (E_norm / (1.0 - ratio)) * g * g
    return (N, references), M, Q, bound, S_M, logdet_N, G, lower


def factor_border(
    Q_d: NDArray,
    S: NDArray,
    U: NDArray,
    generators: BorderGenerators | None,
    *,
    term_name: str,
    term_kind: str = "Nested chain",
) -> BorderFactor:
    """Steps 1-5 of the module docstring on the rest-coordinate ``Q = Q_d + S``.

    ``term_kind`` and ``term_name`` name the term in refusal messages
    (``"FactorSmooth term"`` for the fs and sz factors).
    Refusals (``np.linalg.LinAlgError``): a pivot below minus its own bound
    (material negative curvature), a truncated cluster with a Ritz value below
    minus the uncertainty, or a retained block that is not positive definite
    after deflating the truncated directions, which the verification makes
    unreachable except by a bug.
    """
    m = Q_d.shape[0]
    if m == 0:
        empty = np.zeros((0, 0))
        certificate = BorderCertificate(
            0, 0, 0, 0.0, 0.0, 0, 0, float("nan"), 0.0, (), np.zeros(0), np.zeros(0, dtype=bool)
        )
        return BorderFactor(None, 0.0, empty, empty, empty, 1.0, certificate, None)
    deflation, M, Q, bound, S_M, logdet_N, G, a_lower = _deflate(Q_d, S, U, generators)
    width = len(M)
    diagonal = np.diag(Q).copy()
    if np.any(diagonal < -bound):
        worst = int(np.argmin(diagonal + bound))
        raise np.linalg.LinAlgError(
            f"{term_kind} {term_name!r} has materially negative Schur curvature "
            f"{diagonal[worst]:.6g} on its border (the intercept and the other terms' "
            f"columns), below minus its bound {bound[worst]:.3g}, so the Hessian is "
            "indefinite there."
        )
    exact_null = diagonal <= bound
    live = np.flatnonzero(~exact_null)
    scale = np.ones(width)
    scale[live] = 1.0 / np.sqrt(diagonal[live])
    Q_scaled = scale[:, None] * Q * scale[None, :]
    Q_scaled[exact_null, :] = 0.0
    Q_scaled[:, exact_null] = 0.0
    Q_scaled = 0.5 * (Q_scaled + Q_scaled.T)
    live_matrix = Q_scaled if len(live) == width else Q_scaled[np.ix_(live, live)]
    n_live = len(live)
    u_s = float(np.sum(bound[live] / diagonal[live]))
    live_diagonal = np.diag(live_matrix)
    tau = (
        rump_constant(n_live, float(np.sum(live_diagonal)), float(np.max(live_diagonal))) + u_s
        if n_live
        else u_s
    )

    # Step 2: complete pivoting with the stopping tolerance tau.
    rank, decrements, trailing_bound = 0, 0, float("nan")
    factor = np.zeros((0, 0))
    pivot = np.zeros(0, dtype=np.intp)
    if n_live and verified_positive_definite(live_matrix, tau):
        # Step 3 on the whole live block certifies lambda_min(Q_s) > tau:
        # nothing is truncated, and the unpivoted Cholesky factors Q_s.
        factor, info = dpotrf(live_matrix, lower=1, clean=1)
        if info != 0:  # pragma: no cover - the shifted factorization completed
            raise np.linalg.LinAlgError(
                f"{term_kind} {term_name!r} verified border block failed its Cholesky."
            )
        rank, pivot = n_live, np.arange(n_live)
    elif n_live:
        packed, piv, rank, info = dpstrf(live_matrix, tol=tau, lower=1)
        if info < 0:  # pragma: no cover - argument error
            raise ValueError(f"dpstrf rejected its argument {-info}.")
        factor = np.tril(packed)
        pivot = np.asarray(piv, dtype=np.intp) - 1
        # Step 3: Rump verification of the retained principal block.
        while rank > 0 and not verified_positive_definite(
            live_matrix[np.ix_(pivot[:rank], pivot[:rank])], tau
        ):
            rank -= 1
            decrements += 1

    # Step 4: the truncated subspace in the live scaled coordinates.
    trailing = n_live - rank
    null_live = np.zeros((n_live, 0))
    if trailing:
        L11 = factor[:rank, :rank]
        L21 = factor[rank:, :rank]
        basis = np.zeros((n_live, trailing))
        if rank:
            basis[pivot[:rank]] = -scipy.linalg.solve_triangular(
                L11, L21.T, lower=True, trans="T", check_finite=False
            )
        basis[pivot[rank:], np.arange(trailing)] = 1.0
        null_live = np.linalg.qr(basis)[0]
        product = live_matrix @ null_live
        rayleigh = null_live.T @ product
        rayleigh = 0.5 * (rayleigh + rayleigh.T)
        ritz = np.linalg.eigvalsh(rayleigh)
        residual = float(np.linalg.norm(product - null_live @ rayleigh, 2))
        # a Ritz value of a PSD matrix below minus the uncertainty and the
        # evaluation's rounding is material negative curvature
        # (``BorderCertificate.curvature_floor`` with ||Q_s||_2 <= n_live)
        floor = tau + 2.0 * n_live * n_live * _EPS
        if ritz[0] < -floor:
            raise np.linalg.LinAlgError(
                f"{term_kind} {term_name!r} has materially negative Schur curvature "
                f"{ritz[0]:.3g} on its truncated border subspace."
            )
        gap = tau - float(ritz[-1])
        if gap > residual:
            sine = residual / gap
            trailing_bound = (
                float(ritz[-1]) + residual * residual / (gap * np.sqrt(1.0 - sine * sine)) + u_s
            )
        else:
            trailing_bound = float("inf")

    # The orthonormal scaled null basis on the deflated coordinates: exact
    # nulls by their own bound, then the truncated subspace.
    exact_columns = np.flatnonzero(exact_null)
    k = len(exact_columns) + trailing
    null_scaled = np.zeros((width, k))
    null_scaled[exact_columns, np.arange(len(exact_columns))] = 1.0
    null_scaled[np.ix_(live, np.arange(len(exact_columns), k))] = null_live

    if width == 0:
        kind, solver_factor, logdet_scaled, condition = "empty", None, 0.0, 1.0
    elif k == 0:
        # Full rank: the (pivoted) factor inverts Q_s[P, P] (``_BorderInverse``).
        kind, solver_factor = "full", factor
        logdet_scaled = float(2.0 * np.sum(np.log(np.diag(factor))))
        squares = np.diag(factor) ** 2
        condition = float(squares.max() / squares.min())
    else:
        # The truncated operator: Q_s compressed onto the retained subspace,
        # P Q_s P with P = I - ZZ' (Rayleigh-Ritz), whose null space is
        # exactly span(Z).  A truncated direction keeps curvature up to its
        # Ritz values and residual (step 4), so Q_s Z != 0 and (Q_s + ZZ')^-1
        # - ZZ' is no inverse of anything: it can be indefinite.  P Q_s P =
        # Q_s - (Z Theta Z' + Z R' + R Z'), so it differs from Q_s by the
        # Ritz block and the residual the certificate already bounds; exact
        # nulls have zero rows in Q_s and need no projection.
        operator = Q_scaled
        if trailing:
            coupling = np.zeros((width, trailing))
            coupling[live] = product  # Q_s Z_t: rows off ``live`` are zero
            Z_t = null_scaled[:, len(exact_columns) :]
            operator = Q_scaled - Z_t @ coupling.T - coupling @ Z_t.T + Z_t @ rayleigh @ Z_t.T
            operator = 0.5 * (operator + operator.T)
        deflated = operator + null_scaled @ null_scaled.T
        try:
            cholesky = scipy.linalg.cholesky(deflated, lower=True, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f"{term_kind} {term_name!r} retained Schur block is not positive definite "
                f"after deflating its {k} null directions: {error}"
            ) from error
        kind, solver_factor = "deflated", cholesky
        logdet_scaled = float(2.0 * np.sum(np.log(np.diag(cholesky))))
        condition = float("inf")
    logdet_M = logdet_scaled + float(np.sum(np.log(diagonal[live])))
    if k:
        # Jacobi's complementary minor: pdet(Q) = pdet(Q_s) prod Q_jj det(Z' D_s^2 Z).
        logdet_M += float(np.linalg.slogdet(null_scaled.T @ (scale[:, None] ** 2 * null_scaled))[1])
    source = (
        None
        if kind == "empty"
        else _BorderInverse(
            kind=kind,
            factor=solver_factor,
            pivot=pivot,
            null_scaled=null_scaled,
            scale=scale,
            deflation=deflation,
            M=M,
            G=G,
            a_lower=a_lower,
            m=m,
            tau=float(tau),
            live=live,
            truncated=null_scaled[:, len(exact_columns) :],
        )
    )
    logdet_bound: _BorderInverse | float = source if (n_live and source is not None) else 0.0
    unscaled_null = scale[:, None] * null_scaled
    left_null = null_scaled / scale[:, None]

    # Map back through the elimination of N and the unimodular basis B.
    if deflation is None:
        null, null_left = unscaled_null, left_null
        logdet = logdet_M
    else:
        N, references = deflation
        N_M = N[M]
        # V = B L1^-T [0; D Z] and W = B^-T L1 [0; D^-1 Z]
        top = -G.T @ unscaled_null
        null = N @ top
        null[M] += unscaled_null
        null_left = np.zeros((m, k))
        null_left[M] = left_null
        null_left[references] = -N_M.T @ left_null
        logdet = logdet_N + logdet_M
        if k:
            # ``Q = C^-T diag(a_NN, Q''') C^-1`` with ``C = B L1^-T`` unimodular, so
            # the pseudo-determinant of the congruence (design §3.5) is
            # ``pdet(diag(a_NN, Q''')) det(V'V) / det(Z'Z)`` with ``V = C [0; Z]``
            # the mapped nulls and ``Z`` any basis of ``Q'''``'s null space.  One
            # (exactly) when the nulls vanish on the deflated block; not when a
            # deflated direction couples to a null through the penalty (an ``sz``
            # thin level's penalized alias beside its exact ones).
            logdet += float(
                np.linalg.slogdet(null.T @ null)[1]
                - np.linalg.slogdet(unscaled_null.T @ unscaled_null)[1]
            )

    # Step 5: disclosure of every null direction, exact nulls by their own
    # bound first, then the truncated subspace.
    directions: list[NDArray] = []
    curvature = np.zeros(k)
    weak = np.zeros(k, dtype=bool)
    if k:
        # A direction touches the columns where it is above the resolution of
        # its own computation (sqrt(eps) of its largest entry); its penalty
        # curvature on those columns, positive beyond the rounding of the
        # quadratic form, certifies a direction that is not an exact null of H.
        scaled = np.abs(null_scaled)
        touched = scaled > np.sqrt(_EPS) * np.max(scaled, axis=0, initial=0.0)
        vectors = np.where(touched, unscaled_null, 0.0)
        curvature = np.einsum("ij,ij->j", vectors, S_M @ vectors)
        magnitude = np.einsum("ij,ij->j", np.abs(vectors), np.abs(S_M) @ np.abs(vectors))
        weak = curvature > (width + 2) * _EPS * magnitude
        for column in range(k):
            size = np.abs(null[:, column])
            directions.append(np.flatnonzero(size > np.sqrt(_EPS) * float(np.max(size))))
    certificate = BorderCertificate(
        width=width,
        deflated=0 if deflation is None else int(deflation[0].shape[1]),
        exact_null=len(exact_columns),
        tau=float(tau),
        u_s=u_s,
        rank=int(rank),
        decrements=int(decrements),
        trailing_bound=float(trailing_bound),
        _logdet_bound_source=logdet_bound,
        directions=tuple(directions),
        penalty_curvature=curvature,
        weak=weak,
    )
    return BorderFactor(
        exact_null_columns=M[exact_columns],
        data_side=_DataSide(
            M=M,
            N=None if deflation is None else deflation[0],
            G=G,
            source=source,
        ),
        source=source,
        logdet=float(logdet),
        null=null,
        null_left=null_left,
        scaled_matrix=Q_scaled,
        condition=condition,
        certificate=certificate,
    )
