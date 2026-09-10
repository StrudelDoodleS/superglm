"""Rank-aware structured factorization for sum-to-zero factor smooths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    _BlockDiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _multiply_symmetric_bdlr_coalesced,
    _operator_bdlr,
    _trace_general_bdlr_product,
    _trace_symmetric_bdlr,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    _eigensolver_relative_bar,
    _equilibrate_gram,
    _symmetric_part,
    decompose_gram,
)
from superglm.types import PenaltyComponent


class SumToZeroIdentifiabilityError(np.linalg.LinAlgError):
    """Raised when the globally constrained SZ system is not identifiable."""


@dataclass(frozen=True)
class _LocalPSD:
    """Positive-range inverse and null basis for one symmetric local block."""

    pinv: NDArray
    null: NDArray
    positive_eigenvalues: NDArray
    rank: int


def _decompose_local_psd(
    block: NDArray,
    *,
    term_name: str,
    level_label: Any,
) -> _LocalPSD:
    """Decompose a PSD block without requiring every level to be full rank."""
    locals_, _minimum = _decompose_local_psd_batch(
        np.asarray(block, dtype=np.float64)[None, :, :],
        term_name=term_name,
        level_labels=(level_label,),
    )
    return locals_[0]


def _decompose_local_psd_batch(
    blocks: NDArray,
    *,
    term_name: str,
    level_labels: tuple[Any, ...],
) -> tuple[tuple[_LocalPSD, ...], float]:
    """Decompose every local PSD block in one batched LAPACK dispatch."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Local blocks must have shape (K, k, k).")
    if values.shape[0] != len(level_labels):
        raise ValueError("level_labels length must equal the number of local blocks.")
    symmetric = np.stack([_symmetric_part(block) for block in values])
    finite = np.all(np.isfinite(symmetric), axis=(1, 2))
    if not np.all(finite):
        level = int(np.flatnonzero(~finite)[0])
        raise np.linalg.LinAlgError(
            f"Structured term {term_name!r} level {level_labels[level]!r} has non-finite curvature."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scales = np.max(np.abs(eigenvalues), axis=1)
    # A large smoothing parameter can put O(1) data curvature beside an
    # O(1e11) wiggle penalty.  The usual dimension-scaled roundoff threshold
    # retains that real null-space information; eps**(2/3) would incorrectly
    # discard eigenvalues as large as about five at lambda=1e10.
    thresholds = np.finfo(np.float64).eps * symmetric.shape[1] * scales * 10.0
    negative = eigenvalues[:, 0] < -thresholds
    if np.any(negative):
        level = int(np.flatnonzero(negative)[0])
        raise np.linalg.LinAlgError(
            f"Structured term {term_name!r} level {level_labels[level]!r} "
            f"has negative local curvature ({eigenvalues[level, 0]:.17g})."
        )

    positive = eigenvalues > thresholds[:, None]
    full_rank = np.all(positive, axis=1)
    full_inverses = (
        np.linalg.inv(symmetric[full_rank])
        if np.any(full_rank)
        else np.empty((0, symmetric.shape[1], symmetric.shape[2]), dtype=np.float64)
    )
    full_inverse_index = 0
    locals_list = []
    for level in range(values.shape[0]):
        positive_values = np.asarray(
            eigenvalues[level][positive[level]],
            dtype=np.float64,
        )
        positive_vectors = np.asarray(
            eigenvectors[level][:, positive[level]],
            dtype=np.float64,
        )
        if full_rank[level]:
            # Divide-and-conquer eigenvectors can lose relative inverse
            # accuracy when O(1) data curvature shares a block with an
            # O(1e11) smoothing penalty.  Once the spectrum has certified
            # full rank, a batched direct inverse is both more accurate and
            # still compact.  Rank-deficient blocks retain the spectral
            # pseudo-inverse and null geometry below.
            pinv = full_inverses[full_inverse_index]
            full_inverse_index += 1
        elif positive_values.size:
            pinv = (positive_vectors / positive_values) @ positive_vectors.T
        else:
            pinv = np.zeros_like(symmetric[level])
        locals_list.append(
            _LocalPSD(
                pinv=0.5 * (pinv + pinv.T),
                null=np.asarray(
                    eigenvectors[level][:, ~positive[level]],
                    dtype=np.float64,
                ),
                positive_eigenvalues=positive_values,
                rank=int(positive_values.size),
            )
        )
    locals_ = tuple(locals_list)
    return locals_, float(np.min(eigenvalues))


def _constraint_equilibration(matrix: NDArray) -> NDArray:
    """Whiten the positive range of a small constraint covariance."""
    symmetric = _symmetric_part(np.asarray(matrix, dtype=np.float64))
    if symmetric.shape == (0, 0):
        return symmetric
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        symmetric,
        driver="evr",
        check_finite=False,
    )
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    threshold = np.finfo(np.float64).eps * max(symmetric.shape[0], 1) * scale * 10.0
    factors = np.ones_like(eigenvalues)
    positive = eigenvalues > threshold
    factors[positive] = 1.0 / np.sqrt(eigenvalues[positive])
    return (eigenvectors * factors) @ eigenvectors.T


class _SymmetricBorderFactor:
    """Equilibrated LDL execution on the shared signed-rank representative.

    Congruence preserves inertia, but its diagonal units must be removed
    before numerical rank is assessed. The shared decomposition owns inertia
    and determinant; neither LDL pivots nor a fallback select another rank.
    """

    def __init__(self, matrix: NDArray):
        border = np.asarray(matrix, dtype=np.float64)
        if border.ndim != 2 or border.shape[0] != border.shape[1]:
            raise ValueError("The constrained border must be square.")
        if not np.all(np.isfinite(border)):
            raise np.linalg.LinAlgError("The constrained border contains non-finite values.")
        self._original_matrix = _symmetric_part(border)
        self.size = len(border)
        self._authority = decompose_gram(self._original_matrix, allow_indefinite=True)
        self.matrix, scales, active, _ = _equilibrate_gram(
            self._original_matrix, allow_indefinite=True
        )
        self._scale = np.where(scales > 0.0, scales, 1.0)
        self.zero_count = self.size - self._authority.rank
        retained = self._authority.retained_values
        if retained is None:
            self.positive_count = self._authority.rank
            self.negative_count = 0
        else:
            self.positive_count = int(np.count_nonzero(retained > 0.0))
            self.negative_count = int(np.count_nonzero(retained < 0.0))
        self.logabsdet = self._authority.log_pdet
        self.condition_estimate = self._authority.pre_truncation_condition**2
        self.used_fallback = False
        self.fallback_reason: str | None = None
        self._triangular: NDArray | None = None
        self._permutation: NDArray | None = None
        self._inverse_pivots: tuple[tuple[slice, NDArray], ...] = ()
        if self.zero_count:
            return
        assert len(active) == self.size
        try:
            lu, diagonal, permutation = scipy.linalg.ldl(
                self.matrix, lower=True, hermitian=True, check_finite=False
            )
            action = np.abs(lu) @ np.abs(diagonal) @ np.abs(lu.T)
            reconstructed = lu @ diagonal @ lu.T
            unit = np.finfo(float).eps / 2
            gamma = (3 * self.size + 2) * unit / (1 - (3 * self.size + 2) * unit)
            # Check the represented factorization in componentwise action
            # units. The allowance contains no objective or constraint unit.
            allowance = gamma * (action + np.abs(self.matrix))
            if np.any(np.abs(reconstructed - self.matrix) > allowance):
                raise np.linalg.LinAlgError("LDL reconstruction exceeds its action allowance")
            pivots, inverse_pivots = self._analyze_ldl_pivots(diagonal)
            if (
                np.count_nonzero(pivots > 0.0) != self.positive_count
                or np.count_nonzero(pivots < 0.0) != self.negative_count
            ):
                raise np.linalg.LinAlgError("LDL inertia disagrees with the shared representative")
            self._triangular = lu[permutation, :]
            self._permutation = np.asarray(permutation, dtype=np.intp)
            self._inverse_pivots = inverse_pivots
        except (np.linalg.LinAlgError, ValueError) as error:
            self.used_fallback = True
            self.fallback_reason = f"constrained-border LDL fallback: {error}"

    def _analyze_ldl_pivots(
        self, diagonal: NDArray
    ) -> tuple[NDArray, tuple[tuple[slice, NDArray], ...]]:
        eigenvalues: list[float] = []
        inverse_blocks: list[tuple[slice, NDArray]] = []
        index = 0
        while index < self.size:
            width = 2 if index + 1 < self.size and diagonal[index, index + 1] != 0.0 else 1
            block_slice = slice(index, index + width)
            block = diagonal[block_slice, block_slice]
            values = np.linalg.eigvalsh(block)
            if np.any(values == 0.0):
                raise np.linalg.LinAlgError("LDL cannot execute a retained border direction")
            inverse = np.linalg.inv(block)
            if not np.all(np.isfinite(inverse)):
                raise np.linalg.LinAlgError("LDL pivot inverse is not representable")
            eigenvalues.extend(float(value) for value in values)
            inverse_blocks.append((block_slice, inverse))
            index += width
        return np.asarray(eigenvalues), tuple(inverse_blocks)

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.size:
            raise ValueError(f"border rhs must have shape ({self.size},) or ({self.size}, m)")
        if self.zero_count:
            raise np.linalg.LinAlgError("The constrained border is singular.")
        if self._triangular is None or self._permutation is None:
            solution = np.column_stack([self._authority.solve(column) for column in values.T])
        else:
            scaled_rhs = values / self._scale[:, None]
            forward = scipy.linalg.solve_triangular(
                self._triangular,
                scaled_rhs[self._permutation],
                lower=True,
                unit_diagonal=True,
                check_finite=False,
            )
            middle = np.zeros_like(forward)
            for block_slice, inverse in self._inverse_pivots:
                middle[block_slice] = inverse @ forward[block_slice]
            permuted_solution = scipy.linalg.solve_triangular(
                self._triangular.T,
                middle,
                lower=False,
                unit_diagonal=True,
                check_finite=False,
            )
            solution = np.empty_like(permuted_solution)
            solution[self._permutation] = permuted_solution
            solution /= self._scale[:, None]
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError("The constrained border solution is not representable.")
        return solution[:, 0] if vector_rhs else solution


class SumToZeroBlockFactor:
    """Factor an all-level PSD block system under an exact sum-to-zero constraint."""

    backend = "structured"

    def __init__(
        self,
        *,
        A: NDArray,
        C: NDArray,
        D: NDArray,
        small_indices: NDArray,
        structured_indices: NDArray,
        term_name: str,
        level_labels: tuple[Any, ...] | None = None,
        max_structured_inverse_block: int = 256,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64)
        self.small_indices = np.asarray(small_indices, dtype=np.intp)
        self.structured_indices = np.asarray(structured_indices, dtype=np.intp)
        self.term_name = term_name
        self.dominant_group_name = term_name
        self.max_structured_inverse_block = int(max_structured_inverse_block)
        if self.C.ndim != 3:
            raise ValueError("C must have shape (K, k, q).")
        self.n_levels, self.block_size, q = self.C.shape
        if self.n_levels < 2:
            raise ValueError("Sum-to-zero factors require at least two levels.")
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.D.shape != (self.n_levels, self.block_size, self.block_size):
            raise ValueError("D must have shape (K, k, k).")
        if self.small_indices.shape != (q,):
            raise ValueError("small_indices width does not match A.")
        if self.structured_indices.shape != (self.n_levels - 1, self.block_size):
            raise ValueError("structured_indices must have shape (K - 1, k).")
        if (
            not np.all(np.isfinite(self.A))
            or not np.all(np.isfinite(self.C))
            or not np.all(np.isfinite(self.D))
        ):
            # D belongs here, above the symmetry checks, for the reason
            # BlockSchurFactor already tests its own blocks first: a NaN fails
            # `np.allclose` against its own transpose, so without this it would
            # be refused as an asymmetric block -- a structural complaint about
            # a condition the iterate caused. Callers separate the two by type,
            # so the wrong class sends a recoverable point down the fatal path.
            # An inf never had that problem: matching infs compare equal, so it
            # already reached the curvature check below.
            raise np.linalg.LinAlgError(
                f"Structured SZ term {term_name!r} has non-finite ordinary, cross, or local blocks."
            )

        def symmetric_in_block_units(matrix: NDArray) -> bool:
            scale = float(np.max(np.abs(matrix), initial=0.0))
            if scale == 0.0:
                return True
            normalized = matrix / scale
            allowance = (len(matrix) + 2) * np.finfo(float).eps
            return bool(np.all(np.abs(normalized - normalized.T) <= allowance))

        if not symmetric_in_block_units(self.A):
            raise ValueError("A must be symmetric.")
        if not all(symmetric_in_block_units(block) for block in self.D):
            raise ValueError("Every local D block must be symmetric.")
        all_indices = np.concatenate((self.small_indices, self.structured_indices.ravel()))
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        self.shape = (len(all_indices), len(all_indices))
        if level_labels is None:
            self.level_labels = tuple(range(self.n_levels))
        else:
            self.level_labels = tuple(level_labels)
            if len(self.level_labels) != self.n_levels:
                raise ValueError("level_labels length must equal K.")

        self._locals, self.minimum_local_eigenvalue = _decompose_local_psd_batch(
            self.D,
            term_name=term_name,
            level_labels=self.level_labels,
        )
        self._pinv = np.stack([local.pinv for local in self._locals])
        self._positive_rank = sum(local.rank for local in self._locals)
        self.deficient_levels = tuple(
            label
            for label, local in zip(self.level_labels, self._locals, strict=True)
            if local.rank < self.block_size
        )
        self.minimum_local_diagonal = self.minimum_local_eigenvalue

        null_widths = [local.null.shape[1] for local in self._locals]
        self._null_width = int(sum(null_widths))
        gamma_slices: list[slice] = []
        offset = 0
        for width in null_widths:
            gamma_slices.append(slice(offset, offset + width))
            offset += width
        self._gamma_slices = tuple(gamma_slices)

        Q = np.array(self.A, copy=True)
        R = np.zeros((self.block_size, q))
        M = np.zeros((self.block_size, self.block_size))
        E = np.zeros((q, self._null_width))
        N = np.zeros((self.block_size, self._null_width))
        self._pinv_cross = np.empty_like(self.C)
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            pinv_cross = local.pinv @ self.C[level]
            self._pinv_cross[level] = pinv_cross
            Q -= self.C[level].T @ pinv_cross
            R += pinv_cross
            M += local.pinv
            E[:, gamma_slice] = self.C[level].T @ local.null
            N[:, gamma_slice] = local.null
        Q = _symmetric_part(Q)
        border = np.block(
            [
                [Q, E, -R.T],
                [
                    E.T,
                    np.zeros((self._null_width, self._null_width)),
                    N.T,
                ],
                [-R, N, -M],
            ]
        )
        self._border = border
        constraint_transform = _constraint_equilibration(M)
        border_transform = np.eye(border.shape[0], dtype=np.float64)
        # Ordinary and local-null coordinates both have coefficient units.
        # Whitening only multipliers leaves their coupling proportional to
        # sqrt(curvature), so auxiliary units can manufacture rank loss in a
        # well-conditioned public Hessian. Normalize every coefficient block
        # by the same curvature root before the shared signed-rank decision.
        curvature_scale = max(
            float(np.max(np.abs(self.A), initial=0.0)),
            float(np.max(np.abs(self.C), initial=0.0)),
            float(np.max(np.abs(self.D), initial=0.0)),
        )
        if curvature_scale > 0.0:
            coefficient_indices = np.arange(q + self._null_width)
            border_transform[coefficient_indices, coefficient_indices] = 1.0 / np.sqrt(
                curvature_scale
            )
        border_transform[-self.block_size :, -self.block_size :] = constraint_transform
        scaled_border = border_transform.T @ border @ border_transform
        self._border_transform = border_transform
        self._border_factor = _SymmetricBorderFactor(scaled_border)
        expected_positive = q + self._null_width
        expected_negative = self.block_size
        if (
            self._border_factor.positive_count != expected_positive
            or self._border_factor.negative_count != expected_negative
            or self._border_factor.zero_count
        ):
            raise SumToZeroIdentifiabilityError(
                f"Structured SZ term {term_name!r} is globally unidentifiable after "
                f"enforcing sum-to-zero; deficient fitted levels={self.deficient_levels!r}. "
                "Use basis='fs', reduce k, or provide more numeric support."
            )

        local_logdet = sum(
            float(np.sum(np.log(local.positive_eigenvalues)))
            for local in self._locals
            if local.positive_eigenvalues.size
        )
        transform_logdet = np.linalg.slogdet(border_transform)[1]
        self._logdet = local_logdet + self._border_factor.logabsdet - 2.0 * float(transform_logdet)
        self.rank = self.shape[0]
        self.rank_truncated = False
        self.public_positive_definite = True
        self.used_dense_fallback = self._border_factor.used_fallback
        self.fallback_reason = self._border_factor.fallback_reason
        self.schur_condition_estimate = self._border_factor.condition_estimate
        self._small_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._structured_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._structured_position[self.structured_indices.ravel()] = np.arange(
            (self.n_levels - 1) * self.block_size
        )
        self._border_inverse_cache: NDArray | None = None
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None
        self._raw_border_basis = self._build_raw_border_basis()
        self._public_border_basis = np.zeros((self.shape[0], border.shape[0]))
        self._public_border_basis[self.small_indices, :q] = np.eye(q)
        self._public_border_basis[self.structured_indices] = self._raw_border_basis[:-1]
        self._certify_public_rank()

    def _certify_public_rank(self) -> None:
        """Require public-coordinate rank in addition to auxiliary inertia.

        The ordinary path bounds the residual of the represented compact
        inverse without forming a public matrix. If that sufficient certificate
        is unresolved, only a matrix within the existing inverse-block cap may
        be materialized for the shared Gram authority.
        """
        if self._compact_public_rank_certificate():
            return
        if self.shape[0] <= self.max_structured_inverse_block:
            public = np.zeros(self.shape)
            public[np.ix_(self.small_indices, self.small_indices)] = self.A
            for level, indices in enumerate(self.structured_indices):
                cross = self.C[level] - self.C[-1]
                public[np.ix_(indices, self.small_indices)] = cross
                public[np.ix_(self.small_indices, indices)] = cross.T
                for other, other_indices in enumerate(self.structured_indices):
                    block = self.D[-1] + (self.D[level] if other == level else 0.0)
                    public[np.ix_(indices, other_indices)] = block
            if decompose_gram(public).rank == self.shape[0]:
                return
        raise SumToZeroIdentifiabilityError(
            f"Structured SZ term {self.term_name!r} is globally unidentifiable or its "
            "public numerical rank is unresolved after enforcing sum-to-zero."
        )

    def _compact_public_rank_certificate(self) -> bool:
        """Certify the original public operator using a compact inverse residual.

        For E=S^-1 H S^-1 and the represented Z=S G S, rho>=||I-EZ||_F
        implies ||E^-1||_2 <= ||Z||_2/(1-rho). Absolute row/column actions
        bound both operator norms, so (1-rho)/(E_bound * Z_bound) is a relative
        eigenvalue-magnitude lower bound. No exact unit diagonal or PSD trace
        identity is assumed. Inertia is checked separately above.

        Products remain block-diagonal plus low rank. The squared Frobenius
        residual uses three contractions, with absolute expanded contractions
        bounding their cancellation. A separate formation allowance relates
        that product back to the original A/C/D, including |C_i|+|C_last|.
        The gamma count dominates normalization, block products, thin products,
        their row-length contractions, and the final reductions. All bounds
        use the float epsilon even where extended arithmetic is used.
        """
        p = self.shape[0]
        q = len(self.small_indices)
        k = self.block_size
        indices = self.structured_indices
        extended = np.longdouble
        diagonal = np.empty(p, dtype=extended)
        diagonal[self.small_indices] = np.diag(self.A)
        diagonal[indices] = np.diagonal(self.D[:-1], axis1=1, axis2=2).astype(extended)
        diagonal[indices] += np.diag(self.D[-1]).astype(extended)
        if np.any(diagonal <= 0.0):
            return False
        scales = np.sqrt(diagonal)
        local_scales = scales[indices]
        # The shared Gram authority acts on the symmetric public matrix. Work
        # from these small symmetric moment blocks, retaining the original
        # absolute moments for the formation-error enclosure below.
        symmetric_A = (self.A.astype(extended) + self.A.T.astype(extended)) * extended(0.5)
        symmetric_D = (self.D.astype(extended) + self.D.swapaxes(1, 2).astype(extended)) * extended(
            0.5
        )

        # H = blockdiag(D_i) + V K V'. Normalize columns as well as public
        # coordinates so a uniform curvature unit does not create large thin
        # intermediates before they cancel.
        V = np.zeros((p, 2 * q + k), dtype=extended)
        V[self.small_indices, :q] = np.eye(q) / scales[self.small_indices, None]
        V[indices, q : 2 * q] = (
            self.C[:-1].astype(extended) - self.C[-1].astype(extended)
        ) / local_scales[:, :, None]
        V[indices, 2 * q :] = np.eye(k) / local_scales[:, :, None]
        K = np.zeros((2 * q + k, 2 * q + k), dtype=extended)
        K[:q, :q] = symmetric_A
        K[:q, q : 2 * q] = np.eye(q)
        K[q : 2 * q, :q] = np.eye(q)
        K[2 * q :, 2 * q :] = symmetric_D[-1]
        U = self._public_border_basis.astype(extended) * scales[:, None]
        J = self._border_inverse().astype(extended)

        def normalize(basis, core):
            units = np.max(np.abs(basis), axis=0, initial=0.0)
            units = np.where(units > 0.0, units, 1.0)
            return basis / units, core * units[:, None] * units[None, :]

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            exact_V, exact_K = normalize(V, K)
            exact_Be = symmetric_D[:-1] / local_scales[:, :, None] / local_scales[:, None, :]
            V, K, Be = (np.asarray(value, dtype=float) for value in (exact_V, exact_K, exact_Be))
            U, J = (np.asarray(value, dtype=float) for value in normalize(U, J))
            Bz = np.asarray(
                self._pinv[:-1].astype(extended)
                * local_scales[:, :, None]
                * local_scales[:, None, :],
                dtype=float,
            )
        if not all(np.all(np.isfinite(value)) for value in (V, K, U, J, Be, Bz)):
            return False
        # The moment representation uses a relative formation bound. Refuse
        # casts outside its normal range; the inverse representation may be
        # arbitrary because its entire residual is checked below.
        for source, represented in ((exact_V, V), (exact_K, K), (exact_Be, Be)):
            if np.any((source != 0.0) & (np.abs(represented) < np.finfo(float).tiny)):
                return False

        r = U.shape[1] + V.shape[1]
        # This dominates p + 3*r + 2*k + 32 for complete product/formation
        # paths and pk+2, pr+k+3, 2p+r^2+3 for the three scalar contractions.
        count = 4 * p * r + 8 * r * r + 4 * p * k + 32
        product = extended(count) * (np.finfo(float).eps / 2)
        if product >= 0.5:
            return False
        gamma = np.nextafter(product / (1.0 - product), extended(np.inf))
        inflate = np.nextafter(1.0 / (1.0 - gamma), extended(np.inf))

        def apply(blocks, basis):
            result = np.zeros_like(basis)
            result[indices] = blocks @ basis[indices]
            return result

        with np.errstate(over="ignore", invalid="ignore"):
            left = np.column_stack((apply(Be, U), V))
            right = np.column_stack((U, apply(Bz.swapaxes(1, 2), V)))
            core = np.zeros((r, r))
            width = U.shape[1]
            core[:width, :width] = J
            core[width:, :width] = K @ (V.T @ U) @ J
            core[width:, width:] = K
            thin = left @ core
            base = np.eye(k)[None, :, :] - Be @ Bz
            base_right = apply(base, right)
            base_right[self.small_indices] = right[self.small_indices]
            base_norm = float(np.sum(base * base)) + q
            cross = float(np.sum(thin * base_right))
            low = float(np.sum((thin.T @ thin) * (right.T @ right)))
            squared = base_norm - 2.0 * cross + low

            abs_right = np.abs(right)
            abs_thin = np.abs(thin)
            base_action = apply(np.abs(base), abs_right)
            base_action[self.small_indices] = abs_right[self.small_indices]
            squared_action = (
                base_norm
                + 2.0 * float(np.sum(abs_thin * base_action))
                + float(np.sum((abs_thin.T @ abs_thin) * (abs_right.T @ abs_right)))
            )
            z_rows = np.abs(U) @ (np.abs(J) @ np.sum(np.abs(U), axis=0))
            z_rows[indices] += np.sum(np.abs(Bz), axis=2)
            z_columns = np.abs(U) @ (np.abs(J).T @ np.sum(np.abs(U), axis=0))
            z_columns[indices] += np.sum(np.abs(Bz), axis=1)

        # Absolute actions use the supplied moments, not the possibly cancelled
        # public cross block or the private Schur complement.
        cross_action = (
            (np.abs(self.C[:-1].astype(extended)) + np.abs(self.C[-1].astype(extended)))
            / local_scales[:, :, None]
            / scales[self.small_indices]
        )
        e_rows = np.zeros(p, dtype=extended)
        e_rows[self.small_indices] = np.sum(
            np.abs(self.A.astype(extended))
            / scales[self.small_indices, None]
            / scales[None, self.small_indices],
            axis=1,
        ) + np.sum(cross_action, axis=(0, 1))
        e_rows[indices] = (
            np.sum(np.abs(Be.astype(extended)), axis=2)
            + np.sum(cross_action, axis=2)
            + (np.abs(self.D[-1].astype(extended)) @ np.sum(1.0 / local_scales, axis=0))[None, :]
            / local_scales
        )
        e_columns = np.zeros(p, dtype=extended)
        e_columns[self.small_indices] = np.sum(
            np.abs(self.A.astype(extended))
            / scales[self.small_indices, None]
            / scales[None, self.small_indices],
            axis=0,
        ) + np.sum(cross_action, axis=(0, 1))
        e_columns[indices] = (
            np.sum(np.abs(Be.astype(extended)), axis=1)
            + np.sum(cross_action, axis=2)
            + (np.abs(self.D[-1].astype(extended)).T @ np.sum(1.0 / local_scales, axis=0))[None, :]
            / local_scales
        )
        e_norm = max(np.max(e_rows, initial=0.0), np.max(e_columns, initial=0.0)) * inflate
        z_norm = (
            extended(max(np.max(z_rows, initial=0.0), np.max(z_columns, initial=0.0))) * inflate
        )
        squared_upper = (max(0.0, squared) + gamma * squared_action) * inflate
        formation = extended(gamma) * np.sqrt(extended(p)) * (1.0 + e_norm * z_norm)
        rho = np.sqrt(extended(squared_upper)) + formation
        rho = np.nextafter(rho * inflate, extended(np.inf))
        spectral_error = extended(_eigensolver_relative_bar(p))
        relative_cutoff = max(SHARED_RANK_POLICY.gram_rcond, spectral_error)
        # Clear the shared cutoff even after its stated eigensolver error.
        # An inconclusive bound falls back; this does not change that cutoff.
        required = (
            (relative_cutoff + spectral_error * (1.0 + relative_cutoff)) * e_norm * z_norm * inflate
        )
        return bool(
            np.isfinite(rho)
            and np.isfinite(required)
            and rho < 1.0
            and (1.0 - rho) / inflate > required
        )

    def _build_raw_border_basis(self) -> NDArray:
        q = len(self.small_indices)
        basis = np.zeros(
            (
                self.n_levels,
                self.block_size,
                q + self._null_width + self.block_size,
            )
        )
        basis[:, :, :q] = -self._pinv_cross
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            basis[level, :, q + gamma_slice.start : q + gamma_slice.stop] = local.null
        basis[:, :, q + self._null_width :] = -self._pinv
        return basis

    def _border_inverse(self) -> NDArray:
        if self._border_inverse_cache is None:
            self._border_inverse_cache = self._solve_border(np.eye(self._border.shape[0]))
            self._border_inverse_cache = 0.5 * (
                self._border_inverse_cache + self._border_inverse_cache.T
            )
        return self._border_inverse_cache

    def _solve_border(self, rhs: NDArray) -> NDArray:
        transformed_rhs = self._border_transform.T @ np.asarray(rhs, dtype=np.float64)
        scaled_solution = self._border_factor.solve(transformed_rhs)
        return self._border_transform @ scaled_solution

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve the public ``K - 1`` coordinate system without materializing it."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        q = len(self.small_indices)
        raw_rhs = np.zeros((self.n_levels, self.block_size, values.shape[1]))
        raw_rhs[:-1] = values[self.structured_indices]
        pinv_rhs = np.einsum("kij,kjm->kim", self._pinv, raw_rhs, optimize=True)
        border_small = values[self.small_indices] - np.einsum(
            "kiq,kim->qm",
            self.C,
            pinv_rhs,
            optimize=True,
        )
        border_null = np.empty((self._null_width, values.shape[1]))
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            border_null[gamma_slice] = local.null.T @ raw_rhs[level]
        border_multiplier = -np.sum(pinv_rhs, axis=0)
        border_rhs = np.vstack((border_small, border_null, border_multiplier))
        border_solution = self._solve_border(border_rhs)
        small_solution = border_solution[:q]
        gamma = border_solution[q : q + self._null_width]
        multiplier = border_solution[q + self._null_width :]
        raw_solution = (
            pinv_rhs
            - np.einsum(
                "kiq,qm->kim",
                self._pinv_cross,
                small_solution,
                optimize=True,
            )
            - np.einsum("kij,jm->kim", self._pinv, multiplier, optimize=True)
        )
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            raw_solution[level] += local.null @ gamma[gamma_slice]
        solution = np.empty_like(values)
        solution[self.small_indices] = small_solution
        solution[self.structured_indices] = raw_solution[:-1]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        return self._logdet

    def _validate_selected_indices(self, indices: NDArray) -> NDArray[np.intp]:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def _selected_base_covariance(self, selected: NDArray) -> NDArray:
        base = np.zeros((len(selected), len(selected)))
        structured = self._structured_position[selected]
        for row, left_position in enumerate(structured):
            if left_position < 0:
                continue
            left_level, left_coordinate = divmod(left_position, self.block_size)
            for column, right_position in enumerate(structured):
                if right_position < 0:
                    continue
                right_level, right_coordinate = divmod(right_position, self.block_size)
                if left_level == right_level:
                    base[row, column] = self._pinv[
                        left_level,
                        left_coordinate,
                        right_coordinate,
                    ]
        return base

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        structured_count = int(np.count_nonzero(self._structured_position[selected] >= 0))
        if structured_count > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {structured_count} x {structured_count} "
                f"inverse block for structured term {self.term_name!r}; "
                "request its diagonal instead."
            )
        basis = self._public_border_basis[selected]
        return self._selected_base_covariance(selected) + basis @ self._border_inverse() @ basis.T

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        diagonal = np.zeros(len(selected))
        structured = self._structured_position[selected]
        structured_mask = structured >= 0
        if np.any(structured_mask):
            levels, coordinates = np.divmod(structured[structured_mask], self.block_size)
            diagonal[structured_mask] = self._pinv[levels, coordinates, coordinates]
        basis = self._public_border_basis[selected]
        diagonal += np.sum((basis @ self._border_inverse()) * basis, axis=1)
        return diagonal

    def raw_level_inverse_block(self, level: int) -> NDArray:
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
            raise TypeError("level must be an integer index.")
        level = int(level)
        if level < 0 or level >= self.n_levels:
            raise IndexError("raw level index is outside the fitted level range.")
        basis = self._raw_border_basis[level]
        covariance = self._pinv[level] + basis @ self._border_inverse() @ basis.T
        return 0.5 * (covariance + covariance.T)

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        cached = self._inverse_bdlr_cache
        if cached is None:
            cached = _BlockDiagonalLowRank(
                blocks=self._pinv[:-1],
                structured_indices=self.structured_indices,
                basis=self._public_border_basis,
                core=self._border_inverse(),
                shape=self.shape,
            )
            self._inverse_bdlr_cache = cached
        return cached

    def _penalty_operator(
        self,
        component: PenaltyComponent,
        scale: float,
    ) -> CompactSymmetricOperator:
        indices = _component_indices(component, self.shape[0])
        local_small = self._small_position[indices]
        local_structured = self._structured_position[indices]
        A = np.zeros_like(self.A)
        C = np.zeros_like(self.C)
        if component.penalty_kind == "sum_to_zero":
            if not np.all(local_structured >= 0):
                raise ValueError("Sum-to-zero penalty must lie in the structured block.")
            if (
                component.repeat_count != self.n_levels
                or component.block_width != self.block_size
                or not np.array_equal(
                    indices.reshape(self.n_levels - 1, self.block_size),
                    self.structured_indices,
                )
            ):
                raise ValueError("Sum-to-zero penalty geometry does not match the block factor.")
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (self.block_size, self.block_size):
                raise ValueError("Sum-to-zero penalty local matrix has the wrong shape.")
            return SumToZeroBlockOperator(
                A=A,
                C=C,
                D=np.broadcast_to(scale * omega, self.D.shape),
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
                return SumToZeroBlockOperator(
                    A=A,
                    C=C,
                    D=np.zeros_like(self.D),
                    small_indices=self.small_indices,
                    structured_indices=self.structured_indices,
                )
            if np.all(local_structured >= 0):
                public_D = np.zeros((self.n_levels - 1, self.block_size, self.block_size))
                for position in local_structured:
                    level, coordinate = divmod(position, self.block_size)
                    public_D[level, coordinate, coordinate] = scale
                return BlockSymmetricOperator(
                    A=A,
                    C=np.zeros((self.n_levels - 1, self.block_size, len(self.small_indices))),
                    D=public_D,
                    small_indices=self.small_indices,
                    structured_indices=self.structured_indices,
                )
            raise ValueError("Identity penalty crosses structured partitions.")
        if not np.all(local_small >= 0):
            raise ValueError("Dense penalties must lie in the factor's small block.")
        A[np.ix_(local_small, local_small)] = scale * _component_omega(
            component,
            self.shape[0],
        )
        return SumToZeroBlockOperator(
            A=A,
            C=C,
            D=np.zeros_like(self.D),
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.trace_inverse_operator(self._penalty_operator(component, 1.0))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        return self.operator_cross_trace(
            self._penalty_operator(left, left_scale),
            self._penalty_operator(right, right_scale),
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr_coalesced(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr_coalesced(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
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
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(right, self.structured_indices),
            ),
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        return self.operator_cross_trace(
            self._penalty_operator(component, scale),
            operator,
        )


class ProfiledSumToZeroBlockFactor:
    """Profiled slope view of an augmented sum-to-zero factor."""

    backend = "structured"

    def __init__(
        self,
        *,
        augmented_factor: SumToZeroBlockFactor,
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
        if not len(augmented_factor.small_indices) or augmented_factor.small_indices[0] != 0:
            raise ValueError("The augmented intercept must be the first dense-small coefficient.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self.n_levels = augmented_factor.n_levels
        self.block_size = augmented_factor.block_size
        self.rank = max(augmented_factor.rank - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.used_dense_fallback = augmented_factor.used_dense_fallback
        self.fallback_reason = augmented_factor.fallback_reason
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_eigenvalue = augmented_factor.minimum_local_eigenvalue
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.deficient_levels = augmented_factor.deficient_levels
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None

    @staticmethod
    def _shift_indices(indices: NDArray) -> NDArray[np.intp]:
        return np.asarray(indices, dtype=np.intp) + 1

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start = component.group_sl.start
        stop = component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(
            component,
            group_sl=slice(start + 1, stop + 1, component.group_sl.step),
        )

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
        solution = self.augmented_factor.solve(augmented_rhs)[1:]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_block(self._shift_indices(indices))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_diagonal(self._shift_indices(indices))

    def raw_level_inverse_block(self, level: int) -> NDArray:
        return self.augmented_factor.raw_level_inverse_block(level)

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
        cached = self._inverse_bdlr_cache
        if cached is None:
            augmented = self.augmented_factor._inverse_bdlr()
            cached = _BlockDiagonalLowRank(
                blocks=augmented.blocks,
                structured_indices=self.structured_indices,
                basis=augmented.basis[1:],
                core=augmented.core,
                shape=self.shape,
            )
            self._inverse_bdlr_cache = cached
        return cached

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr_coalesced(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr_coalesced(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
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
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(right, self.structured_indices),
            ),
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        shifted = self._shift_component(component)
        penalty = self.augmented_factor._penalty_operator(shifted, scale)
        if isinstance(penalty, SumToZeroBlockOperator):
            slope_penalty: CompactSymmetricOperator = SumToZeroBlockOperator(
                A=penalty.A[1:, 1:],
                C=penalty.C[:, :, 1:],
                D=penalty.D,
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        elif isinstance(penalty, BlockSymmetricOperator):
            slope_penalty = BlockSymmetricOperator(
                A=penalty.A[1:, 1:],
                C=penalty.C[:, :, 1:],
                D=penalty.D,
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        else:  # pragma: no cover - _penalty_operator contract
            raise TypeError("Profiled SZ penalties must use block-compatible operators.")
        return self.operator_cross_trace(slope_penalty, operator)


__all__ = [
    "ProfiledSumToZeroBlockFactor",
    "SumToZeroBlockFactor",
    "SumToZeroIdentifiabilityError",
]
