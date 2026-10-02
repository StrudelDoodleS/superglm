"""Stable intercept-profiled systems shared by fitting and inference."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import (
    _compensated_add,
    _try_mixed_discrete_centering,
    _try_raw_spline_tabmat_centering,
    _try_tabmat_centering,
    centered_gram_rhs,
    packed_centered_gram_rhs,
    stable_centered_gram_rhs,
    try_raw_moment_centering,
)
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.solvers.mode_score import (
    corrected_two_pass_pair,
    dense_centred_matvec,
    dense_centred_rmatvec,
    dense_columns,
)

_FACTOR_CHUNK_BYTES = 16 * 1024 * 1024
_FACTOR_CHUNK_ROWS = 8192


def _freeze(values: NDArray) -> NDArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CenteredSystem:
    """Complete weighted system after profiling the intercept."""

    sum_w: float
    mean_x: NDArray
    mean_z: float
    data_gram: NDArray
    rhs: NDArray
    penalty: NDArray
    hessian: NDArray
    # The centre the rows were taken about as an exact pair, ``(x - mean_hi) -
    # mean_lo`` (``weighted_mean_pair``): a dense column's anchor and its small
    # remainder, ``mean_x`` and zero elsewhere.  ``None`` for a system formed
    # from raw moments, whose rows are centred about ``mean_x`` alone.
    mean_hi: NDArray | None = None
    mean_lo: NDArray | None = None

    def centre_pair(self) -> tuple[NDArray, NDArray | None]:
        """``(hi, lo)`` for centring rows as ``(x - hi) - lo`` (``lo`` ``None``: ``x - mean_x``)."""
        if self.mean_hi is None or self.mean_lo is None:
            return self.mean_x, None
        return self.mean_hi, self.mean_lo

    def raw_weighted_moments(self) -> tuple[NDArray, NDArray, NDArray, float]:
        """Recover raw Gram/RHS moments from the stable centered system."""
        xtw1 = self.sum_w * self.mean_x
        sum_wz = self.sum_w * self.mean_z
        gram = self.data_gram + self.sum_w * np.outer(self.mean_x, self.mean_x)
        xtwz = self.rhs + self.mean_x * sum_wz
        return gram, xtw1, xtwz, sum_wz


@dataclass
class TabmatCenteringState:
    """Fit-local safety decision for accelerated raw-moment centering."""

    eligible: bool | None = None
    raw_spline_eligible: bool | None = None
    raw_moment_eligible: bool | None = None
    _raw_moment_owners: tuple = ()

    def seed_raw_rejection(self, owners: tuple) -> bool | None:
        """Carry only a refusal within one fixed-coordinate optimizer owner."""
        if len(owners) != len(self._raw_moment_owners) or any(
            current is not previous for current, previous in zip(owners, self._raw_moment_owners)
        ):
            self.raw_moment_eligible = None
        self._raw_moment_owners = owners
        return False if owners and self.raw_moment_eligible is False else None


@dataclass
class _InitialDataReuse:
    """One initial data system, owned by a fixed-design synchronous line search.

    The owner must discard this entry before changing source coordinates or
    invoking user callbacks. No penalty, Hessian, or fitted state is retained.
    """

    key: tuple | None = None
    weights: NDArray | None = None
    response: NDArray | None = None
    before: TabmatCenteringState | None = None
    after: TabmatCenteringState | None = None
    data: tuple | None = None

    def take(self, key, W, z_off, state, penalty):
        if self.key != key or self.before != state or self.data is None:
            return None
        for actual, saved in ((W, self.weights), (z_off, self.response)):
            if (
                actual.dtype != np.float64
                or actual.shape != saved.shape
                or not np.array_equal(actual.view(np.uint64), saved.view(np.uint64))
            ):
                return None
        state.eligible = self.after.eligible
        state.raw_spline_eligible = self.after.raw_spline_eligible
        state.raw_moment_eligible = self.after.raw_moment_eligible
        *data, mean_hi, mean_lo = self.data
        return _attach_centered_penalty(*data, penalty, mean_hi=mean_hi, mean_lo=mean_lo)

    def remember(self, key, W, z_off, before, after, system):
        self.key = key
        self.weights, self.response = _freeze(W), _freeze(z_off)
        self.before, self.after = replace(before), replace(after)
        self.data = (
            system.sum_w,
            system.mean_x,
            system.mean_z,
            system.data_gram,
            system.rhs,
            system.mean_hi,
            system.mean_lo,
        )


@dataclass
class _FisherDataReuse:
    """Unpenalized data owned by one synchronous, fixed-coordinate optimizer."""

    owners: tuple = ()
    weights: NDArray | None = None
    data: tuple | None = None

    def clear(self):
        self.owners = ()
        self.weights = self.data = None

    def take(self, owners, W):
        if (
            len(owners) != len(self.owners)
            or any(current is not saved for current, saved in zip(owners, self.owners))
            or self.weights is None
            or W.dtype != np.float64
            or W.shape != self.weights.shape
            or not np.array_equal(W.view(np.uint64), self.weights.view(np.uint64))
        ):
            self.clear()
        self.owners = owners
        return self.data

    def remember(self, W, system):
        self.weights = _freeze(W)
        self.data = (system.sum_w, system.mean_x, system.data_gram, system.mean_hi, system.mean_lo)


def iter_grouped_design_chunks(dm: DesignMatrix) -> Iterator[tuple[int, int, NDArray]]:
    """Yield bounded dense row blocks from a grouped design."""
    bytes_per_row = 3 * np.dtype(np.float64).itemsize * max(dm.p, 1)
    chunk_rows = max(1, min(_FACTOR_CHUNK_ROWS, _FACTOR_CHUNK_BYTES // bytes_per_row))
    for start in range(0, dm.n, chunk_rows):
        stop = min(start + chunk_rows, dm.n)
        rows = np.arange(start, stop, dtype=np.intp)
        yield start, stop, np.asarray(dm.row_subset(rows).toarray(), dtype=np.float64)


def grouped_weighted_factor(
    dm: DesignMatrix,
    W: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> NDArray:
    """Return a streaming weighted QR factor without retaining all design rows."""
    from superglm.solvers.rank import streamed_weighted_factor

    return streamed_weighted_factor(
        iter_grouped_design_chunks(dm), W, center=center, center_lo=center_lo
    )


def grouped_weighted_factor_rhs(
    dm: DesignMatrix,
    W: NDArray,
    response: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return a bounded weighted QR factor and its transformed response."""
    from superglm.solvers.rank import streamed_weighted_factor_rhs

    return streamed_weighted_factor_rhs(
        iter_grouped_design_chunks(dm),
        W,
        response,
        center=center,
        center_lo=center_lo,
    )


def penalty_factor(penalty: NDArray) -> NDArray:
    """Return a factor ``R`` with ``R'R`` the PSD penalty ``S`` to within its own resolution.

    Every rank decision downstream counts a row of ``R`` along a data-null
    direction as identifying it, so ``R`` keeps only the curvature ``S``
    certifies.  An exactly zero row of ``S`` is an exact null and is left out.
    The rest splits exactly into its contiguous diagonal blocks (no entry
    couples them; a penalty is block diagonal by term).  Each block is
    Jacobi-equilibrated, ``A_b = D S_b D`` with ``D = diag(S_b)^(-1/2)``, keeps
    the eigenpairs of ``A_b`` above its eigensolver resolution ``n_b eps
    ||A_b||_2`` (*LAPACK Users' Guide*, 3rd ed., sec. 4.7;
    ``rank._eigensolver_relative_bar``, the floor of the Gram route's rank cut,
    which equilibrates the same way; accumulated outward, ``_outward_cut``),
    and maps the root back, ``R_b = W^(1/2) V' D^(-1)``.  ``R_b'R_b = D^(-1)
    (A_b)_+ D^(-1)`` is the projection of
    ``S_b`` onto the PSD cone in the norm ``||D X D||_F`` (Higham 2002, IMA J.
    Numer. Anal. 22, Thm 3.2), less the eigenvalues below the bar.

    Below the bar an eigenvalue's magnitude and sign are rounding; keeping it
    (the former ``> 0.0`` test) invented curvature along an exact null.
    Measured: an ``sz`` term's unpenalized natural coordinates (exactly zero
    rows) came out of one ``eigh`` of the whole matrix at up to ``+5.3e-12``
    against ``||S||_2 = 3.9e4``, which made a thin level's exact data-null
    alias identified on gram (rank +1 against the structured solver).

    The cut is on ``A_b``, not ``S_b``, so no coordinate's units decide its
    rank.  A sum of PSD terms (``lambda_j D_j'D_j``, a natural
    parameterization, a Kronecker sum) is rounded entrywise within ``gamma_n
    sqrt(S_ii S_jj)`` (the dot-product bound and Cauchy-Schwarz), the scaled
    perturbation under which ``S``'s eigenvalues are determined to relative
    accuracy ``||A^-1||_2`` times its size, however graded ``S`` is (Demmel &
    Veselic 1992, SIAM J. Matrix Anal. Appl. 13(4); Drmac 2020,
    arXiv:2006.02753, Thms 3.7 and 3.11), and Jacobi scaling is within a
    factor ``n`` of the best diagonal scaling (van der Sluis 1969, Numer.
    Math. 14; Drmac, Thm 3.12).  The unscaled cut ``n_b eps ||S_b||_2``
    dropped the ``1e-6`` mode of ``[[1e10, 1e-10], [1e-10, 1e-6]]`` (bar
    ``4.4e-6``), whose ``A_b`` is the identity to ``1e-12``, and gram then
    rejected every step of a fit with that penalty.

    A block that is not PSD at its scaled resolution has no scaled model:
    a nonzero row on a non-positive diagonal, or an eigenvalue of ``A_b``
    below ``decompose_gram``'s materially-indefinite bar.  Its small entries
    are not known to be data rather than rounding (Drmac, sec. 3.3), so it
    keeps the eigenpairs above ``n_b eps ||S_b||_2`` of ``S_b`` itself.
    """
    width = penalty.shape[0]
    symmetric = 0.5 * (penalty + penalty.T)
    if penalty.shape == (0, 0) or not np.any(symmetric):
        return np.empty((0, width))
    support = np.flatnonzero(np.any(symmetric != 0.0, axis=1))
    coupled = symmetric[np.ix_(support, support)] != 0.0
    order = np.arange(len(support))
    last = (len(support) - 1) - np.argmax(coupled[:, ::-1], axis=1)
    ends = np.flatnonzero(np.maximum.accumulate(np.maximum(last, order)) == order) + 1
    starts = np.concatenate(([0], ends[:-1]))
    single = ends - starts == 1
    # 1 x 1 blocks are their own eigenpairs, exactly: positive is above the bar
    diagonal = support[starts[single]]
    values = symmetric[diagonal, diagonal]
    diagonal = diagonal[values > 0.0]
    factor = np.zeros((len(diagonal), width))
    factor[np.arange(len(diagonal)), diagonal] = np.sqrt(symmetric[diagonal, diagonal])
    rows = [factor]
    for start, stop in zip(starts[~single], ends[~single], strict=True):
        columns = support[start:stop]
        block = symmetric[np.ix_(columns, columns)]
        root = _equilibrated_block_root(block)
        if root is None:
            root = _block_root(block)
        embedded = np.zeros((root.shape[0], width))
        embedded[:, columns] = root
        rows.append(embedded)
    return np.vstack(rows)


def _block_root(block: NDArray) -> NDArray:
    """The eigenpairs of ``block`` above ``n eps ||block||_2``, as a root."""
    from superglm.solvers.rank import _eigensolver_relative_bar

    eigenvalues, eigenvectors = np.linalg.eigh(block)
    bar = _outward_cut(_eigensolver_relative_bar(len(block)), np.max(np.abs(eigenvalues)))
    kept = eigenvalues > bar
    return np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T


def _outward_cut(relative: float, norm: float) -> float:
    """``relative * norm`` rounded up: an upper bound on the exact product.

    ``relative`` (``n eps``) is exact, so the product is one rounding, and
    one ``nextafter`` toward ``+inf`` covers it (Codex review of #440): an
    eigenvalue at the bar is never kept on the product's downward rounding.
    """
    return float(np.nextafter(relative * float(norm), np.inf))


def _equilibrated_block_root(block: NDArray) -> NDArray | None:
    """``penalty_factor``'s root of a coupled block, cut on its Jacobi equilibration.

    The cut is the eigensolver's resolution ``n_b eps ||A_b||_2``, accumulated
    outward (``_outward_cut``).  The block's formation error is controlled
    where the block is formed, not in the cut: every penalty this package
    forms is a sum of ``lambda_t G_t`` with ``G_t`` the Gram of a root
    (``ssp_penalty_matrix``, ``_enclosed_root_gram``), an eigen
    reconstruction ``V Lambda V'`` (``_canonicalize_ssp_penalty``) or an exact
    integer or Kronecker form, so an exact null of the construction is an
    exact null of the rank-``r`` Gram and only rounding, entrywise within
    ``gamma_m sqrt(S_ii S_jj)`` (Higham 2002, eq. 3.5 and Cauchy-Schwarz),
    lifts it.  The worst case of that rounding, ``n_b gamma_m`` in ``||dA||_2``
    (coherent rounding in every entry), is not what the arithmetic delivers:
    across 7,000 random and structured Gram-form blocks the largest rounding
    eigenvalue was 0.15 of the eigensolver's cut, while a cut at the worst
    case dropped real curvature the eigensolver resolves (Opus review of
    #440: a coupled second-difference block at ``1e10`` beside ``1e-2`` kept
    58 of 60 rows, a 529-column REML tensor 507 of 528, ``D_3'D_3`` at
    ``k = 300`` 296 of 297), and so did ``sqrt(n_b) gamma_m`` (137 of 138
    on a Kronecker sum at ``1e11``).

    ``None`` where the block is not PSD at that resolution: a non-positive
    diagonal, or an eigenvalue of ``D S_b D`` below ``-max(100 eps ||D S_b
    D||_2, cut)``, the bar ``rank.decompose_gram`` refuses as materially
    indefinite.
    """
    from superglm.solvers.rank import _EPS, _eigensolver_relative_bar

    diagonal = np.diag(block)
    if not np.all(diagonal > 0.0):
        return None
    scale = np.sqrt(diagonal)
    # |S_ij| / s_i <= s_j on a PSD block, so only an indefinite one overflows
    with np.errstate(over="ignore"):
        equilibrated = (block / scale[:, None]) / scale[None, :]
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    if not np.all(np.isfinite(equilibrated)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    norm = float(np.max(np.abs(eigenvalues)))
    cut = _outward_cut(_eigensolver_relative_bar(len(block)), norm)
    if eigenvalues[0] < -max(100.0 * _EPS * norm, cut):
        return None
    kept = eigenvalues > cut
    return np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T * scale[None, :]


def grouped_augmented_factor(
    dm: DesignMatrix,
    W: NDArray,
    penalty: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> NDArray:
    """Return the bounded weighted-design factor augmented by ``sqrt(S)``."""
    data_factor = grouped_weighted_factor(dm, W, center=center, center_lo=center_lo)
    smooth_factor = penalty_factor(penalty)
    return data_factor if smooth_factor.shape[0] == 0 else np.vstack((data_factor, smooth_factor))


def grouped_augmented_factor_rhs(
    dm: DesignMatrix,
    W: NDArray,
    penalty: NDArray,
    *,
    response: NDArray,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return one compact QR of the weighted data, penalty, and response."""
    data_factor, transformed_rhs = grouped_weighted_factor_rhs(
        dm,
        W,
        response,
        center=center,
        center_lo=center_lo,
    )
    smooth_factor = penalty_factor(penalty)
    if smooth_factor.shape[0] == 0:
        return data_factor, transformed_rhs
    joint = np.column_stack((data_factor, transformed_rhs))
    smooth_joint = np.column_stack((smooth_factor, np.zeros(smooth_factor.shape[0])))
    joint_factor = np.linalg.qr(np.vstack((joint, smooth_joint)), mode="r")
    return np.asarray(joint_factor[:, :-1]), np.asarray(joint_factor[:, -1])


def refresh_centered_rhs(
    *,
    system: CenteredSystem,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
) -> CenteredSystem:
    """Reuse an invariant centered Gram while refreshing its working RHS."""
    sum_w, mean_x, mean_z, data_gram, rhs, mean_hi, mean_lo = _refresh_centered_data_rhs(
        dm=dm,
        W=W,
        z_off=z_off,
        data=(system.sum_w, system.mean_x, system.data_gram, system.mean_hi, system.mean_lo),
    )
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=mean_x,
        mean_z=mean_z,
        data_gram=data_gram,
        rhs=rhs,
        penalty=system.penalty,
        hessian=system.hessian,
        mean_hi=mean_hi,
        mean_lo=mean_lo,
    )


def _refresh_centered_data_rhs(*, dm, W, z_off, data):
    """``(X - 1 m')' W (z - mean_z)`` for a cached Gram, by column type.

    Every column takes its transpose product less ``mean_x`` times the sum of
    ``W (z - mean_z)``, which is zero to its rounding; a ``DenseGroupMatrix``
    column, whose entries its type does not bound, is centred row by row about
    the system's exact centre pair (``centre_pair``) instead, so a column's
    offset never multiplies that rounding (issue #430).
    """
    sum_w, mean_x, data_gram, mean_hi, mean_lo = data
    mean_z = float(np.dot(W, z_off) / sum_w)
    z_centered = z_off - mean_z
    weighted_z = W * z_centered
    rhs = dm.rmatvec(weighted_z) - mean_x * float(np.sum(weighted_z))
    dense = dense_columns(dm)
    if np.any(dense):
        centre = mean_x if mean_hi is None else mean_hi
        centred = dense_centred_rmatvec(dm, weighted_z, centre, mean_lo)
        rhs = np.where(dense, centred, rhs)
    return sum_w, mean_x, mean_z, data_gram, _freeze(rhs), mean_hi, mean_lo


def build_centered_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
    penalty: NDArray,
    tabmat_split=None,
    tabmat_state: TabmatCenteringState | None = None,
    profile: dict | None = None,
    _force_chunked: bool = False,
    _data: tuple | None = None,
    centre: NDArray | None = None,
) -> CenteredSystem:
    """Build a stably centered data Gram, RHS, and penalized Hessian.

    ``centre`` is the fit's fixed state centre (``mode_score.prior_weighted_centre``):
    a dense column's weighted mean is then read as the pair ``(centre, d)``,
    ``d`` the shift of its working-weighted mean from the centre, and the
    dense block is centred by a rank-one correction of its rows centred once
    about the centre (``dense_mean_pair``, ``_attach_dense_split``).
    """
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    z_off = np.asarray(z_off, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if W.shape != (n,) or z_off.shape != (n,):
        raise ValueError("W and z_off must match the design row count")
    if penalty.shape != (p, p):
        raise ValueError("penalty must have shape (p, p)")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("working weights must be finite and non-negative")

    sum_w = float(np.sum(W, dtype=np.float64))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("working weights must have a positive finite sum")
    if _data is not None:
        *refreshed, mean_hi, mean_lo = _refresh_centered_data_rhs(
            dm=dm, W=W, z_off=z_off, data=_data
        )
        return _attach_centered_penalty(*refreshed, penalty, mean_hi=mean_hi, mean_lo=mean_lo)
    mean_z = float(np.dot(W, z_off) / sum_w)
    z_centered = z_off - mean_z
    # A ``DenseGroupMatrix`` column, the one type whose entries its type does
    # not bound, never enters a raw-moment rung: each subtracts ``sum W x x'``
    # and ``(X'W)(X'W)' / sum W`` under a value certificate
    # (``_raw_centering_well_scaled``), so the arithmetic changed with the
    # column's location (offset 0: the raw-moment rung; offset 10: the exact
    # pair).  The rest of the design keeps the rungs: they run on its bounded
    # columns, and the dense columns join them centred about their exact pair
    # (``_attach_dense_split``).  A design with no bounded column, or whose
    # bounded part every rung rejects, takes the exact pair throughout.
    if not np.any(dense_columns(dm)):
        packed = _raw_rung_system(
            dm=dm,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            tabmat_split=tabmat_split,
            tabmat_state=tabmat_state,
            profile=profile,
            force_chunked=_force_chunked,
        )
        if packed is not None:
            mean_x, data_gram, rhs = packed
            return _attach_centered_penalty(sum_w, mean_x, mean_z, data_gram, rhs, penalty)
    else:
        split = _dense_split(dm)
        if split.bounded.p:
            packed = _raw_rung_system(
                dm=split.bounded,
                W=W,
                z_centered=z_centered,
                sum_w=sum_w,
                tabmat_split=(
                    None if tabmat_split is None else split.bounded.tabmat_centering_split
                ),
                tabmat_state=tabmat_state,
                profile=profile,
                force_chunked=_force_chunked,
            )
            if packed is not None:
                return _attach_dense_split(
                    split,
                    W=W,
                    z_centered=z_centered,
                    sum_w=sum_w,
                    mean_z=mean_z,
                    packed=packed,
                    penalty=penalty,
                    centre=centre,
                )

    mean_x, mean_hi, mean_lo = weighted_mean_pair(dm, W, sum_w, anchor=centre)
    data_gram, rhs = centered_gram_rhs(
        dm=dm,
        W=W,
        mean_x=mean_hi,
        z_centered=z_centered,
        mean_lo=mean_lo,
    )
    return _attach_centered_penalty(
        sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_hi, mean_lo=mean_lo
    )


def _raw_rung_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    tabmat_split,
    tabmat_state: TabmatCenteringState | None,
    profile: dict | None,
    force_chunked: bool,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """``(mean_x, data_gram, rhs)`` from the first raw rung that accepts ``dm``, else ``None``.

    Called only with a design free of ``DenseGroupMatrix`` columns.
    """
    packed = packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)
    if packed is None and (tabmat_state is None or tabmat_state.eligible is not False):
        mixed_attempted, mixed = _try_mixed_discrete_centering(
            dm=dm,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            preflight=tabmat_state is None or tabmat_state.eligible is None,
        )
        if mixed_attempted:
            packed = mixed
            if tabmat_state is not None:
                # Only the first call pays for Tabmat's location/scale
                # preflight. Every changed weight vector still receives the
                # authoritative full-moment certificate, and a rejection
                # permanently selects stable chunks for this inner fit.
                tabmat_state.eligible = mixed is not None
    # Raw-spline CSC construction is worthwhile only for a caller that owns a
    # reusable fit-local policy state (direct PIRLS and REML do). One-shot
    # inference/finalization calls retain the bounded stable-chunk path.
    if (
        packed is None
        and tabmat_state is not None
        and tabmat_state.raw_spline_eligible is not False
    ):
        raw_spline_plan = dm.get_raw_spline_tabmat_centering_plan(profile=profile)
        if raw_spline_plan is not None:
            packed = _try_raw_spline_tabmat_centering(
                plan=raw_spline_plan,
                W=W,
                z_centered=z_centered,
                sum_w=sum_w,
                preflight=tabmat_state.raw_spline_eligible is None,
                profile=profile,
            )
            tabmat_state.raw_spline_eligible = packed is not None
            if packed is None and profile is not None:
                profile["centered_spline_tabmat_stable_fallbacks"] = (
                    profile.get("centered_spline_tabmat_stable_fallbacks", 0) + 1
                )
    if (
        packed is None
        and tabmat_split is not None
        and (tabmat_state is None or tabmat_state.eligible is not False)
    ):
        packed = _try_tabmat_centering(
            tabmat_split=tabmat_split,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            preflight=tabmat_state is None or tabmat_state.eligible is None,
        )
        if tabmat_state is not None:
            # A rejection is permanent for this fit.  Later IRLS weights can
            # change the centering ratio, but the stable path remains correct
            # and avoids repeating rejected raw work.
            tabmat_state.eligible = packed is not None
    # General raw-moment rung: the rungs above each require particular group
    # types, so ordinary spline designs reach the chunked pass below even though
    # the per-block moment dispatch can produce the same quantities far more
    # cheaply.  A rejected certificate latches for the fit, as the other
    # accelerated rungs do, so the moments are not recomputed every iteration.
    if (
        packed is None
        and not force_chunked
        and (
            tabmat_state is None
            # `eligible is False` means a preflight already certified this
            # design's raw moments as unsafe.  That verdict is about raw-moment
            # subtraction itself, not about tabmat, so this rung must honour it
            # rather than re-attempting the route that was just locked out.
            or (
                tabmat_state.eligible is not False and tabmat_state.raw_moment_eligible is not False
            )
        )
    ):
        packed = try_raw_moment_centering(
            dm=dm,
            W=W,
            weighted_z=W * z_centered,
            sum_w=sum_w,
        )
        if tabmat_state is not None:
            tabmat_state.raw_moment_eligible = packed is not None
        if packed is not None and profile is not None:
            profile["centered_raw_moment_hits"] = profile.get("centered_raw_moment_hits", 0) + 1
    return packed


@dataclass(frozen=True)
class _DenseSplit:
    """A design's ``DenseGroupMatrix`` groups and its other (bounded) groups, as two designs."""

    dense: DesignMatrix
    bounded: DesignMatrix
    dense_index: NDArray
    bounded_index: NDArray


# One split per design, built on first use.  The bounded design owns its own
# rung caches (execution plan, Tabmat split, bin-space plan, raw-spline Tabmat
# plan), so they persist across a fit's iterations.  Owner: the fit workspace
# that holds the design.  Lifetime: until the fit is published, when
# ``release_dense_split`` drops it with the full design's raw-spline plan
# (``capture_fit_state``), or until the design itself is collected.  The split
# depends only on the design's groups, never on weights, parameters, penalty or
# precision, so nothing else invalidates it; a later fit rebuilds it on use.
_DENSE_SPLITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def release_dense_split(dm: DesignMatrix) -> None:
    """Drop ``dm``'s dense/bounded split and with it every cache of its bounded design."""
    _DENSE_SPLITS.pop(dm, None)


def _dense_split(dm: DesignMatrix) -> _DenseSplit:
    split = _DENSE_SPLITS.get(dm)
    if split is not None:
        return split
    dense_groups, bounded_groups = [], []
    dense_index: list[int] = []
    bounded_index: list[int] = []
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = range(offset, offset + width)
        if type(matrix) is DenseGroupMatrix:
            dense_groups.append(matrix)
            dense_index.extend(columns)
        else:
            bounded_groups.append(matrix)
            bounded_index.extend(columns)
        offset += width
    split = _DenseSplit(
        dense=DesignMatrix(dense_groups, n=dm.n, p=len(dense_index)),
        bounded=DesignMatrix(bounded_groups, n=dm.n, p=len(bounded_index)),
        dense_index=np.asarray(dense_index, dtype=np.intp),
        bounded_index=np.asarray(bounded_index, dtype=np.intp),
    )
    _DENSE_SPLITS[dm] = split
    return split


def _attach_dense_split(
    split: _DenseSplit,
    *,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    mean_z: float,
    packed: tuple[NDArray, NDArray, NDArray],
    penalty: NDArray,
    centre: NDArray | None = None,
) -> CenteredSystem:
    """The centred system from a raw rung on the bounded columns and the dense ones centred apart.

    With the fit's ``centre`` the dense rows ``x~ = fl(x - c)``, centred about
    that fixed centre in a single subtraction, give ``t = sum W x~``, ``G =
    sum W x~ x~'`` and ``r = sum W z x~`` in one pass
    (``anchored_dense_moments``).  The working-weighted mean is the pair ``(c,
    d)``, ``d = t / sum W``, and the block is centred by the rank-one
    correction ``G - t d'`` (``r - d sum W z``): the textbook formula on data
    shifted by ``c`` (Chan, Golub & LeVeque 1983, §3), within its rounding
    envelope while ``d`` lies within one weighted standard deviation of the
    rows (``_anchored_remainder``).  The cross block is ``N~' W D~ = N' (W
    x~) - m_N t'``, one transpose product of the bounded design per dense
    column, ``d``'s share ``d (N'W - m_N sum W)`` vanishing with ``m_N``'s
    definition.

    Without a centre, or past that certificate, the dense block ``D~' W D~``
    and its right-hand side come from rows ``(x - hi) - lo`` about the exact
    pair (``dense_mean_pair``, ``centered_gram_rhs``), and the cross block is
    ``N' (W D~) - m_N (1' W D~)``, with ``1' W D~`` zero to rounding.  Either
    way the bounded columns' raw values never meet a dense column's offset,
    and the bounded block is the rung's own.
    """
    mean_bounded, gram_bounded, rhs_bounded = packed
    dense_width = split.dense.p
    cross = np.empty((split.bounded.p, dense_width), dtype=np.float64)
    anchored = None
    if centre is not None:
        anchor = np.asarray(centre, dtype=np.float64)[split.dense_index]
        first, gram, response = anchored_dense_moments(split.dense, W, anchor, z_centered)
        remainder = _anchored_remainder(first, np.diag(gram), sum_w)
        if remainder is not None and response is not None:
            anchored = anchor, remainder, first, gram, response
    if anchored is not None:
        hi, lo, first, gram, response = anchored
        gram_dense = rank_one_centred_gram(first, gram, lo)
        rhs_dense = response - lo * float(np.dot(W, z_centered))
        # one rows-long buffer serves every dense column: W (x_k - c_k)
        weighted = np.empty(split.dense.n, dtype=np.float64)
        for column, (values, centre) in enumerate(_dense_columns_of(split.dense, hi)):
            np.subtract(values, centre, out=weighted)
            weighted *= W
            cross[:, column] = split.bounded.rmatvec(weighted) - mean_bounded * float(
                np.sum(weighted)
            )
    else:
        pair = dense_mean_pair(split.dense, W, sum_w)
        if pair is None:  # pragma: no cover - the split holds a dense column
            raise RuntimeError("A dense split formed no centre pair.")
        hi, lo = pair
        gram_dense, rhs_dense = centered_gram_rhs(
            dm=split.dense, W=W, mean_x=hi, z_centered=z_centered, mean_lo=lo
        )
        unit = np.zeros(dense_width, dtype=np.float64)
        for column in range(dense_width):
            unit[column] = 1.0
            weighted = W * dense_centred_matvec(split.dense, unit, hi, lo)
            unit[column] = 0.0
            cross[:, column] = split.bounded.rmatvec(weighted) - mean_bounded * float(
                np.sum(weighted)
            )
    p = dense_width + split.bounded.p
    dense_index, bounded_index = split.dense_index, split.bounded_index
    data_gram = np.empty((p, p), dtype=np.float64)
    data_gram[np.ix_(bounded_index, bounded_index)] = gram_bounded
    data_gram[np.ix_(dense_index, dense_index)] = gram_dense
    data_gram[np.ix_(bounded_index, dense_index)] = cross
    data_gram[np.ix_(dense_index, bounded_index)] = cross.T
    rhs = np.empty(p, dtype=np.float64)
    rhs[bounded_index] = rhs_bounded
    rhs[dense_index] = rhs_dense
    mean_x = np.empty(p, dtype=np.float64)
    mean_x[bounded_index] = mean_bounded
    mean_x[dense_index] = hi + lo
    mean_hi = mean_x.copy()
    mean_hi[dense_index] = hi
    mean_lo = np.zeros(p, dtype=np.float64)
    mean_lo[dense_index] = lo
    return _attach_centered_penalty(
        sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_hi, mean_lo=mean_lo
    )


def _dense_sources(dm: DesignMatrix, anchor: NDArray) -> list[tuple[NDArray, NDArray]]:
    """Per dense block ``(values, centre)``, the centre read from ``anchor``."""
    sources = []
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            centre = np.asarray(anchor[offset : offset + width], dtype=np.float64)
            sources.append((matrix.M, centre))
        offset += width
    return sources


def _dense_columns_of(dm: DesignMatrix, anchor: NDArray) -> Iterator[tuple[NDArray, float]]:
    """``(x_k, c_k)`` for each dense column of ``dm`` in order: its values and its centre."""
    for values, centre in _dense_sources(dm, anchor):
        for index in range(values.shape[1]):
            yield values[:, index], float(centre[index])


def anchored_dense_moments(
    dm: DesignMatrix, W: NDArray, anchor: NDArray, response: NDArray | None = None
) -> tuple[NDArray, NDArray, NDArray | None]:
    """``(t, G, r)`` over the dense columns of ``dm``, rows ``x~ = fl(x - anchor)``, in one pass.

    ``t = sum W x~``, ``G = sum W x~ x~'`` and ``r = sum W z x~`` (``z`` the
    ``response``; ``r`` is ``None`` without one), each accumulated across
    row chunks with Kahan's compensation, in ``dense_columns`` order.  Each
    row is centred about the fixed anchor by one subtraction, never about a
    weighted mean of the pass.
    """
    W = np.asarray(W, dtype=np.float64)
    sources = _dense_sources(dm, np.asarray(anchor, dtype=np.float64))
    width = sum(values.shape[1] for values, _ in sources)
    first = np.zeros(width, dtype=np.float64)
    first_compensation = np.zeros_like(first)
    gram = np.zeros((width, width), dtype=np.float64)
    gram_compensation = np.zeros_like(gram)
    rhs = None if response is None else np.zeros(width, dtype=np.float64)
    rhs_compensation = np.zeros(width, dtype=np.float64)
    for start in range(0, dm.n, _FACTOR_CHUNK_ROWS):
        stop = min(start + _FACTOR_CHUNK_ROWS, dm.n)
        blocks = [values[start:stop] - centre for values, centre in sources]
        rows = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
        weights = W[start:stop]
        weighted = rows * weights[:, None]
        _compensated_add(first, first_compensation, rows.T @ weights)
        _compensated_add(gram, gram_compensation, rows.T @ weighted)
        if rhs is not None and response is not None:
            _compensated_add(rhs, rhs_compensation, weighted.T @ response[start:stop])
    return first, 0.5 * (gram + gram.T), rhs


def rank_one_centred_gram(first: NDArray, gram: NDArray, remainder: NDArray) -> NDArray:
    """``sum W (x~ - d)(x~ - d)'`` as ``G - t d'``, symmetrized, from the shifted moments.

    ``t = sum W x~`` and ``G = sum W x~ x~'`` (``anchored_dense_moments``), ``d =
    t / sum W`` (``_anchored_remainder``): the rank-one correction of the
    textbook formula on data shifted by the anchor, as ``_certify_raw_centering``
    applies it to raw moments.
    """
    centred = gram - np.outer(first, remainder)
    return 0.5 * (centred + centred.T)


def _anchored_remainder(first: NDArray, second: NDArray, sum_w: float) -> NDArray | None:
    """``d = t / sum W``, the weighted mean's shift from the anchor, or ``None`` past the certificate.

    ``second`` is ``sum W x~^2``.  The rank-one correction ``sum W x~^2 -
    sum W d^2`` (the textbook formula on shifted data) carries ``n u kappa^2``
    relative, ``kappa^2 = sum W x~^2 / S = 1 + sum W d^2 / S`` with ``S =
    sum W (x~ - d)^2`` (Chan, Golub & LeVeque 1983, eq. 3.2 and Table 1;
    Chan & Lewis 1979 for the condition number).  It is taken only where
    ``d`` lies within one weighted standard deviation, ``sum W d^2 <= S``,
    i.e. ``kappa^2 <= 2`` (their eq. 3.3 with ``p = 1``), the envelope
    ``_raw_centering_well_scaled`` sets for every raw-moment rung.  Weights
    must be non-negative.
    """
    if not (np.isfinite(sum_w) and sum_w > 0.0):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        remainder = first / sum_w
        shifted = 2.0 * sum_w * remainder * remainder
        within = np.isfinite(remainder) & np.isfinite(second) & (shifted <= second)
    return remainder if bool(np.all(within)) else None


def _dense_anchored_pair(
    dm: DesignMatrix, W: NDArray, sum_w: float, anchor: NDArray
) -> tuple[NDArray, NDArray] | None:
    """``(anchor, d)`` on the dense columns from one pass, or ``None`` past ``_anchored_remainder``."""
    first_parts: list[NDArray] = []
    second_parts: list[NDArray] = []
    for values, centre in _dense_sources(dm, anchor):
        width = values.shape[1]
        first = np.zeros(width, dtype=np.float64)
        first_compensation = np.zeros_like(first)
        second = np.zeros(width, dtype=np.float64)
        second_compensation = np.zeros_like(second)
        for start in range(0, dm.n, _FACTOR_CHUNK_ROWS):
            stop = min(start + _FACTOR_CHUNK_ROWS, dm.n)
            rows = values[start:stop] - centre
            weights = W[start:stop]
            _compensated_add(first, first_compensation, rows.T @ weights)
            _compensated_add(second, second_compensation, (rows * rows).T @ weights)
        first_parts.append(first)
        second_parts.append(second)
    remainder = _anchored_remainder(
        np.concatenate(first_parts), np.concatenate(second_parts), sum_w
    )
    if remainder is None:
        return None
    dense = dense_columns(dm)
    hi = np.zeros(dm.p, dtype=np.float64)
    lo = np.zeros(dm.p, dtype=np.float64)
    hi[dense] = anchor[dense]
    lo[dense] = remainder
    return hi, lo


def dense_mean_pair(
    dm: DesignMatrix, W: NDArray, sum_w: float, anchor: NDArray | None = None
) -> tuple[NDArray, NDArray] | None:
    """``(hi, lo)`` over all ``p`` columns, zero off the ``DenseGroupMatrix`` ones; ``None`` without one.

    Each dense column's weighted mean ``sum W x / sum W`` as a pair whose sum
    it is, read without cancellation.  With the fit's fixed centre ``anchor``
    (non-negative ``W``), ``hi`` is that centre and ``lo`` the shift of the
    weighted mean from it, ``sum W x~ / sum W`` on rows centred about it
    (one pass), while the shift stays within one weighted standard deviation
    (``_anchored_remainder``).  Otherwise, by the corrected two-pass algorithm
    (``mode_score.corrected_two_pass_pair``): ``hi`` the rounded mean, ``lo``
    the remainder formed on rows differenced from it.  ``W`` may be signed
    (observed geometry); ``sum_w`` is the caller's own.
    """
    if not any(type(matrix) is DenseGroupMatrix for matrix in dm.group_matrices):
        return None
    W = np.asarray(W, dtype=np.float64)
    if anchor is not None and not np.any(W < 0.0):
        pair = _dense_anchored_pair(dm, W, sum_w, np.asarray(anchor, dtype=np.float64))
        if pair is not None:
            return pair
    hi = np.zeros(dm.p, dtype=np.float64)
    lo = np.zeros(dm.p, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            values = matrix.M

            def chunks(values=values):
                for start in range(0, dm.n, _FACTOR_CHUNK_ROWS):
                    stop = min(start + _FACTOR_CHUNK_ROWS, dm.n)
                    yield start, stop, values[start:stop]

            hi[offset : offset + width], lo[offset : offset + width] = corrected_two_pass_pair(
                chunks, W, sum_w, width
            )
        offset += width
    return hi, lo


def weighted_mean_pair(
    dm: DesignMatrix, W: NDArray, sum_w: float, anchor: NDArray | None = None
) -> tuple[NDArray, NDArray, NDArray | None]:
    """``(mean_x, hi, lo)``: the weighted column mean and the exact pair it rounds.

    One-engine design §3.2 applied to gram's centred system: a
    ``DenseGroupMatrix`` column's mean is ``hi + lo`` (``dense_mean_pair``),
    ``hi`` the rounded mean from a pass shifted by the first row that
    carries weight and ``lo`` the remainder formed on rows differenced from
    ``hi`` (Chan, Golub & LeVeque 1983).  A column constant on its weighted
    rows centres to exact zeros (the unshifted ``sum W x / sum W`` left a
    rounding-level centred diagonal whose rank decision moved the REML
    objective by ~30 between iterates: stage-1 verifier, the Gamma/log
    constant-column fit), and a far row of negligible weight no longer sets
    the remainder's scale, as it did as the anchor.  Rows are centred as
    ``(x - hi) - lo``, which rounds at the column's spread; ``x - mean_x``
    with ``mean_x = fl(hi + lo)`` rounds at ``u |mean|`` and adds ``sum W d
    d'`` (``d`` that rounding) to the profiled Gram, which at a 1e16 offset
    moved a Gaussian fit's eta by 1e-3 (issue #430).  Every other column,
    bounded by its type, has ``hi = mean_x = X'W / sum W`` and ``lo = 0``;
    ``lo`` is ``None`` for a design without a dense column.  With the fit's
    centre ``anchor`` the dense pair is ``(anchor, d)`` while the certificate
    of ``dense_mean_pair`` holds.
    """
    mean = dm.rmatvec(W) / sum_w
    pair = dense_mean_pair(dm, W, sum_w, anchor=anchor)
    if pair is None:
        return mean, mean.copy(), None
    dense = dense_columns(dm)
    hi = np.where(dense, pair[0], mean)
    lo = np.where(dense, pair[1], 0.0)
    return np.where(dense, pair[0] + pair[1], mean), hi, lo


def _attach_centered_penalty(
    sum_w, mean_x, mean_z, data_gram, rhs, penalty, *, mean_hi=None, mean_lo=None
):
    """Attach a fresh trial penalty using the same numerical checks on every path."""
    penalty_symmetric = 0.5 * (penalty + penalty.T)
    hessian = data_gram + penalty_symmetric
    # Both terms are mathematically PSD. Degenerate spline
    # reparameterizations can introduce visible negative round-off, so project
    # only this declared-PSD system back onto its valid cone before rank work.
    # Keep structurally empty coordinates outside the eigensolve: reconstructing
    # the full matrix can scatter round-off into exact zero rows, manufacture
    # numerical rank, and obscure an empty-support spline direction.
    active = np.flatnonzero(np.any(hessian != 0.0, axis=0) | np.any(hessian != 0.0, axis=1))
    active_hessian = hessian[np.ix_(active, active)]
    try:
        np.linalg.cholesky(active_hessian)
    except np.linalg.LinAlgError:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(active_hessian)
        if hessian_eigenvalues.size and hessian_eigenvalues[0] < 0.0:
            active_hessian = (
                hessian_eigenvectors * np.maximum(hessian_eigenvalues, 0.0)[None, :]
            ) @ hessian_eigenvectors.T
            active_hessian = 0.5 * (active_hessian + active_hessian.T)
            hessian = np.zeros_like(hessian)
            hessian[np.ix_(active, active)] = active_hessian
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=_freeze(mean_x),
        mean_z=mean_z,
        data_gram=_freeze(data_gram),
        rhs=_freeze(rhs),
        penalty=_freeze(penalty_symmetric),
        hessian=_freeze(hessian),
        mean_hi=None if mean_hi is None else _freeze(mean_hi),
        mean_lo=None if mean_lo is None else _freeze(mean_lo),
    )


def build_anchor_centered_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
    penalty: NDArray,
) -> CenteredSystem:
    """Build an anchored system for predictors with locations beyond their scale."""
    W = np.asarray(W, dtype=np.float64)
    z_off = np.asarray(z_off, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    if W.shape != (dm.n,) or z_off.shape != (dm.n,):
        raise ValueError("W and z_off must match the design row count")
    if penalty.shape != (dm.p, dm.p):
        raise ValueError("penalty must match the design column count")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("working weights must be finite and non-negative")
    sum_w = float(np.sum(W, dtype=np.float64))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("working weights must have a positive finite sum")
    mean_z = float(np.dot(W, z_off) / sum_w)
    mean_x, data_gram, rhs = stable_centered_gram_rhs(
        dm=dm,
        W=W,
        z_centered=z_off - mean_z,
        sum_w=sum_w,
    )
    penalty_symmetric = 0.5 * (penalty + penalty.T)
    hessian = 0.5 * (data_gram + data_gram.T) + penalty_symmetric
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        if eigenvalues.size and eigenvalues[0] < 0.0:
            hessian = (eigenvectors * np.maximum(eigenvalues, 0.0)[None, :]) @ eigenvectors.T
            hessian = 0.5 * (hessian + hessian.T)
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=_freeze(mean_x),
        mean_z=mean_z,
        data_gram=_freeze(data_gram),
        rhs=_freeze(rhs),
        penalty=_freeze(penalty_symmetric),
        hessian=_freeze(hessian),
    )
