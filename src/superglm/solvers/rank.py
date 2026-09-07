"""Versioned numerical-rank policy and retained-subspace operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


@dataclass(frozen=True)
class RankPolicy:
    """Numerical-rank thresholds, and the version of the rule that applied them.

    ``version`` does not track the threshold VALUES.  Those are constants of
    the floating-point format, pinned in ``test_rank_policy.py``, and they have
    not moved.  It tracks the RULE, because identical thresholds do not imply
    an identical answer: the rank cutoff fixes how many directions are
    retained, and leaves open which columns represent them and whether a
    representative basis is recovered at all.

    That is the one thing a stored ``RankDecomposition`` or ``RankInfo`` cannot
    re-derive for itself, so it is the one thing the version has to carry.

    **Bump contract.**  Bump when the deficient path can return a different
    ``active_columns`` for the same input under the same thresholds.  A newly
    recorded field, a new report, or a change outside this module is NOT a
    bump: the version is a claim about the selection rule, and it can only
    honestly cover what this module decides.  Note that several
    selection-deciding constants live outside this object as module literals,
    so a bump is a claim about the rule as a whole, not about the fields here.
    **Version 3 is exactly that case, so read "have not moved" narrowly.**
    ``gram_rcond`` is still ``eps`` and still pins the normal-equation
    boundary against ``factor_rcond``, but the cutoff the Gram route APPLIES
    is now that field floored at :func:`_eigensolver_relative_bar` -- a module
    literal, not a field -- so the effective threshold moved by a factor of
    the matrix order while every value in this object stayed put.  A reader
    who takes the field for the cut will be wrong by that factor.

    The version reaches a recorded fit through
    ``training_telemetry()["rank_policy"]["version"]``.
    """

    version: int
    factor_rcond: float
    gram_rcond: float
    certification_band: float
    warning_condition: float
    severe_condition: float


_EPS = np.finfo(float).eps
SHARED_RANK_POLICY = RankPolicy(
    # Version 2 -- the deficient path answers differently under version 1's
    # thresholds, in two independent ways.
    #
    # Choosing representatives by walking every prefix was replaced by reading
    # the choice off the null basis.  Where the walk could fill its set the two
    # agree exactly -- over 958 deficient blocks there was NOT ONE where both
    # returned a set and the sets differed -- but the walk tests each prefix
    # against the WHOLE matrix's cutoff, so on a small prefix it can mis-reject
    # a column, run out of candidates and return nothing.  That happened on 126
    # of those 958.  Reading the selection from the null basis makes a
    # representative candidate available where the walk left `gram_eigh`, but
    # the candidate is retained only when its principal condition stays below
    # `severe_condition` AND its smallest eigenvalue numerically clears the
    # same full-matrix cutoff that certified the rank.  The walk fails at the
    # numerical boundary, so most of those candidates deliberately remain
    # spectral.  Safe candidates still recover a coefficient representation,
    # so this is not only a cost difference.
    #
    # Which blocks those are is NOT reproducible across machines: the walk only
    # fails when a prefix eigenvalue lands near `eps * ||A||`, which is below
    # what a symmetric eigensolver resolves.  The rate is stable, the identity
    # of the blocks is not -- see the test of the same name for the measurement
    # and for how that constrains what can be asserted.
    #
    # `_conditioned_representatives` then changed which columns are chosen when
    # index order costs more conditioning than a rank-revealing choice may.  On
    # the three-column near alias it selects [0, 2] where the walk selected
    # [0, 1].
    #
    # Skipping the eager Gram subspace at certification sites is inside this
    # version and contributes no reason for it: that path returns the same
    # decomposition or none, and was accepted on byte-identical fitted output.
    #
    # Version 3 -- the Gram cutoff is floored at the eigensolver's own error
    # bar, so the rank it reports stops being a function of round-off's SIGN.
    # `_eigensolver_relative_bar` carries the derivation and the measurement;
    # what belongs HERE is why it is a version and not a patch.  It moves the
    # cutoff from `eps * max|w|` to `n * eps * max|w|`, which is a factor of
    # `n`, so a direction whose eigenvalue lands in that band is now dropped
    # where it was retained.  That changes `rank`, and hence `active_columns`,
    # for the same input under the same `gram_rcond` -- the bump contract
    # above, exactly.  `factor_rcond` and `gram_rcond` are untouched: the
    # normal-equation boundary they pin is still `lambda = sigma^2`, and the
    # floor binds only where that boundary asks for a decision the arithmetic
    # cannot supply.  See issue #356.
    version=3,
    factor_rcond=float(np.sqrt(_EPS)),
    gram_rcond=float(_EPS),
    certification_band=32.0,
    warning_condition=float(1.0 / np.sqrt(_EPS)),
    severe_condition=float(1.0 / _EPS),
)


def _eigensolver_relative_bar(order: int) -> float:
    """``p(n) eps`` -- the relative accuracy a symmetric eigensolver delivers.

    The *LAPACK Users' Guide*, 3rd ed. (SIAM 1999), sec. 4.7, bounds the
    computed eigenvalues of a symmetric ``A`` by ``|w_i - w_i_exact| <= p(n)
    eps ||A||_2`` with ``p(n)`` "a modestly growing function of n", and states
    in the same section that "large eigenvalues ... are computed to high
    relative accuracy and small ones may not be".  The bound is *absolute* and
    scaled by ``||A||_2``, so nothing below it is a property of the matrix.
    ``p(n) = n`` is taken here; the guide's own code fragment displays the
    bound with ``p(n) = 1``, which is an illustration of the bound and not a
    rank cut.

    **Every established rank tolerance carries such a factor, and this
    module's Gram cut was the only one without it.**  ``numpy.linalg.matrix_rank``
    documents ``S.max() * max(M, N) * eps`` and attributes it to MATLAB's
    ``rank`` and to *Numerical Recipes*, 3rd ed. (Cambridge 2007), p. 795;
    the same edition's alternative is ``eps/2 * sqrt(m + n + 1) * S.max()``,
    described there as based on expected round-off (p. 71).  SuiteSparseQR's
    documented default is ``20 (m + n) eps * max_j ||A(:,j)||_2`` -- Foster &
    Davis, "Algorithm 933", *ACM TOMS* 40(1) Art. 7 (2013).  LAPACK's own
    ``?PSTRF`` -- Cholesky with complete pivoting, which is the closest
    published analogue of what this module does to a semidefinite Gram --
    documents its ``TOL`` argument as "If TOL < 0, then N*U*MAX( A(K,K) ) will
    be used", i.e. ``n u max_k A(k,k)``.  All four are ``p(dimension) * eps *
    scale`` with ``p`` growing at least linearly; ``p == 1`` is not among
    them.  ``screening/_structured.py``'s ``_penalty_root`` already uses
    ``n eps ||S||_2`` in this tree, and cites the same section for it.

    **Why the Gram route needs this floor and the factor route does not.**
    The two rcond fields pin ONE statistical boundary in two coordinate
    systems -- ``factor_rcond = sqrt(eps)`` on singular values is
    ``gram_rcond = eps`` on eigenvalues, because ``lambda = sigma^2``, and
    ``test_shared_rank_policy_matches_normal_equation_boundary`` holds them
    there.  Squaring maps the *boundary* exactly and does *not* map the *noise
    floor*, because the floor is ``p(n) eps`` times the largest value in
    whichever coordinates you are in:

        factor:  cut sqrt(eps) sigma_max,  floor p(n) eps sigma_max
                 -> the cut sits 1/(p(n) sqrt(eps)) ~ 6.7e+07/p(n) ABOVE it
        gram:    cut eps lambda_max,       floor p(n) eps lambda_max
                 -> the cut sits at 1/p(n) of it, i.e. BENEATH it

    So the same policy that is resolved with seven orders to spare on the
    factor is beneath resolution on the Gram, and the gap is exactly ``p(n)``.
    That is not a tolerance that was chosen badly; it is the price of forming
    ``X'X``, and it is the same price -- half the digits -- that makes the
    normal equations a documented hazard for the least-squares problem itself
    -- ``kappa_2(A'A) = kappa_2(A)^2``, the standard argument for preferring QR
    over the normal equations, Golub & Van Loan, *Matrix Computations*, 4th
    ed. (JHU Press 2013), ch. 5, "Orthogonalization and Least Squares".

    **What the floor fixes is the sign, which is the part that is not data.**
    Beneath the floor, an eigenvalue's sign is round-off.  Goulart,
    Nakatsukasa & Rontsis, "Accuracy of approximate projection to the
    semidefinite cone", arXiv:1908.01606, say so in the published form -- that
    "approximations to the small eigenvalues may have the wrong signs", so the
    computed spectrum "may not contain the correct number of positive
    eigenvalues" -- and their main result is that this costs the PROJECTION
    almost nothing (their bound is gap-INDEPENDENT, Thm. 2.1).  It is the
    *count* that is undetermined, not the matrix.  This module publishes the
    count, and inverts on it, which is the one use their result does not
    cover: a retained direction enters ``pseudo_inverse`` as ``1 / w``.

    **The floor is also what makes the PSD clip legitimate rather than a
    decision.**  ``max(w, 0)`` is the projection onto the semidefinite cone --
    Higham, *Linear Algebra Appl.* 103:103-118 (1988) for the Frobenius case,
    Goulart et al. Lemma 2.1 for every unitarily invariant norm -- and it is
    the right thing to do to a RESOLVED negative eigenvalue.  Applied beneath
    the floor it was deciding the rank, because a negative round-off residue
    was clipped to exactly ``0.0`` and always dropped while a positive one of
    the same magnitude was compared against ``eps max|w|`` and could be kept.
    With the cut AT the floor the clip can no longer decide anything: inside
    the bar both signs drop, and outside it the clip only ever meets a
    genuinely negative eigenvalue, which is precisely Higham's case.  The
    asymmetry is removed by raising the cut, not by rewriting the comparison.

    **This reconciles the two modules that answered the question oppositely --
    and they still retain and drop oppositely.**  Issue #356 records that
    ``screening/_structured.py``'s ``_penalty_root`` *keeps* an in-bar
    eigenvalue at its magnitude where this module *dropped* it on the sign, with
    nothing reconciling the two.  The shared rule is not the outcome, it is
    the question: inside the bar, choose the answer that is a function of the
    data rather than of the rounding, on the same bar ``n eps ||A||_2``.  The
    outcomes differ because the objects do.  There, dropping a direction from
    a *penalty* makes it *free* -- filter factor one, the maximum -- so keeping it
    at ``|w|`` is the conservative reading.  Here, keeping a direction in a
    *Hessian* puts ``1/w`` into a covariance, so dropping it is.  Both are
    sign-independent, which is the property the issue is about; neither is a
    coin flip; and the module that is wrong in a given direction is now the
    one whose object says so, rather than whichever one the caller reached.

    **Measured, on the fixture the issue was filed from.**  Seven
    ``OPENBLAS_CORETYPE`` microkernels at one thread, on
    ``TestRankGateSeesCollinearityNotScale``'s correlation matrix and on the
    same matrix with the residue reflected through its own eigenvector
    (``H - 2 w0 v v'``, which is what a different BLAS produces) -- fourteen
    configurations.  Against the OLD cut the residue runs **0.10x to 2.93x**,
    straddling it: rank reads 6 on one configuration and 5 on thirteen, and
    ``||pseudo_inverse||_2`` reads **1.817e+15** on that one against 1.0 on
    the rest.  Against this bar the same residue runs **0.017x to 0.488x** and
    never approaches it, so the direction drops on all fourteen -- worst
    reading 0.488x, i.e. **2.05x of headroom**, min 0.017x.  One number moved
    from a 1.8e+15 spread to no spread at all.

    **The module already believed this bar, in another field, to the bit.**
    ``certification_band * gram_rcond`` is ``32 eps``, which is
    ``_eigensolver_relative_bar(32)`` EXACTLY -- not approximately, the same
    float -- and ``_certification_required`` compares a condition number to
    ``warning_condition / sqrt(certification_band) = 1/sqrt(32 eps) =
    1.186328e+07``, which is ``1/sqrt(bar)`` at order 32 to six figures.  So
    the certification band was always a resolution bar with ``p(n)`` frozen at
    32, applied to the condition number instead of to the cut.  Version 3 does
    not introduce the quantity; it makes the *cut* use it, and track the actual
    order rather than assume 32.  The two agree at width 32 by construction,
    the cut is finer below it and coarser above.

    **What it does not fix, stated because the opposite would be easy to
    assume.**  This does not bound the pseudo-inverse.  A retained direction
    may sit just above the bar, giving ``||pinv||_2 ~ 1/(n eps lambda_max)``,
    and the policy boundary ITSELF admits ``1/(eps lambda_max)`` on both
    routes by construction -- that is what retaining a ``sigma =
    sqrt(eps) sigma_max`` direction means.  A large covariance after this
    change is a conditioning fact about the data.  Before it, it was a fact
    about the machine.  Only the second is this module's to fix, and callers
    that need the first refused rather than reported are what
    ``needs_factor_certification`` is for.
    """
    return float(max(order, 1)) * _EPS


# Largest entry magnitude for which ``M + M.T`` provably cannot overflow: the
# sum is bounded entrywise by twice this.  See ``_symmetric_part``.
_HALF_MAX = float(np.finfo(float).max) / 2.0


def _symmetric_part(values: NDArray) -> NDArray:
    """``0.5 * (M + M.T)``, computed so a finite ``M`` cannot overflow to ``inf``.

    ``M + M.T`` is formed at full magnitude, so a finite ``M`` whose entries
    exceed half the float range overflows before the halving can bring it back.
    ``[[1e308]]`` sums to ``inf``; ``decompose_gram`` then either refuses the
    matrix outright or -- once the caller has pre-symmetrized -- equilibrates
    ``inf / inf`` to ``nan`` and returns a silently wrong answer.  Halving each
    operand first, ``0.5 * M + 0.5 * M.T``, cannot overflow at all: both terms
    are bounded by ``max / 2``, so their sum is bounded by ``max``.

    The two forms are **not** interchangeable, which is why this is a branch
    rather than a rewrite.  Halving is exact only while the halved value stays
    normal, so the split form rounds where the joint form does not.  Swept over
    1.05e6 exhaustive subnormal pairs, 1.05e6 pairs straddling the
    normal/subnormal boundary, 8e3 random subnormal pairs and 1e4 random normal
    pairs spanning the full exponent range, the two forms differ on 393726,
    393728, 2866 and **0** of those respectively.  The normal-range count is the
    load-bearing one -- the forms agree bitwise whenever both operands are
    normal and the sum does not overflow -- but the subnormal disagreement is
    real, and it costs the guarantee that an exactly symmetric ``M`` is
    reproduced bitwise: at ``M = [[3 * 5e-324]]`` the split form returns
    ``4 * 5e-324`` because ``0.5 * M`` rounds to even, while the joint form is
    exact.

    So the joint form is kept verbatim wherever it is provably safe --
    ``max|M| <= max / 2`` bounds ``|M + M.T|`` by ``max`` entrywise -- and the
    split form is taken only in the regime the joint form cannot represent.
    Every in-tree Gram matrix (``XtWX + S``, ``X'X + lambda*P``) is many orders
    below that bound, so this is bitwise inert for all of them.

    Non-finite input needs no special handling: ``max|M|`` is then ``inf`` or
    ``nan``, neither of which satisfies the bound, and the split form
    propagates the ``inf``/``nan`` exactly as the joint form does.
    """
    if float(np.abs(values).max(initial=0.0)) <= _HALF_MAX:
        return 0.5 * (values + values.T)
    return 0.5 * values + 0.5 * values.T


def diagonal_of_square(matrix: NDArray) -> NDArray:
    """Return ``diag(matrix @ matrix)`` with an O(p²) contraction."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return np.einsum("ij,ji->i", matrix, matrix, optimize=True)


def streamed_weighted_factor(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    *,
    center: NDArray | None = None,
) -> NDArray:
    """Build a compact QR factor from bounded weighted row chunks."""
    weights = np.asarray(weights, dtype=float)
    factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        block = np.sqrt(weights[start:stop])[:, None] * block
        stacked = block if factor is None else np.vstack((factor, block))
        factor = np.linalg.qr(stacked, mode="r")
    return np.empty((0, width)) if factor is None else np.asarray(factor)


def streamed_weighted_factor_rhs(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    response: NDArray,
    *,
    center: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Build a compact weighted QR factor and its consistently transformed RHS.

    Appending the response to every bounded design chunk preserves ``Q.T @ b``
    without retaining either the observation matrix or the observation-length
    orthogonal factor.  The returned factor has at most ``p + 1`` rows.
    """
    weights = np.asarray(weights, dtype=float)
    response = np.asarray(response, dtype=float)
    if weights.ndim != 1 or response.shape != weights.shape:
        raise ValueError("weights and response must be matching vectors")
    joint_factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        sqrt_weights = np.sqrt(weights[start:stop])
        joint_block = np.column_stack(
            (sqrt_weights[:, None] * block, sqrt_weights * response[start:stop])
        )
        stacked = joint_block if joint_factor is None else np.vstack((joint_factor, joint_block))
        joint_factor = np.linalg.qr(stacked, mode="r")
    if joint_factor is None:
        return np.empty((0, width)), np.empty(0)
    return np.asarray(joint_factor[:, :width]), np.asarray(joint_factor[:, width])


def _certification_required(
    *,
    method: str,
    width: int,
    rank: int,
    pre_truncation_condition: float,
    resolution_limited: bool,
    policy: RankPolicy,
) -> bool:
    """The certification predicate, over the five fields that decide it.

    ``decompose_gram`` knows all five before it builds the retained subspace,
    so the predicate is kept callable without a decomposition in hand --
    otherwise the eager path and the deferring path would each carry their own
    copy of the band, free to drift apart.
    """
    if method == "qr_svd":
        # A factor decomposition is already the authoritative certificate;
        # never stream and factor the same rows again merely because the
        # factor policy itself truncated a nonzero singular value.
        return False
    certification_condition = policy.warning_condition / np.sqrt(policy.certification_band)
    return bool(
        width > 0
        and (
            (rank == width and pre_truncation_condition >= certification_condition)
            or (rank < width and resolution_limited)
        )
    )


def needs_factor_certification(
    decomposition: RankDecomposition,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
) -> bool:
    """Whether Gram geometry lies inside a band requiring factor certification.

    A certificate governs the retained subspace as well as the integer rank.
    Normal equations can erase a factor-scale direction at the numerical
    boundary, or retain a different direction while reporting the same rank.

    **This shape -- a rank plus a flag saying whether to believe it -- is the
    published state of the art, not a local invention.**  Foster & Davis,
    "Algorithm 933", *ACM TOMS* 40(1) Art. 7 (2013), say of the routine they
    are improving on that it "returns an estimate for the numerical rank that
    is usually, but not always, correct", and their own contribution is
    described as reliable "in the sense that ... the numerical rank is
    accurately determined when a warning flag indicates that the numerical
    rank should be correct" -- the guarantee is *conditional* on the flag.  The
    mechanism is singular-value bounds "used to warn the user if the
    calculated numerical rank may be incorrect".  So the right answer to an
    ambiguous rank is a verdict, not a better threshold, and version 3's cut
    at :func:`_eigensolver_relative_bar` is what makes the verdict honest
    rather than what replaces it.

    **And a verdict only helps where it is read.**  Issue #356 is that call
    sites take :func:`decompose_gram` directly and never ask.  The
    load-bearing pair was ``inference/covariance.py``, where the pseudo-inverse
    IS the published covariance matrix: on the fixture in that issue this
    predicate is ``True`` on both arms of a round-off coin flip, correctly,
    and nothing downstream consulted it.  That pair now reads the verdict and
    falls back to the observation factor, so the published covariance is no
    longer taken from a Gram that cannot certify its own retained subspace.
    Version 3 is what made the verdict honest; reading it is a separate
    change and is not carried by this field.

    Twenty-one direct call sites remain, across nine modules -- seven in
    ``solvers/_structured/geometry.py``, five in ``reml/scop_geometry.py``, two
    each in ``reml/observed_geometry.py`` and ``reml/objective.py``, and one
    each in ``solvers/irls_direct.py``, ``solvers/constrained_qp.py``,
    ``reml/discrete.py``, ``model/state_ops.py`` and ``inference/metrics.py``.
    Each is its own question about whether the quantity it feeds is published
    or internal, and they are tracked on #356 rather than converted wholesale.
    """
    return _certification_required(
        method=decomposition.method,
        width=decomposition.width,
        rank=decomposition.rank,
        pre_truncation_condition=decomposition.pre_truncation_condition,
        resolution_limited=decomposition.resolution_limited,
        policy=policy,
    )


def _freeze(values: NDArray, *, dtype=float) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RankDecomposition:
    policy_version: int
    method: Literal["empty", "cholesky", "pivoted_cholesky", "gram_eigh", "qr_svd"]
    column_scale: NDArray
    active_columns: NDArray
    rank: int
    pre_truncation_condition: float
    cutoff: float
    rank_truncated: bool
    used_svd_fallback: bool
    resolution_limited: bool
    log_pdet: float
    cholesky_factor: NDArray | None = None
    pivots: NDArray | None = None
    solution_basis: NDArray | None = None
    parameter_null_basis: NDArray | None = None
    estimable_functional_basis: NDArray | None = None
    structural_aliases: NDArray | None = None
    retained_values: NDArray | None = None
    factor_rhs_left_basis: NDArray | None = None
    factor_rhs_triangular: NDArray | None = None

    @property
    def width(self) -> int:
        return int(self.column_scale.size)

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.width,):
            raise ValueError("rhs width does not match decomposition")
        if self.rank == 0:
            return np.zeros_like(rhs)
        if self.cholesky_factor is not None:
            active_rhs = rhs[self.active_columns] / self.column_scale[self.active_columns]
            active_solution = scipy.linalg.cho_solve(
                (self.cholesky_factor, True), active_rhs, check_finite=False
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution / self.column_scale[self.active_columns]
            return result
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        return self.solution_basis @ ((self.solution_basis.T @ rhs) / self.retained_values)

    def solve_factor_rhs(self, transformed_rhs: NDArray) -> NDArray:
        """Solve from a response transformed with the certified factor's QR.

        This path avoids re-forming normal equations at the factor-rank
        boundary.  It is available only when ``decompose_factor`` was asked to
        retain the bounded factor solve.
        """
        if self.factor_rhs_left_basis is None:
            raise RuntimeError("factor-RHS solve was not retained")
        transformed_rhs = np.asarray(transformed_rhs, dtype=float)
        if transformed_rhs.shape != (self.factor_rhs_left_basis.shape[0],):
            raise ValueError("transformed RHS length does not match the certified factor")
        if self.rank == 0:
            return np.zeros(self.width)
        projected_rhs = self.factor_rhs_left_basis.T @ transformed_rhs
        if self.factor_rhs_triangular is not None:
            active_solution = scipy.linalg.solve_triangular(
                self.factor_rhs_triangular,
                projected_rhs,
                lower=False,
                check_finite=False,
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution
            return result
        if self.solution_basis is None:
            raise RuntimeError("retained factor solution basis is unavailable")
        return self.solution_basis @ projected_rhs

    def pseudo_inverse(self) -> NDArray:
        if self.rank == 0:
            return np.zeros((self.width, self.width))
        if self.cholesky_factor is not None:
            inverse_equilibrated = scipy.linalg.cho_solve(
                (self.cholesky_factor, True),
                np.eye(len(self.active_columns)),
                check_finite=False,
            )
            inverse = np.zeros((self.width, self.width))
            scale = self.column_scale[self.active_columns]
            inverse[np.ix_(self.active_columns, self.active_columns)] = (
                inverse_equilibrated / np.outer(scale, scale)
            )
            return 0.5 * (inverse + inverse.T)
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        inverse = (self.solution_basis / self.retained_values) @ self.solution_basis.T
        return 0.5 * (inverse + inverse.T)

    def retained_parameter_basis(self) -> NDArray:
        if self.solution_basis is not None:
            return self.solution_basis.copy()
        basis = np.zeros((self.width, self.rank))
        if self.rank:
            basis[self.active_columns, :] = np.diag(1.0 / self.column_scale[self.active_columns])
        return basis

    def null_basis(self) -> NDArray:
        if self.parameter_null_basis is None:
            return np.zeros((self.width, 0))
        return self.parameter_null_basis.copy()

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != (self.width,):
            raise ValueError("contrast width does not match decomposition")
        scaled_columns = self.column_scale > 0.0
        contrast_norm = float(np.linalg.norm(contrast))
        structural_tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            contrast_norm,
            np.finfo(float).tiny,
        )
        if np.linalg.norm(contrast[~scaled_columns]) > structural_tolerance:
            return False
        null = self.null_basis()
        if null.shape[1] == 0:
            return True

        # Test orthogonality in the equilibrated dual coordinates used by the
        # rank decision.  Comparing ``contrast @ parameter_null_basis`` against
        # an unscaled absolute tolerance makes exact aliases appear estimable
        # when one design column is multiplied by a large constant.
        scaled_contrast = contrast[scaled_columns] / self.column_scale[scaled_columns]
        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            return True
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        projection = scaled_contrast @ normalized_null
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            float(np.linalg.norm(scaled_contrast)),
            np.finfo(float).tiny,
        )
        return bool(np.linalg.norm(projection) <= tolerance)

    def coefficient_estimable(self) -> NDArray:
        """Return all unit-coordinate estimability decisions in one projection."""
        scaled_columns = self.column_scale > 0.0
        result = np.zeros(self.width, dtype=bool)
        null = self.null_basis()
        if null.shape[1] == 0:
            result[scaled_columns] = True
            return result

        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            result[scaled_columns] = True
            return result
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        result[scaled_columns] = (
            np.linalg.norm(normalized_null, axis=1) <= SHARED_RANK_POLICY.factor_rcond
        )
        return result


@dataclass(frozen=True)
class RankInfo:
    """Compact fitted-subspace metadata in solver coefficient coordinates."""

    policy_version: int
    coordinate_space: Literal["solver"]
    selected_columns: NDArray
    selected_group_names: Sequence[str]
    sum_w: float
    mean_x: NDArray
    intercept_edf: float
    data: RankDecomposition
    augmented: RankDecomposition
    coefficient: RankDecomposition
    feature_edf: NDArray
    group_edf: Mapping[str, float]
    objective_loss: float | None

    @property
    def total_edf(self) -> float:
        return self.intercept_edf + float(np.sum(self.feature_edf))

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != self.mean_x.shape:
            raise ValueError("rhs width does not match fitted coefficient space")
        result = np.zeros_like(rhs)
        result[self.selected_columns] = self.augmented.solve(rhs[self.selected_columns])
        return result

    def pseudo_inverse(self) -> NDArray:
        width = len(self.mean_x)
        result = np.zeros((width, width))
        result[np.ix_(self.selected_columns, self.selected_columns)] = (
            self.augmented.pseudo_inverse()
        )
        return result

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != self.mean_x.shape:
            raise ValueError("contrast width does not match fitted coefficient space")
        unselected = np.ones(len(contrast), dtype=bool)
        unselected[self.selected_columns] = False
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(1.0, float(np.linalg.norm(contrast)))
        if np.linalg.norm(contrast[unselected]) > tolerance:
            return False
        return self.data.is_estimable(contrast[self.selected_columns])

    def coefficient_estimable(self) -> NDArray:
        result = np.zeros(len(self.mean_x), dtype=bool)
        result[self.selected_columns] = self.data.coefficient_estimable()
        return result


def _equilibrate_gram(
    matrix: NDArray, *, allow_indefinite: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    symmetric = _symmetric_part(values)
    diagonal = np.diag(symmetric)
    scale_reference = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
    if not allow_indefinite and np.any(diagonal < -100.0 * _EPS * scale_reference):
        raise ValueError("matrix has a materially negative diagonal")
    if allow_indefinite:
        row_scale = np.max(np.abs(symmetric), axis=1, initial=0.0)
        diagonal_scale = np.maximum(np.abs(diagonal), _EPS * row_scale)
    else:
        diagonal_scale = np.maximum(diagonal, 0.0)
    active_columns = np.flatnonzero(diagonal_scale > 0.0)
    column_scale = np.zeros(len(diagonal))
    column_scale[active_columns] = np.sqrt(diagonal_scale[active_columns])
    if active_columns.size:
        active_scale = column_scale[active_columns]
        equilibrated = symmetric[np.ix_(active_columns, active_columns)] / np.outer(
            active_scale, active_scale
        )
        equilibrated = 0.5 * (equilibrated + equilibrated.T)
    else:
        equilibrated = np.zeros((0, 0))
    return equilibrated, column_scale, active_columns, symmetric


def _null_basis(
    width: int,
    active_columns: NDArray,
    active_scale: NDArray,
    discarded_vectors: NDArray,
) -> NDArray:
    """Stack the parameter-space null basis: discarded spectral, then structural.

    The **row layout** here is a load-bearing cross-module invariant, not just
    an implementation detail of this function:

    * the discarded-spectral columns are supported only on ``active_columns``,
    * the structural columns are exact unit vectors on the inactive columns,
    * so the two blocks have **disjoint row supports** and split ``null(H)``
      orthogonally.

    ``constrained_qp._null_space_mass`` depends on exactly that.  Changing the
    row supports, or mixing the two kinds of column into shared rows, silently
    breaks that consumer's split.
    """
    pieces: list[NDArray] = []
    if discarded_vectors.shape[1]:
        discarded = np.zeros((width, discarded_vectors.shape[1]))
        discarded[active_columns, :] = discarded_vectors / active_scale[:, None]
        pieces.append(discarded)
    inactive = np.setdiff1d(np.arange(width), active_columns, assume_unique=True)
    if inactive.size:
        inactive_basis = np.zeros((width, inactive.size))
        inactive_basis[inactive, np.arange(inactive.size)] = 1.0
        pieces.append(inactive_basis)
    return np.column_stack(pieces) if pieces else np.zeros((width, 0))


def _earliest_representatives(null_vectors: NDArray, rank: int) -> NDArray | None:
    """Earliest independent representatives, read off a null basis already in hand.

    Index-order greedy selection keeps column ``j`` unless it is a combination
    of the columns before it -- equivalently, unless some null vector has its
    LAST nonzero at ``j``.  Eliminating the null basis from the right-hand end
    pivots on precisely those positions, so the pivots are the rejected columns
    and the complement is the greedy selection, not an approximation of it.

    Deciding the same question by testing each prefix's spectrum costs one
    eigendecomposition per candidate, which is ``O(m**4)`` across the sweep and
    is why a rank-deficient block used to cost hundreds of times a full-rank one
    of the same width.  This is ``O(k**2 m)`` in the NULLITY ``k``, so a block
    that is a few columns short of full rank is nearly free.

    "Still reaches column ``j``" is decided by the null space's LEVERAGE there,
    ``||P_S e_j||**2``, not by any single vector's component.  Those vectors come
    out of an eigendecomposition of a singular system, so a component that is
    mathematically zero arrives as noise many orders above machine epsilon --
    measured at 1e-12 against a 1e-14 absolute threshold on an 8-column block,
    which pivoted on dust and returned a rank-deficient selection.  ``sqrt(eps)``
    is the floor below which a unit-norm null direction carries no information
    here, and leverage is the basis-free way to ask the question.

    It has to be basis-free.  A null SPACE has no canonical basis, and reading
    ``max |N[i, j]|`` instead makes the answer depend on the one ``eigh`` chose:
    for leverage ``s**2`` that maximum lies anywhere in ``[s / sqrt(k), s]``
    depending on the rotation, so any ``s`` in ``(sqrt(eps), sqrt(k) sqrt(eps))``
    is dust for one basis and a dependency for another.  That is not
    hypothetical -- the component form moved on 455 of 4000 subspaces built
    inside that band, and carried ``_conditioned_representatives`` with it on
    21 of them.  Leverage cannot move: ``(N Q)(N Q).T = N N.T``.

    The deflation is basis-free for the same reason.  Rejecting column ``c``
    leaves ``{v in S : v_c = 0}``, a property of the subspace and the column, so
    a Householder maps ``Q[:, c]`` onto the first axis and that axis is dropped.
    Rows stay orthonormal, so the next leverage is read straight off the column
    norms and the cost stays ``O(k**2 m)``.

    Returns ``None`` when the scan cannot reject all ``k`` columns.  The caller
    then keeps whatever exact path it already had rather than proceeding on a
    selection this could not certify.
    """
    width, nullity = null_vectors.shape
    if nullity == 0:
        return np.arange(width, dtype=int)
    basis = np.array(null_vectors.T, dtype=float)
    if not np.all(np.isfinite(basis)):
        return None
    floor = float(np.sqrt(np.finfo(float).eps)) ** 2

    rejected: list[int] = []
    remaining = np.ones(width, dtype=bool)
    for _step in range(nullity):
        if basis.shape[0] == 0:
            break
        columns = np.flatnonzero(remaining)
        if columns.size == 0:
            break
        leverage = np.einsum("ij,ij->j", basis[:, columns], basis[:, columns])
        reachable = np.flatnonzero(leverage > floor)
        if reachable.size == 0:
            break
        # the LATEST column the null space still reaches: the earliest-column
        # convention, decided on a quantity the basis cannot move
        chosen = int(columns[int(reachable[-1])])
        rejected.append(chosen)
        remaining[chosen] = False

        direction = basis[:, chosen].copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            break
        direction[0] += float(np.copysign(norm, direction[0] if direction[0] != 0.0 else 1.0))
        reflector_norm = float(np.linalg.norm(direction))
        if reflector_norm > 0.0:
            direction /= reflector_norm
            basis = basis - 2.0 * np.outer(direction, direction @ basis)
        basis = basis[1:]

    if len(rejected) != nullity:
        return None
    keep = np.setdiff1d(np.arange(width), np.asarray(rejected, dtype=int))
    return keep if keep.size == rank else None


# A fallback pivot may carry this fraction of the largest null-space share
# available, measured on the ``sqrt(leverage)`` scale so the constant keeps the
# meaning it had when the rule read components directly.  See
# ``_leverage_pivot_representatives`` for why it is not 1.
_PIVOT_THRESHOLD = 0.5


def _leverage_pivot_representatives(null_vectors: NDArray, rank: int) -> NDArray | None:
    """Representatives chosen for conditioning, from a quantity the basis cannot move.

    A null SPACE has no canonical basis.  ``eigh`` and ``svd`` return an
    arbitrary orthonormal one, and ``N @ Q`` spans the same subspace for any
    orthogonal ``Q``, so any rule reading individual components of ``N`` is
    reading a coordinate that the eigensolver was free to choose.  The rule this
    replaces did exactly that -- it pivoted on ``max |N[i, j]|`` -- and it moved
    under rotation on 58 of 400 random 6x2 subspaces, rising to 224 of 400 at
    12x4, with the achieved amplification moving too (6.7803 against 2.0732 on
    one subspace).

    ``_earliest_representatives`` did not move on those 400, and that was NOT
    because its question is basis-free.  "The last column any null direction
    still reaches" is a property of the row space, but the ``sqrt(eps)`` floor
    it used to answer that question with was not: it read a single component,
    and for leverage ``s**2`` the largest component ranges over
    ``[s / sqrt(k), s]`` across bases.  Subspaces constructed inside the
    resulting band moved it on 455 of 4000.  The uniform sweep simply never
    produced one -- so it decides on leverage now too, and the two selectors
    share this criterion rather than one of them being safe by nature.

    So the criterion here is the null-space LEVERAGE of each column,
    ``diag(N @ N.T)``, which is invariant by construction: ``(N Q)(N Q).T =
    N Q Q.T N.T = N N.T``.  It is the share of the unit vector ``e_j`` that lies
    in the null space -- 1 when column ``j`` is entirely redundant, 0 when it is
    untouched by any alias -- and rejecting the column with the most of it is
    the classical alias-detection choice.

    The DEFLATION has to be invariant too, or the second step reintroduces what
    the first avoided.  Rejecting column ``c`` leaves ``{v in S : v_c = 0}``,
    which is a property of the subspace and the column, so it is taken as one:
    a Householder reflection maps ``Q[:, c]`` onto the first axis and the
    remaining rows are dropped, leaving an orthonormal basis of exactly that
    subspace.  Row norms therefore stay 1 throughout and the leverage of the
    next step is read straight off the column norms.

    ``_PIVOT_THRESHOLD`` keeps its meaning from the rule this replaces -- it is
    applied to ``sqrt(leverage)``, which is the scale the components were on --
    and so does the reason it is not 1: index order is worth keeping where
    conditioning does not pay for it, and an exact alias splits its leverage
    evenly across the columns it ties, so the tie still falls to the latest.

    Invariance cost nothing here.  On the 287 blocks of a 400-block sweep where
    the earliest rule failed its certificate, this rule and the component rule
    it replaces return selections of IDENTICAL amplification on all 287 --
    better on none, worse on none.  Both take the median from 1.7952e+07 down
    to 2.0000 and the maximum from 7.077e+15 down to 8.555.  The component rule
    was sound in intent and unsound only in its input.

    Both also leave 2 of those 287 above ``_achievable_amplification``, so that
    is a property of greedy selection rather than of either criterion: the
    bound is what a largest-volume subset achieves, and neither rule searches
    for one.  ``_conditioned_representatives`` returns the better of this and
    the earliest selection, so the result is never worse than index order alone
    -- verified over the same 287, worst ratio exactly 1.000000.

    Cost is unchanged at ``O(k**2 m)`` in the nullity ``k``: the Householder is
    ``O(k m)`` per step, exactly what the elimination it replaces cost.
    """
    width, nullity = null_vectors.shape
    if nullity == 0:
        return np.arange(width, dtype=int)
    basis = np.array(null_vectors.T, dtype=float)
    if not np.all(np.isfinite(basis)):
        return None

    rejected: list[int] = []
    remaining = np.ones(width, dtype=bool)
    for _step in range(nullity):
        if basis.shape[0] == 0:
            break
        columns = np.flatnonzero(remaining)
        if columns.size == 0:
            break
        leverage = np.einsum("ij,ij->j", basis[:, columns], basis[:, columns])
        # Leverage is a diagonal of a projector, so it lives in [0, 1] and a
        # column the null space does not reach at all sits at machine dust.
        leverage[leverage <= _EPS] = 0.0
        peak = float(leverage.max(initial=0.0))
        if peak <= 0.0:
            break
        qualifying = np.flatnonzero(leverage >= (_PIVOT_THRESHOLD**2) * peak)
        chosen = int(columns[int(qualifying[-1])])
        rejected.append(chosen)
        remaining[chosen] = False

        # Householder onto {v in S : v_chosen = 0}: reflect Q[:, chosen] to the
        # first axis, then drop that axis.  The result is orthonormal because a
        # reflection is, so no re-orthogonalisation is needed.
        direction = basis[:, chosen].copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            break
        direction[0] += float(np.copysign(norm, direction[0] if direction[0] != 0.0 else 1.0))
        reflector_norm = float(np.linalg.norm(direction))
        if reflector_norm > 0.0:
            direction /= reflector_norm
            basis = basis - 2.0 * np.outer(direction, direction @ basis)
        basis = basis[1:]

    if len(rejected) != nullity:
        return None
    keep = np.setdiff1d(np.arange(width), np.asarray(rejected, dtype=int))
    return keep if keep.size == rank else None


def _selection_amplification(null_vectors: NDArray, keep: NDArray) -> float:
    """How much worse than the retained subspace itself a selection is.

    Split the rows of the null basis ``N`` -- orthonormal columns, so
    ``N.T @ N = I`` -- into the REJECTED rows ``N_R`` and the KEPT rows ``N_K``.
    Then ``N_R.T N_R = I - N_K.T N_K`` gives

        (N_R.T N_R)^-1 - I = (N_K N_R^-1).T (N_K N_R^-1),

    so ``1 / sigma_min(N_R)**2 == 1 + ||N_K N_R^-1||_2**2`` identically, and the
    selected block inherits

        sigma_min(X[:, keep]) >= sigma_rank(X) * sigma_min(N_R).

    The returned ``1 / sigma_min(N_R)`` is therefore the factor by which the
    CHOICE OF REPRESENTATIVES multiplies the condition number the retained
    subspace already has.  It is a property of the selection alone, which is
    why it sees what no test against the rank cutoff can: a block may sit
    comfortably above the cutoff that decided the rank -- not deficient by that
    standard, and accepted by Cholesky -- while still being the worst basis
    available for the subspace it spans.

    Verified over 400 random rank-deficient blocks: the identity holds to
    2.37e-13 relative, and the tightest observed
    ``sigma_min(X_keep) / (sigma_rank(X) * sigma_min(N_R))`` was 1.000002, so
    the bound is attained rather than merely true.
    """
    width = null_vectors.shape[0]
    rejected = np.setdiff1d(np.arange(width), keep)
    if rejected.size == 0:
        return 1.0
    spectrum = np.linalg.svd(null_vectors[rejected, :], compute_uv=False)
    smallest = float(np.min(spectrum)) if spectrum.size else 0.0
    return float("inf") if smallest <= 0.0 else 1.0 / smallest


def _achievable_amplification(width: int, nullity: int) -> float:
    """``sqrt(1 + k*(m-k))``, the amplification a rank-revealing choice reaches.

    The largest-volume ``k x k`` submatrix of a null basis with orthonormal
    columns has ``|N_K N_R^-1|`` bounded entrywise by 1 -- otherwise a swap
    would increase the volume -- so its spectral norm is at most
    ``sqrt(k*(m-k))`` and, by the identity in ``_selection_amplification``, its
    amplification is at most ``sqrt(1 + k*(m-k))``.  A selection worse than
    that is worse than one that provably exists, which makes this the natural
    place to stop trusting index order, rather than a tuned constant.
    """
    return float(np.sqrt(1.0 + float(nullity) * float(width - nullity)))


def _principal_block_condition(equilibrated: NDArray, keep: NDArray) -> float:
    """Condition of the principal block a selection actually hands to the solver.

    This is the matrix that gets factorised and then solved against, so it is
    the quantity a selection should be judged on.  ``_selection_amplification``
    bounds it from one side only, which is enough to SCREEN a selection and not
    enough to choose between two.

    Returns ``inf`` when the block will not factorise, which ranks it below any
    block that will.
    """
    block = equilibrated[np.ix_(keep, keep)]
    try:
        factor = scipy.linalg.cholesky(block, lower=True, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")
    pocon = scipy.linalg.get_lapack_funcs("pocon", (factor,))
    reciprocal, info = pocon(factor, float(np.linalg.norm(block, ord=1)), uplo="L")
    if info != 0 or not np.isfinite(reciprocal) or reciprocal <= 0.0:
        return float("inf")
    return float(1.0 / reciprocal)


def _principal_block_clears_cutoff(
    equilibrated: NDArray,
    keep: NDArray,
    cutoff: float,
) -> bool:
    """Whether a representative numerically clears the full block's cutoff.

    The decomposition decides ``rank`` against one matrix-wide cutoff
    ``cutoff``.  A principal block may be locally well-conditioned and still
    lose a direction against that ORIGINAL cutoff because its own spectral
    scale is smaller.  Handing such a block to Cholesky would make the solve
    rank deficient under the policy that selected it.

    A bare Cholesky of ``B - cutoff * I`` is not a strict floating-point test:
    DPOTRF is backward stable, so it may succeed when the shifted matrix is
    semidefinite.  That happened at exact equality on a two-column public-path
    fixture, with a spurious final diagonal around ``1e-8``.

    Compute only the smallest symmetric eigenvalue and require it to clear the
    cutoff by a backward-error allowance.  Symmetric eigensolvers are
    backward stable to ``O(m * eps * ||B||)``; the factor 64 deliberately makes
    the boundary band conservative and also covers forming and slicing the
    principal block.  This is a NUMERICAL certificate, not a claim that the
    returned eigenvalue is exact.  Equality, values below the cutoff, and
    values too close to distinguish all keep the spectral representation.
    """
    if not np.isfinite(cutoff) or cutoff < 0.0:
        return False
    block = equilibrated[np.ix_(keep, keep)]
    scale = max(
        float(np.linalg.norm(block, ord=np.inf)),
        abs(cutoff),
        np.finfo(float).tiny,
    )
    allowance = 64.0 * max(1, len(keep)) * _EPS * scale
    try:
        smallest = float(
            scipy.linalg.eigvalsh(
                block,
                subset_by_index=[0, 0],
                check_finite=False,
            )[0]
        )
    except (np.linalg.LinAlgError, ValueError):
        return False
    return bool(np.isfinite(smallest) and smallest > cutoff + allowance)


def _compact_factor_qr(
    spectral_factor: NDArray,
    keep: NDArray,
) -> tuple[NDArray, NDArray] | None:
    """Return a canonical thin QR for one factor-space representative.

    ``spectral_factor = diag(s) @ Vh`` is left-orthogonally equivalent to the
    equilibrated observation factor but has at most ``width`` rows.  A QR of
    its selected columns therefore recovers the representative geometry
    without another decomposition whose error or cost scales with the
    observation count.

    QR leaves the signs of corresponding columns of ``Q`` and rows of ``R``
    arbitrary.  Canonicalise them here so ``diag(R)`` is positive.  Then
    ``R.T`` is a conventional lower Cholesky factor satisfying
    ``R.T @ R == spectral_factor[:, keep].T @ spectral_factor[:, keep]`` up to
    the backward error of this width-bounded QR.
    """
    compact = spectral_factor[:, keep]
    try:
        compact_left, upper = scipy.linalg.qr(
            compact,
            mode="economic",
            check_finite=False,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None
    diagonal = np.diag(upper)
    if not np.all(np.isfinite(upper)) or np.any(diagonal == 0.0):
        return None
    row_signs = np.where(diagonal < 0.0, -1.0, 1.0)
    compact_left = compact_left * row_signs
    upper = row_signs[:, None] * upper
    return compact_left, upper


def _triangular_gram_condition(upper: NDArray) -> float:
    """Condition of the Gram represented by a compact factor's upper ``R``.

    The representative solve uses ``L = R.T``.  Estimate the condition from
    that same factor rather than forming a Gram over all observation rows.
    Only the matrix norm needs an explicit product, and its dot products have
    length ``rank <= width``.
    """
    lower = upper.T
    compact_gram = lower @ lower.T
    matrix_norm = float(np.linalg.norm(compact_gram, ord=1))
    if not np.isfinite(matrix_norm) or matrix_norm <= 0.0:
        return float("inf")
    pocon = scipy.linalg.get_lapack_funcs("pocon", (lower,))
    reciprocal, info = pocon(lower, matrix_norm, uplo="L")
    if info != 0 or not np.isfinite(reciprocal) or reciprocal <= 0.0:
        return float("inf")
    return float(1.0 / reciprocal)


def _factor_representative_clears_cutoff(
    spectral_factor: NDArray,
    keep: NDArray,
    *,
    largest_singular_value: float,
    smallest_retained_singular_value: float,
    cutoff: float,
    amplification: float,
) -> bool:
    """Certify a representative in the factor SVD's own coordinates.

    Let the authoritative computed rank-``r`` factor be

    ``F_r = U_r diag(s[:r]) Vh[:r]``

    and let ``N = Vh[r:].T`` be its null basis.  Splitting ``N`` into rows for
    the kept and rejected columns gives the identity documented by
    :func:`_selection_amplification`, hence

    ``sigma_min(F[:, keep]) >= s[r - 1] / amplification``.

    The full factor can only improve that bound: its discarded left-singular
    coordinates add a positive-semidefinite term to the selected Gram.  The
    first test below therefore stays entirely on the singular scale already
    used to compute ``cutoff`` and needs no Gram formation.

    When that bound is inconclusive, assess the candidate in a COMPACT spectral
    factor ``diag(s) @ Vh``.  It has at most ``width`` rows regardless of the
    observation count, so it cannot inherit the ``O(n * eps)`` accumulation of
    ``F.T @ F``.  The compact product is not exact either: a length-``q`` dot
    product has componentwise error bounded by ``gamma_q``.  ``q + 4`` covers
    scaling the right vectors as well as the multiply/accumulate, and the
    positive absolute product turns that into a spectral-norm formation bound.
    The ordinary principal-block certificate then adds its own eigensolver
    backward-error allowance.

    No SVD is performed here.  ``spectral_factor``, the singular values and the
    null-space amplification all come from work the deficient factor path
    already performed.
    """
    if (
        not np.isfinite(cutoff)
        or cutoff < 0.0
        or not np.isfinite(amplification)
        or amplification <= 0.0
    ):
        return False

    singular_scale = max(
        abs(largest_singular_value),
        abs(cutoff),
        np.finfo(float).tiny,
    )
    singular_allowance = 64.0 * spectral_factor.shape[1] * _EPS * singular_scale
    lower_bound = smallest_retained_singular_value / amplification
    if np.isfinite(lower_bound) and lower_bound > cutoff + singular_allowance:
        return True

    compact = spectral_factor[:, keep]
    compact_gram = compact.T @ compact
    absolute_gram = np.abs(compact).T @ np.abs(compact)
    operations = compact.shape[0] + 4
    operation_error = operations * _EPS
    if operation_error >= 0.5:
        return False
    gamma = operation_error / (1.0 - operation_error)
    formation_allowance = (gamma / (1.0 - gamma)) * float(np.linalg.norm(absolute_gram, ord=np.inf))
    if not np.isfinite(formation_allowance):
        return False
    return _principal_block_clears_cutoff(
        compact_gram,
        np.arange(len(keep), dtype=int),
        cutoff**2 + formation_allowance,
    )


def _conditioned_representatives(
    null_vectors: NDArray,
    rank: int,
    block_condition: Callable[[NDArray], float] | None = None,
    *,
    block_admissible: Callable[[NDArray, float], bool] | None = None,
    maximum_condition: float | None = None,
) -> NDArray | None:
    """Earliest representatives, unless index order costs more than it may.

    The earliest rule is a labelling convention -- it decides which of a set of
    aliased columns carries the reproducible zero -- and it is chosen blind to
    conditioning.  Where the earliest independent columns happen to be two
    near-duplicates, that convention is paid for in every downstream solve: the
    block is positive definite, Cholesky accepts it, its smallest eigenvalue is
    above the cutoff that decided the rank, and it is still the worst basis for
    its own span.

    So the convention is kept, and certified -- but the certificate TRIGGERS the
    search and does not decide it.  ``_selection_amplification`` is a bound:
    ``sigma_min(X[:, keep]) >= sigma_rank(X) * sigma_min(N_R)`` puts a floor
    under one end of a ratio whose other end, ``sigma_max`` of the selected
    block, also moves with the selection.  So a smaller amplification improves a
    worst case and says nothing per instance, and on anisotropic factors the two
    orderings genuinely disagree: over 9,654 rank-deficient 3x4 blocks where
    this routine switched, 24 switched to a block whose actual condition was
    WORSE, by up to 1.34x.

    The decision is therefore made on the thing that matters downstream -- the
    condition of the principal block that will be factorised and solved --
    whenever the caller can supply it.  ``block_condition`` takes a candidate
    selection and returns that condition; the alternative is taken only if it is
    strictly better.  Callers that cannot supply one fall back to the bound,
    which is the old behaviour and is documented as weaker.

    The amplification is still the right trigger: it is cheap, it costs
    ``O(k**3)`` in the NULLITY against the ``O(m**3)`` eigendecomposition this
    path has already paid, and being a bound is exactly what makes it a safe
    screen -- it cannot miss a block that is genuinely badly conditioned.

    ``maximum_condition`` is the absolute backstop after that relative choice.
    Picking the better of two representative blocks does not make either one
    usable: where the retained subspace itself sits near the rank boundary,
    both may be catastrophically conditioned even though Cholesky accepts
    them.  In that case ``None`` tells the caller to keep the spectral
    decomposition rather than replace it with an unstable coefficient basis.
    A maximum requires ``block_condition`` because the null-space
    amplification alone cannot certify an individual principal block.

    ``block_admissible`` is a stricter postcondition owned by the caller's
    decomposition policy.  Production uses it to require the ORIGINAL
    full-matrix rank cutoff, which a local condition cannot encode.  It
    receives the candidate and its cached null-space amplification so the
    factor path can stay in the SVD coordinates that decided its rank.  An
    inadmissible earliest candidate triggers the same existing alternative
    search; condition then chooses only among admissible candidates, so a safe
    representative is not discarded merely because an unsafe one had the
    lower local condition.
    """
    if maximum_condition is not None and block_condition is None:
        raise ValueError("maximum_condition requires a block_condition scorer")

    amplifications: dict[tuple[int, ...], float] = {}
    conditions: dict[tuple[int, ...], float] = {}
    admissibility: dict[tuple[int, ...], bool] = {}

    def amplification(keep: NDArray) -> float:
        key = tuple(int(index) for index in keep)
        if key not in amplifications:
            amplifications[key] = _selection_amplification(null_vectors, keep)
        return amplifications[key]

    def condition(keep: NDArray) -> float:
        key = tuple(int(index) for index in keep)
        if key not in conditions:
            assert block_condition is not None
            conditions[key] = float(block_condition(keep))
        return conditions[key]

    def admissible(keep: NDArray) -> bool:
        key = tuple(int(index) for index in keep)
        if key not in admissibility:
            admissibility[key] = (
                True
                if block_admissible is None
                else bool(block_admissible(keep, amplification(keep)))
            )
        return admissibility[key]

    earliest = _earliest_representatives(null_vectors, rank)
    if earliest is None:
        return None
    earliest_amplification = amplification(earliest)
    selected = earliest
    if not admissible(earliest) or earliest_amplification > _achievable_amplification(
        *null_vectors.shape
    ):
        alternative = _leverage_pivot_representatives(null_vectors, rank)
        if alternative is not None and not np.array_equal(alternative, earliest):
            candidates = [
                candidate for candidate in (earliest, alternative) if admissible(candidate)
            ]
            if not candidates:
                return None
            selected = candidates[0]
            if len(candidates) == 2:
                if block_condition is not None:
                    selected = (
                        alternative if condition(alternative) < condition(earliest) else earliest
                    )
                elif amplification(alternative) < earliest_amplification:
                    selected = alternative

    if not admissible(selected):
        return None
    if maximum_condition is not None and not condition(selected) <= maximum_condition:
        return None
    return selected


def try_decompose_verified_spd_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
) -> RankDecomposition | None:
    """Return a conservative full-rank Cholesky decomposition or ``None``.

    This is a fast path for coefficient systems already expected to be
    positive definite.  It uses the shared Gram equilibration, but accepts the
    factor only when every column is structurally active and LAPACK's
    reciprocal condition estimate remains above the stricter factor-scale
    rank tolerance.  Ambiguous systems must use :func:`decompose_gram`.
    """

    if not isinstance(policy, RankPolicy):
        raise TypeError("policy must be a RankPolicy")
    if (
        isinstance(residual_tol, bool)
        or not isinstance(residual_tol, int | float)
        or not np.isfinite(residual_tol)
        or residual_tol <= 0.0
    ):
        raise ValueError("residual_tol must be finite and positive")
    try:
        equilibrated, column_scale, active_columns, _ = _equilibrate_gram(matrix)
    except (ValueError, np.linalg.LinAlgError):
        return None
    width = len(column_scale)
    if width == 0 or len(active_columns) != width:
        return None
    try:
        factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
        matrix_norm = float(np.linalg.norm(equilibrated, ord=1))
        pocon = scipy.linalg.get_lapack_funcs("pocon", (factor,))
        reciprocal_condition, info = pocon(factor, matrix_norm, uplo="L")
        if (
            info != 0
            or not np.isfinite(reciprocal_condition)
            or reciprocal_condition <= policy.factor_rcond
        ):
            return None
        probe = np.arange(1.0, width + 1.0)
        solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
        residual = np.linalg.norm(equilibrated @ solved - probe) / max(
            np.linalg.norm(probe),
            np.finfo(float).tiny,
        )
        if not np.isfinite(residual) or residual > residual_tol:
            return None
    except (ValueError, np.linalg.LinAlgError):
        return None

    log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
        np.sum(np.log(column_scale))
    )
    null = _null_basis(
        width,
        active_columns,
        column_scale,
        np.zeros((width, 0)),
    )
    return RankDecomposition(
        policy_version=policy.version,
        method="cholesky",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=width,
        pre_truncation_condition=float(np.sqrt(1.0 / reciprocal_condition)),
        cutoff=policy.gram_rcond * matrix_norm,
        rank_truncated=False,
        used_svd_fallback=False,
        resolution_limited=False,
        log_pdet=log_pdet,
        cholesky_factor=_freeze(factor),
        parameter_null_basis=_freeze(null),
        structural_aliases=_freeze(np.zeros(width, dtype=bool), dtype=bool),
    )


def _scaled_subspace_logdet(coordinates: NDArray) -> float:
    """Return ``log(det(coordinates.T @ coordinates))`` across extreme row scales."""
    width = coordinates.shape[1]
    if width == 0:
        return 0.0

    # Ordinary QR/SVD only provides absolute accuracy.  DGEJSV's 'F' mode
    # applies full row and column pivoting so diagonal scaling cannot erase a
    # genuine retained direction.  Ask for the unrestricted singular-value
    # range because rank has already been decided in equilibrated coordinates.
    singular_values, _, _, scaling, _, info = scipy.linalg.lapack.dgejsv(
        np.asfortranarray(coordinates),
        joba=2,  # 'F': full-pivoting, high-relative-accuracy preprocessing
        jobu=3,  # 'N': singular values only
        jobv=3,  # 'N': singular values only
        jobr=0,  # 'N': do not truncate the requested singular-value range
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"high-accuracy retained SVD failed with info={info}")
    if np.any(singular_values <= 0.0) or np.any(scaling[:2] <= 0.0):
        raise ValueError("retained coordinate basis is not full rank")
    log_scale = float(np.log(scaling[0]) - np.log(scaling[1]))
    return 2.0 * (float(np.sum(np.log(singular_values))) + width * log_scale)


def _retained_log_pdet(
    active_scale: NDArray,
    retained_vectors: NDArray,
    discarded_vectors: NDArray,
    retained_values: NDArray,
) -> float:
    """Return the retained pseudo-logdet without forming a coordinate Gram."""
    if retained_values.size == 0:
        return 0.0

    # V (retained) and N (discarded) form an orthogonal basis.  Jacobi's
    # complementary-minor identity gives
    #
    # det(V.T D^2 V) = det(D)^2 det(N.T D^-2 N).
    #
    # Evaluate whichever side has fewer columns; this is both cheaper and more
    # accurate for the common one-alias case.
    if retained_vectors.shape[1] <= discarded_vectors.shape[1]:
        coordinate_logdet = _scaled_subspace_logdet(active_scale[:, None] * retained_vectors)
    else:
        coordinate_logdet = 2.0 * float(np.sum(np.log(active_scale)))
        coordinate_logdet += _scaled_subspace_logdet(discarded_vectors / active_scale[:, None])
    return coordinate_logdet + float(np.sum(np.log(np.abs(retained_values))))


def _decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
    allow_indefinite: bool = False,
    omit_uncertifiable: bool = False,
) -> RankDecomposition | None:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix.

    ``omit_uncertifiable`` is a pure optimization hint, never a semantic one:
    when it is set this may return ``None`` instead of a decomposition that
    :func:`needs_factor_certification` would have rejected anyway.  It is
    permitted to return a decomposition in that case too, so the caller still
    owns the predicate -- see :func:`decompose_gram_if_authoritative`.
    """
    equilibrated, column_scale, active_columns, _ = _equilibrate_gram(
        matrix, allow_indefinite=allow_indefinite
    )
    width = len(column_scale)
    structural_aliases = column_scale == 0.0
    if active_columns.size == 0:
        return RankDecomposition(
            policy_version=policy.version,
            method="empty",
            column_scale=_freeze(column_scale),
            active_columns=_freeze(active_columns, dtype=int),
            rank=0,
            pre_truncation_condition=float("inf"),
            cutoff=0.0,
            rank_truncated=width > 0,
            used_svd_fallback=False,
            resolution_limited=False,
            log_pdet=0.0,
            parameter_null_basis=_freeze(np.eye(width)),
            structural_aliases=_freeze(structural_aliases, dtype=bool),
            retained_values=_freeze(np.array([])),
        )

    if not allow_indefinite:
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            matrix_norm = float(np.linalg.norm(equilibrated, ord=1))
            trtri = scipy.linalg.get_lapack_funcs("trtri", (factor,))
            inverse_factor, inverse_info = trtri(
                factor,
                lower=1,
                unitdiag=0,
                overwrite_c=0,
            )
            if inverse_info != 0:
                raise np.linalg.LinAlgError("triangular inverse failed during rank certification")
            inverse_factor_frobenius = float(np.linalg.norm(inverse_factor, ord="fro"))
            min_eigenvalue_lower_bound = 1.0 / inverse_factor_frobenius**2
            pocon = scipy.linalg.get_lapack_funcs("pocon", (factor,))
            reciprocal_condition, info = pocon(factor, matrix_norm, uplo="L")
            # This shortcut returns *full* rank without running `eigh`, so its
            # threshold has to dominate the cutoff `eigh` would have applied,
            # or the two paths disagree on the same matrix.  `certification_band
            # * gram_rcond` is 32 eps and used to dominate a cutoff of eps
            # outright; against a cutoff floored at `n eps` it stops dominating
            # at width 32, so the bar is taken here too.  `matrix_norm` is the
            # 1-norm and `||A||_1 >= ||A||_2 = max|w|` for symmetric `A`, so
            # clearing this clears the spectral cutoff a fortiori.
            safely_full_rank = (
                np.isfinite(min_eigenvalue_lower_bound)
                and min_eigenvalue_lower_bound
                > max(
                    policy.certification_band * policy.gram_rcond,
                    _eigensolver_relative_bar(len(active_columns)),
                )
                * matrix_norm
            )
            if safely_full_rank:
                probe = np.arange(1.0, len(active_columns) + 1.0)
                solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
                residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                    np.linalg.norm(probe), 1e-300
                )
                if residual <= residual_tol:
                    log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                        np.sum(np.log(column_scale[active_columns]))
                    )
                    null = _null_basis(
                        width,
                        active_columns,
                        column_scale[active_columns],
                        np.zeros((len(active_columns), 0)),
                    )
                    return RankDecomposition(
                        policy_version=policy.version,
                        method="cholesky",
                        column_scale=_freeze(column_scale),
                        active_columns=_freeze(active_columns, dtype=int),
                        rank=len(active_columns),
                        pre_truncation_condition=float(
                            np.sqrt(1.0 / reciprocal_condition)
                            if info == 0
                            and np.isfinite(reciprocal_condition)
                            and reciprocal_condition > 0.0
                            else np.sqrt(matrix_norm / min_eigenvalue_lower_bound)
                        ),
                        # The same floored relative factor the spectral path
                        # cuts on, so the field means one thing whichever
                        # branch produced the decomposition.  Informational
                        # here -- this branch never truncates.
                        cutoff=max(
                            policy.gram_rcond,
                            _eigensolver_relative_bar(len(active_columns)),
                        )
                        * matrix_norm,
                        rank_truncated=len(active_columns) < width,
                        used_svd_fallback=False,
                        resolution_limited=False,
                        log_pdet=log_pdet,
                        cholesky_factor=_freeze(factor),
                        parameter_null_basis=_freeze(null),
                        structural_aliases=_freeze(structural_aliases, dtype=bool),
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    raw_eigenvalues = eigenvalues
    max_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues), initial=0.0))
    # Floored at the same bar as the cutoff, for the same reason: a matrix may
    # not be REFUSED as indefinite over a quantity `eigh` cannot resolve.  The
    # 100 eps constant predates the bar and is inert against it up to width
    # 100; past that the bar governs, and the direction it admits is then
    # dropped by the cutoff below, which is the coherent outcome rather than a
    # raise.  Widening only -- nothing that raised on a resolved negative
    # eigenvalue stops raising.
    #
    # It is also inert on the `allow_indefinite=True` side, which is worth
    # SHOWING rather than asserting because widening a definiteness test looks
    # like it should flip semantics.  `eigenvalues[0]` is the MINIMUM, so
    # admitting it means every negative eigenvalue satisfies `|w| <= n eps *
    # max(max_abs, 1)`; the equilibrated matrix has a unit diagonal, so
    # `max_abs >= 1` and that factor is just `max_abs`.  Every such eigenvalue
    # is therefore at or under the cutoff below and is dropped under either
    # semantics.  Nothing retained as indefinite stops being retained.
    negative_tolerance = max(100.0 * _EPS, _eigensolver_relative_bar(len(active_columns))) * max(
        max_abs_eigenvalue, 1.0
    )
    materially_indefinite = bool(eigenvalues[0] < -negative_tolerance)
    if not allow_indefinite and materially_indefinite:
        raise ValueError(
            "matrix is materially indefinite "
            f"(min equilibrated eigenvalue={eigenvalues[0]:.3e}, "
            f"scale={max_abs_eigenvalue:.3e})"
        )
    psd_semantics = not materially_indefinite
    if psd_semantics:
        # This still touches every negative eigenvalue; what it can no longer
        # do is *decide* one.  Beneath the cutoff below -- now the eigensolver's
        # bar -- clipping to `0.0` drops the direction and so does comparing
        # `|w|`, so the outcome is the same on either sign, which is the whole
        # of issue #356.  The only place the clip changes an outcome is on a
        # negative eigenvalue *above* the bar, where the negativity is resolved
        # and clipping is Higham's projection onto the PSD cone -- exactly the
        # case projection is for.  See `_eigensolver_relative_bar`.
        eigenvalues = np.maximum(eigenvalues, 0.0)
        max_abs_eigenvalue = max_eigenvalue
    # `gram_rcond` is the normal-equation boundary; the bar is what `eigh`
    # can resolve.  A cut beneath the bar does not decide rank, it decides the
    # sign of round-off -- so the boundary is FLOORED at the bar rather than
    # replaced by it, and a coarser `gram_rcond` would still win.
    cutoff = (
        max(policy.gram_rcond, _eigensolver_relative_bar(len(active_columns))) * max_abs_eigenvalue
    )
    retained_mask = eigenvalues > cutoff if psd_semantics else np.abs(eigenvalues) > cutoff
    rank = int(np.count_nonzero(retained_mask))
    positive = np.abs(eigenvalues[np.abs(eigenvalues) > 0.0])
    condition = (
        float(np.sqrt(max_abs_eigenvalue / np.min(positive)))
        if positive.size and max_abs_eigenvalue > 0.0
        else float("inf")
    )

    if rank == len(active_columns) and np.all(eigenvalues > 0.0):
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            probe = np.arange(1.0, len(active_columns) + 1.0)
            solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
            residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                np.linalg.norm(probe), 1e-300
            )
            if residual <= residual_tol:
                log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                    np.sum(np.log(column_scale[active_columns]))
                )
                null = _null_basis(
                    width,
                    active_columns,
                    column_scale[active_columns],
                    np.zeros((len(active_columns), 0)),
                )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(active_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=rank < width,
                    used_svd_fallback=False,
                    resolution_limited=False,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(factor),
                    parameter_null_basis=_freeze(null),
                    structural_aliases=_freeze(structural_aliases, dtype=bool),
                    retained_values=_freeze(eigenvalues),
                )
        except (np.linalg.LinAlgError, ValueError):
            pass

    # Normal equations cannot distinguish an exact active-column alias from a
    # full-rank factor direction whose squared singular value rounded to zero.
    # Structural zero columns were removed above; every other PSD truncation
    # therefore needs observation-factor certification when one is available.
    #
    # Hoisted above the subspace construction on purpose: it reads only the
    # spectrum, and it is the last field the certification predicate needs.
    resolution_limited = bool(
        (psd_semantics and rank < len(active_columns))
        or np.any((np.abs(raw_eigenvalues) > 0.0) & ~retained_mask)
        or (fallback_factor is not None and decompose_factor(fallback_factor).rank > rank)
    )
    # Everything past this point -- two width-by-rank bases, the null basis,
    # the retained pseudo-determinant, the representative selection and its
    # Cholesky -- exists only to be read off the returned decomposition.  Both
    # returns below are reached with the ``rank``, ``width``,
    # ``pre_truncation_condition`` and ``resolution_limited`` computed above,
    # and neither reports ``qr_svd``, so the predicate settles here exactly as
    # it would on the finished object.  When it says the caller must certify
    # against the observation factor, none of that work can be read back.
    if omit_uncertifiable and _certification_required(
        method="gram_eigh",
        width=width,
        rank=rank,
        pre_truncation_condition=condition,
        resolution_limited=resolution_limited,
        policy=policy,
    ):
        return None
    retained_vectors = eigenvectors[:, retained_mask]
    discarded_vectors = eigenvectors[:, ~retained_mask]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    active_scale = column_scale[active_columns]
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = eigenvalues[retained_mask]
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    if psd_semantics and 0 < rank < len(active_columns):
        # Choose the earliest original-coordinate representative whose
        # principal system has the certified rank. This gives exact aliases a
        # reproducible zero coefficient while estimability still uses the true
        # spectral null space above.  The selection is read off the null basis
        # this decomposition has already computed -- `_earliest_representatives`
        # documents why eliminating it from the right gives the same columns as
        # walking prefixes, without an eigendecomposition per candidate, and
        # `_conditioned_representatives` documents when index order is too
        # expensive a convention to keep.
        selected_local_array = _conditioned_representatives(
            discarded_vectors,
            rank,
            block_condition=lambda keep: _principal_block_condition(equilibrated, keep),
            block_admissible=lambda keep, _amplification: _principal_block_clears_cutoff(
                equilibrated,
                keep,
                cutoff,
            ),
            maximum_condition=policy.severe_condition,
        )
        if selected_local_array is not None:
            representative_columns = active_columns[selected_local_array]
            representative = equilibrated[np.ix_(selected_local_array, selected_local_array)]
            try:
                representative_factor = scipy.linalg.cholesky(
                    representative, lower=True, check_finite=False
                )
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                return RankDecomposition(
                    policy_version=policy.version,
                    method="pivoted_cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=False,
                    resolution_limited=resolution_limited,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
    return RankDecomposition(
        policy_version=policy.version,
        method="gram_eigh",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=False,
        resolution_limited=resolution_limited,
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(structural_aliases, dtype=bool),
        retained_values=_freeze(retained_values),
    )


def decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
    allow_indefinite: bool = False,
) -> RankDecomposition:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix."""
    decomposition = _decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        fallback_factor=fallback_factor,
        allow_indefinite=allow_indefinite,
    )
    if decomposition is None:  # pragma: no cover - omit_uncertifiable defaults off
        raise RuntimeError("gram decomposition omitted its subspace without being asked to")
    return decomposition


def decompose_gram_if_authoritative(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
) -> RankDecomposition | None:
    """The Gram decomposition when it is authoritative, else ``None``.

    ``None`` means exactly what ``needs_factor_certification`` means on the
    eager result: this Gram cannot certify its own retained subspace, and the
    caller must go to the observation factor.  Callers that do nothing else in
    that case should prefer this to ``decompose_gram`` plus the predicate,
    because a Gram that is about to be superseded never builds the retained
    subspace, the null basis, the representative Cholesky or the retained
    pseudo-determinant that only the superseded object could have exposed.

    The predicate below is the authority, so the contract holds whatever the
    hint inside chooses to skip: the eager and deferring paths agree on every
    field that decides it, and a spared decomposition is one no caller in this
    shape could have read.
    """
    decomposition = _decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        fallback_factor=fallback_factor,
        omit_uncertifiable=True,
    )
    if decomposition is None or needs_factor_certification(decomposition, policy=policy):
        return None
    return decomposition


def decompose_symmetric(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
) -> RankDecomposition:
    """Decompose symmetric full-Newton curvature that may be indefinite."""
    return decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        allow_indefinite=True,
    )


def decompose_factor(
    factor: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    retain_factor_solve: bool = False,
) -> RankDecomposition:
    """Decompose a weighted/augmented factor using the factor-space rule."""
    factor = np.asarray(factor, dtype=float)
    if factor.ndim != 2 or not np.all(np.isfinite(factor)):
        raise ValueError("factor must be a finite matrix")
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active_columns = np.flatnonzero(column_scale > 0.0)
    if active_columns.size == 0:
        decomposition = decompose_gram(np.zeros((width, width)), policy=policy)
        if retain_factor_solve:
            decomposition = replace(
                decomposition,
                factor_rhs_left_basis=_freeze(np.zeros((factor.shape[0], 0))),
            )
        return decomposition
    active_scale = column_scale[active_columns]
    equilibrated = factor[:, active_columns] / active_scale
    # A tall observation factor needs only its thin left singular vectors;
    # requesting a full U would allocate O(n²) memory.  A wide factor still
    # needs full right vectors so exact row-rank null directions are retained.
    full_matrices = equilibrated.shape[0] < equilibrated.shape[1]
    left_vectors, singular_values, Vh = np.linalg.svd(
        equilibrated,
        full_matrices=full_matrices,
    )
    cutoff = policy.factor_rcond * singular_values[0]
    retained_mask = singular_values > cutoff
    rank = int(np.count_nonzero(retained_mask))
    retained_vectors = Vh[: len(singular_values), :].T[:, retained_mask]
    discarded_vectors = Vh.T[:, rank:]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = singular_values[retained_mask] ** 2
    factor_rhs_left_basis = None
    if retain_factor_solve:
        retained_left = left_vectors[:, : len(singular_values)][:, retained_mask]
        factor_rhs_left_basis = retained_left / singular_values[retained_mask]
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    if 0 < rank < len(active_columns):
        # Same certified representative choice as the Gram path, off the right
        # singular vectors that span this factor's null space.  Rank
        # admissibility, condition scoring and the stored solve MUST all stay
        # in these spectral coordinates.  Forming
        # ``equilibrated.T @ equilibrated`` accumulates over the observation
        # count: it can move a direction across the cutoff, corrupt the
        # condition comparison, and leave the result solving a different
        # matrix from the factor whose SVD certified it.
        spectral_factor = singular_values[:, None] * Vh[: len(singular_values), :]
        candidate_qr: dict[tuple[int, ...], tuple[NDArray, NDArray] | None] = {}

        def compact_qr(keep: NDArray) -> tuple[NDArray, NDArray] | None:
            key = tuple(int(index) for index in keep)
            if key not in candidate_qr:
                candidate_qr[key] = _compact_factor_qr(spectral_factor, keep)
            return candidate_qr[key]

        def compact_condition(keep: NDArray) -> float:
            geometry = compact_qr(keep)
            return float("inf") if geometry is None else _triangular_gram_condition(geometry[1])

        selected_local_array = _conditioned_representatives(
            discarded_vectors,
            rank,
            block_condition=compact_condition,
            block_admissible=lambda keep, amplification: _factor_representative_clears_cutoff(
                spectral_factor,
                keep,
                largest_singular_value=float(singular_values[0]),
                smallest_retained_singular_value=float(singular_values[rank - 1]),
                cutoff=float(cutoff),
                amplification=amplification,
            ),
            maximum_condition=policy.severe_condition,
        )
        if selected_local_array is not None:
            representative_columns = active_columns[selected_local_array]
            representative_geometry = compact_qr(selected_local_array)
            if representative_geometry is not None:
                compact_left, representative_upper = representative_geometry
                representative_factor = representative_upper.T
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                representative_rhs_left_basis = None
                representative_rhs_triangular = None
                if retain_factor_solve:
                    representative_rhs_left_basis = (
                        left_vectors[:, : len(singular_values)] @ compact_left
                    )
                    representative_rhs_triangular = (
                        representative_upper * column_scale[representative_columns][None, :]
                    )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="qr_svd",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=True,
                    resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                    factor_rhs_left_basis=(
                        None
                        if representative_rhs_left_basis is None
                        else _freeze(representative_rhs_left_basis)
                    ),
                    factor_rhs_triangular=(
                        None
                        if representative_rhs_triangular is None
                        else _freeze(representative_rhs_triangular)
                    ),
                )
    return RankDecomposition(
        policy_version=policy.version,
        method="qr_svd",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=True,
        resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(column_scale == 0.0, dtype=bool),
        retained_values=_freeze(retained_values),
        factor_rhs_left_basis=(
            None if factor_rhs_left_basis is None else _freeze(factor_rhs_left_basis)
        ),
    )


def selected_group_name_set(result, groups: Sequence, *, penalty=None) -> set[str]:
    """Return explicit solver selection, with a legacy coefficient fallback.

    Legacy results predate explicit rank/selection metadata.  When the fitted
    penalty is available, preserve every group that was not subject to a
    positive nonsmooth penalty; a valid zero estimate is not deselection.
    """
    if getattr(result, "rank_info", None) is not None:
        return set(result.rank_info.selected_group_names)
    if penalty is not None:
        from superglm.penalties.base import penalty_can_zero_groups, penalty_targets_group

        can_zero_groups = penalty_can_zero_groups(penalty)
        return {
            group.name
            for group in groups
            if not can_zero_groups
            or not penalty_targets_group(penalty, group)
            or np.linalg.norm(result.beta[group.sl]) > 1e-12
        }
    return {group.name for group in groups if np.linalg.norm(result.beta[group.sl]) > 1e-12}
