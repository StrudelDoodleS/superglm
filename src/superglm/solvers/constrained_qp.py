"""Active-set constrained penalized least-squares solver.

Solves:
    minimize   0.5 * beta^T H beta - g^T beta
    subject to A @ beta >= b

where H is positive semidefinite.  A rank-deficient H is rank-truncated through
the shared rank policy rather than regularized; a materially indefinite one is
rejected (see ``solve_constrained_qp``).

Uses a primal active-set method:
1. Start with a feasible point (project if needed).
2. Solve the equality-constrained subproblem on the active set.
3. If the step is feasible, check multipliers to drop constraints.
4. If infeasible, find the blocking constraint and add it.

Warm-starting: pass active_set_init from a previous solve to skip
discovery iterations (the active set usually stabilizes after a few
IRLS iterations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm.solvers.rank import _EPS, RankDecomposition, _symmetric_part, decompose_gram

# Headroom on the normal-equation consistency floor (see ``_consistency_floor``).
# The floor estimates the accuracy of the *computed null basis*, and this is the
# safety factor above that estimate.  It is deliberately a constant of its own
# rather than ``SHARED_RANK_POLICY.certification_band``, which happens to hold
# the same value but governs factor certification: retuning that band for its
# own purpose must not silently move this gate.  For the same reason the
# structural floor below carries its own literal rather than deriving from
# this one.
_NULL_BASIS_ACCURACY_SLACK = 32.0

# Dimension term for the spectral floor.  The retained-condition term above
# models error amplification, but ``eigh``'s own backward error grows with the
# problem's width and does not vanish as the retained block becomes perfectly
# conditioned -- so at low condition the condition term bottoms out *below* the
# eigensolver's own roundoff in the null basis and rejects consistent systems.
# Measured worst case across widths 2..24 and ranks 1..width-1 is 17.3 * eps
# per unit width; 512 leaves roughly 30x margin.  It is deliberately separate
# from the condition slack: at any condition worth worrying about the condition
# term dominates, so this cannot move detection at the high end.
_SPECTRAL_DIMENSION_SLACK = 512.0

# Consistency floor for the *structural* half of the null space.  Those basis
# vectors are exact unit vectors -- a structurally zero column gives ``H`` an
# identically zero row -- so no conditioning term belongs here.  The two floors
# model different errors and must move independently: the spectral slack above
# covers ``eps * kappa`` amplification of a *computed* basis, this one covers
# only rounding in the two vector norms that form the ratio.  They happen to
# hold the same numeric value; that coincidence is not a dependency, and
# deriving one from the other would re-create, one level down, exactly the
# coupling ``_NULL_BASIS_ACCURACY_SLACK`` exists to avoid.
_STRUCTURAL_NORM_ROUNDING_SLACK = 32.0
_STRUCTURAL_CONSISTENCY_FLOOR = _STRUCTURAL_NORM_ROUNDING_SLACK * _EPS


@dataclass
class QPResult:
    """Result of a constrained QP solve.

    ``converged`` means the full KKT certificate holds for ``beta``: the
    active-set loop reached its own termination test (a stationary step with
    no negative multiplier) *and* the certified candidate is feasible.  It is
    ``False`` whenever the solver did not complete that certificate, including
    when the loop exhausted ``max_iter`` or a stationary candidate failed its
    primal-feasibility check.  In the latter case a subsequent projection may
    make the returned ``beta`` feasible, but it does not re-establish
    stationarity or dual feasibility; projection can also fail to repair an
    infeasible system.  Thus ``converged=False`` does not imply that the
    returned ``beta`` is infeasible.  It is the best available point, not a
    certified solution.

    A mutually infeasible constraint system is one way for the candidate
    feasibility check to fail, but not the only one and not the common one:
    the loop can also stop at a stationary point on a *subset* active set with
    another row materially violated, on a system that has a feasible point.
    See the early return below for the measurement and projection.

    ``rank``, ``width``, ``method``, ``condition`` and ``used_svd_fallback``
    report the geometry of the ``H`` decomposition this solve actually ran on.
    They are read straight off the decomposition computed below -- attribute
    reads, no extra work -- and exist because the constrained branch of
    :func:`~superglm.solvers.irls_direct.fit_irls_direct` previously reported
    none of it, hardcoding a condition of ``0.0`` where the unconstrained
    branch forty lines above reads the real number.  ``0.0`` reads as
    "perfectly conditioned" to every consumer, so the one branch with the least
    visible linear algebra was also the one claiming the most certainty.

    ``condition`` is ``pre_truncation_condition`` -- deliberately the same
    quantity the unconstrained branch publishes, so the two are comparable.
    The defaults describe "no decomposition was reached".
    """

    beta: NDArray
    active_set: list[int] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True
    rank: int = -1
    width: int = -1
    method: str = ""
    condition: float = 0.0
    used_svd_fallback: bool = False


def _null_space_mass(decomposition: RankDecomposition, g: NDArray) -> tuple[float, float]:
    """Split ``g``'s share of ``null(H)`` into its structural and spectral parts.

    ``g`` is in ``range(H)`` -- the normal equations are consistent -- exactly
    when it is orthogonal to ``null(H)``, so project it onto the null basis the
    decomposition already computed.  This measures the inconsistency directly
    instead of inferring it from a solve residual: a residual is amplified by
    the retained condition number, which at the conditions the rank policy
    happily retains swamps the signal entirely.

    The two halves must not share a threshold, because they are not known to
    the same accuracy.  ``rank._null_basis`` stacks

    * one **exact unit vector** per structurally zero column (``column_scale``
      is 0 there, so ``H`` has an identically zero row and column), and
    * the **discarded spectral directions** ``D^-1 @ V_discarded``, whose
      computed accuracy degrades as ``eps * retained condition``.

    They have disjoint row supports -- the unit vectors live on the unscaled
    columns, the spectral directions on the scaled ones -- so the split is an
    orthogonal decomposition of ``null(H)`` and the two masses can be measured
    and thresholded independently. Judging the exact half against the spectral
    half's floor is what let an ill-conditioned retained block desensitize a
    structural alias where detection is exact.

    Both masses are ratios of two norms of the *same* vector, so they are
    scale-free in exact arithmetic -- but not in floating point, because
    ``np.linalg.norm`` forms ``sqrt(x.dot(x))`` and the squaring leaves the
    representable range an octave early.  Measured on ``[0, x]``: ``x.dot(x)``
    is exactly ``0`` from ``x = 1e-170`` down and exactly ``inf`` from
    ``x = 1e155`` up, while ``np.hypot`` returns the true magnitude at both --
    so the squaring is the site, not the ``sqrt``.  **Both** norms in the ratio
    are affected, not just the denominator, and the two regimes bypass the
    breach test differently:

    * underflow -- numerator and denominator both ``0``, the denominator
      clamped up to ``tiny``, so the ratio is ``0.0`` and ``mass > floor`` is
      simply false;
    * overflow -- both ``inf``, so the ratio is ``nan`` and ``nan > floor`` is
      false as well.

    An ``inf`` ratio is not reachable: the numerator is a norm of a subvector
    of the same ``g``, so it can never exceed the denominator.  Measured
    end-to-end, ``H = diag(1, 0)`` with ``g = (0, 1e-200)`` and with
    ``g = (0, 1e200)`` both returned ``beta = [0, 0]`` with ``converged=True``
    where ``g = (0, 1)`` correctly raises; the spectral half fails the same
    way, returning ``max|beta| = 2.4e202`` at ``g`` scaled by ``1e200``.

    Rescaling by the inf-norm first puts every norm in ``[0.5, sqrt(width)]``,
    which no representable ``g`` can push out of range.  The rescaling is by a
    **power of two** (``frexp``/``ldexp``) rather than by the inf-norm itself,
    which makes it exact: every partial sum inside both norms is then exactly
    the unscaled one times ``2**-exponent``, and a quotient of two exactly
    scaled operands rounds identically to the unscaled quotient.  That is what
    keeps this bitwise inert on everything that was already in range.

    ``g == 0`` is handled before the rescaling -- there is no exponent to take
    -- and returns zero mass, which is what the old ``max(norm, tiny)`` clamp
    produced for it too.  With the exact-zero case split out, that clamp has no
    remaining job: after rescaling the denominator is at least ``0.5``.

    Returns ``(structural_mass, spectral_mass)``, each as a share of ``||g||``.
    """
    largest = float(np.max(np.abs(g), initial=0.0))
    if largest == 0.0:
        return 0.0, 0.0
    g = np.ldexp(g, -int(np.frexp(largest)[1]))

    reference = float(np.linalg.norm(g))
    structural = decomposition.column_scale == 0.0
    structural_mass = float(np.linalg.norm(g[structural])) / reference

    null_basis = decomposition.null_basis()
    scaled = ~structural
    spectral_mass = 0.0
    # Restricting to the scaled rows annihilates the unit-vector columns and
    # leaves exactly the spectral ones.  This also covers both empty cases on
    # its own: with no null columns the column norms are an empty array, and
    # with no scaled columns they are all zero, so ``retained_columns`` is
    # all-False either way.
    spectral_basis = null_basis[scaled, :]
    retained_columns = np.linalg.norm(spectral_basis, axis=0) > 0.0
    if np.any(retained_columns):
        orthonormal, _ = np.linalg.qr(spectral_basis[:, retained_columns])
        spectral_mass = float(np.linalg.norm(orthonormal.T @ g[scaled])) / reference
    return structural_mass, spectral_mass


def _consistency_floor(decomposition: RankDecomposition) -> float:
    """Largest *spectral* null mass the decomposition's own roundoff can manufacture.

    This governs only the spectral half of the split in ``_null_space_mass``.
    The structural half is measured against ``_STRUCTURAL_CONSISTENCY_FLOOR``,
    because a structurally zero column's null vector is an exact unit vector
    and carries none of the error this floor models.

    The computed spectral null basis is accurate only to about ``eps`` times
    the retained condition number, so a genuinely consistent ``g`` still
    projects onto it by roughly that much.  Scaling the threshold with the
    retained conditioning -- rather than fixing it at ``factor_rcond`` -- is
    what keeps a well-posed but ill-conditioned rank-deficient system solvable,
    since the rank policy truncates at ``max(gram_rcond, n eps)`` -- version 3
    floors it at the eigensolver's bar, issue #356 -- and so still retains
    blocks far more ill-conditioned than ``factor_rcond`` would tolerate.  The
    ceiling on a retained condition tightened by exactly ``n`` when that floor
    landed, from ``1/eps`` to ``1/(n eps)``; the table below is unaffected
    because its largest entry is ``1e12``.

    Sensitivity degrades *gradually* as the retained conditioning grows; there
    is no sharp cutoff.  Measured detection of an injected **spectral** null
    component, by retained condition:

    ==============  ============  ===============  =================
    retained cond   median floor  1% mass caught   0.1% mass caught
    ==============  ============  ===============  =================
    ``1e9``         ``6.8e-06``   120/120          120/120
    ``1e10``        ``6.8e-05``   120/120          120/120
    ``1e11``        ``6.8e-04``   118/118          102/118
    ``1e12``        ``6.9e-03``   103/119            8/119
    ``1e13``        ``6.8e-02``     8/120            1/120
    ==============  ============  ===============  =================

    So a 0.1% spectral inconsistency is already partly missed at ``1e11`` and a
    1% one at ``1e12``, well below the ``~1e14`` at which the floor saturates
    at 1 and nothing spectral is detectable at all.  This is a real resolution
    limit rather than a tuning choice: at those conditions the spectral null
    basis is itself only accurate to the floor, and along such a direction the
    objective moves only in the fifth significant figure over a coefficient
    range of ``1e9``.  It fails open -- the solve proceeds rather than refusing
    a system it cannot adjudicate -- which is the safe direction, since
    refusing a solvable system is the defect this floor exists to prevent.

    Two conventions from ``rank.py`` are load-bearing here and are easy to
    confuse:

    * The condition number computed below is on the **Gram** scale, a ratio of
      eigenvalues of the equilibrated Gram matrix.  ``RankDecomposition``'s own
      ``pre_truncation_condition`` is on the **factor** scale -- the square root
      of that ratio.  Do not substitute one for the other; they differ by a
      squaring, which at these magnitudes is the whole gate.
    * ``retained_values is None`` is read as "nothing spectral was dropped".
      That inference holds because only the eigen paths populate the field, but
      it is *not* uniform across ``rank.py``'s two ``method="cholesky"``
      branches: the fast pre-eigendecomposition branch leaves it ``None``,
      while the post-eigendecomposition one sets it to the full spectrum.  Both
      are full rank on the active columns, so both mean the same thing here --
      but a future branch that leaves the field unset after truncating would
      silently collapse this floor to ``32 * eps``.
    """
    retained = decomposition.retained_values
    if retained is None or retained.size == 0:
        # Cholesky without truncated spectrum: nothing spectral was dropped.
        retained_condition = 1.0
    else:
        magnitudes = np.abs(retained)
        smallest = float(np.min(magnitudes))
        retained_condition = (
            float(np.max(magnitudes)) / smallest if smallest > 0.0 else float("inf")
        )
    return float(
        min(
            1.0,
            _EPS
            * max(
                _NULL_BASIS_ACCURACY_SLACK * max(1.0, retained_condition),
                _SPECTRAL_DIMENSION_SLACK * decomposition.width,
            ),
        )
    )


def _feasibility_slack(
    A: NDArray,
    beta: NDArray,
    b: NDArray,
    *,
    abs_b: NDArray | None = None,
    abs_A: NDArray | None = None,
) -> NDArray:
    """Return ``A @ beta - b`` measured against a scale-aware tolerance.

    A step that lands *on* a constraint reproduces ``b_i`` only to within the
    error of the dot product that computed it, so a fixed absolute tolerance
    turns a genuine KKT point into a violation as soon as the constraint row is
    large.  Dividing by a per-row scale keeps the test meaningful under
    rescaling, and is identical to the absolute test for the well-scaled
    problems where that scale is 1.

    **The scale is the sum of absolute terms, not the magnitude of their sum.**
    The standard bound is ``|fl(x'y) - x'y| <= gamma_n sum_i |x_i y_i|`` --
    Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM
    2002), sec. 3.1 -- and the two differ by exactly the dot product's
    cancellation.  Using ``|A_i @ beta|`` was reading the OUTPUT where the error
    is set by the INPUTS, so a row that cancels reported a scale far under the
    accuracy it actually had, and the test on it silently became absolute via
    the ``max(1, ...)`` floor.  Issue #359: a constraint row cancelling by
    ``1.1e17`` -- output ``3.5e-13`` against ``|A_i| @ |beta| = 3.9e+04`` --
    read a scale of ``1.0`` and refused a point whose violation was ``0.0044``
    of its own dot-product bound.

    ``|A_i| @ |beta| >= |A_i @ beta|`` by the triangle inequality, so this scale
    always dominates the previous one and every bound stated against that one
    still holds: the normalized slack moves strictly toward zero, never away.

    Returns the slack already divided by its per-row scale, so callers can
    compare it against a bare ``-tol``.

    ``abs_b`` and ``abs_A`` let a caller in a loop pass ``np.abs(b)`` and
    ``np.abs(A)`` once instead of paying for them every sweep; they must equal
    those values.
    """
    products = A @ beta
    magnitude = (np.abs(A) if abs_A is None else abs_A) @ np.abs(beta)
    return (products - b) / _feasibility_scale(products, b, abs_b=abs_b, abs_products=magnitude)


def _feasibility_scale(
    products: NDArray,
    b: NDArray,
    *,
    abs_b: NDArray | None = None,
    abs_products: NDArray | None = None,
) -> NDArray:
    """Per-row scale for the relative feasibility test: ``max(1, |b|, |A| @ |beta|)``.

    Exposed separately so the active-set loop can divide *both* its slack and
    its directional derivative by the same factor.  Scaling both leaves the
    step ratio ``slack / -a_step`` numerically unchanged while making the
    decisions built on them -- "is this row already satisfied", "does this step
    move the row at all" -- agree with ``_is_feasible``.

    ``abs_products`` is ``|A| @ |beta|``, the dot-product error scale
    :func:`_feasibility_slack` documents.  **Every production call site passes
    it, and the fallback exists for tests that deliberately assert the
    pre-#359 scale.**  Omitting it silently restores exactly the behaviour
    #359 removed -- it is a lower bound on the correct scale, so it is
    conservative and can only refuse a point the full scale would accept -- but
    conservative here means "reintroduces the defect", so a new caller that
    reaches this branch is a mistake rather than a trade-off.
    """
    magnitude = np.abs(products) if abs_products is None else abs_products
    # ``|A| @ |beta|`` can overflow to ``inf`` where the signed ``A @ beta``
    # stays finite -- perfect cancellation of terms near the float ceiling.  An
    # infinite scale makes every slack ``-0.0`` and declares the row feasible
    # whatever it is violated by, so the scale falls back to the row's own
    # magnitude there.  Not reachable in-tree (it needs terms at ~1e308 and the
    # largest drift this module has measured is 4e43), and the pre-#359 scale
    # had no such branch because ``|A @ beta|`` finite meant a finite scale.
    magnitude = np.where(np.isfinite(magnitude), magnitude, np.abs(products))
    return np.maximum(1.0, np.maximum(np.abs(b) if abs_b is None else abs_b, magnitude))


def _is_feasible(
    A: NDArray, beta: NDArray, b: NDArray, tol: float, *, abs_A: NDArray | None = None
) -> bool:
    """Whether ``beta`` satisfies ``A @ beta >= b`` to a scale-aware tolerance.

    ``abs_A`` is the loop-invariant ``np.abs(A)``.  The scale needs it on every
    call, and rebuilding it here is pure waste on the paths that already hold
    it -- the active-set loop, and ``irls_direct``, which tests the aggregate
    constraint system two to three times per IRLS iteration plus once per
    line-search halving.
    """
    return bool(np.all(_feasibility_slack(A, beta, b, abs_A=abs_A) >= -tol))


def _solve_saddle_least_squares(KKT: NDArray, rhs: NDArray) -> NDArray:
    """Least-squares solve of a saddle system, symmetrically equilibrated first.

    ``lstsq``'s rank cutoff is *relative to the largest singular value of the
    matrix it is handed*, so on an unscaled saddle matrix it measures the
    constraint block against the norm of ``H``.  When ``H`` dwarfs the
    constraint rows, every constraint direction falls below the cutoff and is
    discarded as noise -- the solve then returns a point that ignores the
    constraints entirely.  Measured on ``H = diag(1e16, 0)`` with
    ``A = [[1, 1]]``: singular values ``[1e16, 1, 1]``, cutoff
    ``3 * eps * 1e16 = 6.66``, so **both** unit values are truncated and rank 1
    of 3 is retained, for an infeasible answer.  The matrix is nonsingular; only
    the tolerance was wrong.  (``np.linalg.matrix_rank`` reports 1 here too, for
    the same reason -- it is not an independent check.)

    Equilibrating symmetrically -- ``D K D`` with ``D = diag(1 / sqrt(row
    inf-norm))``, solving for ``y`` and returning ``D y`` -- puts every row and
    column on a common scale first, so the cutoff separates directions by
    *conditioning* rather than by which block they came from.  On the case
    above it recovers singular values ``[1, 1, 1]``, rank 3 of 3, and the
    feasible optimum.

    Symmetry of the scaling is load-bearing rather than stylistic.  ``KKT`` is
    only quasi-symmetric -- the off-diagonal blocks are ``-A^T`` and ``+A`` --
    but ``|KKT|`` *is* symmetric, so a scaling built from row inf-norms is
    automatically the same on both sides and ``D K D`` preserves the saddle
    structure the multiplier block relies on.  A one-sided row scaling would
    not, and would leave the returned multipliers on a different scale from the
    step.

    Two properties make the scaling safe on a degenerate block.  Scope them
    carefully: the scale multiplies **three** quantities -- the matrix, the
    right-hand side ``rhs * scale``, and the unscaling ``sol * scale`` -- and
    what is proved below covers only the first.  The equilibrated *matrix
    entries* cannot exceed 1; the other two products carry no such envelope and
    can overflow.  Measured on ``K = diag(1, 1e-320, 1)`` with
    ``rhs = (1, 1e200, 1)``: every equilibrated matrix entry is finite with
    maximum exactly ``1.0``, ``rhs * scale`` overflows to ``inf``, and the
    returned solution is all ``nan``.  That input is far outside anything this
    module assembles -- ``decompose_gram`` refuses a non-finite ``H`` upstream,
    and the smallest nonzero in-tree row inf-norm is of order the constraint
    rows -- but the bound below is not what rules it out, and the Ruiz
    comparison in the second bullet is closer than "provable versus empirical"
    reads, because the provable side covers one product of three.

    * **No zero divide.** A structurally empty row -- an ``H`` row that is zero
      with a matching zero ``A`` column, or an all-zero constraint row -- has
      inf-norm 0 and keeps scale ``1.0`` rather than being sent through
      ``1 / sqrt(0)``.  Leaving it alone is
      correct: there is nothing in that row to normalize, and because ``|KKT|``
      is symmetric the matching column is zero as well, so the row and column
      stay zero and ``lstsq`` discards the direction as it should.  Clamping to
      ``tiny`` instead would manufacture a ``6.7e153`` scale for a row that
      carries no information.
    * **No overflow in the matrix.** For a nonzero row, ``|K[i, j]| <=
      min(m_i, m_j) <= sqrt(m_i * m_j)``, so every equilibrated matrix entry
      satisfies
      ``|K[i, j]| / sqrt(m_i * m_j) <= 1`` by construction, whatever the
      dynamic range of the input.  The bound is exact in real arithmetic and
      holds to a couple of ulps once the two multiplies round, which is what
      the overflow argument needs.  A single pass is what carries this
      guarantee: iterating to the Ruiz fixed point bounds the accumulated
      scale only by the input's dynamic range rather than by ``1/sqrt(m_min)``,
      trading a provable envelope for an empirical one.  Measured over a
      3950-case rank-deficient ensemble the extra passes repair 14 infeasible
      answers against this pass's 7, for the same 2 regressions -- real, but
      not worth the weaker guarantee on this path.  Those 2 are 2 *new
      silently-infeasible answers* against 5 net repairs, inside a population
      this branch created: on ``master`` a singular ``H`` raised
      ``LinAlgError`` before the loop ever ran, so the baseline they regress
      against is a loud refusal, not a feasible answer.

    The solution is minimum-norm **in the equilibrated coordinates**, not in
    the original ones, because that is where ``lstsq`` resolves the null
    directions.  ``RankDecomposition.solve`` follows the same *convention* for
    the pure-``H`` solve -- divide by a scale, solve, divide again -- but not
    the same *scale*: it divides by ``sqrt(diag(H))`` while this divides by
    ``sqrt(row inf-norm of K)``.  So the two solves do not structurally pick
    the same representative on a flat optimal face, and nothing here makes them
    agree; what is claimed is only that they have not been measured to
    disagree in a way that matters.  Where the face is genuinely flat the point
    moves but the objective does not: the rank-one regression below returns
    ``max|beta| = 18.3`` rather than ``15.7`` for the same exact optimum of
    ``-450``.

    The 2 ensemble cases that regress from feasible to infeasible were read as
    that same benign movement, on the strength of a neutral control: rescaling
    ``K`` and ``rhs`` by a common constant rerouted 2607 answers without
    flipping any feasibility outcome.  **The control cannot carry that
    conclusion.**  ``sigma(cK) = c sigma(K)`` and ``rcond`` is relative, so the
    control leaves the retained set identical *by construction* -- measured, it
    changes ``lstsq``'s retained rank in **0 of 110725** rank-deficient KKT
    solves.  It isolates bit noise and nothing else, and cannot tell a
    different minimum-norm representative (benign) from a different retained
    rank (not).  Equilibration changes exactly the latter: across the same
    solves it moves the retained rank on **9.3%** of them.

    Measured directly instead, on the arm-versus-arm rank rather than the
    control: over a production-shaped reconstruction (``b = 0``, structured
    ``A``) the feasibility regressions carry **0 of 2319** KKT solves with a
    changed retained rank, which does support the benign reading there.  Over
    an adversarial reconstruction **196 of 2455 (8.0%)** do.  So the benign
    reading is a property of the in-tree-shaped population, not a structural
    fact about equilibration, and it is the arm-versus-arm rank -- not the
    control -- that establishes it.

    ``rcond`` is deliberately left at ``None``.  The competing proposal is to
    pass ``SHARED_RANK_POLICY.gram_rcond`` so that one rule decides what is
    retained, since the policy keeps ``H`` eigenvalues down to
    ``max(gram_rcond, n eps) * lambda_max`` while ``lstsq`` drops singular
    values below ``max(M, N) * eps * sigma_max``.  **Version 3 (issue #356)
    strengthens that proposal without settling it.**  It gave the two rules the
    same shape -- both are now ``p(dimension) * eps * scale``, where the Gram
    side used to be a bare ``eps`` -- so the constant is no longer the
    difference between them.  What remains is that they are taken on different
    matrices, the ``H`` Gram here and the KKT system there, and the second
    bullet below measures exactly that difference.  Equilibration is *measured*
    to settle it better -- there is no structural argument that it must,
    and the two bullets below are the whole of the evidence, so weakening
    either weakens the decline:

    * A direction that is near-null for ``H`` but pinned by an active
      constraint keeps a KKT singular value of order that constraint row's
      norm, not of order its ``H`` eigenvalue -- a KKT singular value is not an
      ``H`` eigenvalue.  Over a ``p = 180`` ensemble with 50-row active sets and
      an ``H`` eigenvalue planted at ``50 * eps * lambda_max``, the smallest
      retained equilibrated ratio measured ``1.3e-4`` and the largest truncated
      one ``0``; nothing landed between ``eps`` and ``230 * eps``, so the two
      candidate cutoffs cannot disagree there.  Under version 3 that plant sits
      *below* the ``180 eps`` cutoff rather than inside the retention band, so
      the policy now truncates the direction where it used to keep it.  The
      measurement survives that: what the ensemble established is a property of
      the KKT singular values -- an active constraint pins such a direction at
      the constraint row's scale -- and that is unchanged by where the Gram
      cuts.  The ensemble has not been re-run with the plant moved above
      ``180 eps``, and nothing is claimed about that case.
    * Where the active rows leave such a direction *free*, it really does fall
      in the band -- and an explicit ``rcond`` is not enough to save it.  With
      ``H = diag(1, eps, 0)`` and ``A_eq = [[1, 0, 0]]`` the raw ratio is
      ``1.4e-16``, below ``gram_rcond`` as well as below ``4 * eps``, so
      ``rcond=gram_rcond`` still discards it.  Equilibration lifts the same
      ratio to ``0.38``.  See the regression test.

    Lowering the cutoff would also narrow the margin protecting the genuine
    null direction -- on the rank-one regression below it sits at ``2e-17``
    against ``8.9e-16``, and retaining it instead is what produced the ``1e43``
    drift this branch already fixed once.
    """
    row_norm = np.abs(KKT).max(axis=1)
    nonzero = row_norm > 0.0
    scale = np.where(nonzero, 1.0 / np.sqrt(np.where(nonzero, row_norm, 1.0)), 1.0)
    equilibrated = KKT * scale[:, None] * scale[None, :]
    return np.linalg.lstsq(equilibrated, rhs * scale, rcond=None)[0] * scale


def _project_feasible(beta: NDArray, A: NDArray, b: NDArray, tol: float) -> NDArray:
    """Project beta onto the feasible set {x : A @ x >= b}.

    Uses iterative constraint-by-constraint projection (Dykstra-like).

    Each sweep repairs only the single worst violation, so the 100-sweep
    budget can be exhausted with the point still infeasible -- either because
    the constraints are mutually infeasible, or merely because there are more
    violated constraints than sweeps.  Those two cases are not distinguishable
    here and the active-set loop often recovers from the second, so the caller
    must test the feasibility of the point it finally returns rather than
    treating the starting point's status as the answer.

    Uses the same scale-aware stopping test as the caller's convergence check,
    so the two cannot disagree about what "feasible" means -- but takes the
    *selection* from the raw violations.  Those are two different orderings and
    the docstring below says where they part.

    **Issue #359 changed the scale under this argument, and the paragraphs it
    replaced are worth stating because the new behaviour is the point.**  The
    per-row scale was ``max(1, |b|, |A @ beta|)``, so at ``b = 0`` the slack was
    the clamp ``x / max(1, |x|)`` -- a monotone nondecreasing function of the
    raw violation.  That made the raw ``argmin`` attain the minimum scaled slack
    as well, and made ``slack.min() >= -tol`` the same predicate as
    ``violations.min() >= -tol`` for ``tol`` in ``(0, 1)``.  It is now
    ``max(1, |b|, |A| @ |beta|)``, the dot-product error scale, and neither
    property survives:

    * The scale no longer depends on the violation's own magnitude, so it is
      **not** a clamp and the two orderings can genuinely disagree.  The sweep
      still repairs the raw-worst row, which is always a real violation and
      always a valid repair -- but it is no longer guaranteed to be the row
      with the worst *scaled* slack.  The loop re-tests the scaled slack after
      every sweep, so this changes which path it takes, not what it accepts.
    * The stopping test is strictly **weaker** than the raw one, because
      ``|A| @ |beta| >= |A @ beta|`` makes the new scale dominate the old and
      every normalized violation move toward zero.  That is the fix, not a
      side effect: a row whose dot product cancels had its accuracy read off
      the cancelled output, so the old test refused points that were feasible
      to every digit the arithmetic had.

    So this is **no longer bitwise ``master``'s projection**, and the table that
    argued it was has been removed rather than qualified.  What still holds
    term for term is the repair body, and ``products - b`` at ``b = 0`` being
    bitwise ``products``, signed zeros included.

    The ``tol`` domain still matters here for the reason it always did: at
    ``tol >= 1`` the scaled test accepts violations the raw one rejects, which
    is the vacuity the boundary check exists to refuse.
    ``test_the_projection_stops_no_later_than_the_absolute_predicate`` pins what
    survives -- the direction of the change -- against a hand-written reference
    rather than against a recorded number.

    Plain raw violation rather than raw over row norm, though the latter is the
    true Euclidean distance to the hyperplane (the sweep moves ``|violation| /
    ||a||``).  Three reasons, in order of weight.  It is what ``master``
    selected on, so this stays a repair of a defect this branch introduced
    rather than a new selection policy smuggled into a patch.  It is far
    narrower: over 480 constrained fits / 1285 projections / 24300 sweeps the
    raw and clamped orders differ on 4 sweeps, while raw and row-normalized
    differ on 3022 -- in-tree rows are ``D @ P``, not ``D``, with norms
    spanning 0.039 to 0.594, so row normalization would reroute 12% of all
    sweeps to repair 0.016% of them.  And an all-zero constraint row divides
    0 by 0 under normalization, which ``argmin`` then selects, where the raw
    order never picks a row whose *raw* violation is not the worst.

    That last clause is about raw violation only, not about the shared
    predicate: the two come apart whenever a row's scale differs from its
    violation.  With ``b = (0, 1000)``, ``beta = (-0.5, 999)`` and
    ``tol = 0.01``, row 1 is the worse raw violation (``-1`` against ``-0.5``)
    yet is already satisfied against its row scale of 1000, so the sweep spends
    budget repairing a row ``_is_feasible`` accepts.

    **Since #359 that case is reachable at ``b = 0`` too, which is every
    in-tree caller, and the previous wording -- "unreachable at ``b = 0``,
    where the two orders coincide" -- is exactly what this change reverses.**
    The scale no longer depends on the violation, so a row violated worst can
    carry the largest scale and be accepted while a less-violated row with a
    unit scale is not; ``TestProjectionSelectsTheWorstViolation`` demonstrates
    it with ``b = 0``.  Measured over the monotone and constraint fit suites --
    8697 constraint rows -- **46% carry a scale above 1**, worst 23.3, so this
    is an ordinary path rather than a corner.

    It stays self-limiting: the stopping test is the scaled one, so the sweep
    exits as soon as every row is satisfied, the selection is unchanged from
    before this branch, and the repair body never reads the scale -- so the
    iterates are what they were and only the ``break`` moved.
    ``test_the_stopping_test_still_means_every_row`` pins that behaviour.
    """
    beta = beta.copy()
    # Loop-invariant: only ``A @ beta`` changes between sweeps.
    abs_b = np.abs(b)
    abs_A = np.abs(A)
    for _ in range(100):
        # Inlined rather than calling ``_feasibility_slack``, which would
        # recompute the matvec; the arithmetic is identical term for term.
        products = A @ beta
        violations = products - b
        slack = violations / _feasibility_scale(
            products, b, abs_b=abs_b, abs_products=abs_A @ np.abs(beta)
        )
        if slack.min() >= -tol:
            break
        worst = int(np.argmin(violations))
        # Project onto the violated constraint: a^T x >= b_i
        a = A[worst]
        deficit = b[worst] - a @ beta
        beta += deficit / (a @ a) * a
    return beta


BLOCKING_TRACE_CHANNEL = "constrained_qp_blocking"


def _emit_blocking_decision(
    trace_run: TraceRun,
    *,
    iteration: int,
    A: NDArray,
    b: NDArray,
    beta: NDArray,
    beta_new: NDArray,
    tol: float,
    products: NDArray,
    raw_step: NDArray,
    active_lookup: set[int],
    blocking: int,
    alpha_min: float,
) -> None:
    """Record one blocking-constraint decision as a ``step_decision`` event.

    Deliberately **re-derives** everything it records from the raw inputs
    rather than echoing the loop's own booleans or reusing its scaled arrays.
    A hook fed the loop's gating arithmetic would agree with that arithmetic by
    construction and so could never witness the gate being wrong, which is the
    whole reason the decision trace exists.  In particular ``considered_rows``
    is computed from a scale this function derives itself, so a loop that
    gated on the unscaled derivative records a ``blocking_row`` that is *not*
    in its own considered set.
    """
    assert trace_run is not None  # narrowed by the caller's `tracing` guard
    derived_scale = _feasibility_scale(products, b, abs_products=np.abs(A) @ np.abs(beta))
    derived_scaled_step = raw_step / derived_scale
    considered = [
        index
        for index in range(A.shape[0])
        if index not in active_lookup and derived_scaled_step[index] < -tol
    ]
    scaled_slack = _feasibility_slack(A, beta, b)
    trace_run.emit_lazy(
        "step_decision",
        lambda: {
            "iteration": iteration,
            # Independently evaluated: the loop must reach this block only for
            # a step the convergence test rejects.
            "full_step_is_feasible": _is_feasible(A, beta_new, b, tol),
            "considered_rows": tuple(considered),
            # The raw inputs the ratio is defined on, so a reader can recompute
            # alpha exactly rather than trust the recorded value.
            "row_products": tuple(float(products[i]) for i in considered),
            "row_b": tuple(float(b[i]) for i in considered),
            "row_raw_step": tuple(float(raw_step[i]) for i in considered),
            "row_scaled_slack": tuple(float(scaled_slack[i]) for i in considered),
            # The per-row scale the slack was divided by.  Recorded because it
            # is no longer recoverable from `row_products` and `row_b`: since
            # #359 it is `max(1, |b|, |A_i| @ |beta|)`, which depends on the
            # row's inputs rather than on the product they produced, so a
            # reader cannot rebuild it from the recorded output alone.
            "row_scale": tuple(float(derived_scale[i]) for i in considered),
            "blocking_row": int(blocking),
            # Independently derived: a row the scaled gate excludes must never
            # be the one blocked on.
            "blocking_is_considered": blocking < 0 or blocking in considered,
            "blocking_scaled_step": (
                float(derived_scaled_step[blocking]) if blocking >= 0 else 0.0
            ),
            "alpha": float(alpha_min),
        },
        channel=BLOCKING_TRACE_CHANNEL,
        # Fixed rather than taking the caller's ``trace_purpose`` as the
        # sibling emit sites do: this seam is test-only and no caller threads a
        # purpose through yet.  Thread one here if that changes.
        purpose="constrained_qp",
    )


def solve_constrained_qp(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
    *,
    _trace_run: TraceRun | None = None,
) -> QPResult:
    """Solve a convex QP with linear inequality constraints.

    Parameters
    ----------
    H : (p, p) NDArray
        Positive semidefinite Hessian. It is decomposed once through the
        shared rank policy, so a rank-deficient H is truncated rather than
        raising, provided g lies in ``range(H)``.

        Three inputs raise ``ValueError`` rather than returning a plausible
        wrong answer: a materially indefinite H (the problem is then not the
        convex QP this solver assumes); a rank-deficient H whose g has a
        component outside ``range(H)`` (the objective is unbounded below along
        a null direction, and no search direction this method forms can follow
        it); and an H the rank policy cannot equilibrate.

        H is symmetrized once as ``0.5 * (H + H.T)`` and that symmetric part is
        used throughout -- decomposition, KKT blocks, residual and multiplier
        test alike -- so an asymmetric H is solved consistently as its
        symmetric part rather than as two different quadratics on the two
        paths. For an exactly symmetric H whose entries are normal the
        symmetrization is bitwise identity; see ``_symmetric_part`` for the
        overflow branch and for the subnormal case where it is not. Every
        in-tree caller builds H symmetric by construction (``XtWX + S``,
        ``X'X + lambda*P``).
    g : (p,) NDArray
        Linear term (gradient at zero, with sign: objective is
        0.5 * beta^T H beta - g^T beta).
    A : (m, p) NDArray
        Constraint matrix. Constraints are A @ beta >= b.
    b : (m,) NDArray
        Constraint right-hand side.
    active_set_init : list[int] | None
        Warm-start active set from previous solve.
    max_iter : int
        Maximum active-set iterations.
    tol : float
        Tolerance for constraint satisfaction and multiplier signs, required
        to lie in ``(0, 1)``; anything else raises ``ValueError``. The
        constraint test is relative: row ``i`` is satisfied when
        ``A_i @ beta - b_i >= -tol * max(1, |b_i|, |A_i| @ |beta|)``, so a
        badly scaled constraint system does not read as infeasible purely
        because its rows are large. The scale is the **sum of absolute terms**
        of the dot product rather than the magnitude of their sum, which is the
        error bound that dot product actually satisfies; see
        :func:`_feasibility_slack` and issue #359. At ``b_i == 0`` the test is
        no longer algebraically identical to the absolute one -- it is strictly
        weaker by the row's cancellation -- and every in-tree caller passes
        ``b = 0``, so that is the case it changed.

        ``(0, 1)`` is the predicate's actual domain, not a house style, and the
        scale change strengthens the argument rather than weakening it. The
        normalized slack still cannot exceed 1 in magnitude, now for a reason
        that needs no ``max(1, .)`` analysis: ``|A_i @ beta| <= |A_i| @ |beta|``
        by the triangle inequality, so the numerator is bounded by its own
        denominator at ``b = 0``. At ``tol >= 1`` the test is
        ``x >= -tol * scale`` with ``scale >= |x|``, i.e. ``x >= -|x|``, which
        every violation satisfies -- so the solve returns its unconstrained
        answer with ``converged=True``. Measured on ``H = [[1]]``,
        ``g = [-100]``, ``beta >= 0``: ``tol = 0.999999`` returns ``beta = 0``,
        ``tol = 1.0`` returns ``beta = -100`` and calls it converged. That
        fixture is a single row with one term, where cancellation is impossible
        and the two scales coincide, so the measurement is unaffected by the
        change it is quoted beneath.

        Rejecting the value rather than reformulating the predicate is
        deliberate. The vacuity is algebraic, not a rounding artifact: written
        without the division, the test is ``x >= -tol * max(1, |x|)``, and at
        ``tol = 1`` that is ``x >= -|x|``, which holds for every ``x <= 0``
        however it is spelled. The only formulation that cannot saturate is one
        whose scale excludes the row's own magnitude -- which is precisely the
        term this test grew in order to stop reading an exactly-active large
        constraint row as violated. So a non-vacuous formulation is not a
        reformulation but a revert. Validation is also honest about the other
        two jobs this one parameter does: it is the component-scaled step-norm
        convergence threshold
        (``||step / max(1, |beta|)|| < tol``) and the multiplier-sign
        threshold (``min lambda >= -tol``), and neither is meaningful at 1
        either -- the first declares stationarity for any relatively short
        step, the second accepts a materially negative multiplier. No in-tree
        caller passes ``tol`` at all, so the domain restriction is not a
        breaking change.
    _trace_run : TraceRun | None
        Internal seam, default off. When given an *enabled* ``TraceRun`` the
        active-set loop emits one ``step_decision`` event per blocking
        decision on the ``constrained_qp_blocking`` channel, so a test can
        assert the mechanism -- which rows were considered, whether the
        convergence test accepts them, which was blocked on, and the alpha
        taken -- instead of a numeric outcome whose value depends on BLAS.
        Underscore-prefixed and keyword-only because ``solve_constrained_qp``
        is re-exported from ``superglm.solvers`` and this is not public API.
        The default path is bitwise unchanged: the flag is resolved once
        before the loop and the payload is never constructed.

    Returns
    -------
    QPResult
        Solution with beta, active_set, iteration count, convergence flag.
    """
    # Checked at the public boundary rather than at each use: every predicate
    # below reads ``tol``, and a vacuous one is not detectable from the result.
    # Spelled as a negated range so a NaN ``tol`` is rejected as well.
    if not 0.0 < tol < 1.0:
        raise ValueError(
            f"solve_constrained_qp requires 0 < tol < 1, got {tol!r}. tol is a "
            "relative tolerance whose normalized slack saturates at -1, so "
            "tol >= 1 accepts every finite constraint violation and the solve "
            "reports its unconstrained answer as converged. It is also the "
            "step-norm and multiplier-sign threshold, neither of which is "
            "meaningful at 1."
        )

    p = H.shape[0]
    m = A.shape[0]

    # Materialize the symmetric part once and use it everywhere below; see the
    # ``H`` parameter above for why.  The float cast is load-bearing rather
    # than defensive: ``H + H.T`` evaluates in the input dtype, so an integer
    # H silently wraps -- ``[[2**62]]`` as int64 sums to a negative number and
    # a valid PSD problem is then rejected for a "materially negative
    # diagonal".  The raw ``np.linalg.solve`` this replaced upcast internally,
    # so the exposure arrived with the symmetrization.  Casting fixes only the
    # integer half of that exposure; ``rank._symmetric_part`` covers the
    # floating-point half, where a finite ``H`` above half the float range sums
    # to ``inf`` instead of wrapping.  ``decompose_gram`` symmetrizes again on
    # the way in and goes through the same helper, so both symmetrizations on
    # this path carry the same envelope.
    H_asarray = np.asarray(H, dtype=float)
    H_sym = _symmetric_part(H_asarray)

    # Route the pure-H solves through the shared rank policy so a singular or
    # near-singular H is rank-truncated the way it is everywhere else in the
    # solver subsystem, rather than raising LinAlgError.  H does not change
    # during the solve, so one decomposition serves every pure-H solve below.
    try:
        decomposition = decompose_gram(H_sym)
    except ValueError as exc:
        raise ValueError(f"solve_constrained_qp requires a usable PSD H: {exc}") from exc

    def _result(
        beta: NDArray,
        active_set: list[int],
        n_iter: int,
        converged: bool = True,
    ) -> QPResult:
        """Build a result carrying the geometry every pure-H solve ran on.

        Attribute reads only -- no extra factorization.  A closure rather than a
        splatted dict so each field keeps its own type at every return site.
        """
        return QPResult(
            beta=beta,
            active_set=active_set,
            n_iter=n_iter,
            converged=converged,
            rank=int(decomposition.rank),
            width=int(decomposition.width),
            method=str(decomposition.method),
            condition=float(decomposition.pre_truncation_condition),
            used_svd_fallback=bool(decomposition.used_svd_fallback),
        )

    # --- Unconstrained solution ---
    beta_unc = decomposition.solve(g)

    # decomposition.solve is a pseudo-inverse, so it answers even when the
    # normal equations have no solution.  If H is rank-deficient and g has a
    # component outside range(H), the quadratic decreases without bound along
    # that null direction and H^+g is merely a projection, not a stationary
    # point.  Returning it as converged would be a silent wrong answer, so
    # detect the inconsistency before either early return.  Full rank means
    # range(H) is everything, so the check is only needed after truncation.
    if decomposition.rank < decomposition.width:
        # The two halves of null(H) are known to different accuracies, so they
        # get their own floors; see _null_space_mass.
        structural_mass, spectral_mass = _null_space_mass(decomposition, g)
        spectral_floor = _consistency_floor(decomposition)
        structural_breach = structural_mass > _STRUCTURAL_CONSISTENCY_FLOOR
        if structural_breach or spectral_mass > spectral_floor:
            kind, mass, floor = (
                ("a structurally aliased column", structural_mass, _STRUCTURAL_CONSISTENCY_FLOOR)
                if structural_breach
                else ("a truncated spectral direction", spectral_mass, spectral_floor)
            )
            raise ValueError(
                "solve_constrained_qp: H is rank-deficient (rank "
                f"{decomposition.rank} of {decomposition.width}) and g has a "
                f"component in null(H) along {kind} ({mass:.3e} of ||g||, "
                f"against a resolution floor of {floor:.3e}), so the "
                "unconstrained objective is unbounded below along that "
                "direction. The unconstrained solve and the active-set loop's "
                "empty-active-set step both form directions inside range(H), "
                "so neither entry path can follow it. Note the constraints may "
                "still bound the problem, in which case a finite optimum "
                "exists that this solver cannot reach: doing so needs a "
                "null-space descent direction, which is a filed follow-up "
                "rather than a capability it has today. Regularize H (for "
                "example add a ridge term) or drop the aliased columns."
            )

    if m == 0:
        # No constraints: the unconstrained solve above is already the answer.
        return _result(beta_unc, [], 0)

    if _is_feasible(A, beta_unc, b, tol):
        return _result(beta_unc, [], 0)

    # --- Initialize active set ---
    if active_set_init is not None:
        active = list(active_set_init)
    else:
        active = []

    # --- Feasible starting point ---
    # This may still be infeasible; see _project_feasible.  Feasibility is
    # therefore re-tested on the point actually returned, below.
    beta = _project_feasible(beta_unc, A, b, tol)

    # Resolved once: the hot loop must not pay a per-row or per-iteration
    # attribute lookup for a seam that is off in every production call.
    tracing = _trace_run is not None and _trace_run.enabled

    # Route rank-truncated systems to the least-squares solve.  This gate is
    # deliberately narrow, and it is *not* the claim that the direct solve is
    # safe on the other side of it.  That claim was made once and does not
    # survive measurement:
    #
    # ``decompose_gram`` equilibrates by ``sqrt(diag(H))`` *before* deciding
    # rank (``rank._equilibrate_gram``), so the rank decision reports
    # **collinearity**, not scale.  ``H = diag(1, 1e-20)`` equilibrates to the
    # identity and is reported full rank; across diagonal plantings from
    # ``1e-20`` to ``1e-8`` at widths 2, 6 and 12, 0 of 21 dropped rank at all.
    # Such an H takes ``np.linalg.solve`` on a saddle of raw condition ``1e20``
    # without ever approaching the retention threshold -- and a sweep that
    # plants its small eigenvalue diagonally therefore never leaves this side
    # of the gate, whatever it scores.
    #
    # Measured on this side, with the small eigenvalue planted in the
    # *equilibrated* spectrum and the direction it carries made tangent to the
    # active set (``A_eq v = 0``, so no constraint row pins it), over 192 KKT
    # systems the policy calls full rank:
    #
    # ==========================  ==============  ==============
    # quantity                    np.linalg.solve  equilibrated lstsq
    # ==========================  ==============  ==============
    # ``max|step|`` median        ``5.2e2``       ``3.6``
    # ``max|step|`` worst         ``6.6e6``       ``2.5e2``
    # relative residual median    ``1.2e-21``     ``5.0e-21``
    # relative residual worst     ``7.4e-16``     ``1.0e-15``
    # ``|A_eq (beta + step)|``    ``8.4e-10``     ``3.6e-12``
    # ==========================  ==============  ==============
    #
    # Both residuals are at rounding, so the direct solve is not wrong about
    # the linear system: it *numerically succeeds* and picks a far larger
    # representative on a near-flat line.  That is the round-6 drift mechanism
    # -- the one that reached ``4.1e43`` -- and it is the opposite of a
    # truncation artifact, so an objective gap, which is flat along exactly
    # that direction, cannot see it.  Scored on ``max|beta|`` and worst slack
    # instead: end to end with the eigenvalue at ``4 * eps``, 1 of 40 solves on
    # this side of the gate returned a fully saturated ``-1.0`` slack, and
    # ``max|beta|`` reached ``1.9e6`` where the optimum is ``O(1)``.
    #
    # Widening the gate is behaviour-changing for every near-singular fit, so
    # it is filed rather than done here.  **The decline does not rest on
    # byte-identity with master, and an earlier form of this note said it did.**
    # That form cited 9657 of 10924 KKT direct solves as byte-identical; that
    # corpus was measured for the equilibration commit against its *parent*,
    # answering "did equilibration touch the full-rank path", and re-labelling
    # it as a master comparison attaches the wrong baseline.  It is also
    # self-refuting: admitting rank-deficient ``H`` moved fitted values
    # ``2.9e-13`` relative to master on a monotone ``BSplineSmooth`` fit, and if
    # those moved then ``beta_unc = decomposition.solve(g)`` moved, so the
    # projected start moved, so ``rhs[:p] = g - H_sym @ beta`` moved -- the KKT
    # systems on that fit cannot be master's byte for byte.  Measured directly
    # against ``master`` over the 720-solve full-rank corpus, **49** betas are
    # byte-identical and 671 are not.
    #
    # The decline survives on the **routing**, which is a stronger argument and
    # does not decay as fixtures change:
    #
    # ==========================  ========================  ==================
    # ``H`` after equilibration   master                    this branch
    # ==========================  ========================  ==================
    # full rank                   ``np.linalg.solve``       identical routing
    # rank-deficient, LU-solvable ``np.linalg.solve``       equilibrated lstsq
    #                             -- **the drift regime**
    # exactly singular            ``LinAlgError`` before    equilibrated lstsq
    #                             the loop
    # ==========================  ========================  ==================
    #
    # Row 2 carries it.  ``np.linalg.solve`` does not raise on a merely
    # *near*-singular matrix -- that is the whole mechanism of round 6's P1 --
    # so on master an ``H`` whose smallest equilibrated eigenvalue sat just
    # under the retention threshold went into the loop and took the direct KKT
    # solve on a near-singular saddle.  This branch routes exactly that
    # population away from it.  **The branch strictly shrinks the set of inputs
    # reaching the drifting solve and adds none**, and for the set that remains
    # the routing is master's by construction, since the gate keys on
    # ``decomposition.rank < decomposition.width``, a property of ``H`` alone.
    #
    # What is *not* guaranteed is that the specific systems are unchanged, and
    # it is measurably false rather than merely unproven: admitting the
    # rank-deficient case perturbs the QP initialisation, so a fit can move onto
    # or off the drift.  Over that same full-rank corpus 577 of 720 solves take
    # master's iteration count *and* master's active set while only 49 land on
    # master's bytes -- the route is preserved, the arithmetic along it is not.
    kkt_may_be_singular = decomposition.rank < decomposition.width

    # ``A`` is fixed for the whole solve, so the dot-product error scale
    # ``|A| @ |beta|`` costs one elementwise pass here rather than one per
    # iteration; only the ``|beta|`` matvec is per-iteration.  See
    # ``_feasibility_slack``.
    abs_A = np.abs(A)

    for it in range(max_iter):
        # --- Equality-constrained subproblem on active set ---
        if len(active) == 0:
            # No active constraints — unconstrained step.  beta_unc is the same
            # quantity, already computed above the loop.
            step = beta_unc - beta
        else:
            A_eq = A[active]  # (|active|, p)
            b_eq = b[active]  # (|active|,)

            # Solve KKT system:
            # [H    -A_eq^T] [step  ] = [g - H @ beta]
            # [A_eq  0     ] [lambda] = [b_eq - A_eq @ beta]
            n_eq = len(active)
            KKT = np.zeros((p + n_eq, p + n_eq))
            KKT[:p, :p] = H_sym
            KKT[:p, p:] = -A_eq.T
            KKT[p:, :p] = A_eq

            rhs = np.zeros(p + n_eq)
            rhs[:p] = g - H_sym @ beta
            rhs[p:] = b_eq - A_eq @ beta

            if kkt_may_be_singular:
                # A truncated H can leave the KKT system with a
                # constraint-tangent null direction.  np.linalg.solve
                # *numerically succeeds* on such a system rather than raising,
                # so the LinAlgError fallback below never fires and the step
                # drifts along the flat direction -- measured reaching
                # |beta| ~ 4e43 while violating the constraint it was meant to
                # respect.  Take the minimum-norm solution directly instead of
                # waiting for an exception that does not come.
                sol = _solve_saddle_least_squares(KKT, rhs)
            else:
                try:
                    sol = np.linalg.solve(KKT, rhs)
                except np.linalg.LinAlgError:
                    # Singular KKT — use least-squares.  Same equilibration as
                    # the branch above: this is the identical solve on an
                    # identically scaled saddle matrix, and an unscaled cutoff
                    # would discard the constraint block here for exactly the
                    # reason it does there.  The direct solve above is
                    # untouched, so the full-rank path still reaches this line
                    # only after ``np.linalg.solve`` has already refused.
                    sol = _solve_saddle_least_squares(KKT, rhs)

            step = sol[:p]

        # --- Check step feasibility ---
        # Judge movement relative to the current coefficient scale, coordinate
        # by coordinate.  An absolute step threshold is not invariant to
        # scaling the response: at a genuine active-boundary KKT point,
        # cancellation in ``g - H @ beta`` leaves an O(eps * |beta|)
        # correction.  With a large response that correction can exceed an
        # absolute tolerance forever, even after the active set, feasibility,
        # stationarity, and multipliers have stabilized.
        #
        # Per-coordinate scaling avoids letting one large coefficient hide a
        # material update to another.  It also preserves the previous absolute
        # Euclidean test exactly while every ``|beta_i| <= 1``.
        relative_step = step / np.maximum(1.0, np.abs(beta))
        if np.linalg.norm(relative_step) < tol:
            # At a stationary point.  With an empty active set that is already
            # the whole dual condition; otherwise the multipliers decide first.
            #
            # A small ``step`` means stationarity only because the saddle system
            # is *solvable*, and on the ``kkt_may_be_singular`` path that needs
            # an argument: ``step`` there is the **minimum-norm** least-squares
            # solution, which is small by construction whenever ``lstsq``
            # discards a direction the right-hand side needed.  The argument is
            # below the loop, at the return.
            if len(active) != 0:
                # Recompute multipliers at current point.
                # KKT stationarity: H*beta - g = A_eq' * lambda, lambda >= 0
                # => lambda = (A_eq @ A_eq^T)^{-1} @ A_eq @ (H @ beta - g)
                A_eq = A[active]
                residual = H_sym @ beta - g
                try:
                    multipliers = np.linalg.solve(A_eq @ A_eq.T, A_eq @ residual)
                except np.linalg.LinAlgError:
                    multipliers = np.linalg.lstsq(A_eq @ A_eq.T, A_eq @ residual, rcond=None)[0]

                # Drop most negative multiplier (constraint wants to be
                # inactive).  Spelled as ``not (min_mult >= -tol)`` rather than
                # ``min_mult < -tol`` so a NaN multiplier still takes the drop
                # branch, exactly as it did when this was a single comparison.
                min_mult = np.min(multipliers)
                if not min_mult >= -tol:
                    drop_idx = np.argmin(multipliers)
                    active.pop(drop_idx)
                    continue

            # Stationarity and dual feasibility hold; primal feasibility
            # completes the KKT certificate.
            #
            # **Why stationarity holds even though ``step`` may come from
            # ``lstsq``.**  The termination test above is
            # ``||step / max(1, |beta|)|| < tol``, and reading a small relative
            # step as stationarity is immediate for
            # ``np.linalg.solve``, which either returns *the* solution or
            # raises.  It is not immediate for ``_solve_saddle_least_squares``:
            # a minimum-norm least-squares solution is small exactly when
            # ``lstsq`` truncates a direction ``rhs`` needed, so on that path a
            # small step could in principle be a truncation artifact rather
            # than a KKT point, and every early return with a non-empty active
            # set on a rank-deficient ``H`` arrives through it.  It is not an
            # artifact, and the reason is structural rather than empirical:
            #
            # The system is ``[[H, -A_eq^T], [A_eq, 0]] z = [u; v]`` with
            # ``u = g - H beta`` and ``v = b_eq - A_eq beta``.  Solving the
            # second block needs ``v`` in ``range(A_eq)``; what is then left is
            # ``P (u - H x_p)`` in ``range(P H P)``, where ``P`` projects onto
            # ``null(A_eq)`` and ``x_p`` is any particular solution.  Write the
            # PSD ``H`` as ``L L^T``.  Then ``range(P H P) = range(P L)``, and
            # ``P H Q = (P L)(L^T Q)`` has its range inside ``range(P L)`` as
            # well -- so for any ``u`` in ``range(H)`` both ``P u`` and
            # ``P H x_p`` land in ``range(P H P)`` and the condition holds.
            # **The saddle system is consistent for every PSD ``H``, whatever
            # its rank**, given only that ``u`` is in ``range(H)`` and ``v`` is
            # in ``range(A_eq)``.
            #
            # Both hypotheses are discharged here rather than assumed.  ``u`` is
            # in ``range(H)`` because ``g`` is: the consistency gate above the
            # loop refuses any other ``g`` outright, and ``H beta`` is in
            # ``range(H)`` trivially.  ``v`` is in ``range(A_eq)`` whenever
            # ``b_eq`` is, which is automatic at ``b = 0`` -- every in-tree
            # caller.  So on this code path ``lstsq`` is never asked to discard
            # a direction the right-hand side needs.
            #
            # Measured, at 33705 early returns with a non-empty active set --
            # the two 3950-case rank-deficient ensembles, the full-rank
            # byte-identity corpus, and a 26143-solve hunt seeded with
            # positively dependent constraint rows and nonzero ``b`` -- the
            # number whose gradient ``H beta - g`` is both materially nonzero
            # and materially outside ``range(A_eq^T)`` is **0**.  A runtime
            # residual check was written and measured against this population
            # and fires on none of it, so it is not shipped: it would be a
            # guard that no in-tree caller can reach and that no path exercises.
            # ``test_a_converged_result_satisfies_the_kkt_conditions`` verifies
            # the certificate from outside instead, on the public result.
            #
            # What the argument does *not* cover is ``b_eq`` outside
            # ``range(A_eq)``, which needs positively dependent active rows
            # *and* a nonzero ``b``.  Brute-forced over 40000 random PSD
            # saddles, that is the only way to make the system inconsistent
            # (worst relative residual ``0.97`` there, against ``26.7 * eps``
            # with ``v`` in range), and the 26143-solve hunt did not reach it
            # through the loop's own dynamics.  Filed, not guarded: it is a
            # property of the caller's constraint system, and belongs at the
            # boundary rather than inside the iteration if it ever matters.
            #
            # **Known gap, repaired at the return below rather than in the
            # loop.**  The loop can reach this return on a subset active set
            # while another row is materially violated, and it then returns
            # that point.  Over a 3950-case rank-deficient ensemble this fires
            # on 265 cases (production-shaped: ``b = 0``, structured ``A``, so
            # ``x = 0`` is feasible and the problem is not the constraints) and
            # 155 adversarial ones, with slacks saturating at ``-1.0``.
            #
            # It is **not** confined to the rank-deficient population, and that
            # correction matters for who owns the defect.  Over the 600
            # full-rank cold solves of the byte-identity corpus -- ``H`` built
            # as ``M^T M + c I``, so ``np.linalg.solve`` runs on both arms and
            # nothing about this branch's routing applies -- ``master`` returns
            # an infeasible early return on **92** and this branch on **93**.
            # The phenomenon and its rate are master's; only which specific
            # cases land on it moved (67 overlap, 4 byte-identically), because
            # the QP initialisation moved.  So the rank-deficient half below is
            # new to this branch and the full-rank half is inherited.
            #
            # ``converged`` is the inner KKT certificate, while ``beta`` remains
            # the best finite iterate.  ``irls_direct`` deliberately consumes
            # that iterate so a later outer iteration can recover, but attaches
            # the certificate to the retained coefficient state: an incomplete
            # certificate blocks outer convergence and, if still incomplete at
            # termination, produces ``constraint_kkt_incomplete`` plus a
            # warning.  Fixed-lambda and automatic constrained REML reject that
            # terminal reason before publication.  Projection is still needed
            # here because consuming a finite iterate is safe only when the
            # hard constraints are satisfied; the certificate alone does not
            # make an infeasible coefficient vector admissible.  For the
            # rank-deficient half the population is this branch's own: on
            # ``master`` ``np.linalg.solve(H, g)`` ran before the loop, so a
            # singular ``H`` raised ``LinAlgError`` and the loop never ran at
            # all -- measured, ``master`` refuses 2181 of those same 3950 cases
            # outright, and 165 of them are cases where this branch instead
            # returns an infeasible ``beta``.  Loud refusal to silent
            # infeasibility, which is the same correction round 6 already
            # accepted for its own P1.
            #
            # The obvious repair -- add the worst violated inactive row and
            # continue -- was implemented and measured, and is **not** shipped.
            # It repairs 123 of the 265 and regresses no feasibility outcome,
            # but the loop has no anti-cycling rule and this is precisely the
            # manoeuvre that cycles: 9 cases that terminated in 3 to 16
            # iterations then run to ``max_iter``, and 5 of those still do at
            # ``max_iter=4000``, in traced two-state cycles (``drop 3`` /
            # ``block-add 3``, forever).  Bounding re-activations does not
            # close it -- the cycle it steers into never revisits this branch,
            # because the multiplier test drops first every time -- so the
            # bounded variant leaves the same 5 non-terminating.  The residual
            # 142 are not reachable by it at all: in **all** of them the worst
            # violated row is already *active*, so what is left needs the
            # equality block enforced, which is the active-set redesign.  It
            # also moves the full-rank path: 37 corpus records over 8 of 120
            # full-rank seeds, including one that goes from ``n_iter=41`` to
            # exhaustion.
            #
            # So the point is repaired here instead, where the repair cannot
            # cycle.  Feasibility is what the constraint is *for*: at
            # ``irls_direct.py:1614`` an infeasible ``beta`` means the fitted
            # model is not monotone, while a projected one is monotone and
            # merely worse in objective -- and that call site takes ``beta``
            # unconditionally, so the real choice is between shipping a
            # violated constraint and shipping a suboptimal coefficient.
            #
            # **Inert on the currently-feasible population by construction, not
            # by measurement.**  The guard is *precisely* the condition under
            # which ``converged`` is already ``False`` today, so no solve that
            # currently returns a feasible point takes either the projection or
            # a changed flag.  Measured to confirm the construction rather than
            # to establish it: over the 1440 solves of the byte-identity corpus
            # 1295 are byte-identical, all 145 that move were returning an
            # infeasible point, **0** that were returning a feasible one move,
            # and 0 move from an exhaustion return.  All eight end-to-end
            # constrained fits are byte-identical.  It terminates by
            # construction too -- ``_project_feasible`` is 100 bounded sweeps
            # and cannot cycle -- which the loop-side repair could not promise.
            #
            # Reach and cost.  It repairs 146 of the 265 production-shaped and
            # 77 of the 155 adversarial cases, and 51 of the full-rank corpus's
            # cold solves.  The rest exhaust the 100-sweep budget still
            # infeasible, so this is a partial repair by design: what is left
            # needs the equality block enforced, which is the redesign.  The
            # objective is paid for it -- median relative change ``2.4e-13``,
            # but ``1.1`` at the 99th percentile and ``1.9`` at worst -- which
            # is the trade being made and not a defect: those points were
            # outside the feasible set, so their objective was never admissible.
            #
            # ``converged`` deliberately reports the feasibility of the point
            # the loop *found*, taken before the projection runs, and this is
            # not a concession to the sweep budget.  It is the stronger reading:
            # projecting moves ``beta`` off the stationary point, so the KKT
            # certificate does not hold for the projected point even when the
            # projection fully succeeds.  Reporting post-projection feasibility
            # would flip exactly the 223 repaired cases to ``converged=True`` --
            # feasible, but demonstrably not a KKT point -- which is the
            # over-claim the flag exists to prevent.  The budget does also run
            # out, on 119 and 78 of those two populations, so on those the two
            # readings agree; they agree for a reason that does not survive the
            # repair succeeding, which is why the pre-projection one is taken.
            #
            # ``active_set`` is returned unchanged and still describes the
            # pre-projection point.  It is a warm start, not a claim about
            # ``beta``, and narrowing it is the loop work rather than this.
            feasible = _is_feasible(A, beta, b, tol)
            if not feasible:
                beta = _project_feasible(beta, A, b, tol)
            return _result(beta, active, it + 1, converged=feasible)

        # --- Step ratio: find blocking constraint ---
        # Both tests below go through the same per-row scale the convergence
        # test uses.  Leaving them absolute while the boundaries went relative
        # made the loop treat a row as violated that _is_feasible considers
        # satisfied, so it would block on that row with a negative slack and
        # take a backward step.
        beta_new = beta + step

        if _is_feasible(A, beta_new, b, tol, abs_A=abs_A):
            # Full step is feasible
            beta = beta_new
        else:
            # Find blocking constraint (first to be violated along step)
            alpha_min = 1.0
            blocking = -1

            products = A @ beta
            raw_step = A @ step
            # Same scale ``_is_feasible`` uses, which is what makes "does this
            # step move the row" agree with "is this row satisfied".
            scaled_step = raw_step / _feasibility_scale(
                products, b, abs_products=abs_A @ np.abs(beta)
            )
            # Set membership, not ``i in active``: the list scan is O(|active|)
            # per row and dominated the rest of this now-vectorized block.
            # ``active`` is only mutated after this loop finishes.
            active_lookup = set(active)

            for i in range(m):
                if i in active_lookup:
                    continue
                if scaled_step[i] < -tol:
                    # Gate on the *scaled* derivative, so "does this step move
                    # the row at all" agrees with _is_feasible.  Take the ratio
                    # from the *raw* pair: dividing both by the row scale is
                    # algebraically neutral but not neutral in floating point,
                    # since it rounds twice where this rounds once.  On a row
                    # with slack > 1 the scaled numerator is exactly 1.0 and
                    # the two answers differ by up to an ulp, which would put
                    # an ulp of drift into `beta += alpha_min * step` for no
                    # gain -- the gate is what needed to change, not the ratio.
                    alpha = (products[i] - b[i]) / -raw_step[i]
                    if alpha < alpha_min:
                        alpha_min = alpha
                        blocking = i

            if tracing:
                # After selection, before `active` is mutated, so the recorded
                # `active_lookup` is the set the decision was actually made
                # against.  Off by default, so this costs one bool test.
                _emit_blocking_decision(
                    _trace_run,
                    iteration=it,
                    A=A,
                    b=b,
                    beta=beta,
                    beta_new=beta_new,
                    tol=tol,
                    products=products,
                    raw_step=raw_step,
                    active_lookup=active_lookup,
                    blocking=blocking,
                    alpha_min=alpha_min,
                )

            if blocking >= 0:
                beta = beta + alpha_min * step
                active.append(blocking)
            else:
                beta = beta_new

    # Exhaustion is unconditional non-convergence, feasible or not: the loop
    # never reached its stationarity/multiplier test, so there is no KKT
    # certificate to complete.  A merely feasible point is not a solution --
    # any interior point is feasible -- so consulting feasibility here would
    # report success for a search that was cut off mid-flight.
    return _result(beta, active, max_iter, converged=False)
