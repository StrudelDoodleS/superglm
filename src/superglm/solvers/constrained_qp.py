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
    no negative multiplier) *and* ``beta`` is feasible.  It is ``False`` when
    the loop exhausted ``max_iter``, and when the loop terminated but the
    returned point still violates a constraint -- which is what happens for a
    mutually infeasible constraint system.  In either case ``beta`` is the
    best available point, not a certified solution.
    """

    beta: NDArray
    active_set: list[int] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True


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
    since the rank policy truncates at ``gram_rcond`` and so retains blocks far
    more ill-conditioned than ``factor_rcond`` would tolerate.

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
    A: NDArray, beta: NDArray, b: NDArray, *, abs_b: NDArray | None = None
) -> NDArray:
    """Return ``A @ beta - b`` measured against a scale-aware tolerance.

    A step that lands *on* a constraint reproduces ``b_i`` only to about
    ``eps * |A_i @ beta|``, so a fixed absolute tolerance turns a genuine KKT
    point into a violation as soon as the constraint row is large: at
    ``|A_i @ beta| ~ 1e4`` an exactly-active constraint already reads ``-2e-12``.
    Comparing against ``tol * max(1, |b_i|, |A_i @ beta|)`` keeps the test
    meaningful under rescaling, and is identical to the absolute test for the
    well-scaled problems where the scale factor is 1.

    Returns the slack already divided by its per-row scale, so callers can
    compare it against a bare ``-tol``.

    ``abs_b`` lets a caller in a loop pass ``np.abs(b)`` once instead of paying
    for it every sweep; it must equal ``np.abs(b)``.
    """
    products = A @ beta
    return (products - b) / _feasibility_scale(products, b, abs_b=abs_b)


def _feasibility_scale(products: NDArray, b: NDArray, *, abs_b: NDArray | None = None) -> NDArray:
    """Per-row scale for the relative feasibility test: ``max(1, |b|, |A @ beta|)``.

    Exposed separately so the active-set loop can divide *both* its slack and
    its directional derivative by the same factor.  Scaling both leaves the
    step ratio ``slack / -a_step`` numerically unchanged while making the
    decisions built on them -- "is this row already satisfied", "does this step
    move the row at all" -- agree with ``_is_feasible``.
    """
    return np.maximum(1.0, np.maximum(np.abs(b) if abs_b is None else abs_b, np.abs(products)))


def _is_feasible(A: NDArray, beta: NDArray, b: NDArray, tol: float) -> bool:
    """Whether ``beta`` satisfies ``A @ beta >= b`` to a scale-aware tolerance."""
    return bool(np.all(_feasibility_slack(A, beta, b) >= -tol))


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

    Two properties make the scaling safe on a degenerate block, which is the
    reason this cannot emit ``inf`` or ``nan``:

    * **No zero divide.** A structurally empty row -- an ``H`` row that is zero
      with a matching zero ``A`` column, or an all-zero constraint row -- has
      inf-norm 0 and keeps scale ``1.0`` rather than being sent through
      ``1 / sqrt(0)``.  Leaving it alone is
      correct: there is nothing in that row to normalize, and because ``|KKT|``
      is symmetric the matching column is zero as well, so the row and column
      stay zero and ``lstsq`` discards the direction as it should.  Clamping to
      ``tiny`` instead would manufacture a ``6.7e153`` scale for a row that
      carries no information.
    * **No overflow.** For a nonzero row, ``|K[i, j]| <= min(m_i, m_j) <=
      sqrt(m_i * m_j)``, so every equilibrated entry satisfies
      ``|K[i, j]| / sqrt(m_i * m_j) <= 1`` by construction, whatever the
      dynamic range of the input.  The bound is exact in real arithmetic and
      holds to a couple of ulps once the two multiplies round, which is what
      the overflow argument needs.  A single pass is what carries this
      guarantee: iterating to the Ruiz fixed point bounds the accumulated
      scale only by the input's dynamic range rather than by ``1/sqrt(m_min)``,
      trading a provable envelope for an empirical one.  Measured over a
      3950-case rank-deficient ensemble the extra passes repair 14 infeasible
      answers against this pass's 7, for the same 2 regressions -- real, but
      not worth the weaker guarantee on this path.

    The solution is minimum-norm **in the equilibrated coordinates**, not in
    the original ones, because that is where ``lstsq`` resolves the null
    directions.  That is the same convention ``RankDecomposition.solve`` uses
    for the pure-``H`` solve -- it also divides by its column scale, solves,
    and divides again -- so the two solves inside this module now pick the same
    representative on a flat optimal face.  Where the face is genuinely flat
    the point moves but the objective does not: the rank-one regression below
    returns ``max|beta| = 18.3`` rather than ``15.7`` for the same exact
    optimum of ``-450``.

    ``rcond`` is deliberately left at ``None``.  The competing proposal is to
    pass ``SHARED_RANK_POLICY.gram_rcond`` so that one rule decides what is
    retained, since the policy keeps ``H`` eigenvalues down to
    ``gram_rcond * lambda_max`` while ``lstsq`` drops singular values below
    ``max(M, N) * eps * sigma_max``.  Equilibration already settles that
    disagreement, and settles it *better*, for two measured reasons:

    * A direction that is near-null for ``H`` but pinned by an active
      constraint keeps a KKT singular value of order that constraint row's
      norm, not of order its ``H`` eigenvalue -- a KKT singular value is not an
      ``H`` eigenvalue.  Over a ``p = 180`` ensemble with 50-row active sets and
      an in-band ``H`` eigenvalue planted at ``50 * eps * lambda_max``, the
      smallest retained equilibrated ratio measured ``1.3e-4`` and the largest
      truncated one ``0``; nothing landed between ``eps`` and ``230 * eps``, so
      the two candidate cutoffs cannot disagree there.
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
    *selection* from the raw violations, the same split round 4 applied to the
    blocking ratio.  The scale-aware slack is a per-row **clamp**: at ``b = 0``
    it is ``x / max(1, |x|)``, which is exactly ``-1.0`` for every violation
    worse than 1, so rows violated by wildly different amounts become
    indistinguishable and ``argmin`` breaks the exact tie on the lowest index.
    Measured on a first-difference ``A``: raw violations ``[-3, -8, -3]`` scale
    to ``[-1, -1, -1]``, and the sweep repairs row 0 while row 1 is three times
    worse.  Below saturation the two orders agree (``[-0.3, -0.2, -0.4]``
    scales to itself), which is what pins the clamp as the cause rather than a
    coincidence of the fixture.  Clamping is monotone nondecreasing at
    ``b = 0``, so the raw argmin still attains the minimum scaled slack and the
    stopping decision is unchanged there; the ``min`` below keeps the test
    correct for a nonzero ``b``, where it is not.

    **At ``b = 0`` -- every in-tree call site -- this is bitwise ``master``'s
    projection**, and the claim is an argument rather than a measurement, so it
    does not decay as fixtures change:

    ==========================  ===================================================
    step                        at ``b = 0``
    ==========================  ===================================================
    ``products - b``            ``x - 0.0`` is bitwise ``x``, signed zeros
                                included; ``master`` forms the same difference.
    ``argmin(violations)``      the same array, and ``master`` also selected on
                                the raw violations (``git show e8e31f4``).
    ``slack.min() >= -tol``     the clamp ``v / max(1, |v|)`` is exactly ``v``
                                for ``|v| <= 1`` and exactly ``-1`` for
                                ``v < -1``, so for ``tol`` in ``(0, 1)`` --
                                enforced at the public boundary -- ``clamp(v) >=
                                -tol`` and ``v >= -tol`` are the same predicate
                                row by row, and the clamp is nondecreasing so
                                ``min`` commutes with it.  ``master``'s test is
                                ``violations[argmin] >= -1e-12``, and the default
                                ``tol`` **is** ``1e-12``.
    repair body                 unchanged, term for term.
    ==========================  ===================================================

    The ``tol`` domain is what makes the third row exact rather than
    approximate: at ``tol >= 1`` the clamped test accepts violations the raw one
    rejects, which is the vacuity the boundary check exists to refuse.  A
    nonzero ``b``, or a ``tol`` outside the default, narrows the claim to
    "same predicate, possibly different arithmetic" -- no in-tree caller does
    either, and ``test_the_projection_is_bitwise_masters_at_zero_rhs`` pins the
    equality against a hand-written reference rather than against a recorded
    number.

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
    order simply never picks a row that is not violated.
    """
    beta = beta.copy()
    # Loop-invariant: only ``A @ beta`` changes between sweeps.
    abs_b = np.abs(b)
    for _ in range(100):
        # Inlined rather than calling ``_feasibility_slack``, which would
        # recompute the matvec; the arithmetic is identical term for term.
        products = A @ beta
        violations = products - b
        slack = violations / _feasibility_scale(products, b, abs_b=abs_b)
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
    derived_scale = _feasibility_scale(products, b)
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
        ``A_i @ beta - b_i >= -tol * max(1, |b_i|, |A_i @ beta|)``, so a
        badly scaled constraint system does not read as infeasible purely
        because its rows are large. This matters only for callers with a
        nonzero ``b``: at ``b_i == 0`` the relative test is algebraically
        identical to the absolute one for any ``tol`` in ``(0, 1)``, and every
        in-tree caller passes ``b = 0``.

        ``(0, 1)`` is the predicate's actual domain, not a house style. The
        normalized slack saturates at ``-1`` -- at ``b = 0`` it is
        ``x / max(1, |x|)``, which is exactly ``-1`` for every violation worse
        than 1 -- so at ``tol >= 1`` the test accepts *every* finite violation
        and the solve returns its unconstrained answer with
        ``converged=True``. Measured on ``H = [[1]]``, ``g = [-100]``,
        ``beta >= 0``: ``tol = 0.999999`` returns ``beta = 0``, ``tol = 1.0``
        returns ``beta = -100`` and calls it converged.

        Rejecting the value rather than reformulating the predicate is
        deliberate. The vacuity is algebraic, not a rounding artifact: written
        without the division, the test is ``x >= -tol * max(1, |x|)``, and at
        ``tol = 1`` that is ``x >= -|x|``, which holds for every ``x <= 0``
        however it is spelled. The only formulation that cannot saturate is one
        whose scale excludes ``|A_i @ beta|`` -- which is precisely the term
        this test grew in order to stop reading an exactly-active large
        constraint row as violated. So a non-vacuous formulation is not a
        reformulation but a revert. Validation is also honest about the other
        two jobs this one parameter does: it is the step-norm convergence
        threshold (``||step|| < tol``) and the multiplier-sign threshold
        (``min lambda >= -tol``), and neither is meaningful at 1 either -- the
        first declares stationarity for any step shorter than 1, the second
        accepts a materially negative multiplier. No in-tree caller passes
        ``tol`` at all, so the domain restriction is not a breaking change.
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
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

    if _is_feasible(A, beta_unc, b, tol):
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

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

    # Only a rank-truncated H can put a null direction into the KKT system.
    # Resolved once so the full-rank path -- which is nearly every in-tree
    # solve -- keeps taking np.linalg.solve on bitwise the same inputs.
    kkt_may_be_singular = decomposition.rank < decomposition.width

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
        if np.linalg.norm(step) < tol:
            # At a stationary point.  With an empty active set that is already
            # the whole dual condition; otherwise the multipliers decide first.
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
            return QPResult(
                beta=beta,
                active_set=active,
                n_iter=it + 1,
                converged=_is_feasible(A, beta, b, tol),
            )

        # --- Step ratio: find blocking constraint ---
        # Both tests below go through the same per-row scale the convergence
        # test uses.  Leaving them absolute while the boundaries went relative
        # made the loop treat a row as violated that _is_feasible considers
        # satisfied, so it would block on that row with a negative slack and
        # take a backward step.
        beta_new = beta + step

        if _is_feasible(A, beta_new, b, tol):
            # Full step is feasible
            beta = beta_new
        else:
            # Find blocking constraint (first to be violated along step)
            alpha_min = 1.0
            blocking = -1

            products = A @ beta
            raw_step = A @ step
            scaled_step = raw_step / _feasibility_scale(products, b)
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
    return QPResult(beta=beta, active_set=active, n_iter=max_iter, converged=False)
