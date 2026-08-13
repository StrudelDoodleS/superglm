"""Which edf a screening path is supposed to report, against a CERTIFIED oracle.

The two paths that score a ``spline_cat`` pair -- the dense ladder in
:mod:`superglm.screening._score_stat` and the arrow ladder in
:mod:`superglm.screening._structured` -- disagree, and issue #257 records that
disagreement without saying which of them is right.  Neither path can settle
it, and neither can they settle it between them: agreement is not correctness
and disagreement does not say which side moved.

**The oracle these tests pin is arb ball arithmetic** (``python-flint``), which
returns a RIGOROUS enclosure rather than a high-precision guess.  Every constant
below was produced at 800 bits with the enclosure radius printed beside it, and
cross-checked against two algebraically distinct forms of the same quantity --
``tr(A^-1 V_eff)``, ``k - lam tr(A^-1 S)`` and ``sum_j eig_j(A^-1 V_eff)`` --
which agreed to nine decimals.  ``python-flint`` is NOT a test dependency: the
constants and the floors are pinned here and both derivations live in the
commits that added them.

**AN ORACLE IS ONLY A TARGET WHERE THE INPUTS DETERMINE IT, AND A DERIVATIVE IS
ONLY A BOUND WHERE IT IS VALID.**  Both paths receive bit-identical moments, so
everything downstream is algorithm -- but the moments themselves are float64,
and at the ladder's high edge ``lambda`` is 1e10 times the pair's scale.  From
``edf(lam) = tr(A^-1 V)`` with ``A = V + lam S``,

    d edf = lam tr(A^-1 S A^-1 dV) - lam tr(A^-1 V A^-1 dS)

so one ulp of every stored entry -- ``|dV_ij| <= e |V_ij|``,
``|dS_ij| <= e |S_ij|`` at ``e = 2^-52``, a full ulp rather than the unit
round-off ``2^-53``, so the bound covers STEPPING to the neighbouring float and
not only rounding to it -- gives the elementwise contraction

    D = lam e ( sum_ij |(A^-1 S A^-1)_ij| |V_ij|
              + sum_ij |(A^-1 V A^-1)_ij| |S_ij| )

evaluated in arb at 800 bits, since ``A^-1`` in float64 is meaningless at these
condition numbers.  **D is a derivative at the unperturbed A, not a bound on a
finite step**, and it becomes one only where the perturbed inverse exists and
the Neumann series converges:

    r = ||dA||_2 / min_j |eig_j(A)|  <  1,   with ||dA|| <= e (|V|_F + lam |S|_F)

so ``r < 1`` at least says the perturbed inverse exists and is bounded:

    point       lam         min|eig A|   ||dA||      r
    bs(8) hi   1.065e+05    8.908e-08    3.196e-05   3.588e+02
    ps(8) lo   1.381e-11    1.823e-13    3.996e-15   2.192e-02
    ps(8) hi   1.381e+09    1.276e-07    3.436e-05   2.692e+02
    ns(6) lo   5.895e-12    2.223e-12    6.266e-16   2.819e-04

THE TWO HIGH-EDGE ROWS ARE NOT float64 READINGS, AND THE EARLIER ONES WERE.
``min|eig A|`` at those points is 1e-07 against a ``k eps ||A||_2`` of 8.2e-04,
so ``eigvalsh`` does not resolve it: on the ``bs(8)`` pair float64 reports ONE
negative eigenvalue at -7.03e-07, and at 50 digits there are FOUR, all at
-8.909e-08.  float64 ``dsytrf`` gets the inertia right (4) where ``dsyevd`` does
not, which is the tell.  Both high-edge rows are therefore taken at 50 digits
and the low-edge two are float64, where the two agree to four figures
(1.8226e-13 against 1.8230e-13, and 2.2231e-12 against 2.2231e-12).  Correcting
them moved ``r`` from a quoted 2.010e+03 to 3.588e+02 and from 6.119e+01 to
2.692e+02; both are still far above 1, so nothing downstream of ``r > 1``
changes.

At both HIGH edges one ulp of the stored moments is 269x to 359x the smallest
eigenvalue of ``A``, so a one-ulp step can cross a singularity there and NOTHING
bounds the exact edf -- not D, not a multiple of D.  Those two points are not
pinned against ``tr(A^-1 V_eff)`` at all.

WHERE THAT SMALLEST EIGENVALUE COMES FROM, since the answer decides whether it
is fixable.  It is the SCALING and not the penalty builder.  Measured by exact
rational LDL on the stored float64 bytes, because an eigensolver's own backward
error here is larger than the quantity being judged:
``build_difference_penalty`` (``ps``, ``ns``) is EXACTLY positive semidefinite
with exactly ``m`` zero pivots at all eight shapes tried -- it is the Gram
matrix of an integer difference operator, so float64 holds it exactly, and the
-1e-15 an eigensolver reports is the eigensolver's.
``build_integrated_derivative_penalty`` (``bs``, ``cr``) is a different story:
assembled by quadrature, it is genuinely indefinite at K=8 order 2 (exact
inertia 6 positive, 2 negative), K=9 order 2 and K=11 order 1, and where it is
definite it has no null space at all -- exact inertia (11, 0, 0) at K=11 order 2
where the analytic nullity is 2.  What is common to all four bases is
``fl(lambda * S_a)``: rounding each entry independently means the product is not
a scalar multiple of ``S_a``, and on the exactly-PSD difference penalty it
carries a negative exact pivot at 10 of 32 lambdas spanning the ladder's bracket
for (11,2) and 22 of 32 for (8,2) -- never more of them than the nullity, and at
powers of two, where the multiply is exact, never at all.

**AND WHERE r < 1, D / (1 - r) IS STILL NOT A THEOREM.**  ``r`` bounds the
perturbed inverse in an OPERATOR norm; D is an entrywise contraction of the
derivative at the unperturbed ``A``, and dividing one by the other mixes two
norms and still does not bound the rotated entrywise factors of
``(A + dA)^-1``.  The rigorous route is interval arithmetic -- put a BALL of
radius ``e |x|`` on every stored entry and carry the whole neighbourhood
through the solve -- and it was run rather than argued about:

    point      one-ulp ball solve in arb, 800 bits
    bs(8) hi   REFUSED -- A not provably invertible over the ball
    ps(8) lo   admitted, radius 1.5109 df
    ps(8) hi   REFUSED
    ns(6) lo   REFUSED

So the honest position, and the one this file now takes: **the floors below are
exact first-order sensitivities, not certified finite-step bounds.**  What is
known about each is stated in full:

    point      floor       D           rigorous ball   random draws   tolerance
    ps(8) lo   5.6825e-03  5.5544e-03   1.5109         5.61e-03       6e-2  10.8x D
    ns(6) lo   8.2863e-05  8.2837e-05   refused        1.82e-04       1e-3  12.1x D
    bs(8) hi   -- pinned on the truncated functional instead, D = 3.0484e-05 --
    ps(8) hi   -- not pinned at all --

**THE CONSTANTS IN THE "floor" COLUMN ARE ``D / (1 - r)``, NOT ``D``, AND THAT
IS A HELD-OVER LABEL RATHER THAN A HELD-OVER ARGUMENT.**  ``D/(1-r)`` is the
mixed-norm quantity retracted five paragraphs above; the constants were minted
from it and kept their name.  They are retained rather than re-minted because
the difference is 2.3% and 0.03% -- ``5.5544e-03/(1 - 2.192e-02) = 5.679e-03``
and ``8.2837e-05/(1 - 2.819e-04) = 8.2860e-05`` -- so as a stand-in for ``D``
each is very slightly CONSERVATIVE, which is the harmless direction for a floor
a tolerance must clear.  Nothing rests on the difference: against the true ``D``
the tolerances are 10.8x and 12.1x rather than 10.6x and 12.1x, and the random
draws reach 1.01x and 2.2x of ``D`` (0.99x and 2.2x of the tabulated floor).
The tolerances are 10x to 12x D.  That is a stated engineering choice and it is
NOT a certification: the rigorous bound where one exists is 266x looser, and a
6e-2 tolerance widened to 15 df would no longer distinguish the arrow path's
0.525 df error from the dense path's 1.27e-04, which is the finding the file
exists to record.  What makes the choice defensible rather than arbitrary is
that D is very nearly ATTAINED -- random one-ulp draws reach 1.01x and 2.2x of
it -- so it is not an underestimate of the real spread, only of the worst case
interval arithmetic is willing to certify.  Anyone who needs a certified bound
here needs the ball radius, and at three of these four points there is not one.

The earlier form of this file set tolerances from maxima over 40 random 1-ulp
draws at three seeds.  Those figures were 9.4e-04, 5.61e-03, 77.54 and 1.82e-04
df; against D they are 0.06x, 1.01x, 1.10x and 2.2x, so a random draw neither
bounds the worst case nor reliably approaches it, and at the ``bs(8)`` high edge
the resulting 1e-2 tolerance sat at 0.62x its own floor.  Nothing here is set
from a draw.

A second, independent check: re-deriving the four oracle constants on another
machine, all six thread pools pinned, reproduced them to -4.4e-10, +1.09e-04,
-4.0e-10 and +1.3e-05 df.  The ``ps(8)`` low-edge one is the same ORDER as the
error being measured there -- graded against the constant below the dense
clamped rung is 1.27e-04 df out, graded against the re-derived one 1.78e-05, a
factor of 7.1 from nothing but reassembling the moments elsewhere.  The POINT
moves between environments; the tolerance is set from the floor, never from
either error.

**WHAT ``_edge`` ACTUALLY COMPUTES, WHICH IS NOT ``tr(A^-1 V_eff)``.**  The
recommendation this work establishes -- target ``tr(A^-1 V_eff)``, which is
what :func:`superglm.screening._score_stat._edge` already computes -- is true
only on the branch where ``cho_factor`` succeeds.  ``_edge`` factors ``A`` with
``cho_factor`` and FALLS BACK to ``numpy.linalg.pinv(A, hermitian=True)``.
Called with neither ``rcond`` nor ``rtol``, that is ``rcond = 1e-15`` relative
to the largest singular value -- NumPy's documented back-compatible default, not
the Array API's ``max(M, N) * eps``, which applies only when ``rtol=None`` is
passed explicitly.  Verified directly: on a ``k = 200`` matrix whose spectrum is
ones and one small eigenvalue, ``pinv`` inverts 2e-15 and 1e-14 and drops
5e-16, where ``max(M, N) * eps = 4.44e-14`` would have dropped all three.  The
two rules are 27x apart at ``k = 121``, and ``pinv``'s cut is NOT
:func:`_rank_floor`'s -- ``_rank_floor(121)`` is 2.69e-14 against ``pinv``'s
1e-15.

The fallback therefore answers ``tr(P V_eff)`` for a pseudo-inverse that ZEROES
every direction under its cut instead of inverting it.  Measured at the two high
edges pinned here, ``cho_factor`` FAILS on both and the fallback is what runs:

    point      pinv cut     dropped   tr(P V) - tr(A^-1 V)   spectral gap
    bs(8) hi   3.0698e-05    4 / 121   +2.4974e-04 df         2.92e+05
    ps(8) hi   2.1261e-05    4 / 209   -8.8838e-01 df         8.62e+04

so at the ``ps(8)`` high edge nine tenths of a degree of freedom of the gap to
the certified oracle is the truncation and not the arithmetic.  (The Array API
cut would drop the SAME four directions at both points -- the spectral gap is
five orders wide -- so every number in this file is unchanged by which rule is
named.  Only the description was wrong.)  The clamped rung is EXACT for the
functional it evaluates: reconstructing ``tr(P V_eff)`` from ``eigh`` at pinv's
cut reproduces the reported rung to 3.6e-15 and 7.1e-15 df, and in arb to all
16 digits.  That truncated functional IS well posed -- its own ``r`` is 2.7e-05
and 3.9e-05 against the RETAINED spectrum -- which is what makes the ``bs(8)``
high edge pinnable at all, and it is pinned as what it is rather than as the
exact trace.

``pinv`` also scores directions by ``|lambda|`` where :func:`_psd_rank` reads
the sign, so a NEGATIVE curvature direction is inverted rather than dropped; at
both points that rule is inert, but not for the reason first recorded here.
Resolved at 50 digits, the ``bs(8)`` pair carries FOUR negative eigenvalues, all
at -8.909e-08, and the ``ps(8)`` pair one at -1.276e-07 -- not "one each" at
-1.59e-08 and -5.62e-07, which were unresolved float64 readings.  Every one of
them is below its pair's cut (3.0698e-05 and 2.1261e-05), so all are dropped
rather than inverted, and on the ``bs(8)`` pair the four negative directions are
exactly the four ``pinv`` discards.  Inert here is
not inert everywhere, and nothing counts them.

**THE THREE ``xfail(strict=True)`` TESTS DO NOT PIN THEIR OWN DEFECT.**  A
strict xfail records that a test fails; it cannot say the failure is still the
one documented, so a repair and a tenfold worsening look identical.  Each is
therefore PAIRED with an ordinary passing test asserting that the defect is
still THERE and still in the same DIRECTION.

**THOSE PAIRED TESTS ASSERT ONLY THAT THE TWO PATHS DISAGREE, BECAUSE NOTHING
ELSE IS PORTABLE.**  They were written as magnitude brackets first, then as
signed direction assertions, and CI refuted both in turn.  Same numpy 2.4.2,
same scipy 1.18.0, same fixtures, same seeds -- only the interpreter and the
runner differ:

    quantity                            local 3.14   CI 3.12    CI 3.14
    two-estimator gap, bs(8) hi         2.0000 df    4.0000 df  --
    arrow error, ps(8) lo               -0.525 df    -2.780 df  +0.0825 df
    arrow error, ns(6) lo               -30.74 df    -0.0358 df --
    round-off V-mass, ps(8) hi          3.108 df     0.963 df   --

NOT EVERYTHING IN THAT TABLE MOVES, AND WHICH HALF MOVES IS THE POINT.  CI's
3.12 reported the ``bs(8)`` high-edge pair as 7.000002839965082 against
10.999974250213997.  So of the two dense estimators the CLAMPED one is
reproducible -- 1.5e-06 df from this machine's 7.000004364954146, which is 0.05x
its own first-order floor -- and the SEARCHING one moved by 2 df on the same
bytes.  The 4.0000 df gap is not "the defect got bigger there"; it is the
pencil's ``?sygvd`` error bound being realised at ``cond(G) = 5.58e+12``, and it
is why the clamped rung CAN carry an absolute pin below while nothing else at
these edges can.

Two things die there.  On CI's 3.12 the ``ns(6)`` arrow reading does not
collapse at all -- 30.9529 against a certified 30.9887 -- where here it reads
0.2452, so **"the worst arrow reading measured anywhere" is one interpreter's
number**.  And on CI's 3.14 the ``ps(8)`` arrow error changes SIGN: the arrow
path reads 172.9870 against the dense path's 172.9045, HIGH by 0.0825 df, where
here it reads 0.525 df low.  So "the arrow path reads low at the low edge" is
not portable either.

WHAT IS PORTABLE, on every run observed:

* the two paths DISAGREE at the low edge, by 0.0825 to 30.74 df, where a
  repaired arrow path would agree to the dense path's own ~1e-4;
* the DENSE path matches the certified oracle inside its pinned tolerance --
  1.27e-04 here, 9.4e-05 on CI's 3.14, both against 6e-2.

Those two together are the verdict -- they disagree, and the dense one is the
accurate one -- and they need neither the sign nor the size of the arrow error.
That is what these tests assert: ``|dense - arrow| > 1e-3 df``.  They catch
REPAIR, which is what a strict xfail cannot say.  They do NOT catch worsening,
and no portable test can when the same fixture moves 859x and changes sign
between interpreters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Spline
from superglm.screening._score_stat import penalized_score_statistic_ladder
from superglm.screening._structured import spline_cat_moments, structured_ladder

# Every test here fits a model through the public screen, which is what
# pyproject's ``slow`` marker is defined for.  The cost is the same class as the
# tests already carrying it -- 0.03 to 0.46 s each, against 0.03 to 0.39 s for
# the five slow-marked cases in tests/test_screening_worth_gate.py.  Those two
# figures are a RELATIVE comparison taken in one run on a shared machine, which
# is the only form worth quoting: neither is a benchmark and neither should be
# read as an absolute cost.  Everything else in this file is exact arithmetic or
# a mutation result, and so is load-independent.
pytestmark = pytest.mark.slow

BUDGETS = (2.0, 4.0, 8.0, 16.0, 400.0)
_PINV_RCOND = 1e-15  # numpy.linalg.pinv's default when neither rcond nor rtol is given


def _capture(df, y, features, cand, budgets):
    """Both paths' inputs for one pair, from the production assembly."""
    model = SuperGLM(family="gaussian", features=features)
    model.fit_reml(df, y)
    grab = {}
    real_curv, real_cells, real_ladder = (
        ops.pair_score_curvature,
        ops.pair_cell_moments,
        ops.penalized_score_statistic_ladder,
    )

    def spy_cells(*a, **k):
        grab["cells"] = real_cells(*a, **k)
        return grab["cells"]

    def spy_curv(B_a, B_b, S_cell, W_cell):
        grab["menus"] = (B_a, B_b)
        return real_curv(B_a, B_b, S_cell, W_cell)

    def spy_ladder(U, V, C, M, S_ti, **k):
        grab["args"] = (U, V, C, M, S_ti, k.get("U_nuisance"))
        return real_ladder(U, V, C, M, S_ti, **k)

    ops.pair_cell_moments = spy_cells
    ops.pair_score_curvature = spy_curv
    ops.penalized_score_statistic_ladder = spy_ladder
    try:
        model.screen_interactions(df, y, candidates=[cand], edf0=budgets)
    finally:
        ops.pair_cell_moments = real_cells
        ops.pair_score_curvature = real_curv
        ops.penalized_score_statistic_ladder = real_ladder
    return grab


def _band_pair(*, L, reps, kind, n_knots, n_band, width, seed):
    """``L`` levels, ``reps`` rows each, ``n_band`` of them inside a band of x.

    A narrow band is what makes a level's spline geometry nearly singular
    without making the level rare, so the pair stays well populated while its
    profiled curvature loses rank.  That is the geometry both paths are worst
    on and the one #257 reports.
    """
    rng = np.random.default_rng(seed)
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.05, 0.95, n)
    if n_band:
        inside = np.isin(g, [f"L{i}" for i in range(n_band)])
        x[inside] = 0.5 + width * rng.uniform(-1.0, 1.0, int(inside.sum()))
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    return _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind=kind, n_knots=n_knots)},
        ("x", "g"),
        BUDGETS,
    )


def _structured_inputs(grab):
    B_a, B_b = grab["menus"]
    S_cell, W_cell = grab["cells"]
    _, _, _, _, S_ti, _ = grab["args"]
    k_a, k_b = B_a.shape[1], B_b.shape[1]
    S_a = np.ascontiguousarray(S_ti[::k_b, ::k_b][:k_a, :k_a])
    return B_a, S_a, S_cell, W_cell, np.argmax(B_b, axis=0)


def _both(grab):
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(U, V, C, M, S_ti, budgets=BUDGETS, U_nuisance=u_m)
    arrow = structured_ladder(spline_cat_moments(*_structured_inputs(grab)), budgets=BUDGETS)
    return dense, arrow


def _dense_matrices(grab):
    """``V_eff`` and ``S`` exactly as :func:`_edge` is handed them."""
    from superglm.screening._score_stat import _solve_psd

    _, V, C, M, S_ti, _ = grab["args"]
    V = 0.5 * (V + V.T)
    V_eff = V - C.T @ _solve_psd(M, C)
    return 0.5 * (V_eff + V_eff.T), 0.5 * (S_ti + S_ti.T)


def _clamped(rungs, edge):
    """The rung for the budget the ladder must CLAMP, chosen by kind not by value.

    :func:`penalized_score_statistic_ladder` emits exactly one rung per budget,
    in order, so the budget identifies the rung.  ``BUDGETS[0] = 2.0`` is below
    every high-edge edf these fixtures reach and ``BUDGETS[-1] = 400.0`` is above
    every low-edge one, so those two budgets clamp by construction.

    Picking "the clamped rung" as the min or max of ``edf0`` at a fixed lambda
    would work today only because the two dense estimators happen to sit
    7.000004 below 9.000019 -- and a searching rung genuinely does land on the
    bracket edge here, since ``_lambda_for_edf`` returns the edge itself when the
    pencil's bracket disagrees with ``_edge``'s.  Once #257's estimator split is
    fixed, a value-based pick would silently start grading the other estimator
    under this name.

    TWO things are therefore checked rather than assumed, because the lambda
    alone does not identify a clamped rung -- a SEARCHING rung that lands on the
    bracket edge has the extreme lambda too, and on the ``bs(8)`` pair one does
    (budget 8.0, at the same 1.065e+05).  The second check is the clamping
    certificate: a rung clamps at the high edge precisely because the edge value
    it reaches still EXCEEDS its budget, and at the low edge because it falls
    SHORT of it.

    WHY IT IS DIRECTIONAL AND WHAT IT DOES NOT CATCH.  "``edf0 != budget``" was
    the obvious form and it does not work: budget 8.0's searching rung reads
    9.000019, so it differs from its budget too.  The directional form catches a
    searching rung at the edge that UNDERSHOOTS -- which is the case
    ``_lambda_for_edf`` can produce, since it returns the edge itself when the
    pencil's bracket disagrees with ``_edge``'s, and it is exactly the residual
    the lambda check misses.  Exercised directly: a hi-edge rung reading 1.8
    against budget 2.0 is refused, and a lo-edge rung reading 400.0 against
    budget 400.0 is refused.  A searching rung at the edge that OVERSHOOTS its
    budget still passes, and nothing here separates that case; the protection is
    that the rung is chosen by BUDGET INDEX rather than by value, so an
    overshooting rung is never the one returned.
    """
    want = max if edge == "hi" else min
    rung = rungs[0] if edge == "hi" else rungs[-1]
    budget = BUDGETS[0] if edge == "hi" else BUDGETS[-1]
    assert rung.lambda0 == want(r.lambda0 for r in rungs), (
        f"budget {budget} no longer clamps to the {edge} edge",
        rung.lambda0,
    )
    reached = rung.edf0 > budget if edge == "hi" else rung.edf0 < budget
    assert reached, (
        f"budget {budget} is no longer unreachable at the {edge} edge, so this rung "
        f"may have SEARCHED rather than clamped and would grade the other estimator",
        rung.edf0,
    )
    return rung


def _same_lambda(arrow_lam, dense_lam):
    """Both paths must be read at ONE lambda or the comparison means nothing.

    The two ladders are meant to share the dense path's 1e+-10 bracket, but they
    compute the scale it hangs off by different arithmetic -- the arrow path from
    ``profiled_trace / (tr(S_a) L)``, where ``L`` is the LEVEL COUNT
    (``p.dims[0]`` is ``U.shape[0]``), and the dense path from
    ``tr(V_eff) / tr(S_ti)`` with ``tr(S_ti) = k_b tr(S_a)`` -- so the two agree
    only because ``L == k_b`` on these fixtures, and identity is a property to
    CHECK rather than a given.  Measured on
    all three fixtures the two agree to 1.4e-16, 2.3e-16 and 4.1e-16 relative,
    which is 1 to 2 ulp: the same bracket by two routes.  ``rel=1e-12`` is
    ~4500 ulp, loose enough that ulp-level arithmetic differences never fire it
    and tight enough that a bracket which genuinely MOVED does.

    Without this, ``|dense - arrow| > 1e-3`` stays green for a reason that has
    nothing to do with #257: a repaired arrow path whose bracket also shifted
    would still "disagree", which is the exact failure the pairing exists to
    prevent.

    THE RATIO IS FORMED DIRECTLY RATHER THAN THROUGH ``pytest.approx``, and the
    reason is a bug this helper shipped with for one revision.
    ``pytest.approx(x, rel=1e-12)`` also carries a DEFAULT ``abs=1e-12`` and
    takes the LOOSER of the two.  The low-edge lambdas here are 1.4e-11 and
    5.9e-12, so that default absolute tolerance is of the same order as the
    lambdas themselves and the check was vacuous at exactly the two points it
    was added for -- a mutation that moved the arrow bracket by 1e-6 relative
    was caught at the high edge and sailed through both low edges.
    """
    rel = abs(arrow_lam - dense_lam) / max(abs(dense_lam), np.finfo(float).tiny)
    assert rel < 1e-12, (
        "the two paths are no longer scoring the same lambda, so their difference "
        "is not evidence about their algorithms",
        arrow_lam,
        dense_lam,
        rel,
    )


def _edge_branch(V_eff, S, lam):
    """Which arm of :func:`_edge` this ``(V_eff, S, lam)`` takes.

    ``_edge`` factors with ``cho_factor`` and falls back to ``pinv``, and the two
    answer DIFFERENT functionals -- ``tr(A^-1 V)`` against ``tr(P V)``.  Which
    one runs is decided by whether ``dpotrf`` meets a non-positive pivot.

    THAT DECISION IS ROUND-OFF ON THIS FIXTURE AND IS NOT ASSERTED.  It was,
    and the measurement says it should not have been.  ``dpotrf`` fails at the
    111th leading minor with a pivot of -3.9e-07, where the ten pivots before it
    are all 3.07e+07 -- so the failure is not a large negative pivot, it is the
    first one that has collapsed to round-off, and it sits at 0.48x the local
    accumulation bound ``k eps`` times the neighbouring pivot.  Three probes:
    the smallest uniform diagonal shift that makes ``dpotrf`` succeed is
    9.54e-07, which is **0.26 ulp** of ``A``'s largest entry; under the file's
    own one-ulp perturbation model of the stored moments the branch flips on
    2 of 600 draws across three seeds; and a blocked right-looking factorization
    run at ten block sizes from 1 to 121 keeps failing at minor 111 but with the
    failing pivot ranging over -1.5e-06 to -5.4e-06 and one variant reaching
    exactly 0.  The step is stable, the margin is not.  A supported build may
    legitimately take the other arm, so the tests select the reference
    FUNCTIONAL by the arm actually taken instead of requiring one.
    """
    try:
        scipy.linalg.cho_factor(V_eff + lam * S, check_finite=False)
    except scipy.linalg.LinAlgError:
        return "pinv"
    return "cholesky"


def _traces(V_eff, S, lam):
    """``tr(P V_eff)`` at ``pinv``'s own cut, re-formed over the same reduction.

    This is the functional ``_edge`` returns on its ``pinv`` branch, as a sum of
    the retained Rayleigh quotients.  There is deliberately no untruncated
    companion: on the Cholesky branch ``_edge`` answers ``tr(A^-1 V_eff)``, and
    an ``eigh`` reconstruction of THAT would be a second unstable reduction of a
    ``cond = 4.4e+16`` matrix with no entitlement to agree with the first.  The
    caller skips that branch rather than grading it.

    WHAT IS AND IS NOT INDEPENDENT HERE, because the earlier wording overstated
    it.  ``np.linalg.pinv(A, hermitian=True)`` reduces through
    ``svd(..., hermitian=True)``, which IS ``eigh`` -- the same LAPACK
    ``?syevd`` on the same matrix as the ``np.linalg.eigh`` below.  So the
    REDUCTION is shared bit for bit and only the trace FORMATION is independent:
    this sums quotients where ``pinv`` forms a matrix product.  A common-mode
    drift in ``?syevd`` would move both together and this comparison would not
    see it.  That is why the certified absolute pin below exists, and it is not
    a formality -- reducing the same ``A`` through ``?gesdd`` instead
    (``np.linalg.svd`` with no ``hermitian=``, a bidiagonalisation rather than a
    tridiagonalisation) gives 7.000004700567314 against this route's
    7.000004364954149, a genuinely independent 3.36e-07 df apart.

    Returns ``(truncated, dropped, gap)``.
    """
    A = V_eff + lam * S
    w, Q = np.linalg.eigh(A)
    quot = np.einsum("ij,jk,ki->i", Q.T, V_eff, Q)
    keep = np.abs(w) > _PINV_RCOND * np.abs(w).max()
    truncated = float(np.sum(quot[keep] / w[keep]))
    gap = np.abs(w[keep]).min() / np.abs(w[~keep]).max() if (~keep).any() else np.inf
    return truncated, int((~keep).sum()), float(gap)


# --------------------------------------------------------------------------
# bs(8), 12 levels, 30 rows in each, four levels inside a 1e-6 band of x.
# k = 121.  At the ladder's HIGH edge, lambda = 1.06521065611e+05.
#
#   exact tr(A^-1 V)       6.999754628429722  (arb, 800 bits, radius 1.3e-221)
#   NOT CERTIFIABLE        r = 3.588e+02, so no multiple of the derivative
#                          bounds a one-ulp step; this point is NOT pinned
#                          against that constant, by either path.
#   tr(P V) at pinv's cut  7.000004364954146  -- what _edge returns, and what
#                          IS certifiable: r_trunc = 2.745e-05 against the
#                          retained spectrum, first-order floor 3.0484e-05 df.
_BS8_HI_ORACLE = 6.999754628429722
_BS8_HI_TRUNCATED_ORACLE = 7.000004364954146
_BS8_HI_TRUNCATED_FLOOR = 3.0484e-05
# 10x the truncated functional's own first-order floor, which is the convention
# the module docstring states for every tolerance in this file.  It is NOT set
# from observed headroom: the two readings it has to cover are 0.0 df here (the
# reconstruction is exact) and 1.5e-06 df on CI's 3.12, which reported this rung
# as 7.000002839965082 -- 0.05x the floor and 200x inside this bound.
_BS8_HI_TRUNCATED_TOL = 10 * _BS8_HI_TRUNCATED_FLOOR
_BS8 = dict(L=12, reps=30, kind="bs", n_knots=8, n_band=4, width=1e-6, seed=0)
# Same bits into both routes, so no input perturbation enters and what remains is
# arithmetic: at most log2(k) k eps ||V||_2 / min_kept|w| = 5.5e-13 df for the
# summation and the Rayleigh quotients.  Named once because it is asserted at two
# places and the guard exists to stop the two drifting apart.
_SAME_BITS_TOL = 1e-10


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257: the dense ladder's clamped rung reports _edge's trace and its "
        "searching rung reports _build_pencil's filter sum; at one lambda on this "
        "pair they read 7.000004 and 9.000019"
    ),
)
def test_the_dense_ladder_reports_one_edf_per_lambda():
    """Two rungs of ONE dense ladder, at ONE lambda, must report one edf.

    The dense ladder answers a CLAMPED rung from ``_edge``'s
    ``tr((V_eff + lam S)^-1 V_eff)`` and a SEARCHING rung from
    ``_build_pencil``'s filter-factor sum.  Those are the same quantity, so at
    the same lambda they are the same number -- and on this pair they are not:
    the ladder returns 7.000004364954146 for the budgets that clamp and
    9.000019310394286 for the budget that searches, both at
    ``lambda = 106521.06561131448``.

    Mechanism, and it is documented rather than inferred.  ``_build_pencil``
    calls ``scipy.linalg.eigh(V_eff, G)``.  SciPy 1.18 selects the driver as
    ``"evr" if b is None else ("gvx" if subset else "gvd")``, so with a second
    operand and no subset that is LAPACK's ``?sygvd``.  Its error bound degrades
    with the condition number of that operand: "These error bounds are large
    when B is ill-conditioned with respect to inversion", and the backward
    stability the bound rests on "is not necessarily true when B is
    ill-conditioned" (LAPACK Users' Guide, *Further Details: Error Bounds for
    the Generalized Symmetric Definite Eigenproblem*).  ``cond(G)`` measures
    5.58e+12 here.  The same section names the alternative -- Cholesky both
    operands with ``xPOTRF`` and take the generalized SVD of the pair with
    ``xTGSJA``, which "can give a tighter error bound than the above bounds when
    B is ill conditioned but A + B is well-conditioned".  That is the
    cancellation-free filter-factor form, and it is not what this module does.

    Measured over 703 searching rungs with a range-valid oracle, across
    ps/cr/bs/ns at 3 to 8 knots, L in 6/12/20, 12 and 30 rows per level, five
    band layouts and two seeds: the two estimators differ by more than 1e-3 df
    on 32.8% of the rungs carrying a narrow band and on 0.0% of the 120
    without one, and on 0.0% of the 232 ``ns`` rungs, whose penalty is full
    rank so ``G`` is well conditioned.  The clamped estimator is the nearer to
    the oracle on 532 of 697 and the searching one on 165.

    THIS TEST CANNOT SAY THE DEFECT IS STILL THE DOCUMENTED ONE.  A strict
    xfail records only that the assertion fails, so a ladder that returned 700
    instead of 9.000019 would keep it green.  Its EXISTENCE is paired with
    ``test_the_dense_ladder_two_estimators_still_disagree_at_one_lambda``, an
    ordinary passing test that goes red when #257 is fixed.  Its MAGNITUDE is
    not pinned by anything, here or there, and cannot portably be: CI's 3.12
    reads this pair as 7.000002839965082 against 10.999974250213997, a 4.0000 df
    gap where this machine measures 2.0000.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    at_lam = {round(r.edf0, 9) for r in dense if r.lambda0 == lam}
    assert len(at_lam) == 1, (
        "one lambda, two edf: the dense ladder's clamped and searching rungs "
        f"disagree at lambda={lam!r}: {sorted(at_lam)}"
    )


def test_the_dense_ladder_two_estimators_still_disagree_at_one_lambda():
    """Say the thing the strict xfail above cannot: the defect is still THERE.

    A strict xfail is satisfied by any failure, so it cannot distinguish the
    documented defect from an unrelated one or from a tenfold worsening.  This
    asserts that the two estimators still differ at one lambda, which is
    precisely the statement that goes red when #257 is fixed.

    It asserts EXISTENCE and not size, and it is named for what it asserts.
    The size is not portable: the gap is 2.000014945440140 df here (clamped
    7.000004364954146 against searching 9.000019310394286) and 4.0000 df on
    CI's Python 3.12, on the same fixture and seed.  1e-3 sits far under both
    and far over a repaired ladder's 0.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    at_lam = [r.edf0 for r in dense if r.lambda0 == lam]
    gap = max(at_lam) - min(at_lam)
    assert gap > 1e-3, ("the two estimators now agree at one lambda", sorted(set(at_lam)))


def test_the_dense_clamped_rung_is_exactly_the_truncated_trace_it_evaluates():
    """What the clamped rung computes, pinned as what it is.

    ``cho_factor`` fails on this ``A`` on every build measured -- it is
    indefinite -- four negative eigenvalues at -8.909e-08 resolved at 50 digits,
    where float64 ``eigvalsh`` reports one at -7.03e-07, against a largest of
    3.07e+10 --
    so ``_edge`` answers from ``numpy.linalg.pinv(A, hermitian=True)``, which
    zeroes 4 of 121 directions at ``rcond = 1e-15`` of the largest, a cut of
    3.0698e-05.  Re-forming that same functional over the same reduction
    reproduces the reported rung to 3.6e-15 df.  The rung is not approximately
    right for ``tr(P V_eff)``; it is exact for it.

    WHY THIS REPLACED TWO PINS AGAINST ``tr(A^-1 V_eff)``.  The certified
    untruncated value here is 6.999754628429722, and the rung sits 2.4974e-04
    df from it -- all of which is the truncation.  Both this path and the arrow
    path used to be pinned against that constant at 1e-2 df.  They are not any
    more: ``r = 3.588e+02`` at this point, so one ulp of the stored moments can
    cross a singularity of ``A`` and no multiple of the derivative bounds the
    exact edf.  A tolerance there would be an assertion about a quantity
    nothing certifies.  The truncated functional is the opposite case --
    ``r_trunc = 2.745e-05`` against a retained spectrum starting at 1.1644 --
    and it is what the ladder actually reports.

    THE TWO TOLERANCES ANSWER DIFFERENT QUESTIONS AND ARE NAMED SEPARATELY.
    ``_SAME_BITS_TOL`` = 1e-10 grades the reconstruction, which feeds both
    routes the SAME bits, so no input perturbation enters and what remains is
    arithmetic: at most ``log2(k) k eps ||V||_2 / min_kept|w|`` = 5.5e-13 df for
    the summation and the Rayleigh quotients.  1e-10 is 180x that, and the test
    asserts that relation from the fixture rather than trusting it.
    ``_BS8_HI_TRUNCATED_TOL`` = 3.0484e-04 grades the ABSOLUTE pin, and answers
    the other question -- how far can the answer move when the moments are
    reassembled elsewhere -- at 10x the 3.0484e-05 df floor.  Neither is fitted
    to observed headroom: one is 180x a computed arithmetic bound, the other 10x
    a computed sensitivity.

    It bites three ways.  Replacing the clamped rung's ``_edge`` trace with the
    searching rung's ``_pencil_edf`` reports 9.000019310394286 and fails by
    2.000 df, 2.0e+10 x the tolerance.  Removing the truncation -- answering
    from ``numpy.linalg.inv`` rather than from ``pinv`` -- reports
    7.000171676295995 and fails by 1.673e-04 df, 1.7e+06 x.  Note what that
    second number is NOT: it is not the certified 6.999754628429722.  A float64
    inverse of an ``A`` at ``cond = 4.4e+16`` has no accuracy left to deliver
    the exact trace, so dropping ``pinv``'s truncation does not recover the
    exact answer -- it trades a stated truncation for an unstated one.  That is
    the strongest form of the disclosure: the truncation is not only a term in
    the answer, it is what makes this rung reproducible at all.  Widening
    pinv's cut to the Array API's ``max(M, N) * eps`` would NOT move it, and
    that is recorded rather than claimed: the two cuts drop the same four
    directions because the gap across them is 2.92e+05.

    THE SAME-BITS CHECK CANNOT SEE A COMMON-MODE DRIFT, WHICH IS WHY THERE IS
    ALSO AN ABSOLUTE PIN.  ``pinv`` and the reconstruction share their reduction
    -- both are ``?syevd`` on the same ``A`` (see :func:`_traces`) -- so if that
    driver drifted, both routes would move together and agree anyway.  The
    reported rung is therefore ALSO pinned against the certified
    ``tr(P V_eff) = 7.000004364954146``, at 10x this functional's own
    3.0484e-05 df first-order floor.  That pin is portable, which is not an
    assumption: CI's Python 3.12 reported this rung as 7.000002839965082, a
    move of 1.5e-06 df -- 0.05x the floor.  The 4.0000 df gap CI measures on
    this pair is entirely the OTHER estimator, which reads 10.999974 there
    against 9.000019 here.

    ON THE CHOLESKY ARM THIS TEST REFUSES TO GRADE ANYTHING, AND SKIPS.  An
    earlier form asserted ``_edge_branch(...) == "pinv"``, which was wrong: the
    smallest diagonal shift that makes ``dpotrf`` succeed here is 0.26 ulp of
    ``A``'s largest entry and the branch flips on 2 of 600 one-ulp draws (see
    :func:`_edge_branch`), so a build that legitimately takes the other arm
    would have failed a test that is not about it.  The next form graded the
    UNTRUNCATED trace on that arm, which was also wrong and for a subtler
    reason: ``cho_solve`` and ``eigh`` are two different unstable reductions of
    an ``A`` at ``cond = 4.4e+16``, and nothing entitles them to agree to
    1e-10.  This file already measures how far apart such routes land --
    ``numpy.linalg.inv``, a third one, differs from ``pinv``'s answer by
    1.67e-04 df, six orders past that tolerance.  So there is no reconstruction
    to compare against on that arm and no certified constant either, ``r`` being
    3.588e+02 at this point.  The honest move is the one the rest of the file
    makes wherever a quantity has no bound: do not pin it.  The skip carries the
    reason, so a build that lands there says why rather than failing on
    round-off.

    The last assertion is the one that keeps the disclosure honest: the
    truncation, 2.4974e-04 df, must stay ABOVE the point's own first-order
    3.0484e-05 df floor -- 8.19x it -- so it cannot be read as a rounding.  If
    a future ``pinv`` cut left it under the floor, the claim that the
    truncation is a term in the answer would stop being supported, and this
    goes red rather than the docstring going quietly stale.
    """
    grab = _band_pair(**_BS8)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    rung = _clamped(dense, "hi")
    lam, clamped = rung.lambda0, rung.edf0
    truncated, dropped, gap = _traces(V_eff, S, lam)
    assert dropped == 4, ("pinv no longer truncates this A", dropped)
    assert gap > 1e4, ("the truncation is now marginal, not a clean gap", gap)
    if _edge_branch(V_eff, S, lam) != "pinv":
        pytest.skip(
            "cho_factor accepted this A on this build, so _edge returns "
            "tr(A^-1 V_eff) and there is nothing here that can grade it: r = 3.588e+02 "
            "at this point so no certified constant exists, and cho_solve against an "
            "eigh reconstruction are two unstable reductions of a cond = 4.4e+16 matrix "
            "with no entitlement to agree -- numpy.linalg.inv, a third such route, "
            "lands 1.67e-04 df away. This test's subject is the truncation, and on "
            "this arm there is none."
        )
    w = np.linalg.eigvalsh(V_eff + lam * S)
    kept = np.abs(w) > _PINV_RCOND * np.abs(w).max()
    arithmetic = (
        float(np.log2(V_eff.shape[0]))
        * V_eff.shape[0]
        * np.finfo(float).eps
        * float(np.linalg.norm(V_eff, 2))
        / float(np.abs(w[kept]).min())
    )
    assert _SAME_BITS_TOL > 10 * arithmetic, (
        "the same-bits arithmetic floor has grown past the bound this fixture was sized for",
        arithmetic,
    )
    assert clamped == pytest.approx(truncated, abs=_SAME_BITS_TOL), (
        "_edge does not reproduce tr(P V_eff), the functional the pinv arm evaluates",
        clamped,
        truncated,
    )
    assert clamped == pytest.approx(_BS8_HI_TRUNCATED_ORACLE, abs=_BS8_HI_TRUNCATED_TOL), (
        "the clamped rung has left the certified truncated trace by more than 10x "
        "this functional's first-order floor",
        clamped - _BS8_HI_TRUNCATED_ORACLE,
    )
    assert abs(clamped - _BS8_HI_ORACLE) > _BS8_HI_TRUNCATED_FLOOR, (
        "the truncation has stopped being a term in the answer",
        clamped - _BS8_HI_ORACLE,
    )


# --------------------------------------------------------------------------
# ps(8), 20 levels, 30 rows in each, four levels inside a 1e-3 band of x.
# k = 209.  This is issue #257's own geometry.
#
# LOW edge, lambda = 1.3814328670859514e-11:
#   exact edf              172.90456267038329  (arb, 800 bits, radius 5.9e-214)
#   first-order floor D        5.6825e-03 df  (D = 5.5544e-03, r = 2.192e-02)
#   cho_factor succeeds, pinv would drop 0 of 209 -- no truncation here
#
# HIGH edge, lambda = 1381432867.0859513:
#   exact edf              15.888406216250933 (arb, 800 bits, radius 2.4e-220)
#   NOT CERTIFIABLE        r = 2.692e+02
#
# THE HIGH EDGE IS NOT PINNED AGAINST EITHER ORACLE.  What is pinned is the
# ill-posedness, and by an observable rather than by a scale comparison: the
# V-mass carried by the directions of A that are below the round-off the
# penalty term alone injects.  3.1083 df of it here against 0 directions at
# this pair's own low edge.
_PS8 = dict(L=20, reps=30, kind="ps", n_knots=8, n_band=4, width=1e-3, seed=3)
_PS8_LO_ORACLE = 172.90456267038329
_PS8_LO_FLOOR = 5.6825e-03
_PS8_LO_TOL = 6e-2  # 10.6x the first-order floor; the arrow rung misses by 0.525 df
# There is deliberately no _PS8_HI_ORACLE constant.  One existed, was orphaned
# when the high-edge pins were dropped, revived by a bracket, and orphaned again
# when that bracket went.  The value is in the comment block above, where a
# reader wants it; a module-level constant that nothing reads is not flagged by
# ruff and reads as a pin that is no longer made.


def test_the_dense_clamped_rung_matches_the_certified_low_edge_oracle():
    """At the LOW edge the dense clamped rung is right to 1.27e-04 df.

    Pinned at ``_PS8_LO_TOL`` = 6e-2, which is 10.6x this point's FIRST-ORDER
    floor of 5.6825e-03 df and 8.75x below the 0.525 df the arrow path misses it
    by.  ``r = 2.192e-02`` here, so the perturbed inverse at least exists and is
    bounded.

    THAT TOLERANCE IS AN ENGINEERING CHOICE AND IS NOT CERTIFIED, and the
    distinction is not cosmetic.  It was called a "certified finite-perturbation
    floor"; it is not one.  The floor is a derivative at the unperturbed ``A``,
    the only rigorous enclosure this point admits is the one-ulp ball radius of
    1.5109 df -- 266x D and 25x this tolerance -- and no analysis here bounds
    the exact edf inside 6e-2 on a build that reassembles these near-singular
    moments differently.  What supports it is evidence rather than proof, and
    the evidence is stated so a reader can weigh it: three interpreters have
    graded this rung against this constant, with errors of 1.27e-04 df here and
    9.4e-05 df on CI's 3.14 -- CI's 3.12 passed, and its value is not recorded
    because only failures print theirs; a re-derivation of the oracle by an
    independent high-precision route moved the point by 1.09e-04 df; and 40
    random one-ulp draws reached 5.61e-03 df, 1.01x D, so D is very nearly
    attained rather than an underestimate.  The gap between "6e-2 has never been
    approached in any grading yet run" and "the exact edf is provably within
    6e-2" is real, and this file claims only the first.  The tolerance was 1e-2,
    which is 1.8x the floor -- not inverted like the two high-edge pins were,
    but not a margin either.

    ``_edge`` takes the CHOLESKY branch here -- ``A`` is positive definite at
    ``lambda = 1.38e-11`` -- but the branch is NOT asserted, because at this
    edge it does not matter which arm runs.  ``pinv`` would discard 0 of 209
    directions, so ``P`` is a true inverse and ``tr(P V_eff)`` and
    ``tr(A^-1 V_eff)`` are the SAME functional; ``dropped == 0`` below is what
    makes that so, and it is asserted.  This is the load-bearing difference from
    the high edge, where the two arms answer different quantities and the branch
    decision sits at 0.26 ulp.  For the record the margin here is larger but not
    itself certified: ``A``'s smallest eigenvalue is 1.82e-13 against a
    ``k eps ||A||_2`` backward-error scale of 1.70e-13, a ratio of 1.07, and
    600 one-ulp draws across three seeds flipped it 0 times.

    Measured over 20 sweep points where both paths clamp to one lambda AND the
    exact answer is attainable to 1e-2 df, the clamped rung's error has median
    6.9e-05 df and maximum 2.7e-02 df.

    It bites: replacing the ladder's relative bracket scale with an absolute
    1.0 reads 172.49628147315673 and fails by 0.408 df, 6.8x the tolerance.
    """
    grab = _band_pair(**_PS8)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    rung = _clamped(dense, "lo")
    lam, got = rung.lambda0, rung.edf0
    _, dropped, _ = _traces(V_eff, S, lam)
    assert dropped == 0, ("this edge is supposed to be truncation-free", dropped)
    assert _PS8_LO_TOL > 10 * _PS8_LO_FLOOR, "tolerance must clear the first-order floor"
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=_PS8_LO_TOL)


def test_the_arrow_ladder_still_disagrees_with_the_dense_path_at_the_low_edge():
    """Say the thing the strict xfail below cannot: the defect is still THERE.

    Both paths are read from the SAME assembly and compared to each other, so
    no cross-environment constant enters.  A repaired arrow path would agree
    with the dense one to the dense path's own accuracy, ~1e-4 df; the
    threshold is 1e-3.

    Existence, not size, and not even sign -- CI showed both are properties of
    the interpreter rather than of the algorithm.  The arrow reading here is
    172.37960656016872 against a dense 172.90468989250675, 0.525 df LOW; on
    CI's 3.12 it is 2.780 df low; on CI's 3.14 it is 172.9870 against 172.9045,
    0.0825 df HIGH.  See the module docstring.

    The two readings are checked to be at ONE lambda first -- see
    :func:`_same_lambda`.  Without that the disagreement is not evidence about
    the algorithms at all.
    """
    grab = _band_pair(**_PS8)
    dense, arrow = _both(grab)
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    rung = _clamped(dense, "lo")
    _same_lambda(lam, rung.lambda0)
    reference = rung.edf0
    assert abs(reference - got) > 1e-3, ("the two paths now agree at the low edge", got, reference)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: the arrow ladder's rank-minus-trace difference is 0.525 df "
        "low at the ladder's low edge, where the point's first-order floor is "
        "5.6825e-03 df"
    ),
)
def test_the_arrow_ladder_matches_the_certified_low_edge_oracle():
    """The arrow ladder's LOW edge is wrong; its EXISTENCE is paired above.

    172.37960656016872 against a certified 172.90456267038329 -- 0.525 df low,
    92x this point's first-order 5.6825e-03 df floor.  Nothing about the moments
    excuses it: both paths receive the same ones and the dense path reads
    172.90468989250675 from them.

    That 0.525 df is NOT pinned, here or in the paired test, and cannot be: on
    CI's 3.14 the same fixture reads 0.0825 df HIGH.  What the pair asserts is
    that a disagreement is still there.

    The tolerance is the same ``_PS8_LO_TOL`` the dense pin above uses -- an
    engineering choice at 10.6x the first-order floor rather than 1.8x it, not a
    certification; the reasoning is in that test's docstring.

    Over 20 sweep points where both paths clamp to one lambda and the point is
    attainable to 1e-2 df, the arrow error has median 1.17 df and maximum
    30.74 df, and it lands inside the attainable band on 3 of the 20 against the
    dense path's 19.
    """
    _, arrow = _both(_band_pair(**_PS8))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert _PS8_LO_TOL > 10 * _PS8_LO_FLOOR, "tolerance must clear the first-order floor"
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=_PS8_LO_TOL)


def test_the_high_edge_divergence_grades_neither_path():
    """The two paths disagree at the high edge, and nothing here says which is right.

    NAMED FOR WHAT IT ASSERTS.  This was
    ``test_the_high_edge_is_not_determined_by_the_assembled_moments``, and that
    name described the ARGUMENT below rather than the assertion: a
    ``|dense - arrow| > 1e-3`` check can stay green after the point becomes
    well-conditioned, and can go red while it stays uncertifiable, so it does not
    establish non-determinacy.  The argument is still the reason there is nothing
    better to assert, and it is kept in full for that reason -- but the test now
    claims only the divergence it checks.

    #257's headline number is BELOW this point's own noise floor.
    The issue reports 11.57 df of arrow-vs-dense divergence at the high edge and
    treats it as a defect.  On this geometry the two paths differ by 3.71 df --
    15.000025261210718 against 18.706953863891897.

    WHAT THIS TEST NO LONGER DOES.  It used to assert
    ``|dense - oracle| < 5.06`` and ``|arrow - oracle| < 5.06``.  5.06 df was a
    10-draw lower bound on a random spread; the derivative bound at this point
    is 70.7 df and even THAT is not a bound, because ``r = 2.692e+02`` -- one
    ulp of the stored moments is 269x the smallest eigenvalue of ``A``, so a
    one-ulp step can cross a singularity and the exact edf is not bounded by
    any multiple of a derivative taken at the unperturbed point.  Those two
    assertions pinned a quantity nothing certifies.  They are gone.

    THE STRONGEST STATEMENT AVAILABLE IS A REFUSAL, AND IT IS NOT COMPUTABLE
    HERE.  Put a BALL of radius ``e |x|`` on every stored entry of ``V_eff`` and
    ``S`` and solve in arb: the whole one-ulp neighbourhood is carried at once,
    with no norms, no linearization and no Neumann series, and the output radius
    would be a genuine bound.  At this point arb REFUSES -- it cannot prove
    ``A`` invertible anywhere in that ball, raising ``ZeroDivisionError:
    singular matrix in solve()``.  So there is no bound to pin, and the reason
    is not a lack of effort.  (Of the four points this file grades, three refuse
    -- both high edges and the ``ns(6)`` low edge -- and the one that admits a
    solve, the ``ps(8)`` low edge, returns a radius of 1.5109 df, 266x its own
    derivative.  Ball arithmetic through an LU solve is very pessimistic at
    these condition numbers, so a refusal is a FAILURE TO CERTIFY and not a
    proof of singularity.  Both facts belong in the record.)

    WHY THERE IS NO RUNTIME OBSERVABLE EITHER, stated rather than papered over.
    The natural one is the V-mass the round-off subspace carries: take the
    directions of ``A`` below ``lam eps ||S||_2`` and measure
    ``||Q_d' V_eff Q_d||_F / min|w_d|``, which is invariant to rotation WITHIN
    that subspace.  Rotation-invariance is not enough, because MEMBERSHIP is not
    stable -- the threshold sits far below the eigensolver's own resolution
    (``k eps ||A||_2 = 9.87e-04``, 209x the threshold), so which directions fall
    inside it is round-off.  Swept over the threshold on this pair:

        t / base      0.01    0.1     0.3     1.0     3.0     10      100
        high edge     0       0       3.108   3.108   24.41   24.41   24.41
        low  edge     0       0       0       0       0       0       0

    The verdict "the high edge carries mass and the low edge carries none" is
    true over two and a half decades and FALSE below 0.3x, so the observable
    cannot carry an assertion.  It is reported, not asserted.  For contrast the
    ``bs(8)`` high edge reads 4.00e-04 to 9.48e-04 df across the whole sweep --
    3300x less on a pair whose lambda is just as extreme, which is the evidence
    that extreme lambda is not the sufficient condition.

    WHAT IS ASSERTED is only that the two paths still disagree at one lambda,
    ``|dense - arrow| > 1e-3`` against a measured 3.7069 df.  Not a bracket: an
    earlier form asserted ``3.0 < |dense - arrow| < 4.5``, and CI refuted
    magnitude pins on the sibling fixtures by moving them 859x and reversing
    their sign, so no magnitude is claimed at this edge either.  This assertion
    catches the paths CONVERGING and nothing else -- it does not catch either
    one worsening, and it does not say which is right.  Saying which would need
    a bound at this point, and the paragraphs above are the record that there is
    not one.
    """
    grab = _band_pair(**_PS8)
    dense, arrow = _both(grab)
    assert arrow is not None
    rung = _clamped(dense, "hi")
    d = rung.edf0
    lam = max(x.lambda0 for x in arrow)
    _same_lambda(lam, rung.lambda0)
    a = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert abs(d - a) > 1e-3, ("the two paths now agree at the high edge", d, a)


# --------------------------------------------------------------------------
# ns(6), 6 levels, 12 rows in each, two inside a 1e-1 band of x.  k = 35.
# ``ns``'s penalty is full rank, so the arrow ladder's ``free`` set is empty
# and ``block_ranks`` returns the unpenalized count for every level.
#
# LOW edge, lambda = 5.894735595624617e-12:
#   exact edf             30.988667391753435 (arb, 800 bits, radius 2.8e-180)
#   first-order floor D       8.2863e-05 df  (D = 8.2837e-05, r = 2.819e-04)
#   cho_factor succeeds, pinv would drop 0 of 35 -- no truncation here
_NS6 = dict(L=6, reps=12, kind="ns", n_knots=6, n_band=2, width=1e-1, seed=3)
_NS6_LO_ORACLE = 30.988667391753435
_NS6_LO_FLOOR = 8.2863e-05
# TWO tolerances, not one shared constant.  They happen to want the same number
# today and they are not the same quantity: one bounds how far the two DENSE
# estimators may sit apart at a lambda, the other how far the arrow rung may sit
# from the certified oracle.  Sharing a name means tightening either one
# silently retargets the other.
_NS6_PENCIL_TOL = 1e-3  # 12.1x the largest first-order floor the ladder ranges over
_NS6_LO_TOL = 1e-3  # 12.1x the first-order floor; the arrow rung misses by 30.74 df


def test_the_two_dense_estimators_agree_where_the_pencil_is_conditioned():
    """The cross-check that #257 asks for, pinned where it must stay SILENT.

    ``|_edge trace - _pencil_edf|`` at one lambda is a runtime invariant with no
    second path in it: both quantities come out of work the ladder already does,
    and where it stays quiet the accurate estimator's error is bounded.
    Measured on 388 sweep pairs with at least 12 rows per level and at least 12
    levels, at the ladder's low edge: a gate at 0.1 df fires on 4.4% of them and
    the largest error it lets through is 1.85e-02 df.

    The invariant is not vacuous and not universal, which is why it is pinned on
    a family where it holds rather than asserted everywhere.  It is silent on
    0 of 232 ``ns`` searching rungs and 0 of 120 without a narrow band, and
    fires on 32.8% of the banded ones -- exactly the split LAPACK's error bound
    for ``?sygvd`` predicts, since ``ns``'s penalty is full rank so ``G`` stays
    conditioned (``cond(G) = 6.41e+02`` here against 5.58e+12 on the ``bs(8)``
    pair above).

    Each rung is re-read with the OTHER estimator at its own lambda rather than
    rungs being grouped by lambda: grouping only compares two estimators where a
    searching rung happens to land on a bracket edge, which on this pair it
    never does, and the assertion would then be vacuous on the three rungs that
    actually search.

    It bites.  Two mutations of ``_build_pencil``, each reverting a rule that
    module's own docstring states, fail it: deriving ``s`` as ``1 - share``
    instead of carrying both transformed terms fails by 6.361 df, and taking
    the pencil metric as ``V_eff`` rather than ``V_eff + balance * S`` fails by
    30.033 df.  Three others do NOT move this pair and are recorded so the test
    is not credited with catching them: dropping the balance before summing
    (``balance = 1.0``), which
    ``test_a_curvature_that_dwarfs_its_penalty_keeps_the_penalty`` does catch;
    removing the clip of the share to [0, 1]; and reverting ``_rank_floor`` to
    the fixed 1e-12 it started with.  The last two are caught by neither that
    test nor ``test_screening_is_invariant_to_the_units_of_a_numeric_margin``,
    which is a gap this test does not close either.

    ``_NS6_PENCIL_TOL`` IS 1e-3, WHICH IS 12.1x THIS PAIR'S LARGEST FIRST-ORDER
    FLOOR.  This paragraph argued 1e-2 and 121x after the constant had already
    been tightened by a round of review, so both figures were wrong by an order.
    The assertion ranges over all five rungs, whose floors span nine orders --
    9.16e-14 df at ``lambda = 74.3`` up to 8.29e-05 df at ``lambda = 5.9e-12``
    -- and it exists to catch 6.361 and 30.033 df mutations.  So any bound
    between the largest floor it ranges over (8.29e-05) and the smallest
    mutation it must catch (6.361) does the same work: 1e-3 sits 12.1x above the
    first and 6.4e+03 x below the second.  The observed worst over the ladder is
    1.65e-13 df.  What matters is that the bound is ABOVE the largest floor it
    ranges over, which is the property the high-edge tolerances lacked.
    """
    from superglm.screening._score_stat import _edge

    grab = _band_pair(**_NS6)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    worst = 0.0
    for r in dense:
        other, _ = _edge(V_eff, S, r.lambda0)
        worst = max(worst, abs(float(other) - r.edf0))
    assert _NS6_PENCIL_TOL > 10 * _NS6_LO_FLOOR, "tolerance must clear the first-order floor"
    assert worst < _NS6_PENCIL_TOL, ("dense estimators disagree by", worst)


def test_the_arrow_ladder_still_disagrees_on_a_full_rank_penalty():
    """Say the thing the strict xfail below cannot: the defect is still THERE.

    The arrow ladder reads 0.24516151291386734 at this pair's low edge against
    a dense 30.988679169193755 -- 30.74 df low, on a point whose first-order
    floor is 8.2863e-05 df.  On CI's Python 3.12 the same fixture reads 30.9529,
    0.036 df low: the collapse does not happen there at all, so the SIZE is one
    interpreter's.  What holds on every run is that the two paths disagree by
    more than the dense path's own ~1e-5 df, and that is what this asserts,
    against the dense reading from the same assembly.

    The mechanism is visible in the two halves of the difference:
    ``rank - lambda tr(A^-1 S)`` is 35.00 - 34.75 where the exact split is
    35.00 - 4.01, so the arrow inverse's penalty trace is 8.7x too large at a
    lambda of 5.9e-12.  The certified exact ``lam tr(A^-1 S)`` at this point is
    4.011319142520 (arb, radius 2.0e-183).

    It is not an edf-only defect.  ``z = (T - edf0) / sqrt(2 edf0)`` divides by
    the edf, so on THIS interpreter the same pair is ranked at z = 0.13 by the
    dense path and z = 35.42 by the arrow path at a budget of 2.
    """
    grab = _band_pair(**_NS6)
    dense, arrow = _both(grab)
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    rung = _clamped(dense, "lo")
    _same_lambda(lam, rung.lambda0)
    reference = rung.edf0
    assert abs(reference - got) > 1e-3, ("the two paths now agree at the low edge", got, reference)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: on a full-rank penalty the arrow ladder's low edge collapses "
        "to 0.245 df against a certified 30.989, moving z by 35.3"
    ),
)
def test_the_arrow_ladder_low_edge_survives_a_full_rank_penalty():
    """The arrow path should read this well-posed point, and does not.

    Pinned at ``_NS6_LO_TOL`` = 1e-3, 12.1x the first-order 8.2863e-05 df floor,
    which TIGHTENS the 1e-2 this landed with: the point is determined 121x
    better than that bound admitted, and when the arrow path is fixed this
    xfail should flip against a bound the point can actually support.  The
    dense path already reads it to 1.18e-05 df, inside the new bound by 85x.
    ``r = 2.819e-04`` here, the smallest of the four points, but the one-ulp
    ball still REFUSES this one, so 12.1x a derivative is an engineering choice
    here as everywhere else in this file and not a certification.

    Today's failure size is NOT pinned, here or anywhere: CI's 3.12 reads the
    same fixture 0.036 df low against 30.74 df here, an 859x spread.  Its
    EXISTENCE is paired with
    ``test_the_arrow_ladder_still_disagrees_on_a_full_rank_penalty``.
    """
    _, arrow = _both(_band_pair(**_NS6))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert _NS6_LO_TOL > 10 * _NS6_LO_FLOOR, "tolerance must clear the first-order floor"
    assert got == pytest.approx(_NS6_LO_ORACLE, abs=_NS6_LO_TOL)
