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

and then ``|delta edf| <= D / (1 - r)``.  Measured, that condition SPLITS the
four points this file grades, and it is why two of them are no longer pinned
against the untruncated oracle at all:

    point       lam         min|eig A|   ||dA||      r          certified?
    bs(8) hi   1.065e+05    1.590e-08    3.196e-05   2.010e+03  NO
    ps(8) lo   1.381e-11    1.823e-13    3.996e-15   2.192e-02  yes, 1/(1-r) = 1.0224
    ps(8) hi   1.381e+09    5.616e-07    3.436e-05   6.119e+01  NO
    ns(6) lo   5.895e-12    2.223e-12    6.266e-16   2.819e-04  yes, 1/(1-r) = 1.0003

At both HIGH edges one ulp of the stored moments is 61x to 2010x the smallest
eigenvalue of the assembled ``A``, so a one-ulp step can cross a singularity and
NOTHING bounds the exact edf there -- not D, and not a multiple of D.  Those two
points are therefore not pinned against ``tr(A^-1 V_eff)``.  What replaces them
is below.

    point      CERTIFIED finite floor   tolerance      mutation it must catch
    ps(8) lo   5.6825e-03 df            6e-2   10.6x   0.525 df  (arrow rung)
    ns(6) lo   8.2863e-05 df            1e-3   12.1x   30.74 df  (arrow rung)
    bs(8) hi   -- see the truncated functional below, 3.0484e-05 df --
    ps(8) hi   -- not pinned; the ill-posedness itself is, and is measured --

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
both points that rule is inert, since each carries exactly one negative
eigenvalue (-1.59e-08 and -5.62e-07) and both sit under the cut.  Inert here is
not inert everywhere, and nothing counts them.

**THE THREE ``xfail(strict=True)`` TESTS DO NOT PIN THEIR OWN DEFECT.**  A
strict xfail records that a test fails; it cannot say the failure is still the
one documented, so a regression from 0.525 df to 500 df keeps it green.  Each is
therefore PAIRED with an ordinary passing test that brackets the defect's
current magnitude, and those brackets go red on repair AND on worsening.  They
are stated as brackets rather than as tolerances on purpose: the arrow readings
and the pencil rung have no derived accuracy bound to be pinned against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Spline
from superglm.screening._score_stat import penalized_score_statistic_ladder
from superglm.screening._structured import spline_cat_moments, structured_ladder

# Every test here fits a model through the public screen, which is what
# pyproject's ``slow`` marker is defined for.  The cost is the same class as the
# tests already carrying it -- 0.03 to 0.46 s each, against 0.03 to 0.39 s for
# the five slow-marked cases in tests/test_screening_worth_gate.py.
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


def _truncated_trace(V_eff, S, lam):
    """``tr(P V_eff)`` at ``pinv``'s own cut, reached by a different route.

    ``pinv`` reduces with ``eigh`` and then forms a matrix product; this sums
    the retained Rayleigh quotients directly.  Same functional, same bits in,
    independent arithmetic -- so a disagreement is the estimator changing, not
    the moments moving.  Returns the trace, how many directions were dropped,
    and the spectral gap across the cut.
    """
    A = V_eff + lam * S
    w, Q = np.linalg.eigh(A)
    keep = np.abs(w) > _PINV_RCOND * np.abs(w).max()
    Qk = Q[:, keep]
    trace = float(np.sum(np.einsum("ij,jk,ki->i", Qk.T, V_eff, Qk) / w[keep]))
    gap = np.abs(w[keep]).min() / np.abs(w[~keep]).max() if (~keep).any() else np.inf
    return trace, int((~keep).sum()), float(gap)


# --------------------------------------------------------------------------
# bs(8), 12 levels, 30 rows in each, four levels inside a 1e-6 band of x.
# k = 121.  At the ladder's HIGH edge, lambda = 1.06521065611e+05.
#
#   exact tr(A^-1 V)       6.999754628429722  (arb, 800 bits, radius 1.3e-221)
#   NOT CERTIFIABLE        r = 2.010e+03, so no multiple of the derivative
#                          bounds a one-ulp step; this point is NOT pinned
#                          against that constant, by either path.
#   tr(P V) at pinv's cut  7.000004364954146  -- what _edge returns, and what
#                          IS certifiable: r_trunc = 2.745e-05 against the
#                          retained spectrum, certified floor 3.0484e-05 df.
_BS8_HI_ORACLE = 6.999754628429722
_BS8_HI_TRUNCATED_FLOOR = 3.0484e-05
_BS8 = dict(L=12, reps=30, kind="bs", n_knots=8, n_band=4, width=1e-6, seed=0)


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
    calls ``scipy.linalg.eigh(V_eff, G)``, LAPACK's ``?sygv``, whose error
    bound degrades with the condition number of the second operand -- "there
    may be a significant loss of accuracy if B is ill-conditioned with respect
    to inversion" (LAPACK Users' Guide, Error Bounds for the Generalized
    Symmetric Definite Eigenproblem).  ``cond(G)`` measures 5.58e+12 here.
    LAPACK's own recommended alternative for a positive-definite A is the GSVD
    of the two Cholesky factors, which is Van Loan (1976) and is exactly the
    cancellation-free filter-factor form; it is not what this module does.

    Measured over 703 searching rungs with a range-valid oracle, across
    ps/cr/bs/ns at 3 to 8 knots, L in 6/12/20, 12 and 30 rows per level, five
    band layouts and two seeds: the two estimators differ by more than 1e-3 df
    on 32.8% of the rungs carrying a narrow band and on 0.0% of the 120
    without one, and on 0.0% of the 232 ``ns`` rungs, whose penalty is full
    rank so ``G`` is well conditioned.  The clamped estimator is the nearer to
    the oracle on 532 of 697 and the searching one on 165.

    THIS TEST CANNOT SAY THE DEFECT IS STILL THE DOCUMENTED ONE.  A strict
    xfail records only that the assertion fails, so a ladder that returned 700
    instead of 9.000019 would keep it green.  The magnitude is bracketed by
    ``test_the_dense_ladder_two_estimator_gap_is_two_degrees_of_freedom``,
    which is an ordinary passing test and goes red both ways.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    at_lam = {round(r.edf0, 9) for r in dense if r.lambda0 == lam}
    assert len(at_lam) == 1, (
        "one lambda, two edf: the dense ladder's clamped and searching rungs "
        f"disagree at lambda={lam!r}: {sorted(at_lam)}"
    )


def test_the_dense_ladder_two_estimator_gap_is_two_degrees_of_freedom():
    """Bracket the defect the strict xfail above cannot describe.

    At ``lambda = 106521.06561131448`` the clamped rungs read
    7.000004364954146 and the searching rung 9.000019310394286, a gap of
    2.000014945440140 df.  This asserts ``1.5 < gap < 2.5``.

    That is a BRACKET, not a tolerance, and the distinction is deliberate.
    The searching rung comes out of ``?sygv`` at ``cond(G) = 5.58e+12``, where
    LAPACK's own error bound is the one quoted above and no accuracy statement
    is available to pin it against; what IS available is that the defect today
    is two whole degrees of freedom, and both its repair (gap to 0) and its
    worsening (gap past 2.5) are things this suite should notice.  The width
    is one quarter of the gap on each side, which is 2500x the certified floor
    of the clamped half.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    at_lam = [r.edf0 for r in dense if r.lambda0 == lam]
    gap = max(at_lam) - min(at_lam)
    assert 1.5 < gap < 2.5, ("the two-estimator gap is no longer 2 df", sorted(set(at_lam)))


def test_the_dense_clamped_rung_is_exactly_the_truncated_trace_it_evaluates():
    """What the clamped rung computes, pinned as what it is.

    ``cho_factor`` FAILS on this ``A`` -- it is indefinite, smallest eigenvalue
    -7.03e-07 against a largest of 3.07e+10 -- so ``_edge`` answers from
    ``numpy.linalg.pinv(A, hermitian=True)``, which zeroes 4 of 121 directions
    at ``rcond = 1e-15`` of the largest, a cut of 3.0698e-05.  Reconstructing
    that same functional by an independent route reproduces the reported rung
    to 3.6e-15 df.  The rung is not approximately right for ``tr(P V_eff)``;
    it is exact for it.

    WHY THIS REPLACED TWO PINS AGAINST ``tr(A^-1 V_eff)``.  The certified
    untruncated value here is 6.999754628429722, and the rung sits 2.4974e-04
    df from it -- all of which is the truncation.  Both this path and the arrow
    path used to be pinned against that constant at 1e-2 df.  They are not any
    more: ``r = 2.010e+03`` at this point, so one ulp of the stored moments can
    cross a singularity of ``A`` and no multiple of the derivative bounds the
    exact edf.  A tolerance there would be an assertion about a quantity
    nothing certifies.  The truncated functional is the opposite case --
    ``r_trunc = 2.745e-05`` against a retained spectrum starting at 1.1644 --
    and it is what the ladder actually reports.

    THE TOLERANCE IS 1e-10 AND IT IS NOT THE 3.0484e-05 df FLOOR.  Those are
    different comparisons.  The floor answers "how far can the ANSWER move when
    the moments are reassembled elsewhere"; this test feeds both routes the
    SAME bits, so no input perturbation enters and what remains is arithmetic:
    at most ``log2(k) k eps ||V||_2 / min_kept|w|`` = 5.5e-13 df for the
    summation and the Rayleigh quotients.  1e-10 is 180x that.  The floor is
    still asserted below, against the tolerance, so the two cannot be confused.

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

    The last assertion is the one that keeps the disclosure honest: the
    truncation, 2.4974e-04 df, must stay ABOVE the point's own certified
    3.0484e-05 df floor -- 8.19x it -- so it cannot be read as a rounding.  If
    a future ``pinv`` cut left it under the floor, the claim that the
    truncation is a term in the answer would stop being supported, and this
    goes red rather than the docstring going quietly stale.
    """
    grab = _band_pair(**_BS8)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    lam = max(r.lambda0 for r in dense)
    clamped = min(r.edf0 for r in dense if r.lambda0 == lam)
    trace, dropped, gap = _truncated_trace(V_eff, S, lam)
    assert dropped == 4, ("pinv no longer truncates this A", dropped)
    assert gap > 1e4, ("the truncation is now marginal, not a clean gap", gap)
    assert 1e-10 > 3.0e-13, "the same-bits arithmetic floor must sit under the bound"
    assert clamped == pytest.approx(trace, abs=1e-10)
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
#   CERTIFIED floor        5.6825e-03 df  (D = 5.5544e-03, r = 2.192e-02)
#   cho_factor succeeds, pinv would drop 0 of 209 -- no truncation here
#
# HIGH edge, lambda = 1381432867.0859513:
#   exact edf              15.888406216250933 (arb, 800 bits, radius 2.4e-220)
#   NOT CERTIFIABLE        r = 6.119e+01
#
# THE HIGH EDGE IS NOT PINNED AGAINST EITHER ORACLE.  What is pinned is the
# ill-posedness, and by an observable rather than by a scale comparison: the
# V-mass carried by the directions of A that are below the round-off the
# penalty term alone injects.  3.1083 df of it here against 0 directions at
# this pair's own low edge.
_PS8 = dict(L=20, reps=30, kind="ps", n_knots=8, n_band=4, width=1e-3, seed=3)
_PS8_LO_ORACLE = 172.90456267038329
_PS8_LO_FLOOR = 5.6825e-03
_PS8_LO_TOL = 6e-2  # 10.6x the certified floor; the arrow rung misses by 0.525 df
_PS8_HI_ORACLE = 15.888406216250933


def test_the_dense_clamped_rung_matches_the_certified_low_edge_oracle():
    """At the LOW edge the dense clamped rung is right to 1.27e-04 df.

    Pinned at ``_PS8_LO_TOL`` = 6e-2, which is 10.6x this point's CERTIFIED
    finite-perturbation floor of 5.6825e-03 df and 8.75x below the 0.525 df the
    arrow path misses it by.  Certified means the Neumann condition holds:
    ``r = 2.192e-02``, so ``D / (1 - r)`` really does bound a one-ulp step, and
    the inflation it costs is 2.2%.

    The tolerance was 1e-2, which is 1.8x the floor -- not inverted like the
    two high-edge pins were, but not a margin either, and an independent
    re-derivation of ``_PS8_LO_ORACLE`` on another machine moved the point by
    1.09e-04 df, so the constant itself is only good to a few floor-widths.

    ``_edge`` takes the CHOLESKY branch here -- ``A`` is positive definite at
    ``lambda = 1.38e-11`` and ``pinv`` would discard 0 of 209 directions -- so
    unlike the high edge this rung really does evaluate ``tr(A^-1 V_eff)``
    with no truncation in it.

    Measured over 20 sweep points where both paths clamp to one lambda AND the
    exact answer is attainable to 1e-2 df, the clamped rung's error has median
    6.9e-05 df and maximum 2.7e-02 df.

    It bites: replacing the ladder's relative bracket scale with an absolute
    1.0 reads 172.49628147315673 and fails by 0.408 df, 6.8x the tolerance.
    """
    grab = _band_pair(**_PS8)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    lam = min(r.lambda0 for r in dense)
    _, dropped, _ = _truncated_trace(V_eff, S, lam)
    assert dropped == 0, ("this edge is supposed to be truncation-free", dropped)
    got = max(r.edf0 for r in dense if r.lambda0 == lam)
    assert _PS8_LO_TOL > 10 * _PS8_LO_FLOOR, "tolerance must clear the certified floor"
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=_PS8_LO_TOL)


def test_the_arrow_ladder_low_edge_error_is_half_a_degree_of_freedom():
    """Bracket the defect its strict xfail below cannot describe.

    The arrow ladder reads 172.37960656016872 at this pair's low edge against a
    certified 172.90456267038329, so it is 0.52496 df LOW.  This asserts
    ``0.3 < oracle - arrow < 0.8``, which goes red if the path is repaired and
    red if it worsens -- neither of which the xfail below can see, since a
    strict xfail is satisfied by any failure at all.

    A bracket, not a tolerance: the arrow reading has no derived accuracy bound
    to be pinned against, and inventing one would be the defect this file
    documents.  The width is roughly half the error on each side, and it is 53x
    the point's certified 5.6825e-03 df floor, so reassembling the moments
    elsewhere cannot move it out.
    """
    _, arrow = _both(_band_pair(**_PS8))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert 0.3 < _PS8_LO_ORACLE - got < 0.8, ("the arrow low-edge error has moved", got)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: the arrow ladder's rank-minus-trace difference is 0.525 df "
        "low at the ladder's low edge, where the point's certified floor is "
        "5.6825e-03 df"
    ),
)
def test_the_arrow_ladder_matches_the_certified_low_edge_oracle():
    """The arrow ladder's LOW edge is wrong, and by how much is bracketed above.

    172.37960656016872 against a certified 172.90456267038329 -- 0.525 df low,
    92x this point's certified 5.6825e-03 df floor.  Nothing about the moments
    excuses it: both paths receive the same ones and the dense path reads
    172.90468989250675 from them.

    The tolerance is the same certified ``_PS8_LO_TOL`` the dense pin above
    uses, so that when the arrow path is fixed this xfail flips against a bound
    that is 10.6x the floor rather than 1.8x it.

    Over 20 sweep points where both paths clamp to one lambda and the point is
    attainable to 1e-2 df, the arrow error has median 1.17 df and maximum
    30.74 df, and it lands inside the attainable band on 3 of the 20 against the
    dense path's 19.
    """
    _, arrow = _both(_band_pair(**_PS8))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=_PS8_LO_TOL)


def test_the_high_edge_is_not_determined_by_the_assembled_moments():
    """#257's headline number is BELOW this point's own noise floor.

    The issue reports 11.57 df of arrow-vs-dense divergence at the high edge and
    treats it as a defect.  On this geometry the two paths differ by 3.71 df --
    15.000025261210718 against 18.706953863891897.

    WHAT THIS TEST NO LONGER DOES.  It used to assert
    ``|dense - oracle| < 5.06`` and ``|arrow - oracle| < 5.06``.  5.06 df was a
    10-draw lower bound on a random spread; the derivative bound at this point
    is 70.7 df and even THAT is not a bound, because ``r = 6.119e+01`` -- one
    ulp of the stored moments is 61x the smallest eigenvalue of ``A``, so a
    one-ulp step can cross a singularity and the exact edf is not bounded by
    any multiple of a derivative taken at the unperturbed point.  Those two
    assertions pinned a quantity nothing certifies.  They are gone.

    WHAT IT ASSERTS INSTEAD IS AN OBSERVABLE, not a scale comparison.  Norms
    alone do not establish ill-posedness -- an assembly could preserve
    ``||V_eff||``, ``||S||`` and both lambdas while rotating the small
    eigenspaces until the edf is stable.  What cannot survive that is the V-MASS
    the round-off subspace carries.  Take the directions of ``A`` below
    ``lam eps ||S||_2``, the round-off the penalty term alone injects; the edf
    they can contribute is ``||Q_d' V_eff Q_d||_F / min|w_d|``, which is
    invariant to rotation WITHIN that subspace and so does not depend on which
    basis the eigensolver happened to return.  Measured on this pair:

        high edge   1 direction below 4.72e-06,  carrying 3.1083 df
        low  edge   0 directions below 4.72e-26, carrying 0 df

    Three whole degrees of freedom of the answer rest on a direction whose own
    eigenvalue is round-off, at the high edge, and none at the low edge of the
    same pair.  That is why the low edge grades both paths and the high edge
    grades neither, and it is measured rather than argued.

    The bs(8) high edge is the informative contrast and is NOT ill-posed in this
    sense: 4 directions below 6.82e-06 carrying 9.48e-04 df between them, which
    is 3300x less, on a pair whose lambda is just as extreme.  Extreme lambda is
    not sufficient; V-mass on the round-off subspace is what does it.
    """
    grab = _band_pair(**_PS8)
    dense, arrow = _both(grab)
    assert arrow is not None
    V_eff, S = _dense_matrices(grab)
    eps = np.finfo(float).eps
    lam_hi = max(r.lambda0 for r in dense)
    lam_lo = min(r.lambda0 for r in dense)

    def dust_mass(lam):
        w, Q = np.linalg.eigh(V_eff + lam * S)
        dust = np.abs(w) < lam * eps * float(np.linalg.norm(S, 2))
        if not dust.any():
            return 0, 0.0
        Qd = Q[:, dust]
        return int(dust.sum()), float(
            np.linalg.norm(Qd.T @ V_eff @ Qd, "fro") / np.abs(w[dust]).min()
        )

    n_hi, mass_hi = dust_mass(lam_hi)
    n_lo, mass_lo = dust_mass(lam_lo)
    assert n_hi >= 1 and mass_hi > 1.0, ("the high edge is determined after all", n_hi, mass_hi)
    assert n_lo == 0 and mass_lo == 0.0, ("the low edge is not determined either", n_lo, mass_lo)

    d = min(r.edf0 for r in dense if r.lambda0 == lam_hi)
    a = max(r.edf0 for r in arrow if r.lambda0 == max(x.lambda0 for x in arrow))
    assert abs(d - a) < mass_hi + abs(d - _PS8_HI_ORACLE), (
        "the divergence now exceeds what the round-off subspace can explain",
        d,
        a,
    )


# --------------------------------------------------------------------------
# ns(6), 6 levels, 12 rows in each, two inside a 1e-1 band of x.  k = 35.
# ``ns``'s penalty is full rank, so the arrow ladder's ``free`` set is empty
# and ``block_ranks`` returns the unpenalized count for every level.
#
# LOW edge, lambda = 5.894735595624617e-12:
#   exact edf             30.988667391753435 (arb, 800 bits, radius 2.8e-180)
#   CERTIFIED floor       8.2863e-05 df  (D = 8.2837e-05, r = 2.819e-04)
#   cho_factor succeeds, pinv would drop 0 of 35 -- no truncation here
_NS6 = dict(L=6, reps=12, kind="ns", n_knots=6, n_band=2, width=1e-1, seed=3)
_NS6_LO_ORACLE = 30.988667391753435
_NS6_LO_FLOOR = 8.2863e-05
_NS6_LO_TOL = 1e-3  # 12.1x the certified floor; the arrow rung misses by 30.74 df


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
    for ``?sygv`` predicts, since ``ns``'s penalty is full rank so ``G`` stays
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

    The 1e-2 here is 121x this pair's certified floor, and deliberately loose
    rather than derived-tight.  The assertion ranges over all five rungs, whose
    floors span nine orders -- 9.16e-14 df at ``lambda = 74.3`` up to 8.29e-05
    at ``lambda = 5.9e-12`` -- and it exists to catch 6.361 and 30.033 df
    mutations, so a bound anywhere between 1e-3 and 1 does the same work.  The
    observed worst over the ladder is 1.65e-13 df.  What matters is that 1e-2
    is ABOVE the largest floor it ranges over, which is the property the
    high-edge tolerances lacked.
    """
    from superglm.screening._score_stat import _edge

    grab = _band_pair(**_NS6)
    dense, _ = _both(grab)
    V_eff, S = _dense_matrices(grab)
    worst = 0.0
    for r in dense:
        other, _ = _edge(V_eff, S, r.lambda0)
        worst = max(worst, abs(float(other) - r.edf0))
    assert 1e-2 > 100 * _NS6_LO_FLOOR, "tolerance must clear the certified floor"
    assert worst < 1e-2, ("dense estimators disagree by", worst)


def test_the_arrow_ladder_low_edge_collapses_on_a_full_rank_penalty():
    """Bracket the worst arrow reading measured anywhere.

    The arrow ladder reads 0.24516151291386734 at this pair's low edge against
    a certified 30.988667391753435 -- 30.74 df low, on a point whose certified
    floor is 8.2863e-05 df, where the dense path reads 30.988679169193755
    (error 1.18e-05).  This asserts ``25 < oracle - arrow < 35``, which the
    strict xfail below cannot: that one is satisfied by any failure, so a
    reading of -400 would leave it green.

    The mechanism is visible in the two halves of the difference:
    ``rank - lambda tr(A^-1 S)`` is 35.00 - 34.75 where the exact split is
    35.00 - 4.01, so the arrow inverse's penalty trace is 8.7x too large at a
    lambda of 5.9e-12.  The certified exact ``lam tr(A^-1 S)`` at this point is
    4.011319142520 (arb, radius 2.0e-183).

    It is not an edf-only defect.  ``z = (T - edf0) / sqrt(2 edf0)`` divides by
    the edf, so the same pair is ranked at z = 0.13 by the dense path and
    z = 35.42 by the arrow path at a budget of 2, and at 35.4 against -1.65 at
    a budget of 16 -- the arrow reading puts a pair that ranks at the bottom of
    the table at the very top of it.
    """
    _, arrow = _both(_band_pair(**_NS6))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert 25.0 < _NS6_LO_ORACLE - got < 35.0, ("the arrow collapse has moved", got)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: on a full-rank penalty the arrow ladder's low edge collapses "
        "to 0.245 df against a certified 30.989, moving z by 35.3"
    ),
)
def test_the_arrow_ladder_low_edge_survives_a_full_rank_penalty():
    """The arrow path should read this well-posed point, and does not.

    Pinned at ``_NS6_LO_TOL`` = 1e-3, 12.1x the certified 8.2863e-05 df floor,
    which TIGHTENS the 1e-2 this landed with: the point is determined 121x
    better than that bound admitted, and when the arrow path is fixed this
    xfail should flip against a bound the point can actually support.  The
    dense path already reads it to 1.18e-05 df, inside the new bound by 85x.

    The magnitude of today's failure is bracketed by
    ``test_the_arrow_ladder_low_edge_collapses_on_a_full_rank_penalty``.
    """
    _, arrow = _both(_band_pair(**_NS6))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert _NS6_LO_TOL > 10 * _NS6_LO_FLOOR, "tolerance must clear the certified floor"
    assert got == pytest.approx(_NS6_LO_ORACLE, abs=_NS6_LO_TOL)
