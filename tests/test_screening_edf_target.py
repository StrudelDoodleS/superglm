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
constants are pinned here and the derivation lives in the commit that added
them.

**An oracle is only a target where the INPUTS determine it.**  Both paths
receive bit-identical moments, so everything downstream is algorithm -- but the
moments themselves are float64, and at the ladder's high edge ``lambda`` is
1e10 times the pair's scale, so one ulp of the assembled ``S_a`` is amplified
into whole degrees of freedom.  Each test below therefore carries the
ATTAINABLE accuracy of its own point, and no tolerance here is below it.

**THOSE FLOORS ARE DERIVED, NOT OBSERVED.**  ``edf(lam) = tr(A^-1 V)`` with
``A = V + lam S``, so differentiating,

    d edf = lam tr(A^-1 S A^-1 dV) - lam tr(A^-1 V A^-1 dS)

and one ulp of every stored entry is ``|dV_ij| <= e |V_ij|``,
``|dS_ij| <= e |S_ij|`` at ``e = 2^-52`` -- a full ulp, twice the unit round-off
``2^-53``, so that the bound covers a perturbation that STEPS to the neighbouring
float and not only one that rounds to it.  The worst case over all such
perturbations is then the elementwise contraction

    floor = lam e ( sum_ij |(A^-1 S A^-1)_ij| |V_ij|
                  + sum_ij |(A^-1 V A^-1)_ij| |S_ij| )

evaluated in arb at 800 bits, since ``A^-1`` in float64 is meaningless at these
condition numbers -- ``cond(A)`` is 4.4e+16 at the ``bs(8)`` high edge.  It is a
ONE-SIDED bound: a max-minus-min SPREAD over draws is bounded by twice it.

That bound is what the tolerances below are set from, and it is why two of them
moved.  The figures the commit that added these tests called "attainable" were
maxima over 40 random 1-ulp draws at each of three seeds, and a random draw
neither bounds the worst case nor reliably approaches it:

    point       k    lambda       derived floor   40-draw figure     old tol
    bs(8) hi   121   1.065e+05     1.620e-02 df   9.4e-04  (0.06x)   1e-2  0.62x
    ps(8) lo   209   1.381e-11     5.554e-03 df   5.61e-03 (1.01x)   1e-2  1.8x
    ps(8) hi   209   1.381e+09     7.073e+01 df   77.54    (1.10x)   5.06  0.07x
    ns(6) lo    35   5.895e-12     8.284e-05 df   1.82e-04 (2.2x)    1e-2  121x

Three of the four random figures land between 1.0x and 2.2x the one-sided bound,
which is the agreement a spread should show -- but at the ``bs(8)`` high edge
the draws reached 6% of it, and that is the point where the old tolerance sat
BELOW the noise floor at 0.62x.  The ``ps(8)`` high edge sat at 0.07x.  Neither
was a tolerance; both were coincidences of the draw.

    point      floor        tolerance      mutation it must still catch
    bs(8) hi   1.620e-02    2e-1  12.3x    2.000 df  (pencil rung)
    ps(8) lo   5.554e-03    6e-2  10.8x    0.525 df  (arrow rung)
    ps(8) hi   7.073e+01    -- not pinned, no tolerance can be both above the
                              floor and below the 3.71 df it would grade --
    ns(6) lo   8.284e-05    1e-3  12.1x    30.74 df  (arrow rung)

Every tolerance is 10x to 12.3x the one-sided floor, which is 5x to 6.2x the
spread bound, and the margin is there for the second-order terms the
linearization drops rather than for observed comfort.

A second, independent check that the floors are real: re-deriving the four
oracle constants below on another machine, with all six thread pools pinned,
reproduced them to -4.4e-10, +1.09e-04, -4.0e-10 and +1.3e-05 df.  Each of
those is inside its own floor -- but the ``ps(8)`` low-edge one is the SAME
ORDER as the error being measured there.  Graded against the constant below,
the dense clamped rung is 1.27e-04 df out; graded against the re-derived one,
1.78e-05 df.  A factor of 7.1, from nothing but reassembling the moments
somewhere else.  The POINT moves between environments; the pinned constant is
only as good as the floor beside it, which is why the tolerance is set from
the floor and not from either error.

**WHAT ``_edge`` ACTUALLY COMPUTES, WHICH IS NOT ``tr(A^-1 V_eff)``.**  The
recommendation this work establishes -- target ``tr(A^-1 V_eff)``, which is
what :func:`superglm.screening._score_stat._edge` already computes -- is true
only on the branch where ``cho_factor`` succeeds.  ``_edge`` factors ``A``
with ``cho_factor`` and FALLS BACK to ``numpy.linalg.pinv(A, hermitian=True)``,
whose default ``rcond`` is ``max(M, N) * eps`` relative to the largest singular
value, so the fallback answers ``tr(P V_eff)`` for a pseudo-inverse that ZEROES
every direction under that cut instead of inverting it.  Measured at the two
high edges pinned here, ``cho_factor`` FAILS on both and the fallback is what
runs:

    point      cut         dropped   tr(P V) - tr(A^-1 V)
    bs(8) hi   8.25e-04     4 / 121   +2.4974e-04 df
    ps(8) hi   9.87e-04     4 / 209   -8.8838e-01 df

so at the ``ps(8)`` high edge nine tenths of a degree of freedom of the gap to
the certified oracle is the truncation and not the arithmetic.  The clamped
rung is EXACT for the functional it evaluates: reconstructing ``tr(P V)`` from
``eigh`` at ``pinv``'s documented cut reproduces the reported rung to 3.6e-15
and 7.1e-15 df, and evaluating it in arb on the same basis reproduces all 16
digits.  ``|rung - certified oracle|`` is 2.50e-04 and 8.88e-01 df.  The
truncation is not marginal at either point -- the retained spectrum starts at
1.16 and 0.87 against dropped directions at 3.98e-06 and 1.01e-05, a gap of
five orders -- so it is stable, but it is a term in the answer and the
recommendation has to name it.  ``pinv`` also scores directions by ``|lambda|``
where :func:`_psd_rank` reads the sign, so a NEGATIVE curvature direction is
inverted rather than dropped; at both points that rule is inert, since each
carries exactly one negative eigenvalue (-1.59e-08 and -5.62e-07) and both sit
under the cut.  Inert here is not inert everywhere, and nothing counts them.
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

BUDGETS = (2.0, 4.0, 8.0, 16.0, 400.0)


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


# --------------------------------------------------------------------------
# bs(8), 12 levels, 30 rows in each, four levels inside a 1e-6 band of x.
# k = 121.  At the ladder's HIGH edge, lambda = 1.06521065611e+05.
#
#   exact edf              6.999754628429722  (arb, 800 bits, radius 1.3e-221)
#   DERIVED floor, 1 ulp   1.6201e-02 df  (V term 9.75e-09, S term 1.6201e-02)
#   random 40-draw max     9.4e-04 df     <-- 6% of it.  Do NOT pin against.
#
# The derived floor is the worst case over one ulp on every stored entry, by
# the contraction in the module docstring, evaluated in arb; the random figure
# is what 120 draws of that same perturbation happened to reach.  This point IS
# determined -- to sixteen thousandths of a degree of freedom, not to the
# thousandth the draws suggested -- so both paths can be graded on it at a
# tolerance a decade above the floor.
_BS8_HI_ORACLE = 6.999754628429722
_BS8_HI_FLOOR = 1.6201e-02
_BS8_HI_TOL = 2e-1  # 12.3x the floor; the pencil mutation misses by 2.000 df
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

    The certified oracle backs the first: 6.999754628429722, so the clamped
    rung is 2.50e-04 df out and the searching rung is 2.000265 df out, against
    an attainable accuracy of 9.4e-04 df at this point.

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
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    at_lam = {round(r.edf0, 9) for r in dense if r.lambda0 == lam}
    assert len(at_lam) == 1, (
        "one lambda, two edf: the dense ladder's clamped and searching rungs "
        f"disagree at lambda={lam!r}: {sorted(at_lam)}"
    )


def test_the_dense_clamped_rung_matches_the_certified_high_edge_oracle():
    """The clamped rung's trace is the estimator that is right, to a derived bound.

    Pinned at ``_BS8_HI_TOL`` = 2e-1 df, which is 12.3x the derived 1.620e-02 df
    floor of this point and 10x below the 2.000 df mutation it has to catch.
    The tolerance was 1e-2 when these tests landed, set from a 40-draw random
    maximum of 9.4e-04 df; the worst case over the same perturbation is
    1.620e-02 df, so 1e-2 was 0.62x the noise floor -- BELOW it, on a quantity
    the moments' own last bit can move by more than the tolerance allowed.
    Widening it is not loosening a bound to pass: the assertion passes at
    either value, with an observed error of 2.50e-04 df.  It is lifting a bound
    that sat under its floor back above it.

    What the rung computes is ``tr(P V_eff)`` for ``P = pinv(A)``, not
    ``tr(A^-1 V_eff)``: ``cho_factor`` fails on this ``A`` and the fallback
    zeroes 4 of 121 directions at a cut of 8.25e-04.  Of the 2.50e-04 df gap to
    the certified oracle, ALL of it is that truncation -- the rung matches an
    independent reconstruction of ``tr(P V_eff)`` to 3.6e-15 df.  See the module
    docstring.

    It bites: replacing the clamped rung's ``_edge`` trace with the searching
    rung's ``_pencil_edf`` at the same lambda -- the obvious "share one
    estimator" refactor -- reports 9.000019310394286 and fails this by
    2.000 df, which is 20x the tolerance and 247x the floor.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    clamped = min(r.edf0 for r in dense if r.lambda0 == lam)
    assert _BS8_HI_TOL > 10 * _BS8_HI_FLOOR, "tolerance must clear the derived floor"
    assert clamped == pytest.approx(_BS8_HI_ORACLE, abs=_BS8_HI_TOL)


def test_the_arrow_ladder_matches_the_certified_high_edge_oracle():
    """The arrow ladder is accurate at the HIGH edge where the point is posed.

    6.999831294695255 against 6.999754628429722, an error of 7.67e-05 df --
    3x nearer than the dense clamped rung on this point, and it does it
    without a truncation, since the arrow path never forms ``A^-1``.  The
    arrow path's ``rank - lambda tr(A^-1 S)`` difference is not the weak one
    here; the low edge is where it fails, and the two tests below pin that.

    Same derived tolerance as the dense pin above and for the same reason: the
    floor is a property of the POINT, not of the path being graded.  The arrow
    ladder brackets to its own lambda -- 106521.06561131444 against the dense
    path's ...448, one ulp apart -- and the certified oracle moves by less than
    1e-15 df between the two, so the constant serves both.

    It bites, in two different ways.  Adding one to the Guttman accounting --
    ``rank_term = f.rank - rank_m`` becoming ``+ 1.0``, the whole-degree
    miscount this pair of issues is about -- reads 7.999831294695255 and fails
    by 1.000 df, 5x the tolerance.  Reverting :func:`block_ranks` to the
    lambda-INDEPENDENT count that preceded it does not produce a wrong number
    at all: the ladder's own ``0 <= edf <= rank_term`` guard fires and
    ``structured_ladder`` returns ``None``, so the test fails on
    ``arrow is not None``.

    Three mutations do NOT move this pair, and are recorded so the test is not
    credited with catching them: ``_solve_floor`` reverted to a fixed 1e-12,
    the penalty-residue floor applied unconditionally instead of only on an
    exact zero, and the ``dust`` floor dropped from ``block_ranks``.  All three
    leave 6.999831294695255 unchanged to every digit.
    """
    _, arrow = _both(_band_pair(**_BS8))
    assert arrow is not None
    lam = max(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert _BS8_HI_TOL > 10 * _BS8_HI_FLOOR, "tolerance must clear the derived floor"
    assert got == pytest.approx(_BS8_HI_ORACLE, abs=_BS8_HI_TOL)


# --------------------------------------------------------------------------
# ps(8), 20 levels, 30 rows in each, four levels inside a 1e-3 band of x.
# k = 209.  This is issue #257's own geometry.
#
# LOW edge, lambda = 1.3814328670859514e-11:
#   exact edf             172.90456267038329  (arb, 800 bits, radius 5.9e-214)
#   DERIVED floor, 1 ulp  5.5544e-03 df (V term 5.5544e-03, S term 6.32e-15)
#   random 40-draw max    5.61e-03 to 6.78e-03 df   <-- 1.01x to 1.22x the bound,
#                         which is what a max-minus-min spread should show
#                         against a one-sided bound.  The bound is used.
#
# HIGH edge, lambda = 1381432867.0859513:
#   exact edf              15.888406216250933 (arb, 800 bits, radius 2.4e-220)
#   DERIVED floor, 1 ulp   7.0734e+01 df (V term 1.87e-08, S term 7.0734e+01)
#   random 40-draw max     11.11 / 31.83 / 77.54 df at three seeds  (1.10x)
#
# THE HIGH EDGE IS NOT PINNED.  Its floor is 70.7 degrees of freedom on a
# k = 209 pair whose edf is 15.9, so there is no tolerance that is both above
# the floor and small enough to grade the 3.71 df the two paths differ by.  The
# test that used to pin it at 5.06 df -- a 10-draw lower bound, 14x BELOW the
# derived floor -- now asserts the ill-posedness itself, which is robust and is
# the actual finding.  That the linearization is at its own limit here is part
# of it: one ulp of the stored penalty, lam * e * |S|_2 = 4.72e-06, is 8.4x the
# smallest-magnitude eigenvalue of the assembled A, so the derived 70.7 df is
# itself an underestimate.
_PS8 = dict(L=20, reps=30, kind="ps", n_knots=8, n_band=4, width=1e-3, seed=3)
_PS8_LO_ORACLE = 172.90456267038329
_PS8_LO_FLOOR = 5.5544e-03
_PS8_LO_TOL = 6e-2  # 10.8x the floor; the arrow rung misses by 0.525 df
_PS8_HI_ORACLE = 15.888406216250933
_PS8_HI_FLOOR = 7.0734e01


def test_the_dense_clamped_rung_matches_the_certified_low_edge_oracle():
    """At the LOW edge the dense clamped rung is right to 1.27e-04 df.

    Pinned at ``_PS8_LO_TOL`` = 6e-2, which is 10.8x this point's derived
    5.554e-03 df floor and 8.75x below the 0.525 df the arrow path misses it
    by.  The tolerance was 1e-2, which is 1.8x the floor -- not inverted like
    the two high-edge pins, but not a margin either, and an independent
    re-derivation of ``_PS8_LO_ORACLE`` on another machine moved the point by
    1.09e-04 df, so the constant itself is only good to a few floor-widths.

    ``_edge`` takes the CHOLESKY branch here -- ``A`` is positive definite at
    ``lambda = 1.38e-11`` and ``pinv`` would discard 0 of 209 directions -- so
    unlike the high edge this rung really does evaluate ``tr(A^-1 V_eff)``
    with no truncation in it.

    Measured over 20 sweep points where both paths clamp to one lambda AND the
    exact answer is attainable to 1e-2 df, the clamped rung's error has median
    6.9e-05 df and maximum 2.7e-02 df.
    """
    dense, _ = _both(_band_pair(**_PS8))
    lam = min(r.lambda0 for r in dense)
    got = max(r.edf0 for r in dense if r.lambda0 == lam)
    assert _PS8_LO_TOL > 10 * _PS8_LO_FLOOR, "tolerance must clear the derived floor"
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=_PS8_LO_TOL)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: the arrow ladder's rank-minus-trace difference is 0.525 df "
        "low at the ladder's low edge, where the point's derived floor is "
        "5.554e-03 df"
    ),
)
def test_the_arrow_ladder_matches_the_certified_low_edge_oracle():
    """The arrow ladder's LOW edge is wrong, and by how much is now pinned.

    172.37960656016872 against a certified 172.90456267038329 -- 0.525 df low,
    94x this point's derived 5.554e-03 df floor.  Nothing about the moments
    excuses it: both paths receive the same ones and the dense path reads
    172.90468989250675 from them.

    The tolerance is the same derived ``_PS8_LO_TOL`` the dense pin above uses,
    so that when the arrow path is fixed this xfail flips against a bound that
    is 10.8x the floor rather than 1.8x it.

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
    15.000025261210718 against 18.706953863891897 -- and one ulp of the stored
    moments moves the EXACT edf by up to 7.073e+01 df, derived rather than
    sampled.  At the high edge ``lambda`` is 1.38e+09 times the pair's scale,
    and a spline penalty assembled in float64 carries a smallest eigenvalue that
    is round-off of either sign (measured 0.01 to 0.62 eps of ``sigma_max`` on
    the twelve margins this library assembles), so ``lambda S_a`` inherits an
    indefiniteness of order ``lambda eps sigma_max``.  The exact
    ``A = V_eff + lambda S`` is then INDEFINITE, its filter factors leave [0, 1]
    and edf stops being a degrees-of-freedom quantity at all: over an 882-point
    sweep the exact high-edge edf falls outside ``[0, k]`` on 12 points, as far
    out as 524.70 on a k = 209 pair and -69.49 on a k = 35 one.

    WHAT THIS TEST NO LONGER DOES.  It used to assert
    ``|dense - oracle| < 5.06`` and ``|arrow - oracle| < 5.06`` as well.  5.06 df
    was a 10-draw lower bound on the spread, and it is 14x BELOW the derived
    70.7 df floor -- so those two assertions pinned a high-edge quantity to a
    band narrower than the last bit of the moments can move it, which is the
    exact defect this file exists to document.  There is no repair: any bound
    above the floor is 4x wider than the largest value in play.  They are gone,
    and what is left is the ill-posedness, which is robust.

    The two robust facts asserted here are norm comparisons, so they cost no
    accuracy to evaluate and cannot be undone by a last-bit change:

    * ``lam eps |S|_2`` is the size of the round-off the penalty term alone
      injects into ``A``.  At the HIGH edge that is 4.72e-06 against
      ``k eps |V_eff|_2 = 1.70e-13``, the resolution at which ``V_eff``'s own
      near-null spectrum is knowable -- a factor of 2.8e+07, so ``A``'s small
      directions are decided by round-off in ``lam S`` and by nothing else.
    * At the LOW edge the same ratio is 2.8e-13, twenty orders the other way.
      That is why the low edge grades both paths and the high edge grades
      neither, and it is measured on the SAME pair rather than argued.

    The divergence itself is then checked against the derived floor, which is
    #257's correction: 3.71 df of disagreement, 70.7 df of floor.
    """
    grab = _band_pair(**_PS8)
    dense, arrow = _both(grab)
    assert arrow is not None
    V_eff, S = _dense_matrices(grab)
    k = V_eff.shape[0]
    eps = np.finfo(float).eps
    resolution = k * eps * float(np.linalg.norm(V_eff, 2))
    inject = eps * float(np.linalg.norm(S, 2))
    lam_hi = max(r.lambda0 for r in dense)
    lam_lo = min(r.lambda0 for r in dense)
    assert lam_hi * inject > 1e6 * resolution, ("high edge is determined", lam_hi * inject)
    assert lam_lo * inject < 1e-6 * resolution, ("low edge is not", lam_lo * inject)

    d = min(r.edf0 for r in dense if r.lambda0 == lam_hi)
    a = max(r.edf0 for r in arrow if r.lambda0 == max(x.lambda0 for x in arrow))
    assert abs(d - a) < _PS8_HI_FLOOR, ("divergence exceeds the floor", d, a)


# --------------------------------------------------------------------------
# ns(6), 6 levels, 12 rows in each, two inside a 1e-1 band of x.  k = 35.
# ``ns``'s penalty is full rank, so the arrow ladder's ``free`` set is empty
# and ``block_ranks`` returns the unpenalized count for every level.
#
# LOW edge, lambda = 5.894735595624617e-12:
#   exact edf             30.988667391753435 (arb, 800 bits, radius 2.8e-180)
#   DERIVED floor, 1 ulp  8.2837e-05 df (V term 8.2837e-05, S term 4.13e-17)
#   random 40-draw max    1.25e-04 to 1.82e-04 df   (1.5x to 2.2x the one-sided
#                         bound, i.e. a spread, as at the ps(8) low edge)
#
# ``_edge`` takes the CHOLESKY branch here too: A is positive definite and pinv
# would discard 0 of 35 directions.
_NS6 = dict(L=6, reps=12, kind="ns", n_knots=6, n_band=2, width=1e-1, seed=3)
_NS6_LO_ORACLE = 30.988667391753435
_NS6_LO_FLOOR = 8.2837e-05
_NS6_LO_TOL = 1e-3  # 12.1x the floor; the arrow rung misses by 30.74 df


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
    instead of carrying both transformed terms fails by 6.361 df, and whitening
    by ``V_eff`` rather than by ``G`` fails by 30.033 df.  Three others do NOT
    move this pair and are recorded so the test is not credited with catching
    them: dropping the balance before summing (``balance = 1.0``), which
    ``test_a_curvature_that_dwarfs_its_penalty_keeps_the_penalty`` does catch;
    removing the clip of the share to [0, 1]; and reverting ``_rank_floor`` to
    the fixed 1e-12 it started with.  The last two are caught by neither that
    test nor ``test_screening_is_invariant_to_the_units_of_a_numeric_margin``,
    which is a gap this test does not close either.

    The 1e-2 here is 121x this pair's derived floor, and deliberately loose
    rather than derived-tight.  The assertion ranges over all five rungs, whose
    floors span nine orders -- 9.16e-14 df at ``lambda = 74.3`` up to 8.28e-05
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
    assert 1e-2 > 100 * _NS6_LO_FLOOR, "tolerance must clear the derived floor"
    assert worst < 1e-2, ("dense estimators disagree by", worst)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: on a full-rank penalty the arrow ladder's low edge collapses "
        "to 0.245 df against a certified 30.989, moving z by 35.3"
    ),
)
def test_the_arrow_ladder_low_edge_survives_a_full_rank_penalty():
    """The worst arrow reading measured anywhere, on a well-posed point.

    0.24516151291386734 against a certified 30.988667391753435 -- 30.74 df low,
    on a point whose derived floor is 8.284e-05 df, where the dense path reads
    30.988679169193755 (error 1.18e-05).  The mechanism is visible in the two
    halves of the difference: ``rank - lambda tr(A^-1 S)`` is 35.00 - 34.75
    where the exact split is 35.00 - 4.01, so the arrow inverse's penalty trace
    is 8.7x too large at a lambda of 5.9e-12.  The certified exact
    ``lam tr(A^-1 S)`` at this point is 4.011319142520 (arb, radius 2.0e-183).

    Pinned at ``_NS6_LO_TOL`` = 1e-3, 12.1x the derived floor, which TIGHTENS
    the 1e-2 this landed with: the point is determined 121x better than that
    bound admits, and when the arrow path is fixed this xfail should flip
    against a bound the point can actually support.  The dense path already
    reads it to 1.18e-05 df, inside the new bound by 85x.

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
    assert _NS6_LO_TOL > 10 * _NS6_LO_FLOOR, "tolerance must clear the derived floor"
    assert got == pytest.approx(_NS6_LO_ORACLE, abs=_NS6_LO_TOL)
