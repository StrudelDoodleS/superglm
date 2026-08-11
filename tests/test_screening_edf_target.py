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
into whole degrees of freedom.  Each test below therefore carries the measured
ATTAINABLE accuracy of its own point: the spread of the EXACT edf under 1-ulp
symmetric perturbations of the input that edge is sensitive to.  Where that
spread is larger than the disagreement, the disagreement is not a defect of
either path and the tests say so rather than picking a winner.
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


# --------------------------------------------------------------------------
# bs(8), 12 levels, 30 rows in each, four levels inside a 1e-6 band of x.
# k = 121.  At the ladder's HIGH edge, lambda = 1.06521065611e+05.
#
#   exact edf             6.999754628429722  (arb, 800 bits, radius 1.3e-221)
#   attainable, V 1 ulp   3.39e-09 df
#   attainable, S_a 1 ulp 6.6e-04 to 9.4e-04 df
#
# Every attainable figure here is a MAX over 40 symmetric 1-ulp draws at each of
# three generator seeds, so it is a lower bound on the true spread; all three
# seeds are quoted where they differ.  This point IS determined by its inputs,
# to under a thousandth of a degree of freedom, so both paths can be graded on
# it.
_BS8_HI_ORACLE = 6.999754628429722
_BS8_HI_ATTAINABLE = 9.4e-04
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
    """The clamped rung's ``tr(A^-1 V_eff)`` is the estimator that is right.

    Pinned at 1e-2 df against a measured error of 2.50e-04 and an attainable
    accuracy of 9.4e-04 df -- 40x the observed error and 11x the noise floor,
    so this is not a coincidence tolerance.

    It bites: replacing the clamped rung's ``_edge`` trace with the searching
    rung's ``_pencil_edf`` at the same lambda -- the obvious "share one
    estimator" refactor -- reports 9.000019310394286 and fails this by
    2.000 df.
    """
    dense, _ = _both(_band_pair(**_BS8))
    lam = max(r.lambda0 for r in dense)
    clamped = min(r.edf0 for r in dense if r.lambda0 == lam)
    assert clamped == pytest.approx(_BS8_HI_ORACLE, abs=1e-2)


def test_the_arrow_ladder_matches_the_certified_high_edge_oracle():
    """The arrow ladder is accurate at the HIGH edge where the point is posed.

    6.999831294695255 against 6.999754628429722, an error of 7.67e-05 df --
    3x nearer than the dense clamped rung on this point.  The arrow path's
    ``rank - lambda tr(A^-1 S)`` difference is not the weak one here; the low
    edge is where it fails, and the two tests below pin that.
    """
    _, arrow = _both(_band_pair(**_BS8))
    assert arrow is not None
    lam = max(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert got == pytest.approx(_BS8_HI_ORACLE, abs=1e-2)


# --------------------------------------------------------------------------
# ps(8), 20 levels, 30 rows in each, four levels inside a 1e-3 band of x.
# k = 209.  This is issue #257's own geometry.
#
# LOW edge, lambda = 1.3814328670859514e-11:
#   exact edf             172.90456267038329  (arb, 800 bits, radius 5.9e-214)
#   attainable, V 1 ulp   5.61e-03 to 6.78e-03 df
#   attainable, S_a 1 ulp 0.00e+00 df
#
# HIGH edge, lambda = 1381432867.0859513:
#   exact edf              15.888406216250933 (arb, 800 bits, radius 2.4e-220)
#   attainable, V 1 ulp    1.20e-09 df
#   attainable, S_a 1 ulp  11.11 / 31.83 / 77.54 df at the three seeds
#                          <-- the target is NOT determined here.  The assertion
#                          below keeps 5.06, a 10-draw lower bound on the same
#                          spread, because a TIGHTER band makes that test
#                          stronger rather than weaker.
_PS8 = dict(L=20, reps=30, kind="ps", n_knots=8, n_band=4, width=1e-3, seed=3)
_PS8_LO_ORACLE = 172.90456267038329
_PS8_LO_ATTAINABLE = 6.78e-03
_PS8_HI_ORACLE = 15.888406216250933
_PS8_HI_ATTAINABLE = 5.06


def test_the_dense_clamped_rung_matches_the_certified_low_edge_oracle():
    """At the LOW edge the dense clamped rung is right to 1.27e-04 df.

    Pinned at 1e-2, which is 1.5x the 6.78e-03 df this point is determined to
    and 79x the observed error.  Measured over 20 sweep points where both paths
    clamp to one lambda AND the exact answer is attainable to 1e-2 df, the
    clamped rung's error has median 6.9e-05 df and maximum 2.7e-02 df, and it
    lands inside the attainable band on 19 of the 20.
    """
    dense, _ = _both(_band_pair(**_PS8))
    lam = min(r.lambda0 for r in dense)
    got = max(r.edf0 for r in dense if r.lambda0 == lam)
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=1e-2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#257/#249: the arrow ladder's rank-minus-trace difference is 0.525 df "
        "low at the ladder's low edge, where the point is determined to 3.4e-03 df"
    ),
)
def test_the_arrow_ladder_matches_the_certified_low_edge_oracle():
    """The arrow ladder's LOW edge is wrong, and by how much is now pinned.

    172.37960656016872 against a certified 172.90456267038329 -- 0.525 df low,
    77x the 6.78e-03 df this point is determined to.  Nothing about the moments
    excuses it: both paths receive the same ones and the dense path reads
    172.90468989250675 from them.

    Over 20 sweep points where both paths clamp to one lambda and the point is
    attainable to 1e-2 df, the arrow error has median 1.17 df and maximum
    30.74 df, and it lands inside the attainable band on 3 of the 20 against the
    dense path's 19.
    """
    _, arrow = _both(_band_pair(**_PS8))
    assert arrow is not None
    lam = min(r.lambda0 for r in arrow)
    got = max(r.edf0 for r in arrow if r.lambda0 == lam)
    assert got == pytest.approx(_PS8_LO_ORACLE, abs=1e-2)


def test_the_high_edge_divergence_is_inside_what_the_moments_determine():
    """#257's headline number is BELOW this point's own noise floor.

    The issue reports 11.57 df of arrow-vs-dense divergence at the high edge and
    treats it as a defect.  On this geometry the two paths differ by 3.71 df --
    15.000025261210718 against 18.706953863891897 -- and the EXACT edf on these
    moments moves by 11.11, 31.83 and 77.54 df under a 1-ulp symmetric
    perturbation of ``S_a`` alone, over 40 draws at each of three seeds.  At
    the high edge ``lambda`` is 1.38e+09 times the pair's scale, and
    a spline penalty assembled in float64 carries a smallest eigenvalue that is
    round-off of either sign (measured 0.01 to 0.62 eps of ``sigma_max`` on the
    twelve margins this library assembles), so ``lambda S_a`` inherits an
    indefiniteness of order ``lambda * eps * sigma_max`` -- which here is 4.7e-06
    against a ``V_eff`` whose own smallest eigenvalues are 1e-15.  The exact
    ``A = V_eff + lambda S`` is then INDEFINITE, its filter factors leave [0, 1]
    and edf stops being a degrees-of-freedom quantity at all: over an 882-point
    sweep the exact high-edge edf falls outside ``[0, k]`` on 12 points, as far
    out as 524.70 on a k = 209 pair and -69.49 on a k = 35 one.

    So the high-edge divergence is not evidence about either path, and this test
    asserts only what the measurement supports: both paths sit inside the band
    the inputs determine.  The LOW edge is the opposite case -- determined to
    6.8e-03 df there -- and it is where the paths are graded, above.
    """
    grab = _band_pair(**_PS8)
    dense, arrow = _both(grab)
    assert arrow is not None
    d = min(r.edf0 for r in dense if r.lambda0 == max(x.lambda0 for x in dense))
    a = max(r.edf0 for r in arrow if r.lambda0 == max(x.lambda0 for x in arrow))
    assert abs(d - a) < _PS8_HI_ATTAINABLE, ("divergence", d, a)
    assert abs(d - _PS8_HI_ORACLE) < _PS8_HI_ATTAINABLE, ("dense", d)
    assert abs(a - _PS8_HI_ORACLE) < _PS8_HI_ATTAINABLE, ("arrow", a)


# --------------------------------------------------------------------------
# ns(6), 6 levels, 12 rows in each, two inside a 1e-1 band of x.  k = 35.
# ``ns``'s penalty is full rank, so the arrow ladder's ``free`` set is empty
# and ``block_ranks`` returns the unpenalized count for every level.
#
# LOW edge, lambda = 5.894735595624617e-12:
#   exact edf             30.988667391753435 (arb, 800 bits, radius 2.8e-180)
#   attainable, V 1 ulp   1.25e-04 to 1.82e-04 df
#   attainable, S_a 1 ulp 0.00e+00 df
_NS6 = dict(L=6, reps=12, kind="ns", n_knots=6, n_band=2, width=1e-1, seed=3)
_NS6_LO_ORACLE = 30.988667391753435


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
    """
    from superglm.screening._score_stat import _edge, _solve_psd

    grab = _band_pair(**_NS6)
    dense, _ = _both(grab)
    U, V, C, M, S_ti, _ = grab["args"]
    V = 0.5 * (V + V.T)
    V_eff = V - C.T @ _solve_psd(M, C)
    V_eff = 0.5 * (V_eff + V_eff.T)
    S = 0.5 * (S_ti + S_ti.T)
    worst = 0.0
    for r in dense:
        other, _ = _edge(V_eff, S, r.lambda0)
        worst = max(worst, abs(float(other) - r.edf0))
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
    on a point determined to 1.82e-04 df, where the dense path reads
    30.988679169193755 (error 1.18e-05).  The mechanism is visible in the two
    halves of the difference: ``rank - lambda tr(A^-1 S)`` is 35.00 - 34.75
    where the exact split is 35.00 - 4.01, so the arrow inverse's penalty trace
    is 8.7x too large at a lambda of 5.9e-12.

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
    assert got == pytest.approx(_NS6_LO_ORACLE, abs=1e-2)
