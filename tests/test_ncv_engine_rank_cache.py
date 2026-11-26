"""The engine where the downdate alone does not answer: aliased designs, lost rank, cached Grams.

A fit whose own design is rank deficient against the same model without the alias; a leave-out
system whose factorisation margin is questionable against the fit of the rows it keeps; and the
Gaussian cross-product cache against the evaluation that reads rows.  References and tolerances
are those of ``tests/test_ncv_engine.py``.
"""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, RepeatedKFold, TimeSeriesSplit

from superglm import SuperGLM
from superglm._parallel import parallel_config
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.ncv import engine, from_groups, from_splitter, single_rows
from superglm.ncv import rows as ncv_rows
from superglm.ncv.engine import (
    LeaveOutRankError,
    build_problem,
    evaluate,
    evaluate_fit,
    solve_mode,
)
from superglm.reml.penalty_algebra import build_penalty_matrix
from tests.test_ncv_engine import (
    BAR,
    RHO,
    ROUTES,
    KeptRows,
    U,
    assert_pairs_equal,
    assert_same_criterion,
    case,
    pairs_of,
    predictor_slack,
    problem_of,
    refit_pairs,
    solver_ending,
)

ZERO = np.zeros(1)
ORDERS = {
    "as-built": lambda n: np.arange(n),
    "reversed": lambda n: np.arange(n)[::-1],
    "shuffled": lambda n: np.random.default_rng(7).permutation(n),
}


@dataclass(frozen=True)
class Design:
    problem: engine.NCVProblem
    frame: pd.DataFrame


def rebuilt(problem, **changes):
    """``problem`` built again, with ``changes``: a problem of its own, holding nothing."""
    settings = dict(
        dm=problem.dm,
        groups=problem.groups,
        penalties=problem.penalties,
        distribution=problem.distribution,
        link=problem.link,
        y=problem.y,
        weights=problem.weights,
        offset=problem.offset,
        weight_semantics=problem.weight_semantics,
        mode_bar=problem.mode_bar,
    )
    return build_problem(**{**settings, **changes})


def mixed_groups(n):
    """Seventeen groups of two to twenty-eight rows, split between the two solve routes."""
    edges = np.cumsum([2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    return np.searchsorted(edges, np.random.default_rng(11).permutation(n), side="right")


# ── an aliased full design ────────────────────────────────────────────────────


@cache
def twin(family, alias) -> Design:
    """A P-spline and a numeric column, with or without the same column a second time."""
    rng = np.random.default_rng(28)
    n = 240
    x = rng.normal(size=n)
    t = rng.uniform(size=n)
    signal = np.sin(6 * t) + 0.5 * x
    y = signal + rng.normal(0, 0.2, n)
    if family == "poisson":
        y = rng.poisson(np.exp(signal)).astype(float)
    frame = pd.DataFrame({"t": t, "x": x, "twin": x})
    features = {"t": Spline(kind="ps", n_knots=6), "x": Numeric()}
    if alias:
        features["twin"] = Numeric()
    model = SuperGLM(features=features, family=family, selection_penalty=0, spline_penalty=1.0)
    model.fit(frame[list(features)], y)
    return Design(problem_of(model, y, np.ones(n), None), frame)


def twin_neighbourhoods(form, design):
    n = design.problem.n
    if form == "rows":
        return single_rows(n)
    if form == "groups":
        return from_groups(mixed_groups(n), n)
    return from_splitter(TimeSeriesSplit(4), design.frame, design.problem.y)


@cache
def twin_reference(family, form):
    """The neighbourhoods, and the design without the duplicate's Newton steps from its kept rows."""
    clean = twin(family, False)
    nb = twin_neighbourhoods(form, clean)
    return nb, KeptRows(clean.problem, ZERO).one_step(nb)


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
@pytest.mark.parametrize("form", ["rows", "groups", "forward"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_aliased_design_is_the_model_without_the_alias(family, form, route):
    """A duplicated column changes no held-out prediction, criterion or slope.

    Each prediction against the Newton step assembled from the kept rows of the design
    without the duplicate, and the evaluation against that design's own.
    """
    aliased, clean = twin(family, True), twin(family, False)
    fit = solve_mode(aliased.problem, ZERO)
    assert fit.certified and fit.result.reml_hessian_rank == aliased.problem.width - 1
    nb, (want, residual, kappa) = twin_reference(family, form)
    got = evaluate(aliased.problem, nb, ZERO, route=route)
    assert got.feasible
    gap = np.abs(got.heldout.predictor - want)
    slack = residual + predictor_slack(kappa, want)
    assert np.all(gap <= slack), f"gap {gap.max():.3e} against {slack.min():.3e}"
    assert_same_criterion(clean.problem, got, evaluate(clean.problem, nb, ZERO, route=route), kappa)


def spectral_basis(fit):
    """``fit``, rank deficient by one, handing back a basis of its identified slopes.

    The basis is the spectral one of its slope Hessian, Jacobi scaled, and the fit's inverse
    the pseudo-inverse on it; the coefficients are left as the solver gave them, one zero on
    the alias, which differ from the basis's own by a null vector.
    """
    scale = np.sqrt(np.diag(fit.slope_hessian))
    values, vectors = np.linalg.eigh(fit.slope_hessian / np.outer(scale, scale))
    kept = values > np.sqrt(U) * values[-1]
    assert kept.sum() == len(values) - 1
    basis = vectors[:, kept] / scale[:, None]
    spectral = replace(
        fit.result.reml_slope_decomposition,
        solution_basis=basis,
        estimable_functional_basis=vectors[:, kept] * scale[:, None],
        retained_values=values[kept],
    )
    return replace(
        fit,
        result=replace(fit.result, reml_slope_decomposition=spectral),
        slope_inverse=(basis / values[kept]) @ basis.T,
    )


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
def test_retained_basis_is_evaluated_as_representative_columns_are(route):
    """A fit that hands back a basis of its identified slopes gives the same evaluation."""
    aliased, clean = twin("gaussian", True), twin("gaussian", False)
    fit = solve_mode(aliased.problem, ZERO)
    general = spectral_basis(fit)
    for form in ("rows", "groups", "forward"):
        nb, (want, residual, kappa) = twin_reference("gaussian", form)
        got = evaluate_fit(aliased.problem, nb, general, route=route)
        assert got.feasible
        assert np.all(
            np.abs(got.heldout.predictor - want) <= residual + predictor_slack(kappa, want)
        )
        assert_same_criterion(
            clean.problem, got, evaluate(clean.problem, nb, ZERO, route=route), kappa
        )
    bare = replace(general, result=replace(fit.result, reml_slope_decomposition=None))
    unread = evaluate_fit(aliased.problem, single_rows(aliased.problem.n), bare)
    assert not unread.feasible and unread.infeasible_reason == "rank_deficient_fit"


# ── the rank verdict on deletion ──────────────────────────────────────────────

LEVER_H, LEVER_PAIRS = 1e-9, 20


def lever(order):
    """Forty rows at ``x = +-h`` and one at ``x = 1``; returns the problem and that row's place.

    Without the row at one the others still identify the slope, and centred about their own
    mean their Gram is as well conditioned as a Gram can be.  The downdate's margin for that
    row is ``2 * pairs * h**2`` to first order, a tenth of ``P eps``.
    """
    rng = np.random.default_rng(41)
    x = np.concatenate((np.tile([LEVER_H, -LEVER_H], LEVER_PAIRS), [1.0]))
    y = 0.3 * np.sign(x) + rng.normal(0.0, 0.1, len(x))
    rows = ORDERS[order](len(x))
    model = SuperGLM(features={"x": Numeric()}, family="gaussian", selection_penalty=0)
    model.fit(pd.DataFrame({"x": x[rows]}), y[rows])
    problem = problem_of(model, y[rows], np.ones(len(x)), None)
    return problem, x[rows], int(np.flatnonzero(rows == len(x) - 1)[0])


def line_without(x, y, k):
    """The least-squares line through every point but ``k``, at ``x[k]``: closed form."""
    keep = np.arange(len(x)) != k
    centre = x[keep].mean()
    slope = (
        (x[keep] - centre) @ (y[keep] - y[keep].mean()) / ((x[keep] - centre) @ (x[keep] - centre))
    )
    return y[keep].mean() + slope * (x[k] - centre)


@pytest.mark.parametrize("order", list(ORDERS))
def test_full_rank_deletion_below_a_pivot_bar_is_fitted_from_its_kept_rows(order):
    """A downdate margin under ``P eps`` beside rows that identify the model: the kept rows' fit.

    Every held-out prediction is the line through the other rows, to the rounding of a
    perfectly conditioned fit, whatever the order of the rows.
    """
    problem, x, lone = lever(order)
    assert 2 * LEVER_PAIRS * LEVER_H**2 < problem.width * 2 * U
    got = evaluate(problem, single_rows(problem.n), np.zeros(0))
    assert got.feasible and got.work.kept_fits == 1
    want = np.array([line_without(x, problem.y, k) for k in range(problem.n)])
    assert abs(want[lone]) > 1e6
    gap = np.abs(got.heldout.predictor - want)
    assert np.all(gap <= predictor_slack(1.0, want)), f"gap {gap.max():.3e} at {gap.argmax()}"
    error = problem.y - want
    resolved = 2.0 * np.abs(error) @ predictor_slack(1.0, want) + 64 * U * (error @ error)
    assert abs(got.value - error @ error) <= resolved


def paired(order):
    """1,500 rows in pairs ``+-x`` and 100 rows at zero; removing the pairs leaves no slope."""
    rng = np.random.default_rng(34)
    half = rng.normal(size=750)
    x = np.concatenate((half, -half, np.zeros(100)))
    labels = np.repeat([1, 0], [1500, 100])
    y = rng.normal(size=len(x))
    rows = ORDERS[order](len(x))
    model = SuperGLM(features={"x": Numeric()}, family="gaussian", selection_penalty=0)
    model.fit(pd.DataFrame({"x": x[rows]}), y[rows])
    problem = problem_of(model, y[rows], np.ones(len(x)), None)
    return problem, from_groups(labels[rows], len(x))


@pytest.mark.parametrize("order", list(ORDERS))
def test_singular_deletion_above_a_pivot_bar_is_refused(order):
    """A singular leave-out system whose downdate leaves a positive pivot is still refused.

    The kept rows' slope Gram is exactly zero.  A slope Hessian ``64 eps`` above the solver's,
    which is inside the ``gamma_1600`` its formation is good to, leaves the downdate a pivot of
    ``64 eps`` on a unit diagonal: thirty times ``P eps``.
    """
    problem, nb = paired(order)
    fit = solve_mode(problem, np.zeros(0))
    lifted = replace(fit, slope_hessian=fit.slope_hessian * (1.0 + 64 * 2 * U))
    for refused in (
        lambda: evaluate_fit(problem, nb, lifted, route="factor"),
        lambda: evaluate(problem, nb, np.zeros(0)),
    ):
        with pytest.raises(LeaveOutRankError, match="group 1 held out") as refusal:
            refused()
        assert refusal.value.neighbourhood == 1


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_levels_that_coincide_on_the_kept_rows_are_refused(family, route):
    """Outside group 5 level ``q`` occurs only with level ``d``: no level lies in one group.

    The count of levels per neighbourhood finds nothing, and the rows that remain when
    group 5 goes cannot tell the two columns apart.
    """
    rng = np.random.default_rng(22)
    n = 400
    f = rng.choice(list("abcd"), n, p=[0.4, 0.3, 0.2, 0.1])
    groups = rng.integers(0, 12, n)
    frame = pd.DataFrame(
        {"x1": rng.uniform(0, 1, n), "f": f, "h": np.where((f == "d") | (groups == 5), "q", "p")}
    )
    signal = np.sin(5 * frame.x1.to_numpy()) + 0.3 * (f == "b")
    y = signal + rng.normal(0, 0.3, n) if family == "gaussian" else rng.poisson(np.exp(signal))
    model = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=8), "f": Categorical(), "h": Categorical()},
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, y.astype(float))
    problem = problem_of(model, y.astype(float), np.ones(n), None)
    assert solve_mode(problem, ZERO).result.reml_hessian_rank == problem.width
    with pytest.raises(LeaveOutRankError, match="group 5 held out") as refusal:
        evaluate(problem, from_groups(groups, n), ZERO, route=route)
    assert refusal.value.neighbourhood == 5


def one_row_on_the_right():
    """A spline whose right-hand rows form one group, all but the last of them.

    With that group out, the right-hand basis functions meet the kept rows in one row only,
    so the kept rows cannot tell them apart and the penalty must.
    """
    rng = np.random.default_rng(9)
    x = np.sort(np.concatenate((rng.uniform(0.0, 0.3, 120), rng.uniform(0.7, 1.0, 60))))
    y = np.sin(5 * x) + rng.normal(0, 0.2, len(x))
    labels = ((x > 0.5) & (np.arange(len(x)) < len(x) - 1)).astype(int)
    model = SuperGLM(
        features={"x": Spline(kind="ps", n_knots=8)},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(pd.DataFrame({"x": x}), y)
    return problem_of(model, y, np.ones(len(x)), None), from_groups(labels, len(x))


def test_leave_out_system_unusable_at_one_smoothing_is_an_infeasible_trial():
    """Rows the penalty alone tells apart: fitted at an ordinary smoothing, infeasible at none.

    At ``lambda = exp(-200)`` the penalty is ``1e-43`` of the kept rows in their own factor,
    below what the solver's rank rule resolves; the rows and the penalty still identify the
    model, so the trial is infeasible and nothing is refused.
    """
    problem, nb = one_row_on_the_right()
    ordinary = evaluate(problem, nb, ZERO, gradient=False)
    assert ordinary.feasible and ordinary.work.kept_fits == 0
    want = refit_pairs(problem, nb, ZERO)
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)
    assert_pairs_equal(
        problem, pairs_of(ordinary), want, predictor_slack(kappa, want.predictor.to_numpy())
    )
    barely = evaluate(problem, nb, np.array([-200.0]))
    assert not barely.feasible and barely.infeasible_reason == "leave_out_rank"
    assert barely.fit is not None and barely.fit.certified and barely.heldout is None


def test_an_evaluation_fits_few_systems_from_their_kept_rows(monkeypatch):
    """Past ``KEPT_FITS`` systems fitted from their kept rows, the trial is infeasible."""
    problem, _, _ = lever("as-built")
    monkeypatch.setattr(engine, "KEPT_FITS", 0)
    got = evaluate(problem, single_rows(problem.n), np.zeros(0))
    assert not got.feasible and got.infeasible_reason == "leave_out_rank"


# ── Gaussian cross-products ───────────────────────────────────────────────────


def cached_twice(problem, nb, rho, route=None):
    """Two evaluations without held-out pairs on a problem of its own, and that problem."""
    own = rebuilt(problem)
    first = evaluate(own, nb, rho, heldout=False, route=route)
    again = evaluate(own, nb, rho, heldout=False, route=route)
    assert first.feasible and again.feasible and first.heldout is None and again.heldout is None
    return first, again, own


def assert_cached_is_the_row_evaluation(problem, nb, rho, kappa, route=None):
    """Value and gradient from cross-products equal those from rows, on both passes."""
    by_rows = evaluate(rebuilt(problem), nb, rho, route=route)
    first, again, own = cached_twice(problem, nb, rho, route)
    assert_same_criterion(problem, first, by_rows, kappa)
    assert_same_criterion(problem, again, by_rows, kappa)
    return by_rows, again, own


def test_repeated_folds_count_every_scored_occurrence():
    """Two repeats of three folds: 22.7691, each row's two held-out errors both counted.

    A full-fit sum of squares plus the folds' corrections gives 12.6129 on this design.
    """
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_splitter(
        RepeatedKFold(n_splits=3, n_repeats=2, random_state=2), design.frame, problem.y
    )
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)
    by_rows, again, _ = assert_cached_is_the_row_evaluation(problem, nb, ZERO, kappa)
    assert again.work.rows_read == 0 and len(by_rows.heldout.row) == 2 * problem.n
    assert again.value == pytest.approx(22.7691, abs=5e-5)
    refitted = refit_pairs(problem, nb, ZERO)
    assert again.value == pytest.approx(refitted.deviance.sum(), rel=64 * kappa * U)


@pytest.mark.parametrize("rho", [ZERO, np.array([-5.0]), np.array([6.0])], ids=["0", "-5", "6"])
def test_scored_rows_a_strict_subset_of_removed_rows(rho):
    """A forward splitter scores part of what it removes; no row is read on the second pass."""
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_splitter(TimeSeriesSplit(5), design.frame, problem.y)
    assert nb.score_ptr is not None and len(nb.scored_rows()) < int(nb.sizes().sum())
    _, _, kappa = KeptRows(problem, rho).one_step(nb)
    _, again, _ = assert_cached_is_the_row_evaluation(problem, nb, rho, kappa)
    assert again.work.rows_read == 0
    refitted = refit_pairs(problem, nb, rho)
    assert again.value == pytest.approx(refitted.deviance.sum(), rel=64 * kappa * U)


def test_fits_other_than_the_reference_use_the_same_cross_products():
    """Cross-products formed at one smoothing serve every other: the scores move with the fit."""
    design = twin("gaussian", False)
    problem, own = design.problem, rebuilt(design.problem)
    nb = from_splitter(KFold(4, shuffle=True, random_state=1), design.frame, problem.y)
    assert evaluate(own, nb, np.array([4.0]), heldout=False).work.rows_read == problem.n
    for rho in (ZERO, np.array([-6.0]), np.array([9.0])):
        _, _, kappa = KeptRows(problem, rho).one_step(nb)
        got = evaluate(own, nb, rho, heldout=False)
        assert got.work.rows_read == 0
        assert_same_criterion(problem, got, evaluate(rebuilt(problem), nb, rho), kappa)
        with_pairs = evaluate(own, nb, rho)
        assert with_pairs.work.rows_read == problem.n
        want = refit_pairs(problem, nb, rho)
        slack = predictor_slack(kappa, want.predictor.to_numpy())
        assert_pairs_equal(problem, pairs_of(with_pairs), want, slack)


def test_cached_and_uncached_neighbourhoods_in_one_evaluation():
    """Small groups go by rows, large ones by cached cross-products; the sum is the row evaluation's."""
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_groups(mixed_groups(problem.n), problem.n)
    width, sizes = problem.width, nb.sizes()
    small = 3 * sizes**2 * width + sizes**3 < width**3
    assert 0 < small.sum() < nb.m
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)
    _, again, own = assert_cached_is_the_row_evaluation(problem, nb, ZERO, kappa)
    assert sorted(own.cache.held) == np.flatnonzero(~small).tolist()
    assert again.work.rows_read == int(sizes[small].sum())
    assert again.work.factorisations == int((~small).sum())


def test_more_groups_than_the_cache_affords():
    """Under a small ``max_memory`` the largest groups are cached and the rest read rows.

    The cache holds the reference predictor and coefficients and ``8 (P^2 + P)`` bytes a
    group, within a quarter of the allowance; an allowance too small for one group holds
    nothing.
    """
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_groups(mixed_groups(problem.n), problem.n)
    width, sizes = problem.width, nb.sizes()
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)
    by_rows = evaluate(rebuilt(problem), nb, ZERO, route="factor")
    entry, reference = 8 * (width * width + width), 8 * (problem.n + width - 1)
    allowance = reference + 5 * entry + entry // 2
    with parallel_config(max_memory=4 * allowance):
        assert ncv_rows.cache_allowance() == allowance
        first, again, own = cached_twice(problem, nb, ZERO, route="factor")
    afforded = own.cache.afforded
    assert afforded.sum() == 5 == len(own.cache.held)
    assert sizes[afforded].min() >= sizes[~afforded].max()
    assert own.cache.nbytes == reference + 5 * entry <= allowance
    assert again.work.rows_read == 2 * int(sizes[~afforded].sum())
    assert_same_criterion(problem, first, by_rows, kappa)
    assert_same_criterion(problem, again, by_rows, kappa)
    with parallel_config(max_memory=4 * (reference + entry) - 8):
        first, again, own = cached_twice(problem, nb, ZERO, route="factor")
    assert own.cache.nbytes == 0 and not own.cache.held
    assert again.work.rows_read == first.work.rows_read == 2 * problem.n
    assert_same_criterion(problem, again, by_rows, kappa)


@pytest.mark.parametrize("met", [True, False], ids=["bar-met", "short-of-the-bar"])
def test_response_translation_changes_no_criterion(monkeypatch, met):
    """A response moved by 1e7 gives the same criterion and slopes, cached or not.

    Cross-products of the response itself would put the sum of squares together from terms
    of 1e14 a row.  That far from zero the fit ends at the limit of float64, a rounding to
    either side of the acceptance bar: its solves are reported once as meeting that bar and
    once as short of it, and the criterion is the same.
    """
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_splitter(KFold(4, shuffle=True, random_state=1), design.frame, problem.y)
    unmoved = evaluate(rebuilt(problem, mode_bar=None), nb, ZERO)
    moved = rebuilt(problem, y=problem.y + 1e7, mode_bar=None)
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)

    def at_the_limit(arguments):
        return "converged" if met and arguments["_mode_bar"] == BAR else "score_stagnated"

    monkeypatch.setattr(engine, "fit_irls_direct", solver_ending(at_the_limit, []))
    by_rows, again, _ = assert_cached_is_the_row_evaluation(moved, nb, ZERO, kappa)
    assert again.work.rows_read == 0
    assert by_rows.fit.certified == met and by_rows.fit.stagnated != met
    assert again.work.stagnated_fits == int(not met)
    assert_same_criterion(moved, unmoved, by_rows, kappa)


def test_problem_built_again_holds_no_cross_products():
    """Another response on the same design and neighbourhoods starts from its own rows."""
    design = twin("gaussian", False)
    problem = design.problem
    nb = from_splitter(KFold(4, shuffle=True, random_state=1), design.frame, problem.y)
    _, _, kappa = KeptRows(problem, ZERO).one_step(nb)
    first = rebuilt(problem)
    assert evaluate(first, nb, ZERO, heldout=False).feasible and len(first.cache.held) == 4
    other = np.cos(4 * design.frame.t.to_numpy()) + np.random.default_rng(5).normal(
        0, 0.3, problem.n
    )
    for again in (rebuilt(problem, y=other), replace(first, y=other)):
        got = evaluate(again, nb, ZERO, heldout=False)
        refitted = refit_pairs(again, nb, ZERO)
        assert got.value == pytest.approx(refitted.deviance.sum(), rel=64 * kappa * U)
        assert got.work.rows_read == problem.n and again.cache is not first.cache
    assert len(first.cache.held) == 4


def test_another_centre_or_structure_drops_what_is_held():
    """Cross-products are held for one neighbourhood structure about one centre pair."""
    design = twin("gaussian", False)
    problem = rebuilt(design.problem)
    folds = KFold(4, shuffle=True, random_state=1)
    nb = from_splitter(folds, design.frame, problem.y)
    fit = evaluate(problem, nb, ZERO, heldout=False).fit
    held = dict(problem.cache.held)
    assert evaluate_fit(problem, nb, fit, heldout=False).work.rows_read == 0
    assert all(problem.cache.held[k] is entry for k, entry in held.items())
    elsewhere = replace(fit, centre_hi=fit.centre_hi + 2.0**-40)
    assert evaluate_fit(problem, nb, elsewhere, heldout=False).work.rows_read == problem.n
    assert all(problem.cache.held[k] is not entry for k, entry in held.items())
    same_rows = from_splitter(folds, design.frame, problem.y)
    assert evaluate(problem, same_rows, ZERO, heldout=False).work.rows_read == problem.n


def test_gaussian_cross_products_read_no_rows():
    """Five folds: one read to form the cross-products, then none; the held-out pairs cost one."""
    c = case("gaussian", 300)
    problem = rebuilt(c.problem)
    n, width = problem.n, problem.width
    nb = from_groups(np.arange(n) % 5, n)
    first = evaluate(problem, nb, RHO, heldout=False).work
    assert (first.rows_read, first.rows_multiplied) == (n, n + 5)
    again = evaluate(problem, nb, RHO, heldout=False).work
    assert (again.rows_read, again.rows_multiplied) == (0, 5)
    assert (again.inner_solves, again.factorisations, again.solves) == (1, 5, 10)
    value_only = evaluate(problem, nb, RHO, heldout=False, gradient=False).work
    assert (value_only.rows_read, value_only.rows_multiplied, value_only.solves) == (0, 0, 5)
    with_pairs = evaluate(problem, nb, RHO).work
    assert (with_pairs.rows_read, with_pairs.rows_multiplied) == (n, 5)
    assert problem.cache.nbytes == 8 * (n + width - 1) + 5 * 8 * (width * width + width)


def test_cached_scoring_reads_no_response_of_a_zero_weight_row():
    """Zero-weight rows holding 1e200 add nothing to a criterion taken from cross-products."""
    c = case("gaussian", 300, zero_weights=True)
    idle = ~c.problem.active
    whole = rebuilt(c.problem, y=np.where(idle, 1e200, c.problem.y))
    nb = from_groups(np.arange(whole.n) % 5, whole.n)
    _, _, kappa = KeptRows(rebuilt(c.problem), RHO).one_step(nb)
    by_rows, again, _ = assert_cached_is_the_row_evaluation(whole, nb, RHO, kappa)
    assert again.work.rows_read == 0 and np.isfinite(again.value)
    assert np.all(by_rows.heldout.deviance[idle[by_rows.heldout.row]] == 0.0)


def test_poisson_keeps_nothing_between_evaluations():
    """Only a model whose curvature is its weights has cross-products to keep."""
    c = case("poisson", 300)
    problem = rebuilt(c.problem)
    nb = from_groups(np.arange(problem.n) % 5, problem.n)
    first = evaluate(problem, nb, RHO, heldout=False)
    again = evaluate(problem, nb, RHO, heldout=False)
    assert first.work.rows_read == again.work.rows_read > 0 and again.heldout is None
    assert problem.cache.nbytes == 0 and again.value == first.value
    assert again.value == pytest.approx(evaluate(problem, nb, RHO).value, rel=64 * U)


def kept_least_squares(problem, removed, rho):
    """The penalised least-squares fit to the rows outside ``removed``: its predictor on every row.

    A dense least-squares solve of the kept rows over the penalty's root, columns scaled to
    unit length, with that matrix's condition number.  Gaussian with unit weights and no
    offset.
    """
    X = problem.dm.toarray()
    n, p = X.shape
    keep = np.setdiff1d(np.arange(n), removed)
    penalty = build_penalty_matrix(
        list(problem.dm.group_matrices),
        problem.groups,
        problem.lambdas(rho),
        p,
        reml_penalties=problem.penalties,
    )
    values, vectors = np.linalg.eigh(penalty)
    root = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
    centre = X[keep].mean(axis=0)
    rows = np.column_stack([np.ones(len(keep)), X[keep] - centre])
    stacked = np.vstack([rows, np.column_stack([np.zeros(p), root])])
    scale = np.linalg.norm(stacked, axis=0)
    response = np.concatenate([problem.y[keep], np.zeros(p)])
    theta = np.linalg.lstsq(stacked / scale, response, rcond=None)[0] / scale
    singular = np.linalg.svd(stacked / scale, compute_uv=False)
    return np.column_stack([np.ones(n), X - centre]) @ theta, float(singular[0] / singular[-1])


@pytest.mark.parametrize("rho", [-14.0, -22.0])
def test_questionable_group_is_the_least_squares_fit_of_its_kept_rows(rho):
    """A light penalty that alone separates the kept rows: each prediction is their own fit's.

    At ``lambda = exp(-14)`` the downdate's smallest pivot is a healthy 1e-7 of a unit
    diagonal while its smallest eigenvalue is five orders below it; at ``exp(-22)`` both are
    small.  Either way the system is flagged and fitted from the kept rows, where the penalty
    is 1e-3 to 1e-5 of the rows in their own factor.  The bar is the square of that factor's
    condition, the sensitivity of a least-squares solution (Higham 2002, section 20.1).
    """
    problem, nb = one_row_on_the_right()
    got = evaluate(problem, nb, np.array([rho]))
    assert got.feasible and got.work.kept_fits == nb.m
    for k in range(nb.m):
        want, condition = kept_least_squares(problem, nb.drop(k), np.array([rho]))
        at = got.heldout.neighbourhood == k
        want = want[got.heldout.row[at]]
        gap = np.abs(got.heldout.predictor[at] - want)
        assert np.all(gap <= predictor_slack(condition**2, want)), f"{k}: gap {gap.max():.3e}"
