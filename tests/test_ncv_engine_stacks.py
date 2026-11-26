"""Neighbourhood-sized systems solved in stacks against the same systems taken one at a time.

Under the cost rule the engine reads the small neighbourhoods of a block sorted by size and
solves those of one size together.  A forced route takes every neighbourhood alone: the
neighbourhood-sized system by a Cholesky factor of its own (``route="woodbury"``), or the
formed leave-out Hessian (``route="factor"``).  One fit is evaluated all three ways, each on a
problem of its own, and the held-out pairs, the criterion and its gradient are compared within
the bars of ``tests/test_ncv_engine.py``: ``64 kappa U`` times the size of the predictor,
``kappa`` the largest condition number of a Jacobi-scaled leave-out Hessian of the fixture,
computed here from its kept rows.  Which neighbourhoods a stack solves, which are flagged and
fitted from their kept rows and which are refused are compared too, and the work by its counts.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit, RepeatedKFold

from superglm import SuperGLM
from superglm._parallel import parallel_config
from superglm._roundoff import gamma
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.ncv import engine, from_groups, from_neighbours, from_splitter, single_rows
from superglm.ncv import rows as ncv_rows
from superglm.ncv.engine import LeaveOutRankError, evaluate, evaluate_fit, solve_mode
from superglm.solvers.mode_score import linear_predictor
from superglm.solvers.rank import SHARED_RANK_POLICY
from tests.test_ncv_engine import (
    RHO,
    KeptRows,
    assert_same_criterion,
    case,
    predictor_slack,
    problem_of,
)
from tests.test_ncv_engine_rank_cache import ZERO, lever, rebuilt, spectral_basis, twin
from tests.test_ncv_engine_row_work import (
    COMPACT,
    DENSE,
    assert_same_evaluation,
    book,
    condition,
    rho_of,
    row_work,
)


def mixed_sizes(n, common=6, largest=12, seed=3):
    """Labels of groups scattered over ``n`` rows: sizes ``common + 1`` to ``largest`` once each, then 1 to ``common`` in turn."""
    sizes = [*range(common + 1, largest + 1), *(1 + i % common for i in range(n))]
    count = int(np.searchsorted(np.cumsum(sizes), n)) + 1
    labels = np.repeat(np.arange(count), sizes[:count])[:n]
    return np.random.default_rng(seed).permutation(labels)


def small_neighbourhoods(form, c):
    """Small neighbourhoods of ``c``'s rows: mixed groups, windows that score one row, or small folds twice."""
    n = c.problem.n
    if form == "groups":
        return from_groups(mixed_sizes(n), n)
    if form == "neighbours":
        return from_neighbours([np.arange(max(i - 3, 0), min(i + 4, n)) for i in range(n)], n)
    folds = RepeatedKFold(n_splits=n // 4, n_repeats=2, random_state=2)
    return from_splitter(folds, c.frame, c.problem.y)


def three_ways(problem, nb, rho, fit=None):
    """One fit under the cost rule, and with every neighbourhood alone by each forced route."""
    if fit is None:
        stacked = evaluate(rebuilt(problem), nb, rho)
    else:
        stacked = evaluate_fit(rebuilt(problem), nb, fit)
    alone = evaluate_fit(rebuilt(problem), nb, stacked.fit, route="woodbury")
    formed = evaluate_fit(rebuilt(problem), nb, stacked.fit, route="factor")
    assert stacked.feasible and alone.feasible and formed.feasible
    assert (alone.work.stacked, formed.work.stacked) == (0, 0)
    return stacked, alone, formed


def removed_and_weighted(problem, nb):
    """Rows each neighbourhood removes, and how many of them have weight."""
    removed = np.array([len(nb.drop(k)) for k in range(nb.m)])
    return removed, np.array([int(problem.active[nb.drop(k)].sum()) for k in range(nb.m)])


def on_small_route(problem, nb):
    """Neighbourhoods the cost rule gives the neighbourhood-sized system: ``3 a^2 P + a^3 < P^3``.

    ``a`` counts every removed row, and a neighbourhood with no weighted row takes neither route.
    """
    a, weighted = removed_and_weighted(problem, nb)
    return (weighted > 0) & (3 * a**2 * problem.width + a**3 < problem.width**3)


def disc_margins(problem, nb, rho, fit):
    """Per neighbourhood, how far its Gershgorin discs clear ``sqrt(size)`` times the flag level.

    In the test's own coordinates, rows centred about the plain working-weight mean:
    ``I - W^(1/2) Z_a H^-1 Z_a' W^(1/2)`` does not depend on the centre.  The level is the
    solver's cut between a Gram and its row factor, or ``2 P gamma_n`` when that is larger.
    """
    kept = KeptRows(
        problem, rho, at=(linear_predictor(problem.dm, fit.result, problem.offset), fit.result.beta)
    )
    hessian = kept.Z.T @ (kept.curvature[:, None] * kept.Z) + kept.S
    level = max(SHARED_RANK_POLICY.factor_rcond, 2.0 * problem.width * gamma(problem.n))
    margins = np.empty(nb.m)
    for k in range(nb.m):
        rows = nb.drop(k)
        root = np.sqrt(kept.curvature[rows])
        system = (
            np.eye(len(rows))
            - root[:, None]
            * (kept.Z[rows] @ np.linalg.solve(hessian, kept.Z[rows].T))
            * root[None, :]
        )
        reach = np.abs(system).sum(axis=1) - np.abs(np.diag(system))
        margins[k] = np.min(np.diag(system) - reach) - np.sqrt(len(rows)) * level
    return margins


def assert_stacks_take(work, problem, nb, rho, fit):
    """The stacks solve the small-route systems whose discs clear the level, and every other is factored alone.

    Single rows take the scalar form and are in neither count; no margin lies near enough to
    zero for rounding to decide a member.
    """
    removed, _ = removed_and_weighted(problem, nb)
    systems = on_small_route(problem, nb) & (removed > 1)
    margins = disc_margins(problem, nb, rho, fit)[systems]
    assert np.all(np.abs(margins) > 1e-6)
    assert (work.stacked, work.small_factorisations) == (
        int(np.sum(margins > 0)),
        int(np.sum(margins <= 0)),
    )
    assert work.stacks >= len(np.unique(removed[systems][margins > 0]))
    assert work.kept_fits == 0


# ── stacks against neighbourhoods taken alone ────────────────────────────────


@pytest.mark.parametrize("zero_weights", [False, True], ids=["weighted", "zero-weights"])
@pytest.mark.parametrize("form", ["groups", "neighbours", "repeated"])
@pytest.mark.parametrize(
    ("family", "exposure"),
    [("gaussian", "offset"), ("poisson", "offset"), ("poisson", "weight")],
)
def test_stacks_are_the_neighbourhoods_taken_alone(family, exposure, form, zero_weights):
    """Weights, an offset and rows of zero weight: every pair, the criterion and its gradient.

    Groups of one to twelve rows with six sizes that occur once; windows of four to seven
    rows that score one of them, each row removed by several; folds of four rows, each row
    scored twice.  Every neighbourhood is on the neighbourhood-sized route.
    """
    c = case(family, 600, exposure, zero_weights)
    nb = small_neighbourhoods(form, c)
    assert on_small_route(c.problem, nb).sum() == (removed_and_weighted(c.problem, nb)[1] > 0).sum()
    stacked, alone, formed = three_ways(c.problem, nb, RHO)
    kappa = condition(c.problem, nb, RHO, stacked.fit)
    assert_same_evaluation(c.problem, stacked, alone, kappa)
    assert_same_evaluation(c.problem, stacked, formed, kappa)
    assert_stacks_take(stacked.work, c.problem, nb, RHO, stacked.fit)
    assert stacked.work.stacked > 0 == stacked.work.factorisations


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_stacks_under_both_weight_semantics_with_a_select_term(family, semantics):
    """Whole-number weights with zeros, an offset and a ``select=True`` term, on a design of lookups.

    The larger groups are past the cost rule's switch and take the formed leave-out Hessian.
    """
    b = book(family, semantics)
    nb = from_groups(mixed_sizes(b.problem.n), b.problem.n)
    rho = rho_of(b.problem)
    stacked, alone, formed = three_ways(b.problem, nb, rho)
    kappa = condition(b.problem, nb, rho, stacked.fit)
    assert_same_evaluation(b.problem, stacked, alone, kappa)
    assert_same_evaluation(b.problem, stacked, formed, kappa)
    assert_stacks_take(stacked.work, b.problem, nb, rho, stacked.fit)
    _, weighted = removed_and_weighted(b.problem, nb)
    past = int(np.sum((weighted > 0) & ~on_small_route(b.problem, nb)))
    assert stacked.work.stacked > 0 < past == stacked.work.factorisations


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_stacks_in_the_coordinates_an_aliased_fit_identified(family):
    """A duplicated column: stacks of the rows in the fit's representative columns, and in a retained basis."""
    aliased, clean = twin(family, True), twin(family, False)
    n = aliased.problem.n
    nb = from_groups(mixed_sizes(n, common=3, largest=4), n)
    fit = solve_mode(aliased.problem, ZERO)
    assert fit.certified and fit.result.reml_hessian_rank == aliased.problem.width - 1
    kappa = KeptRows(clean.problem, ZERO).one_step(nb)[2]
    for coordinates in (fit, spectral_basis(fit)):
        stacked, alone, formed = three_ways(aliased.problem, nb, ZERO, fit=coordinates)
        assert_same_evaluation(aliased.problem, stacked, alone, kappa)
        assert_same_evaluation(aliased.problem, stacked, formed, kappa)
        assert stacked.work.stacked > 0
        assert stacked.work.stacked + stacked.work.small_factorisations == int(
            np.sum(removed_and_weighted(aliased.problem, nb)[0] > 1)
        )


# ── counts ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_many_small_groups_take_one_stack_a_size(family):
    """Groups of one to twelve rows: one stack a size in a block, no system factored alone, the rows read once.

    Under a small allowance the stacks are those of each block of rows, no read is longer
    than a block and one group, and the evaluation is the same.
    """
    c = case(family, 600)
    p, n = c.problem, c.problem.n
    nb = from_groups(mixed_sizes(n), n)
    removed, _ = removed_and_weighted(p, nb)
    whole = evaluate(p, nb, RHO)
    assert np.all(disc_margins(p, nb, RHO, whole.fit)[removed > 1] > 1e-6)
    work = whole.work
    assert (work.stacks, work.stacked) == (len(np.unique(removed)) - 1, int(np.sum(removed > 1)))
    assert (work.factorisations, work.small_factorisations, work.solves) == (0, 0, 0)
    third = int(family == "poisson")
    assert (work.rows_read, work.rows_multiplied) == (n + third * n, n + nb.m + third * n)
    reads = []

    def recorded(dm, rows, centre_hi, centre_lo):
        reads.append(len(rows))
        return ncv_rows.augmented_rows(dm, rows, centre_hi, centre_lo)

    with pytest.MonkeyPatch.context() as patch, parallel_config(max_memory=64 * 1024):
        patch.setattr(engine, "augmented_rows", recorded)
        limit = ncv_rows.chunk_rows(p.width)
        blocked = evaluate_fit(p, nb, whole.fit)
    assert limit == 256 and max(reads) <= limit + int(removed.max())
    assert sum(reads) == blocked.work.rows_read == work.rows_read
    assert blocked.work.stacks > work.stacks and blocked.work.stacked == work.stacked
    assert blocked.work.small_factorisations == 0
    assert_same_evaluation(p, blocked, whole, condition(p, nb, RHO, whole.fit))


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_rows_of_zero_weight_count_in_the_size_that_decides_the_route(family):
    """Twelve groups of 100 rows, two of them weighted: each formed alone, no 100-row stack.

    A stack carries a removed row of zero weight as a row of the identity, so the rows a
    neighbourhood removes decide its route.  The evaluation is the forced routes'.
    """
    rng = np.random.default_rng(17)
    n, groups = 1200, 12
    frame = pd.DataFrame({"z": rng.normal(size=n), "f": rng.choice(list("abcd"), n)})
    signal = (0.4 * frame.z + 0.3 * (frame.f == "b")).to_numpy()
    y = signal + rng.normal(0.0, 0.5, n)
    if family == "poisson":
        y = rng.poisson(np.exp(signal)).astype(float)
    labels = np.repeat(np.arange(groups), n // groups)
    weights = np.where(np.arange(n) % (n // groups) < 2, rng.choice([0.5, 1.0, 2.0], n), 0.0)
    model = SuperGLM(features={"z": Numeric(), "f": Categorical()}, family=family)
    model.fit(frame, y, sample_weight=weights)
    problem = problem_of(model, y, weights, np.zeros(n))
    nb = from_groups(labels, n)
    assert problem.width == 5 and not on_small_route(problem, nb).any()
    stacked, alone, formed = three_ways(problem, nb, np.zeros(0))
    assert (stacked.work.stacks, stacked.work.stacked) == (0, 0)
    assert (stacked.work.factorisations, stacked.work.small_factorisations) == (groups, 0)
    kappa = condition(problem, nb, np.zeros(0), stacked.fit)
    assert_same_evaluation(problem, stacked, alone, kappa)
    assert_same_evaluation(problem, stacked, formed, kappa)


@pytest.mark.parametrize("zero_weights", [False, True], ids=["weighted", "zero-weights"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_one_row_groups_cost_what_single_rows_cost(family, zero_weights):
    """A group a row, in the rows' order or any other: the single-row form's counts and its evaluation."""
    c = case(family, 300, zero_weights=zero_weights)
    p, n = c.problem, c.problem.n
    rows = evaluate(p, single_rows(n), RHO)
    kappa = condition(p, single_rows(n), RHO, rows.fit)
    assert (rows.work.small_factorisations, rows.work.stacks, rows.work.solves) == (0, 0, 0)
    for labels in (np.arange(n), np.random.default_rng(8).permutation(n)):
        groups = evaluate_fit(rebuilt(p), from_groups(labels, n), rows.fit)
        assert groups.work == replace(rows.work, inner_solves=0, pirls_iterations=0)
        order = np.argsort(groups.heldout.row)
        np.testing.assert_array_equal(groups.heldout.row[order], rows.heldout.row)
        assert np.all(
            np.abs(groups.heldout.predictor[order] - rows.heldout.predictor)
            <= predictor_slack(kappa, rows.heldout.predictor)
        )
        np.testing.assert_array_equal(
            groups.heldout.deviance[order] == 0.0, rows.heldout.deviance == 0.0
        )
        assert_same_criterion(p, groups, rows, kappa)


def test_pairs_of_a_stack_enter_the_forms_kept_row_by_row():
    """Few small folds on a design without tables: each stack's pairs add their forms to every row's.

    Seven folds, six of three rows and one of one row, are few enough for the third-derivative
    pass to keep ``sum_k (z' u_k)(z' Delta_k)`` row by row.  The reference forms it from dense
    blocks.
    """
    c = case("poisson", 600)
    p, n = c.problem, c.problem.n
    fold = np.full(n, -1)
    fold[np.random.default_rng(4).permutation(n)[:19]] = np.repeat(np.arange(7), 3)[:19]
    nb = from_splitter(PredefinedSplit(fold), c.frame, p.y)
    assert sorted(nb.sizes().tolist()) == [1, 3, 3, 3, 3, 3, 3] and 2 * nb.m <= p.width
    with row_work(DENSE):
        dense = evaluate(rebuilt(p), nb, RHO)
    with row_work(COMPACT):
        kept = evaluate_fit(rebuilt(p), nb, dense.fit)
    assert dense.work.rows_streamed == 0 < kept.work.rows_streamed
    assert (
        (kept.work.stacks, kept.work.stacked) == (1, 6) == (dense.work.stacks, dense.work.stacked)
    )
    assert_same_evaluation(p, kept, dense, condition(p, nb, RHO, dense.fit))


# ── flags and refusals ────────────────────────────────────────────────────────

LEVER_H, LEVER_PAIRS = 1e-9, 20


def lever_pairs():
    """Twenty pairs of rows at ``x = +-h`` and one pair at ``x = 1`` and ``x = 2``, a group a pair.

    Two more columns make a pair a neighbourhood-sized system under the cost rule.  Without
    the last pair the others still identify the slope in ``x``, and centred about their own
    mean and scaled their rows are well conditioned; the downdate that removes the pair
    keeps ``2 * pairs * h**2`` of a Hessian entry near five, below what float64 resolves.
    """
    rng = np.random.default_rng(41)
    x = np.concatenate((np.tile([LEVER_H, -LEVER_H], LEVER_PAIRS), [1.0, 2.0]))
    frame = pd.DataFrame({"x": x, "u": rng.normal(size=len(x)), "v": rng.normal(size=len(x))})
    y = 0.3 * np.sign(x) + 0.2 * frame.u - 0.1 * frame.v + rng.normal(0.0, 0.1, len(x))
    model = SuperGLM(
        features={name: Numeric() for name in frame}, family="gaussian", selection_penalty=0
    )
    model.fit(frame, y.to_numpy())
    problem = problem_of(model, y.to_numpy(), np.ones(len(x)), None)
    return problem, frame.to_numpy(), np.arange(len(x)) // 2


def fit_without(X, y, removed):
    """The least-squares fit of every row but ``removed``, at those rows, and its condition.

    The kept rows are centred about their own mean and their columns scaled to unit length.
    """
    keep = np.setdiff1d(np.arange(len(y)), removed)
    centre = X[keep].mean(axis=0)
    scale = np.linalg.norm(X[keep] - centre, axis=0)
    slopes, _, _, singular = np.linalg.lstsq(
        (X[keep] - centre) / scale, y[keep] - y[keep].mean(), rcond=None
    )
    return y[keep].mean() + ((X[removed] - centre) / scale) @ slopes, singular[0] / singular[-1]


def test_flagged_member_is_fitted_from_its_kept_rows_and_its_stack_is_solved():
    """One member of a stack of twenty-one has a questionable system: it alone leaves the stack.

    Every held-out prediction is the least-squares fit of the other pairs, within the square
    of that fit's condition, the sensitivity of a least-squares solution (Higham 2002,
    section 20.1); the twenty other members are solved together.
    """
    problem, X, labels = lever_pairs()
    assert 2 * LEVER_PAIRS * LEVER_H**2 < problem.width * engine.EPS
    nb = from_groups(labels, problem.n)
    got = evaluate(problem, nb, np.zeros(0))
    alone = evaluate_fit(rebuilt(problem), nb, got.fit, route="woodbury")
    assert got.feasible and alone.feasible
    assert (got.work.kept_fits, alone.work.kept_fits) == (1, 1)
    assert (got.work.stacks, got.work.stacked, got.work.small_factorisations) == (1, LEVER_PAIRS, 0)
    assert alone.work.small_factorisations == LEVER_PAIRS
    fits = [fit_without(X, problem.y, nb.drop(k)) for k in range(nb.m)]
    want = np.concatenate([fit for fit, _ in fits])
    slack = predictor_slack(max(condition for _, condition in fits) ** 2, want)
    assert np.abs(want[-2:]).min() > 1e6
    for evaluation in (got, alone):
        gap = np.abs(evaluation.heldout.predictor - want)
        assert np.all(gap <= slack), f"gap {(gap / slack).max():.3e} of its bar at {gap.argmax()}"


def test_flagged_single_row_group_is_fitted_from_its_kept_rows():
    """One-row groups, one of them flagged by ``1 - W h``: the single-row form's flag and its fit."""
    problem, _, _ = lever("as-built")
    rows = evaluate(problem, single_rows(problem.n), np.zeros(0))
    groups = evaluate_fit(rebuilt(problem), from_groups(np.arange(problem.n), problem.n), rows.fit)
    assert rows.feasible and groups.feasible
    assert groups.work == replace(rows.work, inner_solves=0, pirls_iterations=0)
    assert groups.work.kept_fits == 1 and groups.work.stacked == 0
    gap = np.abs(groups.heldout.predictor - rows.heldout.predictor)
    assert np.all(gap <= predictor_slack(1.0, rows.heldout.predictor))


def test_stack_past_the_limit_on_kept_fits_is_an_infeasible_trial(monkeypatch):
    """A flagged member of a stack counts against ``KEPT_FITS`` as it does alone."""
    problem, _, labels = lever_pairs()
    nb = from_groups(labels, problem.n)
    fit = solve_mode(problem, np.zeros(0))
    monkeypatch.setattr(engine, "KEPT_FITS", 0)
    for route in (None, "woodbury"):
        got = evaluate_fit(rebuilt(problem), nb, fit, route=route)
        assert not got.feasible and got.infeasible_reason == "leave_out_rank"
        assert got.past_kept_fits and got.work.stacked == 0


def lost_levels(levels):
    """``case``'s rows with factor levels of their own on the rows of some groups of two and three.

    ``levels`` maps a group to its level.  With such a group held out nothing estimates its level.
    """
    c = case("gaussian", 300)
    n = c.problem.n
    labels = mixed_sizes(n, common=3, largest=3)
    f = c.frame.f.to_numpy().copy()
    for group, level in levels.items():
        f[labels == group] = level
    model = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=8), "f": Categorical()},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(c.frame.assign(f=f), c.problem.y)
    return problem_of(model, c.problem.y, np.ones(n), None), from_groups(labels, n)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("route", [None, "woodbury"], ids=["stacked", "alone"])
def test_singular_member_of_a_stack_is_refused_with_its_neighbourhood(route):
    """A factor level wholly inside a group of two, among fifty groups of two: the typed refusal.

    Pins the refusal such a group gave when every neighbourhood was solved alone: a stack
    must leave the member out and name it as before.
    """
    problem, nb = lost_levels({7: "e"})
    assert len(nb.drop(7)) == 2
    with pytest.raises(LeaveOutRankError, match="group 7 held out") as refusal:
        evaluate(problem, nb, ZERO, route=route)
    assert refusal.value.neighbourhood == 7


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("route", [None, "woodbury"], ids=["stacked", "alone"])
def test_refusal_names_the_first_unidentified_neighbourhood_in_order(route):
    """Two groups lose a level, the earlier one of three rows and the later of two: the earlier is named.

    Pins the neighbourhood a refusal named when every neighbourhood was solved alone, in
    order: stacks are solved by size, and the members a stack leaves out are still decided
    in the neighbourhoods' own order.
    """
    problem, nb = lost_levels({5: "e", 127: "g"})
    assert (len(nb.drop(5)), len(nb.drop(127))) == (3, 2)
    with pytest.raises(LeaveOutRankError, match="group 5 held out") as refusal:
        evaluate(problem, nb, ZERO, route=route)
    assert refusal.value.neighbourhood == 5


def test_member_a_stack_does_not_vouch_for_is_factored_alone(monkeypatch):
    """A member whose discs do not clear the level leaves its stack; the others are solved as before.

    Forced: the stack's test is made to leave out the first member of every stack.  Each is
    then factored alone and found sound, and the evaluation is the one with every member in
    its stack.
    """
    c = case("poisson", 600)
    p, n = c.problem, c.problem.n
    nb = from_groups(mixed_sizes(n), n)
    whole = evaluate(p, nb, RHO)
    assert whole.work.small_factorisations == 0
    dominant = engine._dominant

    def without_the_first(system, level):
        sound = dominant(system, level)
        sound[0] = False
        return sound

    monkeypatch.setattr(engine, "_dominant", without_the_first)
    got = evaluate_fit(rebuilt(p), nb, whole.fit)
    assert got.work.small_factorisations == whole.work.stacks
    assert got.work.solves == 2 * whole.work.stacks
    assert got.work.stacked == whole.work.stacked - whole.work.stacks
    assert_same_evaluation(p, got, whole, condition(p, nb, RHO, whole.fit))
