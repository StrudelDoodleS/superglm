"""The engine's row work through the design's stored form against its dense blocks.

One fit is evaluated twice: with ``rows.COMPACT_ENTRIES`` out of reach, so that every row is read
in a dense block, and with it at one entry, so that every set of rows the compact products serve
goes through them.  The dense blocks are the reference.  The held-out pairs, the criterion and
its gradient are compared within the bars of ``tests/test_ncv_engine.py``: ``64 kappa U`` times
the size of the predictor, ``kappa`` the largest condition number of a Jacobi-scaled leave-out
Hessian of the fixture, computed here from its kept rows.  The cost rule is tested by its
counts at the library's own threshold.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import SuperGLM
from superglm._group_matrix._lookup_tables import lookup_plan, table_leverages
from superglm._roundoff import gamma
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.ncv import engine, from_groups, from_splitter, single_rows
from superglm.ncv import rows as ncv_rows
from superglm.ncv.engine import evaluate, evaluate_fit, solve_mode
from superglm.ncv.rows import CentredRows, augmented_rows, serves_compact
from superglm.ncv.search import search
from superglm.solvers.mode_score import dense_columns, linear_predictor
from tests.test_ncv_engine import (
    RHO,
    KeptRows,
    U,
    assert_same_criterion,
    case,
    deviance_slope,
    neighbourhoods,
    predictor_slack,
    problem_of,
    translated,
)
from tests.test_ncv_engine_rank_cache import (
    ZERO,
    rebuilt,
    spectral_basis,
    twin,
    twin_neighbourhoods,
    twin_reference,
)
from tests.test_ncv_search import TOL, fitted, sizing
from tests.test_ncv_search import Case as SearchCase

DENSE, COMPACT = 1 << 62, 1


@contextmanager
def row_work(entries):
    """Sets of rows of at least ``entries`` dense entries go through compact products."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ncv_rows, "COMPACT_ENTRIES", entries)
        yield


def both(problem, nb, rho, **settings):
    """One fit evaluated from dense blocks and from compact products, each on a problem of its own."""
    with row_work(DENSE):
        dense = evaluate(rebuilt(problem), nb, rho, **settings)
    with row_work(COMPACT):
        compact = evaluate_fit(rebuilt(problem), nb, dense.fit, **settings)
    assert dense.feasible and compact.feasible
    assert dense.work.rows_streamed == 0 < compact.work.rows_streamed
    return dense, compact


def condition(problem, nb, rho, fit):
    """The largest condition number of a Jacobi-scaled leave-out Hessian, from the kept rows."""
    at = linear_predictor(problem.dm, fit.result, problem.offset), fit.result.beta
    return KeptRows(problem, rho, at=at).one_step(nb)[2]


def searching(problem):
    """``problem`` with the inner-fit bars a search runs at, not the tests' tight one."""
    return rebuilt(problem, mode_bar=engine.ncv_mode_bar(), mode_aim=engine.ncv_mode_aim())


def assert_same_evaluation(problem, got, want, kappa):
    """Two evaluations of one fit agree pair by pair, in the criterion and in its gradient.

    A predictor within ``slack`` moves a log-link mean by ``mean expm1(slack)`` and the deviance
    by ``|D'| slack`` and a second-order term of at most ``w mean e^slack slack^2``; an
    identity-link mean is the predictor and its second-order term is ``w slack^2``.
    """
    held, other = got.heldout, want.heldout
    np.testing.assert_array_equal(held.row, other.row)
    np.testing.assert_array_equal(held.neighbourhood, other.neighbourhood)
    slack = predictor_slack(kappa, other.predictor)
    gap = np.abs(held.predictor - other.predictor)
    assert np.all(gap <= slack), f"predictor gap {gap.max():.3e} against {slack.min():.3e}"
    w = problem.weights[other.row]
    curved = other.mean * np.exp(slack) if problem.loss.log_mean else np.ones(len(slack))
    assert np.all(np.abs(held.mean - other.mean) <= curved * slack * (1.0 + slack))
    bar = (deviance_slope(problem, other) + w * curved * slack) * slack + 64.0 * U * other.deviance
    assert np.all(np.abs(held.deviance - other.deviance) <= bar)
    assert_same_criterion(problem, got, want, kappa)


# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Book:
    problem: engine.NCVProblem
    frame: pd.DataFrame
    labels: np.ndarray


@cache
def book(family, semantics="prior", n=1000) -> Book:
    """Whole-number rating factors, so every group is a lookup or a dense column and the design has tables.

    Two P-splines (one with ``select=True``), a numeric column far from zero and a factor;
    whole-number weights with every thirteenth row at zero, and an offset.  From about a
    thousand rows the design stores each spline as its distinct rows and an index.
    """
    rng = np.random.default_rng(71)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 60, n).astype(float),
            "veh": rng.integers(0, 12, n).astype(float),
            "z": rng.normal(2000.0, 5.0, n),
            "f": rng.choice(list("abc"), n, p=[0.5, 0.3, 0.2]),
        }
    )
    weights = rng.choice([1.0, 2.0, 3.0], n)
    weights[::13] = 0.0
    offset = np.log(rng.uniform(0.5, 1.5, n))
    signal = (
        np.sin(frame.age / 8) + 0.05 * frame.veh + 0.03 * (frame.z - 2000) + 0.4 * (frame.f == "b")
    ).to_numpy()
    if family == "gaussian":
        y = signal + offset + rng.normal(0, 0.5, n) / np.sqrt(np.maximum(weights, 1.0))
    else:
        y = rng.poisson(np.exp(signal - 0.5 + offset)).astype(float)
    model = SuperGLM(
        features={
            "age": Spline(kind="ps", n_knots=6),
            "veh": Spline(kind="ps", n_knots=4, select=True),
            "z": Numeric(),
            "f": Categorical(),
        },
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
        weight_semantics=semantics,
    )
    model.fit(frame, y, sample_weight=weights, offset=offset)
    problem = problem_of(model, y, weights, offset)
    assert problem.weight_semantics == semantics
    assert CentredRows(problem.dm, np.zeros(problem.dm.p), None).has_tables
    return Book(problem, frame, rng.integers(0, 9, n))


def book_neighbourhoods(form, b: Book):
    n = b.problem.n
    if form == "rows":
        return single_rows(n)
    if form == "groups":
        return from_groups(b.labels, n)
    return from_splitter(KFold(4, shuffle=True, random_state=3), b.frame, b.problem.y)


def rho_of(problem):
    return np.linspace(-1.0, 1.0, len(problem.names))


@cache
def terms(family) -> Book:
    """A double penalty, a tensor and a spline-by-factor term over whole-number covariates."""
    rng = np.random.default_rng(61)
    n = 420
    frame = pd.DataFrame(
        {
            "x1": rng.integers(0, 25, n) / 25.0,
            "x2": rng.integers(0, 20, n) / 5.0 - 2.0,
            "x3": rng.uniform(0, 1, n),
            "f": rng.choice(list("abc"), n, p=[0.5, 0.3, 0.2]),
        }
    )
    weights = rng.choice([1.0, 2.0], n)
    signal = (
        np.sin(6 * frame.x1) * (1 + 0.5 * (frame.f == "b"))
        + 0.3 * frame.x2**2
        + 0.4 * frame.x1 * frame.x2
        + 0.5 * frame.x3
    ).to_numpy()
    y = (
        signal + rng.normal(0, 0.5, n)
        if family == "gaussian"
        else rng.poisson(np.exp(signal - 0.5)).astype(float)
    )
    model = SuperGLM(
        features={
            "x1": Spline(kind="ps", n_knots=4),
            "x2": Spline(kind="ps", n_knots=4),
            "x3": Spline(kind="ps", n_knots=4, select=True),
            "f": Categorical(),
        },
        interactions=[("x1", "x2"), ("x1", "f")],
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, y, sample_weight=weights)
    return Book(problem_of(model, y, weights, None), frame, rng.integers(0, 6, n))


# ── every hold-out form the compact products serve ────────────────────────────


@pytest.mark.parametrize("zero_weights", [False, True], ids=["weighted", "zero-weights"])
@pytest.mark.parametrize("form", ["groups", "cv", "forward", "repeated"])
@pytest.mark.parametrize(
    ("family", "exposure"),
    [("gaussian", "offset"), ("poisson", "offset"), ("poisson", "weight")],
)
def test_compact_rows_are_the_dense_blocks(family, exposure, form, zero_weights):
    """Weights, an offset and a numeric column at 2000: each neighbourhood through its own row subset.

    A fold is read by no dense block, and the third-derivative term of a few folds is kept
    row by row from their pairs.
    """
    c = case(family, 300, exposure, zero_weights)
    nb = neighbourhoods(form, c)
    dense, compact = both(c.problem, nb, RHO)
    assert_same_evaluation(c.problem, compact, dense, condition(c.problem, nb, RHO, dense.fit))
    if form == "cv":
        # blocks: every fold once and, for Poisson, the weighted rows' third pass; compact
        # products: every row scored, its weighted rows in the Gram and in two third-derivative sums
        n, weighted, poisson = c.problem.n, int(c.problem.active.sum()), family == "poisson"
        assert dense.work.rows_read == n + poisson * weighted
        assert compact.work.rows_streamed == n + (3 if poisson else 1) * weighted
    if form != "groups":
        assert compact.work.rows_read == 0


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize(
    ("family", "form"),
    [
        ("gaussian", "groups"),
        ("gaussian", "cv"),
        ("poisson", "groups"),
        ("poisson", "cv"),
        ("poisson", "rows"),
    ],
)
def test_lookup_design_reads_its_third_pass_from_tables(family, form, semantics):
    """A design of lookups: the forms ``z' M z`` of any rank come from its term-pair tables.

    Single rows read no dense row: their leverages, ``Z' v``, the Gram that is their ``M`` and
    the third pass are four passes of the stored form.  A ``select=True`` term, both weight
    semantics, an offset and rows of zero weight.
    """
    b = book(family, semantics)
    nb = book_neighbourhoods(form, b)
    rho = rho_of(b.problem)
    dense, compact = both(b.problem, nb, rho)
    assert_same_evaluation(b.problem, compact, dense, condition(b.problem, nb, rho, dense.fit))
    weighted = int(b.problem.active.sum())
    assert compact.work.rows_read == 0
    if form == "rows":
        assert dense.work.rows_read == 2 * weighted
        assert compact.work.rows_streamed == 4 * weighted


@pytest.mark.parametrize("form", ["cv", "forward", "repeated"])
def test_gaussian_cross_products_are_kept_from_compact_rows(form):
    """The cache's cross-products, over removed and over scored rows, formed without a dense block.

    A forward splitter scores part of what it removes, so its scored cross-products are
    those of a subset of the neighbourhood's rows.  The cached evaluation equals the dense one
    with its pairs, and a second one reads nothing.
    """
    c = case("gaussian", 300, zero_weights=True)
    nb = neighbourhoods(form, c)
    with row_work(DENSE):
        dense = evaluate(rebuilt(c.problem), nb, RHO)
    own = rebuilt(c.problem)
    with row_work(COMPACT):
        first = evaluate_fit(own, nb, dense.fit, heldout=False)
        again = evaluate_fit(own, nb, dense.fit, heldout=False)
    assert first.work.rows_read == 0 < first.work.rows_streamed
    assert (again.work.rows_read, again.work.rows_streamed) == (0, 0)
    kappa = condition(c.problem, nb, RHO, dense.fit)
    for got in (first, again):
        assert got.heldout is None
        assert_same_criterion(c.problem, got, dense, kappa)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_every_term_type_through_compact_rows(family):
    """A double penalty, a tensor and a spline-by-factor term, each as a row subset of its own."""
    b = terms(family)
    nb = from_groups(b.labels, b.problem.n)
    rho = np.linspace(-1.0, 1.5, len(b.problem.names))
    dense, compact = both(b.problem, nb, rho, route="factor")
    assert len(b.problem.names) >= 6 and compact.work.rows_read == 0
    assert_same_evaluation(b.problem, compact, dense, condition(b.problem, nb, rho, dense.fit))


# ── a column far from zero, and an aliased design ─────────────────────────────


def test_column_translation_leaves_compact_heldout_predictions_unchanged():
    """Adding 2**52 to a column of small integers moves no held-out prediction of the compact rows.

    The leave-out Gram is the solver's about the neighbourhood's own mean plus the outer
    product of that mean's offset from the fit's centre, each formed on rows centred about an
    exact pair: nothing is subtracted at the column's own size.
    """
    rho = np.zeros(1)
    problem, nb = translated(0.0)
    moved, _ = translated(2.0**52)
    with row_work(DENSE):
        want = evaluate(problem, nb, rho, route="factor")
    kappa = condition(problem, nb, rho, want.fit)
    for one in (problem, moved):
        _, compact = both(one, nb, rho, route="factor")
        assert compact.work.rows_read == 0
        assert_same_evaluation(problem, compact, want, kappa)


@pytest.mark.parametrize("form", ["groups", "forward"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_aliased_design_through_compact_rows(family, form):
    """A rank-deficient fit's identified coordinates: compact products lifted to the design's columns."""
    aliased = twin(family, True)
    nb, (_, _, kappa) = twin_reference(family, form)
    dense, compact = both(aliased.problem, nb, ZERO, route="factor")
    # seventeen groups' pairs outnumber half the design's eleven columns: their third pass stays in blocks
    blocks = int(aliased.problem.active.sum()) if (family, form) == ("poisson", "groups") else 0
    assert compact.work.rows_read == blocks < dense.work.rows_read
    assert_same_evaluation(aliased.problem, compact, dense, kappa)


def test_retained_basis_through_compact_rows():
    """The same where the fit hands back a basis of its identified slopes."""
    aliased = twin("gaussian", True)
    general = spectral_basis(solve_mode(aliased.problem, ZERO))
    nb, (_, _, kappa) = twin_reference("gaussian", "forward")
    assert nb.m == twin_neighbourhoods("forward", aliased).m
    with row_work(DENSE):
        dense = evaluate_fit(rebuilt(aliased.problem), nb, general, route="factor")
    with row_work(COMPACT):
        compact = evaluate_fit(rebuilt(aliased.problem), nb, general, route="factor")
    assert compact.work.rows_read == 0 < compact.work.rows_streamed
    assert_same_evaluation(aliased.problem, compact, dense, kappa)


# ── the products themselves ───────────────────────────────────────────────────


@pytest.mark.parametrize("anchor", [0.0, 2.0**40], ids=["near-zero", "far-from-zero"])
def test_centred_rows_products_are_the_dense_rows(anchor):
    """``Z v``, ``Z' v``, ``Z' W Z`` and ``z' M z`` of a row subset against its dense rows.

    Each sum of ``n`` terms is within ``gamma_n`` of the sum of its terms' sizes.  A dense
    column is centred row by row, so its terms are those of ``Z``; every other column takes
    its own product less the centre's: terms of size ``|x| <= |z| + |c|`` and the centre's
    own, ``|z| + 2 |c|`` in all.
    """
    b = book("poisson")
    frame = b.frame.assign(z=np.round(b.frame.z) - 2000.0 + anchor)
    model = SuperGLM(
        features={"age": Spline(kind="ps", n_knots=6), "z": Numeric(), "f": Categorical()},
        family="poisson",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, b.problem.y, sample_weight=b.problem.weights)
    problem = problem_of(model, b.problem.y, b.problem.weights, None)
    fit = solve_mode(problem, np.zeros(1))
    rng = np.random.default_rng(9)
    rows = np.sort(rng.choice(problem.n, 400, replace=False))
    compact = CentredRows(problem.dm, fit.centre_hi, fit.centre_lo).subset(rows)
    Z = augmented_rows(problem.dm, rows, fit.centre_hi, fit.centre_lo)
    centre = np.concatenate(([0.0], np.where(dense_columns(problem.dm), 0.0, fit.centre_hi)))
    sizes = np.abs(Z) + 2.0 * np.abs(centre)
    bound = 4.0 * gamma(len(rows) + problem.width)
    theta, values = rng.normal(size=problem.width), rng.normal(size=len(rows))
    weights = problem.weights[rows] * rng.uniform(0.5, 2.0, len(rows))
    assert np.count_nonzero(weights == 0.0) > 5
    assert np.all(np.abs(compact.matvec(theta) - Z @ theta) <= bound * (sizes @ np.abs(theta)))
    assert np.all(
        np.abs(compact.rmatvec(values) - Z.T @ values) <= bound * (sizes.T @ np.abs(values))
    )
    gram = Z.T @ (weights[:, None] * Z)
    assert np.all(
        np.abs(compact.gram(weights) - gram) <= bound * (sizes.T @ (weights[:, None] * sizes))
    )
    half = rng.normal(size=(problem.width, problem.width))
    matrix = half + half.T
    active = weights > 0.0
    forms = compact.quadratic_forms(matrix, active)
    want = np.einsum("ij,ij->i", Z @ matrix, Z)
    within = np.einsum("ij,ij->i", sizes @ np.abs(matrix), sizes)
    assert np.all(np.abs(forms - want)[active] <= bound * within[active])
    assert not np.any(compact.gram(np.zeros(len(rows))))


# ── the search ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_search_chooses_the_same_curve_by_either_row_work(family):
    """The same smoothing, criterion, EDF and fitted curve from dense blocks and from compact rows."""
    b = book(family)
    problem = searching(b.problem)
    nb = book_neighbourhoods("cv", b)
    c = SearchCase(problem, nb, dict.fromkeys(problem.names, 1.0))
    with row_work(DENSE):
        dense = search(rebuilt(problem), nb, configured=c.configured).result
    with row_work(COMPACT):
        compact = search(rebuilt(problem), nb, configured=c.configured).result
    assert dense.converged and compact.converged
    assert dense.work["rows_streamed"] == 0 < compact.work["rows_streamed"]
    assert compact.work["rows_read"] == 0 < dense.work["rows_read"]
    assert compact.objective == pytest.approx(dense.objective, rel=10 * TOL)
    edf, eta = fitted(c, dense)
    other_edf, other_eta = fitted(c, compact)
    assert other_edf == pytest.approx(edf, rel=0.02)
    assert np.max(np.abs(other_eta - eta)) <= 2e-3 * np.std(eta)


# ── the cost rule, by its counts ──────────────────────────────────────────────


def test_rows_go_compact_from_the_threshold_on():
    assert ncv_rows.COMPACT_ENTRIES == 1 << 18
    assert not serves_compact((1 << 18) // 64 - 1, 64) and serves_compact((1 << 18) // 64, 64)


def test_cost_rule_serves_large_folds_compact_and_small_groups_dense():
    """At the library's threshold: two folds of 12,000 rows read no dense block, 200 groups keep theirs.

    The third pass of the two folds is their pairs' own.  The 200 groups are read once each,
    and their third pass, whose ``M`` has rank 200 on a design without tables, once more.
    """
    p = searching(case("poisson", 24_000).problem)
    n = p.n
    assert serves_compact(n // 2, p.width) and not serves_compact(n // 200, p.width)
    folds = evaluate(p, from_groups(np.arange(n) % 2, n), RHO)
    assert folds.feasible and folds.work.factorisations == 2
    assert (folds.work.rows_read, folds.work.rows_streamed) == (0, 4 * n)
    assert folds.work.rows_multiplied == 2
    groups = evaluate_fit(p, from_groups(np.arange(n) % 200, n), folds.fit)
    assert groups.feasible and groups.work.factorisations == 200
    assert (groups.work.rows_read, groups.work.rows_streamed) == (2 * n, 0)


def test_third_pass_keeps_the_pairs_forms_while_they_are_fewer_than_half_the_width():
    """``2 K <= P`` pairs: two products a pair beat a block's ``P`` a row; one more pair and they do not."""
    p = searching(case("poisson", 24_000).problem)
    n, few = p.n, p.width // 2
    fit = solve_mode(p, RHO)
    assert not serves_compact(n // few, p.width) and serves_compact(n, p.width)
    kept = evaluate_fit(p, from_groups(np.arange(n) % few, n), fit)
    assert (kept.work.rows_read, kept.work.rows_streamed) == (n, n)
    formed = evaluate_fit(p, from_groups(np.arange(n) % (few + 1), n), fit)
    assert (formed.work.rows_read, formed.work.rows_streamed) == (2 * n, 0)


def test_sizing_case_reads_each_group_once(monkeypatch):
    """The search's sizing case, 200 groups of about 100 rows on four continuous P-splines.

    A group is below the compact threshold and within one block: its cross-products, scores
    and third-derivative term share one read, made in blocks of whole groups.  The third
    pass needs ``z' M z`` with ``M`` of rank 200 on a design without tables, a second read.
    """
    c = sizing("poisson", "groups")
    p, n = c.problem, c.problem.n
    fit = solve_mode(p, np.zeros(len(p.names)))
    reads = []

    def recorded(dm, rows, centre_hi, centre_lo):
        reads.append(len(rows))
        return augmented_rows(dm, rows, centre_hi, centre_lo)

    monkeypatch.setattr(engine, "augmented_rows", recorded)
    got = evaluate_fit(p, c.nb, fit, heldout=False)
    block = ncv_rows.chunk_rows(p.width)
    assert got.feasible and got.work.factorisations == c.nb.m == 200
    assert (got.work.rows_read, got.work.rows_streamed) == (2 * n, 0)
    # two blocks of whole groups and two of the third pass, where a read a pass a group made 602
    assert max(reads) <= block < n and len(reads) == 4


def test_lookup_design_serves_single_rows_without_dense_rows_at_the_threshold():
    """Single rows on a design with tables: no dense row once they are many enough, two passes before.

    From the threshold on, the leverages, ``Z' v``, the Gram that is ``M`` and the third pass
    each take one pass of the stored form.
    """
    p = searching(book("poisson", n=20_000).problem)
    weighted = int(p.active.sum())
    assert serves_compact(weighted, p.width)
    got = evaluate(p, single_rows(p.n), rho_of(p))
    assert got.feasible
    assert (got.work.rows_read, got.work.rows_streamed) == (0, 4 * weighted)
    assert got.work.rows_multiplied == 0
    small = book("poisson")
    few = evaluate(small.problem, single_rows(small.problem.n), rho_of(small.problem))
    assert (few.work.rows_read, few.work.rows_streamed) == (2 * int(small.problem.active.sum()), 0)


# ── single rows from the tables ───────────────────────────────────────────────


def single_row_neighbourhoods(form, n):
    """Single rows, a group a row in another order, or groups of one to three rows mixed."""
    if form == "rows":
        return single_rows(n)
    rng = np.random.default_rng(23)
    if form == "groups of one":
        return from_groups(rng.permutation(n), n)
    sizes = np.resize([1, 2, 1, 3, 1], n)
    labels = np.repeat(np.arange(n), sizes)[:n]
    return from_groups(rng.permutation(labels), n)


def stacked_rows(problem, nb):
    """Rows of the neighbourhoods of more than one row that the neighbourhood-sized system takes."""
    sizes = nb.sizes()
    if nb.drop_rows is None:
        return 0
    width = problem.width
    size = np.minimum(sizes, width)
    small = 3 * size * size * width + size**3 < width**3
    weighted = np.bincount(nb.owners()[problem.active[nb.drop_rows]], minlength=nb.m)
    return int(sizes[small & (sizes > 1) & (weighted > 0)].sum())


@pytest.mark.parametrize("form", ["rows", "groups of one", "mixed groups"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_single_rows_from_tables_are_the_dense_rows(family, form):
    """Single rows served by their leverages, ``Z' v`` and one Gram: the dense rows' evaluation.

    Rows of zero weight and an offset; among groups of two and three, whose stacks keep their
    dense blocks, the single rows are left out of those blocks.  The same stacks,
    factorisations and solves, and the criterion alone is the one with its gradient.
    """
    b = book(family)
    p = b.problem
    nb = single_row_neighbourhoods(form, p.n)
    rho = rho_of(p)
    dense, compact = both(p, nb, rho)
    assert_same_evaluation(p, compact, dense, condition(p, nb, rho, dense.fit))
    for count in ("stacks", "stacked", "factorisations", "small_factorisations", "solves"):
        assert getattr(compact.work, count) == getattr(dense.work, count), count
    assert compact.work.rows_read == stacked_rows(p, nb)
    with row_work(COMPACT):
        alone = evaluate_fit(rebuilt(p), nb, dense.fit, gradient=False, heldout=False)
    assert alone.gradient is None and alone.value == compact.value


@cache
def dominant(family):
    """A factor whose base level is three rows of weight 1e-3, and whose level ``b`` holds all but six rows.

    Its contrasts carry the base level's variance, about a thousand, while a row of ``b``
    centres to entries of a few times 1e-4: ``x' V x - 2 c' V x + c' V c`` subtracts terms of
    the variance's size to leave the row's leverage.
    """
    rng = np.random.default_rng(17)
    n = 20_000
    f = rng.permutation(np.array(["a"] * 3 + ["c"] * 3 + ["b"] * (n - 6), dtype=object))
    age = rng.integers(18, 60, n).astype(float)
    signal = np.sin(age / 8) + 0.3 * (f == "c")
    if family == "gaussian":
        y = signal + rng.normal(0, 0.5, n)
    else:
        y = rng.poisson(np.exp(signal - 0.5)).astype(float)
    weights = np.where(f == "a", 1e-3, 1.0)
    model = SuperGLM(
        features={"age": Spline(kind="ps", n_knots=6), "f": Categorical(base="a")},
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(pd.DataFrame({"age": age, "f": f}), y, sample_weight=weights)
    problem = problem_of(model, y, weights, None)
    fit = solve_mode(problem, np.zeros(len(problem.names)))
    Z = augmented_rows(problem.dm, np.arange(n), fit.centre_hi, fit.centre_lo)[:, 1:]
    V = fit.slope_inverse
    dense = np.einsum("ij,ij->i", Z @ V, Z)
    plan = lookup_plan(problem.dm)
    terms, columns = len(plan.terms), plan.dense_columns.size
    summed = 1 + terms + plan.n_pairs + columns * (terms + columns + 3)
    size = np.einsum("ij,ij->i", np.abs(Z) @ np.abs(V), np.abs(Z))
    unit = gamma(3 * problem.dm.p + summed + 2 * problem.width + 8) * size
    return problem, fit, dense, unit


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_dominant_level_leverages_are_their_centred_rows(family):
    """Each row's leverage from the tables is its dense centred row's, within the rounding of its centred entries.

    Within ``gamma_K |z|' |V| |z|`` (``K`` the gather's and the dense product's roundings),
    and within the bound ``CentredRows.leverages`` gives the engine.
    """
    problem, fit, dense, unit = dominant(family)
    found = CentredRows(problem.dm, fit.centre_hi, fit.centre_lo).leverages(
        fit.slope_inverse, np.ones(problem.n, dtype=bool)
    )
    assert found is not None
    gap = np.abs(found[0] - dense)
    assert np.all(gap <= unit), f"{np.max(gap / unit):.3g} of the rounding"
    assert np.max(gap) <= found[1]


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_mutation_dominant_level_read_raw_misses_its_centred_rows(family):
    """The tables read raw, a categorical's centre kept apart as constants: the rounding bar fails.

    Pins that the fixture tells the two reads apart: read raw, as the REML pass reads them,
    the dominant level's leverages are more than ten times the bar from their centred rows.
    """
    problem, fit, dense, unit = dominant(family)
    raw = table_leverages(
        lookup_plan(problem.dm),
        fit.slope_inverse,
        fit.centre_hi,
        fit.centre_lo,
        np.ones(problem.n, dtype=bool),
    )
    assert np.max(np.abs(raw - dense) / unit) > 10.0


def single_row_margins(problem, fit):
    """Each weighted row's margin ``1 - W h`` from its dense row, and ``W`` times the tables' bound."""
    eta = linear_predictor(problem.dm, fit.result, problem.offset)
    _, curvature, _ = problem.loss.derivatives(problem.y, problem.weights, eta)
    rows = np.flatnonzero(problem.active)
    Z = augmented_rows(problem.dm, rows, fit.centre_hi, fit.centre_lo)
    solved = Z.copy()
    solved[:, 0] /= fit.sum_w
    solved[:, 1:] = Z[:, 1:] @ fit.slope_inverse
    found = CentredRows(problem.dm, fit.centre_hi, fit.centre_lo).leverages(
        fit.slope_inverse, problem.active
    )
    W = curvature[rows]
    return 1.0 - W * np.einsum("ij,ij->i", solved, Z), W * found[1] + 2.0 * engine.EPS


def forced_level(problem, fit, where):
    """A level the inverse resolves inside the least margin's band, or between two middle margins."""
    margin, band = single_row_margins(problem, fit)
    if where == "beside the least margin":
        at = int(np.argmin(margin))
        return margin[at] - 0.5 * band[at], margin, band
    ordered = np.sort(margin)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle]), margin, band


@pytest.mark.parametrize("where", ["beside the least margin", "between the middle margins"])
def test_single_rows_the_level_may_not_clear_are_decided_on_their_dense_rows(where, monkeypatch):
    """Forced: the level the inverse resolves moved inside one margin's band, or up to half the margins.

    A row whose margin from the tables clears the level by no more than ``W`` times their
    bound, or not at all, is read densely and decided on that row: below the level it is
    factored alone, as the dense rows' route factors it.
    """
    p = book("poisson").problem
    nb = single_rows(p.n)
    fit = solve_mode(p, rho_of(p))
    level, margin, band = forced_level(p, fit, where)
    monkeypatch.setattr(engine._Trial, "unresolved", property(lambda self: level))
    with row_work(DENSE):
        dense = evaluate_fit(rebuilt(p), nb, fit)
    with row_work(COMPACT):
        compact = evaluate_fit(rebuilt(p), nb, fit)
    assert compact.work.rows_read == int(np.count_nonzero(margin - band <= level)) > 0
    below = int(np.count_nonzero(margin <= level))
    assert compact.work.factorisations == dense.work.factorisations == below
    assert compact.work.kept_fits == dense.work.kept_fits == 0
    assert_same_evaluation(p, compact, dense, condition(p, nb, rho_of(p), fit))


def test_mutation_a_band_of_zero_reads_no_row_beside_the_level(monkeypatch):
    """Without the tables' bound, the row beside the level is taken from the tables: the count above fails."""
    p = book("poisson").problem
    nb = single_rows(p.n)
    fit = solve_mode(p, rho_of(p))
    level, _, _ = forced_level(p, fit, "beside the least margin")
    monkeypatch.setattr(engine._Trial, "unresolved", property(lambda self: level))
    original = CentredRows.leverages

    def unbounded(self, inverse, active):
        slopes, _ = original(self, inverse, active)
        return slopes, 0.0

    monkeypatch.setattr(CentredRows, "leverages", unbounded)
    with row_work(COMPACT):
        compact = evaluate_fit(rebuilt(p), nb, fit)
    assert compact.work.rows_read == 0


def test_mutation_single_rows_without_their_gram_miss_the_dense_gradient(monkeypatch):
    """``M``'s Gram left out of the single rows' terms: the gradient is not the dense rows'."""
    b = book("poisson")
    p, nb, rho = b.problem, single_rows(b.problem.n), rho_of(b.problem)

    def without_gram(self):
        values = self._lean_q
        if values is not None:
            self._lean_q = self._lean_d = None
            self.q += self.design.rmatvec(values)

    monkeypatch.setattr(engine._Trial, "flush_lean", without_gram)
    dense, compact = both(p, nb, rho)
    with pytest.raises(AssertionError):
        assert_same_criterion(p, compact, dense, condition(p, nb, rho, dense.fit))
