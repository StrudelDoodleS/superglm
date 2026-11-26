"""The neighbourhood cross-validation engine at fixed smoothing.

Every held-out prediction and the criterion against library refits with the neighbourhood's rows
removed (the same design object, so the same knots and levels); the Poisson step against a Newton
step assembled from the kept rows alone; the gradient against finite differences of independently
re-solved modes; and the engine's rules for zero weights, offsets, clipped predictors and
uncertified modes.

Tolerances.  ``U`` is the unit roundoff.  Where two exact computations are compared the bar is
``64 kappa U`` times the size of the predictor, ``kappa`` the largest 2-norm condition number of a
Jacobi-scaled leave-out Hessian of the fixture, computed here (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., 2002, section 10.1: a Cholesky solve's forward error is a modest
multiple of ``kappa U``).
"""

from dataclasses import dataclass, replace
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import xlogy
from sklearn.model_selection import KFold, RepeatedKFold, TimeSeriesSplit

from superglm import SuperGLM
from superglm._parallel import parallel_config
from superglm.distributions import clip_mu
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.links import stabilize_eta
from superglm.model.base import model_weight_semantics
from superglm.model.reml_setup import collect_reml_groups
from superglm.ncv import engine, from_groups, from_neighbours, from_splitter, single_rows
from superglm.ncv import rows as ncv_rows
from superglm.ncv.engine import (
    LeaveOutRankError,
    ModeFit,
    build_problem,
    evaluate,
    evaluate_fit,
    solve_mode,
)
from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start, linear_predictor

U = 2.0**-53
MODE_BAR = 200.0 * U
AIM, BAR = engine.ncv_mode_aim(), engine.ncv_mode_bar()
RHO = np.array([0.7, -1.2])
FORMS = ("rows", "groups", "cv", "neighbours", "forward", "repeated")
ROUTES = (None, "woodbury", "factor")


# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    problem: engine.NCVProblem
    frame: pd.DataFrame
    labels: np.ndarray


def problem_of(model, y, weights, offset, *, dm=None, **settings):
    dm = model._dm if dm is None else dm
    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
    settings.setdefault("weight_semantics", model_weight_semantics(model))
    settings.setdefault("mode_bar", MODE_BAR)
    return build_problem(
        dm=dm,
        groups=model._groups,
        penalties=penalties,
        distribution=model._distribution,
        link=model._link,
        y=y,
        weights=weights,
        offset=offset,
        **settings,
    )


def group_labels(rng, n):
    """About ``n / 25`` groups of unequal size, 1 row upwards, scattered over the rows."""
    edges = np.sort(rng.choice(np.arange(10, n - 10), n // 25 - 1, replace=False))
    return rng.permutation(np.searchsorted(edges, np.arange(n), side="right"))


@cache
def case(family, n, exposure="offset", zero_weights=False) -> Case:
    """Two P-splines, a numeric column far from zero and a factor, with weights and an offset.

    ``exposure="weight"`` puts a Poisson exposure in the weights of a rate; ``zero_weights``
    gives every eleventh row a weight of zero.
    """
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(-2, 2, n),
            "z": rng.normal(2000.0, 5.0, n),
            "f": rng.choice(list("abcd"), n, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    exposed = rng.uniform(0.2, 1.5, n)
    weights = rng.choice([0.5, 1.0, 2.0, 3.0], n)
    signal = (
        np.sin(6 * frame.x1) + 0.3 * frame.x2**2 + 0.03 * (frame.z - 2000) + 0.4 * (frame.f == "b")
    ).to_numpy()
    if family == "gaussian":
        offset = 0.5 * np.log(exposed)
        y = signal + offset + rng.normal(0, 0.5, n) / np.sqrt(weights)
    elif exposure == "offset":
        offset = np.log(exposed)
        y = rng.poisson(np.exp(signal - 0.6 + offset)).astype(float)
    else:
        offset = np.zeros(n)
        weights = exposed
        y = rng.poisson(exposed * np.exp(signal - 0.6)) / exposed
    if zero_weights:
        weights = np.where(np.arange(n) % 11 == 3, 0.0, weights)
    model = SuperGLM(
        features={
            "x1": Spline(kind="ps", n_knots=8),
            "x2": Spline(kind="ps", n_knots=8),
            "z": Numeric(),
            "f": Categorical(),
        },
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, y, sample_weight=weights, offset=offset)
    return Case(problem_of(model, y, weights, offset), frame, group_labels(rng, n))


def neighbourhoods(form, c: Case):
    n = c.problem.n
    if form == "rows":
        return single_rows(n)
    if form == "groups":
        return from_groups(c.labels, n)
    if form == "neighbours":
        return from_neighbours([np.arange(max(i - 3, 0), min(i + 4, n)) for i in range(n)], n)
    cv = {
        "cv": KFold(5, shuffle=True, random_state=1),
        "forward": TimeSeriesSplit(5),
        "repeated": RepeatedKFold(n_splits=3, n_repeats=2, random_state=2),
    }[form]
    return from_splitter(cv, c.frame, c.problem.y)


# ── independent references ────────────────────────────────────────────────────


def library_fit(problem, lambdas, rows=None, start=None):
    """The library's fixed-smoothing fit on ``rows`` (every row by default), certified at ``MODE_BAR``."""
    keep = np.arange(problem.n) if rows is None else rows
    warm = {}
    if start is not None:
        warm = {
            "beta_init": start.beta,
            "intercept_init": float(start.intercept),
            "_centred_init": centred_warm_start(start),
        }
    result = fit_irls_direct(
        X=problem.dm if rows is None else problem.dm.row_subset(keep),
        y=problem.y[keep],
        weights=problem.weights[keep],
        family=problem.distribution,
        link=problem.link,
        groups=problem.groups,
        lambda2=lambdas,
        offset=problem.offset[keep],
        reml_penalties=problem.penalties,
        max_iter=200,
        tol=1e-12,
        convergence="mode_score",
        _mode_bar=MODE_BAR,
        compute_rank_info=False,
        _compensate_centred_intercept=False,
        separation="record",
        weight_semantics=problem.weight_semantics,
        **warm,
    )[0]
    assert result.converged
    return result


def mean_and_deviance(problem, rows, eta):
    """The mean and weighted unit deviance of ``rows`` at ``eta``, from the families' textbook forms.

    A row of zero weight has a deviance of zero whatever its response.
    """
    y, w = problem.y[rows], problem.weights[rows]
    y = np.where(w > 0.0, y, 1.0)
    if problem.loss.log_mean:
        mean = np.exp(eta)
        return mean, 2.0 * w * (xlogy(y, y / mean) - (y - mean))
    return eta, w * (y - eta) ** 2


def refit_pairs(problem, nb, rho, every=1):
    """Held-out results from refits without each neighbourhood's rows, one row per scored pair."""
    lambdas = problem.lambdas(rho)
    full = library_fit(problem, lambdas)
    parts = []
    for k in range(0, nb.m, every):
        removed = nb.drop(k)
        pos = nb.scored_positions(k)
        scored = removed if pos is None else removed[pos]
        kept = np.setdiff1d(np.arange(problem.n), removed)
        sub = library_fit(problem, lambdas, kept, start=full)
        eta = linear_predictor(problem.dm.row_subset(scored), sub, problem.offset[scored])
        mean, deviance = mean_and_deviance(problem, scored, eta)
        parts.append(
            pd.DataFrame(
                {
                    "neighbourhood": k,
                    "row": scored,
                    "predictor": eta,
                    "mean": mean,
                    "deviance": deviance,
                }
            )
        )
    return pd.concat(parts).set_index(["neighbourhood", "row"])


@cache
def refits(family, n, form, exposure="offset", every=1):
    c = case(family, n, exposure)
    return refit_pairs(c.problem, neighbourhoods(form, c), RHO, every)


def pairs_of(evaluation):
    """An evaluation's held-out pairs keyed by (neighbourhood, row); the key must be unique."""
    held = evaluation.heldout
    table = pd.DataFrame(
        {
            "neighbourhood": held.neighbourhood,
            "row": held.row,
            "predictor": held.predictor,
            "mean": held.mean,
            "deviance": held.deviance,
        }
    ).set_index(["neighbourhood", "row"])
    assert table.index.is_unique
    return table


class KeptRows:
    """Each neighbourhood's kept-row system, assembled in coordinates of the test's own.

    Rows are centred about the plain working-weight mean of every row, an exact
    reparametrisation that owes nothing to the engine's centre, and each Hessian and score is
    summed from its kept rows alone.  The full fit is the library's at ``rho``, or the one a
    caller hands over as ``at``: its predictor on every row and its slopes.
    """

    def __init__(self, problem, rho, at=None):
        lambdas = problem.lambdas(rho)
        if at is None:
            full = library_fit(problem, lambdas)
            at = linear_predictor(problem.dm, full, problem.offset), full.beta
        self.eta, slopes = at
        mean, _ = mean_and_deviance(problem, np.arange(problem.n), self.eta)
        w, dm = problem.weights, problem.dm
        self.curvature = w * mean if problem.loss.log_mean else w
        self.pull = w * (problem.y - mean)
        X = dm.toarray()
        self.Z = np.column_stack([np.ones(dm.n), X - self.curvature @ X / self.curvature.sum()])
        self.S = np.zeros((dm.p + 1, dm.p + 1))
        self.S[1:, 1:] = build_penalty_matrix(
            list(dm.group_matrices), problem.groups, lambdas, dm.p, reml_penalties=problem.penalties
        )
        self.penalty_score = self.S @ np.concatenate(([0.0], slopes))
        self.full_score = self.Z.T @ self.pull - self.penalty_score

    def solver(self, removed):
        """``(solve, score, condition)`` of the fit without ``removed``, Jacobi-scaled."""
        kept = np.ones(len(self.eta), dtype=bool)
        kept[removed] = False
        Zk = self.Z[kept]
        hessian = Zk.T @ (self.curvature[kept, None] * Zk) + self.S
        scale = 1.0 / np.sqrt(np.diag(hessian))
        scaled = hessian * scale[:, None] * scale[None, :]

        def solve(v):
            return scale * np.linalg.solve(scaled, scale * v)

        return solve, Zk.T @ self.pull[kept] - self.penalty_score, float(np.linalg.cond(scaled))

    def one_step(self, nb):
        """Per scored pair: the Newton-step predictor, and what the full fit's residual score moves it by.

        Also the largest condition number met.
        """
        steps, residual, worst = [], [], 0.0
        for k in range(nb.m):
            removed = nb.drop(k)
            pos = nb.scored_positions(k)
            scored = removed if pos is None else removed[pos]
            solve, score, condition = self.solver(removed)
            steps.append(self.eta[scored] + self.Z[scored] @ solve(score))
            residual.append(np.abs(self.Z[scored] @ solve(self.full_score)))
            worst = max(worst, condition)
        return np.concatenate(steps), np.concatenate(residual), worst


@cache
def kept_rows(family, n, exposure="offset"):
    return KeptRows(case(family, n, exposure).problem, RHO)


@cache
def one_step(family, n, form, exposure="offset"):
    c = case(family, n, exposure)
    return kept_rows(family, n, exposure).one_step(neighbourhoods(form, c))


def predictor_slack(kappa, predictor):
    return 64.0 * kappa * U * np.maximum(1.0, np.abs(predictor))


def deviance_slope(problem, held):
    """``|D'|`` per scored pair, from the families' textbook forms."""
    y, w = problem.y[held.row], problem.weights[held.row]
    return 2.0 * w * np.abs((held.mean if problem.loss.log_mean else held.predictor) - y)


def assert_pairs_equal(problem, got, want, slack):
    """Gaussian pairs of ``got`` are ``want``'s, joined on (neighbourhood, row), within ``slack`` in the predictor.

    The mean is the predictor, and ``D = w (y - eta)^2`` moves by at most
    ``(2 w |y - eta| + w slack) slack`` plus the rounding of its two evaluations.
    """
    joined = got.join(want, how="outer", rsuffix="_ref")
    assert len(joined) == len(got) == len(want)
    assert not joined.isna().to_numpy().any()
    rows = joined.index.get_level_values("row").to_numpy()
    w, y = problem.weights[rows], problem.y[rows]
    gap = np.abs(joined.predictor - joined.predictor_ref).to_numpy()
    assert np.all(gap <= slack), f"predictor gap {gap.max():.3e} against {np.min(slack):.3e}"
    assert np.all(np.abs(joined["mean"] - joined.mean_ref).to_numpy() <= slack)
    slope = 2.0 * w * np.abs(joined.mean_ref.to_numpy() - y)
    deviance_slack = (slope + w * slack) * slack + 64.0 * U * joined.deviance_ref.to_numpy()
    assert np.all(np.abs(joined.deviance - joined.deviance_ref).to_numpy() <= deviance_slack)
    assert abs(got.deviance.sum() - want.deviance.sum()) <= deviance_slack.sum()


# ── rows 2 and 6: Gaussian leave-out fits are exact ───────────────────────────


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
@pytest.mark.parametrize("form", FORMS)
def test_gaussian_heldout_equals_refits(form, route):
    """Weighted Gaussian with an offset: each pair's prediction and the criterion are the refit's."""
    c = case("gaussian", 300)
    nb = neighbourhoods(form, c)
    got = evaluate(c.problem, nb, RHO, gradient=False, route=route)
    assert got.feasible
    want = refits("gaussian", 300, form)
    _, _, kappa = one_step("gaussian", 300, form)
    assert_pairs_equal(
        c.problem, pairs_of(got), want, predictor_slack(kappa, want.predictor.to_numpy())
    )
    assert got.value == pytest.approx(want.deviance.sum(), rel=64 * kappa * U)


def test_each_scored_occurrence_has_its_own_entry():
    """Repeated K-fold scores every row twice and a forward splitter scores only its test rows."""
    c = case("gaussian", 300)
    n = c.problem.n
    repeated = pairs_of(evaluate(c.problem, neighbourhoods("repeated", c), RHO, gradient=False))
    counts = repeated.groupby(level="row").size()
    assert len(repeated) == 2 * n and counts.eq(2).all() and len(counts) == n
    twice = repeated.predictor.groupby(level="row").agg(["min", "max"])
    assert (twice["max"] > twice["min"]).all()

    nb = neighbourhoods("forward", c)
    forward = pairs_of(evaluate(c.problem, nb, RHO, gradient=False))
    tests = [test for _, test in TimeSeriesSplit(5).split(c.frame)]
    assert len(forward) == sum(map(len, tests)) < int(nb.sizes().sum())
    for k, test in enumerate(tests):
        assert forward.loc[k].index.tolist() == test.tolist()
        assert set(test.tolist()) <= set(nb.drop(k).tolist())
    assert all(len(test) < len(nb.drop(k)) for k, test in enumerate(tests[:-1]))


@pytest.mark.parametrize("form", ["forward", "repeated"])
def test_partial_and_repeated_scoring_at_other_smoothings(form):
    """A split that scores part of what it removes, and rows scored twice, at a light and a heavy smoothing."""
    c = case("gaussian", 300)
    nb = neighbourhoods(form, c)
    for rho in (np.array([-4.0, 3.0]), np.array([6.0, -3.0])):
        want = refit_pairs(c.problem, nb, rho)
        _, _, kappa = KeptRows(c.problem, rho).one_step(nb)
        slack = predictor_slack(kappa, want.predictor.to_numpy())
        for route in ROUTES:
            got = evaluate(c.problem, nb, rho, gradient=False, route=route)
            assert_pairs_equal(c.problem, pairs_of(got), want, slack)


# ── row 3: Poisson leave-out fits are one Newton step ─────────────────────────


@pytest.mark.parametrize("exposure", ["offset", "weight"])
@pytest.mark.parametrize("form", FORMS)
def test_poisson_heldout_is_the_one_step_fit(form, exposure):
    """Each held-out predictor is the Newton step of the kept rows' own score and Hessian.

    The engine reads the removed rows' score, the reference the kept rows' score less the
    penalty's: the two differ by the full fit's residual score, whose step is in the slack.
    """
    c = case("poisson", 600, exposure)
    nb = neighbourhoods(form, c)
    want, residual, kappa = one_step("poisson", 600, form, exposure)
    for route in ROUTES:
        got = evaluate(c.problem, nb, RHO, gradient=False, route=route)
        assert got.feasible
        slack = residual + predictor_slack(kappa, want)
        gap = np.abs(got.heldout.predictor - want)
        assert np.all(gap <= slack), f"{route}: gap {gap.max():.3e} against {slack.min():.3e}"
        np.testing.assert_allclose(got.heldout.mean, np.exp(want), rtol=2.0 * slack.max())


@pytest.mark.parametrize("exposure", ["offset", "weight"])
@pytest.mark.parametrize(
    ("form", "every"), [("rows", 5), ("groups", 1), ("cv", 1), ("neighbours", 5), ("forward", 1)]
)
def test_poisson_criterion_within_one_percent_of_refits(form, every, exposure):
    """The acceptance table's bar on this design: held-out deviance within 1% of converged refits.

    Single rows and windows are refitted for every fifth neighbourhood and compared on those
    pairs; the other forms on every pair.
    """
    c = case("poisson", 600, exposure)
    got = pairs_of(evaluate(c.problem, neighbourhoods(form, c), RHO, gradient=False))
    want = refits("poisson", 600, form, exposure, every)
    assert want.index.isin(got.index).all()
    assert got.deviance[want.index].sum() == pytest.approx(want.deviance.sum(), rel=0.01)


# ── rows 2 and 3: the gradient ────────────────────────────────────────────────


def richardson(values, h):
    """The derivative at 0 from values at ``s h``, ``s`` in ``(-4, -2, -1, 1, 2, 4)``, and its truncation.

    Two fourth-order central differences, at ``h`` and ``2 h``, whose errors are ``c h^4`` and
    ``16 c h^4`` to leading order: Richardson's extrapolation removes that term, and its own
    correction, a fifteenth of their difference, is the truncation estimate.
    """
    fine = (values[-2] - 8 * values[-1] + 8 * values[1] - values[2]) / (12 * h)
    coarse = (values[-4] - 8 * values[-2] + 8 * values[2] - values[4]) / (24 * h)
    return fine + (fine - coarse) / 15, abs(fine - coarse) / 15


# The extrapolated difference's weights on the six values sum to 1.65 / h in magnitude.
STENCIL_ROUNDING = (16 * 18 / 12 + 18 / 24) / 15


def criterion_resolution(problem, evaluation, kappa):
    """How finely one criterion value is resolved: ``64 kappa U sum |D'| max(1, |eta|)``."""
    held = evaluation.heldout
    return (
        64.0
        * kappa
        * U
        * float(np.sum(deviance_slope(problem, held) * np.maximum(1.0, np.abs(held.predictor))))
    )


def assert_same_criterion(problem, got, want, kappa):
    """Two evaluations of one criterion agree in value and in every slope within its resolution.

    A slope per unit ``log lambda`` has the criterion's units and is a difference of larger
    terms, so it is held to the criterion's absolute resolution, not to a relative bar.
    """
    resolution = criterion_resolution(problem, want, kappa)
    assert got.feasible and want.feasible
    assert abs(got.value - want.value) <= resolution
    if want.gradient is not None:
        np.testing.assert_allclose(got.gradient, want.gradient, rtol=0.0, atol=resolution)


def assert_gradient_matches_differences(problem, nb, rho, kappa, h=0.05, route=None):
    """Per component and along one coupled direction, against cold re-solves at each point."""
    base = evaluate(problem, nb, rho, route=route)
    assert base.feasible
    rounding = STENCIL_ROUNDING * criterion_resolution(problem, base, kappa) / h
    m = len(rho)
    coupled = np.array([(-0.7) ** j for j in range(m)])
    directions = [*np.eye(m), coupled / np.linalg.norm(coupled)]
    for j, direction in enumerate(directions):
        trials = {
            s: evaluate(problem, nb, rho + s * h * direction, gradient=False, route=route)
            for s in (-4, -2, -1, 1, 2, 4)
        }
        assert all(trial.feasible and trial.work.inner_solves == 1 for trial in trials.values())
        difference, truncation = richardson({s: t.value for s, t in trials.items()}, h)
        analytic = float(base.gradient @ direction)
        bar = truncation + rounding
        assert bar <= 1e-3 * abs(difference), "the difference does not resolve this component"
        assert abs(analytic - difference) <= bar, (j, analytic, difference, bar)
        if j < m:
            # the mode moved with the smoothing, by d beta / d rho_j = -lambda_j H_c^-1 S_j beta
            columns, S = problem.blocks[j]
            moved = (trials[1].fit.result.beta - trials[-1].fit.result.beta) / (2 * h)
            predicted = (
                -np.exp(rho[j])
                * base.fit.slope_inverse[:, columns]
                @ (S @ base.fit.result.beta[columns])
            )
            assert np.linalg.norm(predicted) > 0
            assert np.linalg.norm(moved - predicted) <= h * np.linalg.norm(predicted)
    return base


@pytest.mark.parametrize("form", ["rows", "groups", "cv", "neighbours", "forward"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_gradient_matches_finite_differences(family, form):
    """The analytic gradient at a smoothing away from the optimum, modes re-solved at every point."""
    c = case(family, 300)
    nb = neighbourhoods(form, c)
    _, _, kappa = one_step(family, 300, form)
    assert_gradient_matches_differences(c.problem, nb, RHO, kappa)


@pytest.mark.parametrize("route", ["woodbury", "factor"])
@pytest.mark.parametrize("form", ["groups", "forward"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_forced_routes_give_the_same_gradient(family, form, route):
    """Each solve route alone reproduces the cost rule's criterion and gradient.

    Groups of mixed size, which the cost rule splits between the routes, and a forward
    splitter, whose scored rows are part of the removed ones.
    """
    c = case(family, 300)
    _, _, kappa = one_step(family, 300, form)
    nb = neighbourhoods(form, c)
    assert_same_criterion(
        c.problem, evaluate(c.problem, nb, RHO, route=route), evaluate(c.problem, nb, RHO), kappa
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_every_term_type(family):
    """A double penalty, a tensor and a spline-by-factor term: refits, and each penalty's slope."""
    rng = np.random.default_rng(61)
    n = 500
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(-2, 2, n),
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
    problem = problem_of(model, y, weights, None)
    assert len(problem.names) >= 6
    rho = np.linspace(-1.0, 1.5, len(problem.names))
    nb = from_groups(rng.integers(0, 20, n), n)
    want, residual, kappa = KeptRows(problem, rho).one_step(nb)
    base = assert_gradient_matches_differences(problem, nb, rho, kappa)
    slack = residual + predictor_slack(kappa, want)
    assert np.all(np.abs(base.heldout.predictor - want) <= slack)
    refitted = refit_pairs(problem, nb, rho)
    if family == "gaussian":
        assert_pairs_equal(problem, pairs_of(base), refitted, slack)
    else:
        assert base.value == pytest.approx(refitted.deviance.sum(), rel=0.01)


# ── the review's reproducers ──────────────────────────────────────────────────


def translated(anchor):
    """A spline and a numeric column of small integers moved to ``anchor``; twelve groups."""
    rng = np.random.default_rng(127)
    n = 120
    t = np.arange(n)
    small = (t % 5).astype(float) + (t % 9 == 0)
    frame = pd.DataFrame({"t": rng.uniform(size=n), "z": small + anchor})
    y = np.sin(6 * frame.t.to_numpy()) + small / 2 + rng.normal(0.0, 0.2, n)
    model = SuperGLM(
        features={"t": Spline(kind="ps", n_knots=6), "z": Numeric()},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, y)
    return problem_of(model, y, np.ones(n), None), from_groups(t % 12, n)


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
def test_column_translation_leaves_heldout_predictions_unchanged(route):
    """Adding 2**52 to a column of small integers moves no held-out prediction.

    The column's mean is then not a float64; rows centred about it alone are off by its rounding.
    """
    rho = np.zeros(1)
    problem, nb = translated(0.0)
    moved_problem, _ = translated(2.0**52)
    _, _, kappa = KeptRows(problem, rho).one_step(nb)
    for form in (nb, single_rows(problem.n)):
        here = evaluate(problem, form, rho, route=route)
        there = evaluate(moved_problem, form, rho, route=route)
        slack = predictor_slack(kappa, here.heldout.predictor)
        assert np.all(np.abs(there.heldout.predictor - here.heldout.predictor) <= slack)
        assert_same_criterion(problem, there, here, kappa)
    refitted = refit_pairs(moved_problem, nb, rho)
    there = pairs_of(evaluate(moved_problem, nb, rho, gradient=False, route=route))
    assert_pairs_equal(
        moved_problem, there, refitted, predictor_slack(kappa, refitted.predictor.to_numpy())
    )


def test_one_pirls_iteration_is_not_a_feasible_trial():
    """A mode the solver did not certify gives no criterion, gradient or held-out pairs."""
    c = case("poisson", 300)
    p = c.problem
    hurried = build_problem(
        dm=p.dm,
        groups=p.groups,
        penalties=p.penalties,
        distribution=p.distribution,
        link=p.link,
        y=p.y,
        weights=p.weights,
        offset=p.offset,
        weight_semantics=p.weight_semantics,
        mode_bar=MODE_BAR,
        max_pirls_iter=1,
    )
    nb = neighbourhoods("groups", c)
    got = evaluate(hurried, nb, RHO)
    assert not got.feasible and got.infeasible_reason == "uncertified_mode"
    assert np.isnan(got.value) and got.gradient is None and got.heldout is None and got.fit is None
    uncertified = solve_mode(hurried, RHO)
    assert not uncertified.certified and uncertified.result.termination_reason == "max_iter"
    assert not evaluate_fit(hurried, nb, uncertified).feasible
    assert evaluate(p, nb, RHO).feasible


def test_uncertified_warm_solve_is_retried_from_a_cold_start(monkeypatch):
    """A warm-started solve that ends uncertified is solved once more from the solver's own start."""
    c = case("poisson", 300)
    nb = neighbourhoods("cv", c)
    warm = evaluate(c.problem, nb, RHO + 3.0, gradient=False).fit
    starts = []

    def one_step_when_warm(**arguments):
        starts.append("beta_init" in arguments)
        if "beta_init" in arguments:
            arguments["max_iter"] = 1
        return fit_irls_direct(**arguments)

    monkeypatch.setattr(engine, "fit_irls_direct", one_step_when_warm)
    got = evaluate(c.problem, nb, RHO, warm=warm, gradient=False)
    assert starts == [True, False]
    assert got.feasible and got.work.inner_solves == 2
    monkeypatch.undo()
    _, _, kappa = one_step("poisson", 300, "cv")
    assert_same_criterion(c.problem, got, evaluate(c.problem, nb, RHO, gradient=False), kappa)


def solver_ending(stop, log):
    """The solver, each solve reported as ended the way ``stop(arguments)`` names.

    ``"converged"`` is a solve certified at its bar and any other reason one that ended short
    of it; ``None`` leaves the solver's own report.  The solve is the solver's own, so the
    state it returns is the mode whatever is reported of it.  ``log`` takes every solve's
    bar, the coefficients it was started from (``None`` for the solver's own start) and
    returned, its iterations and whether it is reported certified.
    """

    def solve(**arguments):
        result, *rest = fit_irls_direct(**arguments)
        reason = stop(arguments)
        if reason is not None:
            result = replace(result, converged=reason == "converged", termination_reason=reason)
        log.append(
            SimpleNamespace(
                bar=arguments["_mode_bar"],
                start=arguments.get("beta_init"),
                beta=result.beta,
                iterations=int(result.n_iter),
                certified=bool(result.converged),
            )
        )
        return (result, *rest)

    return solve


def solver_short(when, log, stop="score_stagnated"):
    """The solver, reporting each solve whose arguments ``when`` accepts as ended short of its bar.

    By stagnation, as a solve at the limit of float64 accuracy ends, or by ``stop``.
    """
    return solver_ending(lambda arguments: stop if when(arguments) else None, log)


def at_the_aim(arguments) -> bool:
    return arguments["_mode_bar"] == AIM


def always(arguments) -> bool:
    return True


def test_both_bars_follow_the_tolerance_and_neither_goes_below_a_hundred_eps():
    floor = 200.0 * U
    assert engine.ncv_mode_aim(1e-6) == pytest.approx(1e-9, rel=4 * U) == AIM
    assert engine.ncv_mode_bar(1e-6) == pytest.approx(1e-8, rel=4 * U) == BAR
    assert engine.ncv_mode_aim(1e-11) == floor < engine.ncv_mode_bar(1e-11)
    assert engine.ncv_mode_aim(1e-13) == engine.ncv_mode_bar(1e-13) == floor
    default = case("poisson", 300).problem
    assert default.mode_aim == default.mode_bar == MODE_BAR, "an aim above the bar is the bar"


def test_solve_short_of_its_aim_is_continued_from_where_it_stopped(monkeypatch):
    """Every solve to the aim is made to end short of it, as on data that cannot resolve it.

    The fit is the stopped solve continued to the acceptance bar, not a second solve from
    the solver's own start.  That is one iteration: the stopped state lies within the bar
    already, and the solver takes a step before it reads its certificate.
    """
    problem = replace(case("poisson", 300).problem, mode_aim=AIM, mode_bar=BAR)
    log: list = []
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(at_the_aim, log))
    work = engine.WorkCounts()
    fit = solve_mode(problem, RHO, work=work)
    stopped, continued = log
    assert (work.inner_solves, work.fallback_solves) == (2, 1)
    assert continued.iterations == 1 < stopped.iterations
    assert work.pirls_iterations == stopped.iterations + 1
    assert (stopped.bar, stopped.start, stopped.certified) == (AIM, None, False)
    assert continued.bar == BAR and continued.start is stopped.beta
    assert fit.certified and fit.bar == BAR


def test_fit_carries_its_bar_to_the_solve_it_warms(monkeypatch):
    """A solve warmed by a fit accepted at the acceptance bar runs to that bar and not to the aim.

    One solve, where trying for the aim again would be two.  A fit certified at the aim
    warms a solve that runs to the aim.
    """
    problem = replace(case("poisson", 300).problem, mode_aim=AIM, mode_bar=BAR)
    reached = solve_mode(problem, RHO)
    assert reached.certified and reached.bar == AIM
    log: list = []
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(at_the_aim, log))
    accepted = solve_mode(problem, RHO)
    assert accepted.certified and accepted.bar == BAR
    for warm, bar in ((accepted, BAR), (reached, AIM)):
        del log[:]
        work = engine.WorkCounts()
        fit = solve_mode(problem, RHO + 0.5, warm=warm, work=work)
        assert (log[0].bar, fit.bar) == (bar, BAR) and log[0].start is warm.result.beta
        assert work.inner_solves == len(log) == (1 if bar == BAR else 2)


def test_solve_short_of_both_bars_is_solved_once_from_the_solvers_own_start(monkeypatch):
    """Every solve given a start is made to end on its iteration limit: the third is the solver's own.

    From the solver's own start there are two solves and no third, and a mode that is neither
    certified nor at the limit of float64 gives no criterion.
    """
    c = case("poisson", 300)
    problem = replace(c.problem, mode_aim=AIM, mode_bar=BAR)
    nb = neighbourhoods("cv", c)
    warm = evaluate(problem, nb, RHO + 3.0, gradient=False).fit
    log: list = []
    started = solver_short(lambda arguments: "beta_init" in arguments, log, stop="max_iter")
    monkeypatch.setattr(engine, "fit_irls_direct", started)
    got = evaluate(problem, nb, RHO, warm=warm, gradient=False)
    assert [(call.bar, call.start is not None) for call in log] == [
        (AIM, True),
        (BAR, True),
        (BAR, False),
    ]
    assert got.feasible and got.fit.bar == BAR
    assert (got.work.inner_solves, got.work.fallback_solves) == (3, 2)
    del log[:]
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(always, log, stop="max_iter"))
    lost = evaluate(problem, nb, RHO, gradient=False)
    assert [(call.bar, call.start is not None) for call in log] == [(AIM, False), (BAR, True)]
    assert not lost.feasible and lost.infeasible_reason == "uncertified_mode" and lost.fit is None
    assert (lost.work.inner_solves, lost.work.fallback_solves) == (2, 1)
    assert lost.work.stagnated_fits == 0
    monkeypatch.undo()
    _, _, kappa = one_step("poisson", 300, "cv")
    assert_same_criterion(problem, got, evaluate(problem, nb, RHO, gradient=False), kappa)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_solve_ended_at_the_limit_of_float64_is_the_mode_of_its_trial(monkeypatch, family):
    """Every solve is made to end stagnated, as on data whose limit lies above the acceptance bar.

    The fit is the first solve continued once, and no solve from another start follows.  It
    stands for the mode: the trial has the criterion, the slopes and the held-out pairs that
    the same state gives when it is certified, to the last bit, and the evaluation is counted
    as made at a stagnated fit.  A solve it warms runs to the acceptance bar at once.
    """
    c = case(family, 300)
    problem = replace(c.problem, mode_aim=AIM, mode_bar=BAR)
    nb = neighbourhoods("cv", c)
    log: list = []
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(always, log))
    got = evaluate(problem, nb, RHO)
    assert [(call.bar, call.start is not None) for call in log] == [(AIM, False), (BAR, True)]
    assert log[1].start is log[0].beta
    fit = got.fit
    assert got.feasible and fit.stagnated and fit.usable and not fit.certified
    assert fit.bar == BAR
    assert (got.work.inner_solves, got.work.fallback_solves, got.work.stagnated_fits) == (2, 1, 1)
    del log[:]
    onward = evaluate(problem, nb, RHO + 0.5, warm=fit, gradient=False)
    assert [(call.bar, call.start is fit.result.beta) for call in log] == [(BAR, True)]
    assert onward.feasible and onward.fit.stagnated and onward.work.stagnated_fits == 1
    monkeypatch.undo()
    certified = replace(fit.result, converged=True, termination_reason="converged")
    same = evaluate_fit(problem, nb, replace(fit, result=certified))
    assert same.feasible and same.work.stagnated_fits == 0
    assert got.value == same.value and np.array_equal(got.gradient, same.gradient)
    pd.testing.assert_frame_equal(pairs_of(got), pairs_of(same), check_exact=True)
    assert evaluate(problem, nb, RHO).work.stagnated_fits == 0
    if family == "gaussian":
        want = refits("gaussian", 300, "cv")
        _, _, kappa = one_step("gaussian", 300, "cv")
        slack = predictor_slack(kappa, want.predictor.to_numpy())
        assert_pairs_equal(problem, pairs_of(got), want, slack)


@pytest.mark.parametrize(
    ("stop", "usable"),
    [
        ("converged", True),
        ("score_stagnated", True),
        ("max_iter", False),
        ("step_rejected", False),
        ("nonfinite_deviance", False),
    ],
)
def test_only_the_solvers_stagnation_makes_an_uncertified_fit_usable(stop, usable):
    """A fit short of its certificate stands for the mode only where the solver ended it as stagnated."""
    c = case("poisson", 300)
    nb = neighbourhoods("groups", c)
    solved = solve_mode(c.problem, RHO)
    result = replace(solved.result, converged=stop == "converged", termination_reason=stop)
    fit = replace(solved, result=result)
    assert (fit.certified, fit.stagnated) == (stop == "converged", stop == "score_stagnated")
    got = evaluate_fit(c.problem, nb, fit)
    assert fit.usable == got.feasible == usable
    assert got.infeasible_reason == (None if usable else "uncertified_mode")
    assert (got.fit is fit) == usable and got.work.stagnated_fits == int(fit.stagnated)


@pytest.mark.parametrize("form", ["rows", "groups", "cv"])
def test_zero_weight_rows_with_extreme_responses_contribute_nothing(form):
    """Rows of zero weight, holding a response of 1e200, change nothing and are still predicted.

    The criterion, gradient and every weighted row's held-out prediction equal those of the
    design with the zero-weight rows taken out, and every pair, the zero-weight rows' too, is
    its refit's.
    """
    c = case("gaussian", 300, zero_weights=True)
    p = c.problem
    idle = ~p.active
    assert idle.sum() > 20
    y = np.where(idle, 1e200, p.y)
    kept = np.flatnonzero(p.active)
    common = dict(
        groups=p.groups,
        penalties=p.penalties,
        distribution=p.distribution,
        link=p.link,
        weight_semantics=p.weight_semantics,
        mode_bar=MODE_BAR,
    )
    whole = build_problem(dm=p.dm, y=y, weights=p.weights, offset=p.offset, **common)
    reduced = build_problem(
        dm=p.dm.row_subset(kept),
        y=y[kept],
        weights=p.weights[kept],
        offset=p.offset[kept],
        **common,
    )
    labels = {"rows": np.arange(p.n), "groups": c.labels, "cv": np.arange(p.n) % 5}[form]
    nb = single_rows(p.n) if form == "rows" else from_groups(labels, p.n)
    nb_reduced = single_rows(len(kept)) if form == "rows" else from_groups(labels[kept], len(kept))
    assert form == "rows" or nb.m == nb_reduced.m
    _, _, kappa = KeptRows(reduced, RHO).one_step(nb_reduced)
    refitted = refit_pairs(whole, nb, RHO)
    for route in ROUTES:
        got = evaluate(whole, nb, RHO, route=route)
        want = evaluate(reduced, nb_reduced, RHO, route=route)
        assert_same_criterion(reduced, got, want, kappa)
        slack = predictor_slack(kappa, refitted.predictor.to_numpy())
        assert_pairs_equal(whole, pairs_of(got), refitted, slack)
        scored_idle = idle[got.heldout.row]
        assert scored_idle.sum() == idle.sum()
        assert np.all(got.heldout.deviance[scored_idle] == 0.0)
        assert np.all(np.isfinite(got.heldout.mean))
        gap = np.abs(got.heldout.predictor[~scored_idle] - want.heldout.predictor)
        assert np.all(gap <= predictor_slack(kappa, want.heldout.predictor))


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("form", ["rows", "groups"])
def test_integer_weights_equal_replicated_rows(form, semantics):
    """A row of weight 3 leaves with all three of its copies: the criterion of the copied-out frame."""
    c = case("poisson", 300)
    p = c.problem
    counts = np.random.default_rng(3).integers(1, 4, p.n)
    copies = np.repeat(np.arange(p.n), counts)
    common = dict(
        groups=p.groups,
        penalties=p.penalties,
        distribution=p.distribution,
        link=p.link,
        weight_semantics=semantics,
        mode_bar=MODE_BAR,
    )
    weighted = build_problem(
        dm=p.dm, y=p.y, weights=counts.astype(float), offset=p.offset, **common
    )
    copied = build_problem(
        dm=p.dm.row_subset(copies),
        y=p.y[copies],
        weights=np.ones(len(copies)),
        offset=p.offset[copies],
        **common,
    )
    labels = np.arange(p.n) if form == "rows" else c.labels
    nb = single_rows(p.n) if form == "rows" else from_groups(labels, p.n)
    nb_copied = from_groups(labels[copies], len(copies))
    _, _, kappa = KeptRows(weighted, RHO).one_step(nb)
    by_weight = evaluate(weighted, nb, RHO)
    by_copies = evaluate(copied, nb_copied, RHO)
    assert_same_criterion(weighted, by_copies, by_weight, kappa)


def test_scored_predictor_at_the_log_links_clip_is_an_infeasible_trial():
    """A weighted scored row whose predictor is clipped makes the trial infeasible; an unweighted one does not."""
    c = case("poisson", 300)
    p = c.problem
    nb = neighbourhoods("groups", c)
    common = dict(
        dm=p.dm,
        groups=p.groups,
        penalties=p.penalties,
        distribution=p.distribution,
        link=p.link,
        weight_semantics=p.weight_semantics,
        mode_bar=MODE_BAR,
    )
    offset = p.offset.copy()
    offset[17] = -200.0
    y = p.y.copy()
    y[17] = 0.0
    weighted = build_problem(y=y, weights=p.weights, offset=offset, **common)
    got = evaluate(weighted, nb, RHO)
    assert not got.feasible and got.infeasible_reason == "clipped_prediction"
    assert got.fit is not None and got.fit.certified and got.heldout is None

    weights = p.weights.copy()
    weights[17] = 0.0
    unweighted = evaluate(build_problem(y=y, weights=weights, offset=offset, **common), nb, RHO)
    assert unweighted.feasible
    at = np.flatnonzero(unweighted.heldout.row == 17)
    predictor = unweighted.heldout.predictor[at]
    reported = clip_mu(p.link.inverse(stabilize_eta(predictor, p.link)), p.distribution)
    assert stabilize_eta(predictor, p.link) != predictor
    assert unweighted.heldout.mean[at] == reported and unweighted.heldout.deviance[at] == 0.0


# ── refusals, routes and work ─────────────────────────────────────────────────


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r or "cost-rule")
def test_unidentified_leave_out_system_is_refused_with_its_neighbourhood(route):
    """A factor level wholly inside one neighbourhood leaves a singular system: the typed refusal."""
    c = case("gaussian", 300)
    p = c.problem
    lone = int(np.flatnonzero(c.frame.f.to_numpy() == "d")[0])
    labels = np.where(c.frame.f.to_numpy() == "d", 7, np.arange(p.n) % 7)
    nb = from_groups(labels, p.n)
    with pytest.raises(LeaveOutRankError, match="group 7 held out") as refusal:
        evaluate(p, nb, RHO, route=route)
    assert refusal.value.neighbourhood == 7

    frame = c.frame.assign(f=np.where(np.arange(p.n) == lone, "e", c.frame.f.to_numpy()))
    model = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=8), "f": Categorical()},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(frame, p.y)
    single = problem_of(model, p.y, np.ones(p.n), None)
    with pytest.raises(LeaveOutRankError, match=f"row {lone} held out") as refusal:
        evaluate(single, single_rows(p.n), np.zeros(1), route=route)
    assert refusal.value.neighbourhood == lone


def test_cost_rule_sends_each_neighbourhood_to_the_cheaper_solve():
    """Neighbourhood-sized system when ``3 a^2 P + a^3 < P^3`` in weighted rows ``a``, else form and factor.

    A neighbourhood-sized system is solved in a stack or factored alone; ``solves`` counts the
    solves with one factored system.
    """
    c = case("gaussian", 300, zero_weights=True)
    p = c.problem
    nb = neighbourhoods("groups", c)
    weighted = np.array([int(p.active[nb.drop(k)].sum()) for k in range(nb.m)])
    width = p.width
    small = 3 * weighted**2 * width + weighted**3 < width**3
    assert 0 < small[weighted > 0].sum() < (weighted > 0).sum()
    work = evaluate(p, nb, RHO).work
    assert work.small_factorisations + work.stacked == int((small & (weighted > 0)).sum())
    assert work.factorisations == int((~small & (weighted > 0)).sum())
    assert work.solves == 2 * (work.factorisations + work.small_factorisations)
    forced = evaluate(p, nb, RHO, route="factor").work
    assert (forced.small_factorisations, forced.factorisations) == (0, int((weighted > 0).sum()))


@pytest.mark.parametrize(
    ("family", "form", "gradient", "expected"),
    [
        # (factorisations, neighbourhood-sized factorisations, solves, row reads / n, row products / n, extra products)
        ("gaussian", "rows", True, (0, 0, 0, 1, 2, 0)),
        ("gaussian", "rows", False, (0, 0, 0, 1, 1, 0)),
        ("poisson", "rows", True, (0, 0, 0, 2, 3, 0)),
        ("gaussian", "cv", True, (5, 0, 10, 1, 1, 5)),
        ("gaussian", "cv", False, (5, 0, 5, 1, 1, 0)),
        ("poisson", "cv", True, (5, 0, 10, 2, 2, 5)),
    ],
)
def test_work_counts_per_evaluation(family, form, gradient, expected):
    """One inner solve; single rows factor nothing; ``K`` large folds take ``K`` factorisations and ``2 K`` solves.

    A fold within one block is read once for its cross-products, scores and third-derivative term.
    """
    c = case(family, 300)
    n = c.problem.n
    work = evaluate(c.problem, neighbourhoods(form, c), RHO, gradient=gradient).work
    factorisations, small, solves, reads, products, extra = expected
    assert work.inner_solves == 1
    assert (work.factorisations, work.small_factorisations, work.solves) == (
        factorisations,
        small,
        solves,
    )
    assert (work.rows_read, work.rows_multiplied) == (reads * n, products * n + extra)


def test_supplied_fit_is_evaluated_without_a_solve(monkeypatch):
    """A converged fit and its geometry give the evaluation the solving entry point gives."""
    c = case("poisson", 300)
    nb = neighbourhoods("groups", c)
    solved = evaluate(c.problem, nb, RHO)
    cache: dict = {}
    result, slope_inverse, _ = fit_irls_direct(
        X=c.problem.dm,
        y=c.problem.y,
        weights=c.problem.weights,
        family=c.problem.distribution,
        link=c.problem.link,
        groups=c.problem.groups,
        lambda2=c.problem.lambdas(RHO),
        offset=c.problem.offset,
        reml_penalties=c.problem.penalties,
        convergence="mode_score",
        _mode_bar=MODE_BAR,
        return_xtwx=True,
        cache_out=cache,
        compute_rank_info=False,
        _compensate_centred_intercept=False,
        separation="record",
        weight_semantics=c.problem.weight_semantics,
    )
    fit = ModeFit.from_solve(c.problem.lambdas(RHO), result, slope_inverse, cache)

    def no_solve(**_):
        raise AssertionError("evaluate_fit ran an inner solve")

    monkeypatch.setattr(engine, "fit_irls_direct", no_solve)
    got = evaluate_fit(c.problem, nb, fit)
    assert got.feasible and got.work.inner_solves == 0 and got.work.pirls_iterations == 0
    _, _, kappa = one_step("poisson", 300, "groups")
    assert_same_criterion(c.problem, got, solved, kappa)
    gap = np.abs(got.heldout.predictor - solved.heldout.predictor)
    assert np.all(gap <= predictor_slack(kappa, solved.heldout.predictor))


def test_large_neighbourhoods_are_read_in_bounded_blocks(monkeypatch):
    """Under a small ``max_memory`` no read takes more rows than a block holds, and the sums stand."""
    c = case("poisson", 600)
    whole = {
        form: evaluate(c.problem, neighbourhoods(form, c), RHO)
        for form in ("rows", "forward", "neighbours")
    }
    assert ncv_rows.chunk_rows(c.problem.width) > c.problem.n
    reads = []

    def recorded(dm, rows, centre_hi, centre_lo):
        reads.append(len(rows))
        return ncv_rows.augmented_rows(dm, rows, centre_hi, centre_lo)

    monkeypatch.setattr(engine, "augmented_rows", recorded)
    _, _, kappa = one_step("poisson", 600, "forward")
    for form, over in (("rows", 0), ("forward", 0), ("neighbours", 7)):
        reads.clear()
        with parallel_config(max_memory=64 * 1024):
            limit = ncv_rows.chunk_rows(c.problem.width)
            got = evaluate_fit(c.problem, neighbourhoods(form, c), whole[form].fit)
        assert limit == 256 < int(neighbourhoods("forward", c).sizes().max())
        assert max(reads) <= limit + over and len(reads) > 2
        assert sum(reads) == got.work.rows_read
        # a neighbourhood past one block is read pass by pass, one within a block once
        assert (got.work.rows_read > whole[form].work.rows_read) == (form == "forward")
        assert got.work.rows_read >= whole[form].work.rows_read
        assert_same_criterion(c.problem, got, whole[form], kappa)


def test_unsupported_family_is_refused():
    c = case("gaussian", 300)
    p = c.problem
    gamma = SuperGLM(features={"x1": Spline(kind="ps", n_knots=8)}, family="gamma")
    with pytest.raises(ValueError, match="Gaussian family with an identity link and a Poisson"):
        build_problem(
            dm=p.dm,
            groups=p.groups,
            penalties=p.penalties,
            distribution=gamma._distribution,
            link=p.link,
            y=p.y,
            weights=p.weights,
            offset=p.offset,
            weight_semantics=p.weight_semantics,
        )
