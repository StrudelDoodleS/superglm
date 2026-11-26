"""The search for the smoothing that minimises the neighbourhood cross-validation criterion.

What a user reads of a search is tested on real fits: the chosen smoothing (by criterion, EDF
and fitted values, never by a smoothing parameter where the criterion is flat), the stop and
the bounds disclosed.  Events a trajectory cannot be relied on to reach (an infeasible trial, a
line search that fails, a slope out of the box) are forced, by a criterion written for the test
or by replacing single evaluations of the engine's.

Tolerances.  ``TOL`` is the search's first-order tolerance: two searches that stop within it
agree in the criterion to ``10 TOL`` relative (the test plan's bar; the criterion is second
order in the distance from a minimum).  Bars on EDF and fitted values between two searches are
empirical, since nothing bounds the smoothing where the criterion is flat; each says what was
measured.  Where one smoothing parameter is searched and the criterion curves at its minimum,
a first-order stop lies within ``TOL V / V''`` of the stationary point (``stop_reach``), and
what two searches return may differ by that distance times its own slope in ``log lambda``.
"""

import importlib
import pickle
import warnings
from dataclasses import dataclass, fields, replace
from functools import cache
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import eigh

from superglm import SuperGLM
from superglm._lambda_box import LambdaBox
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.model.base import model_weight_semantics
from superglm.model.reml_setup import collect_reml_groups
from superglm.ncv import engine, from_groups, single_rows
from superglm.ncv import search as search_module
from superglm.ncv.engine import (
    DEFAULT_NCV_TOL,
    Evaluation,
    LeaveOutRankError,
    WorkCounts,
    build_problem,
    evaluate,
    evaluate_fit,
    solve_mode,
)
from superglm.ncv.result import STOP_REASONS, TRIAL_FAILURES, NCVResult, StopReason, TrialFailure
from superglm.ncv.search import Descent, edf_share_start, first_order, search
from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
from superglm.reml.result import REMLResult
from superglm.solvers.mode_score import linear_predictor
from tests.test_import_compat import _LIBRARIES, _loaded_after, _type_checking_imports
from tests.test_ncv_engine import AIM, BAR, always, at_the_aim, solver_short
from tests.test_ncv_engine_rank_cache import one_row_on_the_right, rebuilt, spectral_basis, twin

U = 2.0**-53
TOL = DEFAULT_NCV_TOL
# An inner fit is certified within this share of the search's tolerance (``ncv_mode_bar``).
SHARE = 0.01


# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """A problem, its neighbourhoods, and the smoothing its estimated components were declared with."""

    problem: object
    nb: object
    configured: dict


def case_of(model, frame, y, nb, *, fixed=None, declared=1.0, **settings) -> Case:
    y = np.asarray(y, dtype=float)
    model.fit(frame, y)
    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
    problem = build_problem(
        dm=model._dm,
        groups=model._groups,
        penalties=penalties,
        distribution=model._distribution,
        link=model._link,
        y=y,
        weights=np.ones(len(y)),
        offset=None,
        weight_semantics=model_weight_semantics(model),
        fixed_lambdas=fixed,
        **settings,
    )
    return Case(problem, nb, dict.fromkeys(problem.names, declared))


def spline_model(family="gaussian", penalty=1.0, **spline):
    return SuperGLM(
        features={"t": Spline(kind="ps", n_knots=6, **spline)},
        family=family,
        selection_penalty=0,
        spline_penalty=penalty,
    )


@cache
def curve(scale=1.0, select=False, fixed=()) -> Case:
    """120 rows of a noisy sine in 12 groups; the response in units of ``scale``."""
    rng = np.random.default_rng(127)
    n = 120
    x = rng.uniform(size=n)
    y = np.sin(6 * x) + rng.normal(0.0, 0.2, n)
    nb = from_groups(np.arange(n) % 12, n)
    model = spline_model(select=select)
    return case_of(model, pd.DataFrame({"t": x}), scale * y, nb, fixed=dict(fixed) or None)


@cache
def counts_curve(max_pirls_iter=100) -> Case:
    """The same rows with Poisson counts."""
    rng = np.random.default_rng(127)
    n = 120
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(2 + 3 * np.sin(6 * x))).astype(float)
    nb = from_groups(np.arange(n) % 12, n)
    frame = pd.DataFrame({"t": x})
    return case_of(spline_model("poisson"), frame, y, nb, max_pirls_iter=max_pirls_iter)


@cache
def line(seed, n=300) -> Case:
    """A straight line with noise under a P-spline."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    y = 0.8 * x + rng.normal(0.0, 0.3, n)
    return case_of(spline_model(), pd.DataFrame({"t": x}), y, from_groups(np.arange(n) % 30, n))


@cache
def sizing(family, form) -> Case:
    """The design stage's sizing case: 20,000 rows, four P-splines, ``x2`` truly linear, 200 groups."""
    rng = np.random.default_rng(11)
    n = 20_000
    frame = pd.DataFrame({f"x{i}": rng.uniform(0, 1, n) for i in range(4)})
    group = rng.integers(0, 200, n)
    effect = rng.normal(0, 0.15, 200)[group]
    signal = (
        np.sin(2 * np.pi * frame.x0)
        + np.exp(2 * frame.x1) / 4
        + 0.8 * frame.x2
        + 0.2 * np.cos(9 * frame.x3)
    ).to_numpy()
    if family == "gaussian":
        y = signal + effect + rng.normal(0, 0.6, n)
    else:
        y = rng.poisson(np.exp(signal - 1.2 + effect)).astype(float)
    model = SuperGLM(
        features={f"x{i}": Spline(kind="ps", n_knots=16) for i in range(4)},
        family=family,
        selection_penalty=0,
        spline_penalty=0.1,
    )
    nb = single_rows(n) if form == "rows" else from_groups(group, n)
    return case_of(model, frame, y, nb, declared=0.1)


def share_start(c: Case, share=search_module.START_SHARE) -> np.ndarray:
    """The start at an EDF share, computed as the search computes its default one."""
    declared = np.log([c.configured[name] for name in c.problem.names])
    provisional = solve_mode(c.problem, declared)
    box = search_module._fit_box(c.problem)
    return edf_share_start(c.problem, provisional, box, declared, share)


@cache
def searched(family, form, share=None):
    """The sizing case's search: from the default start, or from the start at another EDF share."""
    c = sizing(family, form)
    start = None
    if share is not None:
        start = dict(zip(c.problem.names, np.exp(share_start(c, share)).tolist(), strict=True))
    return search(c.problem, c.nb, configured=c.configured, start=start).result


@dataclass(frozen=True)
class NarrowBox(LambdaBox):
    """A box with the bounds a test gives it."""

    low: float = 1e-6
    high: float = 1e10

    @property
    def floor(self) -> float:
        return self.low

    @property
    def ceiling(self) -> float:
        return self.high


# ── references of the tests' own ──────────────────────────────────────────────


def log_lambdas(c: Case, result) -> np.ndarray:
    return np.log([result.lambdas[name] for name in c.problem.names])


def penalty_matrix(problem, lambdas):
    return build_penalty_matrix(
        list(problem.dm.group_matrices),
        problem.groups,
        lambdas,
        problem.dm.p,
        reml_penalties=problem.penalties,
    )


def fitted(c: Case, result):
    """Per-term EDF and the fitted predictor of the library's fit at a result's smoothing."""
    problem = c.problem
    fit = solve_mode(problem, log_lambdas(c, result))
    assert fit.certified
    kept = np.eye(problem.dm.p) - fit.slope_inverse @ penalty_matrix(problem, fit.lambdas)
    edf = np.array([np.trace(kept[group.sl, group.sl]) for group in problem.groups])
    return edf, linear_predictor(problem.dm, fit.result, problem.offset)


def gaussian_refit_criterion(c: Case, rho) -> float:
    """The criterion from least-squares refits without each neighbourhood's rows.

    Each refit is the augmented least-squares problem ``[X; S^(1/2)] theta = [y; 0]`` on the
    rows that remain, solved by ``lstsq``: no normal equations and none of the engine's code.
    """
    problem, nb = c.problem, c.nb
    X = np.column_stack([np.ones(problem.n), problem.dm.toarray()])
    S = np.zeros((problem.width, problem.width))
    S[1:, 1:] = penalty_matrix(problem, problem.lambdas(np.asarray(rho, dtype=float)))
    values, vectors = np.linalg.eigh(S)
    root = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
    total = 0.0
    for k in range(nb.m):
        removed = nb.drop(k)
        kept = np.setdiff1d(np.arange(problem.n), removed)
        system = np.vstack([X[kept], root])
        target = np.concatenate([problem.y[kept], np.zeros(problem.width)])
        theta = np.linalg.lstsq(system, target, rcond=None)[0]
        total += float(np.sum((problem.y[removed] - X[removed] @ theta) ** 2))
    return total


def infeasible(reason="uncertified_mode"):
    return Evaluation(False, reason, float("nan"), None, None, None, WorkCounts())


def written(function):
    """A criterion from ``rho -> (value, slopes)``; ``None`` from it is an infeasible trial."""

    def criterion(rho, warm=None):
        answer = function(rho)
        if answer is None:
            return infeasible("non_finite")
        value, slopes = answer
        slopes = np.asarray(slopes, dtype=float)
        return Evaluation(True, None, float(value), slopes, None, None, WorkCounts())

    return criterion


def replace_criterion(monkeypatch, function):
    """Have ``search`` minimise the written criterion ``function`` instead of the engine's."""
    criterion = written(function)
    monkeypatch.setattr(
        search_module, "evaluate", lambda problem, nb, rho, **_: criterion(np.array(rho))
    )


def replace_trials(monkeypatch, rule):
    """Have ``rule(index, rho)`` answer a trial of the engine's criterion when it returns one.

    ``index`` counts the trials from 0.  Returns the list of every certified fit's smoothing.
    """
    seen, lambdas = [], []

    def patched(problem, nb, rho, **settings):
        seen.append(np.array(rho))
        answer = rule(len(seen) - 1, np.array(rho))
        if answer is None:
            answer = evaluate(problem, nb, rho, **settings)
        if isinstance(answer, Exception):
            raise answer
        if answer.fit is not None:
            lambdas.append(dict(answer.fit.lambdas))
        return answer

    monkeypatch.setattr(search_module, "evaluate", patched)
    return lambdas


def assert_first_order(c: Case, result):
    """A converged result: its criterion and slopes meet the first-order test and are the engine's.

    The engine's are re-solved from a cold start.  The inner fits are certified to a hundredth
    of the search's tolerance, so the two agree within a hundredth of the stopping bar.
    """
    rho = log_lambdas(c, result)
    state = first_order(rho, result.gradient, result.objective, result.lambda_box, TOL)
    assert state.met
    assert result.converged and result.termination_reason == "gradient"
    again = evaluate(c.problem, c.nb, rho)
    assert again.feasible
    share = SHARE * TOL * again.value
    assert abs(result.objective - again.value) <= share
    assert np.max(np.abs(result.gradient - again.gradient)) <= share


# ── units ─────────────────────────────────────────────────────────────────────


def test_response_units_do_not_move_the_chosen_smoothing():
    """A Gaussian response in units 10,000 times larger chooses the same curve in as many steps.

    With an additive constant in the stopping bar the rescaled search stopped at its start, in
    no steps, 0.77 away in log lambda and 0.53% worse in the criterion.
    """
    plain, small = curve(1.0), curve(1e-4)
    a = search(plain.problem, plain.nb, configured=plain.configured).result
    b = search(small.problem, small.nb, configured=small.configured).result
    assert_first_order(plain, a)
    assert_first_order(small, b)
    assert a.n_reml_iter >= 1 and b.n_reml_iter >= 1
    assert abs(a.n_reml_iter - b.n_reml_iter) <= 2
    assert abs(a.n_evaluations - b.n_evaluations) <= 2
    assert b.objective / 1e-4**2 == pytest.approx(a.objective, rel=10 * TOL)
    # Each stop has |V'| <= TOL V, so it lies within TOL V / V'' of the stationary point.
    rho, h = log_lambdas(plain, a), 0.05
    slope = [evaluate(plain.problem, plain.nb, rho + s * h).gradient[0] for s in (-1, 1)]
    curvature = (slope[1] - slope[0]) / (2 * h)
    assert curvature > 0
    reach = TOL * a.objective / curvature
    assert abs(log_lambdas(small, b)[0] - rho[0]) <= 4 * reach
    assert a.start_lambdas["t"] == pytest.approx(b.start_lambdas["t"], rel=1e-9)


def test_same_inputs_give_the_same_search():
    c = curve()
    a = search(c.problem, c.nb, configured=c.configured).result
    b = search(c.problem, c.nb, configured=c.configured).result
    assert a.lambdas == b.lambdas and a.objective == b.objective
    assert (a.n_reml_iter, a.n_evaluations) == (b.n_reml_iter, b.n_evaluations)
    assert a.lambda_history == b.lambda_history


# ── starts ────────────────────────────────────────────────────────────────────


@cache
def stationary_maximum() -> tuple[Case, float]:
    """Two groups on one grid whose criterion has a strict maximum at the default start.

    The response is one penalty eigenmode shared by the groups plus one with opposite signs in
    them.  The first asks for less smoothing and the second for more; their amplitudes are set
    so the two slopes cancel at the start (the design review's fixture).
    """
    grid = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"t": np.tile(grid, 2)})
    nb = from_groups(np.repeat([0, 1], len(grid)), 2 * len(grid))
    base = case_of(spline_model(), frame, np.sin(6 * frame.t), nb)
    problem = base.problem
    provisional = solve_mode(problem, np.zeros(1))
    box = search_module._fit_box(problem)
    seed = edf_share_start(problem, provisional, box, np.zeros(1))
    columns, S = problem.blocks[0]
    D = (provisional.slope_hessian - penalty_matrix(problem, provisional.lambdas))[columns, columns]
    ratios, modes = eigh(S, D)
    Z = problem.dm.toarray()[: len(grid), columns]
    Z = Z - Z.mean(axis=0)
    shared = np.tile(Z @ modes[:, -1], 2)
    opposed_half = Z @ modes[:, np.flatnonzero(ratios > 1e-9)[0]]
    opposed = np.concatenate([opposed_half, -opposed_half])

    def slope(y):
        return evaluate(replace(problem, y=y), nb, seed).gradient[0]

    y = shared + np.sqrt(-slope(shared) / slope(opposed)) * opposed
    return Case(replace(problem, y=y), nb, base.configured), float(seed[0])


def test_stationary_maximum_at_the_start_is_not_the_answer():
    c, seed = stationary_maximum()
    at_seed = evaluate(c.problem, c.nb, np.array([seed]))
    assert abs(at_seed.gradient[0]) <= TOL * at_seed.value, "the fixture is not stationary there"
    result = search(c.problem, c.nb, configured=c.configured).result
    assert np.log(result.start_lambdas["t"]) == pytest.approx(seed, abs=1e-12)
    assert result.n_reml_iter >= 1
    assert abs(log_lambdas(c, result)[0] - seed) > 0.1
    beside = min(gaussian_refit_criterion(c, [seed + step]) for step in (-0.1, 0.1))
    assert gaussian_refit_criterion(c, [seed]) > beside, "the fixture has no maximum there"
    assert gaussian_refit_criterion(c, log_lambdas(c, result)) < beside
    assert result.converged and result.termination_reason == "gradient"


def test_default_start_leaves_each_penalty_half_its_range():
    """On a ``select=True`` spline, whose two penalties share a term, each alone is at half its rank."""
    c = curve(select=True)
    problem = c.problem
    assert problem.names == ("t:null", "t:wiggle")
    result = search(problem, c.nb, configured=c.configured).result
    declared = np.zeros(2)
    provisional = solve_mode(problem, declared)
    D = provisional.slope_hessian - penalty_matrix(problem, provisional.lambdas)
    for name, (columns, S), component in zip(
        problem.names, problem.blocks, problem.penalties, strict=True
    ):
        lam = result.start_lambdas[name]
        hessian = D[columns, columns] + lam * S
        consumed = np.trace(np.linalg.solve(hessian, lam * S))
        rank = np.linalg.matrix_rank(S)
        assert rank == component.rank
        assert consumed == pytest.approx(rank / 2, rel=64 * np.linalg.cond(hessian) * U)
    assert set(result.start_lambdas) == set(problem.names)
    assert_first_order(c, result)


def test_penalty_without_a_range_starts_at_its_fallback():
    """A zero penalty, and a data Gram that is not positive definite, divide nothing by zero."""
    c = curve()
    problem = c.problem
    fit = solve_mode(problem, np.zeros(1))
    box = LambdaBox()
    fallback = np.array([0.25])
    columns, S = problem.blocks[0]
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        nothing = replace(problem, blocks=((columns, np.zeros_like(S)),))
        np.testing.assert_array_equal(edf_share_start(nothing, fit, box, fallback), fallback)
        flat = fit.data_gram.copy()
        flat[columns, columns][0, :] = 0.0
        flat[columns, columns][:, 0] = 0.0
        got = edf_share_start(problem, replace(fit, data_gram=flat), box, fallback)
        block = flat[columns, columns]
        assert got[0] == pytest.approx(np.log(np.trace(block) / np.trace(S)), rel=1e-12)
        for without in (np.zeros_like(flat), None):
            empty = replace(fit, data_gram=without)
            np.testing.assert_array_equal(edf_share_start(problem, empty, box, fallback), fallback)


@pytest.mark.parametrize("start", [None, {"t": 1.0}], ids=["default", "given"])
def test_a_declaration_of_zero_searches_as_a_model_declared_without_one(start):
    """``spline_penalty=0`` has no logarithm: the search is the one from the default declaration.

    Its default start, the start the caller gives, and every step after, are those of a model
    given no spline penalty, which declares 0.1.  A start of zero is refused.
    """
    c = curve()
    zero = dict.fromkeys(c.problem.names, 0.0)
    undeclared = dict.fromkeys(c.problem.names, 0.1)
    assert SuperGLM(features={"t": Spline()}).lambda2 == 0.1
    got = search(c.problem, c.nb, configured=zero, start=start).result
    want = search(c.problem, c.nb, configured=undeclared, start=start).result
    assert got.start_lambdas == want.start_lambdas and got.lambdas == want.lambdas
    assert got.lambda_history == want.lambda_history and got.objective == want.objective
    if start is None:
        default = share_start(replace(c, configured=undeclared))
        assert np.log(got.start_lambdas["t"]) == pytest.approx(default[0])
    else:
        assert got.start_lambdas == pytest.approx(start)
    with pytest.raises(ValueError, match="given for 't' must be a positive number, got 0.0"):
        search(c.problem, c.nb, configured=zero, start={"t": 0.0})


@pytest.mark.parametrize("held", [0.0, LambdaBox().floor], ids=["off", "at-the-floor"])
def test_fixed_components_keep_their_values_in_every_inner_fit(monkeypatch, held):
    """A start map moves estimated components only, and a fixed one is in no bound report.

    ``held`` is the wiggle penalty's fixed smoothing: zero is a penalty switched off, and the
    box's floor is a value the bound report would list for an estimated component.
    """
    box = LambdaBox()
    c = curve(select=True, fixed=(("t:wiggle", held),))
    assert c.problem.names == ("t:null",)
    solved = []
    real = search_module.solve_mode
    monkeypatch.setattr(
        search_module,
        "solve_mode",
        lambda *args, **settings: solved.append(real(*args, **settings)) or solved[-1],
    )
    fits = replace_trials(monkeypatch, lambda index, rho: None)
    start = {"t:wiggle": 0.5, "t:null": 2.0, "elsewhere": 3.0}
    result = search(c.problem, c.nb, configured=c.configured, start=start, box=box).result
    assert len(solved) == 1 and len(fits) == result.n_evaluations >= 2
    assert all(fit["t:wiggle"] == held for fit in [solved[0].lambdas, *fits])
    assert result.lambdas["t:wiggle"] == held
    assert result.start_lambdas == {"t:null": pytest.approx(2.0)}
    assert result.lambda_history[0]["t:null"] == pytest.approx(2.0)
    assert all(step["t:wiggle"] == held for step in result.lambda_history)
    assert "t:wiggle" not in dict(result.at_bound) and "t:wiggle" not in dict(result.bound_pressure)
    assert len(result.gradient) == 1


def test_nothing_to_estimate_is_one_evaluation_and_no_optimiser(monkeypatch):
    c = curve(fixed=(("t", 0.5),))
    assert c.problem.names == ()
    monkeypatch.setattr(search_module, "minimize", None)
    monkeypatch.setattr(search_module, "solve_mode", None)
    result = search(c.problem, c.nb, configured={}).result
    assert result.termination_reason == "no_smoothing_parameters" and result.converged
    assert (result.n_evaluations, result.n_reml_iter, result.infeasible_trials) == (1, 0, 0)
    assert result.lambdas == {"t": 0.5} and result.start_lambdas == {}
    assert result.objective == pytest.approx(gaussian_refit_criterion(c, []), rel=1e-9)
    assert result.work["inner_solves"] == 1


@pytest.mark.slow
def test_cold_seed_is_not_the_start_and_does_not_end_a_search():
    """REML's cold seed (1e-4) passes the first-order test on a plateau above the minimum.

    The default start is not that seed, and a search told to start there goes on to another
    start instead of reporting the plateau.
    """
    c = sizing("gaussian", "rows")
    seed = np.full(len(c.problem.names), np.log(1e-4))
    plateau = evaluate(c.problem, c.nb, seed)
    best = searched("gaussian", "rows")
    assert np.max(np.abs(plateau.gradient)) <= TOL * plateau.value, "the seed is not flat here"
    assert plateau.value > best.objective * (1 + 100 * TOL), "the plateau is not above the minimum"
    assert all(abs(np.log(lam) - np.log(1e-4)) > 1.0 for lam in best.start_lambdas.values())
    cold = dict.fromkeys(c.problem.names, 1e-4)
    result = search(c.problem, c.nb, configured=c.configured, start=cold).result
    assert result.start_lambdas == pytest.approx(cold)
    assert result.objective_history[0] == pytest.approx(plateau.value, rel=SHARE * TOL)
    assert result.n_reml_iter >= 1
    assert result.objective == pytest.approx(best.objective, rel=10 * TOL)
    assert result.converged and result.termination_reason == "gradient"


# ── bounds ────────────────────────────────────────────────────────────────────


def test_first_order_state_at_the_bounds():
    """Which slopes a bound blocks, which components sit on one, and which it holds back."""
    box = LambdaBox()
    low, high = box.log_floor, box.log_ceiling
    names = ("out_low", "out_high", "inside", "resting", "in_low", "in_high")
    rho = np.array([low, high, high - 0.005, high, low, high])
    slopes = np.array([1.0, -1.0, -1.0, -1e-9, -1e-9, 1e-9])
    state = first_order(rho, slopes, 1.0, box, TOL)
    assert state.bar == TOL
    assert state.sides == ("floor", "ceiling", None, "ceiling", "floor", "ceiling")
    # a slope out of the box at a bound is blocked; one just inside the ceiling is not
    assert state.projected.tolist() == [0.0, 0.0, -1.0, 0.0, -1e-9, 1e-9]
    assert not state.met
    assert state.at_bound(names) == (
        ("out_low", "floor"),
        ("out_high", "ceiling"),
        ("resting", "ceiling"),
        ("in_low", "floor"),
        ("in_high", "ceiling"),
    )
    # the bound holds back only an outward slope of at least the bar
    assert state.pressure(names) == (("out_low", "floor"), ("out_high", "ceiling"))
    at_rest = first_order(rho[[0, 1, 3]], slopes[[0, 1, 3]], 1.0, box, TOL)
    assert at_rest.met
    # a slope into the box at a bound is a descent direction and is never blocked
    inward = first_order(np.array([low, high]), np.array([-1.0, 1.0]), 1.0, box, TOL)
    assert inward.projected.tolist() == [-1.0, 1.0] and not inward.met
    assert inward.pressure(("a", "b")) == ()


@pytest.mark.parametrize("side", ["floor", "ceiling"])
def test_search_ends_at_a_bound_the_criterion_slopes_out_of(monkeypatch, side):
    c = curve()
    slope = 1.0 if side == "floor" else -1.0
    replace_criterion(monkeypatch, lambda rho: (100.0 + slope * rho[0], [slope]))
    result = search(c.problem, c.nb, configured=c.configured).result
    box = result.lambda_box
    assert result.lambdas["t"] == pytest.approx(getattr(box, side), rel=1e-12)
    assert result.converged and result.termination_reason == "gradient"
    assert result.at_bound == (("t", side),) and result.bound_pressure == (("t", side),)
    assert result.gradient.tolist() == [slope]
    assert result.n_reml_iter >= 1


@pytest.mark.parametrize("seed", [1, 0])
def test_truly_linear_term_ends_as_a_line_inside_the_box_or_at_its_ceiling(seed):
    """Measured on this machine: seed 1 stops inside the box (log lambda 14.8), seed 0 at the ceiling.

    Either is right, and on another BLAS build either sample may take the other end: what holds
    is a converged search, a term that is a line, and a ceiling that is disclosed when reached.
    """
    c = line(seed)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert_first_order(c, result)
    assert result.n_reml_iter <= 20
    edf, _ = fitted(c, result)
    assert edf[0] == pytest.approx(1.0, abs=0.01), "the term is not a line at the chosen smoothing"
    box = result.lambda_box
    assert result.at_bound == ((("t", "ceiling"),) if box.side(result.lambdas["t"]) else ())
    assert result.bound_pressure == ()


def test_ceiling_that_stops_a_falling_criterion_is_disclosed_as_holding_the_term():
    c = line(1)
    wide = search(c.problem, c.nb, configured=c.configured).result
    box = NarrowBox(high=100.0 * wide.start_lambdas["t"])
    at_ceiling = evaluate(c.problem, c.nb, np.array([box.log_ceiling]))
    assert at_ceiling.gradient[0] < -TOL * at_ceiling.value, "the criterion does not fall there"
    held = search(c.problem, c.nb, configured=c.configured, box=box).result
    assert held.lambdas["t"] == pytest.approx(box.ceiling, rel=1e-12)
    assert held.converged and held.termination_reason == "gradient"
    assert held.at_bound == (("t", "ceiling"),) and held.bound_pressure == (("t", "ceiling"),)
    assert held.n_reml_iter >= 1
    assert held.objective == pytest.approx(at_ceiling.value, rel=SHARE * TOL)
    assert held.objective > wide.objective


def test_inner_fits_are_certified_finely_enough_for_the_last_steps():
    """A 240-row Poisson search over single rows meets its first-order test at the default bars.

    Two inner fits at one smoothing differ in the criterion by a share of their certificate
    bar.  At a tenth of the search's tolerance that was the size of the last steps' gains:
    this search ended on its line search after 27 evaluations, its slope 1.1 times the
    stopping bar.  At a hundredth, and at the default aim of a thousandth, it converges in 8.
    """
    design = twin("poisson", False)
    problem = rebuilt(design.problem, mode_bar=engine.ncv_mode_bar(TOL))
    assert (problem.mode_aim, problem.mode_bar) == (AIM, BAR)
    result = search(problem, single_rows(problem.n), configured={"t": 1.0}, tol=TOL).result
    assert result.termination_reason == "gradient" and result.converged
    assert result.n_evaluations <= 12


def test_search_pays_for_an_aim_its_data_cannot_reach_at_one_fit(monkeypatch):
    """Every solve to the aim is made to end short of it, as on data that cannot resolve it.

    The provisional fit pays for the aim: its solve, and the step that takes it on to the
    acceptance bar.  Each later fit is warmed by one accepted at that bar and runs to it, so
    the search's solves are its evaluations and those two, and it ends on the curve of the
    search whose fits reach the aim.
    """
    c = counts_curve()
    assert (c.problem.mode_aim, c.problem.mode_bar) == (AIM, BAR)
    reached = search(c.problem, c.nb, configured=c.configured).result
    log: list = []
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(at_the_aim, log))
    outcome = search(c.problem, c.nb, configured=c.configured)
    result = outcome.result
    assert [call.bar for call in log] == [AIM] + [BAR] * (result.n_evaluations + 1)
    assert result.work["inner_solves"] == result.n_evaluations + 2
    assert result.work["fallback_solves"] == 1 and reached.work["fallback_solves"] == 0
    assert log[1].iterations == 1 and log[1].start is log[0].beta
    assert outcome.fit.certified and outcome.fit.bar == BAR
    assert result.objective == pytest.approx(reached.objective, rel=10 * TOL)
    assert abs(np.log(result.lambdas["t"] / reached.lambdas["t"])) <= 4 * stop_reach(c, reached)


@pytest.mark.parametrize("counts", [False, True], ids=["gaussian", "poisson"])
def test_search_goes_on_from_fits_at_the_limit_of_float64(monkeypatch, counts):
    """Every solve is made to end stagnated, as on data whose limit lies above the acceptance bar.

    Each trial has its criterion, so the search starts where it starts with certified fits
    and takes the same path to the last bit as the search whose fits are accepted at the
    acceptance bar: the states are the same, and only what is reported of them differs.  It
    is not converged, whatever its stop, and every evaluation is counted as made at a
    stagnated fit.  The provisional fit pays for the aim, and each later fit is one solve.
    """
    c = counts_curve() if counts else curve()
    assert (c.problem.mode_aim, c.problem.mode_bar) == (AIM, BAR)
    reached = search(c.problem, c.nb, configured=c.configured).result
    # A problem of its own for each search, so neither reads cross-products the other formed.
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(at_the_aim, []))
    accepted = search(replace(c.problem), c.nb, configured=c.configured).result
    log: list = []
    monkeypatch.setattr(engine, "fit_irls_direct", solver_short(always, log))
    outcome = search(replace(c.problem), c.nb, configured=c.configured)
    result = outcome.result
    assert result.infeasible_trials == 0 and result.infeasible_reasons == {}
    assert result.work["stagnated_fits"] == result.n_evaluations > 1
    assert accepted.work["stagnated_fits"] == reached.work["stagnated_fits"] == 0
    assert not result.converged and result.termination_reason == accepted.termination_reason
    assert accepted.converged == (accepted.termination_reason == "gradient")
    assert [call.bar for call in log] == [AIM] + [BAR] * (result.n_evaluations + 1)
    assert result.work["inner_solves"] == result.n_evaluations + 2
    assert result.work["fallback_solves"] == 1
    assert outcome.fit.stagnated and outcome.fit.bar == BAR
    # The start is read from the provisional fit, here one Newton step on from a state
    # within the aim: it is the unforced search's to well within the acceptance bar.
    assert result.start_lambdas == accepted.start_lambdas
    assert result.start_lambdas["t"] == pytest.approx(reached.start_lambdas["t"], rel=BAR)
    assert result.lambda_history == accepted.lambda_history
    assert result.objective_history == accepted.objective_history
    assert result.lambdas == accepted.lambdas and result.objective == accepted.objective
    assert np.array_equal(result.gradient, accepted.gradient)
    assert result.objective == pytest.approx(reached.objective, rel=10 * TOL)
    assert abs(np.log(result.lambdas["t"] / reached.lambdas["t"])) <= 4 * stop_reach(c, reached)
    if not counts:
        assert result.termination_reason == "gradient"


# ── infeasible trials ─────────────────────────────────────────────────────────


def wall(x):
    """The design stage's probe: a quadratic with no criterion beyond ``x[0] > 1.5``."""
    if x[0] > 1.5:
        return None
    value = 1000.0 + 0.5 * (50 * (x[0] - 1.4) ** 2 + (x[1] + 0.3) ** 2)
    return value, [50 * (x[0] - 1.4), x[1] + 0.3]


@pytest.mark.parametrize("start", [[1.0, 2.0], [0.9, -5.0], [1.45, 0.0]])
def test_infeasible_trial_does_not_end_the_search(start):
    """From the first two starts the first trial, a unit step in ``x[0]``, lands beyond the wall.

    Handed an infinity there, L-BFGS-B stops at the start and reports success.
    """
    handed = []
    descent = Descent(written(wall), LambdaBox())
    real = search_module._Run.objective

    def spy(run, rho):
        handed.append(real(run, rho))
        return handed[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(search_module._Run, "objective", spy)
        descent.descend([np.array(start)])
    point, reason = descent.outcome()
    assert reason == "gradient"
    # slopes within the bar TOL * 1000 where the curvatures are 50 and 1
    assert point.rho == pytest.approx([1.4, -0.3], abs=1e-3)
    assert (descent.infeasible >= 1) == (start[0] < 1.4)
    assert all(np.isfinite(value) and np.all(np.isfinite(slopes)) for value, slopes in handed)


def test_infeasible_trial_inside_a_real_search(monkeypatch):
    c = curve()
    free = search(c.problem, c.nb, configured=c.configured).result
    replace_trials(monkeypatch, lambda index, rho: infeasible("non_finite") if index == 1 else None)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.infeasible_trials == 1
    assert result.objective == pytest.approx(free.objective, rel=10 * TOL)
    assert_first_order(c, result)


def test_infeasible_start_is_retried_from_the_declared_smoothing(monkeypatch):
    """An uncertified inner mode at the start: the search goes on from where the provisional fit stood."""
    c = curve()
    free = search(c.problem, c.nb, configured=c.configured).result
    replace_trials(monkeypatch, lambda index, rho: infeasible() if index == 0 else None)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.start_lambdas == free.start_lambdas
    assert result.lambda_history[0] == {"t": pytest.approx(c.configured["t"])}
    assert result.infeasible_trials == 1
    assert result.objective == pytest.approx(free.objective, rel=10 * TOL)
    assert_first_order(c, result)


def test_unsolvable_leave_out_system_is_a_refusal_only_before_any_criterion(monkeypatch):
    c = curve()
    refusal = LeaveOutRankError(3, "group 3", 0.0)
    replace_trials(monkeypatch, lambda index, rho: refusal if index == 0 else None)
    with pytest.raises(LeaveOutRankError) as raised:
        search(c.problem, c.nb, configured=c.configured)
    assert raised.value.neighbourhood == 3
    replace_trials(monkeypatch, lambda index, rho: refusal if index == 1 else None)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.infeasible_trials == 1 and result.infeasible_reasons == {"leave_out_refused": 1}
    assert_first_order(c, result)


# ── every stop ────────────────────────────────────────────────────────────────


def kink(rho):
    """A criterion whose slope stays above the bar while its steps shrink to nothing."""
    d = rho[0] + 3.0
    return 1.0 + 1e-3 * abs(d) ** 1.01, [1.01e-3 * np.sign(d) * abs(d) ** 0.01]


def forced_gradient(monkeypatch):
    c = curve()
    return c, search(c.problem, c.nb, configured=c.configured).result


def forced_no_smoothing_parameters(monkeypatch):
    c = curve(fixed=(("t", 0.5),))
    return c, search(c.problem, c.nb, configured={}).result


def forced_criterion_settled(monkeypatch):
    c = curve()
    replace_criterion(monkeypatch, kink)
    return c, search(c.problem, c.nb, configured=c.configured).result


def forced_max_iter(monkeypatch):
    c = curve()
    return c, search(c.problem, c.nb, configured=c.configured, max_iter=1).result


def forced_line_search(monkeypatch):
    """A criterion that exists only at the starts: no step from either can be taken."""
    c = curve()
    islands = [share_start(c), np.zeros(1)]

    def island(rho):
        if not any(np.array_equal(rho, at) for at in islands):
            return None
        return 5.0 + rho[0], [1.0]

    replace_criterion(monkeypatch, island)
    return c, search(c.problem, c.nb, configured=c.configured).result


def forced_no_finite_criterion(monkeypatch):
    c = curve()
    replace_criterion(monkeypatch, lambda rho: None)
    return c, search(c.problem, c.nb, configured=c.configured).result


FORCED = {
    "gradient": forced_gradient,
    "no_smoothing_parameters": forced_no_smoothing_parameters,
    "criterion_settled": forced_criterion_settled,
    "max_iter": forced_max_iter,
    "line_search": forced_line_search,
    "no_finite_criterion": forced_no_finite_criterion,
}


def test_stop_reasons_are_a_closed_set():
    assert set(STOP_REASONS) == set(get_args(StopReason)) == set(FORCED)


@pytest.mark.parametrize("reason", list(FORCED))
def test_each_stop_reason_is_forced_and_reads_back(monkeypatch, reason):
    c, result = FORCED[reason](monkeypatch)
    assert isinstance(result, NCVResult)
    assert result.termination_reason == reason
    assert result.converged == (reason in ("gradient", "no_smoothing_parameters"))
    assert result.criterion == "ncv"
    assert len(result.lambda_history) == len(result.objective_history)
    assert set(result.lambdas) == {component.name for component in c.problem.penalties}


def test_search_that_cannot_improve_returns_where_it_stands(monkeypatch):
    c, result = forced_line_search(monkeypatch)
    lowest = min(result.objective_history)
    assert len(result.objective_history) == 2 and result.n_reml_iter == 0
    assert result.objective == lowest
    assert result.lambda_history[result.objective_history.index(lowest)] == result.lambdas
    assert result.gradient.tolist() == [1.0] and not result.converged
    assert result.infeasible_trials == result.n_evaluations - 2 > 0
    assert result.infeasible_reasons == {"non_finite": result.infeasible_trials}


def test_search_stopped_by_its_step_limit_keeps_its_last_step():
    c = curve()
    result = search(c.problem, c.nb, configured=c.configured, max_iter=1).result
    assert result.n_reml_iter == 1 and not result.converged
    assert result.objective == result.objective_history[-1] < result.objective_history[0]
    at = evaluate(c.problem, c.nb, log_lambdas(c, result))
    assert result.gradient == pytest.approx(at.gradient, rel=1e-9)
    assert abs(result.gradient[0]) > TOL * result.objective


def test_settled_search_is_not_a_converged_one(monkeypatch):
    c, result = forced_criterion_settled(monkeypatch)
    assert result.n_reml_iter >= 1
    assert abs(result.gradient[0]) > TOL * result.objective
    assert result.objective < result.objective_history[0]


def test_search_with_no_finite_criterion_invents_no_optimum(monkeypatch):
    c, result = forced_no_finite_criterion(monkeypatch)
    assert result.objective is None and result.gradient is None
    assert result.objective_history == [] and result.n_reml_iter == 0
    assert result.infeasible_trials == result.n_evaluations == 2
    assert result.infeasible_reasons == {"non_finite": 2}
    assert result.lambdas == result.start_lambdas
    assert not result.converged and result.bound_pressure == ()


def test_uncertified_inner_fits_everywhere_are_a_failed_search_at_the_declared_smoothing():
    """One PIRLS iteration certifies no mode, the provisional one included."""
    c = counts_curve(max_pirls_iter=1)
    outcome = search(c.problem, c.nb, configured=c.configured)
    result = outcome.result
    assert result.termination_reason == "no_finite_criterion" and not result.converged
    assert result.objective is None and result.gradient is None
    assert result.lambdas == c.configured == result.start_lambdas
    assert outcome.fit is None
    assert result.work["inner_solves"] >= 2
    assert result.infeasible_reasons == {"uncertified_mode": result.infeasible_trials}
    assert result.infeasible_trials == result.n_evaluations >= 1


def test_zero_criterion_is_its_least_value(monkeypatch):
    c = curve()
    replace_criterion(monkeypatch, lambda rho: (0.0, [0.0]))
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.converged and result.termination_reason == "gradient"
    assert result.objective == 0.0 and result.n_reml_iter == 0


# ── the sizing case ───────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("form", ["rows", "groups"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_sizing_case_converges_within_its_envelope(family, form):
    """20,000 rows, four smooths: the design stage measured 10 or 11 steps and 13 to 15 evaluations."""
    c = sizing(family, form)
    result = searched(family, form)
    assert_first_order(c, result)
    assert 3 <= result.n_reml_iter <= 20
    assert result.n_reml_iter < result.n_evaluations <= 30
    assert result.infeasible_trials == 0
    assert result.work["inner_solves"] == result.n_evaluations + 1
    # the truly linear term: a line, inside the box or at its ceiling, and the search not blocked
    names = list(c.problem.names)
    edf, eta = fitted(c, result)
    assert 1.0 <= edf[names.index("x2")] < 2.0
    assert result.lambdas["x2"] == max(result.lambdas.values()) <= result.lambda_box.ceiling
    assert set(result.at_bound) <= {("x2", "ceiling")}
    # the same curve from another start: by criterion, EDF and fitted values
    other = searched(family, form, 0.25)
    assert other.start_lambdas != result.start_lambdas
    assert other.objective == pytest.approx(result.objective, rel=10 * TOL)
    other_edf, other_eta = fitted(c, other)
    assert other_edf == pytest.approx(edf, rel=0.02)
    assert np.max(np.abs(other_eta - eta)) <= 2e-3 * np.std(eta)


# ── the result ────────────────────────────────────────────────────────────────


def test_result_reads_as_a_reml_result_and_holds_nothing_of_the_search_basis():
    c = curve(select=True)
    outcome = search(c.problem, c.nb, configured=c.configured)
    result = outcome.result
    assert isinstance(result, REMLResult) and result.criterion == "ncv"
    assert result.pirls_result is None and result.reml_penalties is None
    assert result.heldout is None and result.criterion_mean is None
    parameters = len(c.problem.names)
    for entry in fields(result):
        value = getattr(result, entry.name)
        assert not isinstance(value, np.ndarray) or value.size <= parameters, entry.name
    assert outcome.fit is not None and outcome.fit.certified
    assert outcome.fit.lambdas == result.lambdas
    assert result.n_reml_iter == len(result.lambda_history) - 1
    assert result.objective == result.objective_history[-1] == min(result.objective_history)
    assert result.lambda_history[-1] == result.lambdas
    assert result.work["inner_solves"] == result.n_evaluations + 1
    assert result.work["solves"] > 0 and result.infeasible_reasons == {}
    back = pickle.loads(pickle.dumps(result))
    assert back.lambdas == result.lambdas and back.termination_reason == "gradient"
    assert back.lambda_box == result.lambda_box
    np.testing.assert_array_equal(back.gradient, result.gradient)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"tol": 0.0}, "tolerance must be a positive number"),
        ({"tol": float("nan")}, "tolerance must be a positive number"),
        ({"max_iter": 0}, "at least one step"),
        ({"start": {"t": -1.0}}, "'t' must be a positive number"),
        ({"start": {"t": float("inf")}}, "'t' must be a positive number"),
    ],
)
def test_unusable_settings_are_refused_before_any_fit(monkeypatch, settings, message):
    c = curve()
    monkeypatch.setattr(search_module, "evaluate", None)
    monkeypatch.setattr(search_module, "solve_mode", None)
    with pytest.raises(ValueError, match=message):
        search(c.problem, c.nb, configured=c.configured, **settings)


# ── the search on the engine's cache, coordinates and rank verdicts ───────────


def test_gaussian_search_reads_its_rows_at_the_first_evaluation_only(monkeypatch):
    """Groups that are formed and factored: one read forms their cross-products, then none.

    The search asks for no held-out pairs, and neither the provisional fit nor the start value
    reads a row through the engine, so the whole search reads each row once.
    """
    c = curve()
    problem = replace(c.problem)  # a problem of its own: it holds no cross-products yet
    sizes, width = c.nb.sizes(), problem.width
    assert np.all(3 * sizes**2 * width + sizes**3 >= width**3), "a group is not formed and factored"
    trials = []
    real = search_module.evaluate

    def spy(*args, **settings):
        trials.append(real(*args, **settings))
        return trials[-1]

    monkeypatch.setattr(search_module, "evaluate", spy)
    result = search(problem, c.nb, configured=c.configured).result
    assert_first_order(c, result)
    assert len(trials) == result.n_evaluations >= 3
    assert [trial.work.rows_read for trial in trials] == [problem.n] + [0] * (len(trials) - 1)
    assert all(trial.feasible and trial.heldout is None for trial in trials)
    assert result.work["rows_read"] == problem.n
    assert result.work["inner_solves"] == len(trials) + 1
    assert sorted(problem.cache.held) == list(range(c.nb.m))


@cache
def with_twin(family, alias, form) -> Case:
    """A numeric column, the same column again when ``alias``, then a P-spline; 240 rows.

    The duplicate lies before the spline, so the slopes the fit identifies and the design's
    columns are numbered differently across the whole of the spline's block.
    """
    rng = np.random.default_rng(28)
    n = 240
    x = rng.normal(size=n)
    t = rng.uniform(size=n)
    signal = np.sin(6 * t) + 0.5 * x
    y = signal + rng.normal(0, 0.2, n)
    if family == "poisson":
        y = rng.poisson(np.exp(signal)).astype(float)
    frame = pd.DataFrame({"x": x, "twin": x, "t": t})
    features = {"x": Numeric(), "twin": Numeric(), "t": Spline(kind="ps", n_knots=6)}
    if not alias:
        del features["twin"]
    model = SuperGLM(features=features, family=family, selection_penalty=0, spline_penalty=1.0)
    nb = single_rows(n) if form == "rows" else from_groups(np.arange(n) % 12, n)
    return case_of(model, frame[list(features)], y, nb)


def data_gram_of(problem, fit):
    """The data's Gram at ``fit`` from the dense design, its rows centred about their weighted mean."""
    eta = linear_predictor(problem.dm, fit.result, problem.offset)
    w = problem.weights * (np.exp(eta) if problem.loss.log_mean else 1.0)
    X = problem.dm.toarray()
    X = X - w @ X / w.sum()
    return X.T @ (w[:, None] * X)


def read_at(c: Case, rho):
    """What is read of the library's fit at ``rho``: each term's EDF and the held-out predictors.

    A term's EDF is ``tr(H^+ D)`` on its columns, ``D`` from the dense rows, so a column the
    fit did not identify has none.  Also the condition of the Jacobi-scaled Hessian on the
    slopes the fit identified.
    """
    got = evaluate(c.problem, c.nb, np.asarray(rho, dtype=float))
    assert got.feasible
    fit = got.fit
    kept = fit.slope_inverse @ data_gram_of(c.problem, fit)
    edf = {group.name: float(np.trace(kept[group.sl, group.sl])) for group in c.problem.groups}
    on = np.flatnonzero(np.any(fit.slope_inverse != 0.0, axis=0))
    scale = np.sqrt(np.diag(fit.slope_hessian)[on])
    kappa = np.linalg.cond(fit.slope_hessian[np.ix_(on, on)] / np.outer(scale, scale))
    return edf, got.heldout.predictor, float(kappa)


def stop_reach(c: Case, result, h=0.05) -> float:
    """How far in ``log lambda`` a first-order stop of a one-parameter search lies from the stationary point.

    ``|V'| <= TOL V`` at the stop, so at most ``TOL V / V''``, the curvature from the engine's
    slopes ``h`` either side.
    """
    rho = log_lambdas(c, result)
    slope = [evaluate(c.problem, c.nb, rho + s * h).gradient[0] for s in (-1, 1)]
    curvature = (slope[1] - slope[0]) / (2 * h)
    assert curvature > 0
    return TOL * result.objective / curvature


def watch_fits(monkeypatch, reshape=lambda fit: fit):
    """Record every fit a search makes, each reaching the engine as ``reshape`` hands it on."""
    fits = []

    def solve(problem, rho, **settings):
        fits.append(reshape(solve_mode(problem, rho, **settings)))
        return fits[-1]

    def evaluated(problem, nb, rho, *, warm=None, **settings):
        work = WorkCounts()
        answer = evaluate_fit(problem, nb, solve(problem, rho, warm=warm, work=work), **settings)
        answer.work.inner_solves = work.inner_solves
        answer.work.pirls_iterations = work.pirls_iterations
        return answer

    monkeypatch.setattr(search_module, "solve_mode", solve)
    monkeypatch.setattr(search_module, "evaluate", evaluated)
    return fits


@pytest.mark.parametrize("retained", ["columns", "basis"])
@pytest.mark.parametrize(
    ("family", "form"), [("gaussian", "rows"), ("gaussian", "groups"), ("poisson", "groups")]
)
def test_search_on_an_aliased_design_is_the_search_without_the_alias(
    monkeypatch, family, form, retained
):
    """A duplicated column moves neither the start nor what the search returns.

    Through both forms a rank-deficient fit hands the engine: the solver's representative
    columns, and a basis of the identified slopes (forced).  The start is the half-range
    point of the spline's own columns, read in the design's columns in both.
    """
    aliased, clean = with_twin(family, True, form), with_twin(family, False, form)
    want = search(clean.problem, clean.nb, configured=clean.configured).result
    fits = watch_fits(monkeypatch, spectral_basis if retained == "basis" else lambda fit: fit)
    got = search(aliased.problem, aliased.nb, configured=aliased.configured).result
    monkeypatch.undo()

    problem = aliased.problem
    identified = [np.flatnonzero(np.any(fit.slope_inverse != 0.0, axis=0)) for fit in fits]
    assert all(fit.result.reml_hessian_rank == problem.width - 1 for fit in fits)
    if retained == "columns":
        assert all(kept.tolist() == [0, *range(2, problem.dm.p)] for kept in identified)
    else:
        assert all(len(kept) == problem.dm.p for kept in identified)
    columns, S = problem.blocks[0]
    assert (columns.start, columns.stop) == (2, problem.dm.p)

    # the start: half the range of the spline's penalty on its own columns, as without the alias
    lam = got.start_lambdas["t"]
    hessian = data_gram_of(problem, fits[0])[columns, columns] + lam * S
    consumed = np.trace(np.linalg.solve(hessian, lam * S))
    resolved = 64 * np.linalg.cond(hessian) * U
    assert consumed == pytest.approx(np.linalg.matrix_rank(S) / 2, rel=resolved)
    assert lam == pytest.approx(want.start_lambdas["t"], rel=resolved)

    # what the search returns: the criterion, the smoothing, each term's EDF, the held-out predictors
    assert_first_order(aliased, got)
    assert_first_order(clean, want)
    assert got.objective == pytest.approx(want.objective, rel=10 * TOL)
    rho, h = log_lambdas(clean, want), 0.05
    apart = 4 * stop_reach(clean, want)
    assert abs(log_lambdas(aliased, got)[0] - rho[0]) <= apart
    edf, held, kappa = read_at(aliased, log_lambdas(aliased, got))
    edf_want, held_want, _ = read_at(clean, rho)
    (edf_low, held_low, _), (edf_high, held_high, _) = (
        read_at(clean, rho - h),
        read_at(clean, rho + h),
    )
    rounding = 64 * kappa * U
    for mine, theirs in ((edf["t"], "t"), (edf["x"] + edf["twin"], "x")):
        moves = abs(edf_high[theirs] - edf_low[theirs]) / (2 * h)
        assert abs(mine - edf_want[theirs]) <= apart * moves + rounding * problem.dm.p, theirs
    assert edf["twin"] == 0.0
    moves = np.max(np.abs(held_high - held_low)) / (2 * h)
    slack = apart * moves + rounding * np.maximum(1.0, np.abs(held_want))
    assert np.all(np.abs(held - held_want) <= slack)


def test_every_fit_of_a_search_keeps_the_solvers_decomposition(monkeypatch):
    """The provisional fit and each trial's carry what a rank-deficient fit is evaluated in."""
    c = with_twin("gaussian", True, "groups")
    fits = watch_fits(monkeypatch)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert len(fits) == result.n_evaluations + 1 >= 3
    for fit in fits:
        retained = fit.result.reml_slope_decomposition
        assert retained is not None and retained.solution_basis is not None
        assert retained.solution_basis.shape == (c.problem.dm.p, c.problem.dm.p - 1)


# ── refused against infeasible ────────────────────────────────────────────────


@cache
def lone_group() -> Case:
    """The ``curve`` rows with a numeric column that is zero outside group 5.

    With group 5 held out the column is zero on every row that remains, and no penalty sees it.
    """
    rng = np.random.default_rng(127)
    n = 120
    x = rng.uniform(size=n)
    y = np.sin(6 * x) + rng.normal(0.0, 0.2, n)
    group = np.arange(n) % 12
    frame = pd.DataFrame({"t": x, "only": np.where(group == 5, rng.normal(size=n), 0.0)})
    model = SuperGLM(
        features={"t": Spline(kind="ps", n_knots=6), "only": Numeric()},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    return case_of(model, frame, y, from_groups(group, n))


def test_rows_that_never_identify_the_model_end_the_search_at_its_first_trial(monkeypatch):
    """The engine's refusal holds at every smoothing, so the first trial raises it and no other runs."""
    c = lone_group()
    for rho in (-8.0, 0.0, 8.0):
        with pytest.raises(LeaveOutRankError, match="group 5 held out"):
            evaluate(c.problem, c.nb, np.array([rho]), heldout=False)
    trials = []
    real = search_module.evaluate
    monkeypatch.setattr(
        search_module, "evaluate", lambda *args, **s: trials.append(args[2]) or real(*args, **s)
    )
    with pytest.raises(LeaveOutRankError, match="group 5 held out") as refusal:
        search(c.problem, c.nb, configured=c.configured)
    assert refusal.value.neighbourhood == 5 and len(trials) == 1


def test_trial_past_the_limit_on_kept_rows_fits_is_infeasible_and_says_so(monkeypatch):
    """Every leave-out system flagged: a trial stops at ``KEPT_FITS`` and the search fails, unrefused.

    The model is identified at every smoothing, which the same flags show once the limit
    covers every group: each group is then fitted from its kept rows and the search converges
    on the criterion of the unflagged one.
    """
    c = curve()
    assert c.nb.m > engine.KEPT_FITS
    free = search(replace(c.problem), c.nb, configured=c.configured).result
    monkeypatch.setattr(engine._Trial, "questionable", np.inf)
    result = search(replace(c.problem), c.nb, configured=c.configured).result
    assert result.termination_reason == "no_finite_criterion" and not result.converged
    assert result.objective is None and result.infeasible_trials == result.n_evaluations == 2
    assert result.infeasible_reasons == {"kept_rows_limit": 2}
    assert result.work["kept_fits"] == 2 * (engine.KEPT_FITS + 1)
    monkeypatch.setattr(engine, "KEPT_FITS", c.nb.m)
    afforded = search(replace(c.problem), c.nb, configured=c.configured).result
    assert afforded.converged and afforded.infeasible_reasons == {}
    assert afforded.work["kept_fits"] == c.nb.m * afforded.n_evaluations
    assert afforded.objective == pytest.approx(free.objective, rel=10 * TOL)


def test_leave_out_system_unusable_at_one_smoothing_is_one_infeasible_trial():
    """A start where the penalty is below what the kept rows' factor resolves; the search goes on.

    At ``lambda = exp(-50)`` the penalty alone tells the right-hand columns apart and is 1e-22
    of the kept rows in their own factor: infeasible there, identified everywhere.
    """
    problem, nb = one_row_on_the_right()
    box = NarrowBox(low=float(np.exp(-60.0)))
    name = problem.names[0]
    at_start = evaluate(problem, nb, np.array([-50.0]), heldout=False)
    assert not at_start.feasible and at_start.infeasible_reason == "leave_out_rank"
    assert not at_start.past_kept_fits
    result = search(
        problem, nb, configured={name: 1.0}, start={name: float(np.exp(-50.0))}, box=box
    ).result
    assert np.log(result.start_lambdas[name]) == pytest.approx(-50.0)
    assert set(result.infeasible_reasons) == {"leave_out_rank"}
    assert result.infeasible_reasons["leave_out_rank"] == result.infeasible_trials >= 1
    assert result.objective is not None and result.n_reml_iter >= 1
    assert result.lambda_history[0] != result.start_lambdas


def test_rank_deficient_fit_without_its_coordinates_is_one_infeasible_trial(monkeypatch):
    """A fit that kept a basis and hands none back gives no trial; the next start is searched."""
    c = with_twin("gaussian", True, "groups")
    free = search(c.problem, c.nb, configured=c.configured).result

    def unread(index, rho):
        if index:
            return None
        general = spectral_basis(solve_mode(c.problem, rho))
        bare = replace(general, result=replace(general.result, reml_slope_decomposition=None))
        return evaluate_fit(c.problem, c.nb, bare, heldout=False)

    replace_trials(monkeypatch, unread)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.infeasible_reasons == {"rank_deficient_fit": 1}
    assert result.objective == pytest.approx(free.objective, rel=10 * TOL)
    assert_first_order(c, result)


def test_trial_failures_are_a_closed_set():
    theirs = set(get_args(engine.InfeasibleReason))
    assert set(TRIAL_FAILURES) == set(get_args(TrialFailure))
    assert set(TRIAL_FAILURES) == theirs | {"kept_rows_limit", "leave_out_refused"}


@pytest.mark.parametrize("reason", get_args(engine.InfeasibleReason))
def test_each_reason_of_the_engine_is_counted_under_its_name(monkeypatch, reason):
    c = curve()
    replace_trials(monkeypatch, lambda index, rho: infeasible(reason) if index == 0 else None)
    result = search(c.problem, c.nb, configured=c.configured).result
    assert result.infeasible_reasons == {reason: 1} and result.infeasible_trials == 1
    assert_first_order(c, result)


# ── the package ───────────────────────────────────────────────────────────────

SURFACE = {
    # the neighbourhood structure, its builders and its checks
    "FEW_NEIGHBOURHOODS",
    "LevelConflict",
    "Neighbourhoods",
    "build_neighbourhoods",
    "check_leaves_weighted_rows",
    "check_scores_weighted_rows",
    "from_groups",
    "from_neighbours",
    "from_splitter",
    "single_rows",
    "unidentified_levels",
    "warn_few_neighbourhoods",
    # the engine
    "DEFAULT_NCV_TOL",
    "Evaluation",
    "HeldOutPairs",
    "LeaveOutRankError",
    "ModeFit",
    "NCVProblem",
    "WorkCounts",
    "build_problem",
    "evaluate",
    "evaluate_fit",
    "ncv_mode_aim",
    "ncv_mode_bar",
    # the search and its result
    "DEFAULT_MAX_NCV_ITER",
    "SearchOutcome",
    "search_smoothing",
    "NCVResult",
    "STOP_REASONS",
    "TRIAL_FAILURES",
}


def test_package_exports_what_a_fit_calls():
    """One list: what the package names, what a type checker reads, and what each module defines."""
    package = importlib.import_module("superglm.ncv")
    assert set(package.__all__) == SURFACE
    table = {(owner, name) for owner, names in package._LAZY_EXPORTS.items() for name in names}
    assert {name for _, name in table} == SURFACE
    assert table == _type_checking_imports("superglm.ncv")
    for owner, name in sorted(table):
        assert getattr(package, name) is getattr(importlib.import_module(owner), name)
        assert name in dir(package)
    assert package.search_smoothing is search and package.search is search_module


def test_importing_the_package_loads_none_of_its_modules():
    """A name costs its import when it is read: the package alone loads no library and no module.

    A neighbourhood builder then loads its own module, and neither the engine, the solvers nor
    SciPy's optimiser.
    """
    ours = ("superglm.ncv.engine", "superglm.ncv.search", "superglm.ncv.neighbourhoods")
    fitting = ("scipy.optimize", "superglm.solvers", "superglm.reml", *ours)
    assert _loaded_after("import superglm.ncv", (*_LIBRARIES, *fitting)) == []
    after = _loaded_after("from superglm.ncv import single_rows", fitting)
    assert after == ["superglm.ncv.neighbourhoods"]
