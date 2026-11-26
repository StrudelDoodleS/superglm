"""``SuperGLM.fit_ncv`` against mgcv 1.9.3's ``gam(method = "NCV", nei = ...)``, pinned.

mgcv is run as a black box by ``tests/oracles/ncv_mgcv_reference.R`` on the library's own design
and penalties (``paraPen``), its prior weights and the library's neighbourhoods; its outputs are
pinned in ``tests/fixtures/ncv_mgcv_reference.json`` with the simulated rows they were fitted to,
so no test here runs R.  The regeneration commands are at the top of the R script.

What mgcv documents (``?NCV``, ``?gam``; Wood, "On Neighbourhood Cross Validation", arXiv
2404.16490) and these tests rely on: its NCV score is ``V = sum_k sum_{i in d(k)} D(y_i, t_i^a(k))``,
the summed held-out deviance with prior weights, neither averaged nor divided by the scale; for a
non-Gaussian family each leave-out fit is one Newton step from the full fit, as in the library.
Three things are observed of mgcv 1.9.3, not documented: a ``paraPen`` smoothing parameter
multiplies the penalty as given, so mgcv's ``sp`` are the library's lambdas; its NCV score leaves
an offset out of the held-out predictor, so no case here has one (that form is pinned to refits in
``test_ncv_engine.py``); and its default ``Vp`` under ``nei`` carries a neighbourhood correction, so
the ordinary Bayesian covariance is read from a refit with ``nei$jackknife = FALSE``.

Cases.  A Gaussian model with unequal prior weights and a Poisson rate with its exposure as the
prior weight (mgcv's ``weights``; the library under ``weight_semantics="prior"``), each under five
hold-out forms: single rows, 25 groups, ten shuffled folds, each row with the three rows either
side, and a forward-in-time split (``TimeSeriesSplit(5)``: each test block is scored with every
later row removed too).

Tolerances.  ``U`` is the unit roundoff and ``kappa`` the 2-norm condition number of the
Jacobi-scaled penalised Hessian at the comparison's smoothing, computed here; two exact
computations agree within ``64 kappa U`` of their magnitude (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., 2002, section 10.1).  Each engine's full fit at a smoothing lies
one Newton step ``H^-1 g`` of its own penalised score ``g`` from the mode (exactly, for a
Gaussian model; to its stopping rule for Poisson), and that step, computed here from each
engine's coefficients, bounds how far their linear predictors may differ; everything read at one
smoothing is compared within what that difference moves.  Two first-order stops of the smoothing
search lie within ``H_V^-1 g_V`` of the criterion's minimum, ``g_V`` the criterion's slopes at
each stop and ``H_V`` its curvature there; what the two stops' fits may differ by is that step
mapped through each quantity's slopes in ``log lambda``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from superglm import SuperGLM
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.ncv import engine, from_groups, from_neighbours, from_splitter, single_rows
from superglm.ncv.neighbourhoods import Neighbourhoods
from superglm.solvers.mode_score import linear_predictor
from tests.test_ncv_engine import deviance_slope, problem_of

FIXTURE = Path(__file__).parent / "fixtures" / "ncv_mgcv_reference.json"
FAMILIES = ("gaussian", "poisson")
FORMS = ("rows", "groups", "folds", "neighbours", "forward")
N_ROWS = 400
N_GROUPS = 25
U = 2.0**-53


# ── the simulated rows and the model, shared with tests/oracles/ncv_mgcv_reference.py ──


def simulate(family: str) -> pd.DataFrame:
    """Two curves, a numeric column, a factor, 25 groups of unequal size and unequal prior weights.

    Gaussian: weights 1, 2 or 3 and noise of variance ``0.25 / w``.  Poisson: the response is
    a count over its exposure and the exposure is the weight.  Rows are in time order.
    """
    rng = np.random.default_rng({"gaussian": 2026, "poisson": 1126}[family])
    n = N_ROWS
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(-2, 2, n),
            "z": rng.normal(0.0, 1.0, n),
            "f": rng.choice(list("abc"), n, p=[0.5, 0.3, 0.2]),
            "group": rng.choice(N_GROUPS, n, p=rng.dirichlet(np.full(N_GROUPS, 3.0))),
        }
    )
    effect = rng.normal(0.0, 0.3, N_GROUPS)[frame.group]
    signal = (
        np.sin(6 * frame.x1) + 0.3 * frame.x2**2 + 0.2 * frame.z + 0.4 * (frame.f == "b") + effect
    ).to_numpy()
    if family == "gaussian":
        frame["w"] = rng.choice([1.0, 2.0, 3.0], n)
        frame["y"] = signal + rng.normal(0.0, 0.5, n) / np.sqrt(frame.w)
    else:
        frame["w"] = rng.uniform(0.3, 1.5, n)
        frame["y"] = rng.poisson(frame.w * np.exp(signal - 0.6)) / frame.w
    return frame


def model_of(family: str) -> SuperGLM:
    return SuperGLM(
        features={
            "x1": Spline(kind="ps", n_knots=8),
            "x2": Spline(kind="ps", n_knots=8),
            "z": Numeric(),
            "f": Categorical(),
        },
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
        weight_semantics="prior",
    )


def predictors(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["x1", "x2", "z", "f"]]


def hold_out(form: str, frame: pd.DataFrame) -> dict:
    """``fit_ncv``'s hold-out arguments for a form."""
    n = len(frame)
    return {
        "rows": {},
        "groups": {"groups": frame.group.to_numpy()},
        "folds": {"cv": KFold(10, shuffle=True, random_state=3)},
        "neighbours": {"neighbours": [np.arange(max(i - 3, 0), min(i + 4, n)) for i in range(n)]},
        "forward": {"cv": TimeSeriesSplit(5)},
    }[form]


def library_problem(family: str, frame: pd.DataFrame):
    """The plain ``fit`` of the model and the engine's problem on its design, every penalty estimated."""
    y, w = frame.y.to_numpy(), frame.w.to_numpy()
    model = model_of(family).fit(predictors(frame), y, sample_weight=w)
    return model, problem_of(model, y, w, None)


def neighbourhoods_of(form: str, frame: pd.DataFrame) -> Neighbourhoods:
    """The library's neighbourhoods for a form's ``fit_ncv`` arguments."""
    n, arguments = len(frame), hold_out(form, frame)
    if "cv" in arguments:
        return from_splitter(arguments["cv"], predictors(frame), frame.y.to_numpy())
    if "neighbours" in arguments:
        return from_neighbours(arguments["neighbours"], n)
    if "groups" in arguments:
        return from_groups(arguments["groups"], n)
    return single_rows(n)


def mgcv_nei(nb: Neighbourhoods) -> dict[str, list[int]]:
    """mgcv's ``nei``: removed rows ``a`` and scored rows ``d``, 1-based, with end positions."""
    removed, scored = [], []
    for k in range(nb.m):
        drop = nb.drop(k)
        pos = nb.scored_positions(k)
        removed.append(drop)
        scored.append(drop if pos is None else drop[pos])
    return {
        "a": (np.concatenate(removed) + 1).tolist(),
        "ma": np.cumsum([len(r) for r in removed]).tolist(),
        "d": (np.concatenate(scored) + 1).tolist(),
        "md": np.cumsum([len(s) for s in scored]).tolist(),
    }


def from_nei(nei: dict, n: int) -> Neighbourhoods:
    """The library's neighbourhoods read from mgcv's ``nei`` as pinned."""
    a, d = np.asarray(nei["a"]) - 1, np.asarray(nei["d"]) - 1
    a_ptr, d_ptr = (np.concatenate([[0], nei[e]]).astype(np.intp) for e in ("ma", "md"))
    drops = [np.sort(a[a_ptr[k] : a_ptr[k + 1]]) for k in range(len(a_ptr) - 1)]
    positions = [np.searchsorted(r, d[d_ptr[k] : d_ptr[k + 1]]) for k, r in enumerate(drops)]
    return Neighbourhoods(
        n_rows=n,
        form="cv",
        drop_ptr=a_ptr,
        drop_rows=np.concatenate(drops).astype(np.intp),
        score_ptr=d_ptr,
        score_pos=np.concatenate(positions).astype(np.intp),
    )


def design(problem) -> tuple[np.ndarray, list[np.ndarray]]:
    """The dense design with the intercept first and each penalty padded to its width."""
    X = np.column_stack([np.ones(problem.n), problem.dm.toarray()])
    penalties = []
    for columns, block in problem.blocks:
        full = np.zeros((problem.width, problem.width))
        full[1:, 1:][columns, columns] = block
        penalties.append(full)
    return X, penalties


def fingerprint(X: np.ndarray, penalties: list[np.ndarray]) -> dict:
    return {
        "column_sums": X.sum(axis=0).tolist(),
        "column_squares": (X**2).sum(axis=0).tolist(),
        "penalty_norms": [float(np.linalg.norm(S)) for S in penalties],
    }


@cache
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ── what each comparison reads ───────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    family: str
    form: str
    frame: pd.DataFrame
    problem: engine.NCVProblem
    X: np.ndarray
    penalties: list[np.ndarray]
    nb: Neighbourhoods
    mgcv: dict

    @property
    def y(self) -> np.ndarray:
        return self.frame.y.to_numpy()

    @property
    def w(self) -> np.ndarray:
        return self.frame.w.to_numpy()

    @property
    def rho(self) -> np.ndarray:
        """mgcv's chosen smoothing, as the library's ``log lambda``."""
        return np.log(self.mgcv["sp"])


@cache
def rows_of(family: str) -> pd.DataFrame:
    return pd.DataFrame(fixture()[family]["rows"])


@cache
def built(family: str):
    return library_problem(family, rows_of(family))


@cache
def case(family: str, form: str) -> Case:
    _, problem = built(family)
    X, penalties = design(problem)
    pinned = fixture()[family]["forms"][form]
    nb = from_nei(pinned["nei"], problem.n)
    return Case(family, form, rows_of(family), problem, X, penalties, nb, pinned["mgcv"])


@dataclass(frozen=True)
class Mode:
    """A full fit at one smoothing in the exported design's coordinates, with its Newton step left."""

    eta: np.ndarray
    step: np.ndarray
    hessian: np.ndarray

    @property
    def kappa(self) -> float:
        """2-norm condition number of the Jacobi-scaled penalised Hessian."""
        d = 1.0 / np.sqrt(np.diag(self.hessian))
        return float(np.linalg.cond(self.hessian * d[:, None] * d[None, :]))


def mode(c: Case, beta: np.ndarray) -> Mode:
    """The step ``H^-1 g`` of mgcv's objective ``D(b) + sum_j sp_j b'S_j b`` at ``beta``, as ``X`` times it.

    ``g`` and ``H`` are half the gradient and the Hessian, both exact in ``b`` for the Gaussian
    model and the Poisson log link (its observed and expected Hessians coincide).
    """
    eta = c.X @ beta
    mean = eta if c.family == "gaussian" else np.exp(eta)
    variance = np.ones_like(mean) if c.family == "gaussian" else mean
    penalty = sum(s * S for s, S in zip(c.mgcv["sp"], c.penalties, strict=True))
    hessian = c.X.T @ (c.X * (c.w * variance)[:, None]) + penalty
    score = c.X.T @ (c.w * (c.y - mean)) - penalty @ beta
    return Mode(eta, c.X @ np.linalg.solve(hessian, score), hessian)


def coordinates(c: Case, model: SuperGLM) -> np.ndarray:
    """A fitted model's coefficients in the exported design: its linear predictor regressed on ``X``."""
    eta = model.predict(predictors(c.frame))
    eta = eta if c.family == "gaussian" else np.log(eta)
    return np.linalg.lstsq(c.X, eta, rcond=None)[0]


def slope_covariance(c: Case, model: SuperGLM) -> np.ndarray:
    """The model's slope covariance carried into the exported design's slopes.

    Both designs span the same columns with the intercept: the model's slopes ``B`` are
    ``1 c' + A R`` in the exported slopes ``A``, so its coefficients map as ``R`` and its
    covariance as ``R cov R'``.
    """
    covariance, _ = model._coef_covariance
    carry = np.linalg.lstsq(c.X, model._dm.toarray(), rcond=None)[0][1:]
    return carry @ covariance @ carry.T


@cache
def ncv_fit(family: str, form: str) -> SuperGLM:
    """The public ``fit_ncv`` at its defaults on a case's rows and hold-out form.

    ``TimeSeriesSplit(5)`` gives five neighbourhoods, which the library warns about by design.
    """
    frame = rows_of(family)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="fit_ncv has 5 neighbourhoods")
        return model_of(family).fit_ncv(
            predictors(frame),
            frame.y.to_numpy(),
            sample_weight=frame.w.to_numpy(),
            **hold_out(form, frame),
        )


@cache
def fit_at(family: str, form: str, shift: tuple[float, ...] = (0.0, 0.0)) -> SuperGLM:
    """The public ``fit`` at mgcv's chosen smoothing times ``exp(shift)``, on a clone of the NCV fit."""
    c, model = case(family, form), ncv_fit(family, form)
    lambdas = np.exp(c.rho + np.asarray(shift))
    oracle = model._clone_without_features(set(), smoothing_resolved=True)
    oracle.lambda2 = {**oracle.lambda2, **dict(zip(c.problem.names, lambdas.tolist(), strict=True))}
    return oracle.fit(predictors(c.frame), c.y, sample_weight=c.w)


def mean_of(model: SuperGLM, c: Case) -> np.ndarray:
    return model.predict(predictors(c.frame))


def held_out_slack(c: Case, at: Mode, theirs: Mode) -> float:
    """How far the two engines' criteria may differ at one smoothing.

    Every held-out predictor moves with the full fit it starts from; each full fit lies its own
    Newton step from the mode, so the two predictors differ by at most the sum of the two steps,
    plus the rounding of an exact solve, ``64 kappa U`` times the predictor.  Each pair's
    deviance moves by its slope ``|D'| = 2 w |mu - y|`` times that.
    """
    held = engine.evaluate(c.problem, c.nb, c.rho, gradient=False).heldout
    apart = np.max(np.abs(at.step)) + np.max(np.abs(theirs.step))
    slack = apart + 64.0 * max(at.kappa, theirs.kappa) * U * np.maximum(1.0, np.abs(held.predictor))
    return float(np.sum(deviance_slope(c.problem, held) * slack))


def library_mode(c: Case) -> Mode:
    """The engine's own full fit at mgcv's smoothing, the one its criterion starts from."""
    fit = engine.solve_mode(c.problem, c.rho)
    eta = linear_predictor(c.problem.dm, fit.result, c.problem.offset)
    return mode(c, np.linalg.lstsq(c.X, eta, rcond=None)[0])


def weight_drift(at: Mode, theirs: Mode, family: str) -> float:
    """``eps`` with each row's working weight in one fit within ``(1 +- eps)`` of the other's.

    The Gaussian weights are the prior weights in both; a Poisson working weight is ``w mu``,
    so its ratio is ``exp`` of the predictor gap, at most the two Newton steps.
    """
    if family == "gaussian":
        return 0.0
    return float(np.expm1(np.max(np.abs(at.step)) + np.max(np.abs(theirs.step))))


# ── the reference is the one these tests rebuild ─────────────────────────────


@pytest.mark.parametrize("family", FAMILIES)
def test_fixture_is_mgcv_on_this_design(family):
    """mgcv 1.9.3 under R 4.5, and the library still builds the design mgcv was given.

    Each column sum is a sum of ``n`` rounded terms, within ``n U`` of the sum of their sizes.
    """
    pinned = fixture()[family]
    assert fixture()["metadata"]["mgcv_version"] == "1.9.3"
    assert fixture()["metadata"]["r_version"].startswith("R version 4.5")
    c = case(family, "rows")
    assert list(c.problem.names) == pinned["lambda_names"]
    now = fingerprint(c.X, c.penalties)
    size = N_ROWS * U * np.abs(c.X).sum(axis=0)
    for name in ("column_sums", "column_squares"):
        gap = np.abs(np.subtract(now[name], pinned["design"][name]))
        assert np.all(gap <= size * np.maximum(1.0, np.abs(c.X).max(axis=0))), (
            f"the library's {family} design no longer matches the one mgcv fitted ({name}); "
            "regenerate the fixture with the commands in tests/oracles/ncv_mgcv_reference.R"
        )
    np.testing.assert_allclose(now["penalty_norms"], pinned["design"]["penalty_norms"], rtol=64 * U)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("family", FAMILIES)
def test_neighbourhoods_mgcv_was_given_are_the_librarys(family, form):
    """The public hold-out arguments build exactly the removed and scored rows mgcv fitted."""
    nb = neighbourhoods_of(form, rows_of(family))
    assert mgcv_nei(nb) == fixture()[family]["forms"][form]["nei"]


# ── rows 2, 3 and 5: the criterion at mgcv's chosen smoothing ────────────────


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("family", FAMILIES)
def test_criterion_at_mgcvs_smoothing_is_mgcvs_score(family, form):
    """The summed held-out deviance with prior weights is mgcv's NCV score, unscaled.

    Gaussian: both leave-out fits exact.  Poisson: both one Newton step from the full fit.  The
    bar is what the two full fits' own Newton steps and an exact solve's rounding move the
    held-out deviance by (``held_out_slack``); measured gaps are at most 5e-15 relative.
    """
    c = case(family, form)
    value = engine.evaluate(c.problem, c.nb, c.rho, gradient=False).value
    bar = held_out_slack(c, library_mode(c), mode(c, np.asarray(c.mgcv["coefficients"])))
    assert abs(value - c.mgcv["score"]) <= bar, (value, c.mgcv["score"], bar)


@dataclass(frozen=True)
class OneSmoothing:
    """What the library's public fit at mgcv's smoothing reads, and how far each may lie from mgcv's.

    From the two full fits' Newton steps ``s`` and ``kappa``, the largest condition number of
    the two penalised Hessians and of the column-scaled design the coordinates are mapped
    through: the predictor within ``|s_ours| + |s_mgcv| + 64 kappa U``; the working weights
    within ``(1 +- eps)`` (``weight_drift``), so each penalised Hessian lies between ``1 - eps``
    and ``1 + eps`` times the other in the Loewner order, which bounds the EDF
    ``P - tr(H^-1 S)`` by ``tr(H^-1 S) eps / (1 - eps)`` and each variance relatively by
    ``eps / (1 - eps)``; the Gaussian scale is the Pearson sum over ``n - EDF``, which moves
    with each residual and with the EDF.  Every rounding term carries ``64 kappa U``.
    """

    mean: np.ndarray
    mgcv_mean: np.ndarray
    mean_slack: np.ndarray
    edf: float
    edf_slack: float
    scale: float
    scale_rel: float
    se: np.ndarray
    se_rel: float


def one_smoothing(c: Case, model: SuperGLM, coefficients) -> OneSmoothing:
    ours, theirs = mode(c, coordinates(c, model)), mode(c, np.asarray(coefficients))
    scaled = c.X / np.linalg.norm(c.X, axis=0)
    rounding = 64.0 * max(ours.kappa, theirs.kappa, np.linalg.cond(scaled)) * U
    apart = np.max(np.abs(ours.step)) + np.max(np.abs(theirs.step))
    eta_slack = apart + rounding * np.maximum(1.0, np.abs(theirs.eta))
    mgcv_mean, mean_slack = theirs.eta, eta_slack
    if c.family == "poisson":
        mgcv_mean = np.exp(theirs.eta)
        mean_slack = np.expm1(eta_slack) * mgcv_mean
    eps = weight_drift(ours, theirs, c.family)
    width, edf = c.problem.width, model.result.effective_df
    scale_rel = 0.0
    if c.family == "gaussian":
        residual = c.y - mgcv_mean
        pearson = np.sum(c.w * residual**2)
        scale_rel = 2.0 * np.sum(
            c.w * np.abs(residual) * eta_slack
        ) / pearson + rounding * width / (N_ROWS - edf)
    return OneSmoothing(
        mean=mean_of(model, c),
        mgcv_mean=mgcv_mean,
        mean_slack=mean_slack,
        edf=edf,
        edf_slack=(width - edf) * eps / (1.0 - eps) + rounding * width,
        scale=model.result.phi,
        scale_rel=scale_rel,
        se=np.sqrt(np.diag(slope_covariance(c, model))),
        se_rel=0.5 * (eps / (1.0 - eps) + scale_rel) + rounding,
    )


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("family", FAMILIES)
def test_fit_at_mgcvs_smoothing_reads_as_mgcvs(family, form):
    """Fitted values, total EDF, scale and the Bayesian slope standard errors at mgcv's smoothing.

    Against mgcv's refit at its chosen ``sp`` with ``nei$jackknife = FALSE``, whose covariance
    is the ordinary Bayesian one; bars as ``OneSmoothing`` says.
    """
    c = case(family, form)
    got = one_smoothing(c, fit_at(family, form), c.mgcv["coefficients_bayesian"])
    assert np.all(np.abs(got.mean - got.mgcv_mean) <= got.mean_slack)
    assert abs(got.edf - c.mgcv["edf_bayesian"]) <= got.edf_slack
    if family == "gaussian":
        assert abs(got.scale / c.mgcv["scale_bayesian"] - 1.0) <= got.scale_rel
    else:
        assert got.scale == c.mgcv["scale_bayesian"] == 1.0
    np.testing.assert_allclose(got.se, c.mgcv["se_bayesian"][1:], rtol=got.se_rel, atol=0)


@pytest.mark.parametrize("form", ("groups", "folds", "forward"))
@pytest.mark.parametrize("family", FAMILIES)
def test_standard_errors_on_grouped_neighbourhoods_are_mgcvs_default(family, form):
    """Row 7: where each neighbourhood scores a block of rows, mgcv's default ``Vp`` is the library's.

    mgcv's default covariance under ``nei`` accounts for the neighbourhood structure (``?NCV``);
    on these forms it is the ordinary Bayesian one (on single rows and on a row with its
    neighbours it is not, which these fixtures record and no test asserts), and the library
    publishes that one.  Against mgcv's free fit, whose ``Vp`` this is; bars as ``OneSmoothing``.
    """
    c = case(family, form)
    got = one_smoothing(c, fit_at(family, form), c.mgcv["coefficients"])
    np.testing.assert_allclose(got.se, c.mgcv["se"][1:], rtol=got.se_rel, atol=0)
    assert abs(got.edf - c.mgcv["edf"]) <= got.edf_slack
    assert abs(got.scale / c.mgcv["scale"] - 1.0) <= got.scale_rel


@pytest.mark.parametrize("family", FAMILIES)
def test_neighbour_heldout_predictions_are_mgcvs_fitted_values(family):
    """With one scored row per neighbourhood and its neighbours removed, mgcv 1.9.3's ``fitted``
    holds each row's held-out prediction (observed, not documented); the library's are the same.

    Bar: the two full fits' Newton steps plus an exact solve's rounding, on the predictor.
    """
    c = case(family, "neighbours")
    held = engine.evaluate(c.problem, c.nb, c.rho, gradient=False).heldout
    ours, theirs = library_mode(c), mode(c, np.asarray(c.mgcv["coefficients"]))
    apart = np.max(np.abs(ours.step)) + np.max(np.abs(theirs.step))
    slack = apart + 64.0 * max(ours.kappa, theirs.kappa) * U * np.maximum(
        1.0, np.abs(held.predictor)
    )
    theirs_mean = np.asarray(c.mgcv["fitted"])[held.row]
    if family == "poisson":
        slack = np.expm1(slack) * theirs_mean
    assert np.all(np.abs(held.mean - theirs_mean) <= slack)


# ── rows 2 and 3: the public search reaches mgcv's ───────────────────────────


@dataclass(frozen=True)
class Stops:
    """How far the library's and mgcv's first-order stops may lie apart, in ``log lambda``.

    ``reach`` is ``|H_V^-1| (|g_ours| + |g_mgcv|)``, each stop's Newton step to the criterion's
    minimum in absolute value, from the slopes measured at each stop and the criterion's
    curvature ``H_V`` (central differences of the engine's slopes ``h`` either side of mgcv's
    stop).  ``drop`` bounds how far either stop's criterion lies above the minimum,
    ``g' H_V^-1 g / 2`` with the larger ``g``.
    """

    reach: np.ndarray
    drop: float


H_RHO = 1e-3


@cache
def stops(family: str, form: str) -> Stops:
    c, model = case(family, form), ncv_fit(family, form)
    slopes = [
        [
            engine.evaluate(c.problem, c.nb, c.rho + s * H_RHO * e, heldout=False).gradient
            for s in (-1, 1)
        ]
        for e in np.eye(len(c.rho))
    ]
    curvature = np.array([(high - low) / (2 * H_RHO) for low, high in slopes])
    curvature = (curvature + curvature.T) / 2
    assert np.all(np.linalg.eigvalsh(curvature) > 0), "the criterion is not convex at mgcv's stop"
    inverse = np.linalg.inv(curvature)
    ours = np.abs(model.ncv_diagnostics()["gradient"])
    theirs = np.abs(c.mgcv["outer_gradient"])
    reach = np.abs(inverse) @ (ours + theirs)
    drop = 0.5 * max(g @ np.abs(inverse) @ g for g in (ours, theirs))
    return Stops(reach, float(drop))


def slopes_in_rho(family: str, form: str, read) -> np.ndarray:
    """Central differences of ``read(fit)`` in each ``log lambda``, ``H_RHO`` either side, as absolute values."""
    columns = []
    for j in range(2):
        shift = np.zeros(2)
        shift[j] = H_RHO
        high, low = (read(fit_at(family, form, tuple(s * shift))) for s in (1, -1))
        columns.append(np.abs(np.asarray(high) - np.asarray(low)) / (2 * H_RHO))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("family", FAMILIES)
def test_fit_ncv_reaches_mgcvs_fit(family, form):
    """The public ``fit_ncv`` at its defaults and mgcv's own search at its defaults reach one fit.

    Each search stops at a first-order test relative to the criterion (the library's
    ``ncv_tol = 1e-6``; mgcv's ``conv.tol = 1e-6``, ``?gam.control``).  The criterion is flat at
    the minimum, so the smoothing parameters themselves are not compared; the criterion is
    compared within ``Stops.drop`` and EDF and the fitted values within the two stops' reach
    mapped through their slopes in ``log lambda``, each doubled for the quadratic model's own
    error, plus the same quantity's gap at one smoothing (bars of the tests above).  Measured:
    criterion within 1e-10 relative, EDF within 4e-4, fitted values within 1e-4.
    """
    c, model, reach = case(family, form), ncv_fit(family, form), stops(family, form)
    report = model.ncv_diagnostics()
    assert report["converged"] and report["search_converged"]
    theirs = mode(c, np.asarray(c.mgcv["coefficients"]))
    at_mgcv = held_out_slack(c, library_mode(c), theirs)
    assert abs(report["criterion"] - c.mgcv["score"]) <= 2 * reach.drop + at_mgcv

    pairs = int(np.count_nonzero(c.w[c.nb.scored_rows()] > 0))
    assert report["criterion_mean"] == pytest.approx(report["criterion"] / pairs, rel=4 * U)

    fixed = one_smoothing(c, fit_at(family, form), c.mgcv["coefficients"])
    edf_slope = slopes_in_rho(family, form, lambda fit: fit.result.effective_df)
    edf_bar = 2 * edf_slope @ reach.reach + fixed.edf_slack
    assert abs(model.result.effective_df - c.mgcv["edf"]) <= edf_bar

    mean_slope = slopes_in_rho(family, form, lambda fit: mean_of(fit, c))
    gap = np.abs(mean_of(model, c) - fixed.mgcv_mean)
    assert np.all(gap <= 2 * mean_slope @ reach.reach + fixed.mean_slack)
