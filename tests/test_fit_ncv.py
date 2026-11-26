"""``SuperGLM.fit_ncv`` end to end: what a user reads of a fit, its refusals and its disclosures.

References.  The held-out results are compared with fits made without each neighbourhood's
rows: library refits on the remaining rows for a Gaussian model, and for a Poisson model the
Newton step assembled densely from the kept rows alone, taken from the published fit's own
public predictions (``tests.test_ncv_engine.KeptRows``).  The published model is compared with
the public ``fit`` on a clone that declares the chosen smoothing.  Events a search cannot be
relied on to reach (a stop at its step limit aside) are forced: a final fit short of its
certificate, a smoothing parameter on a bound, a search with no criterion, and coefficient
fits at the limit of float64 on either side of the acceptance bar.

Tolerances.  ``U`` is the unit roundoff and ``kappa`` the 2-norm condition number of the
Jacobi-scaled penalised Hessian a comparison goes through, computed here.  Two exact
computations agree within ``64 kappa U`` (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., 2002, section 10.1).  A Poisson fit stops at a certificate, so it lies
one Newton step ``H^-1 g`` of its own residual score ``g`` from the mode; two such fits are
compared within the two steps, computed from each fit.  Two first-order stops of a
one-parameter search lie within ``TOL V / V''`` of the stationary point each
(``tests.test_ncv_search.stop_reach``).
"""

import pickle
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit

from superglm import Constraint, FactorSmooth, RandomEffect, SuperGLM
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.links import LogLink
from superglm.model import fit_method, ncv_ops
from superglm.ncv import engine, from_groups, from_neighbours, from_splitter, single_rows
from superglm.ncv import search as search_module
from superglm.ncv.engine import LeaveOutRankError
from superglm.ncv.result import TRIAL_FAILURES
from superglm.ncv.search import search
from superglm.solvers import pirls
from superglm.solvers.irls_direct import _fit as direct_fit
from superglm.types import LambdaPolicy
from tests.test_ncv_engine import (
    AIM,
    KeptRows,
    U,
    assert_pairs_equal,
    at_the_aim,
    mean_and_deviance,
    predictor_slack,
    problem_of,
    refit_pairs,
    solver_ending,
    solver_short,
)
from tests.test_ncv_engine_rank_cache import twin
from tests.test_ncv_search import Case, NarrowBox, stop_reach

TOL = engine.DEFAULT_NCV_TOL
BAR = engine.ncv_mode_bar(TOL)
NONE = np.zeros(0)
FORMS = ("rows", "groups", "cv", "neighbours", "forward", "repeated")
FEW = "fit_ncv has . neighbourhoods; with fewer than 10"


# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Book:
    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    offset: np.ndarray
    labels: np.ndarray


@cache
def book(family, exposure="offset", n=300) -> Book:
    """Two curves, a numeric column far from zero and a factor; weights, an offset, 15 groups.

    Every eleventh row has a weight of zero.  ``exposure="weight"`` makes the Poisson response
    a rate weighted by its exposure.
    """
    rng = np.random.default_rng(41)
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
    weights = np.where(np.arange(n) % 11 == 3, 0.0, weights)
    labels = np.array([f"g{k:02d}" for k in rng.permutation(np.arange(n) % 15)])
    return Book(frame, y, weights, offset, labels)


def terms(**changed):
    return {
        "x1": Spline(kind="ps", n_knots=8),
        "x2": Spline(kind="ps", n_knots=8),
        "z": Numeric(),
        "f": Categorical(),
        **changed,
    }


def model_of(family="gaussian", features=None, **settings):
    settings.setdefault("selection_penalty", 0)
    return SuperGLM(
        features=terms() if features is None else features,
        family=family,
        spline_penalty=1.0,
        **settings,
    )


def held_out(form, b: Book) -> dict:
    """``fit_ncv``'s hold-out arguments for a form."""
    n = len(b.y)
    return {
        "rows": {},
        "groups": {"groups": b.labels},
        "cv": {"cv": KFold(12, shuffle=True, random_state=1)},
        "forward": {"cv": TimeSeriesSplit(5)},
        "repeated": {"cv": RepeatedKFold(n_splits=5, n_repeats=2, random_state=2)},
        "neighbours": {"neighbours": [np.arange(max(i - 3, 0), min(i + 4, n)) for i in range(n)]},
    }[form]


def structure(form, b: Book):
    """The neighbourhoods of a form, for the references."""
    n, arguments = len(b.y), held_out(form, b)
    if "cv" in arguments:
        return from_splitter(arguments["cv"], b.frame, b.y)
    if "neighbours" in arguments:
        return from_neighbours(arguments["neighbours"], n)
    return from_groups(b.labels, n) if arguments else single_rows(n)


def run(model, b: Book, frame=None, **arguments):
    """``fit_ncv`` on a book; the model and the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_ncv(
            b.frame if frame is None else frame,
            b.y,
            sample_weight=b.weights,
            offset=b.offset,
            **arguments,
        )
    return model, caught


def messages(caught, category=Warning) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, category)]


@cache
def fitted(family, form, exposure="offset", semantics="prior"):
    """The book's model fitted by ``fit_ncv`` over a form's neighbourhoods.

    A Gaussian search meets its first-order test: its inner fit is one exact solve.  A
    Poisson search may end on its line search a little short of it, which depends on the
    rounding of its inner fits; that fit is disclosed, published and compared at the
    smoothing it reached like the others.
    """
    b = book(family, exposure)
    model, caught = run(model_of(family, weight_semantics=semantics), b, **held_out(form, b))
    assert family == "poisson" or not messages(caught, ConvergenceWarning)
    return model


def chosen(model) -> dict:
    return model.ncv_diagnostics()["lambdas"]


def at_chosen(model, b: Book):
    """The engine's problem on the published design, every smoothing parameter held as chosen."""
    return problem_of(model, b.y, b.weights, b.offset, fixed_lambdas=chosen(model))


def fit_at_chosen(model, b: Book, frame=None, lambdas=None, **settings):
    """The public ``fit`` on a clone that declares the chosen smoothing, ``lambdas`` replacing it."""
    oracle = model._clone_without_features(set(), smoothing_resolved=True)
    oracle.lambda2 = {**oracle.lambda2, **(lambdas or {})}
    frame = b.frame if frame is None else frame
    return oracle.fit(frame, b.y, sample_weight=b.weights, offset=b.offset, **settings)


def predictor(model, b: Book, frame=None) -> np.ndarray:
    """The linear predictor of the public predictions, offset included."""
    mean = model.predict(b.frame if frame is None else frame, offset=b.offset)
    return np.log(mean) if isinstance(model._link, LogLink) else mean


def heldout_pairs(model) -> pd.DataFrame:
    """The held-out table keyed by (neighbourhood, row), with the predictor of its held-out mean."""
    table = model.ncv_heldout()
    eta = np.log(table.heldout) if isinstance(model._link, LogLink) else table.heldout
    pairs = pd.DataFrame(
        {
            "neighbourhood": table.neighbourhood,
            "row": table.row,
            "predictor": eta,
            "mean": table.heldout,
            "deviance": table.heldout_deviance,
        }
    ).set_index(["neighbourhood", "row"])
    assert pairs.index.is_unique
    return pairs


@pytest.fixture
def solves(monkeypatch):
    """Every coefficient solve of the library, counted where the solvers themselves begin."""
    counts: Counter = Counter()
    direct, proximal = direct_fit._fit_irls_direct_once, pirls._fit_pirls_inner

    def counted_direct(*args, **kwargs):
        counts["direct"] += 1
        return direct(*args, **kwargs)

    def counted_proximal(*args, **kwargs):
        counts["proximal"] += 1
        return proximal(*args, **kwargs)

    monkeypatch.setattr(direct_fit, "_fit_irls_direct_once", counted_direct)
    monkeypatch.setattr(pirls, "_fit_pirls_inner", counted_proximal)
    return counts


# ── rows 2, 3 and 6: the held-out results are fits without each neighbourhood ─


@pytest.mark.parametrize("form", FORMS)
def test_gaussian_heldout_results_are_refits_without_each_neighbourhood(form):
    """Each scored pair's prediction and deviance, and the criterion, against library refits."""
    b, model = book("gaussian"), fitted("gaussian", form)
    problem, nb = at_chosen(model, b), structure(form, b)
    want = refit_pairs(problem, nb, NONE)
    _, _, kappa = KeptRows(problem, NONE).one_step(nb)
    got = heldout_pairs(model)
    assert_pairs_equal(problem, got, want, predictor_slack(kappa, want.predictor.to_numpy()))
    report = model.ncv_diagnostics()
    assert report["criterion"] == pytest.approx(want.deviance.sum(), rel=64 * kappa * U)
    assert report["criterion"] == model.ncv_heldout().heldout_deviance.to_numpy().sum()


@pytest.mark.parametrize("exposure", ["offset", "weight"])
@pytest.mark.parametrize("form", FORMS)
def test_poisson_heldout_results_are_one_newton_step_without_each_neighbourhood(form, exposure):
    """Each scored pair against the step of its kept rows' own score and Hessian.

    The step starts from the published fit as its public predictions give it.  The engine
    reads the removed rows' score and the reference the kept rows' less the penalty's; they
    differ by the published fit's residual score, whose step is in the slack.
    """
    b, model = book("poisson", exposure), fitted("poisson", form, exposure)
    problem, nb = at_chosen(model, b), structure(form, b)
    reference = KeptRows(problem, NONE, at=(predictor(model, b), model.result.beta))
    want, residual, kappa = reference.one_step(nb)
    table = model.ncv_heldout()
    assert np.array_equal(table.row, nb.scored_rows())
    slack = residual + predictor_slack(kappa, want)
    gap = np.abs(np.log(table.heldout.to_numpy()) - want)
    assert np.all(gap <= slack), f"gap {gap.max():.3e} against {slack.min():.3e}"
    rows = table.row.to_numpy()
    mean, deviance = mean_and_deviance(problem, rows, want)
    # D' = 2 w (mu - y) and D'' = 2 w mu, each over the predictor's slack.
    slope = 2.0 * b.weights[rows] * (np.abs(mean - b.y[rows]) + mean * slack)
    resolved = slope * slack + 64.0 * U * deviance
    assert np.all(np.abs(table.heldout_deviance.to_numpy() - deviance) <= resolved)
    criterion = model.ncv_diagnostics()["criterion"]
    assert abs(criterion - deviance.sum()) <= resolved.sum()
    assert criterion == table.heldout_deviance.to_numpy().sum()


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("form", FORMS)
def test_heldout_table_names_each_pair_and_carries_the_published_fit(family, form):
    """The table's keys, labels, weights and responses are the caller's, its fitted mean ``predict``'s."""
    b, model = book(family), fitted(family, form)
    table, nb = model.ncv_heldout(), structure(form, b)
    rows = table.row.to_numpy()
    assert len(table) == len(nb.scored_rows())
    assert np.array_equal(table.weight, b.weights[rows]) and np.array_equal(table.actual, b.y[rows])
    # One rounding a coefficient on either path, and the public path's own centring.
    public = model.predict(b.frame, offset=b.offset)[rows]
    np.testing.assert_allclose(table.fitted, public, rtol=64 * model._dm.p * U)
    assert np.all(table.heldout_deviance[table.weight == 0.0] == 0.0)
    assert np.all(np.isfinite(table.heldout))
    if form == "groups":
        assert np.array_equal(table.label, b.labels[rows])
        assert np.array_equal(np.unique(b.labels)[table.neighbourhood], table.label)
    else:
        assert np.array_equal(table.label, table.neighbourhood)
    if form in ("cv", "forward", "repeated"):
        splits = held_out(form, b)["cv"].split(b.frame, b.y)
        for k, (_, test) in enumerate(splits):
            assert table.row[table.neighbourhood == k].tolist() == sorted(test.tolist())
    if form in ("rows", "neighbours"):
        assert np.array_equal(table.neighbourhood, rows)
    twice = table.groupby("row").heldout.agg(["size", "min", "max"])
    if form == "repeated":
        assert len(table) == 2 * len(b.y) and twice["size"].eq(2).all()
        assert (twice["max"] > twice["min"]).all()
    else:
        assert twice["size"].eq(1).all()


@pytest.mark.parametrize(
    ("family", "form"), [("gaussian", "groups"), ("poisson", "rows"), ("poisson", "forward")]
)
def test_published_smoothing_and_counts_are_those_of_the_search(family, form):
    """A search run by hand on the declared design returns what the fit publishes.

    The same computation on the same rows, so the smoothing, the counts and the path are
    equal and not merely close, whichever stop the search ends on.  The criterion is read
    again at the published fit, in the basis the chosen smoothing gives, where a fit
    certified to the same bar gives it within that bar.
    """
    b, model = book(family), fitted(family, form)
    declared = model_of(family).fit(b.frame, b.y, sample_weight=b.weights, offset=b.offset)
    problem = problem_of(
        declared,
        b.y,
        b.weights,
        b.offset,
        mode_bar=BAR,
        pirls_tol=declared._tol,
        max_pirls_iter=declared._max_iter,
    )
    want = search(
        problem, structure(form, b), configured=dict.fromkeys(problem.names, 1.0), tol=TOL
    ).result
    assert want.converged or family == "poisson"
    got = model.ncv_diagnostics()
    assert got["lambdas"] == want.lambdas == model._reml_lambdas
    assert got["start_lambdas"] == want.start_lambdas
    assert got["lambda_history"] == want.lambda_history
    assert got["criterion_history"] == want.objective_history
    assert got["gradient"] == want.gradient.tolist()
    assert (got["n_iter"], got["n_evaluations"]) == (want.n_reml_iter, want.n_evaluations)
    assert got["infeasible_trials"] == want.infeasible_trials
    assert got["infeasible_reasons"] == want.infeasible_reasons
    assert got["termination_reason"] == want.termination_reason
    assert got["work"] == want.work and got["work"]["inner_solves"] > want.n_evaluations
    assert (got["lambda_floor"], got["lambda_ceiling"]) == (
        want.lambda_box.floor,
        want.lambda_box.ceiling,
    )
    assert got["at_bound"] == list(want.at_bound) and got["bound_pressure"] == []
    assert got["criterion"] == pytest.approx(want.objective, rel=BAR)
    assert got["converged"] == got["search_converged"] == bool(want.converged)
    assert got["coefficients_converged"] and got["coefficient_termination"] is None
    assert got["method"] == "fit_ncv"
    assert got["heldout_available"] and got["heldout_failure"] is None


@pytest.mark.parametrize(("exposure", "form"), [("offset", "neighbours"), ("weight", "cv")])
def test_poisson_search_meets_its_test_where_fits_to_one_bar_stalled_it(exposure, form):
    """Two searches of the 300-row book, at default settings, stop on their first-order test.

    With every inner fit run only to a hundredth of the tolerance, both ended on the line
    search after 32 evaluations, their slopes 1.5 and 1.6 times the stopping bar: two fits
    at one smoothing then differ in the criterion by 1e-9 of it, the size of the last gains.
    """
    b = book("poisson", exposure)
    model, caught = run(model_of("poisson"), b, **held_out(form, b))
    assert not messages(caught, ConvergenceWarning)
    report = model.ncv_diagnostics()
    assert report["termination_reason"] == "gradient"
    assert report["converged"] and report["search_converged"] and model.result.converged
    assert report["n_evaluations"] <= 20 and report["infeasible_trials"] == 0


# ── row 5: the diagnostics ────────────────────────────────────────────────────


@pytest.mark.parametrize("form", FORMS)
def test_diagnostics_count_the_neighbourhoods_as_the_caller_gave_them(form):
    """The neighbourhood summary recounted from the hold-out arguments themselves."""
    b = book("gaussian")
    n, arguments = len(b.y), held_out(form, b)
    if "cv" in arguments:
        splits = list(arguments["cv"].split(b.frame, b.y))
        removed = [n - len(train) for train, _ in splits]
        scored = np.concatenate([test for _, test in splits])
    elif "neighbours" in arguments:
        removed = [len(set(rows) | {i}) for i, rows in enumerate(arguments["neighbours"])]
        scored = np.arange(n)
    else:
        removed = np.unique(b.labels, return_counts=True)[1] if arguments else np.ones(n)
        scored = np.arange(n)
    times = np.bincount(scored, minlength=n)
    assert fitted("gaussian", form).ncv_diagnostics()["neighbourhoods"] == {
        "form": {"forward": "cv", "repeated": "cv"}.get(form, form),
        "splitter": type(arguments["cv"]).__name__ if "cv" in arguments else None,
        "count": len(removed),
        "size_min": min(removed),
        "size_median": float(np.median(removed)),
        "size_max": max(removed),
        "pairs_scored": len(scored),
        "rows_scored": int(np.count_nonzero(times)),
        "rows_scored_more_than_once": int(np.count_nonzero(times > 1)),
    }


@pytest.mark.parametrize("form", ["groups", "repeated", "forward"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_criterion_mean_is_over_the_likelihood_size_of_the_scored_pairs(semantics, form):
    """The sum of the scored pairs' weights under frequency weights, their weighted count under prior.

    A row scored twice counts twice, a row never scored not at all, and a row of zero weight
    in neither reading.
    """
    b, model = book("gaussian"), fitted("gaussian", form, semantics=semantics)
    scored = np.concatenate(
        [test for _, test in held_out(form, b)["cv"].split(b.frame, b.y)]
        if form != "groups"
        else [np.arange(len(b.y))]
    )
    weights = b.weights[scored]
    size = weights.sum() if semantics == "frequency" else np.count_nonzero(weights)
    assert size not in (len(b.y), len(scored))
    report = model.ncv_diagnostics()
    assert report["criterion_mean"] == pytest.approx(report["criterion"] / size, rel=8 * U)


# ── row 7: the published model is ``fit`` at the chosen smoothing ─────────────

DECLARATIONS = {
    "one penalty a term": {},
    "a second penalty on the null space": {
        "features": terms(x2=Spline(kind="ps", n_knots=8, select=True))
    },
    "a penalty held fixed": {
        "features": terms(x2=Spline(kind="ps", n_knots=8, lambda_policy=LambdaPolicy.fixed(3.0)))
    },
    "the QR solver route": {"direct_solve": "qr"},
    "a duplicated column": {"features": terms(twin=Numeric())},
}


def hessian_condition(model, b: Book, frame=None) -> float:
    """The condition of the Jacobi-scaled penalised Hessian at the published fit, null directions aside."""
    at = predictor(model, b, frame), model.result.beta
    reference = KeptRows(at_chosen(model, b), NONE, at=at)
    hessian = reference.Z.T @ (reference.curvature[:, None] * reference.Z) + reference.S
    scale = 1.0 / np.sqrt(np.diag(hessian))
    spectrum = np.linalg.eigvalsh(hessian * scale[:, None] * scale[None, :])
    kept = spectrum[spectrum > 64 * len(spectrum) * U * spectrum.max()]
    return float(kept.max() / kept.min())


def readings(model, frame, b: Book) -> dict[str, np.ndarray]:
    """What a user reads of a fitted model, each as an array (missing entries as NaN)."""
    fresh = frame.iloc[::-1].reset_index(drop=True)
    result = model.result
    covariance, _ = model._coef_covariance
    contrast = np.cos(np.arange(len(result.beta)))
    read = {
        "predictions": model.predict(frame, offset=b.offset),
        "predictions on other rows": model.predict(fresh, offset=b.offset[::-1]),
        "total EDF": np.array([result.effective_df]),
        "term EDF": np.array([model._group_edf[name] for name in sorted(model._group_edf)]),
        "scale": np.array([result.phi]),
        "likelihood scale": np.array([result.phi_likelihood]),
        "covariance contrast": np.array([contrast @ covariance @ contrast]),
        "deviance": np.array([result.deviance]),
    }
    for name in model._specs:
        read[f"standard errors of {name}"] = model.term_inference(name).se_log_relativity
    return read


def assert_reads_alike(got: dict, want: dict, rel: float) -> None:
    assert got.keys() == want.keys()
    for name, value in got.items():
        mine, theirs = np.asarray(value, dtype=float), np.asarray(want[name], dtype=float)
        assert np.array_equal(np.isnan(mine), np.isnan(theirs)), name
        size = np.nanmax(np.abs(theirs), initial=0.0)
        np.testing.assert_allclose(mine, theirs, rtol=rel, atol=rel * size, err_msg=name)


@pytest.mark.parametrize("declared", DECLARATIONS)
def test_gaussian_fit_reads_as_fit_at_the_chosen_smoothing(declared):
    """Predictions with an offset, standard errors and which coefficients have one, EDF and both scales."""
    b = book("gaussian")
    frame = b.frame.assign(twin=b.frame.z) if declared == "a duplicated column" else b.frame
    model, caught = run(model_of(**DECLARATIONS[declared]), b, frame=frame, groups=b.labels)
    assert not messages(caught, ConvergenceWarning)
    oracle = fit_at_chosen(model, b, frame=frame)
    assert not hasattr(oracle, "ncv") and oracle._reml_result is None
    assert_reads_alike(
        readings(model, frame, b),
        readings(oracle, frame, b),
        64 * hessian_condition(model, b, frame) * U,
    )
    result, theirs = model.result, oracle.result
    assert np.array_equal(
        result.rank_info.coefficient_estimable(), theirs.rank_info.coefficient_estimable()
    )
    assert (result.phi_method, result.phi_likelihood_method) == (
        theirs.phi_method,
        theirs.phi_likelihood_method,
    )
    assert result.direct_backend == theirs.direct_backend
    assert result.direct_backend == ("qr" if declared == "the QR solver route" else "gram")
    assert model._runtime_canonical_state["intercept_shift"] != 0.0
    assert result.reml_slope_decomposition is None
    if declared == "a penalty held fixed":
        assert (
            chosen(model)["x2:wiggle"] == 3.0
            and "x2:wiggle" not in (model.ncv_diagnostics()["start_lambdas"])
        )
    if declared == "a second penalty on the null space":
        assert {"x2:null", "x2:wiggle"} <= chosen(model).keys()
    if declared == "a duplicated column":
        assert not result.rank_info.coefficient_estimable().all()
        assert model._group_edf["twin"] == 0.0


def test_poisson_fit_reads_as_fit_at_the_chosen_smoothing():
    """Each of the two fits lies one Newton step of its own residual score from the mode.

    A step estimates that distance to within its own share of the working weights'
    conditioning, so the predictors agree within the two steps so widened, and what is read
    off the Hessian within the change those steps make to the working weights, times its
    conditioning.  ``fit`` is asked for a tolerance at which its last step is negligible.
    """
    b, model = book("poisson"), fitted("poisson", "groups")
    oracle = fit_at_chosen(model, b, tol=1e-12)
    steps, kappa = [], hessian_condition(model, b)
    for one in (model, oracle):
        reference = KeptRows(
            problem_of(one, b.y, b.weights, b.offset, fixed_lambdas=chosen(model)),
            NONE,
            at=(predictor(one, b), one.result.beta),
        )
        solve, _, _ = reference.solver(np.zeros(0, dtype=int))
        steps.append(np.abs(reference.Z @ solve(reference.full_score)))
    mine, theirs = predictor(model, b), predictor(oracle, b)
    both = steps[0] + steps[1]
    slack = both * (1.0 + kappa * both.max()) + predictor_slack(kappa, theirs)
    assert np.all(np.abs(mine - theirs) <= slack)
    got, want = readings(model, b.frame, b), readings(oracle, b.frame, b)
    for name in ("predictions", "predictions on other rows"):
        del got[name], want[name]
    assert_reads_alike(got, want, 8 * kappa * slack.max())
    assert 8 * kappa * slack.max() < 1e-6
    assert model.result.phi == oracle.result.phi == 1.0
    assert model.result.phi_likelihood == oracle.result.phi_likelihood == 1.0


# ── row 5: what did not converge is disclosed, and the fit returned ───────────


def curve_book(family="gaussian", truth=lambda t: np.sin(6 * t), n=240) -> Book:
    """One curve and a numeric column in twelve groups, unweighted and without an offset."""
    rng = np.random.default_rng(127)
    frame = pd.DataFrame({"t": rng.uniform(size=n), "x": rng.normal(size=n)})
    signal = truth(frame.t.to_numpy()) + 0.5 * frame.x.to_numpy()
    y = signal + rng.normal(0.0, 0.2, n)
    if family == "poisson":
        y = rng.poisson(np.exp(signal)).astype(float)
    labels = np.array([f"g{k:02d}" for k in np.arange(n) % 12])
    return Book(frame, y, np.ones(n), np.zeros(n), labels)


def curve_model(family="gaussian", **settings):
    features = {"t": Spline(kind="ps", n_knots=6), "x": Numeric()}
    return model_of(family, features=features, **settings)


@cache
def curve_fit():
    b = curve_book()
    model, caught = run(curve_model(), b, groups=b.labels)
    assert not caught, messages(caught)
    return model


def test_search_stopped_at_its_step_limit_is_disclosed_and_its_fit_is_good():
    b = curve_book()
    model, caught = run(curve_model(), b, groups=b.labels, max_ncv_iter=1)
    (message,) = messages(caught, ConvergenceWarning)
    assert message.startswith(
        "fit_ncv did not converge: its smoothing search stopped after 1 step because it "
        "reached the max_ncv_iter limit. The model is returned at the lowest held-out deviance"
    )
    assert "Refit with a larger max_ncv_iter, passing lambda2_init=" in message
    assert "final coefficient fit" not in message
    report = model.ncv_diagnostics()
    assert (report["termination_reason"], report["n_iter"]) == ("max_iter", 1)
    assert not report["search_converged"] and not report["converged"]
    assert report["coefficients_converged"] and model.result.converged
    assert report["criterion"] < report["criterion_history"][0]
    assert report["criterion"] > curve_fit().ncv_diagnostics()["criterion"]
    assert report["heldout_available"] and len(model.ncv_heldout()) == len(b.y)
    oracle = fit_at_chosen(model, b)
    np.testing.assert_allclose(
        model.predict(b.frame), oracle.predict(b.frame), rtol=64 * hessian_condition(model, b) * U
    )


def test_final_fit_short_of_its_test_is_disclosed_after_a_search_that_met_its_own(monkeypatch):
    """Every solve of the final coefficient fit is made to end at its iteration limit, uncertified.

    Three solves: to the aim, from where that stopped to the acceptance bar, and to that bar
    from the solver's own start.
    """
    solve = ncv_ops.fit_irls_direct
    solved = []

    def uncertified(**kwargs):
        result, inverse = solve(**kwargs)
        solved.append((kwargs["_mode_bar"], "beta_init" in kwargs))
        return replace(result, converged=False, termination_reason="max_iter"), inverse

    b, good = curve_book(), curve_fit().ncv_diagnostics()
    monkeypatch.setattr(ncv_ops, "fit_irls_direct", uncertified)
    model, caught = run(curve_model(), b, groups=b.labels)
    assert solved == [(AIM, True), (BAR, True), (BAR, False)]
    (message,) = messages(caught, ConvergenceWarning)
    steps = good["n_iter"]
    assert steps > 1 and message.startswith(
        f"fit_ncv did not converge: its smoothing search met its test after {steps} steps, but "
        "the final coefficient fit at that smoothing did not meet its convergence test (it "
        "stopped with termination_reason='max_iter')."
    )
    assert "without held-out results" in message and "larger max_pirls_iter" in message
    report = model.ncv_diagnostics()
    assert report["search_converged"] and report["termination_reason"] == "gradient"
    assert not report["coefficients_converged"] and not report["converged"]
    assert report["coefficient_termination"] == "max_iter" and not model.result.converged
    assert (report["heldout_available"], report["heldout_failure"]) == (False, "uncertified_mode")
    assert report["lambdas"] == good["lambdas"]
    assert report["criterion"] == pytest.approx(good["criterion"], rel=BAR)
    assert report["criterion_mean"] == pytest.approx(report["criterion"] / len(b.y), rel=8 * U)
    with pytest.raises(RuntimeError, match=r"ncv_heldout\(\) has no held-out results to read"):
        model.ncv_heldout()
    np.testing.assert_array_equal(model.predict(b.frame), curve_fit().predict(b.frame))


def test_final_fit_accepted_at_the_looser_bar_reads_as_converged(monkeypatch):
    """The final coefficient fit's solve to the aim is made to end short of it.

    It is continued from where it stopped and certified at the acceptance bar: a converged
    fit with its held-out results and nothing to disclose, one Newton step from the fit
    whose solve reached the aim.
    """
    b, good = curve_book(), curve_fit()
    log: list = []
    monkeypatch.setattr(ncv_ops, "fit_irls_direct", solver_short(at_the_aim, log))
    model, caught = run(curve_model(), b, groups=b.labels)
    assert not messages(caught)
    stopped, continued = log
    assert (stopped.bar, continued.bar) == (AIM, BAR) and continued.start is stopped.beta
    assert not stopped.certified and continued.certified
    report, reached = model.ncv_diagnostics(), good.ncv_diagnostics()
    assert report["converged"] and report["search_converged"] and report["coefficients_converged"]
    assert report["coefficient_termination"] is None and model.result.converged
    assert report["heldout_available"] and report["heldout_failure"] is None
    assert len(model.ncv_heldout()) == len(b.y)
    assert report["lambdas"] == reached["lambdas"]
    kappa = hessian_condition(good, b)
    assert report["criterion"] == pytest.approx(reached["criterion"], rel=64 * kappa * U)
    predicted = good.predict(b.frame)
    gap = np.abs(model.predict(b.frame) - predicted)
    assert np.all(gap <= 64 * kappa * U * np.maximum(1.0, np.abs(predicted)))


@pytest.mark.parametrize("side", ["floor", "ceiling"])
def test_smoothing_a_bound_holds_back_is_disclosed(monkeypatch, side):
    """A box that ends a hundredfold short of the chosen smoothing, on either side of it."""
    b, best = curve_book(), chosen(curve_fit())["t"]
    box = NarrowBox(low=100.0 * best) if side == "floor" else NarrowBox(high=best / 100.0)
    monkeypatch.setattr(search_module, "_fit_box", lambda problem: box)
    model, caught = run(curve_model(), b, groups=b.labels)
    (message,) = messages(caught, ConvergenceWarning)
    which = "smallest" if side == "floor" else "largest"
    assert message.startswith(
        f"fit_ncv's smoothing search ended at a limit of its range on 't' at the {which} "
        "smoothing parameter the search allows (lambda = "
    )
    assert message.endswith(
        "The held-out deviance was still falling at the limit for 't', so the limit, not the "
        "data, set that smoothing."
    )
    report = model.ncv_diagnostics()
    assert report["at_bound"] == report["bound_pressure"] == [("t", side)]
    assert report["lambdas"]["t"] == pytest.approx(getattr(box, side), rel=1e-12)
    assert (report["lambda_floor"], report["lambda_ceiling"]) == (box.floor, box.ceiling)
    assert report["converged"] and report["criterion"] > curve_fit().ncv_diagnostics()["criterion"]


@pytest.mark.parametrize("side", ["floor", "ceiling"])
def test_smoothing_resting_on_a_bound_is_disclosed_without_a_claim_that_it_is_held(
    monkeypatch, side
):
    """The search is made to return a bound of its own box, with no slope out of it."""
    find = ncv_ops.search_smoothing

    def resting(problem, nb, **settings):
        outcome = find(problem, nb, **settings)
        box = outcome.result.lambda_box
        result = replace(
            outcome.result,
            lambdas={**outcome.result.lambdas, "t": getattr(box, side)},
            at_bound=(("t", side),),
            bound_pressure=(),
        )
        return replace(outcome, result=result)

    reference = curve_fit()
    monkeypatch.setattr(ncv_ops, "search_smoothing", resting)
    b = curve_book()
    model, caught = run(curve_model(), b, groups=b.labels)
    (message,) = messages(caught, ConvergenceWarning)
    which = "smallest" if side == "floor" else "largest"
    advice = (
        "nearly unpenalized and can follow noise"
        if side == "floor"
        else "(a straight line for a spline)"
    )
    assert f"on 't' at the {which} smoothing parameter the search allows" in message
    assert advice in message and "still falling" not in message
    report = model.ncv_diagnostics()
    assert report["at_bound"] == [("t", side)] and report["bound_pressure"] == []
    assert report["lambdas"]["t"] == report[f"lambda_{side}"]
    assert report["converged"]
    if side == "ceiling":
        assert model._group_edf["t"] == pytest.approx(1.0, abs=1e-6)
    else:
        assert model._group_edf["t"] > reference._group_edf["t"] + 1.0


def test_search_with_no_criterion_returns_a_fit_at_its_start_and_says_why():
    """One iteration a coefficient fit certifies no Poisson mode, so no trial has a criterion."""
    b = curve_book("poisson")
    model, caught = run(curve_model("poisson"), b, groups=b.labels, max_pirls_iter=1)
    (message,) = messages(caught, ConvergenceWarning)
    assert message.startswith(
        "fit_ncv chose no smoothing: its held-out deviance could not be computed at the 1 "
        f"smoothing setting it tried. At 1 of them, {TRIAL_FAILURES['uncertified_mode']}. The "
        "model is returned as a coefficient fit at the smoothing the search started from"
    )
    assert (
        "the final coefficient fit at that smoothing did not meet its convergence test" in message
    )
    assert message.endswith("Refit with a larger max_pirls_iter.")
    report = model.ncv_diagnostics()
    assert report["termination_reason"] == "no_finite_criterion"
    assert not (report["converged"] or report["search_converged"] or report["heldout_available"])
    assert not report["coefficients_converged"] and report["coefficient_termination"] == "max_iter"
    assert report["criterion"] is None and report["criterion_mean"] is None
    assert report["gradient"] is None and report["heldout_failure"] is None
    assert (
        report["infeasible_reasons"]
        == {"uncertified_mode": 1}
        == {"uncertified_mode": report["infeasible_trials"]}
    )
    assert report["lambdas"] == {"t": 1.0} == report["start_lambdas"]
    assert np.all(np.isfinite(model.predict(b.frame)))
    with pytest.raises(RuntimeError, match="no held-out results to read"):
        model.ncv_heldout_groups()


ADVICE = {
    "uncertified_mode": "Refit with a larger max_pirls_iter.",
    "clipped_prediction": "Check the responses and offsets of the held-out rows",
    "non_finite": "Check the responses, weights and offsets for extreme values.",
    "leave_out_rank": "Choose neighbourhoods that leave rows of every term in the fit",
    "rank_deficient_fit": "Drop a term that repeats what another term already carries.",
    "kept_rows_limit": (
        "Neighbourhoods need fits of their own at every smoothing when two terms carry the "
        "same information (a numeric term beside a spline of the same variable, or a spline "
        "of a column that does not vary): drop one of the two, or choose this model's "
        "smoothing with fit_reml."
    ),
    "leave_out_refused": "Choose neighbourhoods that divide the rows of every factor level",
}


@pytest.mark.parametrize("reason", list(TRIAL_FAILURES))
def test_failed_search_is_disclosed_in_the_words_of_each_reason_its_trials_gave(
    monkeypatch, reason
):
    """The search is made to end without a criterion, three trials failing for ``reason``.

    The model is then the coefficient fit at the smoothing the search started from.
    """
    find = ncv_ops.search_smoothing

    def failed(problem, nb, **settings):
        outcome = find(problem, nb, **settings)
        found = outcome.result
        result = replace(
            found,
            lambdas={**found.lambdas, **found.start_lambdas},
            objective=None,
            gradient=None,
            converged=False,
            termination_reason="no_finite_criterion",
            n_evaluations=3,
            infeasible_trials=3,
            infeasible_reasons={reason: 3},
            at_bound=(),
            bound_pressure=(),
        )
        return replace(outcome, result=result, fit=None)

    monkeypatch.setattr(ncv_ops, "search_smoothing", failed)
    b = curve_book()
    model, caught = run(curve_model(), b, groups=b.labels)
    (message,) = messages(caught, ConvergenceWarning)
    assert message.startswith(
        "fit_ncv chose no smoothing: its held-out deviance could not be computed at the 3 "
        f"smoothing settings it tried. At 3 of them, {TRIAL_FAILURES[reason]}. The model is "
        "returned as a coefficient fit at the smoothing the search started from "
        "(model.ncv_diagnostics()['lambdas']), which held-out prediction did not choose."
    )
    assert ADVICE.keys() == TRIAL_FAILURES.keys() and ADVICE[reason] in message
    report = model.ncv_diagnostics()
    assert not report["converged"] and not report["search_converged"]
    assert report["coefficients_converged"] and model.result.converged
    assert report["criterion"] is None and not report["heldout_available"]
    assert report["lambdas"] == report["start_lambdas"]
    oracle = fit_at_chosen(model, b)
    np.testing.assert_allclose(
        model.predict(b.frame), oracle.predict(b.frame), rtol=64 * hessian_condition(model, b) * U
    )


# ── row 6: the per-group table ────────────────────────────────────────────────


@cache
def uneven_groups() -> tuple[Book, object]:
    """The Gaussian book with one group of zero weight and one whose expected total is negative."""
    b = book("gaussian")
    weights = np.where(b.labels == "g00", 0.0, b.weights)
    offset = np.where(b.labels == "g01", b.offset - 10.0, b.offset)
    y = np.where(b.labels == "g01", b.y - 10.0, b.y)
    uneven = Book(b.frame, y, weights, offset, b.labels)
    model, caught = run(model_of(), uneven, groups=uneven.labels)
    assert not messages(caught, ConvergenceWarning)
    return uneven, model


def summed(pairs: pd.DataFrame, b: Book, labels) -> pd.DataFrame:
    """Reference pairs summed by ``labels`` of their rows, a pair at a time."""
    rows = pairs.index.get_level_values("row").to_numpy()
    w = b.weights[rows]
    table = (
        pd.DataFrame(
            {
                "label": np.asarray(labels)[rows],
                "rows": 1,
                "weight": w,
                "actual": w * b.y[rows],
                "heldout_expected": w * pairs["mean"].to_numpy(),
                "heldout_deviance": pairs.deviance.to_numpy(),
            }
        )
        .groupby("label")
        .sum()
    )
    ratio = table.actual / table.heldout_expected.where(table.heldout_expected > 0.0)
    table.insert(4, "actual_over_expected", ratio)
    return table


def assert_tables_alike(got: pd.DataFrame, want: pd.DataFrame, rel: float) -> None:
    assert got.columns.tolist() == want.columns.tolist()
    assert got.index.tolist() == want.index.tolist()
    assert got.rows.tolist() == want.rows.tolist()
    for column in got.columns[1:]:
        mine, theirs = got[column].to_numpy(), want[column].to_numpy()
        assert np.array_equal(np.isnan(mine), np.isnan(theirs)), column
        size = np.nanmax(np.abs(theirs))
        np.testing.assert_allclose(mine, theirs, rtol=rel, atol=rel * size, err_msg=column)


def test_group_table_is_refits_summed_by_the_callers_labels():
    """Per group and per level of a factor, with a weightless group and a negative expected total."""
    b, model = uneven_groups()
    problem, nb = at_chosen(model, b), from_groups(b.labels, len(b.y))
    want = refit_pairs(problem, nb, NONE)
    _, _, kappa = KeptRows(problem, NONE).one_step(nb)
    rel = 64 * kappa * U * max(1.0, float(np.abs(want.predictor).max()))
    table = model.ncv_heldout_groups()
    assert table.index.name == "group" and table.index.tolist() == sorted(set(b.labels))
    assert_tables_alike(table, summed(want, b, b.labels), rel)
    counts = pd.Series(b.labels).value_counts()
    assert table.rows.tolist() == counts.sort_index().tolist()
    empty = table.loc["g00"]
    assert (empty.rows, empty.weight, empty.actual, empty.heldout_expected) == (20, 0, 0, 0)
    assert np.isnan(empty.actual_over_expected) and empty.heldout_deviance == 0.0
    negative = table.loc["g01"]
    assert negative.heldout_expected < 0.0 and np.isnan(negative.actual_over_expected)
    assert table.actual_over_expected[table.heldout_expected > 0.0].notna().all()
    pd.testing.assert_frame_equal(model.ncv_heldout_groups(by=b.labels), table)
    by_level = model.ncv_heldout_groups(by=b.frame.f)
    assert by_level.index.name == "f"
    assert_tables_alike(by_level, summed(want, b, b.frame.f), rel)


def test_row_scored_by_several_neighbourhoods_counts_once_for_each():
    """Two repeats of five folds: per split, and per group of the caller's, against refits."""
    b, model = book("gaussian"), fitted("gaussian", "repeated")
    problem, nb = at_chosen(model, b), structure("repeated", b)
    want = refit_pairs(problem, nb, NONE)
    _, _, kappa = KeptRows(problem, NONE).one_step(nb)
    rel = 64 * kappa * U * max(1.0, float(np.abs(want.predictor).max()))
    splits = model.ncv_heldout_groups()
    assert splits.index.name == "split" and splits.index.tolist() == list(range(10))
    assert splits.rows.sum() == len(want) == 2 * len(b.y)
    neighbourhood = want.index.get_level_values("neighbourhood").to_numpy()
    for k in range(10):
        assert splits.rows[k] == np.count_nonzero(neighbourhood == k) == 60
        assert splits.heldout_deviance[k] == pytest.approx(
            want.deviance[neighbourhood == k].sum(), rel=rel
        )
    groups = model.ncv_heldout_groups(by=b.labels)
    assert_tables_alike(groups, summed(want, b, b.labels), rel)
    assert groups.rows.tolist() == [40] * 15
    assert groups.weight.sum() == pytest.approx(2 * b.weights.sum(), rel=8 * U)


def test_arguments_the_group_table_cannot_use_are_refused():
    rows, groups = fitted("gaussian", "rows"), fitted("gaussian", "groups")
    with pytest.raises(
        ValueError, match="has no groups to sum over: this fit held out single rows"
    ):
        rows.ncv_heldout_groups()
    with pytest.raises(
        ValueError, match=r"by must hold one label per row of X: .* \(299,\) for 300"
    ):
        groups.ncv_heldout_groups(by=np.zeros(299))
    missing = np.where(np.arange(300) == 7, None, "a")
    with pytest.raises(ValueError, match="by has missing labels on scored rows"):
        groups.ncv_heldout_groups(by=missing)
    fresh = model_of().fit(book("gaussian").frame, book("gaussian").y)
    for read in (fresh.ncv_diagnostics, fresh.ncv_heldout, fresh.ncv_heldout_groups):
        with pytest.raises(
            RuntimeError, match="reads a fit made by fit_ncv, and this model holds none"
        ):
            read()


@pytest.mark.parametrize("form", ["groups", "cv"])
def test_group_table_is_kept_when_the_fit_state_is_released(form):
    """150 groups of two rows, or twelve folds, under ``retain_fit_state=False``, saved and loaded."""
    b = book("gaussian")
    pairs = np.array([f"p{k:03d}" for k in np.random.default_rng(3).permutation(len(b.y)) // 2])
    arguments = {"groups": pairs} if form == "groups" else held_out("cv", b)
    kept, _ = run(model_of(), b, **arguments)
    compact, _ = run(model_of(retain_fit_state=False), b, **arguments)
    assert compact._dm is None and kept._dm is not None
    table = kept.ncv_heldout_groups()
    assert len(table) == (150 if form == "groups" else 12) and table.rows.sum() == len(b.y)
    for model in (compact, pickle.loads(pickle.dumps(compact)), pickle.loads(pickle.dumps(kept))):
        pd.testing.assert_frame_equal(model.ncv_heldout_groups(), table)
        assert model.ncv_diagnostics() == kept.ncv_diagnostics()
    pd.testing.assert_frame_equal(
        pickle.loads(pickle.dumps(kept)).ncv_heldout(), kept.ncv_heldout()
    )
    with pytest.raises(RuntimeError, match=r"ncv_heldout\(\) needs the fit's rows, which"):
        compact.ncv_heldout()
    with pytest.raises(RuntimeError, match=r"ncv_heldout_groups\(by=...\) needs the fit's rows"):
        compact.ncv_heldout_groups(by=b.labels)
    held = compact._reml_result.heldout
    assert held.row is None and held.heldout is None and held.labels is None


def test_accessors_hand_out_copies():
    """Whatever a caller does to a table or to the diagnostics, the next read is the fit's."""
    model = fitted("gaussian", "groups")

    def read():
        return model.ncv_heldout(), model.ncv_heldout_groups(), model.ncv_diagnostics()

    table, groups, report = read()
    table.loc[:, ["heldout", "fitted", "heldout_deviance", "weight", "actual"]] = -1.0
    table["label"] = "changed"
    groups.loc[:, ["weight", "actual", "heldout_expected", "heldout_deviance"]] = -1.0
    for name in ("lambdas", "start_lambdas", "neighbourhoods", "work"):
        report[name].clear()
    report["lambda_history"][0].clear()
    report["at_bound"].append(("x1", "floor"))
    report["criterion_history"][0] = report["gradient"][0] = -1.0
    report["infeasible_reasons"]["non_finite"] = 1
    again = read()
    assert (again[0].heldout_deviance >= 0.0).all() and again[0].label[0] != "changed"
    assert (again[1].weight >= 0.0).all() and (again[1].heldout_deviance >= 0.0).all()
    fresh = again[2]
    assert fresh["lambdas"] == model._reml_lambdas and fresh["start_lambdas"].keys() == {"x1", "x2"}
    assert fresh["neighbourhoods"]["count"] == 15 and fresh["work"]["inner_solves"] > 0
    assert fresh["lambda_history"][0].keys() == {"x1", "x2"} and fresh["at_bound"] == []
    assert fresh["criterion_history"][0] > 0 and fresh["infeasible_reasons"] == {}
    assert fresh["gradient"] == model._reml_result.gradient.tolist()
    pd.testing.assert_frame_equal(again[0], model.ncv_heldout())
    pd.testing.assert_frame_equal(again[1], model.ncv_heldout_groups())


# ── row 8: few neighbourhoods ─────────────────────────────────────────────────


@pytest.mark.parametrize(("count", "warned"), [(9, True), (10, False)])
def test_fewer_than_ten_neighbourhoods_fit_with_a_warning(count, warned):
    b = curve_book()
    model, caught = run(curve_model(), b, groups=np.arange(len(b.y)) % count)
    found = messages(caught)
    assert len(found) == warned
    assert all(m.startswith("fit_ncv has 9 neighbourhoods; with fewer than 10") for m in found)
    assert model.ncv_diagnostics()["neighbourhoods"]["count"] == count
    assert model.ncv_diagnostics()["converged"]


def test_one_supplied_split_fits_with_the_warning():
    """A single train and test split, as ``cross_validate`` accepts one."""
    b = curve_book()
    cv = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    model, caught = run(curve_model(), b, cv=cv)
    (message,) = messages(caught)
    assert message.startswith("fit_ncv has 1 neighbourhood; with fewer than 10 the held-out")
    ((_, test),) = cv.split(b.frame)
    table = model.ncv_heldout()
    assert sorted(table.row) == sorted(test) and len(model.ncv_heldout_groups()) == 1
    report = model.ncv_diagnostics()
    assert report["neighbourhoods"]["rows_scored"] == len(test) == 72
    assert report["converged"] and np.all(np.isfinite(model.predict(b.frame)))


# ── row 9: refusals come before any solve and leave an earlier fit alone ──────


class OneSplit:
    """A splitter that yields the one ``(train, test)`` it is given."""

    def __init__(self, train, test):
        self.pair = np.asarray(train), np.asarray(test)

    def split(self, X, y=None, groups=None):
        yield self.pair


def lone_level(b: Book, level) -> np.ndarray:
    """The book's groups with every row of one factor level in a group of its own."""
    return np.where(b.frame.f == level, "apart", b.labels)


REFUSALS = {
    "binned route": (
        {"discrete": True},
        {},
        r"fit_ncv does not support discrete=True, which fits on binned covariates; construct "
        r"the model with discrete=False",
    ),
    "family": (
        {"family": "gamma"},
        {},
        r"fit_ncv supports a Gaussian family with an identity link and a Poisson family with "
        r"a log link; this model has Gamma with LogLink\. Choose its smoothing with fit_reml",
    ),
    "link": (
        {"link": "log"},
        {},
        r"this model has Gaussian with LogLink\. Choose its smoothing with fit_reml",
    ),
    "selection penalty": (
        {"selection_penalty": 0.1},
        {},
        r"fit_ncv does not support a selection penalty \(selection_penalty=0\.1\); set "
        r"selection_penalty to None or 0, or use fit or fit_path",
    ),
    "selection penalty chosen by the fit": (
        {"selection_penalty": "auto"},
        {},
        r"fit_ncv does not support a selection penalty \(selection_penalty='auto'\)",
    ),
    "monotone term": (
        {"features": terms(x1=Spline(kind="ps", k=8, constraint=Constraint.fit.increasing))},
        {},
        r"fit_ncv does not support monotone or otherwise constrained terms \('x1'\); choose "
        r"this model's smoothing with fit_reml, or drop the fit-time constraint",
    ),
    "random effect": (
        {"features": terms(f=RandomEffect())},
        {},
        r"fit_ncv does not support RandomEffect terms \('f'\); choose this model's smoothing "
        r"with fit_reml",
    ),
    "factor smooth": (
        {"interactions": [FactorSmooth("x1", group="f", basis="sz", k=6)]},
        {},
        r"fit_ncv does not support FactorSmooth terms \('x1:f:sz'\); choose this model's "
        r"smoothing with fit_reml",
    ),
    "one group": (
        {},
        {"groups": lambda b: np.full(len(b.y), "all")},
        r"`groups` holds one label only \('all'\), so holding it out would leave no rows to fit",
    ),
    "folds and neighbours": (
        {},
        {"cv": lambda b: KFold(5), "neighbours": lambda b: [[i] for i in range(len(b.y))]},
        r"fit_ncv takes `cv` or `neighbours`, not both",
    ),
    "groups and neighbours": (
        {},
        {"groups": lambda b: b.labels, "neighbours": lambda b: [[i] for i in range(len(b.y))]},
        r"`groups` cannot be combined with `neighbours`",
    ),
    "labels for other rows": (
        {},
        {"groups": lambda b: b.labels[:-1]},
        r"`groups` must hold one label per row: it has shape \(299,\) for 300 rows",
    ),
    "neighbour that is no row": (
        {},
        {"neighbours": lambda b: [[i, 300] for i in range(len(b.y))]},
        r"`neighbours\[0\]` holds 300, which is not a row position from 0 to 299",
    ),
    "training rows without weight": (
        {},
        {"cv": lambda b: OneSplit(np.flatnonzero(b.weights == 0.0), np.flatnonzero(b.weights))},
        r"Split 0 of OneSplit removes every row with positive weight, leaving nothing to fit",
    ),
    "test row listed twice": (
        {},
        {"cv": lambda b: OneSplit(np.arange(100), [200, 201, 200])},
        r"Split 0 of `cv` \(OneSplit\) lists the test row 200 more than once",
    ),
    "level in one group": (
        {},
        {"groups": lambda b: lone_level(b, "d")},
        r"Every weighted row of level 'd' of 'f' lies in group 'apart', so with it held out "
        r"nothing in the fit estimates that level; merge the level with another or choose "
        r"neighbourhoods that divide its rows\.$",
    ),
    "reference level in one group": (
        {},
        {"groups": lambda b: lone_level(b, "a")},
        r"Every weighted row of level 'a' of 'f' lies in group 'apart', so with it held out "
        r"the other levels of 'f' add up to the intercept",
    ),
    "search tolerance": (
        {},
        {"ncv_tol": lambda b: 0.0},
        r"ncv_tol must be a positive number, got 0\.0; the default is 1e-06",
    ),
    "step limit": (
        {},
        {"max_ncv_iter": lambda b: 0},
        r"max_ncv_iter must be a whole number of at least 1, got 0; the default is 100",
    ),
    "start that is no mapping": (
        {},
        {"lambda2_init": lambda b: 3.0},
        r"lambda2_init must map smoothing components or terms to the smoothing parameter each "
        r"starts from, got float",
    ),
    "start that is no smoothing": (
        {},
        {"lambda2_init": lambda b: {"x1": -1.0}},
        r"lambda2_init\['x1'\] must be a positive number, got -1\.0; give the smoothing "
        r"parameter that term starts from, or leave it out",
    ),
}


def refusal_case(name):
    """A model the refusal applies to, the book it is fitted on, and ``fit_ncv``'s arguments."""
    settings, arguments, message = REFUSALS[name]
    settings = dict(settings)
    if "interactions" in settings:
        settings["features"] = {term: spec for term, spec in terms().items() if term != "f"}
    b = book("gaussian")
    if settings.get("family") == "gamma" or "link" in settings:
        b = replace(b, y=np.abs(b.y) + 1.0)
    return model_of(**settings), b, {k: v(b) for k, v in arguments.items()}, message


@pytest.mark.parametrize("name", list(REFUSALS))
def test_what_is_not_supported_is_refused_before_any_solve(solves, name):
    """Each refusal's sentence, no solve of either solver, and an earlier fit that still predicts."""
    model, b, arguments, message = refusal_case(name)
    structured = name in ("random effect", "factor smooth")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        earlier = model.fit_reml if structured else model.fit
        earlier(b.frame, b.y, sample_weight=b.weights, offset=b.offset)
    before = model.predict(b.frame, offset=b.offset)
    method, revision = fit_method.method_of(model), model._fit_revision
    assert sum(solves.values()) >= 1
    solves.clear()
    with pytest.raises((ValueError, TypeError), match=message):
        run(model, b, **arguments)
    assert sum(solves.values()) == 0
    assert fit_method.method_of(model) == method and model._fit_revision == revision
    np.testing.assert_array_equal(model.predict(b.frame, offset=b.offset), before)


def test_a_solver_route_fit_refuses_is_refused_before_any_solve_with_fit_s_reason(solves):
    """``fit`` refuses a forced structured route on a model without a structured term."""
    b = curve_book()
    with pytest.raises(ValueError) as refused:
        curve_model(direct_solve="structured").fit(b.frame, b.y)
    reason = str(refused.value)
    assert reason.startswith("direct_solve='structured' is ineligible")
    model = curve_model(direct_solve="structured")
    solves.clear()
    with pytest.raises(ValueError) as caught:
        run(model, b, groups=b.labels)
    assert str(caught.value) == (
        f"{reason.rstrip('.')}. Construct the model with direct_solve='auto', 'gram' or 'qr'."
    )
    assert sum(solves.values()) == 0 and fit_method.method_of(model) is None


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("route", ["gram", "qr"])
def test_a_solver_route_fit_accepts_is_the_published_fit_s_route(route):
    """Pins that a route ``fit`` accepts (``gram``, ``qr``) is honoured, not refused.

    The published fit is solved on it and the search converges; only a route ``fit``
    refuses is refused up front (the test above).
    """
    b = curve_book()
    model, _ = run(curve_model(direct_solve=route), b, groups=b.labels)
    assert model.result.direct_backend == route and model.ncv_diagnostics()["converged"]


def test_level_with_one_weighted_row_is_refused_when_single_rows_are_held_out(solves):
    b = book("gaussian")
    frame = b.frame.assign(f=np.where(np.arange(len(b.y)) == 5, "solo", b.frame.f))
    with pytest.raises(ValueError, match="Every weighted row of level 'solo' of 'f' lies in row 5"):
        run(model_of(), b, frame=frame)
    assert sum(solves.values()) == 0


def test_solve_counter_counts_every_fit_and_the_published_fit_is_one_solve(solves):
    """The control of the refusal tests' counter, and the solves of a whole ``fit_ncv``.

    A ``fit`` is one direct solve, a fit with a selection penalty one proximal solve, and a
    ``fit_ncv`` the search's solves and one more: the held-out table costs none.  (With an
    offset that varies, the search's smoothing box takes an intercept-only solve as well.)
    """
    b = curve_book()
    curve_model().fit(b.frame, b.y)
    assert solves == {"direct": 1}
    curve_model(selection_penalty=0.05).fit(b.frame, b.y)
    assert solves == {"direct": 1, "proximal": 1}
    solves.clear()
    model, _ = run(curve_model(), b, groups=b.labels)
    report = model.ncv_diagnostics()
    assert solves == {"direct": report["work"]["inner_solves"] + 1}
    assert report["work"]["inner_solves"] == report["n_evaluations"] + 1
    assert report["heldout_available"]


def test_levels_that_coincide_once_a_group_is_held_out_are_refused_before_the_search(solves):
    """Outside one group, a level of one factor marks the same rows as a level of another.

    No count of rows shows it, so the refusal follows the search's first two coefficient
    fits and precedes any step.
    """
    b = curve_book()
    n = len(b.y)
    rng = np.random.default_rng(9)
    one = rng.choice(list("abc"), n)
    inside = b.labels == "g00"
    other = np.where(inside, rng.choice(list("pq"), n), np.where(one == "b", "q", "p"))
    frame = b.frame.assign(one=one, other=other)
    features = {"t": Spline(kind="ps", n_knots=6), "one": Categorical(), "other": Categorical()}
    model = model_of(features=features).fit(frame, b.y)
    before, method = model.predict(frame), fit_method.method_of(model)
    solves.clear()
    with pytest.raises(
        LeaveOutRankError, match="With group 'g00' held out, the rows that remain do not identify"
    ):
        run(model, b, frame=frame, groups=b.labels)
    assert solves == {"direct": 2}
    assert fit_method.method_of(model) == method == "fit"
    np.testing.assert_array_equal(model.predict(frame), before)


def test_start_names_components_or_terms_and_ignores_other_names():
    """A term's name starts each of its components, a component's own name wins, a stranger warns."""
    b = curve_book()
    features = {"t": Spline(kind="ps", n_knots=6, select=True), "x": Numeric()}
    start = {"t": 4.0, "t:null": 9.0, "x": 2.0, "nobody": 1.0}
    model, caught = run(model_of(features=features), b, groups=b.labels, lambda2_init=start)
    (message,) = messages(caught)
    assert message.startswith("lambda2_init names no smoothing parameter of this model: nobody.")
    started = model.ncv_diagnostics()["start_lambdas"]
    assert started == pytest.approx({"t:null": 9.0, "t:wiggle": 4.0}, rel=4 * U)


# ── units and aliases ─────────────────────────────────────────────────────────


def stops_apart(model, b: Book, frame=None, h=0.05):
    """How far two first-order stops may lie apart in ``log lambda``, and what that moves.

    The EDF and the predictions per unit of ``log lambda``, from ``fit`` at the chosen
    smoothing times ``exp(-h)`` and ``exp(h)``.
    """
    nb = from_groups(b.labels, len(b.y))
    problem = problem_of(model, b.y, b.weights, b.offset, mode_bar=BAR)
    apart = 4 * stop_reach(Case(problem, nb, {}), model._reml_result)
    lam = chosen(model)["t"]
    low, high = (fit_at_chosen(model, b, frame, {"t": lam * np.exp(s * h)}) for s in (-1, 1))
    edf = abs(high.result.effective_df - low.result.effective_df) / (2 * h)
    frame = b.frame if frame is None else frame
    moves = np.max(np.abs(high.predict(frame) - low.predict(frame))) / (2 * h)
    return apart, edf, moves


def test_response_units_do_not_move_the_chosen_curve():
    """The response in units 10,000 times smaller chooses the same EDF and the same curve."""
    b, plain = curve_book(), curve_fit()
    small, caught = run(curve_model(), replace(b, y=1e-4 * b.y), groups=b.labels)
    assert not caught, messages(caught)
    apart, edf, moves = stops_apart(plain, b)
    rounding = 64 * hessian_condition(plain, b) * U
    mine, theirs = small.ncv_diagnostics(), plain.ncv_diagnostics()
    assert mine["criterion"] / 1e-4**2 == pytest.approx(theirs["criterion"], rel=10 * TOL)
    assert abs(np.log(mine["lambdas"]["t"] / theirs["lambdas"]["t"])) <= apart
    assert abs(small.result.effective_df - plain.result.effective_df) <= apart * edf + rounding * 9
    predicted = plain.predict(b.frame)
    gap = np.abs(small.predict(b.frame) / 1e-4 - predicted)
    assert np.all(gap <= apart * moves + rounding * np.maximum(1.0, np.abs(predicted)))
    assert abs(mine["n_iter"] - theirs["n_iter"]) <= 2 and mine["converged"]


# ── coefficient fits at the limit of float64 ──────────────────────────────────

# What a fit says of a coefficient fit the solver ended at the limit of float64 accuracy.
AT_THE_LIMIT = (
    "stopped at the limit of floating-point accuracy for this data, short of {their} "
    "convergence test (termination_reason='score_stagnated')"
)
THE_SEARCH_S = (
    "The smoothing is the one the search chose, and the model is returned with the "
    "coefficients, effective degrees of freedom (how flexible each term is), standard errors "
    "and held-out results of the fit there."
)
NEARER_ZERO = (
    "Coefficient fits stop at that limit when the response, less its offset where it has one, "
    "lies very far from zero compared with its spread: subtract a constant from the response "
    "before fitting and add it back to the predictions."
)
# What is reported of each coefficient solve: the solver's own report, every solve to the
# aim stagnated and every solve to the acceptance bar certified, every solve stagnated, or
# every solve stopped by its iteration limit.
REPORTS = {
    "as-solved": lambda arguments: None,
    "bar-met": lambda arguments: (
        "converged" if arguments["_mode_bar"] == BAR else "score_stagnated"
    ),
    "short-of-the-bar": lambda arguments: "score_stagnated",
    "iteration-limit": lambda arguments: "max_iter",
}
# The three a response far from zero can meet.
AT_THE_FLOAT_LIMIT = ("as-solved", "bar-met", "short-of-the-bar")


def limit_statement(report: dict, final: bool) -> str:
    """The disclosure of a search that met its test on coefficient fits at the limit of float64.

    ``report`` is the fit's ``ncv_diagnostics()``; ``final`` says the final fit was one of them.
    """
    used, tried = report["work"]["stagnated_fits"], report["n_evaluations"]
    several = used > 1 or (used == 1 and final)
    fits = f"the coefficient fit{'' if used == 1 else 's'} at {used} of the {tried} smoothing "
    fits += "settings it tried"
    if used and final:
        fits += ", and the final coefficient fit,"
    if not used:
        fits = "the final coefficient fit at that smoothing"
    stopped = AT_THE_LIMIT.format(their="their" if several else "its")
    return (
        f"fit_ncv did not converge: its smoothing search met its test after {report['n_iter']} "
        f"steps, but {fits} {stopped}. {THE_SEARCH_S} {NEARER_ZERO}"
    )


def fit_reported(model, b: Book, search="as-solved", final="as-solved", **arguments):
    """``fit_ncv`` with its coefficient solves reported as ``REPORTS`` says; also both logs of solves."""
    logs: tuple[list, list] = ([], [])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine, "fit_irls_direct", solver_ending(REPORTS[search], logs[0]))
        patch.setattr(ncv_ops, "fit_irls_direct", solver_ending(REPORTS[final], logs[1]))
        model, caught = run(model, b, groups=b.labels, **arguments)
    return model, caught, logs


@cache
def twin_book() -> Book:
    """The rank-cache tests' 240-row Gaussian book in twelve groups."""
    design = twin("gaussian", False)
    n = design.problem.n
    labels = np.array([f"g{k:02d}" for k in np.arange(n) % 12])
    return Book(design.frame[["t", "x"]], design.problem.y, np.ones(n), np.zeros(n), labels)


@cache
def at_zero(name):
    """A book as it is, its fit, and how far another first-order stop of its search may lie."""
    b = curve_book() if name == "curve" else twin_book()
    model, caught = run(curve_model(), b, groups=b.labels)
    assert not caught, messages(caught)
    return b, model, stops_apart(model, b), hessian_condition(model, b)


@cache
def far_fit(name, level, report):
    """A book's fit with its response far from zero, every coefficient solve reported as ``report``.

    The curve book is moved by ``level`` of its spread and the twin book by ``level``.
    """
    b = at_zero(name)[0]
    shift = level * float(np.std(b.y)) if name == "curve" else level
    fit = fit_reported(curve_model(), replace(b, y=b.y + shift), search=report, final=report)
    return shift, *fit


FAR = [("curve", 1e7), ("curve", 1e8), ("twin", 1e7), ("twin", 1e8)]


@pytest.mark.parametrize("report", AT_THE_FLOAT_LIMIT)
@pytest.mark.parametrize(("name", "level"), FAR)
def test_response_far_from_zero_chooses_the_curve_of_the_response_at_zero(name, level, report):
    """A Gaussian response 1e7 and 1e8 from zero: the smoothing, EDF and curve of the response at zero.

    That far out a coefficient fit ends at the limit of float64, a rounding to either side
    of the acceptance bar.  The fit is made as the solver reports its solves, with every one
    reported as meeting that bar, and with every one reported short of it.  The curve is
    the same each way; the model reads as converged where no fit was short of the bar, and
    as not converged, with one warning that says why and what to do, where one was.

    Tolerances: two first-order stops (``stops_apart``), and ``64 kappa U`` times the size
    of what is compared, which for a response at the shift is the shift.  The held-out
    results are compared with library refits of the response at zero, moved by the shift: a
    least-squares fit with an unpenalised intercept moves with its response.
    """
    b, plain, (apart, edf, moves), kappa = at_zero(name)
    shift, model, caught, (searched, final) = far_fit(name, level, report)
    mine, theirs = model.ncv_diagnostics(), plain.ncv_diagnostics()
    rounding = 64 * kappa * U
    assert abs(np.log(mine["lambdas"]["t"] / theirs["lambdas"]["t"])) <= apart
    assert abs(model.result.effective_df - plain.result.effective_df) <= apart * edf + rounding * 9
    predicted = plain.predict(b.frame)
    gap = np.abs(model.predict(b.frame) - shift - predicted)
    assert np.all(gap <= apart * moves + rounding * np.abs(predicted + shift))
    table = plain.ncv_heldout()
    slope = 2 * table.weight * np.abs(table.heldout - table.actual)
    resolved = rounding * float(np.sum(slope * np.abs(table.heldout + shift)))
    assert abs(mine["criterion"] - theirs["criterion"]) <= 10 * TOL * theirs["criterion"] + resolved

    problem, nb = at_chosen(model, b), from_groups(b.labels, len(b.y))
    want = refit_pairs(problem, nb, NONE)
    _, _, kappa_out = KeptRows(problem, NONE).one_step(nb)
    got = heldout_pairs(model)
    got = got.assign(predictor=got.predictor - shift, mean=got["mean"] - shift)
    slack = predictor_slack(kappa_out, want.predictor.to_numpy() + shift)
    assert_pairs_equal(problem, got, want, slack)
    assert mine["heldout_available"] and mine["heldout_failure"] is None
    assert mine["criterion"] == model.ncv_heldout().heldout_deviance.to_numpy().sum()

    used, at_limit = mine["work"]["stagnated_fits"], not model.result.converged
    assert mine["search_converged"] and mine["termination_reason"] == "gradient"
    assert mine["infeasible_trials"] == 0 and final[-1].certified != at_limit
    assert mine["converged"] == mine["coefficients_converged"] == (not used and not at_limit)
    assert mine["coefficient_termination"] == (None if mine["converged"] else "score_stagnated")
    expected = [] if mine["converged"] else [limit_statement(mine, at_limit)]
    assert messages(caught) == expected
    if report == "bar-met":
        assert mine["converged"]
    if report == "short-of-the-bar":
        assert used == mine["n_evaluations"] and at_limit

    # The aim is paid for once: the provisional fit and one continuation beside the
    # evaluations, and a final fit of one solve unless it is the first short of the aim.
    work = mine["work"]
    assert work["inner_solves"] == len(searched) <= mine["n_evaluations"] + 2
    assert work["fallback_solves"] <= 1 and len(final) <= 2 - work["fallback_solves"]
    if report != "as-solved":
        assert [solve.bar for solve in searched] == [AIM] + [BAR] * (mine["n_evaluations"] + 1)
        assert [solve.bar for solve in final] == [BAR]


@pytest.mark.parametrize(("name", "level"), FAR)
def test_the_curve_does_not_depend_on_which_bar_a_fit_at_the_limit_met(name, level):
    """The same solves reported as meeting the acceptance bar, and as short of it: one fit, to the last bit."""
    _, met, _, _ = far_fit(name, level, "bar-met")
    _, short, _, _ = far_fit(name, level, "short-of-the-bar")
    mine, theirs = short.ncv_diagnostics(), met.ncv_diagnostics()
    assert theirs["converged"] and not mine["converged"]
    for key in ("lambdas", "criterion", "gradient", "lambda_history", "criterion_history"):
        assert mine[key] == theirs[key], key
    assert short.result.effective_df == met.result.effective_df
    frame = at_zero(name)[0].frame
    assert np.array_equal(short.predict(frame), met.predict(frame))
    pd.testing.assert_frame_equal(short.ncv_heldout(), met.ncv_heldout(), check_exact=True)
    pd.testing.assert_frame_equal(
        short.ncv_heldout_groups(), met.ncv_heldout_groups(), check_exact=True
    )


@pytest.mark.parametrize("where", ["search", "final", "both"])
def test_fits_at_the_limit_of_float64_are_used_and_disclosed_in_one_warning(where):
    """The solves of the search, of the final fit, or of both are made to end stagnated.

    The search goes on from such fits and the final fit keeps its held-out results, which
    are library refits without each group.  The model is not converged: the search's verdict
    stands, and the coefficient fits' does not.  A stagnated final fit is one solve after a
    search that stopped aiming, and two after one that did not: no solve from another start.
    After the unforced search it is one exact step on from the certified fit, and everything
    read of the model, its standard errors among it, is that fit's.
    """
    b, good = curve_book(), curve_fit()
    forced = {
        "search": "short-of-the-bar" if where != "final" else "as-solved",
        "final": "short-of-the-bar" if where != "search" else "as-solved",
    }
    model, caught, (_, final) = fit_reported(curve_model(), b, **forced)
    report, reached = model.ncv_diagnostics(), good.ncv_diagnostics()
    assert messages(caught) == [limit_statement(report, final=where != "search")]
    assert report["search_converged"] and report["termination_reason"] == "gradient"
    assert not report["converged"] and not report["coefficients_converged"]
    assert report["coefficient_termination"] == "score_stagnated"
    assert model.result.converged == (where == "search")
    used = 0 if where == "final" else report["n_evaluations"]
    assert report["work"]["stagnated_fits"] == used and reached["work"]["stagnated_fits"] == 0
    assert [(solve.bar, solve.start is not None) for solve in final] == {
        "search": [(BAR, True)],
        "final": [(AIM, True), (BAR, True)],
        "both": [(BAR, True)],
    }[where]

    apart, edf, moves = stops_apart(good, b)
    kappa = hessian_condition(good, b)
    assert abs(np.log(report["lambdas"]["t"] / reached["lambdas"]["t"])) <= apart
    if where == "final":
        assert report["lambdas"] == reached["lambdas"]
        assert_reads_alike(readings(model, b.frame, b), readings(good, b.frame, b), 64 * kappa * U)
    predicted = good.predict(b.frame)
    gap = np.abs(model.predict(b.frame) - predicted)
    assert np.all(gap <= apart * moves + 64 * kappa * U * np.maximum(1.0, np.abs(predicted)))
    problem, nb = at_chosen(model, b), from_groups(b.labels, len(b.y))
    want = refit_pairs(problem, nb, NONE)
    _, _, kappa_out = KeptRows(problem, NONE).one_step(nb)
    slack = predictor_slack(kappa_out, want.predictor.to_numpy())
    assert_pairs_equal(problem, heldout_pairs(model), want, slack)
    assert report["heldout_available"] and report["heldout_failure"] is None
    assert len(model.ncv_heldout_groups()) == 12


def test_fits_at_the_limit_of_float64_are_named_beside_whatever_else_fell_short():
    """After a search on stagnated fits: one stopped at its step limit, and a final fit at its iteration limit.

    The first keeps its held-out results.  The second has none: a fit stopped by its
    iteration limit stands for no mode, and is solved once more from the solver's own start.
    """
    b, short = curve_book(), "short-of-the-bar"
    model, caught, _ = fit_reported(curve_model(), b, search=short, final=short, max_ncv_iter=1)
    mine = model.ncv_diagnostics()
    tried = mine["n_evaluations"]
    fits = f"the coefficient fits at {tried} of the {tried} smoothing settings it tried"
    (message,) = messages(caught)
    assert message.startswith(
        "fit_ncv did not converge: its smoothing search stopped after 1 step because it "
        f"reached the max_ncv_iter limit, and {fits}, and the final coefficient fit, "
        f"{AT_THE_LIMIT.format(their='their')}. The model is returned at the lowest held-out "
        "deviance the search reached"
    )
    assert message.endswith(f"to continue from this fit. {NEARER_ZERO}")
    assert not mine["search_converged"] and not mine["coefficients_converged"]
    assert mine["heldout_available"] and mine["work"]["stagnated_fits"] == tried

    model, caught, (_, final) = fit_reported(
        curve_model(), b, search=short, final="iteration-limit"
    )
    mine = model.ncv_diagnostics()
    tried = mine["n_evaluations"]
    fits = f"the coefficient fits at {tried} of the {tried} smoothing settings it tried"
    assert messages(caught) == [
        f"fit_ncv did not converge: its smoothing search met its test after {mine['n_iter']} "
        f"steps, but {fits} {AT_THE_LIMIT.format(their='their')}, and the final coefficient "
        "fit at that smoothing did not meet its convergence test (it stopped with "
        "termination_reason='max_iter'). The model is returned with the coefficients, "
        "effective degrees of freedom and standard errors of that fit's last step, and "
        "without held-out results. Refit with a larger max_pirls_iter to let the final fit "
        f"finish. {NEARER_ZERO}"
    ]
    assert [(solve.bar, solve.start is not None) for solve in final] == [(BAR, True), (BAR, False)]
    assert mine["coefficient_termination"] == "max_iter" and not model.result.converged
    assert (mine["heldout_available"], mine["heldout_failure"]) == (False, "uncertified_mode")


def test_poisson_fits_at_the_limit_of_float64_are_used_and_disclosed():
    """Every coefficient solve of a Poisson fit is made to end stagnated.

    As for a Gaussian fit: each trial counts, the model is not converged and says so once,
    and it keeps its held-out results, here the Newton step of each group's kept rows from
    the published fit.  The search's stop is not asserted: its fits run to the acceptance
    bar after the first, where the last steps of a Poisson search depend on their rounding.
    """
    b = curve_book("poisson")
    good, quiet = run(curve_model("poisson"), b, groups=b.labels)
    assert not messages(quiet) and good.ncv_diagnostics()["work"]["stagnated_fits"] == 0
    report = "short-of-the-bar"
    model, caught, (searched, final) = fit_reported(
        curve_model("poisson"), b, search=report, final=report
    )
    mine, theirs = model.ncv_diagnostics(), good.ncv_diagnostics()
    assert mine["work"]["stagnated_fits"] == mine["n_evaluations"] > 1
    assert mine["infeasible_trials"] == 0 and mine["termination_reason"] != "no_finite_criterion"
    assert not mine["converged"] and not mine["coefficients_converged"]
    assert mine["coefficient_termination"] == "score_stagnated" and not model.result.converged
    assert [solve.bar for solve in searched] == [AIM] + [BAR] * (mine["n_evaluations"] + 1)
    assert [solve.bar for solve in final] == [BAR]
    (message,) = messages(caught, ConvergenceWarning)
    fits = (
        f"the coefficient fits at {mine['n_evaluations']} of the {mine['n_evaluations']} "
        "smoothing settings it tried, and the final coefficient fit, "
    )
    assert fits + AT_THE_LIMIT.format(their="their") in message and message.endswith(NEARER_ZERO)
    if mine["search_converged"]:
        assert message == limit_statement(mine, final=True)

    apart, _, _ = stops_apart(good, b)
    assert abs(np.log(mine["lambdas"]["t"] / theirs["lambdas"]["t"])) <= apart
    assert mine["criterion"] == pytest.approx(theirs["criterion"], rel=10 * TOL)
    problem, nb = at_chosen(model, b), from_groups(b.labels, len(b.y))
    reference = KeptRows(problem, NONE, at=(predictor(model, b), model.result.beta))
    want, residual, kappa = reference.one_step(nb)
    table = model.ncv_heldout()
    assert np.array_equal(table.row, nb.scored_rows())
    slack = residual + predictor_slack(kappa, want)
    held_gap = np.abs(np.log(table.heldout.to_numpy()) - want)
    assert np.all(held_gap <= slack), f"gap {held_gap.max():.3e} against {slack.min():.3e}"
    assert mine["heldout_available"] and mine["criterion"] == table.heldout_deviance.sum()


def test_duplicated_column_fits_as_the_model_without_it():
    """The same numeric column twice: the criterion, the EDF and the predictions of the model with one."""
    b, clean = curve_book(), curve_fit()
    frame = b.frame.assign(twin=b.frame.x)[["x", "twin", "t"]]
    features = {"x": Numeric(), "twin": Numeric(), "t": Spline(kind="ps", n_knots=6)}
    aliased, caught = run(model_of(features=features), b, frame=frame, groups=b.labels)
    assert not caught, messages(caught)
    assert not aliased.result.rank_info.coefficient_estimable().all()
    apart, edf, moves = stops_apart(clean, b)
    rounding = 64 * hessian_condition(clean, b) * U
    mine, theirs = aliased.ncv_diagnostics(), clean.ncv_diagnostics()
    assert mine["converged"] and mine["infeasible_trials"] == 0
    assert mine["criterion"] == pytest.approx(theirs["criterion"], rel=10 * TOL)
    assert abs(np.log(mine["lambdas"]["t"] / theirs["lambdas"]["t"])) <= apart
    assert (
        abs(aliased.result.effective_df - clean.result.effective_df) <= apart * edf + rounding * 9
    )
    assert aliased._group_edf["x"] + aliased._group_edf["twin"] == pytest.approx(1.0, abs=rounding)
    predicted = clean.predict(b.frame)
    slack = apart * moves + rounding * np.maximum(1.0, np.abs(predicted))
    assert np.all(np.abs(aliased.predict(frame) - predicted) <= slack)
