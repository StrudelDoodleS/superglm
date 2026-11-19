"""Whether the penalised estimate exists, decided exactly from the responses and the design.

A coefficient whose estimate is infinite has no standard error.  ``diagnostics.separation``
decides whether a direction of recession exists (Verbeek 1989; Santos Silva & Tenreyro 2010;
Correia, Guimaraes & Zylkin 2019, arXiv:1903.01633, Prop. 1; Konis 2007 for the binomial):
the fit's score proves existence (Stiemke's alternative), or a separating direction is verified
on the rows exactly, and a separated fit publishes the fit on the rows it keeps (Prop. 3).  Each
case below printed a number at ``92302bc40`` where it now prints none, or the walked fit's SE
where it now prints the kept rows'; the controls say so.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from superglm import Categorical, Numeric, RandomEffect, Spline, SuperGLM

FIXTURES = Path(__file__).parent / "fixtures"


def published_separation(model):
    """The published fit's existence decision, read from its identifiability record."""
    return getattr(model._fit_inference_info.get("identifiability"), "separation", None)


def _patch(monkeypatch, module: str, name: str, value) -> None:
    """Patch ``module.name`` where it exists (a change's new name is absent before it)."""
    import importlib
    import importlib.util

    if importlib.util.find_spec(module) is not None:
        monkeypatch.setattr(importlib.import_module(module), name, value, raising=False)


def _fit(family, features, X, y, *, method="fit", **kwargs):
    model = SuperGLM(family=family, features=features, selection_penalty=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y, **kwargs)
    return model


def _summary_se(model) -> dict[str, float]:
    return {row.name: row.se for row in model.summary()._coef_rows if not row.is_spline}


def _notes(model) -> str:
    record = model._fit_inference_info.get("identifiability")
    return "" if record is None else "\n".join(record.notes())


def _kept_row_variance(model, rows) -> tuple[np.ndarray, np.ndarray]:
    """The published variance of the linear predictor on ``rows``, and an independent one.

    The oracle is the pseudo-inverse of the kept rows' augmented Hessian ``A'WA + S`` at the
    published fit's Fisher weights, formed from the explicit design by eigendecomposition: the
    separated span is its null space, and a row it does not move has one quadratic form there.
    """
    from superglm.model.state_ops import (
        _public_intercept_shift,
        _rank_curvature,
        _solver_space_working_weights,
    )

    inference = model._fit_inference_info
    n = model._dm.n
    design = np.hstack(
        [np.ones((n, 1))]
        + [np.asarray(g.toarray(), float).reshape(n, -1) for g in model._dm.group_matrices]
    )
    public = design.copy()
    public[:, 1:] -= _public_intercept_shift(model, inference["active_groups"], design.shape[1] - 1)
    published = np.einsum(
        "ij,jk,ik->i", public[rows], np.asarray(inference["XtWX_inv_aug"]), public[rows]
    )
    W = _solver_space_working_weights(model)
    penalty = np.zeros((design.shape[1],) * 2)
    penalty[1:, 1:] = _rank_curvature(model, list(model._groups))
    values, vectors = np.linalg.eigh(design.T @ (W[:, None] * design) + penalty)
    kept = values > 1e-10 * values[-1]
    inverse = (vectors[:, kept] / values[kept]) @ vectors[:, kept].T
    return published, np.einsum("ij,jk,ik->i", design[rows], inverse, design[rows])


# ------------------------------------------------------------------ a separated reference level
def _reference_level_frame():
    """Opus's gate case C3: a Poisson factor whose reference level E has every response 0."""
    rng = np.random.default_rng(1)
    n = 2000
    f = rng.choice(list("ABCDE"), n)
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(-1 + 0.5 * (f == "B") + np.sin(2 * np.pi * x))).astype(float)
    y[f == "E"] = 0.0
    return pd.DataFrame({"f": f, "x": x}), y, f != "E"


def test_a_reference_level_with_every_response_zero_has_no_standard_error_but_a_footnote():
    """``f``'s reference level E has no response off zero: the intercept and every level of
    ``f`` move to infinity with it, so none has a standard error, and the footnote names E.

    A direction of recession (intercept down, every other level up) is verified on the rows
    exactly; the fit published is the fit on the rows it keeps, whose covariance on those rows
    is the pseudo-inverse of their Hessian (to 1e-10), and E's rows stay at the boundary.  At
    ``92302bc40`` the summary printed SE 74,345 on the intercept and on each level, from
    wherever PIRLS stopped.
    """
    X, y, kept = _reference_level_frame()
    model = _fit("poisson", {"f": Categorical(), "x": Spline(n_knots=8)}, X, y, method="fit_reml")
    se = _summary_se(model)
    for name in ("Intercept", "f[A]", "f[B]", "f[C]", "f[D]"):
        assert np.isnan(se[name]), name
    notes = _notes(model)
    assert "5 coefficients have no finite estimate" in notes
    assert "level E of f" in notes
    metrics = model.metrics(X, y)
    assert np.all(np.isnan(metrics.coefficient_se["f"])) and np.isnan(metrics.intercept_se)
    assert np.all(np.isfinite(metrics.coefficient_se["x"]))
    assert float(np.max(model.predict(X)[~kept])) < 1e-6
    separation = published_separation(model)
    assert separation is not None and separation.state == "present"
    assert separation.rows.size == int(np.sum(~kept))
    published, oracle = _kept_row_variance(model, kept)
    np.testing.assert_allclose(published, oracle, rtol=1e-10)


def test_the_separation_decision_does_not_depend_on_where_the_walk_stopped():
    """The same separation, decided at two depths of the walk, publishes one fit.

    A loose and a tight PIRLS tolerance stop the walk along E's direction at different depths
    (the event is forced, not left to a BLAS kernel's rounding): both decide the same rows and
    the same coefficients, and the kept rows' fit, standard errors and predictions agree, since
    they come from the fit on those rows.  At ``92302bc40`` the level SEs were the walk's, and
    moved with its depth.
    """
    X, y, kept = _reference_level_frame()
    loose = _fit("poisson", {"f": Categorical(), "x": Spline(n_knots=8)}, X, y, tol=1e-5)
    tight = _fit("poisson", {"f": Categorical(), "x": Spline(n_knots=8)}, X, y, tol=1e-12)
    assert abs(loose.result.intercept - tight.result.intercept) > 1.0, "the walks must differ"
    for model in (loose, tight):
        se = _summary_se(model)
        assert all(np.isnan(se[name]) for name in ("Intercept", "f[A]", "f[B]", "f[C]", "f[D]"))
    first, second = published_separation(loose), published_separation(tight)
    assert first.state == second.state == "present"
    assert first.names == second.names
    np.testing.assert_array_equal(first.rows, second.rows)
    assert _summary_se(loose).keys() == _summary_se(tight).keys()
    for name, value in _summary_se(loose).items():
        assert np.isnan(value) == np.isnan(_summary_se(tight)[name])
    np.testing.assert_allclose(
        loose.metrics(X, y).coefficient_se["x"], tight.metrics(X, y).coefficient_se["x"], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.log(loose.predict(X[kept])), np.log(tight.predict(X[kept])), rtol=0, atol=1e-6
    )


@pytest.mark.pins_behaviour
def test_a_penalised_factor_keeps_its_estimates_finite():
    """Control: with ``f`` a random effect the separating ray has penalty curvature, so the
    penalised estimate exists and every coefficient keeps its standard error, as at
    ``92302bc40``.  The decision, kept on the record, is a proved absence."""
    X, y, _ = _reference_level_frame()
    model = _fit("poisson", {"f": RandomEffect(), "x": Spline(n_knots=8)}, X, y, method="fit_reml")
    separation = published_separation(model)
    assert separation is None or separation.state == "absent"
    metrics = model.metrics(X, y)
    assert np.isfinite(metrics.intercept_se)
    assert np.all(np.isfinite(metrics.coefficient_se["f"]))
    assert "no finite estimate" not in _notes(model)


def test_a_zero_working_weight_on_a_separated_row_is_judged_by_the_rows(monkeypatch):
    """A boundary row whose working weight is exactly zero gives no multiplier, so the score
    cannot prove existence and the rows decide, as they do with the weight there: the same
    coefficients lose their standard errors.  At ``92302bc40`` they had them."""
    X, y, kept = _reference_level_frame()
    real = []

    def vanishing(model, solver, weights):
        W, residual = real[0](model, solver, weights)
        return np.where(kept, W, 0.0), residual

    try:
        from superglm.model.separation_ops import fisher_rows
    except ImportError:
        pass
    else:
        real.append(fisher_rows)
        _patch(monkeypatch, "superglm.model.separation_ops", "fisher_rows", vanishing)
    model = _fit("poisson", {"f": Categorical(), "x": Spline(n_knots=8)}, X, y)
    se = _summary_se(model)
    assert all(np.isnan(se[name]) for name in ("Intercept", "f[A]", "f[B]", "f[C]", "f[D]"))


def test_a_direction_that_cannot_be_verified_is_disclosed_unresolved(monkeypatch):
    """Where no separating direction verifies on the rows exactly and the score proves nothing,
    the summary says existence is not resolved at float64, and no fit is withheld.  Forced: the
    rows' program is made to find nothing.  At ``92302bc40`` nothing was said."""
    _patch(
        monkeypatch,
        "superglm.model.separation_ops",
        "separating_direction",
        lambda matrix, sides, priority=None: (None, np.zeros(0, dtype=np.intp), {}),
    )
    X, y, _ = _reference_level_frame()
    model = _fit("poisson", {"f": Categorical(), "x": Spline(n_knots=8)}, X, y)
    assert "not resolved at float64" in _notes(model)
    assert published_separation(model).state == "unresolved"
    assert getattr(model, "_separated_rows", None) is None


# ------------------------------------------------------------------- binomial, against Konis's LP
def _konis_infinite(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which coefficients Konis's linear program (2007, ch. 4) finds infinite, from its definition.

    Maximise ``sum_i s_i a_i'b`` under ``s_i a_i'b >= 0`` and ``-1 <= b <= 1`` (``s = 2y - 1``):
    the data are separated exactly when the optimum is positive, and a coefficient is infinite
    where the solution's ``b_j`` is not zero (R ``detectseparation``'s convention; the package is
    not installed here, so its program is written out).
    """
    signed = (2.0 * y - 1.0)[:, None] * design
    program = linprog(
        -signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(len(y)),
        bounds=[(-1.0, 1.0)] * design.shape[1],
        method="highs",
    )
    assert program.status == 0
    return np.abs(program.x) > 1e-9


def _complete_frame():
    rng = np.random.default_rng(3)
    x1, x2 = rng.uniform(size=300), rng.uniform(size=300)
    y = (x1 + 0.5 * x2 > 0.6).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


def _quasi_complete_frame():
    rng = np.random.default_rng(3)
    x1 = np.r_[rng.uniform(0.1, 1, 100), -rng.uniform(0.1, 1, 100), np.zeros(40)]
    x2 = rng.normal(size=240)
    y = np.r_[np.ones(100), np.zeros(100), (np.arange(40) % 2).astype(float)]
    return pd.DataFrame({"x1": x1, "x2": x2}), y


@pytest.mark.parametrize("frame", [_complete_frame, _quasi_complete_frame])
def test_binomial_separation_withholds_what_konis_program_finds_infinite(frame):
    """Logistic regression on complete and quasi-complete separation: the coefficients without a
    standard error are exactly those Konis's program finds infinite.

    Complete: every coefficient (no row is kept).  Quasi-complete: the ties at ``x1 = 0`` carry
    both outcomes, so the intercept and ``x2`` are finite and fitted on those rows, ``x1`` is
    infinite.  At ``92302bc40`` every coefficient printed a finite SE.
    """
    X, y = frame()
    model = _fit("binomial", {"x1": Numeric(), "x2": Numeric()}, X, y)
    design = np.column_stack([np.ones(len(y)), X["x1"], X["x2"]])
    infinite = _konis_infinite(design, y)
    metrics = model.metrics(X, y)
    blank = np.array(
        [
            np.isnan(metrics.intercept_se),
            np.isnan(metrics.coefficient_se["x1"][0]),
            np.isnan(metrics.coefficient_se["x2"][0]),
        ]
    )
    np.testing.assert_array_equal(blank, infinite)
    assert published_separation(model).state == "present"


# ------------------------------------------------------------------- Correia et al.'s table 2
def test_quasi_complete_poisson_separation_along_two_rays_is_found_whole():
    """Correia, Guimaraes & Zylkin's table 2: two combinations vanish on the positive rows and
    each overlaps the zeros, yet a mixture separates.  The program finds the whole face: rows 1-3
    are separated and ``x2``, ``x3`` and ``x4`` are infinite; the intercept is finite and is the
    maximum-likelihood fit of the kept rows (``x2 = x3 = x4`` there).  At ``92302bc40`` all four
    printed SEs.
    """
    table = np.array(
        [
            [0, -1, 5, 3],
            [0, 2, 0, 1],
            [0, 0, -6, -3],
            [0, 0, 0, 0],
            [1, 3, 3, 3],
            [2, 6, 6, 6],
            [3, 5, 5, 5],
            [4, 7, 7, 7],
            [5, 4, 4, 4],
        ],
        dtype=float,
    )
    X = pd.DataFrame(table[:, 1:], columns=["x2", "x3", "x4"])
    y = table[:, 0]
    model = _fit("poisson", {"x2": Numeric(), "x3": Numeric(), "x4": Numeric()}, X, y)
    se = model.metrics(X, y).coefficient_se
    assert all(np.isnan(se[name][0]) for name in ("x2", "x3", "x4"))
    separation = published_separation(model)
    assert separation.state == "present"
    np.testing.assert_array_equal(separation.rows, [0, 1, 2])
    # the kept rows' Poisson fit on (1, x), by Newton's method
    A = np.column_stack([np.ones(6), table[3:, 1]])
    b = np.zeros(2)
    for _ in range(100):
        mu = np.exp(A @ b)
        b = b + np.linalg.solve(A.T @ (mu[:, None] * A), A.T @ (y[3:] - mu))
    assert model.result.intercept == pytest.approx(b[0], abs=1e-6)


def test_a_separated_set_short_of_the_whole_face_is_not_certified(monkeypatch):
    """A verified separating direction that separates only some of the rows that can be is not
    the decision: the fit on the rows it keeps must prove they separate no further, and here
    they do (rows 2 and 3 of table 2 still separate).  Forced: the program returns the mixture
    ``(x2 + x3) / 2 - x4``, which separates row 1 alone.  The outcome is a proven lower bound
    (``"partial"``): the coefficients that direction moves have no finite estimate and are
    withheld, and the footnote says more rows may be separated.  At ``92302bc40`` every
    coefficient printed an SE; at ``07a1401a5`` the footnote called them unresolved, although
    the verified direction proves them infinite."""
    from scipy import sparse

    table = np.array(
        [
            [0, -1, 5, 3],
            [0, 2, 0, 1],
            [0, 0, -6, -3],
            [0, 0, 0, 0],
            [1, 3, 3, 3],
            [2, 6, 6, 6],
            [3, 5, 5, 5],
            [4, 7, 7, 7],
            [5, 4, 4, 4],
        ],
        dtype=float,
    )
    partial = np.array([0.0, 0.5, 0.5, -1.0])

    def short(matrix, sides, priority=None):
        assert sparse.issparse(matrix)
        return partial, np.array([0]), {}

    _patch(monkeypatch, "superglm.model.separation_ops", "separating_direction", short)
    X = pd.DataFrame(table[:, 1:], columns=["x2", "x3", "x4"])
    y = table[:, 0]
    model = _fit("poisson", {"x2": Numeric(), "x3": Numeric(), "x4": Numeric()}, X, y)
    se = model.metrics(X, y).coefficient_se
    assert all(np.isnan(se[name][0]) for name in ("x2", "x3", "x4"))
    notes = _notes(model)
    assert "not resolved" not in notes
    assert "(row 0)" in notes and "more rows may be" in notes
    assert published_separation(model).state == "partial"


# ------------------------------------------------------------------- Sol's gate counterexamples
def test_a_separating_ray_one_unit_in_the_last_place_from_an_alias_is_found():
    """Sol's first counterexample: rows ``(1, -1, -1), (1, 1, 1), (1, 0, 2^-52)``, ``y = (1, 1,
    0)``.  The slopes' difference separates the third row by ``2^-52``; a score whose residual is
    small cannot set it aside as an alias, and the exact verification finds it, so both slopes
    are infinite.  At ``92302bc40`` ``b`` was an alias and ``a`` printed SE 0.866."""
    delta = 2.0**-52
    X = pd.DataFrame({"a": [-1.0, 1.0, 0.0], "b": [-1.0, 1.0, delta]})
    y = np.array([1.0, 1.0, 0.0])
    model = _fit("poisson", {"a": Numeric(), "b": Numeric()}, X, y)
    se = model.metrics(X, y).coefficient_se
    assert np.isnan(se["a"][0]) and np.isnan(se["b"][0])
    separation = published_separation(model)
    assert separation.state == "present"
    np.testing.assert_array_equal(separation.rows, [2])


def test_a_proposal_that_violates_a_row_by_one_unit_in_the_last_place_is_rejected():
    """Sol's second counterexample, on the program itself: signed rows ``(1, 1), (-1, -1),
    (1, 1 + 2^-52), (0, -1)`` admit no separating direction, though HiGHS reports one, ``(1,
    -1)``, whose third row is ``-2^-52``.  The exact verification rejects it (a check within the
    evaluation's rounding bound accepts it).  Control: the program is new in this change."""
    from scipy import sparse

    from superglm.diagnostics.separation import separating_direction

    delta = 2.0**-52
    rows = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0 + delta], [0.0, -1.0]])
    # at the upper boundary every row's sign is its own
    direction, strict, _ = separating_direction(sparse.csr_matrix(rows), np.ones(4, np.int8))
    assert direction is None and strict.size == 0


def test_an_untouched_slope_takes_its_standard_error_from_the_kept_rows():
    """Sol's third counterexample: kept rows ``x = +-1e-8`` with response 1, separated rows ``x =
    +-1`` with response 0 and indicator 1.  Only the indicator is infinite; the slope's standard
    error is the kept rows' fit's, ``1 / sqrt(2e-16) = 70,710,678.12`` at fixed scale.  At
    ``92302bc40`` it was the walk's, 70,710.64: a thousand times too small."""
    X = pd.DataFrame({"x": [1e-8, -1e-8, 1.0, -1.0], "s": [0.0, 0.0, 1.0, 1.0]})
    y = np.array([1.0, 1.0, 0.0, 0.0])
    model = _fit("poisson", {"x": Numeric(), "s": Numeric()}, X, y)
    se = model.metrics(X, y).coefficient_se_raw
    assert se["x"][0] == pytest.approx(1.0 / np.sqrt(2e-16), rel=1e-6)
    assert np.isnan(se["s"][0])
    assert published_separation(model).names == ("s",)


# ------------------------------------------------------------------- the boundary's other cases
@pytest.mark.pins_behaviour
def test_a_log_binomial_success_at_the_finite_boundary_is_not_separation():
    """Control: under the log link a binomial mean reaches 1 at a finite predictor, so a level
    whose responses are all 1 sits on the mean space's edge, not at infinity: no separation is
    reported, as at ``92302bc40``."""
    rng = np.random.default_rng(5)
    f = rng.choice(list("ABC"), 600)
    y = (rng.uniform(size=600) < np.where(f == "A", 0.2, 0.4)).astype(float)
    y[f == "C"] = 1.0
    model = SuperGLM(
        family="binomial", link="log", features={"f": Categorical(base="A")}, selection_penalty=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"f": f}), y)
    separation = published_separation(model)
    assert separation is None or separation.state != "present"
    assert "no finite estimate" not in _notes(model)


def test_a_cauchit_fit_says_its_maximiser_need_not_be_unique():
    """The sign test proves a cauchit maximiser exists, not that it is unique (its likelihood is
    not concave), and the summary says so.  At ``92302bc40`` it said nothing."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=500)
    y = (rng.uniform(size=500) < 0.5 + np.arctan(0.8 * x) / np.pi).astype(float)
    model = SuperGLM(family="binomial", link="cauchit", features={"x": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"x": x}), y)
    assert "need not be unique" in _notes(model)
    assert published_separation(model).state == "absent"


# ------------------------------------------------------------------- D6: the resumed fit, mgcv
def _poisson_near_duplicate_frame(d: float):
    rng = np.random.default_rng(1)
    n = 2000
    z, x1, h = rng.uniform(size=n), rng.normal(size=n), rng.normal(size=n)
    y = rng.poisson(np.exp(-0.5 + 0.3 * x1 + np.sin(2 * np.pi * z))).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x1 + d * h, "z": z}), y


def _poisson_near_duplicate(oracle):
    X, y = _poisson_near_duplicate_frame(oracle["d"])
    model = SuperGLM(
        family="poisson", features={"x1": Numeric(), "x2": Numeric(), "z": Spline(n_knots=8)}
    )
    model.lambda2 = oracle["lambda"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, tol=1e-12, max_iter=500)
    return model, X, y


def test_a_poisson_near_duplicate_is_fitted_as_mgcv_fits_it():
    """``x2 = x1 + 1e-8 h`` under Poisson: the penalised fit identifies the pair (mgcv rank
    12/12), and the resumed IRLS on the identified subspace reaches mgcv's mode.

    The fixture is mgcv 1.9-3 run on superglm's own design and penalty at the pinned lambda
    (``paraPen``).  The solver truncates the pair's difference; the rows identify it, and the
    resumed fit takes joint Newton steps (the certified part's and the weak directions') to
    stationarity, so the coefficients (near ``+-3.1e6``), every SE^2 (mgcv's ``Vp``) and the EDF
    agree to 1e-6.  At ``92302bc40`` the coefficients stayed where the solver stopped (``x2 =
    0``) and the EDF was 9.7354 against mgcv's 9.7349.
    """
    oracle = json.loads((FIXTURES / "mgcv_poisson_near_duplicate.json").read_text())
    model, X, y = _poisson_near_duplicate(oracle)
    design = np.hstack(
        [np.asarray(m.toarray(), float).reshape(model._dm.n, -1) for m in model._dm.group_matrices]
    )
    assert float(np.sum(np.abs(design))) == pytest.approx(oracle["design_abs_sum"], rel=1e-12)
    assert oracle["rank"] == oracle["width"]
    metrics = model.metrics(X, y)
    se = np.concatenate(
        [[metrics.intercept_se_raw]]
        + [np.atleast_1d(metrics.coefficient_se_raw[g.name]) for g in model._groups]
    )
    np.testing.assert_allclose(se**2, oracle["Vp_diagonal"], rtol=1e-6)
    beta = np.r_[model.result.intercept, model.result.beta]
    mgcv = np.asarray(oracle["coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.effective_df == pytest.approx(oracle["edf"], rel=1e-6)


@pytest.mark.pins_behaviour
def test_a_resumed_fit_that_cannot_reach_stationarity_keeps_the_solvers_coefficients(
    monkeypatch,
):
    """Control: a resumption cut short (forced: one iteration) is not published; the fit keeps
    6a's coefficients along the weak direction (``x2 = 0``, where the solver stopped) and says
    so, as at ``92302bc40``."""
    _patch(monkeypatch, "superglm.model.state_ops", "_RESUME_ITERATIONS", 1)
    oracle = json.loads((FIXTURES / "mgcv_poisson_near_duplicate.json").read_text())
    model, _, _ = _poisson_near_duplicate(oracle)
    record = model._fit_inference_info.get("identifiability")
    assert record is not None and record.weak is not None and not record.weak.fitted
    assert model.result.beta[1] == 0.0
    assert "stay where the solver stopped" in _notes(model)


# ------------------------------------------------------------------- the score's proof, at size
def _strong_logistic(weighted: bool):
    """A healthy logistic fit with a strong slope: its rows reach far into both tails."""
    rng = np.random.default_rng(5)
    n = 20_000
    x = rng.normal(size=n)
    f = rng.choice(list("ABCDEFGH"), n)
    eta = -1.0 + 6.0 * x + 0.3 * (f == "B")
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    weights = rng.uniform(1e-4, 1.0, n) if weighted else None
    return pd.DataFrame({"x": x, "f": f}), y, weights


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "prior weights"])
def test_a_healthy_logistic_fit_proves_existence_from_its_score(monkeypatch, weighted):
    """The score proves a healthy logistic estimate finite; the rows' program is not run.

    A row deep in its tail has a multiplier about its own working weight, so the score's old
    test, ``min mu sqrt(lambda / max W)``, failed here (2e-9 against a defect of 6e-8 at 200,000
    rows); the weighted Cauchy-Schwarz form, ``min mu_i / sqrt(W_i) sqrt(lambda)``, clears it
    with room (gate review k3p8).  At ``07a1401a5`` both fits ran the program over all 20,000
    rows, found no direction, and printed that existence was not resolved at float64.
    Mutation check: with the unweighted margin the summary printed that note again.
    """
    import superglm.model.separation_ops as separation_ops

    ran = []
    real = separation_ops.separating_direction
    monkeypatch.setattr(
        separation_ops,
        "separating_direction",
        lambda matrix, sides, priority=None: ran.append(1) or real(matrix, sides, priority),
    )
    X, y, weights = _strong_logistic(weighted)
    model = SuperGLM(family="binomial", features={"x": Numeric(), "f": Categorical()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=weights)
    assert "not resolved" not in str(model.summary())
    assert ran == []
    separation = published_separation(model)
    assert (separation.state, separation.decided_by) == ("absent", "certificate")


def _counted_program(monkeypatch) -> list[int]:
    """Patch HiGHS so each call records the rows its program reads."""
    import scipy.optimize

    rows: list[int] = []
    real = scipy.optimize.linprog

    def counted(*args, **kwargs):
        rows.append(int(kwargs["A_ub"].shape[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", counted)
    return rows


def _separated_indicator_frame(n: int):
    """6b's ``dense-binomial-sep`` case at ``n`` rows: ``w = +-1`` on 2% of them separates them.

    A healthy logistic in a factor, a spline and a slope, and an indicator ``w`` whose rows all
    have response 1 where it is ``+1`` and 0 where it is ``-1``: ``w`` alone has no finite
    estimate.
    """
    rng = np.random.default_rng(7)
    f = rng.choice(list("ABCDEFGH"), n)
    x, z = rng.uniform(size=n), rng.normal(size=n)
    eta = -1.5 + 0.3 * (f == "B") + np.sin(2 * np.pi * x) + 0.2 * z
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    w = np.where(rng.uniform(size=n) < 0.02, rng.choice([-1.0, 1.0], n), 0.0)
    y = np.where(w > 0, 1.0, np.where(w < 0, 0.0, y))
    X = pd.DataFrame({"f": f, "x": x, "z": z, "w": w})
    features = {"f": Categorical(), "x": Spline(n_knots=10), "z": Numeric(), "w": Numeric()}
    return X, y, features


def _decided_past_the_cap(model, X, rows_read, cap):
    """What a user reads of a separation decided from a capped program, and its work."""
    se = _summary_se(model)
    assert np.isnan(se["w"]) and np.isfinite(se["z"])
    separation = published_separation(model)
    assert separation.state == "present" and separation.names == ("w",)
    np.testing.assert_array_equal(separation.rows, np.flatnonzero(X["w"].to_numpy() != 0))
    assert "not resolved" not in _notes(model)
    # one program, of at most the cap's entries, over a subset of the rows
    width = separation.work["candidates"]
    assert len(rows_read) == 1 and rows_read[0] * width <= cap
    assert separation.work["lp_rows"] * width > cap
    assert rows_read[0] == separation.work["sampled"] < separation.work["lp_rows"]


@pytest.mark.slow
def test_a_separation_past_the_programs_cap_is_decided_from_a_subset_of_its_rows(monkeypatch):
    """6b's ``dense-binomial-sep`` case, 200,000 rows by ten directions, is decided again.

    Its program (2,000,000 entries) exceeds the cap (``PROGRAM_CAP``, 524,288), so it reads
    52,428 of the rows (``program_rows``); the direction it proposes is verified on all
    200,000 exactly, and the fit on the rows it keeps certifies that it separates no more.  So
    ``w`` has no standard error, the fit warns once naming it, and the work is one program of
    at most the cap.  At ``9ac9df407`` the program was not run past the cap: the decision was
    unresolved and the summary printed ``w``'s walked standard error (17,835 at 4,000 rows).
    Mutation check: with the program not run past the cap, unresolved again.
    """
    from superglm.diagnostics.separation import PROGRAM_CAP

    rows_read = _counted_program(monkeypatch)
    X, y, features = _separated_indicator_frame(200_000)
    model = SuperGLM(family="binomial", features=features, selection_penalty=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(X, y)
    _decided_past_the_cap(model, X, rows_read, PROGRAM_CAP)
    stated = [str(w.message) for w in caught if str(w.message).startswith("Separation:")]
    assert len(stated) == 1 and "direction moving w" in stated[0]


def test_a_capped_program_reads_the_rows_the_walk_drove_to_the_boundary(monkeypatch):
    """Three of 4,000 rows are separated (``w = 1``, response 1); a cap of 80 entries lets the
    program read 40 rows, half of them the ones the walk drove nearest their boundary.

    The walk drives the three rows' predictor up, so they are among those 20, and the program
    finds ``w`` on 40 rows; the direction verifies on all 4,000, and ``w`` has no standard
    error.  An evenly strided sample of 40 rows holds none of the three.  At ``9ac9df407`` the
    program was not run past the cap: unresolved, and ``w``'s walked standard error printed.
    Mutation check: with the strided rows alone, no proposal and unresolved.
    """
    _patch(monkeypatch, "superglm.diagnostics.separation", "PROGRAM_CAP", 80)
    rows_read = _counted_program(monkeypatch)
    rng = np.random.default_rng(11)
    n = 4000
    w = np.zeros(n)
    w[[17, 1234, 3501]] = 1.0
    y = np.where(w > 0, 1.0, (rng.uniform(size=n) < 0.3).astype(float))
    X = pd.DataFrame({"w": w, "z": rng.normal(size=n)})
    model = _fit("binomial", {"w": Numeric(), "z": Numeric()}, X, y)
    _decided_past_the_cap(model, X, rows_read, 80)


def test_a_capped_program_whose_proposal_does_not_verify_is_disclosed_unresolved(monkeypatch):
    """Past its cap the program reads a subset of the rows; a direction that separates the
    subset but not every row leaves the decision unresolved, and the summary says how.

    Forced: the healthy logistic fit's score is kept from certifying it, and the cap is 8
    entries, so the program reads 4 of the 20,000 rows (two of each sign, by the walk's
    predictor), which the slope separates.  The proposal fails on the other rows, so nothing is
    withheld, and the footnote names the program's size, its cap and the rows it read.  Its
    work grows faster than its rows (56 s at 200,000 logistic rows of nine directions), so an
    uncapped program made a fit of that size take a minute.  At ``9ac9df407`` the program was
    not run past the cap, and the footnote said so.  Mutation checks: with the proposal checked
    on the rows it read only, the slope was called infinite; without the cap the program read
    every row.
    """
    _patch(monkeypatch, "superglm.diagnostics.separation", "PROGRAM_CAP", 8)
    _patch(
        monkeypatch, "superglm.model.separation_ops", "existence_certified", lambda *a, **k: False
    )
    rows_read = _counted_program(monkeypatch)
    X, y, _ = _strong_logistic(False)
    model = SuperGLM(family="binomial", features={"x": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    separation = published_separation(model)
    assert separation.state == "unresolved"
    assert rows_read == [4] and separation.work["sampled"] == 4
    assert (
        "linear program (20000 rows by 2 directions) exceeds its size cap of 8 entries; no "
        "direction it proposed from 4 of those rows separates the responses at 0 or 1 on every "
        "row" in _notes(model)
    )
    assert np.isfinite(_summary_se(model)["x"])
    assert getattr(model, "_separated_rows", None) is None


def test_complete_separation_leaves_a_penalised_term_its_estimate():
    """Complete separation drives only the penalty's null space to infinity (gate w5m2, sec. 3).

    Every response is ``u > 0.5``, so no row is kept.  A direction strict on every row stays
    so nearby, so every direction the penalty leaves free recedes: the intercept and ``u``.  A
    random effect's ridge curves every direction of its levels, so they keep finite estimates
    and standard errors, from the penalty alone (no row informs them, and the footnote says
    so).  At ``07a1401a5`` the span was every coordinate: ``g``'s levels too were called
    infinite and blanked.  Mutation check: with the identity span, ``g`` was named.
    """
    rng = np.random.default_rng(3)
    n = 300
    u = rng.uniform(size=n)
    g = rng.choice(list("ABCDEF"), n)
    X, y = pd.DataFrame({"u": u, "g": g}), (u > 0.5).astype(float)
    model = _fit("binomial", {"u": Numeric(), "g": RandomEffect()}, X, y, method="fit_reml")
    metrics = model.metrics(X, y)
    assert np.all(np.isfinite(metrics.coefficient_se["g"]))
    assert np.isnan(metrics.intercept_se) and np.isnan(metrics.coefficient_se["u"][0])
    assert "penalty alone" in _notes(model)
    separation = published_separation(model)
    assert separation.state == "present" and separation.complete
    assert separation.names == ("Intercept", "u")


def test_a_publication_reads_its_rows_once_and_refines_no_further_than_its_rounding(monkeypatch):
    """Work counts: one evaluation of the fitted rows per publication, one solve at its rounding.

    The record and the existence certificate read one evaluation of the fitted state's
    Fisher rows (``state_ops._publication_rows``), and a certificate whose defect is its own
    rounding takes no refining step, which could not lower it.  Here five rows sit at offset
    -95, so their multipliers are about ``e^-40`` and the score cannot prove existence; the
    interior rows do.  At ``07a1401a5`` the publication evaluated the linear predictor twice
    and the certificate solved three times.  Mutation checks: with the certificate forming
    its own rows the count was two; with refinement continuing past the rounding, three solves.
    """
    import superglm.model.separation_ops as separation_ops
    import superglm.model.state_ops as state_ops
    import superglm.solvers.mode_score as mode_score

    counts = {"eta": 0, "solves": [], "inside": False}
    real_eta, real_publish = mode_score.linear_predictor, state_ops.publish_identified_fit
    real_certificate = separation_ops.existence_certified

    def eta(*args, **kwargs):
        counts["eta"] += counts["inside"]
        return real_eta(*args, **kwargs)

    def publish(model):
        counts["inside"] = True
        try:
            return real_publish(model)
        finally:
            counts["inside"] = False

    class Counted:
        def __init__(self, inner, tally):
            self.inner, self.tally = inner, tally

        def solve(self, vector):
            self.tally.append(1)
            return self.inner.solve(vector)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def certificate(*args, **kwargs):
        tally: list[int] = []
        kwargs["decomposition"] = Counted(kwargs["decomposition"], tally)
        certified = real_certificate(*args, **kwargs)
        counts["solves"].append((len(tally), certified))
        return certified

    monkeypatch.setattr(mode_score, "linear_predictor", eta)
    monkeypatch.setattr(state_ops, "publish_identified_fit", publish)
    monkeypatch.setattr(separation_ops, "existence_certified", certificate)
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    offset = np.zeros(n)
    offset[:5] = -95.0
    y = rng.poisson(np.exp(-0.5 + 0.3 * x + offset)).astype(float)
    model = SuperGLM(family="poisson", features={"x": Numeric()})
    model.fit(pd.DataFrame({"x": x}), y, offset=offset)

    assert counts["eta"] == 1
    assert counts["solves"] == [(1, False)]
    assert published_separation(model).decided_by == "interior rank"
