"""``Constraint.postfit`` rules on SuperLSS terms (``distributional/shape_repair.py``).

A post-fit rule means what it means for SuperGLM: the term is fitted without it, and
``apply_shape_postfit`` projects the fitted curve onto the shape in the term's own basis,
re-estimates the parameter's intercept and withholds the term's inference. Each repair is checked
against an independent projection (a dense-grid QP solved through its NNLS dual) and the intercept against
``scipy.stats.gamma``'s likelihood.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from superglm import Constraint, GammaLS, SuperLSS, s
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.kernels.generalized_gamma import mean_exists
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning
from superglm.distributional.revision import EditedCoefficientsWarning, ShapeRepairedWarning
from superglm.distributional.shape_repair import POSTFIT_MOVES, PostfitState, postfit_status
from superglm.editor import edit
from superglm.editor.cv import InForce
from superglm.editor.lss.view import LSSModelView
from tests._shape_violation import curvature_violation, monotonicity_violation

_GRID = pd.DataFrame({"x": np.linspace(0.05, 9.95, 400)})


def _data(dip_in: str = "mean", *, n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    dip = 0.4 + 0.18 * x - 0.8 * np.exp(-(((x - 6.0) / 0.9) ** 2))
    if dip_in == "mean":
        y = rng.gamma(6.0, np.exp(dip) / 6.0)
    else:  # the scale (CV) carries the dip
        cv = np.exp(dip - 1.0)
        y = rng.gamma(1.0 / cv**2, np.exp(dip) * cv**2)
    return pd.DataFrame({"x": x}), y


def _fit(parameter: str, rule, *, kind: str, k: int, retain_rows: bool = True) -> SuperLSS:
    family = GammaLS()
    term = s("x", kind=kind, k=k, constraint=rule)
    mean, scale = (
        (family.mean(term), family.scale())
        if parameter == "mean"
        else (
            family.mean(),
            family.scale(term),
        )
    )
    X, y = _data(parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SuperLSS(family, mean, scale).fit_reml(X, y, retain_rows=retain_rows)


def _curve(model: SuperLSS, parameter: str) -> np.ndarray:
    return model.predict_link(_GRID)[parameter].to_numpy()


def _term_of(fitted, parameter: str):
    return next(p for p in fitted.compiled_predictors if p.name == parameter).compiled.specs["x"]


def _term(model: SuperLSS, parameter: str):
    fitted = model._model
    spec = next(p for p in fitted.compiled_predictors if p.name == parameter).compiled.specs["x"]
    return spec, fitted.layout.term_slices[f"{parameter}:x"]


def _oracle_projection(model: SuperLSS, parameter: str, kind: str, n_grid: int = 500):
    """The weighted projection of the fitted curve onto the shape, solved independently.

    The grid and its weights are SuperGLM's definition (a 500-bin histogram of the training
    axis plus one, read at evenly spaced points); the shape is imposed by finite-difference
    derivative rows at 3000 points, and the QP solved exactly through its non-negative
    least-squares dual (Lawson and Hanson 1974, ``scipy.optimize.nnls``).
    """
    spec, columns = _term(model, parameter)
    beta = np.asarray(model._model.result.coefficients)[columns]
    x = _data(parameter)[0]["x"].to_numpy()
    counts, edges = np.histogram(x, bins=n_grid)
    grid = np.linspace(spec._lo, spec._hi, n_grid)
    w = np.maximum(np.interp(grid, 0.5 * (edges[:-1] + edges[1:]), counts + 1.0), 1e-6)
    basis = spec.transform(grid)
    target = basis @ beta
    points = np.linspace(spec._lo, spec._hi, 3000)
    h = 1e-4 * (spec._hi - spec._lo)
    inner = np.clip(points, spec._lo + 2 * h, spec._hi - 2 * h)
    if kind == "increasing":
        rows = (spec.transform(inner + h) - spec.transform(inner - h)) / (2 * h)
    else:
        rows = (
            spec.transform(inner + h) - 2 * spec.transform(inner) + spec.transform(inner - h)
        ) / h**2
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    gram = basis.T @ (basis * w[:, None]) / w.sum()
    linear = basis.T @ (target * w) / w.sum()
    # min b'Gb/2 - l'b over R b >= 0 has the dual min ||L^-1 (R' lam + l)|| over lam >= 0,
    # G = L L', a non-negative least-squares problem, and b = G^-1 (l + R' lam).
    factor = np.linalg.cholesky(gram)
    solve = lambda v: np.linalg.solve(factor, v)  # noqa: E731
    lam, _ = optimize.nnls(solve(rows.T), -solve(linear), maxiter=50 * rows.shape[0])
    projected = np.linalg.solve(gram, linear + rows.T @ lam)
    return grid, w, target, basis @ projected


def _gamma_log_likelihood(model: SuperLSS, mean_shift: float, weights=None) -> float:
    """The book's log-likelihood under ``scipy.stats.gamma`` with the mean's intercept moved."""
    X, y = _data("mean")
    link = model.predict_link(X)
    mean = np.exp(link["mean"].to_numpy() + mean_shift)
    cv = np.exp(link["scale"].to_numpy())
    rows = stats.gamma.logpdf(y, a=1.0 / cv**2, scale=mean * cv**2)
    return float(np.sum(rows if weights is None else weights * rows))


def _intercept_slopes(model: SuperLSS, shift: float, weights=None) -> tuple[float, float]:
    """The likelihood's slope in the mean intercept at the model, and at ``shift`` back."""
    step = 1e-4

    def at(h: float) -> float:
        upper = _gamma_log_likelihood(model, h + step, weights)
        return (upper - _gamma_log_likelihood(model, h - step, weights)) / (2 * step)

    return at(0.0), at(-shift)


@pytest.mark.parametrize(
    ("parameter", "kind", "k", "rule"),
    [
        ("mean", "bs", 8, "increasing"),
        ("mean", "ps", 8, "convex"),
        ("scale", "ps", 10, "increasing"),
    ],
)
def test_a_postfit_rule_repairs_the_curve_of_any_parameter_to_the_projection(
    parameter, kind, k, rule
):
    model = _fit(parameter, getattr(Constraint.postfit, rule), kind=kind, k=k)
    before = _curve(model, parameter)
    violation = monotonicity_violation if rule == "increasing" else curvature_violation
    assert violation(before, rule) > 1e-5, "the fixture's fitted curve breaks the rule"

    assert model.apply_shape_postfit() is model
    after = _curve(model, parameter)
    assert violation(after, rule) <= 1e-9 * np.ptp(after)

    grid, w, target, oracle = _oracle_projection(model, parameter, rule)
    spec, columns = _term(model, parameter)
    repaired = spec.transform(grid) @ np.asarray(model._model.coefficients)[columns]
    # the same weighted least-squares projection, up to how each imposes the shape
    scale = np.sqrt(np.average(target**2, weights=w))
    assert np.max(np.abs(repaired - oracle)) <= 1e-3 * scale
    loss = np.average((repaired - target) ** 2, weights=w)
    oracle_loss = np.average((oracle - target) ** 2, weights=w)
    assert loss <= oracle_loss * (1 + 1e-3) + 1e-12 * scale**2
    # the fit's records still describe the fit
    assert not np.array_equal(model.result_.coefficients, model._model.coefficients)


def test_the_repair_re_estimates_the_intercept_on_the_likelihood():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    model.apply_shape_postfit()
    (record,) = model._model.fit_state.hand_edit.repairs
    assert record.intercept_shift != 0.0
    # the repaired mean's intercept maximizes scipy's gamma likelihood with everything else held
    slope, held = _intercept_slopes(model, record.intercept_shift)
    assert abs(slope) <= 1e-4 * abs(held)


@pytest.mark.parametrize("level", [-5.0, 5.0])
def test_the_intercept_is_re_estimated_however_far_an_edit_moved_the_level(level):
    # An editor edit can move the term's level far beyond what the projection moves the curve;
    # the search widens until it holds the likelihood's optimum.
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = edit(model, train_data=(X, y))
        session.select_x("mean:x", 0.0, 10.0)
        session.shift("mean:x", level)
        edited = session.to_model()
    edited.apply_shape_postfit()
    (record,) = edited._model.fit_state.hand_edit.repairs
    assert record.intercept_shift == pytest.approx(-level, abs=1.0)
    slope, held = _intercept_slopes(edited, record.intercept_shift)
    assert abs(slope) <= 1e-4 * abs(held)
    mean = edited.predict_parameters(X)["mean"].to_numpy()
    assert np.sum(y) / np.sum(mean) == pytest.approx(1.0, abs=0.05)


def test_a_repaired_term_withholds_its_inference_and_keeps_its_edf():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    edf = model.result_.term_edf["mean:x"]
    model.apply_shape_postfit()

    with pytest.warns(ShapeRepairedWarning, match="mean:x"):
        effect = model.term_inference("mean", "x", n_sim=200)
    assert np.all(np.isnan(effect.se)) and np.all(np.isnan(effect.lower))
    assert effect.lower_simultaneous is None and effect.critical_value is None
    assert effect.edf == pytest.approx(edf)
    with pytest.warns(ShapeRepairedWarning):
        test = model.term_test("mean", "x")
    assert np.isnan(test.p_value) and "inference withheld" in test.note

    with pytest.warns(ShapeRepairedWarning):
        table = model.summary()
    row = table[(table["parameter"] == "mean") & (table["term"] == "x")].iloc[0]
    assert np.isnan(row["p_value"]) and np.isnan(row["statistic"])
    assert row["edf"] == pytest.approx(edf)
    assert "inference withheld" in row["note"]

    # the repair re-estimated the mean's intercept, so the fit's inference does not describe it;
    # the scale's intercept was not moved, and keeps its inference
    intercepts = table[table["term"] == "(intercept)"].set_index("parameter")
    assert intercepts.loc["mean", "estimate"] == pytest.approx(
        model._model.coefficients[model._model.layout.predictor("mean").intercept_index]
    )
    assert np.isnan(intercepts.loc["mean", ["se", "statistic", "p_value"]].to_numpy(float)).all()
    assert "re-estimated after the post-fit repair of mean:x" in intercepts.loc["mean", "note"]
    assert np.isfinite(
        intercepts.loc["scale", ["se", "statistic", "p_value"]].to_numpy(float)
    ).all()
    with pytest.warns(ShapeRepairedWarning, match="intercept.*other rows.*before the repair"):
        model.summary()
    with pytest.warns(ShapeRepairedWarning, match="covariance of the fit before"):
        model.predict_link(_GRID, se_fit=True)


def test_an_unrepaired_term_keeps_its_inference_and_draws_no_warning():
    family = GammaLS()
    X, y = _data("mean")
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing)),
        family.scale(s("x", kind="ps", k=6)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    reference = model.term_inference("scale", "x", n_sim=200)
    model.apply_shape_postfit()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeRepairedWarning)
        effect = model.term_inference("scale", "x", n_sim=200)
    np.testing.assert_array_equal(effect.se, reference.se)


def test_before_the_repair_the_fit_is_unconstrained_and_says_so_without_refusing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    free = _fit("mean", None, kind="bs", k=8)
    np.testing.assert_array_equal(model.result_.coefficients, free.result_.coefficients)
    rendered = model.diagnose().render(detail="full")
    assert "has not been applied" in rendered and "apply_shape_postfit" in rendered
    notes = model.rating_table_payload(model._training_frame).notes
    assert any("mean:x" in note and "has not been applied" in note for note in notes)


def test_the_repair_is_disclosed_by_diagnose_rating_tables_and_the_editor():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, _ = _data("mean")
    view = LSSModelView(model, X)
    assert view.term_inference("mean:x").ci_lower is not None
    unrepaired_block = model.rating_table_payload(X).blocks()["mean:x"].table
    model.apply_shape_postfit()

    findings = [f for f in model.diagnose().findings if f.code == "shape.repair"]
    assert len(findings) == 1 and "mean:x" in findings[0].headline
    assert "repaired" in findings[0].headline and "withheld" in findings[0].headline

    payload = model.rating_table_payload(X)
    assert any("mean:x was repaired" in note for note in payload.notes)
    # the table is the repaired curve, not the fitted one
    repaired_block = payload.blocks()["mean:x"].table
    assert not repaired_block.equals(unrepaired_block)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeRepairedWarning)
        repaired = LSSModelView(model, X).term_inference("mean:x")
    assert repaired.ci_lower is None and repaired.ci_upper is None


def test_a_second_repair_and_an_obeyed_rule_change_nothing():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    model.apply_shape_postfit()
    revision = model._model
    model.apply_shape_postfit()
    assert model._model is revision

    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 10.0, 400)
    family = GammaLS()
    obeyed = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obeyed.fit_reml(pd.DataFrame({"x": x}), rng.gamma(6.0, np.exp(0.3 * x) / 6.0))
    assert monotonicity_violation(_curve(obeyed, "mean"), "increasing") == 0.0
    before = obeyed._model
    obeyed.apply_shape_postfit()
    assert obeyed._model is before and before.fit_state.hand_edit is None
    assert "needs no repair" in obeyed.diagnose().render(detail="full")


def test_a_repair_that_fits_worse_than_no_term_is_refused_and_nothing_is_published():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=10)
    before = model._model
    # the repaired curve fits the data better than no term; it loses on the penalty, and the
    # refusal says which objective it compared
    with pytest.raises(RuntimeError, match="penalized objective.*smoothing penalty") as refused:
        model.apply_shape_postfit()
    assert "fits worse" not in str(refused.value)
    assert "Constraint.fit.increasing" in str(refused.value)
    assert model._model is before


def test_a_repair_that_leaves_a_row_without_a_finite_mean_is_refused():
    # Mean-form generalized gamma has a finite mean only where scale * |shape| < 1 for a
    # negative shape. Every row has one at this revision; the decreasing repair of the shape
    # flattens it to about -0.6, which leaves the rows of largest scale without one.
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x})
    family = GeneralizedGammaLSS()
    model = SuperLSS(
        family,
        family.mean(),
        family.scale(s("x", kind="bs", k=6)),
        family.shape(s("x", kind="bs", k=6, constraint=Constraint.postfit.decreasing)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, np.random.default_rng(0).gamma(4.0, 0.25, 200))
    fitted = model._model
    beta = np.array(fitted.coefficients, dtype=np.float64)
    targets = {"scale": np.log(0.49) + x * np.log(1.99 / 0.49), "shape": -1.0 + 0.8 * x}
    for parameter, target in targets.items():
        spec = _term_of(fitted, parameter)
        design = np.column_stack([np.ones(len(x)), spec.transform(x)])
        solution = np.linalg.lstsq(design, target, rcond=None)[0]
        beta[fitted.layout.predictor(parameter).intercept_index] = solution[0]
        beta[fitted.layout.term_slices[f"{parameter}:x"]] = solution[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        revised = model._with_revised_coefficients(beta, edited_terms=["scale:x", "shape:x"])
    theta = revised.predict_parameters(X)
    assert mean_exists(theta["scale"].to_numpy(), theta["shape"].to_numpy()).all()
    before = revised._model
    with pytest.raises(RuntimeError, match="shape:x.*training rows have no valid parameters"):
        revised.apply_shape_postfit()
    assert revised._model is before
    revised.predict_cdf(X, np.ones(len(x)))


def test_a_repair_needs_the_fitted_rows():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8, retain_rows=False)
    with pytest.raises(RuntimeError, match="retain_rows=True"):
        model.apply_shape_postfit()


def test_fit_time_and_postfit_rules_combine_as_they_do_on_superglm():
    # one term: Constraint.postfit takes no combination, whichever model it is declared for
    with pytest.raises(ValueError, match="fit-time only"):
        Constraint.fit.increasing & Constraint.postfit.convex
    # two terms: the fit-time rule is enforced in the fit, the post-fit one repaired afterwards
    family = GammaLS()
    X, y = _data("mean")
    X = X.assign(z=np.random.default_rng(9).uniform(0.0, 10.0, len(X)))
    model = SuperLSS(
        family,
        family.mean(
            s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing),
            s("z", kind="ps", k=6, constraint=Constraint.fit.increasing),
        ),
        family.scale(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    assert model._model.fit_state.shape_fit is not None
    model.apply_shape_postfit()
    grid = pd.DataFrame({"x": _GRID["x"], "z": np.full(len(_GRID), 5.0)})
    assert monotonicity_violation(model.predict_link(grid)["mean"].to_numpy(), "increasing") <= 1e-9
    z_grid = pd.DataFrame({"x": np.full(len(_GRID), 5.0), "z": _GRID["x"]})
    z_curve = model.predict_link(z_grid)["mean"].to_numpy()
    assert monotonicity_violation(z_curve, "increasing") <= 1e-9 * max(np.ptp(z_curve), 1.0)
    codes = {f.code for f in model.diagnose().findings}
    assert {"shape.rule", "shape.repair"} <= codes


def test_a_repaired_model_pickles_with_its_repair():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    model.apply_shape_postfit()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.predict_link(_GRID)["mean"].to_numpy(), _curve(model, "mean")
    )
    assert restored._model.fit_state.hand_edit.repaired_terms == ("mean:x",)


def test_a_restored_model_is_repaired_with_the_frame_it_was_fitted_on():
    # A pickle carries no training frame, and the fit dropped the zero-weight rows: the frame
    # the user fitted on is the one they have.
    family = GammaLS()
    X, y = _data("mean")
    weights = np.ones(len(y))
    weights[:20] = 0.0
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing)),
        family.scale(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y, sample_weight=weights)
    restored = pickle.loads(pickle.dumps(model))
    restored.apply_shape_postfit(X)
    model.apply_shape_postfit()
    np.testing.assert_allclose(_curve(restored, "mean"), _curve(model, "mean"), rtol=1e-12)
    assert restored._model.fit_state.hand_edit.repaired_terms == ("mean:x",)
    # the intercept is re-estimated on the rows the fit kept
    (record,) = restored._model.fit_state.hand_edit.repairs
    slope, held = _intercept_slopes(restored, record.intercept_shift, weights)
    assert abs(slope) <= 1e-4 * abs(held)


def _break_in_the_editor(model: SuperLSS) -> SuperLSS:
    """Lower the 6-7 stretch of the repaired curve in the editor, which breaks increasing."""
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = edit(model, train_data=(X, y))
        session.select_x("mean:x", 6, 7)
        session.shift("mean:x", -0.4)
        return session.to_model()


def test_a_repaired_curve_edited_to_break_its_rule_is_disclosed_and_repaired_once_more():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, _ = _data("mean")
    model.apply_shape_postfit()
    broken = _break_in_the_editor(model)
    assert monotonicity_violation(_curve(broken, "mean"), "increasing") > 1e-5

    (finding,) = [f for f in broken.diagnose().findings if f.code == "shape.repair"]
    assert finding.severity == "warning" and "breaks it again" in finding.headline
    notes = [note for note in broken.rating_table_payload(X).notes if "mean:x" in note]
    assert len(notes) == 1 and "edited since" in notes[0] and "not increasing" in notes[0]

    broken.apply_shape_postfit()
    assert monotonicity_violation(_curve(broken, "mean"), "increasing") <= 1e-9
    # one current repair: the new one replaced the first
    assert [r.term for r in broken._model.fit_state.hand_edit.repairs] == ["mean:x"]
    (finding,) = [f for f in broken.diagnose().findings if f.code == "shape.repair"]
    assert finding.severity == "info" and "repaired the fitted curve" in finding.headline
    notes = [note for note in broken.rating_table_payload(X).notes if "mean:x" in note]
    assert len(notes) == 1 and "was repaired" in notes[0] and "edited since" not in notes[0]


def test_each_move_of_a_postfit_rule_lands_where_its_table_says():
    X, y = _data("mean")
    walked: set[tuple[PostfitState, str]] = set()

    def state(model: SuperLSS) -> PostfitState:
        (status,) = postfit_status(model._model)
        return status.state

    def move(model: SuperLSS, event: str, data=(X, y)) -> SuperLSS:
        before = state(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if event == "repair":
                moved = model.apply_shape_postfit()
            elif event == "refit":
                moved = copy.copy(model).fit_reml(*data)
            else:  # an edit that keeps the curve, or one that puts the fit's curve back
                beta = np.array(model._model.coefficients, dtype=np.float64)
                if event == "edit back":
                    columns = model._model.layout.term_slices["mean:x"]
                    beta[columns] = model.result_.coefficients[columns]
                moved = model._with_revised_coefficients(beta, edited_terms=["mean:x"])
        key = (before, "edit" if event.startswith("edit") else event)
        assert state(moved) in POSTFIT_MOVES[key], (key, state(moved))
        walked.add(key)
        return moved

    obeyed, obeyed_data = _obeyed_fit()
    assert state(obeyed) is PostfitState.OBEYED
    for event in ("repair", "edit", "refit"):
        move(obeyed, event, obeyed_data)

    pending = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    assert state(pending) is PostfitState.PENDING
    move(pending, "edit")
    move(pending, "refit")
    repaired = move(copy.copy(pending), "repair")
    assert state(repaired) is PostfitState.REPAIRED
    move(repaired, "repair")
    move(repaired, "edit")
    move(repaired, "refit")
    broken = move(repaired, "edit back")
    assert state(broken) is PostfitState.BROKEN
    move(broken, "edit")
    move(broken, "refit")
    assert state(move(broken, "repair")) is PostfitState.REPAIRED
    assert walked == set(POSTFIT_MOVES)


def _obeyed_fit():
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 10.0, 400)
    X, y = pd.DataFrame({"x": x}), rng.gamma(6.0, np.exp(0.3 * x) / 6.0)
    family = GammaLS()
    obeyed = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return obeyed.fit_reml(X, y), (X, y)


def test_an_open_editor_session_sees_a_repair_made_on_its_model():
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = edit(model, train_data=(X, y))
    assert session.model.term_inference("mean:x").ci_lower is not None
    in_force = InForce.of(session)
    assert in_force.describes(session)

    model.apply_shape_postfit()
    # a result made before the repair is not current: predictions moved
    assert not in_force.describes(session)
    assert InForce.of(session).describes(session)
    # the view follows the repair: the repaired curve, with its band withheld rather than NaN
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeRepairedWarning)
        repaired = session.model.term_inference("mean:x")
    assert repaired.ci_lower is None and repaired.ci_upper is None
    spec, columns = _term(model, "mean")
    np.testing.assert_allclose(
        session.model.term_coefficients("mean:x"), model._model.coefficients[columns]
    )
