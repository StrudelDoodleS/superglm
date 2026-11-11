"""``Constraint.postfit`` rules on SuperLSS terms (``distributional/shape_repair.py``).

A post-fit rule means what it means for SuperGLM: the term is fitted without it, and
``apply_shape_postfit`` projects the fitted curve onto the shape in the term's own basis,
re-estimates the parameter's intercept and withholds the term's inference. Each repair is checked
against an independent projection (a dense-grid QP solved through its NNLS dual) and the intercept against
``scipy.stats.gamma``'s likelihood.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from superglm import Constraint, GammaLS, SuperLSS, s
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning
from superglm.distributional.revision import ShapeRepairedWarning
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


def _gamma_log_likelihood(model: SuperLSS, mean_shift: float) -> float:
    """The book's log-likelihood under ``scipy.stats.gamma`` with the mean's intercept moved."""
    X, y = _data("mean")
    link = model.predict_link(X)
    mean = np.exp(link["mean"].to_numpy() + mean_shift)
    cv = np.exp(link["scale"].to_numpy())
    return float(np.sum(stats.gamma.logpdf(y, a=1.0 / cv**2, scale=mean * cv**2)))


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
    step = 1e-4
    slope = (_gamma_log_likelihood(model, step) - _gamma_log_likelihood(model, -step)) / (2 * step)
    held = (
        _gamma_log_likelihood(model, step - record.intercept_shift)
        - _gamma_log_likelihood(model, -step - record.intercept_shift)
    ) / (2 * step)
    assert abs(slope) <= 1e-4 * abs(held)


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

    # an intercept is not a repaired term: its standard error stays
    intercepts = table[table["term"] == "(Intercept)"]
    assert np.all(np.isfinite(intercepts["se"].to_numpy()))
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
    with pytest.raises(RuntimeError, match="worse than dropping the term"):
        model.apply_shape_postfit()
    assert model._model is before


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
