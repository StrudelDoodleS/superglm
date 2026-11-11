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

from superglm import Constraint, GammaLS, SuperLSS, cat, s
from superglm._frame import as_eager_frame
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


def test_a_column_named_like_a_free_level_block_keeps_its_repair_record():
    # A column whose own name ends in ":special" is a term of its own. At f6346704d the suffix
    # was stripped before the exact match, so its inference was quoted, unwithheld and unwarned.
    family = GammaLS()
    X, y = _data("mean")
    X = X.rename(columns={"x": "x:special"})
    rule = Constraint.postfit.increasing
    model = SuperLSS(
        family, family.mean(s("x:special", kind="bs", k=8, constraint=rule)), family.scale()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    model.apply_shape_postfit()
    assert model._model.fit_state.hand_edit.repaired_terms == ("mean:x:special",)
    with pytest.warns(ShapeRepairedWarning, match="mean:x:special"):
        effect = model.term_inference("mean", "x:special")
    assert np.all(np.isnan(effect.se))
    with pytest.warns(ShapeRepairedWarning, match="mean:x:special"):
        test = model.term_test("mean", "x:special")
    assert np.isnan(test.p_value)


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


def _profile_optimum(model: SuperLSS, parameter: str, X, y) -> float:
    """The further shift of ``parameter``'s intercept that minimizes the model's own log score,
    found by SciPy's bounded Brent on the public scoring path."""
    fitted = model._model
    index = fitted.layout.predictor(parameter).intercept_index
    beta = np.array(fitted.coefficients, dtype=np.float64)
    edited = list(fitted.fit_state.hand_edit.edited_terms)

    def log_score(shift: float) -> float:
        moved = beta.copy()
        moved[index] += shift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            there = model._with_revised_coefficients(moved, edited_terms=edited)
            value = float(there.scores(X, y, which=("log",))["log"].sum())
        return value if np.isfinite(value) else 1e300

    search = optimize.minimize_scalar(
        log_score, bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-9}
    )
    return float(search.x)


def test_a_repair_that_needs_its_intercept_to_keep_every_row_valid_is_published():
    # Mean-form generalized gamma has a finite mean only where scale * |shape| < 1 for a
    # negative shape. Every row has one at this revision. The decreasing repair of the shape
    # flattens it to about -0.6, which at the held intercept leaves the rows of largest scale
    # without one; the re-estimated intercept gives every row one again, so the repair is
    # published. At b7cc25f0c it was refused: the comparison with the held, infinite
    # likelihood was NaN and threw the re-estimate away.
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
        model.fit_reml(X, y := np.random.default_rng(0).gamma(4.0, 0.25, 200))
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

    revised.apply_shape_postfit()
    (record,) = revised._model.fit_state.hand_edit.repairs
    assert record.term == "shape:x"
    # The re-estimate is the profile's optimum. At f6346704d the search started from its
    # farthest valid probe and published +419,428.8, where predict_cdf(X, 1) was 0 on every row.
    assert record.intercept_shift == pytest.approx(1.18338, abs=1e-4)
    assert _profile_optimum(revised, "shape", X, y) == pytest.approx(0.0, abs=1e-5)
    assert revised.predict_cdf(X, np.ones(len(x))).min() > 0.5
    theta = revised.predict_parameters(X)
    assert mean_exists(theta["scale"].to_numpy(), theta["shape"].to_numpy()).all()
    assert monotonicity_violation(theta["shape"].to_numpy(), "decreasing") == 0.0
    assert np.isfinite(revised.predict_cdf(X, np.ones(len(x)))).all()
    # the scenario: without the re-estimated intercept, rows had no mean
    held = np.array(revised._model.coefficients, dtype=np.float64)
    held[revised._model.layout.predictor("shape").intercept_index] -= record.intercept_shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta = revised._with_revised_coefficients(held, edited_terms=["shape:x"])
        theta = theta.predict_parameters(X)
    assert not mean_exists(theta["scale"].to_numpy(), theta["shape"].to_numpy()).all()


def _generalized_gamma_with_a_narrow_band(spread: float):
    """Codex's revision: a scale offset falling ``spread`` over x, the scale link predictor
    ``0.5 - x`` and the shape ``-0.01 - 1.49 x``. Flattening the scale's term to increasing
    leaves valid only the shifts between an overflow wall below and a mean wall above."""
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x, "off": -spread + (spread - 5.0) * x})
    family = GeneralizedGammaLSS()
    model = SuperLSS(
        family,
        family.mean(),
        family.scale(
            s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing), offset="off"
        ),
        family.shape(s("x", kind="bs", k=6)),
    )
    y = np.random.default_rng(0).gamma(4.0, 0.25, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    fitted = model._model
    beta = np.array(fitted.coefficients, dtype=np.float64)
    targets = {"scale": 0.5 - x - X["off"].to_numpy(), "shape": -0.01 - 1.49 * x}
    for parameter, target in targets.items():
        design = np.column_stack([np.ones(len(x)), _term_of(fitted, parameter).transform(x)])
        solution = np.linalg.lstsq(design, target, rcond=None)[0]
        beta[fitted.layout.predictor(parameter).intercept_index] = solution[0]
        beta[fitted.layout.term_slices[f"{parameter}:x"]] = solution[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        revised = model._with_revised_coefficients(beta, edited_terms=["scale:x", "shape:x"])
    return revised, X


def test_a_valid_band_between_two_intercept_probes_is_found():
    # Codex's case. The valid shifts lie between -27.5 and -15.42, inside the gap between the
    # probes -15.5 and -31 (in the scale's own units, -0.5 and -1); at 19a6ab028 the search
    # tried only the probes and refused. The band's sides are told apart by the rows invalid
    # at each midpoint, and the published shift sits on the mean wall, where the likelihood
    # still improves.
    revised, X = _generalized_gamma_with_a_narrow_band(35.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        revised.apply_shape_postfit()
    (record,) = revised._model.fit_state.hand_edit.repairs
    assert record.intercept_shift == pytest.approx(-15.42058, abs=1e-4)
    theta = revised.predict_parameters(X)
    sigma, shape = theta["scale"].to_numpy(), theta["shape"].to_numpy()
    assert mean_exists(sigma, shape).all()
    assert [status.state for status in postfit_status(revised._model)] == [PostfitState.REPAIRED]
    # the wall: a further 1e-5 leaves a row with no mean
    assert not mean_exists(sigma * np.exp(1e-5), shape).all()


def test_the_repair_is_refused_when_no_intercept_shift_leaves_every_row_valid():
    # With the offset falling 60 rather than 35, the overflow wall lies above the mean wall:
    # no shift is valid, as a grid of 6001 shifts confirms. The gap search ends within its
    # budget and the repair is refused with nothing published.
    revised, X = _generalized_gamma_with_a_narrow_band(60.0)
    before = revised._model
    evaluations = []
    real = GeneralizedGammaLSS.evaluate_natural

    def counted(self, *args, **kwargs):
        evaluations.append(1)
        return real(self, *args, **kwargs)

    with warnings.catch_warnings(), pytest.MonkeyPatch.context() as patch:
        warnings.simplefilter("ignore")
        patch.setattr(GeneralizedGammaLSS, "evaluate_natural", counted)
        with pytest.raises(RuntimeError, match="rejected before publication"):
            revised.apply_shape_postfit()
    assert revised._model is before
    # the held intercept, 122 probes, the held shift again and the gap search's budget
    assert len(evaluations) <= 1 + 122 + 1 + 256 + 4


def _gaussian_in_units(unit: float) -> SuperLSS:
    """A Gaussian location repair with the response and its known sigma in ``unit``."""
    from superglm import GaussianLS

    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 10.0, 400)
    sigma = 0.1 * np.exp(0.2 * x)
    y = 0.4 + 0.18 * x - 0.8 * np.exp(-(((x - 6.0) / 0.9) ** 2)) + sigma * rng.normal(size=400)
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family,
        family.location(s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing)),
        family.scale(offset=np.log(unit * sigma)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"x": x}), unit * y, lambdas={"location:x#wiggle": 0.0})
    return model.apply_shape_postfit()


def test_the_re_estimated_intercept_does_not_depend_on_the_response_units():
    # On an identity link the intercept carries the response's units. At b7cc25f0c the search
    # had a fixed radius of 1 and an absolute tolerance of 1e-10, so in units of 1e-15 it
    # published no shift at all.
    (natural,) = _gaussian_in_units(1.0)._model.fit_state.hand_edit.repairs
    (tiny,) = _gaussian_in_units(1e-15)._model.fit_state.hand_edit.repairs
    assert abs(natural.intercept_shift) > 1e-3
    # Brent locates the optimum to sqrt(u) of the location's range on the link
    np.testing.assert_allclose(tiny.intercept_shift / 1e-15, natural.intercept_shift, rtol=1e-5)


def test_the_intercept_is_re_estimated_however_far_a_revision_moved_it_in_scale_units():
    # Sol's case: a response of scale 1e-9 whose location intercept a revision moved by 0.2,
    # 2e8 of its own units. At f6346704d the bracket widened at most 6 times from 4 units and
    # refused at a shift of -0.000562; it now widens until the likelihood rises.
    from superglm import GaussianLS

    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x, "off": np.full(200, np.log(1e-9))})
    y = 1.0 + 1e-9 * (-x + 0.1 * np.random.default_rng(3).normal(size=200))
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family,
        family.location(s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing)),
        family.scale(offset="off"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas={"location:x#wiggle": 0.0})
        fitted = model._model
        beta = np.array(fitted.coefficients, dtype=np.float64)
        beta[fitted.layout.predictor("location").intercept_index] += 0.2
        revised = model._with_revised_coefficients(beta, edited_terms=["location:x"])
        revised.apply_shape_postfit()
    (record,) = revised._model.fit_state.hand_edit.repairs
    assert record.intercept_shift == pytest.approx(-0.2, abs=1e-6)
    # with the scale held, the location intercept's optimum is the precision-weighted mean
    # residual's zero: here to a millionth of the response's scale
    theta = revised.predict_parameters(X)
    precision = 1.0 / theta["scale"].to_numpy() ** 2
    residual = y - theta["location"].to_numpy()
    assert abs(np.sum(precision * residual) / np.sum(precision)) <= 1e-6 * 1e-9


def test_a_probe_whose_likelihood_overflows_is_a_wall_of_the_intercept_search():
    # Sol's case: the scale's offset runs from -100 to 0, so a trial shift of the scale
    # intercept far enough down overflows the Gaussian likelihood. At 19a6ab028 that raised
    # ValueError at the search's lower probe instead of bounding the search there.
    from superglm import GaussianLS

    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x, "off": -100.0 + 100.0 * x, "zero": np.zeros(200)})
    y = np.exp(X["off"].to_numpy() + 0.2 * np.sin(6.0 * x))
    y = y * np.random.default_rng(3).normal(size=200)
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family,
        family.location("zero", intercept=False),
        family.scale(
            s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing), offset="off"
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas={"scale:x#wiggle": 0.0}, max_inner_iter=500)
        model.apply_shape_postfit()
    (status,) = postfit_status(model._model)
    assert status.state is PostfitState.REPAIRED
    theta = model.predict_parameters(X)
    log_likelihood = stats.norm.logpdf(y, theta["location"], theta["scale"]).sum()
    assert np.isfinite(log_likelihood)
    # the published intercept is the profile's optimum
    assert abs(_profile_optimum(model, "scale", X, y)) <= 1e-5


def _perturbed_repairer(monkeypatch, block):
    """Make the shared repairer return ``block(fitted block)`` for the repaired term."""
    import dataclasses

    from superglm.model import shape_ops

    real = shape_ops._repairer

    class Perturbed:
        def __init__(self, kind):
            self.inner = real(kind)

        def repair(self, spec, coefficients, groups, **kwargs):
            result = self.inner.repair(spec, coefficients, groups, **kwargs)
            beta = np.array(result.repaired_beta_reparam, dtype=np.float64)
            columns = groups[0].sl
            beta[columns] = block(np.asarray(coefficients, dtype=np.float64)[columns])
            return dataclasses.replace(result, repaired_beta_reparam=beta)

    monkeypatch.setattr(shape_ops, "_repairer", Perturbed)


def test_a_projection_onto_the_apex_is_published_as_the_zero_curve(monkeypatch):
    # The projection certifies at the fitted curve's scale; the state is read at the repaired
    # curve's. A result that is zero only to the first scale's rounding read as BROKEN right
    # after its repair at b7cc25f0c, a move POSTFIT_MOVES does not have. Forced here: the
    # repairer returns the fitted, rule-breaking block times 1e-14.
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    _perturbed_repairer(monkeypatch, lambda fitted: 1e-14 * fitted)
    (before,) = postfit_status(model._model)
    model.apply_shape_postfit()
    (after,) = postfit_status(model._model)
    assert after.state in POSTFIT_MOVES[(before.state, "repair")]
    assert after.state is PostfitState.REPAIRED
    _, columns = _term(model, "mean")
    assert not np.any(np.asarray(model._model.coefficients)[columns])
    repaired = model._model
    model.apply_shape_postfit()
    assert model._model is repaired


def test_a_projection_that_breaks_its_rule_at_its_own_scale_is_refused(monkeypatch):
    # Half the fitted curve breaks the rule as the fitted curve does; at b7cc25f0c it was
    # published and read as BROKEN at once.
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    _perturbed_repairer(monkeypatch, lambda fitted: 0.5 * fitted)
    before = model._model
    with pytest.raises(RuntimeError, match="mean:x rejected.*at its own scale"):
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
    # the page's token must change wherever ``describes`` does, a repair in place included
    assert in_force.token != InForce.of(session).token


def _open_then_repair(*, edit_before=None):
    """An editor session opened on a fit, then the model repaired in place behind it."""
    from superglm.editor.payloads import session_payload

    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = edit(model, train_data=(X, y))
        if edit_before is not None:
            edit_before(session)
        before = session_payload(session)["mean:x"]
        revision = session.model_revision
        model.apply_shape_postfit()
        after = session_payload(session)["mean:x"]
    return model, session, before, after, revision


def test_an_open_session_shows_the_repaired_curve_without_its_band():
    # At b7cc25f0c the page kept the curve and band read when the session opened.
    model, session, before, after, revision = _open_then_repair()
    assert before["ci_lower_y"] is not None
    assert after["ci_lower_y"] is None and after["ci_upper_y"] is None
    assert session.model_revision > revision
    drawn = np.log(np.asarray(after["y"]))
    assert monotonicity_violation(drawn, "increasing") <= 1e-12
    curve = session.model.term_inference("mean:x")
    np.testing.assert_allclose(drawn, np.asarray(curve.log_relativity), atol=1e-12)
    assert np.max(np.abs(drawn - np.log(np.asarray(before["y"])))) > 0.05
    # and an edit made now moves the repaired curve: shifting all of it by 0.1 moves every
    # prediction by 0.1, rather than bringing back the fit's dip
    X, _ = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.select_x("mean:x", 0.0, 10.0)
        session.shift("mean:x", 0.1)
        moved = session.to_model().predict_link(X)["mean"] - model.predict_link(X)["mean"]
    np.testing.assert_allclose(moved, 0.1, atol=1e-9)


def test_an_edit_made_before_an_in_place_repair_carries_over_onto_the_repaired_curve():
    def shift(session):
        session.select_x("mean:x", 2.0, 4.0)
        session.shift("mean:x", 0.2)

    model, session, before, after, _ = _open_then_repair(edit_before=shift)
    term = session.terms["mean:x"]
    np.testing.assert_allclose(
        term.original_log_effect,
        session.model.term_inference("mean:x").log_relativity,
        atol=1e-12,
    )
    chosen = (term.x >= 2.0) & (term.x <= 4.0)
    change = term.edited_log_effect - term.original_log_effect
    np.testing.assert_allclose(change[chosen], 0.2, atol=1e-12)
    np.testing.assert_allclose(change[~chosen], 0.0, atol=1e-12)
    assert after["edited"]
    # its undo takes the curve back to the repaired one, not the fit's
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.undo()
    np.testing.assert_array_equal(
        term.original_log_effect, session.terms["mean:x"].original_log_effect
    )
    np.testing.assert_allclose(
        session.terms["mean:x"].edited_log_effect, term.original_log_effect, atol=1e-12
    )
    assert not session.edited_terms()


def test_with_an_edit_pending_the_evidence_model_follows_an_in_place_repair():
    # The metrics and reports read the edited model from the session's cache. At b7cc25f0c an
    # in-place repair changed neither its epoch, its revision nor the model's identity, so they
    # kept the model from before the repair, marked current.
    from superglm.editor.widget import EditorWidget

    rng = np.random.default_rng(3)
    X, y = _data("mean")
    X = X.assign(z=rng.uniform(0.0, 1.0, len(X)))
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(
            s("x", kind="bs", k=8, constraint=Constraint.postfit.increasing),
            s("z", kind="bs", k=6),
        ),
        family.scale(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
        session = edit(model, train_data=(X, y))
        widget = EditorWidget(session, mode="server")
        session.select_x("mean:z", 0.2, 0.6)
        session.shift("mean:z", 0.2)
        stale, _ = widget._current_model_for_evidence()
        revision = session.model_revision
        model.apply_shape_postfit()
        current, _ = widget._current_model_for_evidence()
        wanted = session.to_model()
    assert session.model_revision > revision and current is not stale
    assert current._model.fit_state.hand_edit.repaired_terms == ("mean:x",)
    np.testing.assert_allclose(
        current.predict_link(X)["mean"], wanted.predict_link(X)["mean"], atol=1e-12
    )


def _session_with(*edits):
    """The PR's Gamma ``bs``/k=8 mean fit, an editor session on it, and ``edits`` made on it."""
    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = edit(model, train_data=(X, y))
        for start, stop, delta in edits:
            session.select_x("mean:x", start, stop)
            session.shift("mean:x", delta)
    return model, session


def _off_by(session, edits) -> float:
    """How far the session's edited curve is from the repaired curve plus ``edits``."""
    term = session.terms["mean:x"]
    repaired = session.model.term_inference("mean:x").log_relativity  # the model's, repaired
    wanted = np.asarray(repaired, dtype=np.float64).copy()
    for start, stop, delta in edits:
        wanted[(term.x >= start) & (term.x <= stop)] += delta
    return float(np.max(np.abs(term.edited_log_effect - wanted)))


_EDIT_A, _EDIT_B = (2.0, 4.0, 0.2), (3.0, 5.0, -0.1)


def test_undo_and_redo_straight_after_an_in_place_repair_act_on_the_repaired_curve():
    # The review's sequence, with nothing read between the repair and the undo. At c3c56d40a
    # undo took its record before the session followed the repair, so it wrote the fit's
    # pre-repair values back: 0.156, 0.143 and 0.143 off after undo, undo and redo.
    model, session = _session_with(_EDIT_A, _EDIT_B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.apply_shape_postfit()
        session.undo()
        assert _off_by(session, [_EDIT_A]) <= 1e-12
        session.undo()
        assert _off_by(session, []) <= 1e-12 and not session.edited_terms()
        session.redo()
        assert _off_by(session, [_EDIT_A]) <= 1e-12


def test_redo_straight_after_an_in_place_repair_redoes_the_same_change():
    # At c3c56d40a the redone change on A's points was [0.198, 0.356] where 0.2 is wanted.
    model, session = _session_with(_EDIT_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.undo()
        model.apply_shape_postfit()
        session.redo()
    assert _off_by(session, [_EDIT_A]) <= 1e-12


def test_undo_of_a_whole_curve_shift_straight_after_a_repair_restores_the_repaired_curve():
    # Sol's case: at c3c56d40a the undo brought back the fit's dip (0.1879 off), the term stayed
    # edited and the model it published read BROKEN.
    model, session = _session_with((0.0, 10.0, 0.2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.apply_shape_postfit()
        session.undo()
        published = session.to_model()
    assert _off_by(session, []) <= 1e-12 and not session.edited_terms()
    (status,) = postfit_status(published._model)
    assert status.state is PostfitState.REPAIRED


def test_the_page_undo_straight_after_a_repair_acts_on_the_repaired_curve():
    # The page's Undo posts no term, so nothing reads the terms before the undo.
    from superglm.editor.widget import EditorWidget

    model, session = _session_with(_EDIT_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widget = EditorWidget(session, mode="server")
        model.apply_shape_postfit()
        widget._operate("undo")
    assert _off_by(session, []) <= 1e-12


def test_a_session_follows_a_repair_only_under_the_widget_lock_and_its_getters_change_nothing(
    monkeypatch,
):
    # At c3c56d40a reading ``model_revision`` followed the repair, and the widget read it
    # before taking its lock, so the follow could rewrite the history beside a request.
    from superglm.editor.widget import EditorWidget

    model, session = _session_with(_EDIT_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widget = EditorWidget(session, mode="server")
        model.apply_shape_postfit()
    history, terms, revision = session.history, session.terms, session.model_revision
    assert session.edit_epoch == session._edit_epoch and session.history is history
    assert session.terms is terms and session.model_revision == revision

    owned = []
    follow = session._carry_onto_the_fit_in_force
    monkeypatch.setattr(
        session,
        "_carry_onto_the_fit_in_force",
        lambda: owned.append(widget._lock._is_owned()) or follow(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widget._metrics("deviance")
    assert owned == [True] and session.model_revision > revision


def test_the_intercept_is_re_estimated_where_valid_shifts_are_a_narrow_interval():
    # Sol's case: a generalized gamma whose scale rows are valid over a narrow band of shifts
    # of the scale intercept. At c3c56d40a the bounded search probed only invalid shifts and
    # kept the held intercept; an independent SciPy profile finds +4.55323, lowering the
    # negative log-likelihood from 1035.04 to 808.92.
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x, "off": -35.0 + 30.0 * x})
    y = 1.0 + 0.005 * np.random.default_rng(0).normal(size=200)
    family = GeneralizedGammaLSS()
    rule = Constraint.postfit.increasing
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6)),
        family.scale(s("x", kind="bs", k=6, constraint=rule), offset="off"),
        family.shape(s("x", kind="bs", k=6)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    fitted = model._model
    layout = fitted.layout
    beta = np.array(fitted.coefficients, dtype=np.float64)
    for parameter, intercept in (("mean", np.log(1.03)), ("scale", 0.0), ("shape", -1.5)):
        beta[layout.predictor(parameter).intercept_index] = intercept
        beta[layout.term_slices[f"{parameter}:x"]] = 0.0
    design = _term_of(fitted, "scale").transform(x)
    beta[layout.term_slices["scale:x"]] = np.linalg.lstsq(design, 0.2 * np.sin(6 * x), rcond=None)[
        0
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        revised = model._with_revised_coefficients(
            beta, edited_terms=["mean:x", "scale:x", "shape:x"]
        )
    assert revised.predict_parameters(X)["scale"].max() < 0.017

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        revised.apply_shape_postfit()
    (record,) = revised._model.fit_state.hand_edit.repairs
    assert record.intercept_shift == pytest.approx(4.55323, abs=1e-4)
    assert revised.predict_parameters(X)["scale"].max() == pytest.approx(0.64969, abs=1e-4)


def test_a_session_follows_a_refit_in_place_that_changed_a_categorical_reference():
    # Sol's case: frequency weights make A the reference level of the first fit and B of the
    # refit. At 19a6ab028 the session kept the first fit's compiled specs, so an edit to C was
    # written in A's encoding and the untouched A and B predicted each other's means.
    from superglm.editor.lss.availability import session_refit_reason
    from superglm.editor.payloads import session_payload

    g = np.repeat(["A", "B", "C"], [200, 120, 80])
    X = pd.DataFrame({"g": g})
    mean = pd.Series({"A": 1.0, "B": 2.0, "C": 1.5})[g].to_numpy()
    y = np.random.default_rng(0).gamma(4.0, mean / 4.0)
    first, again = np.ones(len(g)), np.where(g == "B", 10.0, 1.0)
    family = GammaLS()
    model = SuperLSS(family, family.mean(cat("g")), family.scale(), weight_semantics="frequency")
    levels = pd.DataFrame({"g": ["A", "B", "C"]})

    def shifted_c(session):
        session.select_levels("mean:g", ["C"])
        session.shift("mean:g", 0.1)
        return session.to_model().predict(levels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=first)
        session = edit(model, train_data=(X, y, first))
        assert session_refit_reason(session) is None
        model.fit(X, y, sample_weight=again)
        # the reason kept for the first fit goes with it: the session takes the refit's weights
        session_payload(session)
        assert session_refit_reason(session) is None
        np.testing.assert_array_equal(session._evaluation_data["train"].sample_weight, again)
        followed = shifted_c(session)
        fresh = shifted_c(edit(model, train_data=(X, y, again)))
        refitted = model.predict(levels)
    np.testing.assert_allclose(followed, fresh, rtol=1e-12)
    np.testing.assert_allclose(followed[:2], refitted[:2], rtol=1e-12)
    np.testing.assert_allclose(followed[2], refitted[2] * np.exp(0.1), rtol=1e-12)


def test_a_session_follows_a_refit_in_place_that_kept_other_rows():
    # The term grids and reference values are read from the rows a fit kept. At 19a6ab028 the
    # session kept the first fit's 400 rows after a refit that kept 220, so its grid differed
    # from a fresh session's and its curve was 0.028 off.
    from superglm.editor.payloads import session_payload

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 10.0, 400)
    X = pd.DataFrame({"x": x})
    y = rng.gamma(4.0, np.exp(0.1 * x) / 4.0)
    kept = np.where(x < 5.0, 0.0, 1.0)
    kept[:5] = 1.0
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", k=6)), family.scale(), weight_semantics="frequency")
    lambdas = {"mean:x#wiggle": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=np.ones(400), lambdas=lambdas)
        session = edit(model, train_data=(X, y, np.ones(400)))
        model.fit(X, y, sample_weight=kept, lambdas=lambdas)
        session_payload(session)
        fresh = edit(model, train_data=(X, y, kept))
    assert len(session.model.X_train) == len(fresh.model.X_train) == 220
    followed, wanted = session.terms["mean:x"], fresh.terms["mean:x"]
    np.testing.assert_array_equal(followed.x, wanted.x)
    np.testing.assert_allclose(followed.original_log_effect, wanted.original_log_effect, atol=1e-12)


def _gamma_on(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"x": x}), rng.gamma(4.0, np.exp(0.3 * x) / 4.0)


@pytest.mark.parametrize("n_refit", [150, 60])
def test_a_session_takes_the_rows_of_a_refit_in_place_on_other_data(n_refit):
    # Sol's case. At 9f0916e6e the follow applied the refit's kept positions to the session's
    # 100-row frame: IndexError on 150 rows, and on 60 rows the first 60 of the old rows.
    from superglm.editor.cv import final_fit_reason
    from superglm.editor.lss.availability import TRAINING_ROWS_FOLLOWED, session_refit_reason
    from superglm.editor.payloads import session_payload
    from superglm.editor.session import EditorFitReplacedWarning
    from superglm.editor.widget import EditorWidget

    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", kind="bs", k=6)), family.scale())
    lambdas = {"mean:x#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    X2, y2 = _gamma_on(n_refit, 1)
    weights = np.linspace(1.0, 2.0, n_refit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas=lambdas)
        session = edit(model, train_data=(X, y))
        widget = EditorWidget(session, mode="server")
        model.fit(X2, y2, sample_weight=weights, lambdas=lambdas)
    with pytest.warns(EditorFitReplacedWarning, match="refitted in place on other data"):
        session_payload(session)
    assert TRAINING_ROWS_FOLLOWED.format(n=n_refit) in widget._state()["notice"]
    train = session._evaluation_data["train"]
    np.testing.assert_array_equal(np.asarray(train.y), y2)
    np.testing.assert_array_equal(train.sample_weight, weights)
    assert session_refit_reason(session) is None and final_fit_reason(session) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = edit(model, train_data=(X2, y2, weights))
        for opened in (session, fresh):
            opened.select_x("mean:x", 0.2, 0.4)
            opened.shift("mean:x", 0.1)
        followed, wanted = session.terms["mean:x"], fresh.terms["mean:x"]
        np.testing.assert_array_equal(followed.x, wanted.x)
        np.testing.assert_allclose(followed.edited_log_effect, wanted.edited_log_effect, atol=1e-12)
        np.testing.assert_allclose(
            session.to_model().predict(X2), fresh.to_model().predict(X2), rtol=1e-12
        )
        assert widget._state()["notice"] is None  # said once


def _follows_the_refit(session) -> None:
    from superglm.editor.payloads import session_payload
    from superglm.editor.session import EditorFitReplacedWarning

    with pytest.warns(EditorFitReplacedWarning, match="refitted in place on other data"):
        session_payload(session)


def test_a_session_takes_a_refit_in_place_on_the_same_rows_with_other_weights():
    # Codex's case. At 02a6773bc the rows matched on count and responses alone, so the session
    # kept its unweighted split, and its exposure, scores, Refit and Final fit had no weights.
    from superglm.editor.lss.availability import session_refit_reason

    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", kind="bs", k=6)), family.scale())
    lambdas = {"mean:x#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    weights = np.arange(1.0, 101.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas=lambdas)
        session = edit(model, train_data=(X, y))
        model.fit(X, y, sample_weight=weights, lambdas=lambdas)
    _follows_the_refit(session)
    np.testing.assert_array_equal(session._evaluation_data["train"].sample_weight, weights)
    assert session_refit_reason(session) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = edit(model, train_data=(X, y, weights))
    np.testing.assert_array_equal(session.terms["mean:x"].weights, fresh.terms["mean:x"].weights)


def test_a_session_takes_a_refit_in_place_on_the_same_rows_with_another_offset():
    from superglm.editor.payloads import session_payload

    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", kind="bs", k=6), offset="exposure"), family.scale())
    lambdas = {"mean:x#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    X = X.assign(exposure=np.zeros(100))
    moved = X.assign(exposure=np.linspace(-0.2, 0.2, 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas=lambdas)
        session = edit(model, train_data=(X, y))
        # unit weights resolve as no weights do: a refit on the same inputs is followed quietly
        model.fit(X, y, sample_weight=np.ones(100), lambdas=lambdas)
        warnings.simplefilter("error", UserWarning)
        session_payload(session)
        warnings.simplefilter("ignore")
        model.fit(moved, y, lambdas=lambdas)
    _follows_the_refit(session)
    train = session._evaluation_data["train"]
    np.testing.assert_array_equal(
        as_eager_frame(train.X).column_array("exposure"), moved["exposure"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = edit(model, train_data=(moved, y))
        for opened in (session, fresh):
            opened.select_x("mean:x", 0.2, 0.4)
            opened.shift("mean:x", 0.1)
        np.testing.assert_allclose(
            session.to_model().predict(moved), fresh.to_model().predict(moved), rtol=1e-12
        )


def test_a_refit_in_place_rebuilds_what_a_ruled_edit_reads():
    # #621 reads a ruled term's rule rows, the reference row and its reference design lazily
    # and keeps them on the view; each is read from the fit, so a refit in place drops them.
    from superglm.editor.lss.rules import term_rows
    from superglm.editor.payloads import session_payload

    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.fit.increasing)),
        family.scale(),
    )
    lambdas = {"mean:x#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    X2, y2 = X.assign(x=np.linspace(1.0, 3.0, 100)), _gamma_on(100, 1)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas=lambdas)
        session = edit(model, train_data=(X, y))
        session._prepare_edit("mean:x")
        before = term_rows(session.model, "mean:x"), session.model.reference_row()
        model.fit(X2, y2, lambdas=lambdas)
        session_payload(session)
        fresh = edit(model, train_data=(X2, y2))
        fresh._prepare_edit("mean:x")
    followed, wanted = term_rows(session.model, "mean:x"), term_rows(fresh.model, "mean:x")
    assert followed is not before[0]
    np.testing.assert_array_equal(followed.A, wanted.A)
    np.testing.assert_array_equal(followed.c, wanted.c)
    pd.testing.assert_frame_equal(session.model.reference_row(), fresh.model.reference_row())
    assert not before[1].equals(session.model.reference_row())
    np.testing.assert_array_equal(
        session.model.reference_design("mean:x"), fresh.model.reference_design("mean:x")
    )


def test_a_session_takes_a_refit_in_place_on_other_feature_values():
    # Codex's case: the same rows, responses, weights and offsets, with x replaced by x**3. At
    # 3f528d859 the rows still matched, so the session kept its old features and exposure.
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", kind="bs", k=6)), family.scale())
    lambdas = {"mean:x#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    cubed = X.assign(x=X["x"] ** 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas=lambdas)
        session = edit(model, train_data=(X, y))
        model.fit(cubed, y, lambdas=lambdas)
    _follows_the_refit(session)
    train = session._evaluation_data["train"]
    np.testing.assert_array_equal(as_eager_frame(train.X).column_array("x"), cubed["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = edit(model, train_data=(cubed, y))
    followed, wanted = session.terms["mean:x"], fresh.terms["mean:x"]
    np.testing.assert_array_equal(followed.x, wanted.x)
    np.testing.assert_array_equal(followed.weights, wanted.weights)


@pytest.mark.parametrize("change", ["repair", "same data"])
def test_a_follow_onto_the_same_data_takes_the_fits_rows_quietly(change):
    # The session's split becomes the fit's kept rows after every refit or repair in place;
    # where they hold the data it had, nothing is said and nothing the page shows moves.
    from superglm.editor.payloads import session_payload
    from superglm.editor.session import EditorFitReplacedWarning

    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(
            s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing),
            s("z", kind="bs", k=5),
        ),
        family.scale(),
    )
    lambdas = {"mean:x#wiggle": 1.0, "mean:z#wiggle": 1.0}
    X, y = _gamma_on(100, 0)
    X = X.assign(z=np.random.default_rng(5).uniform(size=100))
    weights = np.linspace(1.0, 2.0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=weights, lambdas=lambdas)
        session = edit(model, train_data=(X, y, weights))
        before = session_payload(session)["mean:z"]
        if change == "repair":
            model.apply_shape_postfit()
        else:
            model.fit(X, y, sample_weight=weights, lambdas=lambdas)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        after = session_payload(session)["mean:z"]
    assert not [w for w in seen if issubclass(w.category, EditorFitReplacedWarning)]
    train = session._evaluation_data["train"]
    assert train.X is model._training_frame  # the fit's own rows, now
    np.testing.assert_array_equal(train.sample_weight, weights)
    np.testing.assert_array_equal(np.asarray(train.y), y)
    assert after == before


_OFFSETS = ("none", "column", "array", "column and array", "several parameters")
_WEIGHTS = ("none", "prior", "frequency", "exposure")


def _follow_case(offsets: str, weights: str, seed: int):
    """A 100-row SuperLSS whose mean term has a dip a post-fit increasing rule repairs, its
    offsets declared as ``offsets`` names, and its data for ``seed``: ``(model, X, y, w, given)``,
    ``given`` being the offsets a ``train_data`` supplies (its array parts)."""
    from superglm import NegativeBinomialLS

    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 10.0, 100))
    X = pd.DataFrame({"x": x, "exposure": rng.uniform(-0.3, 0.3, 100)})
    array = np.linspace(-0.2, 0.2, 100)  # declared on the model, so the same for every seed
    mean_sources = {
        "none": (),
        "column": ("exposure",),
        "array": (array,),
        "column and array": ("exposure", array),
        "several parameters": ("exposure",),
    }[offsets]
    other_sources = (0.5 * array,) if offsets == "several parameters" else ()
    family = NegativeBinomialLS() if weights == "exposure" else GammaLS()
    other = family.theta if weights == "exposure" else family.scale
    rule = Constraint.postfit.increasing
    term = s("x", kind="bs", k=8, constraint=rule)
    model = SuperLSS(
        family,
        family.mean(term, **({"offset": mean_sources} if mean_sources else {})),
        other(**({"offset": other_sources} if other_sources else {})),
        weight_semantics="frequency" if weights == "frequency" else "prior",
    )
    log_mean = 0.4 + 0.18 * x - 0.8 * np.exp(-(((x - 6.0) / 0.9) ** 2))
    log_mean = log_mean + sum(
        X[s_].to_numpy() if isinstance(s_, str) else s_ for s_ in mean_sources
    )
    w = {
        "none": None,
        "prior": rng.uniform(0.5, 2.0, 100),
        "frequency": rng.integers(1, 4, 100).astype(float),
        "exposure": rng.uniform(0.5, 2.0, 100),
    }[weights]
    if weights == "exposure":  # a rate: counts over the exposure
        y = rng.negative_binomial(5.0, 5.0 / (5.0 + 4.0 * w * np.exp(log_mean))) / w
    else:
        y = rng.gamma(6.0, np.exp(log_mean) / 6.0)
    given = {
        name: sum(source for source in sources if not isinstance(source, str))
        for name, sources in (("mean", mean_sources), (other.__name__, other_sources))
        if any(not isinstance(source, str) for source in sources)
    }
    return model, X, y, w, given or None


def _shown(session, X):
    """Everything a session shows of its training rows, and of one edit and its export."""
    from superglm.editor.lss.validation import lss_dataset_metrics
    from superglm.editor.payloads import session_payload

    session_payload(session)  # an action: the session follows its model, as the page's do
    train = session._evaluation_data["train"]
    frame = as_eager_frame(train.X)
    lss = session.model.lss
    shown = {
        "x": frame.column_array("x"),
        "exposure": frame.column_array("exposure"),
        "y": np.asarray(train.y, dtype=np.float64),
        "w": np.ones(train.n_obs) if train.sample_weight is None else train.sample_weight,
        "link": lss.predict_link(train.X, offsets=train.offset).to_numpy(),
        "metrics": np.array(list(lss_dataset_metrics(session.model, train).values())),
        "exposure weights": session.terms["mean:x"].weights,
    }
    session.select_x("mean:x", 2.0, 5.0)
    session.shift("mean:x", 0.1)
    shown["payload"] = session_payload(session)
    shown["edited"] = session.to_model().predict_link(train.X, offsets=train.offset).to_numpy()
    return shown


@pytest.mark.parametrize("follow", ["refit on the same data", "refit on other data", "repair"])
@pytest.mark.parametrize("weights", _WEIGHTS)
@pytest.mark.parametrize("offsets", _OFFSETS)
def test_a_followed_session_shows_what_a_fresh_one_on_the_refit_shows(offsets, weights, follow):
    # The invariant that closes the follow's family of findings: after any follow, the
    # session's training rows and every number it shows equal those of a session opened fresh
    # on the refitted model with the refit's own data. Codex's case is a column and an array
    # offset: at b28e71770 the follow stored the total offset as the split's array part, and
    # shape_prediction_offsets added the column a second time.
    model, X, y, w, given = _follow_case(offsets, weights, 0)
    lambdas = {"mean:x#wiggle": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=w, lambdas=lambdas)
        session = edit(model, train_data=(X, y, w, given))
        opened = model._model
        if follow == "repair":
            model.apply_shape_postfit()
            assert model._model.fit_state.hand_edit.repairs  # a repair was made
            data = (X, y, w, given)
        else:
            _, X2, y2, w2, _ = _follow_case(offsets, weights, 0 if "same" in follow else 1)
            model.fit(X2, y2, sample_weight=w2, lambdas=lambdas)
            data = (X2, y2, w2, given)
        assert model._model is not opened
        followed = _shown(session, data[0])
        fresh = _shown(edit(model, train_data=data), data[0])
    assert followed.keys() == fresh.keys()
    for name in followed:
        if name == "payload":
            assert followed[name] == fresh[name], name
        else:
            np.testing.assert_allclose(followed[name], fresh[name], rtol=1e-12, err_msg=name)


def test_a_refit_in_place_that_kept_no_rows_says_the_editor_needs_its_rows():
    from superglm.editor.errors import EditorValueError
    from superglm.editor.lss.availability import TRAINING_ROWS_LOST
    from superglm.editor.payloads import session_payload

    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", kind="bs", k=6)), family.scale())
    X, y = _gamma_on(100, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lambdas={"mean:x#wiggle": 1.0})
        session = edit(model, train_data=(X, y))
        model.fit(*_gamma_on(150, 1), lambdas={"mean:x#wiggle": 1.0}, retain_rows=False)
        with pytest.raises(EditorValueError) as refused:
            session_payload(session)
    assert str(refused.value) == TRAINING_ROWS_LOST


def test_the_diagnose_rows_follow_an_in_place_repair_made_after_the_report_was_opened():
    # The rows are kept per fit. At f6346704d they were kept per model view, which a repair
    # keeps, so row 303 still read predicted 8.511 and cost 8.674 against 9.381 and 7.590.
    from superglm.editor.cv import diagnose_report_payload
    from superglm.editor.widget import EditorWidget

    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widget = EditorWidget(edit(model, train_data=(X, y)), mode="server")
        before = diagnose_report_payload(widget)["fits"]
        model.apply_shape_postfit()
        after = diagnose_report_payload(widget)["fits"]
        fresh = EditorWidget(edit(model, train_data=(X, y)), mode="server")
        wanted = diagnose_report_payload(fresh)["fits"]
    assert after != before
    assert after == wanted


def test_supplied_cv_results_say_they_were_made_on_the_fit_before_an_in_place_repair():
    # A cv= result describes the fit the session was opened on. At f6346704d the session
    # compared models only, which a repair keeps, so after it the supplied scores and fold
    # curves read stale=False and Diagnose did not say the folds were of the earlier model.
    from sklearn.model_selection import KFold

    from superglm import cross_validate
    from superglm.editor.cv import capture_cv_view, cv_tab_payload, diagnose_report_payload
    from superglm.editor.payloads import session_payload
    from superglm.editor.widget import EditorWidget

    model = _fit("mean", Constraint.postfit.increasing, kind="bs", k=8)
    X, y = _data("mean")
    earlier = "fitted on the model as first opened"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplied = cross_validate(
            model,
            X,
            y,
            cv=KFold(2, shuffle=True, random_state=0),
            fit_mode="fit_reml",
            return_estimators=True,
            error_score="raise",
        )
        session = edit(model, train_data=(X, y), cv=supplied)
        widget = EditorWidget(session, mode="server")

        def stale() -> tuple[list[bool], bool, bool]:
            report = cv_tab_payload(capture_cv_view(session, run=None, final_fit=None), jobs={})
            note = diagnose_report_payload(widget)["note"]
            return (
                [r["stale"] for r in report["results"]],
                report["relativities"]["stale"],
                (earlier in note),
            )

        assert stale() == ([False], False, False)
        model.apply_shape_postfit()
        session_payload(session)
        assert not session.edited_terms()
        assert stale() == ([True], True, True)
