"""A coefficient revision publishes one predictor in every coordinate (#447).

An editor edit and a post-fit shape repair each write new coefficients and
move the public intercept.  Three representations then have to read the same
function of the rows: the public predictor ``predict`` scores, its centred
pair ``(alpha, c)`` (which keeps a numeric column far from zero exact), and
the solver predictor that a later shape repair profiles its intercept from.
Before #447 the editor left the solver predictor at the pre-edit coefficients'
shift ``m' beta``, so a repair after an edit published the difference, and an
intercept change reached only the raw intercepts, where it rounded away beside
a numeric column at an offset of 1e16.

References are exact (``Fraction``) or the repair's own definition: the
projection of the edited coefficients onto the shape cone, then the intercept
that minimizes the deviance at them.  Bounds derive from the dimensions,
``eps`` and the repair's stopping rules (Higham 2002, section 3.1, for the
``gamma_k`` of each evaluation), never from what passed locally.
"""

from __future__ import annotations

import copy
import math
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, MonotoneRepairer, Numeric, PSpline, SuperGLM
from superglm.editor import EditorSession
from superglm.model import shape_ops

EPS = float(np.finfo(np.float64).eps)
_U = EPS / 2.0


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _shaped_fit(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> SuperGLM:
    """The issue's Gaussian fit: one increasing-by-repair P-spline, frequency weights."""
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.8,
        weight_semantics="frequency",
        features={
            "x": PSpline(
                n_knots=6, knot_strategy="uniform", constraint=Constraint.postfit.increasing
            )
        },
    ).fit(pd.DataFrame({"x": x}), y, sample_weight=w)


def _edit(model: SuperGLM, frame: pd.DataFrame, y, w, name: str, scale: float, shift: float):
    session = EditorSession.from_model(model, terms=[name], train_data=(frame, y, w))
    term = session.terms[name]
    original = np.asarray(term.edited_log_effect, dtype=np.float64)
    term.edited_log_effect = scale * original + shift
    return session.to_model(), term, original


def _weighted_mean(values, w) -> Fraction:
    weights = [Fraction(float(v)) for v in w]
    total = sum(
        (wi * Fraction(float(v)) for wi, v in zip(weights, values, strict=True)), Fraction(0)
    )
    return total / sum(weights, Fraction(0))


def _evaluation_bound(edited, columns, beta_edit, beta_repaired) -> float:
    """What two float evaluations of the edited model's predictors can differ by.

    The solver predictor is ``alpha_s + (alpha_lo + X beta)`` over the solver
    columns ``X = X_pub + 1 m'``, the public one ``alpha + (alpha_lo + X_pub
    beta)``; a repair moves the first by the term's change ``(X - 1 m')
    dbeta``.  Each row is within ``gamma_(p+4)`` of its magnitudes (Higham
    2002, section 3.1); the TwoSum that carries an intercept change is exact.
    """
    solver, public = edited._solver_pirls_result(), edited.result
    term = next(iter(edited._runtime_canonical_state["terms"].values()))
    means = np.asarray(term["groups"][0]["column_means"], dtype=np.float64)
    spread = (np.abs(columns) + np.abs(means)) @ (2.0 * np.abs(beta_edit) + np.abs(beta_repaired))
    magnitude = (
        abs(float(solver.centred_intercept or solver.intercept))
        + abs(float(solver.centred_intercept_lo or 0.0))
        + abs(float(public.centred_intercept or public.intercept))
        + abs(float(public.centred_intercept_lo or 0.0))
        + float(np.max(spread))
    )
    return 2.0 * _gamma(beta_edit.size + 4) * magnitude


def _gaussian_repair_bound(edited, columns, beta_edit, beta_repaired, y, w, reference) -> float:
    """What the repair's float evaluation can miss the exact profiled predictor by.

    The repair profiles its intercept from the solver predictor moved by the
    term's change (``_evaluation_bound``).  The weighted residual sum adds
    ``gamma_(n+1) sum |w r|``, and the profile skips a sum below ``64 eps
    max(1, sum |w r|)`` (``shape_ops._profile_repaired_intercept``).
    """
    residual = max(1.0, math.fsum(np.abs(w * (y - reference))))
    return _evaluation_bound(edited, columns, beta_edit, beta_repaired) + (
        _gamma(len(y) + 1) + 64.0 * EPS
    ) * residual / math.fsum(w)


def _projected_reference(edited, x, y, w):
    """The repair's definition applied outside any revision: projection, then exact profile.

    ``apply_shape_postfit`` projects the term's coefficients onto the shape
    cone (``MonotoneRepairer`` on the grid weights) and profiles the intercept
    at them; for a Gaussian identity fit that is the weighted mean residual,
    ``a = sum w (y - X_pub beta_r) / sum w``, formed here in exact arithmetic.
    Returns the projected coefficients, the public columns and the reference.
    """
    spec, groups = edited._specs["x"], list(edited._groups)
    beta_edit = np.asarray(edited.result.beta, dtype=np.float64)
    projected = MonotoneRepairer(direction="increasing").repair(
        spec, beta_edit, groups, weights=shape_ops._grid_weights(spec, x, w, 80), n_grid=80
    )
    beta_reference = np.asarray(projected.repaired_beta_reparam, dtype=np.float64)
    columns = np.asarray(spec.transform(x), dtype=np.float64)
    curve = [
        sum(
            (Fraction(float(c)) * Fraction(float(b)) for c, b in zip(row, beta_reference)),
            Fraction(0),
        )
        for row in columns
    ]
    intercept = _weighted_mean([Fraction(float(v)) - c for v, c in zip(y, curve)], w)
    return beta_reference, columns, np.array([float(intercept + c) for c in curve])


@pytest.mark.parametrize("reload", [False, True], ids=["fresh", "pickled"])
def test_a_repair_after_an_edit_predicts_the_weighted_mean(reload):
    """The issue's fixture: the repair flattens the edited curve to the weighted mean.

    Halving a decreasing curve keeps it decreasing, so the increasing repair
    flattens it and the profiled Gaussian intercept is the weighted mean of
    ``y``.  It predicted 0.94969657 against 0.94614946 on 0.35.0 and 31544462,
    off by the shift ``m' (beta_new - beta_old)`` the editor left in the solver
    predictor; a pickle round trip of the edited model kept it.  Metrics read
    the same solver predictor on the fit rows (``metrics(...).eta``) and were
    off by as much.  The reference is the exact profile at the projected
    coefficients (``_projected_reference``), the weighted mean to the size of
    whatever curve the optimizer leaves, so no exact zero is asserted.
    Mutation: drop ``_read_solver_state_from_public`` from
    ``publish_revised_coefficients``.
    """
    x = np.linspace(0.0, 1.0, 60)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), x.size)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _shaped_fit(y, x, w)
        edited, _, _ = _edit(model, frame, y, w, "x", 0.5, 0.0)
    if reload:
        edited = pickle.loads(pickle.dumps(edited))
    beta_edit = np.asarray(edited.result.beta, dtype=np.float64).copy()
    columns = np.asarray(edited._specs["x"].transform(x), dtype=np.float64)
    eta = edited.metrics(frame, y, sample_weight=w).eta
    published = edited._predict_eta_raw_exact(frame)
    assert np.all(
        np.abs(eta - published) <= _evaluation_bound(edited, columns, beta_edit, beta_edit)
    )
    beta_reference, _, reference = _projected_reference(edited, x, y, w)
    bound = _gaussian_repair_bound(edited, columns, beta_edit, beta_reference, y, w, reference)
    edited.apply_shape_postfit(frame, n_grid=80)

    np.testing.assert_array_equal(edited.result.beta, beta_reference)
    error = np.abs(edited.predict(frame) - reference)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}, bound {bound:.3g}"


@pytest.mark.parametrize("shift", [0.0, 0.1], ids=["no_intercept_change", "intercept_change"])
def test_a_repair_after_an_edit_projects_the_edited_curve(shift):
    """A repair that keeps a curve: the projection of the edited coefficients, then the profile.

    ``apply_shape_postfit`` projects the term's coefficients onto the shape
    cone (``MonotoneRepairer`` on the grid weights) and profiles the intercept
    at them; for a Gaussian identity fit that is the weighted mean residual.
    The reference is that definition applied to the edited model outside any
    revision: the same projection, then ``a = sum w (y - X_pub beta_r) / sum
    w`` in exact arithmetic.  31544462 missed it by 1.5e-3 with or without an
    intercept change in the edit.
    """
    x = np.linspace(0.0, 1.0, 60)
    y = 1.5 + 0.4 * x + 0.08 * np.sin(9.0 * x)
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), x.size)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _shaped_fit(y, x, w)
        edited, _, _ = _edit(model, frame, y, w, "x", 0.5, shift)
    beta_edit = np.asarray(edited.result.beta, dtype=np.float64).copy()
    beta_reference, columns, reference = _projected_reference(edited, x, y, w)
    assert np.any(beta_reference) and not np.array_equal(beta_reference, beta_edit)
    bound = _gaussian_repair_bound(edited, columns, beta_edit, beta_reference, y, w, reference)
    edited.apply_shape_postfit(frame, n_grid=80)

    np.testing.assert_array_equal(edited.result.beta, beta_reference)
    error = np.abs(edited.predict(frame) - reference)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}, bound {bound:.3g}"


def _offset_frame(offset: float):
    """An even-integer numeric column (exact at 1e16) beside a spline and a four-level factor."""
    rng = np.random.default_rng(447)
    n = 120
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(0.0, 1.0, n)
    g = np.resize(np.array(["a", "b", "c", "d"], dtype=object), n)
    level = np.resize(np.array([0.0, 0.2, 0.3, -0.4]), n)
    eta = 0.2 * z + 0.8 * s + 0.1 * np.sin(9.0 * s) + level
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), n)
    return pd.DataFrame({"x": offset + z, "s": s, "g": g}), eta, w, rng


def _public_columns(model, frame) -> np.ndarray:
    """The public design ``X_pub``, each term's columns as its published spec scores them."""
    columns = np.zeros((len(frame), model.result.beta.size))
    for group in model._groups:
        block = model._specs[group.feature_name].transform(frame[group.feature_name].to_numpy())
        block = block.toarray() if hasattr(block, "toarray") else block
        columns[:, group.sl] = np.asarray(block, dtype=np.float64).reshape(len(frame), -1)
    return columns


def _centred_magnitude(model, frame) -> np.ndarray:
    """Each row's ``|alpha| + |alpha_lo| + sum |X_pub - c_pub| |beta|``."""
    result = model.result
    centre = np.asarray(result.state_center, dtype=np.float64)
    spread = np.abs(_public_columns(model, frame) - centre[None, :]) @ np.abs(result.beta)
    return spread + abs(float(result.centred_intercept)) + abs(result.centred_intercept_lo or 0.0)


@pytest.mark.parametrize(
    ("retain", "reload"), [(True, False), (False, True)], ids=["retained", "released_pickled"]
)
def test_an_intercept_change_beside_a_far_offset_numeric_reaches_the_prediction(retain, reload):
    """Part 2 of #447: the edit's intercept change goes into the centred pair.

    With ``base="first"`` the edit ``0.5 effect + 0.1`` moves the intercept by
    0.1 and each level's coefficient by the rest, so every row's prediction
    moves by exactly ``edited[level] - original[level]``.  Beside ``x = 1e16
    + z`` both raw intercepts are about -2e15, whose ulp is 0.25: 31544462
    added the 0.1 there and cleared the centred pair, and predicted none of it.
    Each predictor evaluates to ``gamma_(p+3)`` of its centred magnitudes;
    the change is exact.  Mutation: drop the carry in ``move_public_intercept``.
    """
    frame, eta, w, _ = _offset_frame(1e16)
    y = 3.0 + eta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric(), "g": Categorical(base="first")},
            retain_fit_state=retain,
        ).fit(frame.drop(columns="s"), y, sample_weight=w)
        model = pickle.loads(pickle.dumps(fitted)) if reload else fitted
        edited, term, original = _edit(model, frame, y, w, "g", 0.5, 0.1)
    row_level = {str(level): index for index, level in enumerate(term.levels)}
    index = np.array([row_level[str(level)] for level in frame["g"]])
    change = (0.5 * original + 0.1) - original
    before = model.predict(frame)
    expected = np.array(
        [float(Fraction(float(b)) + Fraction(float(c))) for b, c in zip(before, change[index])]
    )
    p = model.result.beta.size
    bound = (
        2.0 * _gamma(p + 3) * (_centred_magnitude(model, frame) + _centred_magnitude(edited, frame))
    )
    bound += _U * np.abs(expected)
    error = np.abs(edited.predict(frame) - expected)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}"


def test_a_repair_beside_a_far_offset_numeric_keeps_its_profiled_intercept():
    """A repair's own intercept change reaches the centred pair at an offset of 1e16 too.

    The repair profiles the Poisson intercept at the projected coefficients
    and added the shift to the raw public intercept (about -1e15 here, ulp
    0.125), where it rounded away; the solver relation then still held bit
    for bit and the pair was re-read without it: 31544462 missed the profiled
    predictor by 5.6e-4.  Reference: the projected coefficients the repair
    published, and the exact Poisson profile ``a = log sum w y - log sum w
    exp(X_pub beta_r)``.  Bound: the profile's acceptance bar ``|score| <=
    1e-9 (1 + sum |w (y - mu)|)`` over the information ``sum w mu``, doubled
    for its curvature, and each evaluation's ``gamma_(n+p+4)``.
    """
    frame, eta, w, rng = _offset_frame(1e16)
    y = rng.poisson(np.exp(0.3 + 0.5 * eta)).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.8,
            features={
                "x": Numeric(),
                "s": PSpline(
                    n_knots=6, knot_strategy="uniform", constraint=Constraint.postfit.increasing
                ),
            },
            weight_semantics="frequency",
        ).fit(frame.drop(columns="g"), y, sample_weight=w)
        before = np.asarray(model.result.beta, dtype=np.float64).copy()
        model.apply_shape_postfit(frame, sample_weight=w, n_grid=80)
    beta = np.asarray(model.result.beta, dtype=np.float64)
    assert not np.array_equal(beta, before)

    exact = [
        sum((Fraction(float(c)) * Fraction(float(b)) for c, b in zip(row, beta)), Fraction(0))
        for row in _public_columns(model, frame)
    ]
    relative = np.array([float(value - exact[0]) for value in exact])
    level = math.log(math.fsum(w * y)) - math.log(math.fsum(w * np.exp(relative)))
    reference = level + relative
    mu = np.exp(reference)
    profile = 2e-9 * (1.0 + math.fsum(np.abs(w * (y - mu)))) / math.fsum(w * mu)
    bound = profile + 4.0 * _gamma(len(y) + beta.size + 4) * float(
        np.max(_centred_magnitude(model, frame) + np.abs(reference))
    )
    error = np.abs(model._predict_eta_raw_exact(frame) - reference)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}, bound {bound:.3g}"


def test_a_revision_starts_from_the_published_predictor_of_an_older_edit():
    """A model an editor revised before #447 is repaired from its published predictor.

    Such a model kept the solver intercept, centred pair and recorded shift of
    its pre-edit coefficients; a pickle carries them.  It is emulated here by
    writing the fit's values back over an edit's.  ``predict`` reads the
    published pair either way; a revision has to start from it, or the repair
    profiles from the old shift as before the fix.  For this edit the editor
    skips its -2.7e-17 intercept change, so these are the values 31544462
    wrote: a pickle of its edit matches them field for field.  Check: the
    first test's reference and bound.  Mutation: drop the reconcile in
    ``FittedStateRevision.start``.
    """
    x = np.linspace(0.0, 1.0, 60)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), x.size)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _shaped_fit(y, x, w)
        edited, _, _ = _edit(model, frame, y, w, "x", 0.5, 0.0)
    stale = pickle.loads(pickle.dumps(edited))
    fitted_solver = model._solver_pirls_result()
    for name in ("intercept", "centred_intercept", "centred_intercept_lo"):
        object.__setattr__(stale._solver_result, name, getattr(fitted_solver, name))
    stale._runtime_canonical_state = copy.deepcopy(model._runtime_canonical_state)
    np.testing.assert_array_equal(stale.predict(frame), edited.predict(frame))

    beta_edit = np.asarray(edited.result.beta, dtype=np.float64)
    beta_reference, columns, reference = _projected_reference(edited, x, y, w)
    stale.apply_shape_postfit(frame, n_grid=80)
    np.testing.assert_array_equal(stale.result.beta, beta_reference)
    bound = _gaussian_repair_bound(edited, columns, beta_edit, beta_reference, y, w, reference)
    error = np.abs(stale.predict(frame) - reference)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}, bound {bound:.3g}"


def _slope_edit_bound(model, edited, frame, expected) -> np.ndarray:
    """Each row's evaluation error in the coordinates a slope edit is defined in.

    The edit pivots at ``x = 0``, so the rows read ``intercept + x beta``:
    each evaluation (the fit's, the edit's) is within ``gamma_(p+3)`` of the
    raw magnitudes ``|eta| + |x| (|beta_old| + |beta_new|)``, plus the fit's
    centred constant ``|c beta_old|`` that the edit moves into the intercept
    (one product rounding), and the reference's own rounding ``u |eta|``.
    """
    x = frame["x"].to_numpy(dtype=np.float64)
    old, new = abs(float(model.result.beta[0])), abs(float(edited.result.beta[0]))
    centre = abs(float(np.asarray(model.result.state_center)[0]))
    magnitude = np.abs(model.predict(frame)) + np.abs(x) * (old + new) + centre * old
    return 2.0 * _gamma(model.result.beta.size + 3) * magnitude + _U * np.abs(expected)


def _slope_edited(model, frame, y, slope) -> SuperGLM:
    session = EditorSession.from_model(model, terms=["x"], train_data=(frame, y))
    session.terms["x"].edited_log_effect = np.array([slope], dtype=np.float64)
    return session.to_model()


def _raw_reference(model, frame, slope) -> np.ndarray:
    """``eta_before + x (slope - beta_old)``, exact before its one rounding."""
    change = Fraction(slope) - Fraction(float(model.result.beta[0]))
    return np.array(
        [
            float(Fraction(float(b)) + Fraction(float(v)) * change)
            for b, v in zip(model.predict(frame), frame["x"])
        ]
    )


@pytest.mark.parametrize("slope", [1.0, -1.0], ids=["up", "down"])
@pytest.mark.parametrize("scale", [1e16, -1e16])
def test_a_numeric_slope_edit_keeps_the_intercept_remainder(scale, slope):
    """A slope edit on a column spanning zero to 2e16 predicts its raw intercept at zero.

    Sol's review of #448 (P2): ``x = 1e16 {0, 1, 2}`` beside ``y = {2, 3, 2}``
    fits a slope of about 5e-35 about the centre 1e16 and the raw intercept
    2.3333333333333335.  An editor slope is defined about ``x = 0``, so the row
    at zero predicts that intercept.  26321fd6 carried ``c dbeta`` into the
    pair, which became ``(1e16 + 2, 1/3)``, and predicted 2.0 there: the
    remainder joined the row's ``-1e16`` before that cancelled the high part.
    The moved column is now scored about zero (``_recentre_revised_columns``).
    Reference: ``eta_before + x dbeta`` exactly; bound ``_slope_edit_bound``;
    through ``predict``, ``metrics`` on the fit rows and a pickle round trip.
    """
    x = scale * np.resize([0.0, 1.0, 2.0], 60)
    y = np.resize([2.0, 3.0, 2.0], 60)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            frame, y
        )
        edited = _slope_edited(model, frame, y, slope)
    expected = _raw_reference(model, frame, slope)
    bound = _slope_edit_bound(model, edited, frame, expected)
    for label, values in (
        ("predict", edited.predict(frame)),
        ("metrics", edited.metrics(frame, y).eta),
        ("pickled", pickle.loads(pickle.dumps(edited)).predict(frame)),
    ):
        error = np.abs(values - expected)
        assert np.all(error <= bound), f"{label}: max error {float(np.max(error)):.3g}"


@pytest.mark.parametrize("slopes", [(-1e308, 1e308), (1e308, -1e308)], ids=["up", "down"])
@pytest.mark.parametrize("scale", [1e-308, 1e-300])
def test_slope_edits_across_the_float_range_form_no_coefficient_difference(scale, slopes):
    """Successive slope edits of -1e308 and 1e308 complete and predict ``intercept + x beta``.

    Sol's review of #448 (P2): ``x = scale {-1, 0, 1}`` has a zero centre, so a
    slope edit moves no centred column.  26321fd6 formed ``beta - beta_before``,
    which overflowed to ``inf`` on the second edit, and ``0 * inf`` poisoned the
    pair: the revision refused with non-finite scalars where 31544462 predicted
    about ``2.33 +- 1`` (or ``+- 1e8``).  A zero-centre column is now skipped
    without a product.  Reference and bound as in the previous test.
    """
    x = scale * np.resize([-1.0, 0.0, 1.0], 60)
    y = np.resize([2.0, 3.0, 2.0], 60)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
            frame, y
        )
        first = _slope_edited(model, frame, y, slopes[0])
        second = _slope_edited(first, frame, y, slopes[1])
    for edited, slope in ((first, slopes[0]), (second, slopes[1])):
        expected = _raw_reference(model, frame, slope)
        error = np.abs(edited.predict(frame) - expected)
        assert np.all(error <= _slope_edit_bound(model, edited, frame, expected))
