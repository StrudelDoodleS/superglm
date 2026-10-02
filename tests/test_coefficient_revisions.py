"""A coefficient revision publishes one predictor in every coordinate (#447).

An editor edit and a post-fit shape repair each write new coefficients and
move the public intercept.  Three representations then have to read the same
function of the rows: the public predictor ``predict`` scores, its centred
pair ``(alpha, c)`` (which keeps a numeric column far from zero exact), and
the solver predictor that a later shape repair profiles its intercept from.
Before #447 the editor left the solver predictor at the pre-edit coefficients'
shift ``m' beta``, so a repair after an edit published the difference, and an
intercept change reached only the raw intercepts, where it rounded away beside
a numeric column at an offset of 1e16.  A numeric slope edit keeps its fitted
centre and carries its change into the pair exactly, and the pair is then
evaluated as one compensated sum (#449).

References are exact (``Fraction``) or the repair's own definition: the
projection of the edited coefficients onto the shape cone, then the intercept
that minimizes the deviance at them.  Bounds derive from the dimensions,
``eps`` and the repair's stopping rules (Higham 2002, section 3.1, for the
``gamma_k`` of each evaluation), never from what passed locally.
"""

from __future__ import annotations

import math
import pickle
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, MonotoneRepairer, Numeric, PSpline, SuperGLM
from superglm.editor import EditorSession
from superglm.model import shape_ops
from tests.test_saved_fs_models import assert_predicts_as_saved

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


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("source", ["saved_v0_35_0", "saved_31544462"])
def test_an_edit_saved_before_447_is_repaired_from_its_published_predictor(source):
    """An edited model saved by v0.35.0 or 31544462 repairs to the first test's reference.

    Their editor left the solver intercept and the recorded shift, and on
    31544462 the solver's centred pair, at the pre-edit coefficients
    (``scripts/make_saved_editor_fixtures.py`` wrote these pickles).  The model
    predicts as it was saved, to two evaluations' rounding on another BLAS
    kernel or SIMD target (``assert_predicts_as_saved``), and a revision starts
    from its published predictor (``FittedStateRevision.start``).  The repair then reaches
    the projection-and-profile reference within the first test's bound; the
    writer's own repair missed it by 3.55e-3.  Mutation: drop the re-read in
    ``FittedStateRevision.start``.
    """
    with open(FIXTURES / source / "editor_halved_spline.pkl", "rb") as handle:
        record = pickle.load(handle)
    model, frame = record["model"], record["frame"]
    y, w = record["y"], record["sample_weight"]
    assert_predicts_as_saved(model, frame, record["prediction"])

    x = frame["x"].to_numpy(dtype=np.float64)
    beta_edit = np.asarray(model.result.beta, dtype=np.float64).copy()
    beta_reference, columns, reference = _projected_reference(model, x, y, w)
    bound = _gaussian_repair_bound(model, columns, beta_edit, beta_reference, y, w, reference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.apply_shape_postfit(frame, n_grid=80)
    np.testing.assert_array_equal(model.result.beta, beta_reference)
    error = np.abs(model.predict(frame) - reference)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}, bound {bound:.3g}"


def _slope_edited(model, frame, y, slopes: dict[str, float]) -> SuperGLM:
    """The model with each named numeric column's slope edited, one session per column."""
    edited = model
    for name, slope in slopes.items():
        session = EditorSession.from_model(edited, terms=[name], train_data=(frame, y))
        session.terms[name].edited_log_effect = np.array([slope], dtype=np.float64)
        edited = session.to_model()
    return edited


def _raw_reference(model, frame, slopes: dict[str, float]) -> np.ndarray:
    """``eta_before + sum_j x_j (slope_j - beta_old_j)``, exact before its one rounding."""
    groups = {group.feature_name: group for group in model._groups}
    changes = {
        name: Fraction(slope) - Fraction(float(model.result.beta[groups[name].sl][0]))
        for name, slope in slopes.items()
    }
    rows = []
    for index, before in enumerate(model.predict(frame)):
        value = Fraction(float(before))
        for name, change in changes.items():
            value += Fraction(float(frame[name].iloc[index])) * change
        rows.append(float(value))
    return np.array(rows)


def _slope_edit_bound(model, edited, frame, expected) -> np.ndarray:
    """Each row's error budget for numeric slope edits of an all-numeric model.

    The reference starts from the fit's own prediction, which is within
    ``gamma_(p+2)`` of the fit's centred magnitudes ``|alpha| + |alpha_lo| +
    sum |x - c| |beta_old|`` (Higham 2002, section 3.1).  An edit that moves
    a centred column is one compensated sum of the pair and each such column's
    exact pieces about its fitted centre.  That is within ``u |eta|`` and
    ``gamma_k^2`` of the addends' magnitudes ``|alpha'| + |alpha_lo'| + sum |x
    - c| |beta|``, ``k = 3p + 2`` (Ogita, Rump & Oishi 2005, Proposition 4.5),
    plus the pieces' own ``u^2``.  A column without a centre is one product
    added as fitted, ``gamma_2 |x beta|``.  Nothing a moved centre or an
    uncompensated pair would add to the magnitudes is allowed.  Each term is
    doubled for the magnitudes' own rounding.
    """
    p = model.result.beta.size
    centre = np.asarray(model.result.state_center, dtype=np.float64)
    columns = _public_columns(model, frame)
    beta = np.abs(np.asarray(edited.result.beta, dtype=np.float64))
    centred = centre != 0.0
    pieces = np.abs(columns[:, centred] - centre[centred]) @ beta[centred]
    plain = np.abs(columns[:, ~centred]) @ beta[~centred]
    pair = abs(float(edited.result.centred_intercept)) + abs(
        edited.result.centred_intercept_lo or 0.0
    )
    compensated = (_gamma(3 * p + 2) ** 2 + _U**2) * (pair + pieces)
    fitted = _gamma(p + 2) * _centred_magnitude(model, frame)
    return 2.0 * (fitted + compensated + _gamma(2) * plain + _U * np.abs(expected))


def _numeric_fit(frame, y) -> SuperGLM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={name: Numeric() for name in frame.columns},
        ).fit(frame, y)


def _assert_edit_reads(model, edited, frame, y, slopes) -> None:
    """``predict``, ``metrics`` on the fit rows and a pickle all read the raw reference."""
    expected = _raw_reference(model, frame, slopes)
    bound = _slope_edit_bound(model, edited, frame, expected)
    for label, values in (
        ("predict", edited.predict(frame)),
        ("metrics", edited.metrics(frame, y).eta),
        ("pickled", pickle.loads(pickle.dumps(edited)).predict(frame)),
    ):
        error = np.abs(values - expected)
        assert np.all(error <= bound), f"{label}: max error {float(np.max(error)):.3g}"


@pytest.mark.parametrize("slope", [1.0, -1.0, 1.1], ids=["up", "down", "inexact"])
@pytest.mark.parametrize("scale", [1e16, -1e16])
def test_a_numeric_slope_edit_keeps_the_intercept_remainder(scale, slope):
    """A slope edit on a column spanning zero to 2e16 predicts its raw intercept at zero.

    Sol's review of #448 (P2): ``x = 1e16 {0, 1, 2}`` beside ``y = {2, 3, 2}``
    fits a slope of about 5e-35 about the centre 1e16 and the raw intercept
    2.3333333333333335.  An editor slope is defined about ``x = 0``, so the row
    at zero predicts that intercept.  26321fd6 carried ``c dbeta`` into the
    pair, which became ``(1e16 + 2, 1/3)``, and predicted 2.0 there: the
    remainder joined the row's ``-1e16`` before that cancelled the high part.
    The centre stays, the change is carried exactly, and the sum is
    compensated with the row's exact pieces, so a slope whose product with the
    centre rounds (1.1) reads it too.  Reference: ``eta_before + x dbeta``
    exactly; bound ``_slope_edit_bound``; through ``predict``, ``metrics`` and a
    pickle round trip.
    """
    frame = pd.DataFrame({"x": scale * np.resize([0.0, 1.0, 2.0], 60)})
    y = np.resize([2.0, 3.0, 2.0], 60)
    model = _numeric_fit(frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = _slope_edited(model, frame, y, {"x": slope})
    np.testing.assert_array_equal(edited.result.state_center, model.result.state_center)
    _assert_edit_reads(model, edited, frame, y, {"x": slope})


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_cancelling_slope_edits_on_two_columns_keep_their_rows_apart(sign):
    """Two columns a few ulps apart at 1e12, their slopes edited to +-5e7, keep the rows' differences.

    #449 (Sol's bounded check of #448): ``x = 1e12 + s a`` and ``t = 1e12 + s
    b``, ``s`` the spacing at 1e12 and ``a, b`` in ``{-3, -1, 1, 3}``, with ``y =
    2.5 + 1e8 (x - t)``.  v0.36.0 re-centred both columns at ``c beta_before /
    beta = 2e12``.  Their contributions became about 5e19 each, and their
    cancellation lost the rows' differences: 4177 off.  Keeping the centres
    keeps each row's ``(x - c) beta`` at about 1e4, and the carried ``c dbeta``
    of the two columns is summed exactly.  Reference and bound as in the
    previous test.
    """
    spacing = float(np.spacing(1e12))
    a = np.resize([-3.0, -1.0, 1.0, 3.0], 64)
    b = np.repeat(np.resize([-3.0, -1.0, 1.0, 3.0], 16), 4)[:64]
    frame = pd.DataFrame({"x": sign * (1e12 + spacing * a), "t": sign * (1e12 + spacing * b)})
    y = 2.5 + 1e8 * (frame["x"].to_numpy() - frame["t"].to_numpy())
    model = _numeric_fit(frame, y)
    slopes = {"x": 5e7, "t": -5e7}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = _slope_edited(model, frame, y, slopes)
    _assert_edit_reads(model, edited, frame, y, slopes)


def test_a_slope_edited_to_zero_carries_the_columns_fitted_constant():
    """A numeric effect removed in the editor leaves ``eta_before - x beta_before``.

    A year column (centre about 2010, slope 0.01) edited to a slope of 0: the
    column then contributes nothing, and the pair takes ``-c beta_before``
    from an exact product.  0.36.0 reached the same value through its
    zero-slope fallback, a centre of 0 (#449).  Reference and bound as above.
    """
    year = 2000.0 + np.resize(np.arange(20.0), 60)
    frame = pd.DataFrame({"x": year})
    y = 1.0 + 0.01 * (year - 2010.0) + np.resize([0.1, -0.1, 0.05], 60)
    model = _numeric_fit(frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = _slope_edited(model, frame, y, {"x": 0.0})
    assert edited.result.beta[0] == 0.0
    _assert_edit_reads(model, edited, frame, y, {"x": 0.0})


@pytest.mark.parametrize("slopes", [(-1e308, 1e308), (1e308, -1e308)], ids=["up", "down"])
@pytest.mark.parametrize(
    ("scale", "pattern"),
    [(1e-308, (-1.0, 0.0, 1.0)), (1e-300, (-1.0, 0.0, 1.0)), (1e-300, (1.0, 2.0, 3.0))],
    ids=["zero_centre_1e-308", "zero_centre_1e-300", "centred_1e-300"],
)
def test_slope_edits_across_the_float_range_form_no_coefficient_difference(scale, pattern, slopes):
    """Successive slope edits of -1e308 and 1e308 complete and predict ``intercept + x beta``.

    Sol's review of #448 (P2): ``x = scale {-1, 0, 1}`` has a zero centre, so a
    slope edit moves no centred column.  26321fd6 formed ``beta - beta_before``,
    which overflowed to ``inf`` on the second edit, and ``0 * inf`` poisoned the
    pair.  The revision refused with non-finite scalars where 31544462
    predicted about ``2.33 +- 1`` (or ``+- 1e8``).  ``x = 1e-300 {1, 2, 3}`` keeps
    a centre of 2e-300 and predictions of about 3e8 (#449).  There the carried
    change is ``c beta - c beta_before`` from two finite exact products, about
    4e8; the difference ``c (beta - beta_before)`` would be ``c * inf``.  A
    zero-centre column is skipped without a product.  Reference and bound as
    above.
    """
    frame = pd.DataFrame({"x": scale * np.resize(list(pattern), 60)})
    y = np.resize([2.0, 3.0, 2.0], 60)
    model = _numeric_fit(frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = _slope_edited(model, frame, y, {"x": slopes[0]})
        second = _slope_edited(first, frame, y, {"x": slopes[1]})
    for edited, slope in ((first, slopes[0]), (second, slopes[1])):
        _assert_edit_reads(model, edited, frame, y, {"x": slope})


def test_a_revision_far_from_its_public_intercept_commits():
    """The intercepts are checked to ``gamma_2`` of their magnitudes, not a bare 1e-13 (#449).

    ``y = 1e5 sin(2 pi x)`` under weights that differ between the halves puts
    the public intercept at about 0 and the solver's at about -3.3e4.  An edit
    re-reads ``S = fl(P - h)``, one rounding of up to ``u |S|`` (3.6e-12).  The
    bare ``1e-13 (1 + |P|)`` before #449 refused the commit as an inconsistent
    intercept relation (Claude's review of #448), and v0.36.0 refuses this
    edit.  The derived bound is ``fit_state._intercepts_read_the_shift``.
    Check: the edit commits and predicts the raw public predictor ``P' + X_pub
    beta'`` to each evaluation's ``gamma_(p+3)``.
    """
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 1.0, 200))
    w = np.where(x < 0.5, 1.0, 4.0)
    y = 1e5 * np.sin(2.0 * np.pi * x) + rng.normal(0.0, 1.0, x.size)
    frame = pd.DataFrame({"x": x})

    def fit(target):
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=1e-3,
            features={"x": PSpline(n_knots=8)},
            weight_semantics="frequency",
        ).fit(frame, target, sample_weight=w)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = y - fit(y).result.intercept
        model = fit(y)
        assert abs(model.result.intercept) < 1.0 < 2048.0 < abs(model._solver_result.intercept)
        edited, _, _ = _edit(model, frame, y, w, "x", 0.5, 0.0)
    columns = _public_columns(edited, frame)
    beta = np.asarray(edited.result.beta, dtype=np.float64)
    raw = float(edited.result.intercept) + columns @ beta
    magnitude = _centred_magnitude(edited, frame) + abs(float(edited.result.intercept))
    magnitude += np.abs(columns) @ np.abs(beta)
    bound = 2.0 * _gamma(beta.size + 3) * magnitude
    assert np.all(np.abs(edited.predict(frame) - raw) <= bound)


def test_holdout_drop_term_reads_a_slope_edit_exactly():
    """Dropping an edited numeric term in holdout diagnostics keeps the carried change's cancellation.

    Claude's review of #453: the drop subtracted the term's ``c' beta`` after the
    compensated sum had rounded at ``|alpha|``, and ``alpha`` holds the carried
    ``c dbeta``.  On Sol's P2a fixture with a slope of 1.1 and holdout rows at
    ``x = 0``, where dropping ``x`` changes nothing, the drop read 4.0 against
    2.3333 and ``delta_deviance`` was 16.7.  v0.36.0 read 0.  The shift is now
    subtracted inside the sum as exact products.  Bound: the two predictors
    differ by at most twice the compensated sum's ``u |eta| + gamma_k^2``
    magnitudes ``d``, so the deviances differ by ``2 sum |r| d + n d^2``.
    """
    frame = pd.DataFrame({"x": 1e16 * np.resize([0.0, 1.0, 2.0], 60)})
    y = np.resize([2.0, 3.0, 2.0], 60)
    model = _numeric_fit(frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = _slope_edited(model, frame, y, {"x": 1.1})
    holdout = pd.DataFrame({"x": np.zeros(6)})
    y_holdout = np.full(6, 2.5)
    table = edited.term_drop_diagnostics(
        frame, y, mode="holdout", X_val=holdout, y_val=y_holdout
    ).set_index("feature")
    result = edited.result
    p = result.beta.size
    magnitude = abs(float(result.centred_intercept)) + abs(result.centred_intercept_lo or 0.0)
    magnitude += abs(float(np.asarray(result.state_center)[0]) * float(result.beta[0]))
    eta = edited.predict(holdout)
    d = 2.0 * ((_gamma(3 * p + 4) ** 2 + _U**2) * magnitude + _U * float(np.max(np.abs(eta))))
    bound = 2.0 * float(np.sum(np.abs(y_holdout - eta))) * d + len(eta) * d**2
    assert abs(float(table.loc["x", "delta_deviance"])) <= bound
