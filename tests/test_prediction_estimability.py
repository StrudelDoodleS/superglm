"""Predictions and term values that lean on what a fit does not determine are disclosed once.

``diagnostics.estimability``: a row is off support for a kind of direction when
``||r Q_k|| > 1e-6 ||r||`` in the record's dual Jacobi scale (R's ``predict.lm``
``rankdeficient = "warnif"``).  ``predict`` returns the published fit's value and gives one
``NonEstimablePredictionWarning`` naming the rows and their kinds; a relativity or term curve
off support has a NaN standard error.  Each fixture forces its direction by construction: a
level at one value of its by-level slope (an alias), a response-free indicator (a separation).
At ``253c6d353`` none of these was said and every standard error was printed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm._frame import as_eager_frame
from superglm.model import base

U = 2.0**-53  # the unit roundoff

rng = np.random.default_rng(7)
N = 2000
X_SPREAD = rng.uniform(0, 10, N)
LEVEL = rng.choice(list("ABCDE"), N)
Y = rng.gamma(2.0, np.exp(0.1 * np.sin(X_SPREAD)) / 2.0)
# level E sits at x = 5 alone, so ``x:f[E] = 5 f[E]`` on every row: an alias
ALIASED = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
NEW = pd.DataFrame({"x": [5.0, 7.0, 3.0, 5.0, 2.0], "f": ["E", "E", "A", "B", "E"]})


def _fit(X, y=Y, *, x_spec=None, link="log", method="fit_reml", **kwargs):
    model = SuperGLM(
        family=superglm.families.gamma(),
        link=link,
        features={"x": x_spec or Numeric(), "f": Categorical()},
        interactions=[("x", "f")],
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y)
    return model


def _caught(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    return value, caught


def _disclosures(caught):
    """The off-support warnings, matched by class name (the class is new at this stage)."""
    return [
        c.message for c in caught if type(c.message).__name__ == "NonEstimablePredictionWarning"
    ]


def _design(model, X) -> np.ndarray:
    """Each coefficient's column on the rows of ``X``, scored as ``predict`` scores it."""
    frame, p = as_eager_frame(X), len(model.result.beta)
    plan = base._prediction_plan(model)
    return np.column_stack(
        [
            sum(
                base._score_prediction_term_exact(term, frame, unit)
                for term in (*plan["features"], *plan["interactions"])
            )
            for unit in np.eye(p)
        ]
    )


def _published_eta(model, X) -> np.ndarray:
    """``b0 + x beta`` at the published coefficients, from the dense design rows."""
    return model.result.intercept + _design(model, X) @ model.result.beta


@pytest.mark.parametrize("x_spec", [Numeric, lambda: Spline(n_knots=6)], ids=["numeric", "spline"])
def test_rows_off_a_levels_alias_are_named_once_at_the_gauges_value(x_spec):
    """Level E at one ``x``: its rows at another ``x`` lean on the alias the gauge fixes.

    The alias is ``f[E]`` against ``x:f[E]``'s slope (numeric) or its by-level spline's
    unpenalised part (spline).  Rows 1 and 4 (``E`` away from ``x = 5``) are named, with kind
    ``"alias"``; ``E`` at ``x = 5`` and the other levels are not.  The predictions are the
    published gauge's (``b0 + x beta`` at the published coefficients, to ``kappa u``).  The
    fit's own rows are all on support: no warning.
    """
    model = _fit(ALIASED, x_spec=x_spec())
    mu, caught = _caught(lambda: model.predict(NEW))
    (warning,) = _disclosures(caught)
    assert warning.rows.tolist() == [1, 4]
    assert warning.kinds == ("alias", "alias")
    assert "rows 1, 4" in str(warning) and warning.no_mean == 0
    eta = _published_eta(model, NEW)
    np.testing.assert_allclose(np.log(mu), eta, rtol=0, atol=1e3 * U * np.max(np.abs(eta)))
    _, caught = _caught(lambda: model.predict(ALIASED))
    assert not _disclosures(caught)


def test_a_rows_share_is_measured_in_the_dual_jacobi_scale_against_the_dense_rows():
    """The share ``||r Q|| / ||r||`` against a dense oracle of the record's metric.

    Unpenalised Gamma with the log link: the Fisher weights are one, so the centred Hessian is
    ``H = X_c' X_c`` with ``X_c`` the training design centred on its mean ``c``.  In the
    centred coordinates the scoring row is ``[1, x - c]``, the Jacobi scale
    ``diag(sqrt(n), sqrt(diag H))``, and ``Q`` an orthonormal basis of ``D null(X_c)``.  The
    disclosed products and norms equal the oracle's to ``kappa u`` (the share's condition is
    the dense null space's, about ``1 / sigma_min`` of the scaled design).
    """
    model = _fit(ALIASED)
    _, caught = _caught(lambda: model.predict(NEW))
    assert [w.rows.tolist() for w in _disclosures(caught)] == [[1, 4]]
    X = _design(model, ALIASED)
    centre = X.mean(axis=0)
    D = np.sqrt(np.sum((X - centre) ** 2, axis=0))
    _, singular, right = np.linalg.svd((X - centre) / D, full_matrices=False)
    Q = right[singular < 1e-8 * singular[0]].T
    assert Q.shape[1] == 1
    rows = (_design(model, NEW) - centre) / D
    oracle_share = np.abs(rows @ Q).ravel()
    oracle_norm = np.sqrt(1.0 / N + np.sum(rows**2, axis=1))

    from superglm.model.estimability_ops import published_directions

    frame, plan = as_eager_frame(NEW), base._prediction_plan(model)
    directions = published_directions(model)
    share = np.abs(
        base._direction_products(
            model,
            frame,
            plan,
            directions.slopes["alias"],
            directions.intercepts["alias"],
            "conditional",
        )
    ).ravel()
    norm = base._row_norms(model, frame, plan, directions, "conditional")
    kappa = singular[0] / singular[singular >= 1e-8 * singular[0]][-1]
    np.testing.assert_allclose(norm, oracle_norm, rtol=1e3 * kappa * U)
    np.testing.assert_allclose(share, oracle_share, rtol=0, atol=1e3 * kappa * U)


def test_a_rows_norm_is_the_same_across_its_blocks(monkeypatch):
    """``_row_norms`` reads a term's design rows in blocks: two-row blocks give the same norms.

    Each row's square is its own sum, so the blocking changes no rounding: bitwise equal.
    """
    from superglm.model.estimability_ops import published_directions

    model = _fit(ALIASED)
    frame, plan = as_eager_frame(NEW), base._prediction_plan(model)
    directions = published_directions(model)
    whole = base._row_norms(model, frame, plan, directions, "conditional")
    widest = max(len(term["beta_idx"]) for term in (*plan["features"], *plan["interactions"]))
    monkeypatch.setattr(base, "_NORM_BLOCK_ENTRIES", 2 * widest)
    np.testing.assert_array_equal(
        base._row_norms(model, frame, plan, directions, "conditional"), whole
    )


def test_a_levels_alias_gives_its_relativity_and_curve_no_standard_error():
    """``f[E]``'s relativity is the gauge's value, and the fit does not determine it: SE NaN.

    ``relativities(with_se=True)`` and ``term_inference`` test each level's functional (its
    coordinate alone) as ``predict`` tests a row; ``E`` leans on the alias and has a NaN
    standard error and interval, with one warning naming ``'f'`` and the interaction ``'x:f'``,
    whose level ``E`` leans on it too.  The other levels keep the standard errors
    ``term_inference`` printed before, and the relativity values do not move.
    ``non_estimable`` names each level's kind, which the rating table and the editor read.
    """
    model = _fit(ALIASED)
    frames, caught = _caught(lambda: model.relativities(with_se=True))
    (warning,) = _disclosures(caught)
    assert list(warning.points) == ["f", "x:f"]
    frame = frames["f"]
    levels = frame["level"].tolist()
    e = levels.index("E")
    assert np.isnan(frame["se_log_relativity"].iloc[e])
    others = np.delete(frame["se_log_relativity"].to_numpy(), e)
    assert np.all(np.isfinite(others)) and np.count_nonzero(others) == len(levels) - 2

    inference, caught = _caught(lambda: model.term_inference("f"))
    assert len(_disclosures(caught)) == 1
    assert inference.non_estimable.tolist() == ["alias" if lv == "E" else "" for lv in levels]
    assert np.isnan(inference.se_log_relativity[e])
    assert np.isnan(inference.ci_lower[e]) and np.isnan(inference.ci_upper[e])
    np.testing.assert_array_equal(np.delete(inference.se_log_relativity, e), others)
    np.testing.assert_array_equal(inference.log_relativity, frame["log_relativity"].to_numpy())
    # without standard errors the points are flagged and nothing is said
    plain, caught = _caught(lambda: model.term_inference("f", with_se=False))
    assert not _disclosures(caught)
    assert plain.non_estimable.tolist() == inference.non_estimable.tolist()


def test_a_levels_alias_gives_its_feature_se_no_value():
    """``metrics().feature_se("f")``: ``E``'s level leans on the alias, so its SE is NaN.

    The same test as the term readers' (``explain_ops._curve_labels``), on the functional this
    reader reports: each level's coordinate.  The other levels keep the quasi-likelihood SEs
    ``feature_se`` printed before, which are ``term_inference``'s times the Pearson dispersion's
    square root over the fitted scale's (to the rounding of one product); one warning names
    ``'f'``.  At ``253c6d353`` ``E`` printed a finite SE.
    """
    model = _fit(ALIASED)
    metrics = model.metrics(ALIASED, Y)
    values, caught = _caught(lambda: metrics.feature_se("f"))
    (warning,) = _disclosures(caught)
    assert list(warning.points) == ["f"]
    levels = list(values["levels"])
    e = levels.index("E")
    se = np.asarray(values["se_log_relativity"])
    assert np.isnan(se[e]) and np.all(np.isfinite(np.delete(se, e)))
    reference, _ = _caught(lambda: model.term_inference("f"))
    order = [list(reference.levels).index(level) for level in levels]
    expected = np.asarray(reference.se_log_relativity)[order] * np.sqrt(
        metrics._coefficient_dispersion / model.result.phi
    )
    np.testing.assert_allclose(np.delete(se, e), np.delete(expected, e), rtol=16 * U)


def test_a_spline_curves_points_off_support_have_no_simultaneous_band():
    """A spline beside its own linear column: its curve leans on the alias the gauge holds.

    ``xl = x`` duplicates the spline's unpenalised linear direction, so ``null(H)`` moves the
    spline's curve by a line and ``xl`` against it; the published constraint holds ``xl`` at
    zero (D8: the later term), and every point of the spline's curve but those where that line
    vanishes leans on it.  ``simultaneous_bands`` gives those points no standard error and no
    band, with one warning, as ``term_inference`` labels them; a point the fit determines keeps
    its band.  At ``253c6d353`` the bands were the constraint's.  Mutation: the bands read
    without the off-support test.
    """
    rows = np.random.default_rng(3)
    x = rows.uniform(0.0, 10.0, N)
    frame = pd.DataFrame({"x": x, "xl": x})
    response = rows.gamma(2.0, np.exp(0.1 * np.sin(x)) / 2.0)
    model = SuperGLM(
        family=superglm.families.gamma(),
        link="log",
        features={"x": Spline(n_knots=6), "xl": Numeric()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, response)
    inference, _ = _caught(lambda: model.term_inference("x", with_se=False))
    flagged = np.asarray(inference.non_estimable) != ""
    assert np.count_nonzero(flagged) > len(flagged) // 2
    bands, caught = _caught(lambda: model.simultaneous_bands("x", n_sim=500))
    assert len(_disclosures(caught)) == 1
    np.testing.assert_array_equal(bands["x"].to_numpy(), np.asarray(inference.x))
    for column in ("se", "ci_lower_simultaneous", "ci_upper_pointwise"):
        values = bands[column].to_numpy()
        assert np.all(np.isnan(values[flagged])) and np.all(np.isfinite(values[~flagged])), column


def test_a_rating_table_flags_the_levels_the_fit_does_not_determine():
    """The categorical block of ``f`` carries each level's kind; its workbook note names ``E``."""
    from superglm.export.rating_tables import _categorical_block

    model = _fit(ALIASED)
    block = _categorical_block(model, as_eager_frame(ALIASED), "f", None, "native")
    levels = block.table["f"].tolist()
    flags = getattr(block, "non_estimable", None)
    assert flags is not None
    assert flags.tolist() == ["alias" if lv == "E" else "" for lv in levels]
    from superglm.export.excel import _non_estimable_note

    note = _non_estimable_note(block)
    assert note.startswith("Rows E: the fit does not determine their relativity")


def test_an_inverse_link_row_off_support_without_a_mean_is_named_once():
    """Each row is named in one warning: off support, with or without a mean, in the disclosure.

    Under the inverse link a row has a mean only at ``eta > 0``; an offset of -100 pushes rows 0
    (``E`` at ``x = 7``, off support) and 1 (``A``, on support) outside it.  Row 2 is off support
    with a mean, row 3 neither.  The disclosure names rows 0 and 2 and counts row 0 as having no
    mean ("outside the inverse link's domain at the returned predictor": its offset, not the
    gauge, put it there); the no-mean warning names row 1 alone.  The readers' refusal is
    unchanged: it names rows 0 and 1.  Under a reader's quiet scope the disclosure stands without
    the count, and a silent internal prediction says nothing.
    """
    model = _fit(ALIASED, link="inverse")
    X = pd.DataFrame({"x": [7.0, 3.0, 7.0, 3.0], "f": ["E", "A", "E", "A"]})
    offset = np.array([-100.0, -100.0, 0.0, 0.0])
    mu, caught = _caught(lambda: model.predict(X, offset=offset))
    assert np.isnan(mu[:2]).all() and np.isfinite(mu[2:]).all()
    (warning,) = _disclosures(caught)
    assert warning.rows.tolist() == [0, 2] and warning.no_mean == 1
    assert "1 of them have no mean at the returned predictor" in str(warning)
    assert "outside the inverse link's domain" in str(warning)
    no_mean = [str(c.message) for c in caught if "have no predicted mean" in str(c.message)]
    assert no_mean == [no_mean[0]] and "1 of 4 rows" in no_mean[0]
    assert "(row positions 1)" in no_mean[0]

    with pytest.raises(base.NoPredictedMeanError) as refused:
        _caught(lambda: base.predict_priced(model, X, offset, np.ones(4), "the deviance"))
    assert refused.value.positions.tolist() == [0, 1]

    with base.no_mean_warning_quiet():
        _, caught = _caught(lambda: model.predict(X, offset=offset))
    (warning,) = _disclosures(caught)
    assert warning.rows.tolist() == [0, 2] and warning.no_mean == 0
    assert not [c for c in caught if "have no predicted mean" in str(c.message)]

    _, caught = _caught(lambda: base.predict_exact(model, X, offset, warn=False))
    assert not caught


def test_off_support_disclosure_costs_one_pass_per_direction_and_one_norm_pass(monkeypatch):
    """With one alias direction ``predict`` makes one extra coefficient pass and one norm pass.

    Counted on the predictor's own term pass (``_score_terms``) and the norm pass
    (``_row_norms``), with the disclosure as the public output.
    """
    model = _fit(ALIASED)
    _, caught = _caught(lambda: model.predict(NEW))
    assert len(_disclosures(caught)) == 1
    counts = {"passes": 0, "norms": 0}
    score_terms, row_norms = base._score_terms, base._row_norms

    def counted_terms(*args, **kwargs):
        counts["passes"] += 1
        return score_terms(*args, **kwargs)

    def counted_norms(*args, **kwargs):
        counts["norms"] += 1
        return row_norms(*args, **kwargs)

    monkeypatch.setattr(base, "_score_terms", counted_terms)
    monkeypatch.setattr(base, "_row_norms", counted_norms)
    _, caught = _caught(lambda: model.predict(NEW))
    assert len(_disclosures(caught)) == 1
    assert counts == {"passes": 2, "norms": 1}


@pytest.mark.pins_behaviour
def test_a_fit_without_such_directions_makes_no_extra_pass(monkeypatch):
    """Control: a fit whose record has no such direction scores each term once, forms no row.

    Level E spread over ``x`` identifies ``x:f[E]``, so the record holds no direction and
    ``predict`` reads nothing more than its own pass: every term is scored once and no term's
    design rows are formed (``transform``).  It passed at ``253c6d353``, which had no
    disclosure; it pins that the disclosure costs a fit without directions nothing.
    """
    model = _fit(pd.DataFrame({"x": X_SPREAD, "f": LEVEL}))
    plan = base._prediction_plan(model)
    calls = {"scored": 0, "rows": 0}
    exact, centred = base._score_prediction_term_local_exact, base._centred_term_contribution

    def scored(*args, **kwargs):
        calls["scored"] += 1
        return exact(*args, **kwargs)

    def scored_centred(*args, **kwargs):
        calls["scored"] += 1
        return centred(*args, **kwargs)

    monkeypatch.setattr(base, "_score_prediction_term_local_exact", scored)
    monkeypatch.setattr(base, "_centred_term_contribution", scored_centred)
    for term in (*plan["features"], *plan["interactions"]):
        spec, transform = term["spec"], term["spec"].transform

        def formed(*args, _transform=transform, **kwargs):
            calls["rows"] += 1
            return _transform(*args, **kwargs)

        monkeypatch.setattr(spec, "transform", formed)
    _, caught = _caught(lambda: model.predict(NEW))
    assert not _disclosures(caught)
    assert calls == {"scored": len(plan["features"]) + len(plan["interactions"]), "rows": 0}


def _separated_indicator():
    """A Poisson indicator whose rows all have response zero: ``z`` has no finite estimate."""
    z = np.repeat([0.0, 1.0], 64)
    frame, y = pd.DataFrame({"z": z}), 1.0 - z
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="poisson", features={"z": Numeric()}).fit(frame, y)
    return model, frame, y


def test_rows_along_a_separated_direction_are_named_separated():
    """``z``'s rows lean on the separated direction and are named ``"separated"``.

    The decision's span is ``z``; rows at ``z = 0`` are on support.  The values are where the
    walk stopped (``b0 + z beta``).  A revision that sets ``z``'s slope (the editor) leaves no
    separated value, so the edited model's predictions say nothing: the published statement,
    not the record's decision, is read.  Opening the editor flags the point without a warning.
    """
    from superglm.editor import EditorSession

    model, frame, y = _separated_indicator()
    assert model.result.separation_statement.tag == "present"
    new = pd.DataFrame({"z": [0.0, 1.0, 0.5]})
    mu, caught = _caught(lambda: model.predict(new))
    (warning,) = _disclosures(caught)
    assert warning.rows.tolist() == [1, 2] and warning.kinds == ("separated", "separated")
    eta = model.result.intercept + model.result.beta[0] * new["z"].to_numpy()
    np.testing.assert_allclose(np.log(mu), eta, rtol=0, atol=1e3 * U * np.max(np.abs(eta)))

    # the editor shows each point's kind on its curve, so opening it gives no warning
    session, caught = _caught(
        lambda: EditorSession.from_model(model, terms=["z"], train_data=(frame, y))
    )
    assert not _disclosures(caught)
    assert session.terms["z"].metadata["non_estimable"] == ["separated"]
    session.terms["z"].edited_log_effect = np.array([0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = session.to_model()
    _, caught = _caught(lambda: edited.predict(new))
    assert not _disclosures(caught)


def test_a_separated_level_is_named_once_by_the_decision():
    """A level with only zero responses: its rows are disclosed once, as ``"separated"``.

    The build scan named the level at fit; at predict the decision's span names its rows, so
    the per-level warning (which a fit without the decision still gives) is not repeated.  Its
    relativity has no standard error.
    """
    rng = np.random.default_rng(5)
    c = rng.choice([f"c{i}" for i in range(8)], size=2500)
    y = np.where(c == "c6", 0.0, rng.poisson(0.5, size=c.size).astype(float) + 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="poisson", features={"c": Categorical()}).fit(
            pd.DataFrame({"c": c}), y
        )
    _, caught = _caught(lambda: model.predict(pd.DataFrame({"c": ["c6", "c1", "c6"]})))
    (warning,) = _disclosures(caught)
    assert warning.rows.tolist() == [0, 2] and warning.kinds == ("separated", "separated")
    assert not [c for c in caught if "only boundary responses at fit" in str(c.message)]
    frames, caught = _caught(lambda: model.relativities(with_se=True))
    levels = frames["c"]["level"].tolist()
    assert np.isnan(frames["c"]["se_log_relativity"].iloc[levels.index("c6")])
    assert len(_disclosures(caught)) == 1


@pytest.mark.parametrize("x_spec", [Numeric, lambda: Spline(n_knots=6)], ids=["numeric", "spline"])
def test_an_interactions_points_off_the_alias_are_flagged_by_the_same_test(x_spec):
    """``x:f[E]``'s slope (numeric) or curve (spline) leans on the alias; the other levels' do not.

    ``term_inference`` on the interaction flags its points (``non_estimable``, keyed as the
    inference keys them) and, with standard errors asked for, warns once;
    ``relativities(with_se=True)`` names the interaction's frames in its warning.  Without
    standard errors nothing is said.
    """
    model = _fit(ALIASED, x_spec=x_spec())
    inference, caught = _caught(lambda: model.term_inference("x:f"))
    flagged = getattr(inference, "non_estimable", None)
    assert flagged
    if inference.kind == "numeric_categorical":
        assert flagged == {"E": "alias"}
    else:
        assert {level for level, _ in flagged} == {"E"}
        assert set(flagged.values()) == {"alias"}
        assert len(flagged) >= len(inference.x) - 1
    assert len(_disclosures(caught)) == 1
    _, caught = _caught(lambda: model.term_inference("x:f", with_se=False))
    assert not _disclosures(caught)
    _, caught = _caught(lambda: model.relativities(with_se=True))
    (warning,) = _disclosures(caught)
    assert any(str(name).startswith("x:f") for name in warning.points)


def test_a_spline_by_level_rating_block_flags_the_levels_cells():
    """The ``x:f`` cell block flags level ``E``'s cells alone; the workbook note names them."""
    from superglm.export.rating_tables import build_rating_table_payload

    model = _fit(ALIASED, x_spec=Spline(n_knots=6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = build_rating_table_payload(model, ALIASED, Y, n_bins=10, impact_bins=(10,))
    (block,) = [b for b in payload.interactions if b.name == "x:f"]
    flags = getattr(block, "non_estimable", None)
    assert flags is not None
    columns = list(block.table.columns[1:])
    flagged_columns = {columns[j] for j in np.nonzero(flags != "")[1]}
    assert flagged_columns == {"E"}
    assert np.count_nonzero(flags[:, columns.index("E")] == "alias") >= len(block.table) - 1
    from superglm.export.excel import _non_estimable_cells_note

    assert _non_estimable_cells_note(block).startswith("Cells (")


def _shifted_alias_fit():
    """Two splines on ``z = 2 x + 1``: their linear parts are one alias the gauge holds.

    The rows of weight zero sit off the line, so the public columns' training means ``m`` (all
    rows) differ from the fit's weighted centre ``c`` and the alias's public intercept carries
    ``m'q`` (``estimability_ops``, "Coordinates").
    """
    gen = np.random.default_rng(3)
    n, k = 1500, 300
    x = gen.uniform(0, 10, n)
    z = 2.0 * x + 1.0
    w = np.ones(n)
    x[:k], z[:k], w[:k] = gen.uniform(8, 10, k), gen.uniform(0, 5, k), 0.0
    y = gen.gamma(2.0, np.exp(0.1 * np.sin(x)) / 2.0)
    frame = pd.DataFrame({"x": x, "z": z})
    model = SuperGLM(
        family=superglm.families.gamma(),
        link="log",
        features={"x": Spline(n_knots=6), "z": Spline(n_knots=6)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=w)
    return model, frame, y, w


def test_rows_on_an_alias_of_two_centred_splines_are_on_support():
    """The fit's own weighted rows lie on the alias; its weightless rows, off the line, do not.

    The public intercept of the alias carries the materialised spline columns' shift; without
    it every row on the line would lean on the alias by ``|m'q|``.
    """
    model, frame, _, w = _shifted_alias_fit()
    _, caught = _caught(lambda: model.predict(frame[w > 0]))
    assert not _disclosures(caught)
    _, caught = _caught(lambda: model.predict(frame[w == 0]))
    (warning,) = _disclosures(caught)
    assert warning.rows.size == int(np.count_nonzero(w == 0))
    assert set(warning.kinds) == {"alias"}


def test_a_binned_spline_block_flags_its_bins_on_the_alias():
    """The binned blocks of ``x`` and ``z``: a bin's value is the term's mean over its rows.

    Both terms carry the alias's linear part, so their bins lean on it and are flagged.
    """
    from superglm.export.rating_tables import build_rating_table_payload

    model, frame, y, w = _shifted_alias_fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = build_rating_table_payload(
            model, frame, y, sample_weight=w, n_bins=8, impact_bins=(8,)
        )
    for block in payload.main_effects:
        flags = getattr(block, "non_estimable", None)
        assert block.kind == "continuous" and flags is not None
        assert np.count_nonzero(flags == "alias") >= len(block.table) - 1


def test_the_editor_keeps_the_band_of_every_level_the_fit_determines():
    """Only ``E`` loses its band in the editor's payload; the other levels keep theirs.

    A band whose end is not finite on a determined level still has no band at all (the rule
    for any other non-finite end).
    """
    from superglm.editor import EditorSession
    from superglm.editor.display import display_for
    from superglm.editor.payloads import _ci_payload, session_payload

    model = _fit(ALIASED)
    session = EditorSession.from_model(model, terms=["f"], train_data=(ALIASED, Y))
    term = session_payload(session)["f"]
    levels = [str(level) for level in term["levels"]]
    e = levels.index("E")
    assert term["ci_lower_y"][e] is None and term["ci_upper_y"][e] is None
    kept = [i for i in range(len(levels)) if i != e]
    assert all(isinstance(term["ci_lower_y"][i], float) for i in kept)
    assert all(term["ci_lower_y"][i] <= term["ci_upper_y"][i] for i in kept)
    assert term["non_estimable"][e] == "alias"

    editable = session.terms["f"]
    editable.ci_lower_log_effect = editable.ci_lower_log_effect.copy()
    editable.ci_lower_log_effect[kept[0]] = np.nan
    delta = np.zeros(editable.size)
    assert _ci_payload(editable, delta, display_for(model, "f")) == (None, None)


def test_an_sz_terms_row_norm_is_the_functional_predict_scores(monkeypatch):
    """A saved ``sz`` model whose levels its data identify only in part: rows read as predicted.

    Its term predicts a level left out of the population at the population curve, a row that
    is not the term's design row.  With a direction forced along one of the term's coordinates
    (``published_directions``), ``predict`` names the rows that reach it, and each row's norm in
    the dual scale equals the norm of the predictor's own row: the predictor at each unit
    coordinate (``_score_terms``), centred and scaled, to ``kappa u``.
    """
    import importlib
    import importlib.util
    import pickle
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "saved_v0_36_0" / "sz_gaussian_one_row.pkl"
    with open(path, "rb") as handle:
        record = pickle.load(handle)
    model, grid = record["model"], record["grid"]
    spec = model._interaction_specs["x:g:sz"]
    assert spec._has_population_offset
    plan = base._prediction_plan(model)
    p = len(model.result.beta)
    gen = np.random.default_rng(11)
    scale, centre = gen.uniform(0.5, 2.0, p), gen.normal(size=p)
    (term,) = [t for t in plan["interactions"] if t["name"] == "x:g:sz"]
    direction = np.zeros((p, 1))
    direction[term["beta_idx"][0], 0] = 1.0 / scale[term["beta_idx"][0]]
    forced = None
    if importlib.util.find_spec("superglm.model.estimability_ops") is not None:
        estimability_ops = importlib.import_module("superglm.model.estimability_ops")
        forced = estimability_ops.PublishedDirections(
            {"unresolved": direction}, {"unresolved": np.zeros(1)}, scale, centre, 3.0
        )
        monkeypatch.setattr(estimability_ops, "published_directions", lambda model: forced)
    _, caught = _caught(lambda: model.predict(grid))
    (warning,) = _disclosures(caught)
    assert set(warning.kinds) == {"unresolved"} and warning.rows.size

    frame = as_eager_frame(grid)

    def predictor(beta):
        eta = base.EtaSum(len(frame), 0.0, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base._score_terms(
                model, frame, plan, beta, eta, None, random_effects="conditional", fitted=False
            )
        return eta.finish()

    rows = np.column_stack([predictor(unit) for unit in np.eye(p)])
    oracle = np.sqrt(1.0 / 9.0 + np.sum(((rows - centre) / scale) ** 2, axis=1))
    norms = base._row_norms(model, frame, plan, forced, "conditional")
    np.testing.assert_allclose(norms, oracle, rtol=1e3 * p * U)
