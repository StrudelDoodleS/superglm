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


def test_a_levels_alias_gives_its_relativity_and_curve_no_standard_error():
    """``f[E]``'s relativity is the gauge's value, and the fit does not determine it: SE NaN.

    ``relativities(with_se=True)`` and ``term_inference`` test each level's functional (its
    coordinate alone) as ``predict`` tests a row; ``E`` leans on the alias and has a NaN
    standard error and interval, with one warning naming ``'f'``.  The other levels keep the
    standard errors ``term_inference`` printed before, and the relativity values do not move.
    ``non_estimable`` names each level's kind, which the rating table and the editor read.
    """
    model = _fit(ALIASED)
    frames, caught = _caught(lambda: model.relativities(with_se=True))
    (warning,) = _disclosures(caught)
    assert list(warning.points) == ["f"]
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
