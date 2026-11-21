"""What a user reads about a categorical level, or crossed cell, with exposure and no claims.

Such a level is separated: its effect has no finite maximum-likelihood estimate, and every
engine (R ``glm``, mgcv, statsmodels, glum) fits it and stops somewhere on the way to minus
infinity with ``converged`` true.  The fit keeps the level and its price; these tests pin
the disclosure around it, one acceptance row each:

1. one ``SeparationWarning`` after the fit, in plain words, priced by the fit it describes,
   and no second notice of the same level from REML or the runtime backstop;
2. coefficient labels name levels and cells, not column positions;
3. ``summary()``, ``term_inference()``, ``relativities()`` and the workbook summary keep the
   estimate, flag it and leave its Wald standard error, z, p and interval blank;
4. ``centering="mean"`` centres on the levels that are not separated;
6. the rating-table block carries a note naming the level;
7. ``predict`` warns once per call, for levels and cells;
8. a level a fit-time cap bounds is not separated;
9. ``Categorical(grouping=dict)`` is refused by name, and the warning's grouping call works.

Each test fails at master 05004eb95.  Assertions read labels, flags, blanks and the warning's
count and fields; the one number checked against an independent reference is the response
the level's rows would expect at the base rate, which ``predict`` gives at the base level.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from superglm import (
    Categorical,
    Constraint,
    GroupLasso,
    Numeric,
    RandomEffect,
    SuperGLM,
    collapse_levels,
)
from superglm import Spline as SplineTerm
from superglm.diagnostics.separation import (
    SeparationError,
    SeparationWarning,
    format_runtime_message,
)
from superglm.reml.identified import WeakIdentificationWarning, coefficient_labels


def _book(n=8000, seed=0, levels=tuple("ABCDE")):
    """Poisson claims on exposure; the last level (about 40 rows, 2 expected claims) has none."""
    rng = np.random.default_rng(seed)
    lev = rng.choice(list(levels), n, p=[0.4, 0.3, 0.2, 0.095, 0.005])
    x = rng.uniform(0.0, 1.0, n)
    exposure = rng.uniform(0.2, 1.0, n)
    mu = exposure * 0.08 * np.exp(0.5 * x) * np.where(lev == levels[1], 1.3, 1.0)
    y = rng.poisson(mu).astype(float)
    y[lev == levels[-1]] = 0.0
    return pd.DataFrame({"lev": lev, "x": x}), y, exposure


def _model(**options) -> SuperGLM:
    lev = options.pop("lev", Categorical())
    return SuperGLM(
        family="poisson",
        features={"lev": lev, "x": SplineTerm(n_knots=6)},
        selection_penalty=0.0,
        **options,
    )


def _fit(model, X, y, exposure, method="fit"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(model, method)(X, y, offset=np.log(exposure))
    return caught


def _of(caught, category):
    return [str(w.message) for w in caught if issubclass(w.category, category)]


@pytest.fixture(scope="module")
def fitted():
    X, y, exposure = _book()
    models = {}
    for method in ("fit", "fit_reml"):
        model = _model()
        models[method] = (model, _fit(model, X, y, exposure, method))
    return X, y, exposure, models


# ── row 1: one warning after the fit, priced by it ────────────────────────


@pytest.mark.parametrize("method", ["fit", "fit_reml", "discrete"])
def test_a_fit_warns_once_after_it_with_the_level_priced(fitted, method):
    """One ``SeparationWarning``, stating rows, exposure, the base-rate expectation and price.

    The expected response is checked against ``predict`` at the base level on the level's
    own rows, the definition the warning states.  On master the warning was the build's
    ("IRLS will drift"), with no exposure or price, and ``fit_reml`` added a
    ``WeakIdentificationWarning`` for ``lev[3]``.
    """
    X, y, exposure, models = fitted
    if method == "discrete":
        model = _model(discrete=True)
        caught = _fit(model, X, y, exposure, "fit_reml")
    else:
        model, caught = models[method]
    separation = _of(caught, SeparationWarning)
    assert len(separation) == 1
    assert not _of(caught, WeakIdentificationWarning)
    message = separation[0]
    assert model.result.converged

    rows = (X["lev"] == "E").to_numpy()
    at_base = model.predict(X[rows].assign(lev="A"), offset=np.log(exposure[rows]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        fitted = model.predict(X[rows], offset=np.log(exposure[rows]))
    line = next(text for text in message.splitlines() if "'lev' level 'E'" in text)
    assert f"{rows.sum()} rows" in line
    assert f"exposure {exposure[rows].sum():.4g}" in line
    assert "every response 0" in line
    assert f"{at_base.sum():.3g} expected in total at the rate of base level 'A'" in line
    assert f"fitted {fitted.sum():.3g} in total" in line
    assert "where the fit stopped, not an estimate" in message
    assert "IRLS" not in message


def test_a_refusal_states_the_level_before_any_fit():
    """``separation="error"`` refuses at build with the plain text and no fitted numbers."""
    X, y, exposure = _book()
    model = _model(separation="error")
    with pytest.raises(SeparationError) as refused:
        model.fit(X, y, offset=np.log(exposure))
    message = str(refused.value)
    assert "'lev' level 'E'" in message and "every response 0" in message
    assert "in total" not in message
    assert model._result is None


def _caught(fit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit()
    return [w for w in caught if issubclass(w.category, SeparationWarning)]


def _two_levels(y_b, **options):
    """100 rows at A, a response 1 on every fifth, and 100 at B responding ``y_b``."""
    X = pd.DataFrame({"a": np.repeat(["A", "B"], 100)})
    y = np.r_[(np.arange(100) % 5 == 0).astype(float), np.full(100, float(y_b))]
    model = SuperGLM(
        family="poisson", features={"a": Categorical(base="A")}, selection_penalty=0.0, **options
    )
    return model, X, y


@pytest.mark.pins_behaviour
def test_a_refit_whose_warning_is_raised_keeps_the_previous_fit():
    """Under warnings as errors, the warning is raised before the refit is installed.

    Control: pins master's behaviour, whose build raised the warning before any fit.
    At 20f77dd6f it was raised after the install: B's price moved from 0.2 to 8e-8.
    """
    model, X, y = _two_levels(1.0)
    model.fit(X, y)
    revision, price = model._fit_revision, model.predict(X.tail(1))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        with pytest.raises(SeparationWarning):
            model.fit(X, np.where(X["a"] == "B", 0.0, y))
    assert model._fit_revision == revision
    assert model.predict(X.tail(1))[0] == price


def test_the_warning_names_the_callers_line():
    """Both the published fit's warning and ``drop1``'s, which publishes none, land here.

    At 20f77dd6f both were located at ``contextlib.py``.
    """
    model, X, y = _two_levels(0.0)
    (published,) = _caught(lambda: model.fit(X, y))
    assert published.filename == __file__
    X = X.assign(x=np.linspace(0.0, 1.0, 200))
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A"), "x": Numeric()},
        selection_penalty=0.0,
    )
    _caught(lambda: model.fit(X, y))
    (compared,) = _caught(lambda: model.drop1(X, y))
    assert compared.filename == __file__ and "'a' level 'B'" in str(compared.message)


def _cells(cells, **options):
    """A Poisson ``a x b`` book, 100 rows per cell; ``cells`` lists (a, b, rows with a 1)."""
    data = [(a, b, float(i < events)) for a, b, events in cells for i in range(100)]
    frame = pd.DataFrame(data, columns=["a", "b", "y"])
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A"), "b": Categorical(base="P")},
        interactions=[("a", "b")],
        selection_penalty=0.0,
        **options,
    )
    caught = _caught(lambda: model.fit(frame[["a", "b"]], frame["y"].to_numpy()))
    return model, frame[["a", "b"]], caught


def _line(caught, name):
    (warning,) = caught
    return next(text for text in str(warning.message).splitlines() if name in text)


def test_the_base_rate_moves_the_level_in_every_term_it_enters():
    """C beside ``a:b``: 4 expected at A, what ``predict`` gives C's rows set to A.

    Codex's case. At 20f77dd6f the walked main effect alone was taken off, leaving C's
    interaction cells in: the warning read 2.
    """
    model, X, caught = _cells(
        [("A", "P", 1), ("A", "Q", 3), ("B", "P", 2), ("B", "Q", 2), ("C", "P", 0), ("C", "Q", 0)]
    )
    rows = (X["a"] == "C").to_numpy()
    at_base = model.predict(X[rows].assign(a="A")).sum()
    assert at_base == pytest.approx(4.0, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        fitted = model.predict(X[rows]).sum()
    line = _line(caught, "'a' level 'C'")
    assert f"{at_base:.3g} expected in total at the rate of base level 'A'" in line
    assert f"fitted {fitted:.3g} in total" in line


def test_the_base_rate_beside_a_walk_past_float64():
    """Binomial/log ``fit_reml`` walks c to -5.5e27: 24.0 expected at a, as ``predict`` says.

    At 20f77dd6f the predictor less the walk cancelled the other terms: 200.
    """
    rng = np.random.default_rng(15)
    c = np.repeat(["a", "b", "c"], 200)
    group = np.tile(np.repeat([f"g{i}" for i in range(10)], 20), 3)
    y = (rng.uniform(size=600) < np.repeat([0.2, 0.3, 0.35], 200)).astype(float)
    y[c == "c"] = 0.0
    X = pd.DataFrame({"c": c, "g": group})
    model = SuperGLM(
        family="binomial",
        link="log",
        features={"c": Categorical(base="a"), "g": RandomEffect()},
        selection_penalty=0.0,
    )
    caught = _caught(lambda: model.fit_reml(X, y))
    at_base = model.predict(X[c == "c"].assign(c="a")).sum()
    assert 20.0 < at_base < 30.0
    line = _line(caught, "'c' level 'c'")
    assert f"{at_base:.3g} expected in total at the rate of base level 'a'" in line


@pytest.mark.parametrize("link", ["probit", "cauchit"])
def test_every_link_states_finite_totals(link):
    """All-1 B against P(A) = 0.5: 50 expected at A, about 100 fitted, no overflow.

    At 20f77dd6f the relativity ``exp`` of the walked slope read 151 (probit), and the
    cauchit fit raised ``FloatingPointError`` under ``np.errstate(over="raise")``.
    """
    X = pd.DataFrame({"a": np.repeat(["A", "B"], 100)})
    y = np.r_[np.tile([0.0, 1.0], 50), np.ones(100)]
    model = SuperGLM(
        family="binomial", link=link, features={"a": Categorical(base="A")}, selection_penalty=0.0
    )
    with np.errstate(over="raise"):
        caught = _caught(lambda: model.fit(X, y))
    line = _line(caught, "'a' level 'B'")
    assert "every response 1; 50 expected in total at the rate of base level 'A'" in line
    assert "fitted 100 in total" in line


def test_no_base_rate_for_an_empty_base_cell():
    """A x P holds no rows, so the fit does not determine its rate: no figure is stated.

    At 20f77dd6f the warning read 160 expected at A x P, and an equivalent fit with the
    same training predictions gives 320.
    """
    X = pd.DataFrame({"a": np.repeat(["A", "B", "B"], 40), "b": np.repeat(["Q", "P", "Q"], 40)})
    y = np.repeat([4.0, 2.0, 0.0], 40)
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A"), "b": Categorical(base="P")},
        interactions=[("a", "b")],
        selection_penalty=0.0,
    )
    line = _line(_caught(lambda: model.fit(X, y)), "'a:b' cell 'B' x 'Q'")
    assert "expected" not in line and "fitted" in line


@pytest.mark.parametrize(
    ("semantics", "offset", "counted"),
    [
        ("prior", True, "100 rows, exposure 200, every response 0; 20 expected"),
        ("frequency", True, "100 rows, exposure 2000, every response 0; 200 expected"),
        ("prior", False, "100 rows, every response 0; 20 expected"),
        ("frequency", False, "100 rows, weight 1000, every response 0; 200 expected"),
    ],
)
def test_totals_count_each_row_as_its_weights_say(semantics, offset, counted):
    """Weight 10 on every row, two years each: exposure and totals per physical row.

    Prior weights state a row's precision and count it once, frequency weights count it
    ten times; without an offset there is no exposure to state.  At 20f77dd6f every
    case multiplied by the weight: 2000 exposure and 200 expected under prior weights.
    """
    model, X, y = _two_levels(0.0, weight_semantics=semantics)
    options = {"sample_weight": np.full(200, 10.0)}
    if offset:
        options["offset"] = np.full(200, np.log(2.0))
    line = _line(_caught(lambda: model.fit(X, y, **options)), "'a' level 'B'")
    assert counted in line


def test_every_separated_level_is_listed():
    """16 claim-free levels, 16 lines: each names a price that is not an estimate.

    At 20f77dd6f the warning named 15 and said "... and 1 more".
    """
    X = pd.DataFrame({"a": np.repeat(["base", *[f"z{i:02d}" for i in range(16)]], 20)})
    y = np.where(X["a"] == "base", np.tile([1.0, 0.0, 0.0, 0.0, 0.0], 68), 0.0)
    model = SuperGLM(
        family="poisson", features={"a": Categorical(base="base")}, selection_penalty=0.0
    )
    (warning,) = _caught(lambda: model.fit(X, y))
    assert len(re.findall(r"^  'a' level 'z\d\d'", str(warning.message), re.M)) == 16
    assert "more in" not in str(warning.message)


@pytest.mark.pins_behaviour
def test_a_crossed_cells_remedy_names_no_cap():
    """A cap bounds a main effect's levels, not a crossed cell: the remedy does not offer one.

    Control: pins master's remedy, which named no cap.  At 20f77dd6f it said "bound it
    with Constraint.fit.cap", and capping both parents left the cell separated.
    """
    _, _, caught = _cells([("A", "P", 20), ("A", "Q", 20), ("B", "P", 20), ("B", "Q", 0)])
    (warning,) = caught
    remedy = str(warning.message).splitlines()[-1]
    assert "Constraint.fit.cap" not in remedy and "selection penalty" in remedy


def _forced_runtime_finding(monkeypatch, carried_by):
    """Every solve reports a runtime finding whose walk ``carried_by`` carries farthest.

    The backstop's terminal signature depends on the trajectory, so a test forces it.
    """
    import superglm.diagnostics.separation as separation
    from superglm.diagnostics.separation import RuntimeFinding

    monkeypatch.setattr(
        separation,
        "runtime_finding",
        lambda *args, **kwargs: RuntimeFinding(
            format_runtime_message(1e13, 3, list(carried_by), True), tuple(carried_by)
        ),
    )
    X, y, exposure = _book()
    return _of(_fit(_model(), X, y, exposure), SeparationWarning)


def test_the_runtime_backstop_does_not_restate_a_disclosed_level(monkeypatch):
    """A runtime finding carried by ``lev``, whose level E the scan recorded, is that walk.

    It is not warned a second time; master warned it beside the build's warning.
    """
    found = _forced_runtime_finding(monkeypatch, ["lev"])
    assert len(found) == 1 and "signature of separation" not in found[0]


@pytest.mark.pins_behaviour
def test_the_runtime_backstop_still_warns_a_walk_another_term_carries(monkeypatch):
    """Control: a runtime finding carried by the spline is not the level's, and is warned.

    Pins master's behaviour for a finding the scan's record does not explain: the
    separated level's disclosure and the backstop's statement, two warnings.
    """
    found = _forced_runtime_finding(monkeypatch, ["x"])
    assert len(found) == 2 and any("signature of separation" in text for text in found)


@pytest.mark.pins_behaviour
def test_the_runtime_backstop_warns_a_walk_it_shares_with_another_term(monkeypatch):
    """A finding naming ``lev`` and then the spline: the spline's part is not the level's.

    It is warned beside the level's disclosure.  Control: pins master's behaviour, which
    warned every finding.  At 20f77dd6f only the first group named was read, so the
    finding was dropped.
    """
    found = _forced_runtime_finding(monkeypatch, ["lev", "x"])
    assert len(found) == 2 and any("signature of separation" in text for text in found)


def test_ignore_says_nothing_about_a_separated_level():
    """``separation="ignore"`` silences REML's weak-identification line for the level too."""
    X, y, exposure = _book()
    model = _model(separation="ignore")
    caught = _fit(model, X, y, exposure, "fit_reml")
    assert not _of(caught, SeparationWarning)
    assert not [text for text in _of(caught, WeakIdentificationWarning) if "lev" in text]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(X[X["lev"] == "E"].head(3))
    assert not _of(caught, SeparationWarning)


# ── row 2: labels name levels and cells ──────────────────────────────────


def test_weak_identification_names_the_level_not_its_column():
    """Integer levels 0..4 with level 4 separated: REML's record names ``lev[4]``.

    Master labelled the column's position, ``lev[3]``, which names the healthy level 3.
    """
    X, y, exposure = _book(levels=(0, 1, 2, 3, 4))
    model = _model(lev=Categorical(base=0))
    _fit(model, X, y, exposure, "fit_reml")
    assert model._reml_profile["reml_weakly_identified_labels"] == ("lev[4]",)


def test_every_coefficient_label_names_its_level():
    """Integer levels 0..3: the unresolved-rows warning and ``diagnostics()`` name levels.

    At 20f77dd6f both labelled column positions, naming the healthy level 1 among
    ``A[2], B[2], B[1], A[1]``.
    """
    heavy = [(u, v) for u in (0, 1) for v in (0, 1)] + [(u, v) for u in (2, 3) for v in (2, 3)]
    rows = [(u, v, r, 1e8, 1.3) for u, v in heavy for r in (0.0, 1.0)]
    rows += [(u, v, r, 1e-8, -20.0) for u, v in [(1, 2), (1, 3)] for r in (0.0, 1.0)]
    frame = pd.DataFrame(rows, columns=["A", "B", "y", "w", "off"])
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve="gram",
        features={"A": Categorical(base=0), "B": Categorical(base=0)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            frame[["A", "B"]],
            frame["y"].to_numpy(),
            sample_weight=frame["w"].to_numpy(),
            offset=frame["off"].to_numpy(),
        )
    named = {"A[2]", "A[3]", "B[2]", "B[3]"}
    (record,) = model.diagnostics()["_model"]["unresolved_rows"]
    assert set(record["coefficients"]) == named
    (message,) = [m for m in _of(caught, WeakIdentificationWarning) if "rows" in m.lower()]
    assert all(label in message for label in named) and "A[1]" not in message


def _crossed():
    rng = np.random.default_rng(1)
    n = 6000
    a = rng.choice(list("ABC"), n, p=[0.5, 0.3, 0.2])
    b = rng.choice(list("PQ"), n, p=[0.8, 0.2])
    exposure = rng.uniform(0.2, 1.0, n)
    y = rng.poisson(exposure * 0.1).astype(float)
    y[(a == "C") & (b == "Q")] = 0.0
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A"), "b": Categorical(base="P")},
        interactions=[("a", "b")],
        selection_penalty=0.0,
    )
    X = pd.DataFrame({"a": a, "b": b})
    caught = _fit(model, X, y, exposure)
    return model, X, caught


def test_a_crossed_cell_is_labelled_by_its_levels():
    model, _, caught = _crossed()
    group = next(g for g in model._groups if g.feature_name == "a:b")
    specs = {**model._specs, **model._interaction_specs}
    column = model._interaction_specs["a:b"]._pairs.index(("C", "Q"))
    assert coefficient_labels(model._groups, [group.start + column], specs) == ("a:b[C×Q]",)
    (message,) = _of(caught, SeparationWarning)
    assert "'a:b' cell 'C' x 'Q'" in message and "base cell 'A' x 'P'" in message


# ── row 3: flagged, Wald statistics blank ─────────────────────────────────


def test_the_summary_blanks_the_wald_test_an_exact_test_contradicts():
    """0 claims in 25 years against 20 in 100: the exact two-rate test gives p = 0.0215.

    Conditional on the 20 claims, the level's share is binomial with p = 25 / 125, so
    ``binomtest(0, 20, 0.2)`` is the exact test (R's ``poisson.test``).  Master's Wald test
    read p = 0.968 beside a standard error in the tens.  The row keeps its estimate, is
    flagged ``SEP`` and shows no standard error, z, p or interval.
    """
    assert binomtest(0, 20, 0.2).pvalue == pytest.approx(0.0215, abs=5e-5)
    lev = np.repeat(["A", "E"], [100, 25])
    y = np.zeros(125)
    y[:20] = 1.0
    model = SuperGLM(family="poisson", features={"lev": Categorical(base="A")})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        model.fit(pd.DataFrame({"lev": lev}), y)
    row = next(r for r in model.summary()._coef_rows if r.name == "lev[E]")
    assert (row.se, row.z, row.p, row.ci_low, row.ci_high) == (None,) * 5
    assert row.coef < -5 and row.separated
    line = next(text for text in str(model.summary()).splitlines() if "lev[E]" in text)
    assert "SEP" in line and "0.968" not in line


def test_every_reader_flags_the_level_and_keeps_its_estimate(fitted):
    from superglm.export.summary import build_summary_export_payload

    _, _, _, models = fitted
    model, _ = models["fit_reml"]
    rows = {r.name: r for r in model.summary()._coef_rows}
    assert rows["lev[E]"].p is None and rows["lev[E]"].separated
    assert all(not rows[f"lev[{k}]"].separated and rows[f"lev[{k}]"].se > 0 for k in "BCD")

    ti = model.term_inference("lev")
    assert list(ti.separated) == [False] * 4 + [True]
    assert np.isnan(ti.se_log_relativity[-1]) and np.isnan(ti.ci_upper[-1])
    assert np.all(np.isfinite(ti.ci_upper[:-1])) and ti.relativity[-1] < 1e-3

    table = model.relativities(with_se=True)["lev"]
    assert table["separated"].tolist() == [False] * 4 + [True]
    assert np.isnan(table["se_log_relativity"].iloc[-1])

    payload = build_summary_export_payload(model)
    warning = {row.term: row.warning for row in payload.terms}
    assert warning["lev[E]"] == "Separated" and warning["lev[B]"] == ""
    assert next(t for t in payload.terms if t.term == "lev[E]").std_error is None


def test_a_separated_base_flags_every_contrast_of_its_term():
    X, y, exposure = _book()
    model = _model(lev=Categorical(base="E"))
    _fit(model, X, y, exposure)
    rows = [r for r in model.summary()._coef_rows if r.group == "lev"]
    assert rows and all(r.se is None and r.separated for r in rows)
    assert bool(np.all(model.term_inference("lev").separated))


def test_a_crossed_cell_row_is_flagged():
    model, _, _ = _crossed()
    rows = {r.name: r for r in model.summary()._coef_rows if r.group == "a:b"}
    assert rows["a:b[C:Q]"].p is None and rows["a:b[C:Q]"].separated
    assert not rows["a:b[B:Q]"].separated and rows["a:b[B:Q]"].p is not None


def _grid(cells):
    """``_cells`` without the warnings: (a, b, rows with a 1), 100 rows per cell."""
    model, X, _ = _cells(cells)
    return model, X, {row.name: row for row in model.summary()._coef_rows}


_SPARSE = [("A", "P", 20), ("A", "Q", 20), ("A", "R", 20)]


@pytest.mark.parametrize(
    ("cells", "blank", "kept"),
    [
        # B x P empty, B x R separated: a[B] and B:Q carry its walk (pruned B:R)
        (_SPARSE + [("B", "Q", 30), ("B", "R", 0)], {"a[B]", "a:b[B:Q]"}, {"b[Q]", "b[R]"}),
        # the base cell separated, B x P empty: B:Q is determined by the other four
        (
            [("A", "P", 0), *_SPARSE[1:], ("B", "Q", 30), ("B", "R", 40)],
            {"Intercept", "b[Q]", "b[R]"},
            {"a[B]", "a:b[B:Q]"},
        ),
        # full grid, the base-column cell B x P separated: the main effect a[B] moves
        (
            _SPARSE + [("B", "P", 0), ("B", "Q", 30), ("B", "R", 40)],
            {"a[B]", "a:b[B:Q]", "a:b[B:R]"},
            {"Intercept", "b[Q]", "b[R]"},
        ),
    ],
    ids=["sparse_cell", "sparse_base_cell", "base_column_cell"],
)
def test_a_crossed_cell_blanks_every_coefficient_it_moves(cells, blank, kept):
    """One rule for every grid: a coefficient the healthy rows leave undetermined is blank.

    At 20f77dd6f the four-corner rule left the sparse grid's B:Q unflagged (SE 156.8),
    blanked the healthy B:Q beside a separated base cell, and left a[B] unflagged beside
    a separated base-column cell (SE 142.9).
    """
    _, _, rows = _grid(cells)
    assert {name for name, row in rows.items() if row.separated} == blank
    assert all(rows[name].se is None and rows[name].p is None for name in blank)
    assert all(rows[name].se is not None and rows[name].se > 0 for name in kept)


@pytest.mark.pins_behaviour
def test_a_healthy_cell_beside_a_separated_base_cell_keeps_its_wald_test():
    """B:Q against its reference: log((30/40) / (20/20)), SE sqrt(1/30 + 1/40 + 1/20 + 1/20).

    Control: pins master's standard error for the healthy cell, which 20f77dd6f blanked.
    """
    _, _, rows = _grid([("A", "P", 0), *_SPARSE[1:], ("B", "Q", 30), ("B", "R", 40)])
    row = rows["a:b[B:Q]"]
    assert row.coef == pytest.approx(np.log(0.75), abs=1e-6)
    assert row.se == pytest.approx(np.sqrt(1 / 30 + 1 / 40 + 1 / 20 + 1 / 20), rel=1e-6)


def test_a_separated_base_cell_blanks_the_intercept():
    """A x P separated in a 2 x 2 grid: every coefficient moves, the intercept too.

    At 20f77dd6f the intercept read p = 0.993.
    """
    _, _, rows = _grid([("A", "P", 0), ("A", "Q", 40), ("B", "P", 20), ("B", "Q", 30)])
    assert all(row.separated and row.se is None for row in rows.values())


def test_the_interaction_readers_flag_a_separated_cell():
    """``term_inference`` and ``relativities(with_se=True)`` flag B:Q as the summary does.

    At 20f77dd6f both returned its factor, 1.7e-6, unflagged.
    """
    model, _, _ = _grid([("A", "P", 20), ("A", "Q", 20), ("B", "P", 20), ("B", "Q", 0)])
    assert model.term_inference("a:b").separated.tolist() == [True]
    assert model.relativities(with_se=True)["a:b"]["separated"].tolist() == [True]


def test_a_reader_factors_each_block_once(monkeypatch):
    """B x P separated in a 2 x 2 grid: each reader call factors the cross's block once.

    A plain level's block is read from its labels and factors nothing.  At 46bcd5990
    ``relativities()`` factored the same block five times, a centred ``term_inference``
    four times, and the plain level's block once per reader.
    """
    import superglm.diagnostics.separation as separation

    calls = []
    factor = separation._rank_and_null_space
    monkeypatch.setattr(
        separation, "_rank_and_null_space", lambda design: calls.append(1) or factor(design)
    )
    model, _, _ = _grid([("A", "P", 20), ("A", "Q", 20), ("B", "P", 0), ("B", "Q", 30)])
    for read in (
        lambda: model.relativities(with_se=False),
        lambda: model.relativities(with_se=True, centering="mean"),
        lambda: model.term_inference("a", centering="mean"),
    ):
        calls.clear()
        read()
        assert len(calls) == 1
    plain, X, y = _two_levels(0.0)
    _caught(lambda: plain.fit(X, y))
    calls.clear()
    plain.relativities(with_se=True, centering="mean")
    plain.summary()
    assert not calls


def test_a_healthy_factor_beside_a_separated_one_costs_its_readers_nothing_more():
    """A healthy 400-level factor reads the same memory whether or not ``small`` is recorded.

    Its readers skip it: no record reaches its columns.  At 46bcd5990 every categorical
    term's reader built a levels x coefficients contrast matrix, 1.3 MB here (Codex: 22
    MiB at 1,200 levels).
    """
    import tracemalloc

    rng = np.random.default_rng(3)
    n = 8000
    small = np.where(np.arange(n) % 40 == 0, "B", "A")
    X = pd.DataFrame({"big": rng.integers(0, 400, n).astype(str), "small": small})
    y = rng.poisson(2.0, n).astype(float)  # every level of big has claims
    y[small == "B"] = 0.0
    model = SuperGLM(
        family="poisson",
        features={"big": Categorical(), "small": Categorical(base="A")},
        selection_penalty=0.0,
    )
    _caught(lambda: model.fit(X, y))
    spec = model._specs["small"]
    held = spec._separated_levels_at_fit
    assert held == ("B",) and not model._specs["big"]._separated_levels_at_fit

    def peak() -> int:
        model.relativities(with_se=False)  # imports and lazy state, outside the count
        model.term_inference("big", with_se=False)
        tracemalloc.start()
        model.relativities(with_se=False)
        model.term_inference("big", with_se=False)
        top = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return top

    recorded = peak()
    spec._separated_levels_at_fit = ()
    unrecorded = peak()
    spec._separated_levels_at_fit = held
    assert recorded <= unrecorded + 64 * 1024


def test_mean_centering_leaves_out_a_level_a_separated_cell_moves():
    """C x P separated, C x Q with claims: a[C] carries the cell's walk, so it leaves the centre.

    A and B are centred on their own mean, each with half a[B]'s standard error.  At
    20f77dd6f a[C] was unflagged and the centre ran through its walk: A 24.3, B 25.0.
    """
    model, _, rows = _grid(
        [
            ("A", "P", 10),
            ("A", "Q", 20),
            ("B", "P", 15),
            ("B", "Q", 25),
            ("C", "P", 0),
            ("C", "Q", 5),
        ]
    )
    assert rows["a[C]"].separated and not rows["a[B]"].separated
    native = model.term_inference("a").log_relativity
    centred = model.term_inference("a", centering="mean")
    expected = native[:2] - native[:2].mean()
    np.testing.assert_allclose(centred.log_relativity[:2], expected, rtol=0, atol=1e-12)
    assert centred.separated.tolist() == [False, False, True]
    se = centred.se_log_relativity
    np.testing.assert_allclose(se[:2], 0.5 * rows["a[B]"].se, rtol=1e-9)
    assert np.isnan(se[2])


def test_a_separated_base_blanks_the_intercept():
    """Base E separated: the intercept is E's log rate, walked with it.

    At 20f77dd6f the intercept read SE 4e4 and p = 0.9995 above the blank rows.
    """
    X, y, exposure = _book()
    model = _model(lev=Categorical(base="E"))
    _fit(model, X, y, exposure)
    intercept = model.summary()._coef_rows[0]
    assert intercept.name == "Intercept" and intercept.separated and intercept.se is None


def test_a_thin_separated_level_shows_no_low_credibility_note():
    """Nine claim-free rows: the row reads SEP, and no LC note speaks of its blank interval.

    At 20f77dd6f the console printed the LC note and footnote beside the blank row, and
    the editor showed the low-credibility marker rather than the flag.
    """
    from superglm.editor.summaries import _compact_summary_row

    X, y, exposure = _book(n=2000, seed=1)
    model = _model()
    _fit(model, X, y, exposure)
    summary = model.summary()
    row = next(r for r in summary._coef_rows if r.name == "lev[E]")
    assert row.separated and not row.quasi_separated
    assert "LC: low credibility" not in str(summary) and "Low-credibility" not in str(summary)
    compact = _compact_summary_row(row)
    assert (compact["advisory_code"], compact["advisory_kind"]) == ("SEP", "separated")


def test_the_grouped_display_carries_the_flag():
    """Collapsing a grouping keeps one flag per displayed group.

    At 20f77dd6f ``to_dataframe()`` raised: four levels against five flags.
    """
    from superglm.plotting.group_display import project_grouped_term_for_display

    X, y, exposure = _book()
    y = np.where(X["lev"] == "D", 0.0, y)
    spec = Categorical(grouping=collapse_levels(X["lev"], groups={"D+E": ["D", "E"]}))
    model = _model(lev=spec)
    _fit(model, X, y, exposure)
    shown = project_grouped_term_for_display(model, model.term_inference("lev"), mode="collapsed")
    assert shown.term.to_dataframe()["separated"].tolist() == [False, False, False, True]


def test_the_editor_draws_every_healthy_level_interval(fitted):
    """Only the separated level's whisker is left out; A..D keep theirs.

    At 20f77dd6f the editor drew no interval for any level of the term.
    """
    from superglm.editor.payloads import session_payload
    from superglm.editor.session import EditorSession

    _, _, _, models = fitted
    model, _ = models["fit"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        payload = session_payload(EditorSession.from_model(model))["lev"]
    lower, upper = payload["ci_lower_y"], payload["ci_upper_y"]
    assert lower[-1] is None and upper[-1] is None
    assert all(lo is not None and lo <= hi for lo, hi in zip(lower[:-1], upper[:-1], strict=True))


def _binomial_reml(cells_zero, mode, crossed=False, seed=15):
    """Binomial/log REML beside a random effect, the rows of ``cells_zero`` all 0."""
    rng = np.random.default_rng(seed)
    n = 3000
    a = rng.choice(["A", "B", "C"], n, p=[0.4, 0.3, 0.3])
    b = rng.choice(["P", "Q"], n, p=[0.6, 0.4])
    y = (rng.uniform(size=n) < 0.15 + 0.05 * (a == "B") + 0.05 * (b == "Q")).astype(float)
    y[cells_zero(a, b)] = 0.0
    X = pd.DataFrame({"a": a, "b": b, "g": [f"g{i % 25}" for i in range(n)]})
    features = {"a": Categorical(base="A"), "b": Categorical(base="P"), "g": RandomEffect()}
    model = SuperGLM(
        family="binomial",
        link="log",
        features=features,
        interactions=[("a", "b")] if crossed else [],
        selection_penalty=0.0,
        separation=mode,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(X, y)
    return _of(caught, WeakIdentificationWarning), _of(caught, SeparationWarning)


@pytest.mark.parametrize("mode", ["warn", "ignore"])
def test_reml_adds_nothing_for_a_separated_base(mode):
    """The base level A has no event: its notice is the separation warning's, or none.

    At 20f77dd6f ``ignore`` still gave REML's weak-identification line for ``a[A]``.
    """
    weak, separated = _binomial_reml(lambda a, b: a == "A", mode)
    assert not weak and len(separated) == (mode == "warn")


@pytest.mark.parametrize("mode", ["warn", "ignore"])
def test_reml_adds_nothing_for_a_separated_base_row_cell(mode):
    """A x Q, in the base row: its walk is the parent b's, and no second notice names it.

    At 20f77dd6f both modes gave a weak-identification line beside it.
    """
    weak, separated = _binomial_reml(lambda a, b: (a == "A") & (b == "Q"), mode, crossed=True)
    assert not weak and len(separated) == (mode == "warn")


# ── row 4: mean centering leaves the separated level out of the centre ────


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_mean_centering_centres_on_the_levels_that_are_not_separated(fitted, method):
    """Master's mean ran through E's walked effect: A..D read about 840-1260, SE about 4e6."""
    _, _, _, models = fitted
    model, _ = models[method]
    native = model.term_inference("lev").log_relativity
    centred = model.term_inference("lev", centering="mean")
    expected = native - native[:4].mean()
    np.testing.assert_allclose(centred.log_relativity, expected, rtol=0, atol=1e-12)
    assert np.all((centred.relativity[:4] > 0.5) & (centred.relativity[:4] < 2.0))
    assert np.all(centred.se_log_relativity[:4] < 0.2)
    assert np.isnan(centred.se_log_relativity[-1]) and centred.separated[-1]
    table = model.relativities(with_se=True, centering="mean")["lev"]
    np.testing.assert_allclose(table["log_relativity"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        table["se_log_relativity"].iloc[:4], centred.se_log_relativity[:4], rtol=1e-12
    )


# ── row 6: the rating table says which factor is not an estimate ──────────


def test_the_rating_table_block_names_its_separated_level(fitted, tmp_path):
    from openpyxl import load_workbook

    X, y, exposure, models = fitted
    model, _ = models["fit"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        payload = model.rating_table_payload(X, y, offset=np.log(exposure))
        path = tmp_path / "rates.xlsx"
        model.export_rating_tables(path, X, y, offset=np.log(exposure))
    block = next(b for b in payload.main_effects if b.name == "lev")
    assert block.note is not None and "Separated levels: E." in block.note
    factor = block.table.set_index("lev")["Relativity"]["E"]
    assert factor == pytest.approx(model.term_inference("lev").relativity[-1], rel=1e-12)
    cells = [c.value for row in load_workbook(path).worksheets[0].iter_rows() for c in row]
    assert any(isinstance(v, str) and "Separated levels: E." in v for v in cells)


# ── row 7: predict warns once, for levels and cells ───────────────────────


def test_predict_warns_once_for_a_separated_level_and_keeps_the_price(fitted):
    X, _, _, models = fitted
    model, _ = models["fit"]
    rows = X[X["lev"] == "E"].head(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price = model.predict(rows)
    (message,) = _of(caught, SeparationWarning)
    assert "'lev' levels E" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = model.predict(rows.assign(lev="A"))
    rel = model.term_inference("lev").relativity[-1]
    np.testing.assert_allclose(price, quiet * rel, rtol=1e-10)


def test_predict_warns_for_a_separated_cell():
    model, X, _ = _crossed()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pd.DataFrame({"a": ["C", "B"], "b": ["Q", "Q"]}))
    (message,) = _of(caught, SeparationWarning)
    assert "'a:b' cells C x Q" in message
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pd.DataFrame({"a": ["B"], "b": ["Q"]}))
    assert not _of(caught, SeparationWarning)


_CROSS = [("A", "P", 20), ("A", "Q", 0), ("B", "P", 20), ("B", "Q", 20)]


@pytest.mark.parametrize("routed", ["A", "new"])
def test_predict_warns_for_a_label_routed_to_a_separated_cell(routed):
    """``unseen="base"`` scores "new" at A, so new x Q is the separated A x Q: one warning.

    Codex's case. At 20f77dd6f the routed row was priced at A x Q without a word.
    """
    data = [(a, b, float(i < k)) for a, b, k in _CROSS for i in range(100)]
    frame = pd.DataFrame(data, columns=["a", "b", "y"])
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A", unseen="base"), "b": Categorical(base="P")},
        interactions=[("a", "b")],
        selection_penalty=0.0,
    )
    _caught(lambda: model.fit(frame[["a", "b"]], frame["y"].to_numpy()))
    (warning,) = _caught(lambda: model.predict(pd.DataFrame({"a": [routed], "b": ["Q"]})))
    assert "'a:b' cells A x Q" in str(warning.message)


def test_predict_warns_for_a_label_routed_to_a_separated_base():
    """``unseen="base"`` with base E separated: an unseen label is scored at E, and said so.

    At 20f77dd6f the row was priced at E's 2e-5 without a separation warning.
    """
    X, y, exposure = _book()
    model = _model(lev=Categorical(base="E", unseen="base"))
    _fit(model, X, y, exposure)
    (warning,) = _caught(lambda: model.predict(pd.DataFrame({"lev": ["Z"], "x": [0.5]})))
    assert "'lev' levels E" in str(warning.message)


# ── row 8: a level a cap bounds is not separated ─────────────────────────


@pytest.mark.parametrize("mode", ["warn", "error"])
def test_a_capped_level_is_not_reported_as_separated(fitted, mode):
    """The cap holds E's relativity at 0.5, a finite optimum: no warning, refusal or flag."""
    X, y, exposure, _ = fitted
    model = _model(lev=Categorical(constraint=Constraint.fit.cap(0.5, 2.0)), separation=mode)
    caught = _fit(model, X, y, exposure)
    assert not _of(caught, SeparationWarning)
    assert model.term_inference("lev").relativity[-1] == pytest.approx(0.5, rel=1e-6)
    assert not any(r.separated for r in model.summary()._coef_rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(X[X["lev"] == "E"].head(2))
    assert not _of(caught, SeparationWarning)


def _capped_path_model(mode):
    model = SuperGLM(
        family="poisson",
        features={
            "a": Categorical(base="A", constraint=Constraint.fit.cap(0.5, 2.0)),
            "x": Numeric(),
        },
        penalty=GroupLasso(lambda1=0.1, features=["x"]),
        separation=mode,
    )
    X = pd.DataFrame({"a": np.repeat(["A", "B"], 100), "x": np.tile([0.0, 1.0], 100)})
    y = np.r_[1.0 + X["x"].to_numpy()[:100], np.zeros(100)]
    return model, X, y


def test_fit_path_scans_a_capped_level_it_does_not_cap():
    """``fit_path`` fits without its rules, so a cap bounds nothing there: B is separated.

    At 20f77dd6f the cap exempted B from the scan on the path too: no warning, and
    ``"error"`` returned B's factor at 2e-28.
    """
    model, X, y = _capped_path_model("warn")
    (warning,) = _caught(lambda: model.fit_path(X, y, lambda_seq=[0.1, 0.01]))
    assert "'a' level 'B'" in str(warning.message)
    model, X, y = _capped_path_model("error")
    with warnings.catch_warnings(), pytest.raises(SeparationError, match="'a' level 'B'"):
        warnings.simplefilter("ignore")  # the path's notice that it ignores the cap
        model.fit_path(X, y, lambda_seq=[0.1, 0.01])


def test_reml_says_nothing_of_a_capped_level_beside_a_random_effect(fitted):
    """The cap bounds E, so the random-effect variance warning does not name it either.

    At 20f77dd6f that warning still called E a claim-free level that separates.
    """
    X, y, exposure, _ = fitted
    model = SuperGLM(
        family="poisson",
        features={"lev": Categorical(constraint=Constraint.fit.cap(0.5, 2.0)), "g": RandomEffect()},
        selection_penalty=0.0,
    )
    X = X.assign(g=[f"g{i % 25}" for i in range(len(X))])
    assert not _of(_fit(model, X, y, exposure, "fit_reml"), SeparationWarning)


def test_a_saved_model_drops_a_capped_levels_old_record():
    """A model saved by a build that recorded a capped level: restored, the level is priced.

    The cap held B at 0.5, a finite optimum.  At 20f77dd6f the restored record blanked
    B's standard error and ``predict`` warned of it.
    """
    import pickle

    _, X, y = _two_levels(0.0)
    model = SuperGLM(
        family="poisson",
        features={"a": Categorical(base="A", constraint=Constraint.fit.cap(0.5, 2.0))},
        selection_penalty=0.0,
    )
    model.fit(X, y)
    model._specs["a"]._separated_levels_at_fit = ("B",)  # what master's build recorded
    restored = pickle.loads(pickle.dumps(model))
    inference = restored.term_inference("a")
    assert inference.separated is None and np.isfinite(inference.se_log_relativity[-1])
    assert not _caught(lambda: restored.predict(X.tail(1)))


def test_a_saved_crossed_record_without_occupied_cells_keeps_its_figures():
    """A model saved before the build kept occupied cells: its crossed record flags nothing.

    Its readers keep the figures they had until it is refitted (a known limit).  At
    46bcd5990 such a record read every cell as occupied and flagged a[B] and B:Q beside
    the base-column cell B x P.
    """
    import pickle

    model, _, _ = _grid([*_SPARSE, ("B", "P", 0), ("B", "Q", 30), ("B", "R", 40)])
    model._interaction_specs["a:b"]._occupied_cells_at_fit = None  # what master saved
    restored = pickle.loads(pickle.dumps(model))
    rows = restored.summary()._coef_rows
    assert not any(row.separated for row in rows) and all(row.se is not None for row in rows)
    assert restored.term_inference("a:b").separated is None


def test_a_saved_summary_of_a_separated_level_is_formed_again(fitted):
    """A summary cached before its readers flagged E is dropped when the model is restored.

    At 20f77dd6f the restored model served the stale summary: E's SE 2e7, p 1.0, unflagged.
    """
    import pickle

    _, _, _, models = fitted
    model = pickle.loads(pickle.dumps(models["fit_reml"][0]))
    summary = model.summary()
    stale = next(r for r in summary._coef_rows if r.name == "lev[E]")
    stale.separated, stale.se, stale.p = False, 2.09e7, 1.0  # what an older summary held
    restored = pickle.loads(pickle.dumps(model))
    row = next(r for r in restored.summary()._coef_rows if r.name == "lev[E]")
    assert row.separated and row.se is None and row.p is None


# ── row 9: the grouping remedy is a call that works ─────────────────────


def test_the_grouping_call_names_a_level_the_data_hold():
    """A declared level with no rows sits beside E: the call pairs E with an observed level.

    At 20f77dd6f it named the empty level, and ``collapse_levels`` refused the call.
    """
    X, y, exposure = _book()
    model = _model(lev=Categorical(levels=["A", "B", "C", "D", "D_empty", "E"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the declared level is pinned
        (message,) = _of(_fit(model, X, y, exposure), SeparationWarning)
    pattern = r"Categorical\(grouping=collapse_levels\(X\[[^\]]*\], groups=\{[^}]*\}\)\)"
    call = re.search(pattern, message).group(0)
    assert "D_empty" not in call
    spec = eval(call, {"Categorical": Categorical, "collapse_levels": collapse_levels, "X": X})
    assert spec._grouping.original_to_group["E"] == spec._grouping.original_to_group["D"]


def test_a_dict_grouping_is_refused_by_name():
    with pytest.raises(TypeError, match="grouping= takes a LevelGrouping.*collapse_levels"):
        Categorical(grouping={"E": "D"})


def test_the_warnings_grouping_call_prices_the_level(fitted):
    X, y, exposure, models = fitted
    _, caught = models["fit"]
    (message,) = _of(caught, SeparationWarning)
    pattern = r"Categorical\(grouping=collapse_levels\(X\[[^\]]*\], groups=\{[^}]*\}\)\)"
    call = re.search(pattern, message).group(0)
    spec = eval(call, {"Categorical": Categorical, "collapse_levels": collapse_levels, "X": X})
    model = _model(lev=spec)
    caught = _fit(model, X, y, exposure)
    assert not _of(caught, SeparationWarning)
    assert spec._grouping.original_to_group["E"] == spec._grouping.original_to_group["D"]
