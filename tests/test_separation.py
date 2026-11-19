"""Separation detection: build-time cell scan and in-solver backstop.

Issues #340 (name separated interaction cells at build time, before any IRLS
iteration) and #341 (refuse, or at minimum loudly flag, instead of returning
finite garbage on a separated design).
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Numeric,
    SeparationError,
    SeparationWarning,
    Spline,
    SuperGLM,
)
from superglm.distributions import Tweedie

# ── Shared designs ───────────────────────────────────────────────


def _tweedie_response(rng, n, rate=0.4):
    npos = rng.poisson(rate, size=n)
    return np.where(npos > 0, rng.gamma(npos * 2.0 + 1e-12, 0.6), 0.0)


def separated_crossed_design(seed=42, n=4000):
    """Crossed categorical design with two cells forced to zero response."""
    rng = np.random.default_rng(seed)
    a = rng.choice([f"a{i}" for i in range(6)], size=n)
    b = rng.choice([f"b{i}" for i in range(5)], size=n)
    w = rng.uniform(0.3, 1.5, size=n)
    y = _tweedie_response(rng, n)
    sep_mask = ((a == "a3") & (b == "b2")) | ((a == "a4") & (b == "b4"))
    assert sep_mask.sum() > 50, "separated cells must carry real exposure"
    y = np.where(sep_mask, 0.0, y)
    # Keep every other occupied cell anchored by at least one positive claim.
    df = pd.DataFrame({"a": a, "b": b})
    cell_max = pd.Series(y).groupby([df["a"], df["b"]]).transform("max").to_numpy()
    thin = (~sep_mask) & (cell_max == 0.0)
    y = np.where(thin, 0.7, y)
    return df, y, w, sep_mask


def crossed_model(**kwargs):
    defaults = dict(
        family=Tweedie(p=1.5),
        features={"a": Categorical(), "b": Categorical()},
        interactions=[("a", "b")],
        max_iter=60,
    )
    defaults.update(kwargs)
    return SuperGLM(**defaults)


def fit_catching(model, df, y, w):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w)
    return [c for c in caught if issubclass(c.category, SeparationWarning)]


# ── #340: build-time detection ───────────────────────────────────


def test_build_warning_names_separated_cells():
    df, y, w, _ = separated_crossed_design()
    model = crossed_model()
    sep = fit_catching(model, df, y, w)
    assert len(sep) == 1
    message = str(sep[0].message)
    assert "'a:b'" in message
    assert "('a3' x 'b2')" in message
    assert "('a4' x 'b4')" in message
    # The remedy is stated, per the issue.
    assert "collapse_levels" in message
    assert "RandomEffect" in message


def test_build_warning_fires_before_any_irls_iteration():
    df, y, w, _ = separated_crossed_design()
    model = crossed_model(separation="error")
    with pytest.raises(SeparationError, match=r"\('a3' x 'b2'\)"):
        model.fit(df, y, sample_weight=w)
    # Refusal happened at build: no solver result was produced.
    assert model._result is None


def test_discrete_path_detects_the_same_cells():
    df, y, w, _ = separated_crossed_design()
    rng = np.random.default_rng(3)
    df = df.assign(x=rng.uniform(0.0, 1.0, size=len(df)))
    model = SuperGLM(
        family=Tweedie(p=1.5),
        features={"a": Categorical(), "b": Categorical(), "x": Spline(n_knots=6)},
        interactions=[("a", "b")],
        discrete=True,
        max_iter=60,
    )
    sep = fit_catching(model, df, y, w)
    assert len(sep) == 1
    assert "('a3' x 'b2')" in str(sep[0].message)
    with pytest.raises(SeparationError):
        SuperGLM(
            family=Tweedie(p=1.5),
            features={"a": Categorical(), "b": Categorical(), "x": Spline(n_knots=6)},
            interactions=[("a", "b")],
            discrete=True,
            separation="error",
        ).fit(df, y, sample_weight=w)


def test_thin_cells_with_any_positive_response_do_not_fire():
    """Separation means zero positive response, not 'few rows'."""
    rng = np.random.default_rng(11)
    n = 4000
    a = rng.choice([f"a{i}" for i in range(6)], size=n)
    b = rng.choice([f"b{i}" for i in range(5)], size=n)
    w = rng.uniform(0.3, 1.5, size=n)
    y = _tweedie_response(rng, n)
    df = pd.DataFrame({"a": a, "b": b})
    # Give EVERY occupied cell at least one positive claim, however thin.
    first_row_of_cell = ~df.duplicated(subset=["a", "b"])
    y = np.where(first_row_of_cell & (y == 0.0), 0.05, y)
    model = crossed_model()
    sep = fit_catching(model, df, y, w)
    assert sep == []


def test_main_effect_level_detected():
    rng = np.random.default_rng(5)
    n = 2500
    c = rng.choice([f"c{i}" for i in range(8)], size=n)
    w = np.ones(n)
    y = _tweedie_response(rng, n)
    y = np.where(c == "c6", 0.0, np.where(y == 0.0, 0.4, y))
    df = pd.DataFrame({"c": c})
    model = SuperGLM(family=Tweedie(p=1.5), features={"c": Categorical()}, max_iter=60)
    sep = fit_catching(model, df, y, w)
    assert len(sep) == 1
    message = str(sep[0].message)
    assert "'c'" in message
    assert "'c6'" in message


@pytest.mark.pins_behaviour
def test_predict_names_a_separated_categorical_level_it_meets():
    """``predict`` names a main-effect level that had only zero responses at fit (sz_explosion round 2).

    The fit warns at build; the level has no finite effect, and its
    predictions sit near zero wherever the walk stopped.  ``predict`` said
    nothing.  It now names the level when its rows reach it, at the caller's
    line, and stays silent on rows of other levels.  The caller's line is a
    control: master's fixed stack level already found it, and it pins that for
    the deeper prediction path that reads the raw predictor (``warn_caller``).
    Mutation: no record of the level; then the fixed stack level, one frame
    too deep on that path.
    """
    rng = np.random.default_rng(5)
    n = 2500
    c = rng.choice([f"c{i}" for i in range(8)], size=n)
    w = np.ones(n)
    y = _tweedie_response(rng, n)
    y = np.where(c == "c6", 0.0, np.where(y == 0.0, 0.4, y))
    model = SuperGLM(family=Tweedie(p=1.5), features={"c": Categorical()}, max_iter=60)
    fit_catching(model, pd.DataFrame({"c": c}), y, w)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pd.DataFrame({"c": ["c6", "c1"]}))
    named = [x for x in caught if "only boundary responses at fit" in str(x.message)]
    assert len(named) == 1 and "'c' levels c6" in str(named[0].message)
    assert named[0].filename == __file__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pd.DataFrame({"c": ["c1", "c2"]}))
    assert not [x for x in caught if "only boundary responses at fit" in str(x.message)]


def test_predict_names_a_separated_grouped_level_by_its_members():
    """``predict`` names a collapsed level that separates when a member's raw value meets it (PR 508 round 1).

    ``Categorical(grouping=...)`` is the remedy the guide recommends for
    sparse levels, and the separation scan records the COLLAPSED label
    ("c67"), while ``predict`` reads the frame's raw values ("c6", "c7").  The
    check compared the two directly, so no row ever matched a group.  It now
    resolves the column as ``score`` does: raw to collapsed to fitted.
    Mutation: the raw values matched against the recorded labels.
    """
    from superglm import collapse_levels

    rng = np.random.default_rng(5)
    n = 2500
    c = rng.choice([f"c{i}" for i in range(8)], size=n)
    y = _tweedie_response(rng, n)
    y = np.where(np.isin(c, ["c6", "c7"]), 0.0, np.where(y == 0.0, 0.4, y))
    grouping = collapse_levels(sorted(set(c)), groups={"c67": ["c6", "c7"]})
    model = SuperGLM(
        family=Tweedie(p=1.5), features={"c": Categorical(grouping=grouping)}, max_iter=60
    )
    fit_catching(model, pd.DataFrame({"c": c}), y, np.ones(n))
    assert model._specs["c"]._separated_levels_at_fit == ("c67",)
    for member in ("c6", "c7"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.predict(pd.DataFrame({"c": [member, "c1"]}))
        named = [x for x in caught if "only boundary responses at fit" in str(x.message)]
        assert len(named) == 1 and "'c' levels c67" in str(named[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pd.DataFrame({"c": ["c1", "c2"]}))
    assert not [x for x in caught if "only boundary responses at fit" in str(x.message)]


@pytest.fixture
def program_sizes(monkeypatch):
    """``(stored entries, rows)`` of every linear program ``linprog`` is handed."""
    from scipy import optimize, sparse

    stored: list[tuple[int, int]] = []
    real = optimize.linprog

    def spy(c, *, A_ub, **kwargs):
        # the build-time scans' programs; the published fit's existence test is its own
        if sys._getframe(1).f_code.co_name != "separating_direction":
            stored.append((A_ub.nnz if sparse.issparse(A_ub) else A_ub.size, A_ub.shape[0]))
        return real(c, A_ub=A_ub, **kwargs)

    monkeypatch.setattr(optimize, "linprog", spy)
    return stored


def test_a_levels_recession_span_program_stores_a_linear_number_of_entries(program_sizes):
    """A level with many distinct x builds a sparse span program (PR 508 round 1).

    ``_recession_span`` joined ``-D`` (``n x q``) to ``np.eye(n)``, ``n`` the
    level's distinct boundary values: ``n (q + n)`` stored entries, about
    800 MB at 10,000 values of a continuous covariate.  The slack identity is
    sparse, ``n (q + 1)`` entries in all.  Mutation: the dense identity.
    """
    from superglm.diagnostics.separation import _level_recession

    n = 400
    x = np.linspace(0.0, 1.0, n)
    natural = np.column_stack([np.ones(n), x, x**2])
    lower = np.ones(n, dtype=bool)
    none = np.zeros(n, dtype=bool)
    found = _level_recession(natural, lower, none, none, np.eye(3))
    assert found is not None and found.shape == (3, 3)
    # a row holds its q = 3 coefficients and its own slack, at most
    assert program_sizes and all(entries <= rows * 4 for entries, rows in program_sizes)


def test_detecting_separated_varying_coefficient_levels_solves_no_span_program(program_sizes):
    """A Numeric x Categorical build reads only whether a level recedes (PR 508 round 1).

    ``separated_level_ranges`` keeps a level's rows' range and discards the
    span, but solved the span's program for every level whose curve recedes,
    zero-claim levels included.  Here a level with 300 distinct x and no claim
    is recorded without one; the programs left store at most the ``q = 2``
    coefficients per row.  Mutation: the span computed.
    """
    rng = np.random.default_rng(2)
    m = 300
    g = np.concatenate([rng.choice(["a", "c"], 1500), np.full(m, "b")])
    x = np.concatenate([rng.uniform(size=1500), np.linspace(0.0, 1.0, m)])
    y = np.concatenate([rng.poisson(0.5, 1500).astype(float), np.zeros(m)])
    y[:1500][g[:1500] == "c"] += 1.0
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "g": Categorical(base="a")},
        interactions=[("x", "g")],
        selection_penalty=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"x": x, "g": g}), y)
    assert set(model._interaction_specs["x:g"]._separated_ranges) == {"b"}
    assert all(entries <= rows * 2 for entries, rows in program_sizes)


def test_gaussian_identity_never_fires():
    df, y, w, _ = separated_crossed_design()
    model = SuperGLM(
        family="gaussian",
        features={"a": Categorical(), "b": Categorical()},
        interactions=[("a", "b")],
        max_iter=60,
    )
    sep = fit_catching(model, df, y, w)
    assert sep == []


def test_selection_penalty_exempts_bounded_terms():
    df, y, w, _ = separated_crossed_design()
    model = crossed_model(selection_penalty=0.5)
    sep = fit_catching(model, df, y, w)
    assert sep == []


def test_ignore_mode_is_silent_and_fits():
    df, y, w, sep_mask = separated_crossed_design()
    model = crossed_model(separation="ignore")
    sep = fit_catching(model, df, y, w)
    assert sep == []
    # The unchecked fit still returns the collapsed predictions -- the
    # opt-out reproduces the old behaviour bit for bit.
    pred = model.predict(df)
    assert pred[sep_mask].max() < 1e-6


def test_invalid_mode_rejected_at_construction():
    with pytest.raises(ValueError, match="separation"):
        SuperGLM(separation="strict")


def test_clone_preserves_separation_mode():
    model = crossed_model(separation="error")
    clone = model.clone_unfitted()
    assert clone._separation == "error"


def test_an_unpenalised_refit_keeps_the_models_separation_mode():
    """``refit_unpenalised`` lifts the selection penalty that exempted a separated level.

    Under ``"error"`` the refit refuses the level, as a fit without the
    penalty does; under ``"warn"`` it names it and fits. The refit's model
    used to fall back to ``"warn"`` whatever the model declared.
    """
    rng = np.random.default_rng(4)
    n = 600
    df = pd.DataFrame(
        {"c": rng.choice(list("ABCZ"), n, p=[0.4, 0.3, 0.28, 0.02]), "x": rng.normal(size=n)}
    )
    y = rng.poisson(np.exp(0.3 + 0.2 * df["x"].to_numpy())).astype(np.float64)
    y[(df["c"] == "Z").to_numpy()] = 0.0

    def fitted(separation):
        return SuperGLM(
            family="poisson",
            selection_penalty=2.0,
            separation=separation,
            features={"c": Categorical(base="A"), "x": Numeric()},
        ).fit(df, y)

    strict = fitted("error")
    with pytest.raises(SeparationError, match="'Z'"):
        strict.refit_unpenalised(df, y)
    lenient = fitted("warn")
    with pytest.warns(SeparationWarning, match="'Z'"):
        lenient.refit_unpenalised(df, y)


# ── #341: runtime backstop for separation the build scan cannot see ──


def separated_numeric_design(seed=7, n=3000, share=0.25):
    rng = np.random.default_rng(seed)
    z = (rng.uniform(size=n) < share).astype(float)
    y = _tweedie_response(rng, n, rate=0.5)
    y = np.where(z == 1.0, 0.0, y)
    return pd.DataFrame({"z": z}), y, np.ones(n)


@pytest.mark.pins_behaviour
def test_runtime_backstop_warns_on_pinned_predictor(monkeypatch):
    """Control: where no existence decision is built, the backstop names the pinned group.

    As on master: the walk of a claim-free indicator stops at the log link's guard and the
    backstop's finding names ``z``.  A decided fit says so from its decision instead
    (``test_a_separated_indicator_warns_once_from_the_decision``).
    """
    force_decision_not_built(monkeypatch)
    df, y, w = separated_numeric_design()
    with warnings.catch_warnings():
        # convergence='coefficients' carries its own experimental warning.
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"z": Numeric()},
            max_iter=150,
            convergence="coefficients",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w)
    sep = [c for c in caught if issubclass(c.category, SeparationWarning)]
    assert len(sep) == 1
    message = str(sep[0].message)
    assert "signature of separation" in message
    assert "'z'" in message  # the drifting group is named


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("share", [0.25, 0.6])
def test_runtime_backstop_names_the_group_that_moves_the_stopped_rows(monkeypatch, share):
    """A coefficient's size is no guide across groups; what it moves on the stopped rows is.

    Control, on the route where no existence decision is built: the backstop's naming rule,
    as on master.

    ``x`` is a real effect on a column of scale 1e-3, so its coefficient
    (about 260) dwarfs the separated ``z`` (pinned near -80).  Naming the
    largest coefficient named ``x``; the rows the guard stopped are moved by
    ``z``.  On the freMTPL2 guide model the same rule named a tensor term for
    a brand's spline-by-factor curve.  At ``share=0.6`` the stopped rows are
    most of the rows, so a median over every row sits on them and ``z``
    moved nothing from it; the move is measured from the rows that did not
    stop.
    """
    force_decision_not_built(monkeypatch)
    df, y, w = bystander_design(share)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"z": Numeric(), "x": Numeric()},
            max_iter=150,
            convergence="coefficients",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w)
    beta = dict(zip(["z", "x"], model.result.beta, strict=True))
    assert abs(beta["x"]) > abs(beta["z"]) + 2.0, (
        "fixture must make the bystander's coefficient larger"
    )
    message = next(str(c.message) for c in caught if issubclass(c.category, SeparationWarning))
    assert "overflow guard" in message, "the regression needs the pinned branch"
    assert "group(s): ['z']" in message


def bystander_design(share):
    """The claim-free indicator ``z`` beside ``x``, a real effect on a column of scale 1e-3."""
    df, y, w = separated_numeric_design(share=share)
    rng = np.random.default_rng(11)
    df = df.assign(x=rng.normal(size=len(df)) * 1e-3)
    y = np.where(df["z"].to_numpy() == 1.0, 0.0, y * np.exp(300.0 * df["x"].to_numpy()))
    return df, y, w


def year_design():
    """The claim-free indicator ``z`` beside ``year``, a covariate near 2015."""
    df, y, w = separated_numeric_design()
    year = 2015.0 + np.random.default_rng(11).integers(-5, 6, size=len(df)).astype(float)
    df = df.assign(year=year)
    y = np.where(df["z"].to_numpy() == 1.0, 0.0, y * np.exp(0.1 * (year - 2015.0)))
    return df, y, w


@pytest.mark.pins_behaviour
def test_runtime_backstop_measures_a_group_from_its_bulk_not_from_zero(monkeypatch):
    """A covariate far from zero carries a large constant the intercept cancels.

    ``year`` sits near 2015 with a coefficient of 0.1, so its raw contribution
    is about 212 on every row, beyond the separated ``z`` (pinned near -80).
    What it moves the stopped rows away from the bulk is under 1.  Control, on the
    route where no existence decision is built: the backstop's rule, as on master.
    """
    force_decision_not_built(monkeypatch)
    df, y, w = year_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"z": Numeric(), "year": Numeric()},
            max_iter=150,
            convergence="coefficients",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w)
    beta = dict(zip(["z", "year"], model.result.beta, strict=True))
    assert abs(beta["year"]) * 2010 > abs(beta["z"]), "fixture must make the raw offset larger"
    message = next(str(c.message) for c in caught if issubclass(c.category, SeparationWarning))
    assert "group(s): ['z']" in message


@pytest.mark.pins_behaviour
def test_runtime_backstop_reports_the_expected_curvature_ratio(monkeypatch):
    """Tweedie/log rows are observed-Newton; the gate compares the Fisher ratio.

    The iteration log keeps the observed weights, which are the Fisher weights
    times (2 - p) + (p - 1) y / mu, so their ratio is not the one the warning reports.
    Control, on the route where no existence decision is built, as on master.
    """
    force_decision_not_built(monkeypatch)
    df, y, w = separated_numeric_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"z": Numeric()},
            max_iter=150,
            convergence="coefficients",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w, record_diagnostics=True)
    message = next(str(c.message) for c in caught if issubclass(c.category, SeparationWarning))
    observed = model.result.iteration_log[-1].w_ratio
    # The negative check reads the ratio the message reports, in this format.
    assert re.search(r"ratio \d\.\de[+-]\d+", message)
    assert f"ratio {observed:.1e}" not in message


def test_runtime_backstop_error_mode_refuses():
    """Under ``"error"`` the separated indicator is refused with its decision's statement.

    The rows' direction ``z`` is verified exactly, so the refusal names it and the rows it
    separates, at least: a refused fit runs no withheld fit to show the set maximal.  Master
    refused with the backstop's guard signature.  Mutation check: with the withheld fit run
    before the refusal, the statement was the maximal one, without "at least".
    """
    df, y, w = separated_numeric_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"z": Numeric()},
            max_iter=150,
            convergence="coefficients",
            separation="error",
        )
    with pytest.raises(SeparationError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(df, y, sample_weight=w)
    message = str(refused.value)
    assert message.startswith("Separation: at least ")
    assert says_infinite(message, "z")
    assert not refused.value.levels


@pytest.mark.parametrize("design", ["pinned", "share 0.25", "share 0.6", "year"])
def test_a_separated_indicator_warns_once_from_the_decision(design):
    """A separation the decision proves is said once, naming exactly the coefficients it moves.

    The claim-free indicator ``z`` walks to the log link's guard beside a bystander whose
    coefficient is larger (``x``, on a 1e-3 column) or whose raw contribution is (``year`` near
    2015): the backstop needed its naming rules to point at ``z``; the decision's span names it
    by its coordinates.  The fit returned is the fit on the rows ``z`` does not touch, its
    standard error withheld.  At ``07a1401a5`` none of these fits warned: the published fit was
    the withheld refit, which carries no finding.  Mutation check: with the statement read from
    the published solve's raw finding, nothing was said.
    """
    df, y, w = {
        "pinned": separated_numeric_design,
        "share 0.25": lambda: bystander_design(0.25),
        "share 0.6": lambda: bystander_design(0.6),
        "year": year_design,
    }[design]()
    features = {"z": Numeric(), **{name: Numeric() for name in df.columns if name != "z"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=Tweedie(p=1.5), features=features, max_iter=150, convergence="coefficients"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(df, y, sample_weight=w)

    claims = separation_warnings(caught)
    assert len(claims) == 1
    assert claims == decision_claims(caught)
    assert says_infinite(claims[0], "z")
    separation = published_decision(model)
    assert separation.state == "present" and separation.names == ("z",)
    statement = model.result.separation_statement
    assert (statement.tag, statement.source) == ("present", "decision")
    assert np.isnan(model.metrics(df, y, sample_weight=w).coefficient_se["z"][0])


def test_runtime_backstop_silent_on_healthy_fit():
    rng = np.random.default_rng(13)
    n = 2000
    z = rng.uniform(0.0, 1.0, size=n)
    mu = np.exp(0.2 + 0.5 * z)
    npos = rng.poisson(mu * 0.5)
    y = np.where(npos > 0, rng.gamma(npos * 2.0 + 1e-12, 0.6), 0.0)
    df = pd.DataFrame({"z": z})
    model = SuperGLM(family=Tweedie(p=1.5), features={"z": Numeric()}, max_iter=100)
    sep = fit_catching(model, df, y, np.ones(n))
    assert sep == []


def test_runtime_backstop_is_silent_when_the_budget_ends_mid_descent(monkeypatch):
    """Exhausting a budget is not evidence of separation on its own.

    The gate reads the LAST iteration's deviance movement.  That value has to
    be captured inside the loop, before ``dev_prev`` advances: read afterwards
    it is identically zero whenever the loop exhausts ``max_iter``, which makes
    the stagnation clause vacuously true and brands every extreme-weight fit
    that merely ran out of budget -- slow convergence, not separation.

    The weight threshold is lowered so the gate is reachable at all; the point
    under test is the stagnation clause, not the threshold.
    """
    import superglm.diagnostics.separation as separation_module

    monkeypatch.setattr(separation_module, "EXTREME_WEIGHT_RATIO", 1.0)

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 4.0, n)
    mu = np.exp(1.0 + 2.5 * x)
    y = np.where(rng.random(n) < 0.3, rng.gamma(2.0, np.clip(mu, 1e-8, 1e8) / 2.0), 0.0)

    # The intercept-only start ignores the offset, so every row starts 30 log
    # units below its mode. A full-Newton Tweedie/log working response rises
    # less than 1/(p - 1) = 2 above eta, so the climb takes about twenty
    # bounded steps and a budget at the gate's own floor of ten ends in it.
    offset = np.full(n, -30.0)
    max_iter = 10
    model = SuperGLM(
        family=Tweedie(p=1.5),
        link="log",
        features={"x": Numeric()},
        max_iter=max_iter,
        tol=1e-14,
        separation="warn",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(pd.DataFrame({"x": x}), y, offset=offset, record_diagnostics=True)

    # Non-vacuity: every other input of the exhausted-and-stagnant clause
    # holds, so the deviance movement alone decides it.
    log = model.result.iteration_log
    assert len(log) == max_iter
    assert log[-1].termination_reason == "max_iter"
    assert log[-1].w_ratio > separation_module.EXTREME_WEIGHT_RATIO
    movement = abs(log[-1].deviance - log[-2].deviance) / (abs(log[-2].deviance) + 1)
    assert movement > separation_module.STAGNANT_DEVIANCE_DELTA

    exhaustion = [
        str(w.message)
        for w in caught
        if issubclass(w.category, separation_module.SeparationWarning)
        and "budget" in str(w.message)
    ]
    assert exhaustion == [], f"budget exhaustion alone must not read as separation: {exhaustion}"


def test_the_runtime_gate_reads_expected_curvature_under_observed_newton_rows():
    """Tweedie/log observed rows are Fisher rows times (2 - p) + (p - 1) y / mu.

    At equal means a response 1e13 times its mean spreads the observed weights
    past the 1e12 bar while the expected curvature, which the bar was set on,
    stays flat.
    """
    from superglm.links import LogLink
    from superglm.solvers.irls_direct import _separation_weight_ratio

    mu, eta, weights = np.ones(3), np.zeros(3), np.ones(3)
    common = {"family": Tweedie(p=1.5), "link": LogLink(), "mu": mu, "eta": eta}
    assert _separation_weight_ratio("observed", 1e13, weights=weights, **common) == 1.0
    assert _separation_weight_ratio("fisher", 1e13, weights=weights, **common) == 1e13


# ── The runtime warning describes the published fit (maths hunt F13) ──

#: The log link's overflow guard on the linear predictor (``links.stabilize_eta``).
LOG_GUARD = 80.0
U = np.finfo(np.float64).eps / 2


def runtime_claims(caught):
    return [
        str(c.message)
        for c in caught
        if issubclass(c.category, SeparationWarning) and "signature of separation" in str(c.message)
    ]


#: How the existence decision's statements open (``Separation.statement``).
DECISION_OPENINGS = ("Separation: ", "Separation not resolved", "Complete separation")


def decision_claims(caught):
    """The separation warnings the fit's existence decision gave."""
    return [
        str(c.message)
        for c in caught
        if issubclass(c.category, SeparationWarning)
        and str(c.message).startswith(DECISION_OPENINGS)
    ]


def says_infinite(message, names):
    """Whether a decision's statement names ``names`` as the coefficients with no finite estimate."""
    return f"moving {names}," in message and "no finite estimate" in message


def separation_warnings(caught):
    return [str(c.message) for c in caught if issubclass(c.category, SeparationWarning)]


def force_decision_not_built(monkeypatch):
    """Publication builds no existence decision, so the backstop's finding is the fit's statement.

    The route a selection penalty, a constrained or a SCOP fit takes: no response boundary is
    read (``separation_ops.separation_boundaries``).  Before the decision existed there was none
    to skip, so this patches nothing there.
    """
    import importlib
    import importlib.util

    if importlib.util.find_spec("superglm.model.separation_ops") is not None:
        module = importlib.import_module("superglm.model.separation_ops")
        monkeypatch.setattr(module, "separation_boundaries", lambda model: (), raising=False)


def published_decision(model):
    """The published fit's existence decision, from its identifiability record."""
    record = model._fit_inference_info.get("identifiability")
    return getattr(record, "separation", None)


def forced_runtime_finding():
    """The backstop's message for a pinned walk, as ``runtime_finding`` formats it."""
    from superglm.diagnostics.separation import format_runtime_message

    return format_runtime_message(1e35, 5, ["x"], True)


def force_runtime_finding(monkeypatch):
    """Every coefficient solve that runs the backstop records a finding, whatever its state.

    Both solvers read the one check (``separation.runtime_finding``), so this
    forces the event on every route without bending the data to reach it.  The
    existence decision is not built (``force_decision_not_built``): it would
    prove these estimates finite and overrule the forced finding.
    """
    import superglm.diagnostics.separation as separation_module

    message = forced_runtime_finding()
    monkeypatch.setattr(separation_module, "runtime_finding", lambda *_a, **_k: message)
    # these fits' estimates exist, so a decision would overrule the forced finding
    force_decision_not_built(monkeypatch)
    return message


def overdispersed_separated_curve(seed=0, n=2000):
    """NB2 claim counts on exposure; level C of the spline-by-factor term has no claim past x 22.

    The freMTPL2 witness's shape in small: levels A and B reach the far x
    values with claims, C's twelve far policies have none, so the build holds
    C's curve there and names it.
    """
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [
            np.minimum(rng.geometric(0.15, size=n) - 1, 22),
            rng.uniform(23, 50, size=60).round(),
            rng.uniform(30, 45, size=12).round(),
        ]
    ).astype(float)
    f = np.concatenate(
        [rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]), rng.choice(["A", "B"], 60)]
        + [["C"] * 12]
    )
    exposure = rng.uniform(0.3, 1.0, size=len(x))
    mean = exposure * np.exp(-1.0 + 0.03 * np.minimum(x, 22) + np.where(f == "B", 0.2, 0.0))
    counts = rng.poisson(mean * rng.gamma(1.0, 1.0, size=len(x))).astype(float)
    counts[n : n + 60] = np.maximum(counts[n : n + 60], 1.0)
    counts[-12:] = 0.0
    return pd.DataFrame({"x": x, "f": pd.Categorical(f)}), counts, np.log(exposure)


def test_a_search_solve_at_the_guard_does_not_warn_about_the_published_fit(monkeypatch):
    """The separation warning describes the fit returned, not a trial solve of its search.

    On the freMTPL2 NB2 witness (45,000 policies, ``theta="auto"``) the first
    REML candidate, at theta 8.65, stopped with seven rows pinned at the log
    link's guard and warned that those rows' predictions were unusable; the
    published fit, at theta 2.28, converged with their log means between
    -11.4 and -3.5, which an independent NB2 refit matched to 2e-8.  Here every
    solve of the smoothing search finds the backstop's signature (forced at the
    check, inside the solve), and the published refit is left to its own.  The
    theta calibration, the REML build and each theta/REML refit also built the
    design and named C's held curve, four notices on the witness.  Mutation
    checks: with the search's solves passing ``"warn"`` each of them warned;
    with the per-call scope removed the build named the curve more than once.
    """
    import superglm.diagnostics.separation as separation_module
    import superglm.reml.direct as reml_direct
    from superglm.distributions import NegativeBinomial

    X, counts, offset = overdispersed_separated_curve()
    real_finding, solve = separation_module.runtime_finding, reml_direct.fit_irls_direct
    message = forced_runtime_finding()
    in_trial, forced = [], []

    def finding(*args, **kwargs):
        if not in_trial:
            return real_finding(*args, **kwargs)
        forced.append(message)
        return message

    def trial(*args, **kwargs):
        in_trial.append(True)
        try:
            return solve(*args, **kwargs)
        finally:
            in_trial.pop()

    monkeypatch.setattr(separation_module, "runtime_finding", finding)
    monkeypatch.setattr(reml_direct, "fit_irls_direct", trial)
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        features={"x": Spline(n_knots=12), "f": Categorical()},
        interactions=[("x", "f")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(X, counts, offset=offset)

    notices = [str(c.message) for c in caught if issubclass(c.category, SeparationWarning)]
    assert len(notices) == 1
    assert notices[0].startswith("Separation in a spline-by-factor term")
    # The event happened: the search's solves found the signature.
    assert len(forced) > 1
    # What the runtime warning would claim, read from predict: no row is there.
    log_mean = np.log(model.predict(X, offset=offset))
    assert np.all(np.abs(log_mean) < LOG_GUARD / 2)
    assert model.result.separation_message is None


def test_a_separated_fit_warns_once_however_many_solves_its_theta_search_ran():
    """A separated fit says so once, after its theta search, from its decision.

    ``theta="auto"`` calibrates theta by refitting the mean at each trial
    theta, and each refit of this separated indicator walks to the guard as
    the published fit does.  The published fit's decision is the one warning,
    naming ``z``; the fit on the other rows holds the separated span where the
    walk stopped, so those rows' predictions sit at the guard, as the warning
    says, and nowhere else.  At ``07a1401a5`` it said nothing (the published
    withheld refit carried no finding).  Mutation check: with the calibration
    solves passing the model's mode, each warned (four warnings for one fit).
    """
    from superglm.distributions import NegativeBinomial

    rng = np.random.default_rng(7)
    n = 2000
    z = (rng.uniform(size=n) < 0.25).astype(float)
    x = rng.uniform(0.0, 1.0, n)
    mean = np.exp(0.3 + np.sin(3.0 * x))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mean)).astype(float)
    y[z == 1.0] = 0.0
    frame = pd.DataFrame({"z": z, "x": x})
    with warnings.catch_warnings():
        # convergence='coefficients' carries its own experimental warning.
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"z": Numeric(), "x": Numeric()},
            max_iter=150,
            convergence="coefficients",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, y)

    claims = separation_warnings(caught)
    assert len(claims) == 1
    assert claims == decision_claims(caught)
    assert says_infinite(claims[0], "z")
    log_mean = np.log(model.predict(frame))
    separated = z == 1.0
    np.testing.assert_allclose(log_mean[separated], -LOG_GUARD, rtol=0, atol=64 * U * LOG_GUARD)
    assert np.all(np.abs(log_mean[~separated]) < LOG_GUARD / 2)


def poisson_curve(seed=3, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.3 + np.sin(3.0 * x))).astype(float)
    return pd.DataFrame({"x": x}), y


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("route", ["fit_reml", "fit_path"])
@pytest.mark.parametrize("mode", ["warn", "ignore", "error"])
def test_a_fit_reads_the_models_separation_mode_for_what_it_returns(monkeypatch, route, mode):
    """``fit_reml`` and ``fit_path`` warn once, stay silent or refuse, as ``fit`` does.

    Control, on the route where no existence decision is built (a decision would prove these
    estimates finite and overrule the forced finding): the backstop's statement, as on master.

    The backstop's finding is forced at the check both solvers read: on every
    solve of ``fit_reml``, which publishes its terminal refit's finding, and
    on the first point of the path.  ``fit_path`` returns every point, so
    each point carries its own finding, the fitted model's result (the last
    point) describes only its own state, and the call warns once, naming the
    points that have one.  Under ``"error"`` the refusal publishes nothing.
    Mutation checks: with the search's solves passing ``"warn"``,
    ``fit_reml`` warned once per solve under every mode; with the BCD solver
    recording no finding, or the path disclosing its last point's finding
    only, the path warned nothing; with the path's statement written onto the
    last point's result, that result carried the first point's finding.
    """
    import superglm.diagnostics.separation as separation_module

    frame, y = poisson_curve()
    message = forced_runtime_finding()
    solves = []

    def finding(*_args, **_kwargs):
        solves.append(1)
        return message if route == "fit_reml" or len(solves) == 1 else None

    monkeypatch.setattr(separation_module, "runtime_finding", finding)
    force_decision_not_built(monkeypatch)
    features = {"x": Spline(kind="cr", k=8)}
    if route == "fit_reml":
        model = SuperGLM(family="poisson", features=features, separation=mode)

        def fit():
            return model.fit_reml(frame, y)
    else:
        model = SuperGLM(
            family="poisson", features=features, selection_penalty=1e-3, separation=mode
        )

        def fit():
            return model.fit_path(frame, y, n_lambda=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "error":
            with pytest.raises(SeparationError, match="overflow guard"):
                fit()
        else:
            returned = fit()

    claims = runtime_claims(caught)
    if mode == "error":
        assert claims == []
        with pytest.raises(RuntimeError, match="Not fitted"):
            model.result
        return
    assert len(claims) == (1 if mode == "warn" else 0)
    if route == "fit_reml":
        assert model.result.separation_message == message
        return
    assert returned.separation_path == (message, None, None, None)
    first = f"{returned.lambda_seq[0]:.4g}"
    assert all(
        claim.startswith(f"fit_path: 1 of 4 path points (lambda {first}) ") for claim in claims
    )
    assert model.result.separation_message is None


@pytest.mark.parametrize("mode", ["warn", "error"])
def test_estimate_p_warns_for_its_published_refit_not_for_each_power_it_tried(monkeypatch, mode):
    """A power-search candidate is ranked and discarded, so its fit says nothing.

    Every candidate's fit and the refit at the selected power record the
    backstop's finding (forced); the refit is the fit the caller receives, and
    it warns once, or under ``"error"`` is what refuses.  Mutation checks:
    publishing a candidate's finding warned once per power the search
    evaluated; refusing a strict candidate as it finished refused the first
    power tried.
    """
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    y = _tweedie_response(rng, n, rate=0.6) * np.exp(np.sin(3.0 * x))
    frame = pd.DataFrame({"x": x})
    force_runtime_finding(monkeypatch)
    model = SuperGLM(family=Tweedie(p=1.5), features={"x": Spline(kind="cr", k=8)}, separation=mode)
    if mode == "error":
        with pytest.raises(SeparationError) as refused:
            model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="reml")
        assert "_refit_selected" in {entry.name for entry in refused.traceback}
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="reml")

    assert len(result.evaluations) > 1
    assert len(runtime_claims(caught)) == 1


@pytest.mark.parametrize("route", ["fit", "fit_reml", "estimate_p", "estimate_theta"])
@pytest.mark.parametrize("mode", ["warn", "ignore", "error"])
def test_every_route_decides_a_row_at_the_guard_as_fit_does(route, mode):
    """A row only its offset holds at the guard is not separation, on every route and mode.

    One claim-free policy's offset sits 100 below the rest, so every solve,
    the published one included, holds that row's predictor at the log link's
    guard, and the backstop reports it although no coefficient runs off (a
    known false call of the guard heuristic: master warned, or under
    ``"error"`` refused).  The existence decision proves every estimate finite
    (no direction of the design separates that row), and a decided absence
    overrules the heuristic: each route fits and says nothing about separation,
    under every mode, and the row is still predicted at the guard.  Mutation
    check: with the heuristic's finding published beside a decided absence,
    every route warned or refused again.
    """
    from superglm.distributions import NegativeBinomial

    frame, y = poisson_curve()
    offset = np.zeros(len(y))
    offset[0], y[0] = -100.0, 0.0
    family = {"estimate_p": Tweedie(p=1.5), "estimate_theta": NegativeBinomial(theta=2.0)}
    model = SuperGLM(
        family=family.get(route, "poisson"),
        features={"x": Spline(kind="cr", k=8)},
        separation=mode,
    )
    fits = {
        "fit": lambda: model.fit(frame, y, offset=offset),
        "fit_reml": lambda: model.fit_reml(frame, y, offset=offset),
        "estimate_p": lambda: model.estimate_p(
            frame,
            y,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="reml",
            p_bounds=(1.3, 1.7),
            xatol=0.1,
        ),
        "estimate_theta": lambda: model.estimate_theta(frame, y, offset=offset, fit_mode="reml"),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fits[route]()

    assert separation_warnings(caught) == []
    assert published_decision(model).state == "absent"
    assert model.result.separation_statement.tag == "absent"
    assert model.result.separation_message is not None, "the event: the backstop fired"
    log_mean = np.log(model.predict(frame, offset=offset))
    assert log_mean[0] == pytest.approx(-LOG_GUARD, abs=64 * U * LOG_GUARD)
    assert np.all(np.abs(log_mean[1:]) < LOG_GUARD / 2)


def test_an_independent_fit_inside_a_progress_callback_reports_its_own_findings():
    """Each public fit call names its design findings once; a search's candidates share its call.

    ``estimate_p`` runs one ``fit_reml`` per power it tries, and each checks the
    random effect beside the claim-free level: the search names that hazard,
    and the level, once.  A separate model fitted from its progress callback is
    a call of its own, and names its claim-free level as it does when fitted
    alone; so does each of two distributional fits there, whose build holds a
    level's curve over a range with no claim.  Mutation checks: a candidate
    opening its own scope named the hazard once per power; every nested call
    joining the enclosing scope left the callback's fit silent; a SuperLSS fit
    without a scope of its own left the second distributional fit silent.  A
    ``drop1`` there names the level once, as alone; with its comparison fits
    joining the enclosing call's scope instead of its own, it said nothing.
    """
    import superglm as sg
    from superglm import RandomEffect
    from superglm.features.interaction import SplineCategorical

    rng = np.random.default_rng(5)
    n = 440
    c = np.where(np.arange(n) < 25, "zero", "positive")
    count = rng.poisson(0.9, size=n)
    y = np.where(count > 0, rng.gamma(2 * count + 1e-12, 0.6), 0.0)
    y[c == "zero"] = 0.0
    frame = pd.DataFrame({"c": c, "r": rng.choice([f"R{i}" for i in range(12)], size=n)})
    frame["z"] = rng.normal(size=n)
    other = SuperGLM(family=Tweedie(p=1.5), features={"c": Categorical(), "z": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other.fit(frame[["c", "z"]], y)
    # Level C has no claim past x = 3; A and B have claims everywhere.
    level = np.tile(np.repeat(["A", "B", "C"], 9), 4)
    x = np.tile(np.arange(9.0), 12)
    curve_frame = pd.DataFrame({"x": x, "f": level})
    curve_y = np.where((level == "C") & (x > 3), 0.0, np.where(np.arange(108) // 27 == 1, 8.0, 1.0))

    def notices(caught, start):
        return [
            c
            for c in caught
            if issubclass(c.category, SeparationWarning) and str(c.message).startswith(start)
        ]

    def distributional_fit():
        family = sg.NegativeBinomialLS()
        mean = family.mean(
            sg.term("x", Spline(n_knots=4)),
            sg.cat("f"),
            sg.interaction(SplineCategorical("x", "f")),
        )
        model = sg.SuperLSS(family, mean, family.theta())
        lambdas = {"mean:x#wiggle": 1.0, "mean:x:f[B]#wiggle": 1.0, "mean:x:f[C]#wiggle": 1.0}
        model.fit(curve_frame, curve_y, max_inner_iter=2, lambdas=lambdas)

    inner = []

    def progress(phase, payload):
        if not inner:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                SuperGLM(family=Tweedie(p=1.5), features={"c": Categorical()}).fit(frame, y)
                distributional_fit()
                distributional_fit()
            inner.append(notices(caught, "Separation detected"))
            inner.append(notices(caught, "Separation in a spline-by-factor term"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                other.drop1(frame[["c", "z"]], y)
            inner.append(notices(caught, "Separation detected"))

    model = SuperGLM(family=Tweedie(p=1.5), features={"c": Categorical(), "r": RandomEffect()})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.estimate_p(
            frame,
            y,
            fit_mode="reml",
            search_fit_mode="reml",
            p_bounds=(1.3, 1.7),
            xatol=0.1,
            progress_callback=progress,
        )

    assert len(result.evaluations) > 1
    assert [len(found) for found in inner] == [1, 2, 1]
    assert len(notices(caught, "RandomEffect variance component(s) 'r'")) == 1
    assert len(notices(caught, "Separation detected")) == 1


@pytest.mark.parametrize("solver", ["fit_irls_direct", "fit_pirls"])
@pytest.mark.parametrize("mode", [None, "warn", "error", "ignore", "record", "recrod"])
def test_the_exported_solvers_keep_their_separation_keyword(monkeypatch, solver, mode):
    """``fit_irls_direct`` and ``fit_pirls`` take ``separation=`` as they did, default ``"warn"``.

    Called directly on a state carrying the backstop's finding (forced), a
    solve warns by default and under ``"warn"``, refuses under ``"error"`` and
    is silent under ``"ignore"``; ``"record"``, which a model's own solves
    pass, leaves the finding on the result and says nothing.  Any other value
    is refused before the solve, so a misspelt mode cannot drop the warning.
    """
    import superglm.solvers as solvers
    from superglm import GroupLasso
    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    message = force_runtime_finding(monkeypatch)
    x = np.tile([-1.0, 1.0], 40)[:, None]
    y = np.ones(len(x))
    arguments = {
        "X": DesignMatrix([DenseGroupMatrix(x)], n=len(x), p=1),
        "y": y,
        "weights": np.ones_like(y),
        "family": Poisson(),
        "link": LogLink(),
        "groups": [GroupSlice("x", 0, 1)],
        "weight_semantics": "frequency",
        **(
            {"lambda2": 0.0}
            if solver == "fit_irls_direct"
            else {"penalty": GroupLasso(lambda1=0.0)}
        ),
        **({} if mode is None else {"separation": mode}),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "recrod":
            with pytest.raises(ValueError, match="separation must be 'warn', 'error', 'ignore'"):
                getattr(solvers, solver)(**arguments)
            return
        if mode == "error":
            with pytest.raises(SeparationError, match="overflow guard"):
                getattr(solvers, solver)(**arguments)
            return
        result = getattr(solvers, solver)(**arguments)

    result = result[0] if isinstance(result, tuple) else result
    assert len(runtime_claims(caught)) == (1 if mode in (None, "warn") else 0)
    assert result.separation_message == (None if mode == "ignore" else message)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("dropped", [("x",), ("x", "z")])
@pytest.mark.parametrize("mode", ["warn", "ignore", "error"])
def test_drop1_names_the_feature_whose_comparison_fit_is_separated(monkeypatch, mode, dropped):
    """``drop1`` publishes its comparison fits' deviances, so it reports their findings once.

    Control, on the route where no existence decision is built (these estimates exist, so a
    decision would overrule the forced finding): the backstop's statement, as on master.

    The full model fits cleanly; then the backstop's finding is forced on the
    solves of the comparison fit without ``x``, or of both comparison fits.
    The call warns once, naming each comparison fit that carries it (one
    without ``x``, or one without ``x`` and one without ``z``, never one
    without both), stays silent under ``"ignore"`` and refuses under
    ``"error"``; the comparison fit itself, discarded, says nothing.  Mutation
    checks: with comparison fits publishing, the generic warning came without
    the feature's name; with ``drop1`` reading no finding, it said nothing
    under any mode.
    """
    import superglm.diagnostics.separation as separation_module

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.2 + 0.3 * frame["x"].to_numpy())).astype(float)
    model = SuperGLM(
        family="poisson", features={"x": Numeric(), "z": Numeric()}, separation=mode
    ).fit(frame, y)
    message = forced_runtime_finding()

    def finding(design, groups, *_args, **_kwargs):
        missing = {"x", "z"} - {group.name for group in groups}
        return message if missing & set(dropped) else None

    opening = (
        "drop1: the comparison fit without 'x' carries "
        if dropped == ("x",)
        else "drop1: the comparison fits without 'x' and without 'z' each carry "
    )
    monkeypatch.setattr(separation_module, "runtime_finding", finding)
    force_decision_not_built(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "error":
            with pytest.raises(SeparationError) as refused:
                model.drop1(frame, y)
            assert str(refused.value).startswith(opening)
            return
        table = model.drop1(frame, y)

    claims = runtime_claims(caught)
    assert sorted(table["feature"]) == ["x", "z"]
    assert len(claims) == (1 if mode == "warn" else 0)
    assert all(claim.startswith(opening) for claim in claims)


@pytest.mark.parametrize("separated", [True, False], ids=["separated", "offset row only"])
@pytest.mark.parametrize("route", ["fit", "fit_reml", "fit_reml without smooth terms"])
def test_a_refused_fit_says_nothing_else_about_itself(monkeypatch, route, separated):
    """Under ``"error"`` a separated fit is refused before any statement about it is made.

    Separated: the indicator ``z`` has four active rows, every response 0, so the decision
    verifies its direction and the fit is refused with its statement.  A ``fit_reml`` would
    first name ``z`` weakly identified, and the unresolved-row report, which every route runs
    after its solve, is forced to speak; neither is reached.  The offset row only (one
    claim-free row at offset -100, at the log link's guard): the decision proves every
    estimate finite, so nothing refuses and nothing is said about separation; master refused
    it on the guard heuristic (the log-binomial ``fit`` and both ``fit_reml`` designs), a false
    call.  Mutation checks: with the weak-identification notice given as the REML finish ran,
    ``fit_reml`` warned before its refusal; with the decided absence not overruling the
    backstop, the offset-only fits were refused.
    """
    from superglm.distributions import Binomial
    from superglm.links import LogLink
    from superglm.model import fit_ops

    def unresolved_rows(model):
        warnings.warn("unresolved rows", UserWarning)

    monkeypatch.setattr(fit_ops, "_warn_unresolved_rows", unresolved_rows)
    z_rows = [1, 2, 3, 4] if separated else []
    if route != "fit_reml":
        rng = np.random.default_rng(4)
        n = 400
        x = rng.uniform(-2.0, 2.0, n)
        y = rng.binomial(1, np.clip(np.exp(-0.5 + 0.4 * x), 0.01, 0.99)).astype(float)
        y[np.argmax(x)] = 1.0
        z = np.zeros(n)
        z[z_rows] = 1.0
        y[z_rows] = 0.0
        frame = pd.DataFrame({"x": x, "z": z})
        model = SuperGLM(
            family=Binomial(),
            link=LogLink(),
            features={"x": Numeric(), **({"z": Numeric()} if separated else {})},
            separation="error",
        )
    else:
        rng = np.random.default_rng(3)
        n = 1500
        x = rng.uniform(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.3 + np.sin(3.0 * x))).astype(float)
        z = np.zeros(n)
        z[z_rows] = 1.0
        y[z_rows] = 0.0
        frame = pd.DataFrame({"x": x, "z": z})
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", k=8), **({"z": Numeric()} if separated else {})},
            separation="error",
        )
    offset = np.zeros(n)
    offset[0], y[0] = -100.0, 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if separated:
            with pytest.raises(SeparationError, match="moving z,"):
                getattr(model, route.split()[0])(frame, y, offset=offset)
        else:
            getattr(model, route.split()[0])(frame, y, offset=offset)

    if separated:
        assert [c.category.__name__ for c in caught] == []
    else:
        assert separation_warnings(caught) == []
        assert published_decision(model).state == "absent"


def test_strict_fit_reml_decides_on_the_theta_refit_it_returns(monkeypatch):
    """An NB2 ``theta="auto"`` REML fit refits at each new theta; only the last is returned.

    The backstop's finding is forced on every refit but the last.  Under
    ``"error"`` the fit is returned, its state clean: a refit the alternation
    supersedes is no fit the caller receives.  Mutation check: refusing each
    strict refit as it finished refused the first.
    """
    import superglm.diagnostics.separation as separation_module
    from superglm.distributions import NegativeBinomial
    from superglm.model import fit_ops

    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    mean = np.exp(0.3 + np.sin(3.0 * x))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mean)).astype(float)
    frame = pd.DataFrame({"x": x})
    real_finalize, real_finding = fit_ops.finalize_reml_fit, separation_module.runtime_finding
    message = forced_runtime_finding()
    refits, last = [], []

    def finalize(*args, **kwargs):
        refits.append(1)
        return real_finalize(*args, **kwargs)

    def finding(*args, **kwargs):
        if last and len(refits) < last[0]:
            return message
        return real_finding(*args, **kwargs)

    monkeypatch.setattr(fit_ops, "finalize_reml_fit", finalize)
    monkeypatch.setattr(separation_module, "runtime_finding", finding)

    def fit():
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"x": Spline(kind="cr", k=8)},
            separation="error",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit_reml(frame, y)

    fit()
    assert len(refits) > 1
    last.append(len(refits))
    refits.clear()
    model = fit()

    assert len(refits) == last[0]
    assert model.result.separation_message is None


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("mode", [None, "warn", "error", "ignore", "record", "recrod"])
def test_the_exported_reml_optimizers_disclose_the_state_they_return(monkeypatch, discrete, mode):
    """``optimize_direct_reml`` and ``optimize_discrete_reml_cached_w`` speak as the solvers do.

    Each is called directly, with the arguments a model's ``fit_reml`` passes
    it, on solves carrying the backstop's finding (forced).  It warns once by
    default and under ``"warn"``, refuses under ``"error"`` and says nothing
    under ``"ignore"`` and ``"record"`` (which the model passes, since its fit
    publishes the finding); a misspelt mode is refused first.  Mutation check:
    with the optimizer disclosing nothing, a direct call returned the finding
    silently.
    """
    from superglm.model import reml_ops
    from superglm.reml import optimize_direct_reml, optimize_discrete_reml_cached_w

    frame, y = poisson_curve()
    message = force_runtime_finding(monkeypatch)
    calls = []

    def standalone(*args, **kwargs):
        assert kwargs.pop("separation") == "record"
        function = optimize_direct_reml
        if discrete:
            function = optimize_discrete_reml_cached_w
            args = args[:4] + args[5:]
            kwargs.pop("w_correction_order")
            kwargs.pop("presolve")
        modes = {} if mode is None else {"separation": mode}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = function(*args, **kwargs, **modes)
            except (SeparationError, ValueError) as exc:
                result = exc
        calls.append((result, runtime_claims(caught)))
        return function(*args, **kwargs, separation="record")

    monkeypatch.setattr(reml_ops, "optimize_direct_reml", standalone)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="cr", k=8)},
        discrete=discrete,
        separation="ignore",
    )
    model.fit_reml(frame, y)

    [(result, claims)] = calls
    if mode == "recrod":
        assert isinstance(result, ValueError)
        assert "separation must be 'warn', 'error', 'ignore'" in str(result)
        return
    if mode == "error":
        assert isinstance(result, SeparationError)
        assert str(result) == message
        return
    assert claims == ([message] if mode in (None, "warn") else [])
    assert result.pirls_result.separation_message == message


def test_separation_error_takes_the_arguments_a_value_error_does():
    """``SeparationError`` keeps ``ValueError``'s positional arguments, and loads old pickles.

    ``levels`` is keyword-only.  An exception pickled before ``levels``
    existed (its class and positional arguments only) loads with ``levels``
    false.  Mutation check: with a single positional message, two arguments
    raised ``TypeError`` and the no-argument form held one empty argument.
    """
    import pickle

    assert SeparationError("x", "foo").args == ("x", "foo")
    assert SeparationError().args == ()
    assert SeparationError("level", levels=True).levels

    class Saved:
        def __reduce__(self):
            return SeparationError, ("x", "foo")

    loaded = pickle.loads(pickle.dumps(Saved()))
    assert type(loaded) is SeparationError
    assert (loaded.args, loaded.levels) == (("x", "foo"), False)
    again = pickle.loads(pickle.dumps(SeparationError("level", levels=True)))
    assert (again.args, again.levels) == (("level",), True)


def test_a_coefficient_revision_clears_the_solves_separation_finding():
    """A revised state is not one a solve returned, and an edit that sets ``z`` drops its claim.

    A numeric indicator whose rows have no claim is fitted; its decision says ``z`` has no
    finite estimate.  The editor sets its slope to zero, so every revised prediction is the
    claim rows' mean and reads nothing of where the walk stopped: the edited result
    (``coefficients_revised``) says nothing about separation, as it carries no convergence,
    and the source model keeps its statement.  At ``07a1401a5`` the source fit said nothing
    either.  Mutation check: with the revision keeping every statement, the edited model still
    said ``z`` was separated.
    """
    from superglm.editor import EditorSession

    z = np.repeat([0.0, 1.0], 64)
    frame, y = pd.DataFrame({"z": z}), 1.0 - z
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SuperGLM(family="poisson", features={"z": Numeric()}).fit(frame, y)
    assert len(decision_claims(caught)) == 1
    source = model.result.separation_statement
    assert source.tag == "present" and "moving z" in source.text
    session = EditorSession.from_model(model, terms=["z"], train_data=(frame, y))
    session.terms["z"].edited_log_effect = np.array([0.0])
    edited = session.to_model()

    assert edited.result.termination_reason == "coefficients_revised"
    np.testing.assert_allclose(edited.predict(frame), 1.0, rtol=1e-6)
    assert edited.result.separation_message is None
    assert edited.result.separation_statement is None
    assert model.result.separation_statement is source


def test_a_levels_anchor_rank_is_read_from_an_economy_factorization(monkeypatch):
    """A level with many distinct anchors allocates no square left factor (PR 508 round 4).

    ``_level_recession`` took the SVD of its ``(n, k)`` anchor rows with
    ``full_matrices=True`` and read only the rank and the right null space:
    an ``(n, n)`` left factor, about 3.2 GB at 20,000 anchors.  The work is
    bounded by the factors' sizes, not time.  Mutation: ``full_matrices=True``.
    """
    from superglm.diagnostics.separation import _level_recession

    sizes: list[int] = []
    real = np.linalg.svd

    def spy(a, *args, **kwargs):
        factors = real(a, *args, **kwargs)
        sizes.append(max(factor.size for factor in factors))
        return factors

    monkeypatch.setattr(np.linalg, "svd", spy)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    natural = np.column_stack([np.ones(n), x])
    inside = np.ones(n, dtype=bool)
    none = np.zeros(n, dtype=bool)
    assert _level_recession(natural, none, none, inside, np.eye(2)) is None
    assert sizes and max(sizes) <= n * 2
    from superglm.diagnostics.separation import _rank_and_null_space

    # fewer rows than columns still returns the whole null space
    rank, null = _rank_and_null_space(np.array([[1.0, 0.0, 0.0, 1.0]]))
    assert rank == 1 and null.shape == (4, 3)
    assert np.allclose(np.array([[1.0, 0.0, 0.0, 1.0]]) @ null, 0.0)


@pytest.mark.parametrize("origin", [0.0, 1e8])
def test_two_positive_anchors_fix_a_lines_both_coefficients_at_any_origin(origin):
    """Moving the covariate's origin does not make a level separate (PR 508 round 4).

    Positive responses at ``x = origin`` and ``origin + 1`` fix both affine
    coefficients of the level's line, so the zero at ``origin + 2`` cannot
    recede.  The anchors' rank was decided on the unscaled ``[1, x]``, where at
    ``1e8`` the second singular value falls below ``max(shape) eps`` of the
    first: rank one, and a false direction of recession.  The columns are now
    equilibrated.  Mutation: the columns unscaled.
    """
    from superglm.diagnostics.separation import _level_recession

    x = origin + np.array([0.0, 1.0, 2.0])
    natural = np.column_stack([np.ones(3), x])
    inside = np.array([True, True, False])
    lower = np.array([False, False, True])
    none = np.zeros(3, dtype=bool)
    assert _level_recession(natural, lower, none, inside, np.eye(2)) is None
    assert _level_recession(natural, lower, none, inside, np.eye(2), span=False) is None


# ── The scan reads the build's level codes ───────────────────────


def _separated_level_column(seed=5, n=2500):
    rng = np.random.default_rng(seed)
    c = rng.choice([f"c{i}" for i in range(8)], size=n)
    y = np.where(c == "c6", 0.0, rng.poisson(0.5, size=n).astype(float) + (c != "c6"))
    return c, y


def _build_separated_main_effect(spec, c, y):
    from superglm._frame import as_eager_frame
    from superglm.dm_builder import build_design_matrix

    specs = {"c": spec}
    return build_design_matrix(
        as_eager_frame(pd.DataFrame({"c": c})),
        y,
        np.ones(len(y)),
        None,
        family="poisson",
        link_spec=None,
        specs=specs,
        feature_order=["c"],
        interaction_specs={},
        interaction_order=[],
        pending_interactions=[],
        model_discrete=False,
        n_bins_config=32,
        lambda2=0.1,
        weight_semantics="frequency",
    )


def _spec_for(kind, c):
    from superglm import collapse_levels

    if kind == "declared":
        return Categorical(levels=sorted(set(c)))
    if kind == "grouped":
        grouping = collapse_levels(sorted(set(c)), groups={"c01": ["c0", "c1"]})
        return Categorical(grouping=grouping)
    return Categorical()


@pytest.mark.parametrize("kind", ["inferred", "declared", "grouped"])
def test_a_main_effect_columns_level_codes_are_computed_once_per_build(monkeypatch, kind):
    """The build codes the column against its universe; the separation scan reads those codes.

    Both used to call ``_codes_against`` on the same column, one pass over every
    row each. A declared universe codes once in the build (an inferred or grouped
    one comes from ``factorize``) and the scan then codes nothing. Mutation:
    a scan that codes the column itself again makes this one call more.
    """
    import superglm.features.categorical as categorical

    c, y = _separated_level_column()
    calls = []
    real = categorical._codes_against
    monkeypatch.setattr(
        categorical,
        "_codes_against",
        lambda values, levels: calls.append(len(values)) or real(values, levels),
    )
    with pytest.warns(SeparationWarning, match="'c6'"):
        built = _build_separated_main_effect(_spec_for(kind, c), c, y)
    assert built.compiled.specs["c"]._separated_levels_at_fit == ("c6",)
    assert calls == ([len(c)] if kind == "declared" else [])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array(["b", "a", "c", "a", "b"] * 40, dtype=object), id="object"),
        pytest.param(np.array(["b", "a", "c", "a", "b"] * 40), id="unicode"),
        pytest.param(np.array([3, 1, 2, 1, 3] * 40), id="int"),
        pytest.param(np.array([0.5, -0.0, 0.0, 2.5, 0.5] * 40), id="float"),
        pytest.param(np.array([True, False, True, True, False] * 40), id="bool"),
        pytest.param(np.array([1, 1.0, True, "x", 2] * 40, dtype=object), id="mixed-object"),
    ],
)
@pytest.mark.parametrize("declared", [False, True])
def test_build_codes_are_the_codes_the_scan_would_make(values, declared):
    """``_build_with_codes`` returns exactly ``_codes_against(column, levels)``: coding the column again is redundant.

    The inferred universe comes out of ``factorize``, whose row codes must be
    the unique-index lookup's for the scan to take them as its own.
    """
    from superglm.diagnostics.separation import scan_categorical_term
    from superglm.features.categorical import _codes_against

    spec = Categorical(levels=pd.unique(values).tolist()) if declared else Categorical()
    info, codes = spec._build_with_codes(values)
    assert info.cat_codes is not None
    expected = _codes_against(values, spec._levels)
    assert codes.dtype == expected.dtype
    assert np.array_equal(codes, expected)
    # The level with code 0 has an all-zero response, so it separates under "zero".
    y = (expected != 0).astype(float)
    weight = np.ones(len(values))
    found = scan_categorical_term("c", spec, values, y, weight, ("zero",), codes)
    assert found == scan_categorical_term("c", spec, values, y, weight, ("zero",))
    assert [f.labels for f in found] == [[spec._levels[0]]]


class _WithoutPenalty(Categorical):
    """A subclass whose ``build`` marks its group unpenalised."""

    def build(self, x, sample_weight=None):
        info = super().build(x, sample_weight)
        info.penalized = False
        return info


@pytest.mark.parametrize("separation", ["warn", "ignore"])
def test_a_categorical_subclass_build_override_is_kept_whether_or_not_the_fit_scans(separation):
    """Scanning for separation must not change a fit: the term's own ``build`` runs.

    The scan reads the base class's build codes, which a subclass that overrides
    ``build()`` does not hand over, so the compiler calls its override and the scan
    codes the column itself. Fails on de1b48ca, which called the base build whenever
    the scan was on and predicted [2.5, 2.5] here.
    """
    c = np.array(["a", "b"] * 100)
    y = np.array([1.0, 4.0] * 100)
    model = SuperGLM(
        family="poisson",
        selection_penalty=1e6,
        separation=separation,
        features={"c": _WithoutPenalty()},
    )
    model.fit(pd.DataFrame({"c": c}), y)
    np.testing.assert_allclose(
        model.predict(pd.DataFrame({"c": ["a", "b"]})), [1.0, 4.0], rtol=1e-6
    )


def test_a_categorical_subclass_build_override_is_still_scanned(monkeypatch):
    import superglm.features.categorical as categorical

    c, y = _separated_level_column()
    calls = []
    real = categorical._codes_against
    monkeypatch.setattr(
        categorical,
        "_codes_against",
        lambda values, levels: calls.append(len(values)) or real(values, levels),
    )
    with pytest.warns(SeparationWarning, match="'c6'"):
        built = _build_separated_main_effect(_WithoutPenalty(), c, y)
    assert built.compiled.specs["c"]._separated_levels_at_fit == ("c6",)
    assert calls == [len(c)]  # the scan's own pass: the override handed it no codes


@pytest.mark.parametrize(
    ("spec", "x", "message"),
    [
        pytest.param(
            lambda: Categorical(base="z", levels=["a", "b", "z"]),
            np.array(["a", "b", "a"], dtype=object),
            "falling back",
            id="base-fallback",
        ),
        pytest.param(
            lambda: Categorical(base="a", levels=["a", "b", "c"]),
            np.array(["a", "b", "a"], dtype=object),
            "pinned to base",
            id="pinned-level",
        ),
    ],
)
def test_build_warnings_are_attributed_to_the_callers_line(spec, x, message):
    """``build()`` is one frame deeper than the method that warns; the stacklevel counts it."""
    import inspect

    with pytest.warns(UserWarning, match=message) as record:
        spec().build(x)
        line = inspect.currentframe().f_lineno - 1
    attributed = [w for w in record if message in str(w.message)]
    assert [(w.filename, w.lineno) for w in attributed] == [(__file__, line)]


def test_a_fit_reports_the_build_warnings_where_master_did():
    """A fit re-emits the term's build warnings from the model's build, as it always has."""
    c = np.array(["a", "b"] * 20)
    y = np.array([1.0, 4.0] * 20)
    with pytest.warns(UserWarning, match="falling back") as record:
        SuperGLM(
            family="poisson", features={"c": Categorical(base="z", levels=["a", "b", "z"])}
        ).fit(pd.DataFrame({"c": c}), y)
    attributed = [w for w in record if "falling back" in str(w.message)]
    assert [Path(w.filename).name for w in attributed] == ["dm_builder.py"]


# ── The decision's statement: once, under the model's mode, on every route ──


def separated_indicator(seed=7, n=2000):
    """A Poisson response with a claim-free numeric indicator ``z`` beside a real effect ``x``."""
    rng = np.random.default_rng(seed)
    z = (rng.uniform(size=n) < 0.25).astype(float)
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.3 + np.sin(3.0 * x))).astype(float)
    y[z == 1.0] = 0.0
    return pd.DataFrame({"z": z, "x": x}), y


@pytest.mark.parametrize("mode", ["warn", "ignore", "error"])
@pytest.mark.parametrize("route", ["fit", "fit_reml", "NB2 theta auto", "fit_path", "Final fit"])
def test_a_decided_separation_reads_the_models_mode_on_every_route(route, mode):
    """A decided separation warns once, says nothing or refuses, as the model's mode says.

    ``z`` separates its rows; ``fit``, ``fit_reml``, NB2 ``theta="auto"`` (its decision taken
    on the published refit of the alternation) and Final fit decide it, and under ``"warn"``
    say so once, from the decision; under ``"error"`` they are refused with that statement and
    nothing said before (Final fit with its own sentence for a solver's finding).  ``fit_path``
    builds no decision (a selection penalty is not quadratic): its walk stops short of the
    guard, so it is silent, as on master; it is listed in the pull request as a known silent
    case.  At ``07a1401a5`` every decided route was silent under ``"warn"`` and fitted under
    ``"error"``.  Mutation checks: with the published NB2 refit still marked as a search refit,
    the strict alternation fitted; with the refusal given where the fit is handed over, the
    NB2 fit warned that ``z`` was weakly identified before it was refused.
    """
    from superglm.distributions import NegativeBinomial
    from superglm.editor import EditorSession
    from superglm.editor.cv import capture_final_fit, run_final_fit
    from superglm.editor.errors import EditorValueError

    frame, y = separated_indicator()
    features = {"z": Numeric(), "x": Spline(n_knots=6)}
    options = {"separation": mode}
    if route == "NB2 theta auto":
        options["family"] = NegativeBinomial(theta="auto")
    else:
        options["family"] = "poisson"
    if route == "fit_path":
        options.update(selection_penalty=1e-3, penalty_features="x")
    model = SuperGLM(features=features, **options)

    def run():
        if route == "fit":
            return model.fit(frame, y)
        if route in ("fit_reml", "NB2 theta auto"):
            return model.fit_reml(frame, y)
        if route == "fit_path":
            return model.fit_path(frame, y, n_lambda=4)
        # the template is fitted where z's rows have claims; Final fit refits on the rows given
        claims = np.where(frame["z"].to_numpy() == 1.0, 1.0, y)
        template = SuperGLM(features=features, family="poisson", separation=mode).fit(frame, claims)
        session = EditorSession.from_model(template, train_data=(frame, y))

        class Context:
            def progress(self, *args, **kwargs):
                pass

            def check(self):
                pass

        return run_final_fit(capture_final_fit(session), Context())

    refused = (SeparationError, EditorValueError)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "error" and route != "fit_path":
            with pytest.raises(refused) as raised:
                run()
        else:
            run()

    if route == "fit_path":
        assert separation_warnings(caught) == []
        assert model.result.separation_statement is None
        return
    if mode == "error":
        assert [c.category.__name__ for c in caught] == []
        if route == "Final fit":
            return
        assert str(raised.value).startswith("Separation: at least ")
        assert says_infinite(str(raised.value), "z")
        return
    claims = separation_warnings(caught)
    if mode == "ignore":
        assert claims == []
        return
    assert len(claims) == 1 and claims == decision_claims(caught)
    assert says_infinite(claims[0], "z")


def separated_level(seed=11, n=3000, with_indicator=False):
    """Level D of ``f`` has no claim; with ``with_indicator`` neither has ``z``'s rows."""
    rng = np.random.default_rng(seed)
    f = rng.choice(list("ABCD"), n)
    x = rng.uniform(0.0, 1.0, n)
    z = (rng.uniform(size=n) < 0.1).astype(float)
    y = rng.poisson(np.exp(0.2 + np.sin(3.0 * x) + 0.3 * (f == "B"))).astype(float)
    y[f == "D"] = 0.0
    if with_indicator:
        y[z == 1.0] = 0.0
    frame = pd.DataFrame({"f": f, "x": x, "z": z})
    return frame, y


@pytest.mark.parametrize("overlap", ["the scan's rows", "beyond the scan's rows"])
def test_a_separation_the_build_scan_named_is_warned_about_once(overlap):
    """A separation on rows the build scan already named is the scan's one warning.

    The scan names level D before any solve; the decision then separates D's rows (and with
    the indicator ``z`` claim-free too, ``z``'s rows beyond D).  On D's rows alone the fit warns
    once, the scan's, and the decision's statement stays on the fit, naming ``f[D]`` (the
    summary footnote too).  Beyond them the decision adds what the scan could not see, and says
    so: two warnings, the second the decision's, naming both.  At ``07a1401a5`` the second fit
    was silent about ``z``.  Mutation check: without the scan's rows kept in the call's scope,
    the first fit warned twice about D.
    """
    beyond = overlap == "beyond the scan's rows"
    frame, y = separated_level(with_indicator=beyond)
    model = SuperGLM(
        family="poisson",
        features={"f": Categorical(base="A"), "x": Spline(n_knots=6), "z": Numeric()},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, y)

    claims = separation_warnings(caught)
    assert claims[0].startswith("Separation detected") and "'f'" in claims[0]
    statement = model.result.separation_statement
    assert statement.tag == "present"
    if not beyond:
        assert len(claims) == 1
        assert published_decision(model).names == ("f[D]",)
        return
    assert len(claims) == 2 and decision_claims(caught) == claims[1:]
    assert says_infinite(claims[1], "f[D], z")
    assert re.search(r"\(level D of f, and \d+ other rows\)", claims[1])


def test_drop1_says_a_separated_comparison_fit_is_the_limit_fit():
    """A comparison fit whose decision proves separation is the limit fit, and ``drop1`` says so.

    ``z`` separates its rows in the full model and in the comparison fit without ``x``; that
    fit is discarded, so it says nothing itself, and ``drop1`` names it once.  Its deviance is
    the supremum the likelihood approaches, not a fit short of the maximum (the backstop's
    case), so the table compares with that limit.  The comparison fit without ``z`` is not
    separated.  At ``07a1401a5`` ``drop1`` said nothing.  Mutation check: with every finding
    worded as the backstop's, the statement called the comparison fit not a maximum-likelihood
    fit.
    """
    frame, y = separated_indicator()
    frame = frame.assign(x=frame["x"] - 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        model = SuperGLM(family="poisson", features={"z": Numeric(), "x": Numeric()}).fit(frame, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = model.drop1(frame, y)

    claims = separation_warnings(caught)
    assert sorted(table["feature"]) == ["x", "z"]
    assert len(claims) == 1
    assert claims[0].startswith("drop1: the comparison fit without 'x' carries ")
    assert "the limit of a fit" in claims[0] and "not a maximum-likelihood fit" not in claims[0]
    assert "moving z" in claims[0]


def test_the_separation_statement_moves_only_as_its_table_says(monkeypatch):
    """The statement is one tagged value; its writers make only the moves its table declares.

    The table itself is checked (every tag is a state, every state is reached from ``"none"``
    and left by some move), then each declared revision move is made by its writer
    (``revised_statement``) from a statement of that tag, and an undeclared move is refused.
    Publication's moves are made by fits throughout this file.  Mutation check: with a
    partial statement revised away, the writer made a move the table does not declare.
    """
    from superglm.diagnostics.separation import Separation
    from superglm.model import separation_ops as ops

    moves = ops.STATEMENT_MOVES
    tags = set(moves)
    assert tags == {"none", "absent", "present", "partial", "unresolved", "heuristic"}
    for state, events in moves.items():
        assert set(events.values()) <= tags
        assert any(target != state for target in events.values())
    reached, frontier = {"none"}, ["none"]
    while frontier:
        for target in moves[frontier.pop()].values():
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    assert reached == tags

    frame, y = separated_indicator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="poisson", features={"z": Numeric(), "x": Numeric()}).fit(frame, y)
    present = model.result.separation_statement
    assert present.tag == "present"
    z = int(present.reached[0])
    partial = ops.SeparationStatement(
        "partial", "text", reached=present.reached, decision=present.decision
    )
    absent = ops.SeparationStatement("absent", decision=Separation("absent", "certificate"))
    guard = ops.SeparationStatement("heuristic", "the backstop's", guard_rows=3)
    budget = ops.SeparationStatement("heuristic", "the backstop's")
    unresolved = ops.SeparationStatement("unresolved", "not decided", guard_rows=3)
    monkeypatch.setattr(ops, "_guard_rows", lambda model, eta: 0)
    cases = [
        (None, (), "revise", None),
        (absent, (z,), "revise", "absent"),
        (present, (), "revise", "present"),
        (present, (z,), "revise away", None),
        (partial, (z,), "revise", "partial"),
        (unresolved, (), "revise", "unresolved"),
        (budget, (), "revise", "heuristic"),
        (guard, (), "revise away", None),
    ]
    for before, replaced, event, tag in cases:
        after = ops.revised_statement(model, before, np.asarray(replaced, dtype=np.intp))
        assert (None if after is None else after.tag) == tag
        assert moves["none" if before is None else before.tag][event] == (
            "none" if tag is None else tag
        )
    revised = ops.revised_statement(model, unresolved, np.zeros(0, dtype=np.intp))
    assert revised.text is None, "an unresolved decision's warning stands only on its evidence"
    with pytest.raises(RuntimeError, match="partial --revise away--> none is not a move"):
        ops._moved(partial, "revise away", None)
