"""Separation detection: build-time cell scan and in-solver backstop.

Issues #340 (name separated interaction cells at build time, before any IRLS
iteration) and #341 (refuse, or at minimum loudly flag, instead of returning
finite garbage on a separated design).
"""

import re
import warnings

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


def test_predict_names_a_separated_categorical_level_it_meets():
    """``predict`` names a main-effect level that had only zero responses at fit (sz_explosion round 2).

    The fit warns at build; the level has no finite effect, and its
    predictions sit near zero wherever the walk stopped.  ``predict`` said
    nothing.  It now names the level when its rows reach it, and stays
    silent on rows of other levels.  Mutation: no record of the level.
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


def test_runtime_backstop_warns_on_pinned_predictor():
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


@pytest.mark.parametrize("share", [0.25, 0.6])
def test_runtime_backstop_names_the_group_that_moves_the_stopped_rows(share):
    """A coefficient's size is no guide across groups; what it moves on the stopped rows is.

    ``x`` is a real effect on a column of scale 1e-3, so its coefficient
    (about 260) dwarfs the separated ``z`` (pinned near -80).  Naming the
    largest coefficient named ``x``; the rows the guard stopped are moved by
    ``z``.  On the freMTPL2 guide model the same rule named a tensor term for
    a brand's spline-by-factor curve.  At ``share=0.6`` the stopped rows are
    most of the rows, so a median over every row sits on them and ``z``
    moved nothing from it; the move is measured from the rows that did not
    stop.
    """
    df, y, w = separated_numeric_design(share=share)
    rng = np.random.default_rng(11)
    df = df.assign(x=rng.normal(size=len(df)) * 1e-3)
    y = np.where(df["z"].to_numpy() == 1.0, 0.0, y * np.exp(300.0 * df["x"].to_numpy()))
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


def test_runtime_backstop_measures_a_group_from_its_bulk_not_from_zero():
    """A covariate far from zero carries a large constant the intercept cancels.

    ``year`` sits near 2015 with a coefficient of 0.1, so its raw contribution
    is about 212 on every row, beyond the separated ``z`` (pinned near -80).
    What it moves the stopped rows away from the bulk is under 1.
    """
    df, y, w = separated_numeric_design()
    year = 2015.0 + np.random.default_rng(11).integers(-5, 6, size=len(df)).astype(float)
    df = df.assign(year=year)
    y = np.where(df["z"].to_numpy() == 1.0, 0.0, y * np.exp(0.1 * (year - 2015.0)))
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


def test_runtime_backstop_reports_the_expected_curvature_ratio():
    """Tweedie/log rows are observed-Newton; the gate compares the Fisher ratio.

    The iteration log keeps the observed weights, which are the Fisher weights
    times (2 - p) + (p - 1) y / mu, so their ratio is not the one the warning reports.
    """
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
    with pytest.raises(SeparationError, match="signature of separation"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(df, y, sample_weight=w)


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
    assert [w.filename.rsplit("/", 1)[-1] for w in attributed] == ["dm_builder.py"]
