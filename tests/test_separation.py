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


def separated_numeric_design(seed=7, n=3000):
    rng = np.random.default_rng(seed)
    z = (rng.uniform(size=n) < 0.25).astype(float)
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
