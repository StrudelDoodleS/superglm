"""A REML search conditions on the first finite price of a separated level."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, RandomEffect, Spline, SuperGLM
from superglm.distributions import NegativeBinomial, Tweedie


def test_refused_and_accepted_trials_use_the_cold_price(monkeypatch):
    import superglm.model.reml_finalize as final
    import superglm.reml.direct as direct

    rng = np.random.default_rng(39)
    x = rng.uniform(size=1300)
    mask = np.arange(x.size) < 15
    y = rng.poisson(np.exp(np.sin(8 * x)))
    y[mask] = 0
    X = pd.DataFrame({"r": np.where(mask, "Z", "A"), "x": x})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={"r": Categorical(base="A"), "x": Spline(n_knots=5)},
    )
    solve = direct.fit_irls_direct
    objective = direct.reml_laml_objective
    phases = []
    prices = []
    rejected = False

    def record(**kw):
        phase = (kw.get("debug_context") or {}).get("phase", "terminal")
        phases.append(phase)
        result = solve(**kw)
        if phase == "bootstrap":
            prices.append(result[0].beta[0])
        else:
            assert result[0].beta[0] == 0
            np.testing.assert_array_equal(kw["offset"], np.where(mask, prices[0], 0))
        return result

    def reject_once(*args, **kw):
        nonlocal rejected
        evaluation = objective(*args, **kw)
        if phases[-1] == "line_search" and not rejected:
            rejected = True
            return np.inf
        return evaluation

    monkeypatch.setattr(direct, "fit_irls_direct", record)
    monkeypatch.setattr(final, "fit_irls_direct", record)
    monkeypatch.setattr(direct, "reml_laml_objective", reject_once)
    model.fit_reml(X, y)
    assert rejected and phases.count("line_search") >= 2 and "candidate" in phases
    assert phases[-1] == "terminal" and model._reml_result.converged
    assert _effect(model, "r", "Z") == prices[0]


def test_only_the_group_without_a_positive_weight_event_is_held():
    from superglm import collapse_levels

    rng = np.random.default_rng(86)
    x = rng.uniform(size=1500)
    r = np.where(x < 0.5, "A", "B")
    r[:12], r[12:22], r[22:45] = "C", "D", "E"
    y = rng.poisson(0.6 + x)
    y[:45] = 0
    y[0] = y[23] = 1
    weights = np.ones(x.size)
    weights[0] = 0
    X = pd.DataFrame({"r": r, "x": x})
    grouping = collapse_levels(sorted(set(r)), groups={"CD": ["C", "D"]})

    def fit(mode, k):
        return SuperGLM(
            family="poisson",
            selection_penalty=0,
            separation=mode,
            features={"r": Categorical(base="A", grouping=grouping), "x": Spline(n_knots=5)},
        ).fit_reml(X, y, sample_weight=weights, max_reml_iter=k)

    m = fit("warn", 20)
    term = m.term_inference("r")
    assert list(np.asarray(term.levels)[term.separated]) == ["C", "D"]
    price = _effect(m, "r", "C")
    assert _effect(fit("warn", 1), "r", "D") == price
    assert _effect(fit("ignore", 20), "r", "C") < price - 1


def _effect(model, name, level):
    term = model.term_inference(name)
    return term.log_relativity[list(term.levels).index(level)]


def _fit_with_separated_z(base, family="poisson"):
    from superglm import PSpline

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 2000)
    r = rng.choice(list("ABC"), len(x), p=[0.4, 0.4, 0.2])
    r[:30] = "Z"
    y = rng.poisson(np.exp(-1.5 + 0.3 * np.sin(x) + 0.4 * (r == "B") - 0.3 * (r == "C")))
    y[:30] = 0
    if family == "binomial":
        y = (y > 0).astype(float)
    X = pd.DataFrame({"x": x, "r": r})
    model = SuperGLM(family=family, features={"x": PSpline(n_knots=8), "r": Categorical(base=base)})
    return model.fit_reml(X, y), X, y


@pytest.mark.parametrize("base", ["A", "Z"])
def test_metrics_blank_only_contrasts_that_move_under_separation(base):
    model, X, y = _fit_with_separated_z(base)
    metrics = model.metrics(X, y)
    feature = metrics.feature_se("r")
    blank = np.array([base == "Z" or level == "Z" for level in feature["levels"]])
    for se in (
        metrics.coefficient_se["r"],
        metrics.coefficient_se_raw["r"],
        feature["se_log_relativity"],
    ):
        np.testing.assert_array_equal(np.isnan(se), blank)
        assert (se[~blank] > 0).all()
    assert np.isnan(metrics.intercept_se) == (base == "Z")
    assert np.isnan(metrics.intercept_se_raw) == (base == "Z")
    native = model.term_inference("r", centering="native")
    raw = dict(zip(native.levels, native.se_log_relativity, strict=True))
    np.testing.assert_allclose(
        metrics.coefficient_se_raw["r"], [raw[lvl] for lvl in feature["levels"]]
    )


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize(
    "family,link",
    [
        ("poisson", "log"),
        ("binomial", "logit"),
        ("binomial", "log"),
        (NegativeBinomial(theta=5), "log"),
        (Tweedie(p=1.5), "log"),
    ],
)
def test_a_separated_price_is_independent_of_the_smoothing_budget(family, link, discrete):
    rng = np.random.default_rng(42)
    n = 2400
    x = rng.uniform(size=n)
    region = rng.choice(list("AB"), n)
    region[:12] = "Z"
    y = rng.binomial(1, 0.10 + 0.15 * np.sin(3 * x) ** 2).astype(float)
    y[:12] = 0
    if link == "logit":
        region[12:22] = "Y"
        y[12:22] = 1
    frame = pd.DataFrame({"r": region, "x": x})
    model = SuperGLM(
        family=family,
        link=link,
        discrete=discrete,
        selection_penalty=0,
        features={"r": Categorical(base="A"), "x": Spline(n_knots=5)},
    )
    levels = ["Z", "Y"] if link == "logit" else ["Z"]
    model.fit(frame, y)
    ordinary = [_effect(model, "r", level) for level in levels]
    model.fit_reml(frame, y, max_reml_iter=1)
    held = [_effect(model, "r", level) for level in levels]
    np.testing.assert_allclose(held, ordinary, atol=1.0, rtol=0)
    model.fit_reml(frame, y, max_reml_iter=20)
    assert [_effect(model, "r", level) for level in levels] == held
    ti = model.term_inference("r")
    assert np.isnan(ti.se_log_relativity[ti.separated]).all()
    assert model.result.converged and model._reml_result.converged
    from superglm.model.estimability_ops import NonEstimablePredictionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", NonEstimablePredictionWarning)
        model.predict(frame)


def test_random_effect_reml_matches_the_same_rows_with_an_explicit_fixed_offset():
    rng = np.random.default_rng(0)
    n = 20000
    lev = rng.choice(list("ABCDE"), n, p=[0.4, 0.3, 0.2, 0.098, 0.002])
    grp = rng.integers(0, 30, n)
    u = rng.normal(0, 0.2, 30)
    y = rng.poisson(0.08 * np.where(lev == "B", 1.3, 1) * np.exp(u[grp])).astype(float)
    hit = lev == "E"
    y[hit] = 0
    frame = pd.DataFrame({"lev": lev, "grp": grp.astype(str)})

    def fit(frame, offset=None):
        return SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"lev": Categorical(base="A"), "grp": RandomEffect()},
        ).fit_reml(frame, y, offset=offset)

    model = fit(frame)
    price = _effect(model, "lev", "E")
    assert price == pytest.approx(-7.89027, abs=1e-3)  # first cold fit on #694
    healthy = frame.assign(lev=np.where(hit, "A", lev))
    reference = fit(healthy, np.where(hit, price, 0.0))
    assert model._reml_result.lambdas["grp"] == pytest.approx(
        reference._reml_result.lambdas["grp"], rel=1e-7
    )
    assert model._reml_result.objective == pytest.approx(
        reference._reml_result.objective, abs=1e-7, rel=0
    )
    np.testing.assert_allclose(
        model.predict(frame),
        reference.predict(healthy, offset=np.where(hit, price, 0.0)),
        rtol=1e-7,
    )
    assert model.result.effective_df == pytest.approx(reference.result.effective_df, rel=1e-7)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
    model.fit_reml(healthy, y)
    assert not np.any(model.term_inference("lev").separated)
    np.testing.assert_allclose(model.predict(healthy), fit(healthy).predict(healthy), rtol=1e-12)


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_an_exact_presolve_holds_the_same_first_cold_contrast(monkeypatch, family):
    import superglm.reml.direct as direct

    monkeypatch.setattr(direct, "_PRESOLVE_MIN_ESTIMATED", 1)
    rng = np.random.default_rng(68)
    x = rng.uniform(size=1800)
    a = rng.choice(list("ABC"), len(x))
    b = rng.choice(list("PQ"), len(x))
    a[:18], b[12:24] = "Z", "R"  # Six rows carry both held contributions.
    if family == "binomial":
        y = rng.binomial(1, 0.45 + 0.25 * np.sin(5 * x))
    else:
        y = rng.poisson(np.exp(0.6 * np.sin(5 * x)))
    y[(a == "Z") | (b == "R")] = 0
    X = pd.DataFrame({"a": a, "b": b, "x": x})
    model = SuperGLM(
        family=family,
        selection_penalty=0,
        features={"a": Categorical(base="Z"), "b": Categorical(base="P"), "x": Spline(n_knots=5)},
    )
    model.fit_reml(X, y, max_reml_iter=1)
    a_price, b_price = _effect(model, "a", "A"), _effect(model, "b", "R")
    model.fit_reml(X, y, max_reml_iter=20)
    assert model._reml_profile["reml_presolve_n_iter"] > 0
    assert _effect(model, "a", "A") == a_price
    assert _effect(model, "b", "R") == b_price
    frame = X.assign(a=np.where(a == "Z", "A", a), b=np.where(b == "R", "P", b))
    offset = np.where(a == "Z", -a_price, 0.0) + np.where(b == "R", b_price, 0.0)
    oracle = SuperGLM(
        family=family,
        selection_penalty=0,
        features={"a": Categorical(base="A"), "b": Categorical(base="P"), "x": Spline(n_knots=5)},
    ).fit_reml(frame, y, offset=offset)
    np.testing.assert_allclose(model.predict(X), oracle.predict(frame, offset=offset), rtol=2e-6)


def test_logit_with_no_healthy_reference_keeps_the_cold_odds_ratio(monkeypatch):
    import superglm.reml.direct as direct

    rng = np.random.default_rng(45)
    x = rng.uniform(size=800)
    y = rng.binomial(1, 0.45, size=x.size)
    X = pd.DataFrame({"r": np.where(y, "events", "zeros"), "x": x})
    model = SuperGLM(
        family="binomial",
        selection_penalty=0,
        features={"r": Categorical(base="zeros"), "x": Spline(n_knots=4)},
    )
    solve = direct.fit_irls_direct
    cold = []

    def record(**kw):
        out = solve(**kw)
        if (kw.get("debug_context") or {}).get("phase") == "bootstrap":
            cold.append(out[0].beta[0])
        return out

    monkeypatch.setattr(direct, "fit_irls_direct", record)
    model.fit_reml(X, y, max_reml_iter=1)
    assert _effect(model, "r", "events") == cold[0]
    model.fit_reml(X, y, max_reml_iter=20)
    assert _effect(model, "r", "events") == cold[0]
    assert model.term_inference("r").separated.all()
    probabilities = model.predict(X)
    assert np.all((probabilities > 0) & (probabilities < 1))


@pytest.mark.parametrize("route", ["smooth", "ordinary", "fixed_qp"])
def test_power_profile_publishes_the_separated_price_at_its_selected_power(route):
    from superglm import BSplineSmooth, Constraint, LambdaPolicy, Numeric

    rng = np.random.default_rng(240)
    n = 800
    x = rng.uniform(size=n)
    r = np.where(np.arange(n) < 8, "Z", "A")
    count = rng.poisson(np.exp(np.sin(6 * x)))
    y = rng.gamma(np.maximum(count, 1), 0.8)
    y[(count == 0) | (r == "Z")] = 0
    X = pd.DataFrame({"r": r, "x": x})

    def model(p):
        if route == "ordinary":
            x_term = Numeric()
        elif route == "fixed_qp":
            x_term = BSplineSmooth(
                n_knots=4,
                constraint=Constraint.fit.increasing,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        else:
            x_term = Spline(n_knots=4)
        return SuperGLM(
            family=Tweedie(p=p),
            selection_penalty=0,
            features={"r": Categorical(base="A"), "x": x_term},
        )

    m = model(1.5)
    result = m.estimate_p(X, y, fit_mode="reml", p_bounds=(1.3, 1.7), xatol=0.03)
    direct = model(result.p_hat).fit_reml(X, y)
    assert _effect(m, "r", "Z") == _effect(direct, "r", "Z")
    assert m.result.effective_df == pytest.approx(direct.result.effective_df)


def test_power_profile_with_a_zero_level_matches_same_knot_deletion():
    from superglm import PSpline

    rng = np.random.default_rng(123)
    x = rng.uniform(0, 10, 2000)
    r = rng.choice(list("ABC"), len(x), p=[0.4, 0.4, 0.2])
    r[:80] = "Z"
    p, phi = 1.85, 1.5
    mu = np.exp(0.5 + 0.4 * np.sin(x) + 0.5 * (r == "B") - 0.3 * (r == "C"))
    count = rng.poisson(mu ** (2 - p) / (phi * (2 - p)))
    pos = count > 0
    y = np.zeros_like(x)
    y[pos] = rng.gamma(count[pos] * (2 - p) / (p - 1), phi * (p - 1) * mu[pos] ** (p - 1))
    y[r == "Z"] = 0
    X = pd.DataFrame({"x": x, "r": r})

    def profile(frame, response, spline, lower_bound):
        model = SuperGLM(family=Tweedie(p=1.5), features={"x": spline, "r": Categorical(base="A")})
        fit = model.estimate_p(
            frame, response, fit_mode="reml", p_bounds=(lower_bound, 1.95), xatol=0.005
        )
        return model, fit

    full, fit = profile(X, y, PSpline(n_knots=8), 1.05)
    keep = r != "Z"
    spec = full._specs["x"]
    knots = {"knots": spec.fitted_knots, "boundary": spec.fitted_boundary}
    deleted, reference = profile(X.loc[keep], y[keep], PSpline(**knots), 1.05)
    narrow, narrow_fit = profile(X, y, PSpline(**knots), 1.60)
    for model, result in ((full, fit), (narrow, narrow_fit)):
        # Within the power search tolerance, including the likelihood-ratio interval.
        np.testing.assert_allclose(
            [result.p_hat, *result.ci(0.05)], [reference.p_hat, *reference.ci(0.05)], atol=0.005
        )
        assert result.phi_hat == pytest.approx(reference.phi_hat, rel=0.005)
        assert model._reml_lambdas["x"] == pytest.approx(deleted._reml_lambdas["x"], rel=0.02)


def test_a_qp_refit_keeps_the_separated_price(monkeypatch):
    import superglm.model.reml_finalize as final
    from superglm import BSplineSmooth, Constraint

    rng = np.random.default_rng(12)
    x = rng.uniform(size=1200)
    r = np.where(np.arange(x.size) < 9, "Z", "A")
    y = rng.poisson(0.5 + x)
    y[r == "Z"] = 0
    X = pd.DataFrame({"r": r, "x": x})
    qp_offsets = []
    solve = final.fit_irls_direct

    def record(**kw):
        if kw.get("trace_purpose") == "reml_qp_final":
            qp_offsets.append(kw["offset"].copy())
        return solve(**kw)

    monkeypatch.setattr(final, "fit_irls_direct", record)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={
            "r": Categorical(base="A"),
            "x": BSplineSmooth(n_knots=5, constraint=Constraint.fit.increasing),
        },
    )
    model.fit_reml(X, y, max_reml_iter=1)
    price = _effect(model, "r", "Z")
    np.testing.assert_array_equal(qp_offsets[-1], np.where(r == "Z", price, 0.0))
    model.fit_reml(X, y)
    assert _effect(model, "r", "Z") == price
    ref = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={"x": BSplineSmooth(n_knots=5, constraint=Constraint.fit.increasing)},
    ).fit_reml(X[["x"]], y, offset=np.where(r == "Z", price, 0.0))
    assert model._reml_result.objective == pytest.approx(ref._reml_result.objective, rel=2e-8)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_fixed_lambda_tweedie_keeps_the_rows_in_the_profile(semantics):
    from superglm import LambdaPolicy

    rng = np.random.default_rng(45)
    n = 420
    x = rng.uniform(size=n)
    r = np.where(np.arange(n) < 30, "Z", "A")
    weights = rng.choice([1.0, 2.0], n)
    counts = rng.poisson(0.6 + x)
    y = rng.gamma(np.maximum(counts, 1), 0.8)
    y[(counts == 0) | (r == "Z")] = 0
    X = pd.DataFrame({"r": r, "x": x})
    for lam in (0.5, 20.0):

        def fit(features, offset=None):
            features["x"] = Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(lam))
            return SuperGLM(
                family=Tweedie(p=1.4),
                selection_penalty=0,
                weight_semantics=semantics,
                features=features,
            ).fit_reml(X, y, sample_weight=weights, offset=offset)

        m = fit({"r": Categorical(base="A")})
        held = _effect(m, "r", "Z")
        offset = np.where(r == "Z", held, 0.0)
        reference = fit({}, offset)
        assert m._reml_result.objective == pytest.approx(reference._reml_result.objective, rel=1e-9)
        assert m.result.phi_likelihood == pytest.approx(reference.result.phi_likelihood, rel=1e-7)
        assert m.result.effective_df == pytest.approx(reference.result.effective_df, rel=1e-7)
        np.testing.assert_allclose(m.predict(X), reference.predict(X, offset=offset), rtol=1e-7)


def test_auto_theta_uses_the_cold_reml_price_at_the_published_theta(monkeypatch):
    import superglm.reml.direct as direct

    rng = np.random.default_rng(25)
    x = rng.uniform(size=1600)
    r = np.where(np.arange(x.size) < 18, "Z", "A")
    mu = np.exp(np.sin(6 * x))
    y = rng.negative_binomial(2.0, 2.0 / (2 + mu)).astype(float)
    y[r == "Z"] = 0.0
    X = pd.DataFrame({"r": r, "x": x})
    cold = []
    solve = direct.fit_irls_direct

    def record(**kwargs):
        out = solve(**kwargs)
        if (kwargs.get("debug_context") or {}).get("phase") == "bootstrap":
            cold.append(out[0].beta[0])
        return out

    monkeypatch.setattr(direct, "fit_irls_direct", record)
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        selection_penalty=0,
        features={"r": Categorical(base="A"), "x": Spline(n_knots=5)},
        spline_penalty=1e4,
    ).fit_reml(X, y)
    assert len(cold) > 1
    assert _effect(model, "r", "Z") == cold[-1]
