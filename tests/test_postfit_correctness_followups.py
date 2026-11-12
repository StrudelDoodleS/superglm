"""Regressions for the boundary, continuation and metric follow-ups to #626."""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import special

from superglm import Constraint, GammaLS, GaussianLS, PSpline, SuperGLM, SuperLSS, s
from superglm.constraints import CurvatureRepairer, MonotoneRepairer
from superglm.distributions import Poisson, clip_mu
from superglm.links import IdentityLink, stabilize_eta
from superglm.model import shape_ops
from superglm.solvers.mode_score import linear_predictor
from tests._shape_projection_oracle import projection_certificate


def _deviance(model, eta, y):
    mu = clip_mu(model._link.inverse(stabilize_eta(eta, model._link)), model._distribution)
    return float(np.sum(model._distribution.deviance_unit(y, mu)))


@pytest.mark.parametrize("zero_weight", [1.0, 2.0])
def test_a_poisson_identity_intercept_can_minimize_at_a_zero_count_boundary(zero_weight):
    # At shift -.1, the zero row's left score is zero, its right score -w.
    # The other row's score is 2/1.4-1, strictly between 0 and w. Thus the
    # one-sided derivatives bracket zero, although neither smooth score is zero.
    model = SimpleNamespace(_distribution=Poisson(), _link=IdentityLink())
    eta, y, weights = np.array([0.1, 1.5]), np.array([0.0, 2.0]), np.array([zero_weight, 1.0])

    profiled, shift, _, deviance = shape_ops._profile_repaired_intercept(
        model, eta, y=y, weights=weights
    )

    assert shift == pytest.approx(-0.1, abs=1e-14)
    np.testing.assert_allclose(profiled, [0.0, 1.4], atol=1e-14)
    expected = 2.0 * (1.4 - 2.0 + 2.0 * np.log(2.0 / 1.4))
    assert deviance == pytest.approx(expected, rel=1e-12)
    for h in (-1e-5, 1e-5):
        mu = clip_mu(profiled + h, model._distribution)
        assert np.dot(weights, model._distribution.deviance_unit(y, mu)) > deviance


@pytest.mark.parametrize("kind", ["increasing", "decreasing", "convex", "concave"])
@pytest.mark.parametrize(("seed", "origin", "slope"), [(18, 0.1, 2.0), (9, 0.05, 1.5)])
def test_a_poisson_identity_step_repair_publishes_its_boundary_intercept(kind, seed, origin, slope):
    x = np.linspace(0.0, 1.0, 200)
    X, offset = pd.DataFrame({"x": x}), origin + slope * x
    mean = np.maximum(offset + 0.8 * np.tanh((x - 0.5) / 0.05), 0.01)
    y = np.random.default_rng(seed).poisson(mean).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        link="identity",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=6, constraint=getattr(Constraint.postfit, kind))},
    ).fit(X, y, offset=offset)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.apply_shape_postfit(X, offset=offset)

    eta = linear_predictor(model._dm, model._solver_result, offset)
    published = _deviance(model, eta, y)
    assert "x" in model._shape_repairs
    for h in (-1e-5, 1e-5):
        assert _deviance(model, eta + h, y) >= published - 1e-10 * abs(published)


def test_a_poisson_identity_intercept_can_minimize_just_beyond_a_zero_count_corner():
    # The zero row has information ~1/.0011 at the minimum, but its actual
    # deviance curvature is zero. Fisher scoring exhausts its steps at 93e82a.
    model = SimpleNamespace(_distribution=Poisson(), _link=IdentityLink())
    eta, y = np.array([0.1, 1.5989]), np.array([0.0, 3.0])
    profiled, shift, _, deviance = shape_ops._profile_repaired_intercept(
        model, eta, y=y, weights=np.ones(2)
    )
    assert shift == pytest.approx(-0.0989, abs=1e-11)
    np.testing.assert_allclose(profiled, [0.0011, 1.5], atol=1e-11)
    assert deviance == pytest.approx(0.0022 + 6.0 * np.log(2.0) - 3.0, rel=1e-11)


@pytest.mark.parametrize(("link", "shape"), [("log", "bump"), ("identity", "sin")])
def test_a_failed_candidate_intercept_continues_from_the_flat_binomial_curve(link, shape):
    x = np.linspace(0.0, 1.0, 200)
    rng = np.random.default_rng(18)
    f = (np.exp(-(((x - 0.3) / 0.12) ** 2)) - 0.4) if shape == "bump" else np.sin(12.0 * x)
    offset = -0.05 - 1.2 * x if link == "log" else 0.92 - 0.6 * x
    mean = (
        np.exp(np.minimum(offset + 0.6 * f, -1e-3))
        if link == "log"
        else np.clip(offset + 0.3 * f, 1e-3, 1.0 - 1e-3)
    )
    y = rng.binomial(1, mean).astype(np.float64)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="binomial",
        link=link,
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=6 if link == "log" else 10, constraint=Constraint.postfit.concave)
        },
    ).fit(X, y, offset=offset)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.apply_shape_postfit(X, offset=offset)

    eta = linear_predictor(model._dm, model._solver_result, offset)
    # The base publishes the flat curve at 237.426 / 219.221. Both have
    # rule-obeying curves that improve it; a failed profile is not optimality.
    flat_deviance = 237.42604507637324 if link == "log" else 219.22094172727157
    assert np.any(model.result.beta)
    assert _deviance(model, eta, y) < 0.995 * flat_deviance


@pytest.mark.parametrize(("degree", "kind"), [(0, "increasing"), (1, "convex"), (1, "concave")])
def test_an_uncertified_one_solve_projection_publishes_feasible_with_disclosure(
    monkeypatch, degree, kind
):
    from superglm import constraints

    x = np.linspace(0.0, 1.0, 100)
    y = -x if degree == 0 else (-1.0 if kind == "convex" else 1.0) * (x - 0.4) ** 2
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                n_knots=6,
                degree=degree,
                m=1,
                penalty="none",
                constraint=getattr(Constraint.postfit, kind),
            )
        },
    ).fit(X, y)
    groups = [g for g in model._groups if g.feature_name == "x"]
    repairer = MonotoneRepairer(kind) if degree == 0 else CurvatureRepairer(kind)
    # Force certification failure on every BLAS build, including the feasible
    # retry. The apex is always feasible and its distance bound is the zero loss.
    monkeypatch.setattr(constraints, "_solve_cone_qp", lambda *a, **k: None)

    with pytest.warns(RuntimeWarning, match="stopped before it certified the closest"):
        result = repairer.repair(model._specs["x"], model.result.beta, groups)

    assert np.all(np.isfinite(result.repaired_beta_reparam))
    assert not np.any(result.repaired_beta_reparam)


def test_a_structurally_singular_public_grid_metric_discloses_its_uncertified_distance():
    x = np.linspace(0.0, 1.0, 200)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                n_knots=5, degree=7, m=1, penalty="none", constraint=Constraint.postfit.increasing
            )
        },
    ).fit(pd.DataFrame({"x": x}), 0.9 * x + 0.25 * np.sin(12.0 * x))
    beta = np.random.default_rng(0).normal(size=model.result.beta.size)
    groups = [g for g in model._groups if g.feature_name == "x"]

    with pytest.warns(RuntimeWarning, match="rank-deficient.*grid|grid.*rank-deficient"):
        repair = MonotoneRepairer("increasing").repair(model._specs["x"], beta, groups, n_grid=7)

    assert np.all(np.isfinite(repair.repaired_log_effect))


def test_a_concentrated_weight_lss_projection_uses_the_centered_metric():
    x = np.linspace(0.0, 1.0, 100)
    y = 0.9 * x + 0.25 * np.sin(12.0 * x)
    weights = np.full(x.size, 1e-15)
    weights[50] = 1.0
    X = pd.DataFrame({"x": x, "zero": np.zeros_like(x)})
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family,
        family.location(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale("zero", intercept=False),
    ).fit(X, y, sample_weight=weights, lambdas={"location:x#wiggle": 0.0})
    fitted = model._model
    spec = next(p for p in fitted.compiled_predictors if p.name == "location").compiled.specs["x"]
    columns = fitted.layout.term_slices["location:x"]
    before = fitted.coefficients[columns].copy()
    basis = spec.transform(x)
    centered = basis - (weights @ basis) / weights.sum()
    H = centered.T @ (weights[:, None] * centered)

    model.apply_shape_postfit()

    after = model._model.coefficients[columns]
    certificate = projection_certificate(spec, "increasing", after, H, H @ before, before)
    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= 5e-8


def test_two_parameter_lss_repairs_settle_both_intercepts_on_the_final_likelihood():
    rng = np.random.default_rng(1240)
    x = rng.uniform(0.0, 1.0, 500)

    def wiggle(t):
        return 0.9 * t + 0.25 * np.sin(12.0 * t)

    mean = np.exp(0.5 + 0.8 * wiggle(x))
    cv = np.exp(np.log(0.4) + 0.5 * wiggle(1.0 - x))
    y = rng.gamma(1.0 / cv**2, mean * cv**2)
    X = pd.DataFrame({"x": x})
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale(s("x", kind="bs", k=6, constraint=Constraint.postfit.decreasing)),
    ).fit_reml(X, y)
    fitted = model._model

    model.apply_shape_postfit()

    theta = model.predict_parameters(X)
    mu, a = theta["mean"].to_numpy(), 1.0 / theta["scale"].to_numpy() ** 2
    # Independent Gamma likelihood derivatives in log mean and log CV.
    score_mean = a * (y / mu - 1.0)
    score_scale = (
        -2.0 * a * (np.log(a) + 1.0 - np.log(mu) + np.log(y) - y / mu - special.digamma(a))
    )
    for score in (score_mean, score_scale):
        assert abs(score.sum()) <= 1e-6 * np.abs(score).sum()
    records = model._model.fit_state.hand_edit.repairs
    assert len(records) == 2
    for record in records:
        parameter = record.term.split(":", 1)[0]
        index = fitted.layout.predictor(parameter).intercept_index
        assert record.intercept_shift == pytest.approx(
            model._model.coefficients[index] - fitted.coefficients[index], abs=1e-12
        )


def test_joint_intercept_settling_discloses_exhaustion_and_keeps_its_last_estimates(monkeypatch):
    from superglm.distributional import shape_repair

    fitted = SimpleNamespace(
        layout=SimpleNamespace(
            predictor=lambda name: SimpleNamespace(intercept_index={"mean": 0, "scale": 1}[name])
        )
    )
    monkeypatch.setattr(shape_repair, "_NEWTON_STEPS", 2)
    monkeypatch.setattr(shape_repair, "_profiled_merit", lambda *a: (0.1, 1.0, 0))
    with pytest.warns(RuntimeWarning, match="did not settle jointly.*published"):
        beta = shape_repair._settled_intercepts(fitted, None, None, np.zeros(2), ("mean", "scale"))
    np.testing.assert_allclose(beta, [0.2, 0.2])


@pytest.mark.parametrize("source", ["observed", "fisher"])
def test_centered_lss_curvature_follows_the_requested_source_at_revised_coefficients(
    monkeypatch, source
):
    from superglm.distributional import shape_repair

    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = np.random.default_rng(3).gamma(2.0, np.exp(0.5 + 0.4 * np.sin(12 * x)) / 2.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale(),
    ).fit(X, y, lambdas={"mean:x#wiggle": 1.0})
    fitted = model._model
    (term,) = shape_repair.postfit_terms(fitted)
    beta = fitted.coefficients.copy()
    own = fitted.layout.predictor("mean").intercept_index
    beta[own] += 0.2
    beta[fitted.layout.predictor("scale").intercept_index] += 0.1
    theta = model.predict_parameters(X)
    mu = theta["mean"].to_numpy() * np.exp(0.2)
    a = np.exp(-0.2) / theta["scale"].to_numpy() ** 2
    information = a * y / mu if source == "observed" else a
    basis = term.curve_spec.transform(x)
    centered = basis - (information @ basis) / information.sum()
    penalty = fitted.result.penalty[term.columns, term.columns]
    expected = centered.T @ (information[:, None] * centered) + penalty
    curvature = fitted.result.terminal_penalized_curvature.copy()
    columns = np.arange(term.columns.start, term.columns.stop)
    coupling = information @ basis
    curvature[own, own] = information.sum()
    curvature[columns, own] = coupling
    curvature[own, columns] = coupling
    curvature[np.ix_(columns, columns)] = basis.T @ (information[:, None] * basis) + penalty
    monkeypatch.setattr(shape_repair, "_profile_cancels", lambda *a: True)

    actual, _ = shape_repair._profiled_curvature(
        fitted, term, curvature, coefficients=beta, source=source
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.linalg.norm(expected))


def test_a_stored_lss_metric_is_replayed_at_the_fit_after_a_hand_edit(monkeypatch):
    from superglm.distributional import shape_repair

    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = np.random.default_rng(4).gamma(2.0, np.exp(0.5 + 0.4 * np.sin(12 * x)) / 2.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
        family.scale(),
    ).fit(X, y, lambdas={"mean:x#wiggle": 1.0})
    theta = model.predict_parameters(X)
    information = y / theta["mean"].to_numpy() / theta["scale"].to_numpy() ** 2
    fitted = model._model
    (term,) = shape_repair.postfit_terms(fitted)
    basis = term.curve_spec.transform(x)
    centered = basis - (information @ basis) / information.sum()
    expected = centered.T @ (information[:, None] * centered)
    expected += fitted.result.penalty[term.columns, term.columns]
    beta = fitted.coefficients.copy()
    beta[fitted.layout.predictor("mean").intercept_index] += 0.4
    revised = model._with_revised_coefficients(beta, edited_terms=["mean:x"])
    # Force cancellation on every platform, as concentrated weights do. This
    # is the default metric used when curvature_at rejects an edited state.
    monkeypatch.setattr(shape_repair, "_profile_cancels", lambda *a: True)

    actual = shape_repair._term_precision(revised._model, term)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.linalg.norm(expected))


def test_joint_intercept_settling_records_each_parameters_extra_shift_once(monkeypatch):
    from superglm.distributional import shape_repair

    rng = np.random.default_rng(27)
    x, z = rng.uniform(size=(2, 200))
    X = pd.DataFrame({"x": x, "z": z})
    y = rng.gamma(3.0, np.exp(1.0 - x - z) / 3.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(
            s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing),
            s("z", kind="bs", k=6, constraint=Constraint.postfit.increasing),
        ),
        family.scale(s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing)),
    ).fit(X, y, lambdas={"mean:x#wiggle": 1.0, "mean:z#wiggle": 1.0, "scale:x#wiggle": 1.0})
    fitted = model._model
    beta = fitted.coefficients.copy()
    for term in shape_repair.postfit_terms(fitted):
        values = X[term.label].to_numpy()
        beta[term.columns] = np.linalg.lstsq(
            term.curve_spec.transform(values), 0.5 - values, rcond=None
        )[0]
    revised = model._with_revised_coefficients(beta, edited_terms=["mean:x", "mean:z", "scale:x"])

    def extra_shift(fitted, rows, frame, coefficients, parameters):
        result = coefficients.copy()
        for parameter in parameters:
            result[fitted.layout.predictor(parameter).intercept_index] += 0.1
        return result

    monkeypatch.setattr(shape_repair, "_settled_intercepts", extra_shift)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        revised.apply_shape_postfit()

    records = revised._model.fit_state.hand_edit.repairs
    assert len(records) == 3
    for parameter in ("mean", "scale"):
        index = fitted.layout.predictor(parameter).intercept_index
        total = sum(r.intercept_shift for r in records if r.term.startswith(parameter + ":"))
        assert total == pytest.approx(revised._model.coefficients[index] - beta[index], abs=1e-12)
