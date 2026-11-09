"""Public complete-composition lifecycle, independent of the family kernel."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import KFold, RepeatedKFold

from superglm import DirichletLSS, SuperLSS, cross_validate
from superglm.distributional.checks.scores import log_score, threshold_weighted_crps

from ._dirichlet_lifecycle_reference import (
    cardinal,
    natural_constant,
    objective,
    spline_profile,
    stationary,
)


def book(n=90):
    rng = np.random.default_rng(6021)
    x = np.linspace(-1, 1, n)
    X = pd.DataFrame({"x": x, "o": 0.12 * x}, index=pd.Index(np.arange(n) + 200, name="row"))
    alpha = np.exp(np.array([0.3, 0.8, 1.2]) + x[:, None] * [-0.2, 0.3, 0.1])
    y = pd.DataFrame(
        [rng.dirichlet(a) for a in alpha], columns=("red", "green", "blue"), index=X.index
    )
    return X, y


def estimator(*, components=("red", "green", "blue"), **kwargs):
    family = DirichletLSS(components)
    return SuperLSS(
        family,
        *(family.alpha(c, "x", offset="o") for c in components),
        weight_semantics="frequency",
        **kwargs,
    )


@pytest.fixture(scope="module")
def fitted():
    X, y = book()
    return estimator().fit(X, y), X, y


def test_public_mean_labels_index_and_alpha_order(fitted):
    model, X, _ = fitted
    alpha = model.predict_parameters(X)
    mean = model.predict(X)
    assert isinstance(mean, pd.DataFrame)
    assert tuple(mean.columns) == model.family.components
    assert tuple(alpha.columns) == ("alpha_0", "alpha_1", "alpha_2")
    pd.testing.assert_index_equal(mean.index, X.index)
    np.testing.assert_allclose(
        mean, alpha.to_numpy() / alpha.to_numpy().sum(axis=1)[:, None], rtol=2e-15
    )
    np.testing.assert_allclose(mean.sum(axis=1), 1, atol=5e-16)
    assert (mean > 0).all().all()
    assert not model._model.predict(X).flags.writeable


def test_joint_log_scores_canonicalize_before_retained_conversion(fitted):
    model, X, y = fitted
    weight = np.arange(len(X)) % 3 + 1.0
    weight[4] = 0
    raw = y.astype(object)
    raw.iloc[4] = ["dropped", None, True]
    result = model.scores(X, raw[["blue", "red", "green"]], which=("log",), sample_weight=weight)
    alpha = model.predict_parameters(X).to_numpy()
    want = -np.array([stats.dirichlet.logpdf(v, a) for v, a in zip(y.to_numpy(), alpha)]) * weight
    assert result.shape == (len(X), 1)
    assert np.isnan(result.iloc[4, 0])
    np.testing.assert_allclose(result.log[weight > 0], want[weight > 0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        log_score(model._model, X, y)[weight > 0],
        (want / np.where(weight, weight, 1))[weight > 0],
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("problem", ["index", "columns", "bool", "string", "sum"])
def test_scoring_checks_original_identity_and_retained_values(fitted, problem):
    model, X, y = fitted
    y = y.astype(object)
    if problem == "index":
        y.index = y.index[::-1]
    elif problem == "columns":
        y.columns = ["red", "green", "green"]
    elif problem == "bool":
        y.iloc[0, 0] = True
    elif problem == "string":
        y.iloc[0, 0] = str(y.iloc[0, 0])
    else:
        y.iloc[0, 0] += 0.05
    with pytest.raises(ValueError):
        model.scores(X, y, which=("log",))


def test_cv_named_response_oof_and_default_joint_nll():
    X, y = book()
    cv = cross_validate(
        estimator(),
        X,
        y[["blue", "green", "red"]],
        cv=KFold(3),
        fit_mode="fit",
        return_oof=True,
        return_estimators=True,
        error_score="raise",
    )
    assert set(cv.mean_scores) == {"nll"}
    assert cv.oof_predictions.shape == y.shape
    total = 0.0
    for model, (_, rows) in zip(cv.estimators, cv.fold_indices):
        expected = model.predict(X.iloc[rows])
        np.testing.assert_allclose(cv.oof_predictions[rows], expected)
        total += model.scores(X.iloc[rows], y.iloc[rows], which=("log",)).log.sum()
    assert cv.pooled_scores["nll"] == pytest.approx(total / len(X))


@pytest.mark.parametrize("problem", ["index", "columns"])
def test_cv_checks_original_labels_before_any_fit(problem, monkeypatch):
    X, y = book()
    if problem == "index":
        y.index = y.index[::-1]
    else:
        y.columns = ["red", "green", "green"]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid original response labels reached a training fit")

    monkeypatch.setattr(SuperLSS, "fit", forbidden)
    with pytest.raises(ValueError, match=problem):
        cross_validate(estimator(), X, y, cv=KFold(2), fit_mode="fit", error_score="raise")


@pytest.mark.parametrize("name", ["nll", "crps", "actual_expected"])
def test_custom_scalar_score_named_like_builtin_keeps_its_own_meaning(name):
    X, y = book()

    def share_error(model, X, y, *, sample_weight, offset):
        return np.mean((model.predict(X, offsets=offset).iloc[:, 0].to_numpy() - y[:, 0]) ** 2)

    share_error.__name__ = name
    result = cross_validate(
        estimator(),
        X,
        y,
        cv=KFold(2),
        fit_mode="fit",
        scoring=share_error,
        return_estimators=True,
        error_score="raise",
    )
    for i, (model, (_, rows)) in enumerate(zip(result.estimators, result.fold_indices)):
        expected = np.mean((model.predict(X.iloc[rows]).iloc[:, 0] - y.iloc[rows, 0]) ** 2)
        assert result.fold_scores[name].iloc[i] == pytest.approx(expected)
    assert result.builtin_scores == ()
    assert result.pooled_scores == {}


@pytest.mark.parametrize("repeats", [1, 2])
def test_cv_drops_raw_invalid_zero_rows_and_counts_unpriced_rows(repeats):
    X, y = book()
    weight = np.ones(len(X))
    weight[3] = 0
    X.loc[X.index[3], "x"] = np.nan
    y = y.astype(object)
    y.iloc[3] = [None, "drop", True]
    with pytest.warns(UserWarning, match="1 held-out rows of weight zero") as caught:
        cv = cross_validate(
            estimator(),
            X,
            y,
            cv=RepeatedKFold(n_splits=3, n_repeats=repeats, random_state=31),
            sample_weight=weight,
            fit_mode="fit",
            return_oof=True,
            error_score="raise",
        )
    assert np.isnan(cv.oof_predictions[3]).all()
    assert np.isfinite(cv.oof_predictions[weight > 0]).all()
    reports = [item for item in caught if "could not be priced" in str(item.message)]
    assert len(reports) == 1


def test_paired_named_log_comparison_and_positional_order_refusal(fitted):
    model, X, y = fitted
    other = estimator(components=("blue", "red", "green")).fit(X, y)
    comparison = model.compare(other, X, y[["green", "blue", "red"]])
    assert abs(comparison.overall["mean_diff"]) < 1e-8
    with pytest.raises(ValueError, match="component.*order"):
        model.compare(other, X, y.to_numpy())

    different = estimator(components=("red", "green", "yellow")).fit(
        X, y.rename(columns={"blue": "yellow"})
    )
    with pytest.raises(ValueError, match="same component set"):
        model.compare(different, X, y)


@pytest.mark.parametrize(
    "operation,call",
    [
        ("CDF", lambda m, X, y: m.predict_cdf(X, y)),
        ("quantile", lambda m, X, y: m.predict_quantile(X, 0.5)),
        ("residual", lambda m, X, y: m.residuals(X, y)),
        ("residual", lambda m, X, y: m.residuals(X, y, kind="pit")),
        ("check", lambda m, X, y: m.check(X, y, "x")),
        ("check_2d", lambda m, X, y: m.check_2d(X, y, "x", "o")),
        ("actual_expected", lambda m, X, y: m.actual_expected(X, y, "x")),
        ("calibration", lambda m, X, y: m.calibration(X, y)),
        ("CRPS", lambda m, X, y: m.scores(X, y)),
        ("CRPS", lambda m, X, y: m.compare(m, X, y, which="crps")),
        ("threshold", lambda m, X, y: m.scores(X, y, which=("log",), thresholds=(0.2,))),
        ("threshold", lambda m, X, y: threshold_weighted_crps(m._model, X, y, 0.2)),
        ("Murphy", lambda m, X, y: m.compare(m, X, y, murphy_quantile=0.5)),
        ("share callable", lambda m, X, y: m.posterior_bounds(X, "mean")),
        ("cdf", lambda m, X, y: m.posterior_bounds(X, quantity=("cdf", 0.3))),
        ("quantile", lambda m, X, y: m.posterior_bounds(X, ("quantile", 0.5))),
        ("exceedance", lambda m, X, y: m.posterior_bounds(X, ("exceedance", 0.3))),
        ("expected_shortfall", lambda m, X, y: m.posterior_bounds(X, ("expected_shortfall", 0.9))),
        ("posterior_predictive", lambda m, X, y: m.posterior_predictive(X)),
        ("portfolio", lambda m, X, y: m.portfolio(X)),
        ("risk_curves", lambda m, X, y: m.risk_curves({}, "x")),
        ("density_fan", lambda m, X, y: m.density_fan({}, "x")),
        ("parameter_spread", lambda m, X, y: m.parameter_spread(X, threshold=0.2)),
        ("plot_diagnostics", lambda m, X, y: m.plot_diagnostics(X, y)),
        ("qq", lambda m, X, y: m.plot_data("qq", X=X, y=y)),
        ("worm", lambda m, X, y: m.plot_data("worm", X=X, y=y)),
        ("pit", lambda m, X, y: m.plot_data("pit", X=X, y=y)),
    ],
)
def test_scalar_operations_refuse_by_name(fitted, operation, call, monkeypatch):
    from superglm.distributional.model import DenseDistributionalModel

    model, X, y = fitted

    def no_prediction(*args, **kwargs):
        pytest.fail("scalar-only operation started evaluating vector predictions")

    monkeypatch.setattr(DenseDistributionalModel, "predict_parameters", no_prediction)
    monkeypatch.setattr(DenseDistributionalModel, "_predict_parameters_resolved", no_prediction)
    with pytest.raises(NotImplementedError, match=operation):
        call(model, X, y)


@pytest.mark.parametrize("score", ["crps", "gini", "actual_expected"])
def test_explicit_scalar_cv_scores_refuse_before_any_fit(fitted, score, monkeypatch):
    model, X, y = fitted

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported scorer reached a training fit")

    monkeypatch.setattr(SuperLSS, "fit", forbidden)
    with pytest.raises(NotImplementedError, match=score):
        cross_validate(model, X, y, cv=KFold(2), scoring=score, fit_mode="fit", error_score="raise")


def test_vector_editor_refuses_before_opening(fitted):
    from superglm.editor import edit

    model, X, y = fitted
    with pytest.raises(NotImplementedError, match="editor"):
        edit(model, train_data=(X, y))


def test_natural_alpha_intercept_reference_and_joint_share_uncertainty():
    from superglm.distributional.posterior import posterior_parameters

    X, y = book(180)
    weight = np.arange(len(X)) % 3 + 30.0
    family = DirichletLSS(tuple(y.columns))
    model = SuperLSS(
        family, *(family.alpha(c) for c in family.components), weight_semantics="frequency"
    ).fit(X, y, sample_weight=weight)
    alpha, ref = natural_constant(y.to_numpy(), weight)
    actual = model.predict_parameters(X).iloc[0].to_numpy()
    assert np.linalg.norm(ref(actual)[1], ord=np.inf) / weight.sum() < 1e-7
    np.testing.assert_allclose(actual, alpha, rtol=3e-7)
    np.testing.assert_allclose(-model.result_.log_likelihood, ref(alpha)[0], rtol=2e-12)
    natural_cov = np.linalg.inv(ref(alpha)[2])
    eta_cov = natural_cov / np.outer(alpha, alpha)
    np.testing.assert_allclose(model.covariance_, eta_cov, rtol=2e-6)
    assert model._model.inference.rank == 3
    assert model.result_.total_effective_df == pytest.approx(3)
    _, se = model.predict_link(X.iloc[:2], se_fit=True)
    np.testing.assert_allclose(se.iloc[0], np.sqrt(np.diag(eta_cov)), rtol=2e-6)

    mu = alpha / alpha.sum()
    jac = (np.eye(3) - mu[:, None]) / alpha.sum()
    want = jac @ natural_cov @ jac.T
    draws = model.posterior_draws(24000, seed=62)
    _, parameters = next(posterior_parameters(model._model, X.iloc[:1], draws))
    shares = parameters[:, 0] / parameters[:, 0].sum(axis=1, keepdims=True)
    # MC error is O(1/sqrt(draws)); covariance is small enough for the delta approximation.
    np.testing.assert_allclose(
        np.cov(shares.T), want, rtol=0.07, atol=0.025 * np.max(np.diag(want))
    )
    assert np.max(np.abs(eta_cov - np.diag(np.diag(eta_cov)))) > 0.2 * np.max(np.diag(eta_cov))
    functions = [
        lambda a: a[:, 0] / a.sum(axis=1),
        lambda a: a[:, :2].sum(axis=1) / a.sum(axis=1),
        lambda a: np.log(a[:, 0] / a[:, 2]),
        lambda a: a.sum(axis=1),
    ]
    for fun in functions:
        bounds, values = model.posterior_bounds(X.iloc[:2], fun, draws=draws, return_draws=True)
        np.testing.assert_allclose(bounds.estimate, fun(np.tile(alpha, (2, 1))), rtol=4e-7)
        assert values.shape == (draws.n_draws, 2)
        assert (bounds.sd > 0).all()
    with pytest.raises(ValueError, match="one value per row"):
        model.posterior_bounds(X.iloc[:2], lambda a: a / a.sum(axis=1)[:, None], draws=draws)


def smooth_case(*, discrete=False, retain=True, reml=False, order=("red", "green", "blue")):
    from superglm import s

    rng = np.random.default_rng(606)
    x = np.tile(np.linspace(-1, 1, 31), 5)
    X = pd.DataFrame({"x": x, "o": 0.13 * x})
    truth = np.exp(
        np.column_stack(
            [0.5 + 0.7 * np.cos(3 * x), 0.8 - 0.9 * np.sin(4 * x), np.full(len(x), 1.1)]
        )
    )
    y = pd.DataFrame([rng.dirichlet(a) for a in truth], columns=("red", "green", "blue"))
    family = DirichletLSS(order)
    terms = {
        "red": (s("x", kind="cr", knots=[-0.5, 0, 0.5], penalty="none"),),
        "green": (s("x", kind="cr", knots=[-0.6, -0.2, 0.2, 0.6], penalty="none"),),
        "blue": (),
    }
    strengths = {"red": 0.3, "green": 0.8}
    model = SuperLSS(
        family,
        *(family.alpha(c, *terms[c], offset="o" if c == "red" else None) for c in order),
        discrete=discrete,
        n_bins=31,
        weight_semantics="frequency",
    )
    kwargs = dict(retain_rows=retain)
    if reml:
        model.fit_reml(X, y, max_reml_iter=30, outer="efs+newton", practical_reml=False, **kwargs)
    else:
        model.fit(
            X,
            y,
            lambdas={f"{family.component_parameters[c]}:x#wiggle": v for c, v in strengths.items()},
            **kwargs,
        )
    return model, X, y


@pytest.mark.parametrize("discrete,retain", [(False, True), (True, False)])
def test_distinct_spline_fixed_fit_against_independent_design_penalty(discrete, retain):
    model, X, y = smooth_case(discrete=discrete, retain=retain)
    knots = [np.array([-1, -0.5, 0, 0.5, 1]), np.array([-1, -0.6, -0.2, 0.2, 0.6, 1])]
    pairs = [cardinal(X.x.to_numpy(), k) for k in knots]
    designs = [pairs[0][0], pairs[1][0], np.ones((len(X), 1))]
    penalties = [0.3 * pairs[0][1], 0.8 * pairs[1][1], np.zeros((1, 1))]
    offsets = np.column_stack([X.o, np.zeros((len(X), 2))])
    ref = objective(y.to_numpy(), designs, penalties, offsets, np.ones(len(X)))
    beta = stationary(ref, 12)
    actual = np.concatenate(
        [
            model.predict_link(pd.DataFrame({"x": k, "o": np.zeros(len(k))})).iloc[:, j]
            for j, k in enumerate([*knots, np.array([0.0])])
        ]
    )
    assert np.linalg.norm(ref(actual)[1], ord=np.inf) / len(X) < 2e-7
    np.testing.assert_allclose(ref(actual)[0], ref(beta)[0], rtol=2e-12)
    alpha = np.exp(
        np.column_stack([d @ b for d, b in zip(designs, np.split(beta, [5, 11]))]) + offsets
    )
    np.testing.assert_allclose(model.predict_parameters(X), alpha, rtol=3e-6)
    assert model._model.result.execution_backend_identifier == (
        "distributional-chunked-v1" if discrete else "distributional-dense-v1"
    )
    assert (model._model.fit_state.retained_rows is not None) == retain
    covariance = np.linalg.inv(ref(beta)[2])
    _, se = model.predict_link(X.iloc[:1], se_fit=True)
    for j, (d, indices) in enumerate(zip(designs, (slice(0, 5), slice(5, 11), slice(11, 12)))):
        want = d[0] @ covariance[indices, indices] @ d[0]
        assert se.iloc[0, j] ** 2 == pytest.approx(want, rel=3e-6)
    penalty = np.zeros((12, 12))
    for p, indices in zip(penalties, (slice(0, 5), slice(5, 11), slice(11, 12))):
        penalty[indices, indices] = p
    assert model.result_.total_effective_df == pytest.approx(
        np.trace(np.eye(12) - covariance @ penalty), rel=3e-6
    )
    assert model._model.inference.rank == 12


def test_permuted_distinct_spline_specs_and_penalties():
    first, X, y = smooth_case()
    other, _, _ = smooth_case(order=("blue", "red", "green"))
    np.testing.assert_allclose(other.predict(X)[list(y.columns)], first.predict(X), rtol=3e-6)
    np.testing.assert_allclose(
        other.predict_parameters(X), first.predict_parameters(X).iloc[:, [2, 0, 1]], rtol=3e-6
    )
    assert first.result_.total_effective_df == pytest.approx(
        other.result_.total_effective_df, rel=2e-7
    )


def test_response_covariance_and_numerical_range(fitted):
    model, X, _ = fitted
    alpha = model.predict_parameters(X).to_numpy()
    cov = model.family.response_covariance(alpha)
    expected = np.array([stats.dirichlet.cov(a) for a in alpha])
    np.testing.assert_allclose(cov, expected, rtol=2e-15, atol=1e-17)
    np.testing.assert_allclose(cov.sum(axis=2), 0, atol=8e-17)
    assert np.linalg.eigvalsh(cov).min() >= -8e-17
    with pytest.raises(ValueError, match="underflow|range|positive"):
        model.family.default_prediction(np.array([[1e-300, 1e300, 1.0]]))


def test_linear_term_inference_and_saved_summary_use_joint_hessian(fitted):
    import pickle

    model, X, y = fitted
    design = np.column_stack([np.ones(len(X)), X.x])
    reference = objective(
        y.to_numpy(),
        [design] * 3,
        [np.zeros((2, 2))] * 3,
        np.tile(X.o.to_numpy()[:, None], (1, 3)),
        np.ones(len(X)),
    )
    beta = stationary(reference, 6)
    covariance = np.linalg.inv(reference(beta)[2])
    table = model.summary()
    assert set(table.parameter) == set(model.parameter_names_)
    for j, parameter in enumerate(model.parameter_names_):
        row = table[(table.parameter == parameter) & (table.term == "x")].iloc[0]
        assert row.estimate == pytest.approx(beta[2 * j + 1], rel=3e-6)
        assert row.se**2 == pytest.approx(covariance[2 * j + 1, 2 * j + 1], rel=3e-6)
        effect = model.term_inference(parameter, "x", n_points=9, n_sim=100)
        assert len(effect.effect) == 9
        assert np.isfinite(effect.se).all()
        assert model.term_test(parameter, "x").statistic == pytest.approx(
            row.estimate**2 / row.se**2, rel=3e-6
        )
    restored = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(restored.summary(X_train=X), table)


@pytest.fixture(scope="module")
def smooth_fit():
    return smooth_case(reml=True)


def test_actual_smoothing_and_corrected_covariance_against_profile_reference(smooth_fit):
    from superglm.distributional.posterior import _resolved_posterior_covariance

    model, X, y = smooth_fit
    assert model.result_.coefficient_converged
    assert model.smoothing_convergence_reason_ == "stationary"
    assert model.smoothing_certified_
    rho = np.log(list(model.smoothing_parameters_.values()))
    offsets = np.column_stack([X.o, np.zeros((len(X), 2))])

    def profile(r):
        return spline_profile(y.to_numpy(), X.x.to_numpy(), offsets, r)

    center, beta, hessian, designs = profile(rho)
    h = 0.001
    eye = np.eye(2)
    h_rho = np.empty((2, 2))
    jac = np.empty((len(beta), 2))
    for j in range(2):
        plus, minus = profile(rho + h * eye[j]), profile(rho - h * eye[j])
        assert abs((plus[0] - minus[0]) / (2 * h)) < 2e-5
        h_rho[j, j] = (plus[0] - 2 * center + minus[0]) / h**2
        jac[:, j] = (plus[1] - minus[1]) / (2 * h)
    h_rho[0, 1] = h_rho[1, 0] = sum(
        a * b * profile(rho + h * np.array([a, b]))[0] for a in (-1, 1) for b in (-1, 1)
    ) / (4 * h**2)
    assert np.linalg.eigvalsh(h_rho).min() > 0.1
    corrected_ref = np.linalg.inv(hessian) + jac @ np.linalg.solve(h_rho, jac.T)
    covariance, kind = _resolved_posterior_covariance(model._model, "corrected")
    assert kind == "corrected"
    assert np.linalg.eigvalsh(covariance - model.covariance_).min() > -1e-12
    _, se = model.predict_link(X.iloc[:1], se_fit=True, covariance="corrected")
    for j, (d, indices) in enumerate(zip(designs, (slice(0, 5), slice(5, 11), slice(11, 12)))):
        expected = d[0] @ corrected_ref[indices, indices] @ d[0]
        assert se.iloc[0, j] ** 2 == pytest.approx(expected, rel=2e-4)
    assert model.posterior_draws(4, covariance="corrected").covariance_kind == "corrected"


def test_chunked_smoothing_agrees_and_retained_replay_uses_analytic_directions(
    smooth_fit, monkeypatch
):
    import superglm.distributional.posterior as posterior
    from superglm.distributional.results.endpoint_evidence import ANALYTIC_DIRECTION_AUTHORITY

    model, X, _ = smooth_fit
    other, _, _ = smooth_case(reml=True, discrete=True)
    assert other._model.result.execution_backend_identifier == "distributional-chunked-v1"
    assert other.smoothing_convergence_reason_ == "stationary"
    np.testing.assert_allclose(other.predict(X), model.predict(X), rtol=3e-6)
    np.testing.assert_allclose(
        list(other.smoothing_parameters_.values()),
        list(model.smoothing_parameters_.values()),
        rtol=3e-5,
    )
    original = posterior.laml_derivatives
    authorities = []

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        authorities.append(result.third_derivative_authority)
        return result

    monkeypatch.setattr(posterior, "laml_derivatives", recording)
    covariance, kind = posterior._resolved_posterior_covariance(other._model, "corrected")
    assert kind == "corrected"
    assert authorities == [ANALYTIC_DIRECTION_AUTHORITY]
    np.testing.assert_allclose(
        covariance,
        posterior.posterior_covariance(model._model, kind="corrected"),
        rtol=2e-5,
        atol=1e-9,
    )


def test_optional_curvature_failure_discloses_conditional_covariance(smooth_fit, monkeypatch):
    import superglm.distributional.smoothing.derivatives as derivatives
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.distributional.posterior import _resolved_posterior_covariance

    model, X, _ = smooth_fit
    before = model.predict(X)

    def unresolved(*args, **kwargs):
        raise FloatingPointError("forced optional Dirichlet second direction range")

    monkeypatch.setattr(
        DirichletLSS, "predictor_curvature_second_directional_derivative", unresolved
    )
    # The existing numerical fallback is allowed. Force its optional second
    # direction to be unresolved too, so this test requires an honest fallback.
    monkeypatch.setattr(derivatives, "_second_derivatives", unresolved)
    with pytest.warns(
        ConvergenceWarning, match="unavailable.*row curvature could not be differentiated"
    ):
        covariance, kind = _resolved_posterior_covariance(model._model, "corrected")
    assert kind == "fixed"
    np.testing.assert_array_equal(covariance, model.covariance_)
    pd.testing.assert_frame_equal(model.predict(X), before)


def offset_case():
    X, y = book()
    X["o2"] = 0.05 * X.x**2
    a, b = 0.1 * X.x.to_numpy(), -0.03 * X.x.to_numpy() ** 2
    family = DirichletLSS(tuple(y.columns))
    model = SuperLSS(
        family,
        family.alpha("red", "x", offset=["o", pd.Series(a, index=X.index), b]),
        family.alpha("green", offset=["o", "o2"]),
        family.alpha("blue", "x", offset=[a, b]),
        weight_semantics="frequency",
    )
    replacements = {"alpha_0": a + b, "alpha_2": a + b}
    return model, X, y, replacements


def test_mixed_offsets_joint_scoring_cv_custom_scalar_and_replay():
    model, X, y, offsets = offset_case()
    model.fit(X, y)
    with pytest.raises(ValueError, match="replacement|array"):
        model.predict(X)
    with pytest.raises(ValueError, match="column"):
        model.predict(X, offsets={**offsets, "alpha_1": np.zeros(len(X))})
    family = model.family
    control_X = X.assign(sum_array=offsets["alpha_0"])
    control = SuperLSS(
        family,
        family.alpha("red", "x", offset=["o", "sum_array"]),
        family.alpha("green", offset=["o", "o2"]),
        family.alpha("blue", "x", offset="sum_array"),
        weight_semantics="frequency",
    ).fit(control_X, y)
    np.testing.assert_allclose(
        model.predict(X, offsets=offsets), control.predict(control_X), rtol=1e-12
    )
    np.testing.assert_allclose(
        model.scores(X, y, which=("log",), offsets=offsets),
        control.scores(control_X, y, which=("log",)),
        rtol=1e-12,
    )
    clone = model._clone_with_replaced_terms({})
    assert clone._model is None
    clone._replay_fit(X, y, None, offsets)
    np.testing.assert_allclose(
        clone.predict(X, offsets=offsets), model.predict(X, offsets=offsets), rtol=1e-12
    )

    def share_error(est, rows, response, *, sample_weight, offset):
        mean = est.predict(rows, offsets=offset).to_numpy()
        return np.mean((mean[:, 0] - response[:, 0]) ** 2)

    cv = cross_validate(
        model,
        X,
        y[["blue", "red", "green"]],
        cv=KFold(3),
        fit_mode="fit",
        scoring=("nll", share_error),
        return_oof=True,
        return_estimators=True,
        error_score="raise",
    )
    assert np.isfinite(cv.fold_scores[["nll", "share_error"]]).all().all()
    for est, (_, rows) in zip(cv.estimators, cv.fold_indices):
        off = {name: values[rows] for name, values in offsets.items()}
        np.testing.assert_allclose(cv.oof_predictions[rows], est.predict(X.iloc[rows], offsets=off))
    with pytest.raises(ValueError, match="Declare SuperLSS offsets"):
        cross_validate(model, X, y, cv=KFold(2), offset=offsets)


@pytest.mark.parametrize("bad", [True, np.timedelta64(1, "D"), np.datetime64("2026-01-01")])
def test_vector_scoring_offsets_keep_types_positions_and_drop_authority(bad):
    model, X, y, offsets = offset_case()
    model.fit(X, y)
    raw = offsets["alpha_0"].astype(object)
    raw[6] = bad
    replacements = {**offsets, "alpha_0": raw}
    weight = np.ones(len(X))
    weight[[1, 3]] = 0
    with pytest.raises(ValueError, match="6"):
        model.scores(X, y, which=("log",), offsets=replacements, sample_weight=weight)
    weight[6] = 0
    scores = model.scores(X, y, which=("log",), offsets=replacements, sample_weight=weight)
    assert scores.log.isna().sum() == 3
    wrong_index = pd.Series(raw, index=X.index[::-1])
    with pytest.raises(ValueError, match="index"):
        model.scores(
            X, y, which=("log",), offsets={**offsets, "alpha_0": wrong_index}, sample_weight=weight
        )


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_saved_predictions_inference_hand_edits_and_refit(retain, format, tmp_path, monkeypatch):
    import pickle

    import joblib

    import superglm.distributional.api as api
    from superglm.distributional.revision import EditedCoefficientsWarning

    model, X, y, offsets = offset_case()
    model.fit(X, y, retain_rows=retain)
    path = tmp_path / ("dirichlet." + format)
    if format == "pickle":
        path.write_bytes(pickle.dumps(model))
        restored = pickle.loads(path.read_bytes())
    else:
        joblib.dump(model, path)
        restored = joblib.load(path)
    expected = model.predict(X, offsets=offsets)

    def no_fit(*args, **kwargs):
        pytest.fail("prediction/inference tried to run a training optimizer")

    with monkeypatch.context() as scoped:
        scoped.setattr(api, "fit_dense_distributional", no_fit)
        pd.testing.assert_frame_equal(restored.predict(X, offsets=offsets), expected)
        pd.testing.assert_frame_equal(
            restored.predict_parameters(X, offsets=offsets),
            model.predict_parameters(X, offsets=offsets),
        )
        np.testing.assert_array_equal(restored.covariance_, model.covariance_)
        pd.testing.assert_frame_equal(
            restored.predict_link(X, offsets=offsets, se_fit=True)[1],
            model.predict_link(X, offsets=offsets, se_fit=True)[1],
        )
        bounds = restored.posterior_bounds(
            X.iloc[:3],
            lambda a: a[:, 0] / a.sum(axis=1),
            offsets={k: v[:3] for k, v in offsets.items()},
            n_draws=50,
        )
        np.testing.assert_allclose(bounds.estimate, expected.iloc[:3, 0])
    beta = restored._model.coefficients.copy()
    beta[restored._model.layout.term_slices["alpha_0:x"]] += 0.25
    edited = restored._with_revised_coefficients(beta, edited_terms=("alpha_0:x",), X_train=X)
    assert np.max(abs(edited.predict(X, offsets=offsets) - expected).to_numpy()) > 0.01
    np.testing.assert_array_equal(edited.covariance_, restored.covariance_)
    with pytest.warns(EditedCoefficientsWarning):
        edited.predict_link(X, offsets=offsets, se_fit=True)
    with pytest.raises(ValueError, match="fitted coefficients"):
        edited.posterior_draws(4, covariance="corrected")
    fresh = edited._clone_with_replaced_terms({})
    fresh._replay_fit(X, y, None, offsets)
    np.testing.assert_allclose(fresh.predict(X, offsets=offsets), expected, rtol=2e-12)
    assert fresh._hand_edited_terms == ()
    old_draws = restored.posterior_draws(4)
    restored.fit(X, y[["blue", "green", "red"]])
    with pytest.raises(ValueError, match="different fit"):
        restored.posterior_bounds(X, ("parameter", "alpha_0"), draws=old_draws, offsets=offsets)


@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_saved_smoothing_rebinds_inference_without_optimizer(
    smooth_fit, format, tmp_path, monkeypatch
):
    import pickle

    import joblib

    import superglm.distributional.solver.solver as solver
    from superglm.distributional.posterior import posterior_covariance

    model, X, _ = smooth_fit
    expected = posterior_covariance(model._model, kind="corrected")
    if format == "pickle":
        restored = pickle.loads(pickle.dumps(model))
    else:
        path = tmp_path / "smooth.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)

    def no_optimizer(*args, **kwargs):
        pytest.fail("saved prediction/inference called a training optimizer")

    monkeypatch.setattr(solver, "fit_dense_fixed_lambda", no_optimizer)
    np.testing.assert_allclose(
        posterior_covariance(restored._model, kind="corrected"), expected, rtol=1e-12
    )
    pd.testing.assert_frame_equal(restored.predict(X), model.predict(X))


def test_compact_smoothing_predicts_and_discloses_unavailable_correction(smooth_fit):
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.distributional.posterior import _resolved_posterior_covariance

    model, X, _ = smooth_fit
    compact, _, _ = smooth_case(reml=True, retain=False, discrete=True)
    assert compact._model.fit_state.retained_rows is None
    np.testing.assert_allclose(compact.predict(X), model.predict(X), rtol=3e-6)
    with pytest.warns(ConvergenceWarning, match="unavailable.*retained training rows"):
        covariance, kind = _resolved_posterior_covariance(compact._model, "corrected")
    assert kind == "fixed"
    np.testing.assert_array_equal(covariance, compact.covariance_)


@pytest.mark.parametrize("change", ["response", "labels"])
def test_changed_response_or_component_identity_refuses_corrected_replay(smooth_fit, change):
    from dataclasses import replace

    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.distributional.model import DenseDistributionalModel
    from superglm.distributional.posterior import _resolved_posterior_covariance

    model, _, _ = smooth_fit
    state = model._model.fit_state
    family = model.family
    if change == "response":
        y = state.retained_rows.response.copy()
        y[0] = y[0, [1, 0, 2]]
        state = replace(state, retained_rows=replace(state.retained_rows, response=y))
    else:
        family = DirichletLSS(("green", "red", "blue"))
    altered = DenseDistributionalModel(family, state)
    with pytest.warns(ConvergenceWarning, match="different likelihood plan identifier"):
        cov, kind = _resolved_posterior_covariance(altered, "corrected")
    assert kind == "fixed"
    np.testing.assert_array_equal(cov, model.covariance_)


def test_bound_identity_weights_bounds_provenance_and_derived_log_rounding(smooth_fit, monkeypatch):
    import superglm.distributional.families.dirichlet as module
    from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch
    from superglm.distributional.observation import ResponseBounds
    from superglm.distributional.weights import (
        LikelihoodWeightError,
        UnsupportedLikelihoodContractError,
        resolve_likelihood_weights,
    )

    model, _, _ = smooth_fit
    state = model._model.fit_state
    rows = state.retained_rows
    family = model.family
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION)

    def mismatch(other):
        return rebound_plan_mismatch(other, state.family_likelihood_plan_identifier, 0.0)

    assert mismatch(plan) is None
    for values in (np.full(len(rows.response), 2.0), np.r_[0.0, np.ones(len(rows.response))]):
        weights = resolve_likelihood_weights(
            values, n_observations=len(values), contract=state.weight_contract
        )
        assert (
            mismatch(family.bind_likelihood(rows.response, weights, COMPLETE_OBSERVATION))
            == "identifier"
        )
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete vector"):
        bounds = ResponseBounds.from_arrays(
            np.zeros(len(rows.response)),
            np.ones(len(rows.response)),
            np.zeros(len(rows.response), dtype=np.int8),
        )
        family.bind_likelihood(rows.response, rows.likelihood_weights, bounded_observation(bounds))
    with pytest.raises(LikelihoodWeightError, match="duplicate"):
        plan.take(np.array([0, 0]))
    take = plan.take(np.array([3, 1, 4]))
    np.testing.assert_array_equal(take.weights.input_positions, [3, 1, 4])
    assert mismatch(take) == "identifier"
    original = module.prepare_log_shares
    monkeypatch.setattr(module, "prepare_log_shares", lambda y: np.nextafter(original(y), np.inf))
    # Cross-platform transcendental rounding is derived, never identity authority.
    rebound = family.bind_likelihood(rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION)
    assert not np.array_equal(rebound.log_shares, plan.log_shares)
    assert mismatch(rebound) is None


def test_distinct_spline_frequency_replication():
    base, X, y = smooth_case()
    counts = np.arange(len(X)) % 3 + 1
    weighted = base._clone_with_replaced_terms({}).fit(
        X, y, sample_weight=counts, lambdas=base.smoothing_parameters_
    )
    positions = np.repeat(np.arange(len(X)), counts)
    replicated = base._clone_with_replaced_terms({}).fit(
        X.iloc[positions], y.iloc[positions], lambdas=base.smoothing_parameters_
    )
    np.testing.assert_allclose(weighted.predict(X), replicated.predict(X), rtol=2e-6)
    np.testing.assert_allclose(weighted.covariance_, replicated.covariance_, rtol=4e-6, atol=1e-10)


def test_public_prediction_underflow_is_not_a_structural_zero(fitted):
    model, X, _ = fitted
    # The generic undeclared-predictor map remains available at prediction.
    family = model.family
    plain = SuperLSS(family, *(family.alpha(c) for c in family.components)).fit(X, book()[1])
    desired = np.log([1e-300, 1e300, 1.0])
    fitted_eta = plain.predict_link(X.iloc[:1]).iloc[0].to_numpy()
    offsets = {
        p: np.array([desired[j] - fitted_eta[j]]) for j, p in enumerate(plain.parameter_names_)
    }
    assert (plain.predict_parameters(X.iloc[:1], offsets=offsets) > 0).all().all()
    with pytest.raises(ValueError, match="underflow|range|positive"):
        plain.predict(X.iloc[:1], offsets=offsets)


@pytest.mark.parametrize("consumer", ["scores", "compare"])
def test_unresolved_joint_density_cannot_publish_zero_log_scores(fitted, consumer):
    _, X, y = fitted
    family = DirichletLSS(tuple(y.columns))
    model = SuperLSS(
        family, *(family.alpha(c) for c in family.components), weight_semantics="frequency"
    ).fit(X, y)
    frame, response = X.iloc[:3], y.iloc[:3]
    eta = model.predict_link(frame).to_numpy()
    offsets = {
        name: np.array([0.0, np.log(1e308) - eta[1, j], np.log(1e308) - eta[2, j]])
        for j, name in enumerate(model.parameter_names_)
    }
    assert np.isfinite(model.predict_parameters(frame, offsets=offsets)).all().all()
    with pytest.raises(ValueError, match="concentration.*range"):
        model.predict(frame, offsets=offsets)
    weights = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match=r"log score.*original rows \[2\]"):
        if consumer == "scores":
            model.scores(frame, response, which=("log",), offsets=offsets, sample_weight=weights)
        else:
            model.compare(model, frame, response, offsets=offsets, sample_weight=weights)
    # The numerical range of omitted rows must not affect the retained score.
    retained = model.scores(
        frame, response, which=("log",), offsets=offsets, sample_weight=np.array([1.0, 0.0, 0.0])
    )
    assert np.isfinite(retained.log.iloc[0])
    assert retained.log.iloc[1:].isna().all()
