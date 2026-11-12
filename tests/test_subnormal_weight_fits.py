"""A common weight unit preserves Gaussian predictions, uncertainty and EDF."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, Numeric, Piecewise, PSpline, SuperGLM, Tweedie

U = np.finfo(np.float64).eps / 2


@pytest.mark.parametrize(("n_knots", "degree"), [(6, 3), (10, 4)])
@pytest.mark.parametrize("weight_unit", [1e-315, 1e-320])
def test_subnormal_gaussian_weights_agree_with_rectangular_least_squares(
    n_knots, degree, weight_unit
):
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = 0.9 * x + 0.25 * np.sin(12 * x)
    weights = weight_unit * np.random.default_rng(0).uniform(0.5, 1.5, len(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=0,
        features={
            "x": PSpline(
                n_knots=n_knots,
                degree=degree,
                m=1,
                penalty="none",
                constraint=Constraint.postfit.increasing,
            )
        },
    ).fit(frame, y, sample_weight=weights)

    basis = model._specs["x"].transform(x)
    design = np.column_stack([np.ones(len(x)), basis])
    # Lift the ACTUAL represented weights, preserving even their subnormal
    # input quantization. The rectangular oracle never forms a tiny Gram.
    lifted = np.ldexp(weights, 1000)
    root = np.sqrt(lifted)
    weighted = root[:, None] * design
    beta = np.linalg.lstsq(weighted, root * y, rcond=None)[0]
    prediction = design @ beta
    condition = np.linalg.cond(weighted)
    bound = 128 * U * condition * np.max(np.abs(prediction))
    np.testing.assert_allclose(model.predict(frame), prediction, rtol=0, atol=bound)
    assert model.result.converged
    assert model.result.effective_df == pytest.approx(design.shape[1], rel=128 * U * condition)

    phi = float(lifted @ (y - prediction) ** 2 / (len(x) - design.shape[1]))
    covariance = phi * np.linalg.inv(design.T @ (lifted[:, None] * design))
    effect = model.term_inference("x", n_points=17)
    grid = model._specs["x"].transform(effect.x)
    variance = np.einsum("ij,jk,ik->i", grid, covariance[1:, 1:], grid)
    se = np.sqrt(variance)
    np.testing.assert_allclose(
        effect.se_log_relativity,
        se,
        rtol=0,
        atol=128 * U * condition**2 * np.max(se),
    )


def test_subnormal_penalty_keeps_its_ratio_to_the_data_information():
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = 0.9 * x + 0.25 * np.sin(12 * x)
    weights = 1e-320 * np.random.default_rng(0).uniform(0.5, 1.5, len(x))
    penalty = 3e-320

    def fit(w, lam):
        return SuperGLM(
            family="gaussian",
            selection_penalty=0,
            spline_penalty=lam,
            features={"x": PSpline(n_knots=6, m=1)},
        ).fit(frame, y, sample_weight=w)

    fitted = fit(weights, penalty)
    reference = fit(np.ldexp(weights, 1064), np.ldexp(penalty, 1064))
    np.testing.assert_allclose(fitted.predict(frame), reference.predict(frame), rtol=2e-12)
    assert fitted.result.effective_df == pytest.approx(reference.result.effective_df, rel=2e-12)
    np.testing.assert_allclose(
        fitted.term_inference("x").se_log_relativity,
        reference.term_inference("x").se_log_relativity,
        rtol=2e-12,
    )


def test_subnormal_poisson_weights_keep_known_scale_and_representable_standard_errors():
    x = np.linspace(-1, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = np.random.default_rng(12).poisson(np.exp(0.4 + 0.7 * x))
    weights = 1e-320 * np.random.default_rng(13).uniform(0.5, 1.5, len(x))

    def fit(w):
        return SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Numeric()},
        ).fit(frame, y, sample_weight=w)

    fitted = fit(weights)
    reference = fit(np.ldexp(weights, 1064))
    np.testing.assert_allclose(fitted.predict(frame), reference.predict(frame), rtol=2e-10)
    assert fitted.result.phi == 1.0
    assert fitted.result.converged
    se = fitted.term_inference("x").se_log_relativity
    reference_se = np.ldexp(reference.term_inference("x").se_log_relativity, 532)
    assert np.all(np.isfinite(se))
    np.testing.assert_allclose(se, reference_se, rtol=2e-10)
    rows = {row.name: row for row in fitted.summary()._coef_rows}
    reference_rows = {row.name: row for row in reference.summary()._coef_rows}
    for name in ("Intercept", "x"):
        assert np.isfinite(rows[name].se)
        assert rows[name].se == pytest.approx(np.ldexp(reference_rows[name].se, 532), rel=2e-10)
    metrics = fitted.metrics(frame, y, sample_weight=weights)
    np.testing.assert_allclose(metrics.coefficient_se_raw["x"], se, rtol=2e-10)
    assert metrics.intercept_se_raw == pytest.approx(rows["Intercept"].se, rel=2e-10)


@pytest.mark.parametrize("family", ["gaussian", "gamma", "tweedie"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_subnormal_fit_statistics_and_diagnostics_keep_the_weight_contract(family, semantics):
    import mpmath as mp
    from scipy import stats

    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = np.exp(0.4 + 0.2 * x + 0.1 * np.cos(13 * x))
    weights = 1e-320 * np.random.default_rng(14).uniform(0.5, 1.5, len(x))
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        selection_penalty=0,
        weight_semantics=semantics,
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=weights)
    mu = model.predict(frame)
    lifted = np.ldexp(weights, 1064)
    variance = model._distribution.variance(mu)
    df = len(x) - 2 if semantics == "prior" else 1
    phi = np.sum(lifted * (y - mu) ** 2 / variance) / df
    design = np.column_stack([np.ones(len(x)), x])
    information = design.T @ (
        (lifted / variance * model._link.deriv_inverse(model._link.link(mu)) ** 2)[:, None] * design
    )
    covariance = phi * np.linalg.inv(information)
    metrics = model.metrics(frame, y, sample_weight=weights)
    assert metrics.intercept_se == pytest.approx(np.sqrt(covariance[0, 0]), rel=2e-9)
    assert metrics.coefficient_se["x"][0] == pytest.approx(np.sqrt(covariance[1, 1]), rel=2e-9)
    expected_residuals = np.sqrt(lifted) * (y - mu) / np.sqrt(variance * phi)
    expected_residuals /= np.sqrt(1 - metrics.leverage)
    np.testing.assert_allclose(metrics.std_pearson_residuals, expected_residuals, rtol=2e-9)
    assert np.all(np.isfinite(metrics.residuals("quantile")))

    if semantics == "prior":
        if family == "gaussian":
            expected_ll = np.sum(stats.norm.logpdf(y, loc=mu, scale=np.sqrt(phi / lifted)))
        elif family == "gamma":
            expected_ll = np.sum(stats.gamma.logpdf(y, a=lifted / phi, scale=mu * phi / lifted))
        else:
            with mp.workdps(70):
                terms = []
                for value, mean, weight in zip(y, mu, lifted, strict=True):
                    observed, location = mp.mpf(float(value)), mp.mpf(float(mean))
                    scale = mp.mpf(float(phi)) / mp.mpf(float(weight))
                    terms.append(
                        mp.log(2 / (scale * mp.sqrt(observed)))
                        - 2 * (mp.sqrt(location) + observed / mp.sqrt(location)) / scale
                        + mp.log(mp.besseli(1, 4 * mp.sqrt(observed) / scale))
                    )
                expected_ll = float(mp.fsum(terms))
    else:
        # Independent density oracle: physical phi need not be representable.
        with mp.workdps(400):
            exact_phi = mp.mpf(float(phi)) * mp.power(2, -1064)
            terms = []
            for observed, mean, weight in zip(y, mu, weights, strict=True):
                value, location, count = map(mp.mpf, (float(observed), float(mean), float(weight)))
                if family == "gaussian":
                    density = (
                        -(mp.log(2 * mp.pi * exact_phi) + (value - location) ** 2 / exact_phi) / 2
                    )
                elif family == "gamma":
                    shape = 1 / exact_phi
                    density = (
                        shape * mp.log(shape / location)
                        + (shape - 1) * mp.log(value)
                        - shape * value / location
                        - mp.loggamma(shape)
                    )
                else:
                    density = (
                        mp.log(2 / (exact_phi * mp.sqrt(value)))
                        - 2 * (mp.sqrt(location) + value / mp.sqrt(location)) / exact_phi
                        + mp.log(mp.besseli(1, 4 * mp.sqrt(value) / exact_phi))
                    )
                terms.append(count * density)
            expected_ll = float(mp.fsum(terms))
    assert model._fit_stats.log_likelihood == pytest.approx(expected_ll, rel=2e-10, abs=2e-11)
    assert metrics.log_likelihood == pytest.approx(expected_ll, rel=2e-10, abs=2e-11)
    assert model._fit_stats.likelihood_size == pytest.approx(
        len(x) if semantics == "prior" else np.sum(weights),
        rel=2e-12,
        abs=0,
    )


def test_scaled_fit_keeps_inference_after_serialization_and_releasing_rows():
    import pickle

    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = 0.4 + 0.2 * x + 0.1 * np.cos(13 * x)
    weights = 1e-320 * np.ones(len(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        retain_fit_state=False,
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=weights)
    se = model.term_inference("x").se_log_relativity
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
    np.testing.assert_array_equal(restored.term_inference("x").se_log_relativity, se)
    assert np.all(np.isfinite(se))
    assert restored.summary()._coef_rows[0].se > 0


def test_subnormal_postfit_repair_uses_the_same_penalty_unit_as_the_fit():
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = 0.9 * x + 0.25 * np.sin(12 * x)
    weights = 1e-320 * np.random.default_rng(0).uniform(0.5, 1.5, len(x))
    penalty = 3e-320

    def fit(w, lam):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            spline_penalty=lam,
            features={"x": PSpline(n_knots=6, m=1, constraint=Constraint.postfit.increasing)},
        ).fit(frame, y, sample_weight=w)
        return model.apply_shape_postfit(frame, sample_weight=w)

    fitted = fit(weights, penalty)
    reference = fit(np.ldexp(weights, 1064), np.ldexp(penalty, 1064))
    np.testing.assert_allclose(fitted.predict(frame), reference.predict(frame), rtol=0, atol=2e-8)
    assert fitted.result.phi == float(np.ldexp(reference.result.phi, -1064))


def test_a_large_active_penalty_caps_the_weight_lift_without_changing_its_target():
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = 0.9 * x + 0.25 * np.sin(12 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e300,
        features={"x": PSpline(n_knots=6, m=1)},
    ).fit(frame, y, sample_weight=np.full(len(x), 1e-320))
    assert model.result.weight_exponent == 0
    assert np.all(np.isfinite(model.predict(frame)))
    assert np.ptp(model.predict(frame)) < 1e-12


@pytest.mark.parametrize("kind", ["spline", "piecewise"])
def test_known_scale_curve_errors_are_finite_before_a_variance_can_be_materialized(kind):
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    y = np.random.default_rng(15).poisson(np.exp(0.4 + 0.7 * x))
    weights = np.full(len(x), 1e-320)

    def fit(w):
        feature = PSpline(n_knots=6) if kind == "spline" else Piecewise([0.3, 0.7])
        return SuperGLM(
            family="poisson",
            selection_penalty=0,
            spline_penalty=0,
            features={"x": feature},
        ).fit(frame, y, sample_weight=w)

    fitted, reference = fit(weights), fit(np.ldexp(weights, 1064))
    for centering in ("native", "mean"):
        kwargs = dict(centering=centering)
        if kind == "spline":
            kwargs.update(simultaneous=True, n_sim=100)
        actual = fitted.term_inference("x", **kwargs)
        expected = reference.term_inference("x", **kwargs)
        assert np.all(np.isfinite(actual.se_log_relativity))
        np.testing.assert_allclose(
            actual.se_log_relativity,
            np.ldexp(expected.se_log_relativity, 532),
            rtol=2e-9,
        )
        if kind == "piecewise":
            assert not np.any(np.isnan(actual.knot_covariance))


def test_old_solver_result_restores_unscaled_units():
    from superglm.solvers.pirls import PIRLSResult

    old_state = dict(
        beta=np.zeros(1),
        intercept=0.0,
        n_iter=1,
        deviance=1.0,
        converged=True,
        phi=2.0,
        effective_df=2.0,
    )
    restored = PIRLSResult.__new__(PIRLSResult)
    restored.__setstate__(old_state)
    assert restored.weight_exponent == 0
    assert restored.solver_phi is None


def test_evaluation_weights_can_use_a_different_unit_from_the_fitted_geometry():
    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = 0.4 + 0.2 * x + 0.1 * np.cos(13 * x)
    weights = np.full(len(x), 1e-320)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=weights)
    metrics = model.metrics(frame, y)
    lifted = np.ldexp(weights, 1064)
    phi = np.sum(lifted * (y - model.predict(frame)) ** 2) / (len(x) - 2)
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.inv(design.T @ design)
    expected = np.ldexp(np.sqrt(phi * np.diag(inverse)), -532)
    assert metrics.intercept_se == pytest.approx(expected[0], rel=2e-12, abs=0)
    assert metrics.coefficient_se["x"][0] == pytest.approx(expected[1], rel=2e-12, abs=0)
    row = next(row for row in metrics._build_coef_rows() if row.name == "x")
    assert row.se == pytest.approx(expected[1], rel=2e-12, abs=0)
    np.testing.assert_allclose(metrics.feature_se("x")["se_coef"], expected[1], rtol=2e-12, atol=0)

    # An exactly predicted evaluation row has a finite log density even when
    # the physical variance rounds to zero. No infinite prior weight is formed.
    perfect = model.metrics(frame, model.predict(frame))
    expected_ll = -0.5 * len(x) * (np.log(2 * np.pi * phi) - 1064 * np.log(2.0))
    assert perfect.log_likelihood == pytest.approx(expected_ll, rel=2e-12)


def test_known_scale_path_tables_use_unit_dispersion():
    from superglm.inference.covariance import covariance_selected_diagonal

    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 400)
    z = rng.choice(list("abcd"), len(x))
    frame = pd.DataFrame({"x": x, "z": z})
    mu = np.exp(0.3 + 0.6 * x + (z == "b") * 0.4)
    y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    model = SuperGLM(family="poisson", features={"x": PSpline(n_knots=8), "z": Categorical()})
    model.fit_path(frame, y, n_lambda=5)
    # Path results retain Pearson dispersion for quasi-likelihood diagnostics.
    # A fitted-family coefficient table still uses the known unit dispersion.
    assert model.result.phi > 1.5
    inference = model._fit_inference_info
    augmented = inference["XtWX_inv_aug"]
    categorical = next(g for g in inference["active_groups"] if g.feature_name == "z")
    indices = np.r_[0, np.arange(categorical.start, categorical.end) + 1]
    expected = np.sqrt(covariance_selected_diagonal(augmented, indices))
    labels = ["Intercept", *[f"z[{level}]" for level in model._specs["z"]._non_base]]
    for rows in (model.summary()._coef_rows, model.metrics(frame, y)._build_coef_rows()):
        actual = {row.name: row.se for row in rows}
        np.testing.assert_allclose([actual[label] for label in labels], expected, rtol=2e-12)


@pytest.mark.parametrize("retain", [True, False])
def test_subnormal_drop1_keeps_its_table_after_releasing_fit_rows(retain):
    x = np.linspace(-1, 1, 80)
    frame = pd.DataFrame({"x": x, "u": np.cos(5 * x)})
    y = 0.4 + 0.2 * x + 0.1 * np.cos(13 * x)
    weights = np.full(len(x), 1e-320)

    def table(w):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            retain_fit_state=retain,
            features={"x": Numeric(), "u": Numeric()},
        ).fit(frame, y, sample_weight=w)
        return model.drop1(frame, y, sample_weight=w).set_index("feature").sort_index()

    actual, expected = table(weights), table(np.ldexp(weights, 1064))
    np.testing.assert_allclose(actual["statistic"], expected["statistic"], rtol=2e-10)
    np.testing.assert_allclose(actual["p_value"], expected["p_value"], rtol=2e-9, atol=0)


@pytest.mark.parametrize("unit", [1e-315, 1e-320])
def test_editor_response_influence_keeps_the_dispersion_exponent(unit):
    from superglm.editor.free_levels import _gap_variances

    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = 0.4 + 0.2 * x + 0.1 * np.cos(13 * x)
    weights = np.full(len(x), unit)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=weights)
    actual, note = _gap_variances(model, np.ones((1, 1)), model, None, "x", frame)
    assert note is None
    lifted = np.ldexp(weights, 1064)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(np.sqrt(lifted)[:, None] * design, np.sqrt(lifted) * y, rcond=None)[0]
    phi = np.sum(lifted * (y - design @ beta) ** 2) / (len(x) - 2)
    expected = phi * np.linalg.inv(design.T @ (lifted[:, None] * design))[1, 1]
    np.testing.assert_allclose(actual, [expected], rtol=2e-11)


@pytest.mark.parametrize("family", ["gaussian", "gamma", "tweedie"])
@pytest.mark.parametrize("unit", [1e-315, 1e-320])
def test_quantile_residuals_keep_finite_row_scales_on_other_weights(family, unit):
    from scipy.stats import norm

    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = np.exp(0.4 + 0.2 * x + 0.1 * np.cos(13 * x))
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        selection_penalty=0,
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=np.full(len(x), unit))
    response = model.predict(frame).copy()
    response[1::3] = np.nextafter(response[1::3], np.inf)
    response[2::3] = np.nextafter(response[2::3], -np.inf)
    actual = model.metrics(frame, response).residuals("quantile")
    # At these physical dispersions even one ulp of the mean is far beyond
    # the existing probability clipping endpoints. At the mean the normal
    # limit's skewness correction is far below u for Gamma/Tweedie as well.
    expected = np.zeros(len(x))
    expected[1::3] = norm.ppf(1 - 1e-10)
    expected[2::3] = norm.ppf(1e-10)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("family", ["gaussian", "gamma", "tweedie"])
@pytest.mark.parametrize("unit", [1e-315, 1e-320])
def test_frequency_likelihood_at_the_mean_is_finite_on_other_weights(family, unit):
    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = np.exp(0.4 + 0.2 * x + 0.1 * np.cos(13 * x))
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        selection_penalty=0,
        weight_semantics="frequency",
        features={"x": Numeric()},
    ).fit(frame, y, sample_weight=np.full(len(x), unit))
    mu = model.predict(frame)
    result = model.result
    log_phi = np.log(result.solver_phi) - result.weight_exponent * np.log(2.0)
    power = {"gaussian": 0.0, "gamma": 2.0, "tweedie": 1.5}[family]
    # The Gaussian normalizer is exact. At these scales the Gamma Stirling
    # and Tweedie saturated-density corrections are far below rounding.
    expected = float(np.sum(-0.5 * (np.log(2 * np.pi) + log_phi + power * np.log(mu))))
    actual = model.metrics(frame, mu).log_likelihood
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-12)


@pytest.mark.parametrize("unit", [1e-315, 1e-320])
@pytest.mark.parametrize("at_mean", [False, True])
def test_editor_export_refreshes_statistics_in_the_evaluation_weight_unit(unit, at_mean):
    from superglm.editor import EditorSession

    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = 0.4 + 0.2 * x + 0.1 * np.cos(13 * x)
    model = SuperGLM(family="gaussian", selection_penalty=0, features={"x": Numeric()}).fit(
        frame, y, sample_weight=np.full(len(x), unit)
    )
    response = model.predict(frame) if at_mean else y
    edited = EditorSession.from_model(model, terms=["x"]).to_model(X=frame, y=response)
    mu = edited.predict(frame)
    deviance = float(np.sum((response - mu) ** 2))
    null_deviance = float(np.sum((response - np.mean(response)) ** 2))
    assert np.isfinite(edited.result.deviance)
    assert edited.result.deviance == pytest.approx(deviance, rel=2e-12, abs=1e-28)
    assert edited._fit_stats.null_deviance == pytest.approx(null_deviance, rel=2e-12)
    assert edited._fit_stats.pearson_chi2 == pytest.approx(deviance, rel=2e-12, abs=1e-28)
    assert edited._fit_stats.explained_deviance == pytest.approx(1 - deviance / null_deviance)
    if at_mean:
        result = model.result
        log_phi = np.log(result.solver_phi) - result.weight_exponent * np.log(2.0)
        expected_ll = -0.5 * len(x) * (np.log(2 * np.pi) + log_phi)
        assert edited._fit_stats.log_likelihood == pytest.approx(expected_ll, rel=2e-12)


@pytest.mark.parametrize("unit", [1e-160, 1e-315, 1e-320])
def test_weight_lift_preserves_headroom_for_large_numeric_columns(unit):
    t = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": 1e154 * t})
    y = 0.4 + 0.2 * t + 0.1 * np.cos(13 * t)
    model = SuperGLM(family="gaussian", selection_penalty=0, features={"x": Numeric()}).fit(
        frame, y, sample_weight=np.full(len(t), unit)
    )
    # The independent rectangular problem is expressed in a safe column unit.
    design = np.column_stack([np.ones(len(t)), t])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    expected = design @ beta
    bound = 128 * U * np.linalg.cond(design) * np.max(np.abs(expected))
    assert model.result.converged
    np.testing.assert_allclose(model.predict(frame), expected, rtol=0, atol=bound)
    assert model.result.effective_df == pytest.approx(2.0, rel=128 * U)


@pytest.mark.parametrize("unit", [1.0, 1e-315, 1e-320])
def test_exact_gaussian_fit_keeps_zero_cooks_distance(unit):
    x = np.linspace(-1, 1, 60)
    frame = pd.DataFrame({"x": x})
    y = np.zeros(len(x))
    weights = np.full(len(x), unit)
    model = SuperGLM(family="gaussian", selection_penalty=0, features={"x": Numeric()}).fit(
        frame, y, sample_weight=weights
    )
    np.testing.assert_array_equal(model.metrics(frame, y, sample_weight=weights).cooks_distance, 0)
