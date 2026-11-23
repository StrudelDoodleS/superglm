"""The fit-entry advisory describes the first working system without steering it."""

import math
import pickle
import warnings
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, PSpline, Ridge, SuperGLM, generate_tweedie_cpg
from superglm.diagnostics import convergence
from superglm.distributions import NegativeBinomial, Tweedie
from superglm.links import IdentityLink, LogLink
from superglm.profiling import nb
from superglm.solvers.working_rows import CoefficientWorkingRows


def _advisories(caught):
    return [warning for warning in caught if warning.category.__name__ == "FitPrecisionWarning"]


def _findings(model):
    return model.diagnostics()["_model"]["initial_precision"]


@pytest.mark.parametrize("family, offset", [("binomial", 800.0), ("poisson", -800.0)])
def test_recoverable_constant_offsets_stay_silent(family, offset):
    X = pd.DataFrame({"level": ["a", "a", "b", "b"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SuperGLM(family=family, link="log", features={"level": Categorical()}).fit(
            X, np.array([0.0, 1.0, 0.0, 1.0]), offset=np.full(4, offset), max_iter=2
        )

    assert _advisories(caught) == []
    assert _findings(model) == {}


def test_first_predictor_combines_offset_and_intercept_before_classifying_log_range():
    X = pd.DataFrame({"x": np.arange(8.0)})
    y = np.full(8, 2.0)
    # The starting intercept is log(2). Row 1 would round to zero if the
    # offset were read alone. Row 4 has no likelihood and is excluded.
    offset = np.array([0.0, -745.5, -746.0, 711.0, -900.0, 0.0, 0.0, 0.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    assert 0.0 < math.exp(offset[1] + math.log(2.0)) < np.finfo(float).tiny
    assert math.exp(offset[2] + math.log(2.0)) == 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SuperGLM(family="poisson", features={"x": Numeric()}).fit(
            X, y, sample_weight=weight, offset=offset, max_iter=1
        )

    (advisory,) = _advisories(caught)
    assert advisory.filename == __file__
    message = str(advisory.message).lower()
    assert "offset" in message and "subnormal" in message and "rounds to zero" in message
    findings = _findings(model)
    assert findings["log_mean_subnormal"]["rows"] == (1,)
    assert findings["log_mean_zero"]["rows"] == (2,)
    assert findings["log_mean_overflow"]["rows"] == (3,)
    assert findings["zero_sample_weight"]["rows"] == (4,)


def test_weight_and_information_spreads_are_distinct_on_a_light_level():
    X = pd.DataFrame({"level": ["a", "a", "b", "b"]})
    y = np.array([0.0, 1.0, 0.0, 1.0])
    weight = np.array([1e-8, 1e-8, 1.0, 1.0])
    offset = np.array([-15.0, -15.0, 0.0, 0.0])
    # Each level's mixed Bernoulli likelihood has an interior optimum at 1/2.
    # At the start, Fisher information is w * p / (1 - p), a much wider spread.
    p_light = 0.5 * math.exp(-15.0)
    assert (1.0 - p_light) / (weight[0] * p_light) > 2.0**49
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SuperGLM(family="binomial", link="log", features={"level": Categorical()}).fit(
            X, y, sample_weight=weight, offset=offset
        )

    (advisory,) = _advisories(caught)
    assert "spread" in str(advisory.message) and "Fisher" in str(advisory.message)
    findings = _findings(model)
    assert "sample_weight_spread" not in findings
    assert findings["information_spread"]["rows"] == (0, 2)
    assert findings["information_spread"]["values"][0] > 0.0
    assert "zero_information" not in findings


def test_input_zero_is_distinct_from_positive_weight_with_zero_fisher_information():
    X = pd.DataFrame({"x": np.arange(7.0)})
    y = np.array([0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 2.0**-1074, 2.0**-1000])
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -60.0, -40.0])
    assert weight[5] * math.exp(-60.0) == 0.0
    assert 0.0 < weight[6] * math.exp(-40.0) < np.finfo(float).tiny
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SuperGLM(family="poisson", features={"x": Numeric()}).fit(
            X, y, sample_weight=weight, offset=offset
        )

    (advisory,) = _advisories(caught)
    assert "Zero initial Fisher information" in str(advisory.message)
    findings = _findings(model)
    assert findings["zero_sample_weight"]["rows"] == (4,)
    assert findings["subnormal_sample_weight"]["rows"] == (5,)
    assert findings["sample_weight_spread"]["rows"] == (5, 0)
    assert findings["zero_information"]["rows"] == (5,)
    assert findings["subnormal_information"]["rows"] == (6,)


def test_report_counts_across_blocks_and_bounds_examples_after_fit_state_release(monkeypatch):
    if hasattr(convergence, "_PRECISION_BLOCK_SIZE"):
        monkeypatch.setattr(convergence, "_PRECISION_BLOCK_SIZE", 3)
    x = np.arange(12.0)
    X = pd.DataFrame({"x": x})
    weight = np.ones_like(x)
    weight[[1, 2, 4, 5, 9]] = 0.0
    model = SuperGLM(family="gaussian", features={"x": Numeric()}, retain_fit_state=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, x + 0.1 * np.cos(x), sample_weight=weight)

    (advisory,) = _advisories(caught)
    assert "e.g." in str(advisory.message)
    restored = pickle.loads(pickle.dumps(model))
    finding = _findings(restored)["zero_sample_weight"]
    assert finding == {"count": 5, "rows": (1, 2, 4)}
    finding["count"] = 0
    assert _findings(restored)["zero_sample_weight"]["count"] == 5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored.fit(X, x + 0.1 * np.cos(x))
    assert not _advisories(caught)
    assert not _findings(restored)


def test_spread_threshold_compares_mantissas_at_equal_exponent_gap():
    # Both cases have the same exponent gap. Their significands put the
    # actual ratio at and just below the 2**49 threshold.
    at = np.array([2.0**-49, 1.0])
    below = np.array([np.nextafter(at[0], np.inf), 1.0])
    scan = convergence._scan_initial_precision
    equal = scan(at, at, np.zeros(2), None, IdentityLink())
    assert equal["sample_weight_spread"]["values"] == tuple(at)
    assert scan(below, below, np.zeros(2), None, IdentityLink()) == {}


def test_log_range_limits_distinguish_binary64_neighbours():
    with localcontext() as context:
        context.prec = 80
        tiny = Decimal(2) ** -1022
        half_subnormal = Decimal(2) ** -1075
        overflow_midpoint = Decimal(np.finfo(float).max) + Decimal(2) ** 970
        cutoffs = tuple(
            value.ln() for value in (half_subnormal, tiny - half_subnormal, overflow_midpoint)
        )
    eta = np.array(
        sorted(
            neighbour
            for cutoff in cutoffs
            for neighbour in (
                np.nextafter(float(cutoff), -np.inf),
                float(cutoff),
                np.nextafter(float(cutoff), np.inf),
            )
        )
    )
    weight = np.ones_like(eta)
    scan = convergence._scan_initial_precision
    findings = scan(weight, weight, eta, eta, LogLink())
    for name, condition in (
        ("log_mean_zero", lambda value: value < cutoffs[0]),
        ("log_mean_subnormal", lambda value: cutoffs[0] < value < cutoffs[1]),
        ("log_mean_overflow", lambda value: value > cutoffs[2]),
    ):
        expected = tuple(i for i, value in enumerate(eta) if condition(Decimal(value)))
        assert findings[name] == {"count": len(expected), "rows": expected[:3]}
    assert scan(weight, weight, eta, eta, IdentityLink()) == {}


def test_advisory_uses_the_primary_rows_and_labels_their_curvature():
    group_list = []
    model = SimpleNamespace(_groups=group_list, _fit_weights=np.ones(2), _fit_offset=None)
    eta = np.zeros(2)
    nested = CoefficientWorkingRows(np.ones(2), eta, "fisher")
    primary = CoefficientWorkingRows(np.array([0.0, 1.0]), eta, "observed")
    link = IdentityLink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with convergence.initial_precision_advisory(model, enabled=True) as findings:
            convergence.inspect_initial_precision([], nested, eta, link)
            convergence.inspect_initial_precision(group_list, primary, eta, link)
            convergence.inspect_initial_precision(group_list, primary, eta, link)
        assert not _advisories(caught)
        convergence.warn_initial_precision(findings)

    (advisory,) = _advisories(caught)
    assert "Zero initial observed information" in str(advisory.message)
    assert findings["zero_information"] == {
        "count": 1,
        "rows": (0,),
        "curvature_source": "observed",
    }


@pytest.mark.parametrize("route", ["direct", "bcd", "exact_reml", "discrete_reml"])
def test_each_fit_scans_once_without_changing_results(route, monkeypatch):
    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = np.sin(8.0 * x) + 0.1 * np.cos(125.0 * x)
    weight = np.ones_like(y)
    weight[12] = 0.0
    reml = route.endswith("reml")

    def fit():
        model = SuperGLM(
            family="gaussian",
            features={"x": PSpline(n_knots=6) if reml else Numeric()},
            penalty=Ridge(lambda1=0.05 if route == "bcd" else 0.0),
            discrete=route == "discrete_reml",
        )
        if reml:
            return model.fit_reml(X, y, sample_weight=weight)
        return model.fit(X, y, sample_weight=weight, max_iter=1 if route == "bcd" else None)

    scan = getattr(convergence, "_scan_initial_precision", None)
    calls = []

    def record(weights, *args):
        calls.append(len(weights))
        return scan(weights, *args)

    reml_candidates = []
    if reml:
        from superglm.reml import direct, discrete

        engine = discrete if route == "discrete_reml" else direct
        solve = engine.fit_irls_direct

        def record_candidate(*args, **kwargs):
            before = len(calls)
            result = solve(*args, **kwargs)
            if kwargs.get("trace_purpose") == "reml_candidate":
                reml_candidates.append(len(calls) - before)
            return result

        monkeypatch.setattr(engine, "fit_irls_direct", record_candidate)

    if scan is not None:
        monkeypatch.setattr(convergence, "_scan_initial_precision", record)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = fit()

    (advisory,) = _advisories(caught)
    assert advisory.category is convergence.FitPrecisionWarning
    assert calls == [len(y)]
    assert _findings(actual)["zero_sample_weight"]["rows"] == (12,)
    if reml:
        assert reml_candidates and all(count == 0 for count in reml_candidates)

    monkeypatch.setattr(convergence, "_scan_initial_precision", lambda *args: {})
    with warnings.catch_warnings(record=True) as control_warnings:
        warnings.simplefilter("always")
        control = fit()
    assert not _advisories(control_warnings)
    assert not _findings(control)
    _assert_same_fit(actual, control, X)


def _assert_same_fit(actual, control, X):
    np.testing.assert_array_equal(actual.result.beta, control.result.beta)
    np.testing.assert_array_equal(actual.predict(X), control.predict(X))
    for name in (
        "intercept",
        "deviance",
        "phi",
        "effective_df",
        "n_iter",
        "termination_reason",
        "converged",
    ):
        assert getattr(actual.result, name) == getattr(control.result, name)
    if actual._reml_result is not None:
        for name in ("lambdas", "converged", "n_reml_iter", "termination_reason"):
            assert getattr(actual._reml_result, name) == getattr(control._reml_result, name)
        assert actual._reml_result.lambda_history == control._reml_result.lambda_history


@pytest.mark.parametrize("route", ["fit", "fit_reml", "estimate_theta"])
def test_nb_theta_mean_fits_leave_the_advisory_to_the_public_fit(route, monkeypatch):
    rng = np.random.default_rng(32)
    x = np.linspace(0.0, 1.0, 150)
    X = pd.DataFrame({"x": x})
    mu = np.exp(1.0 + np.sin(5.0 * x))
    y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    weight = np.ones_like(y)
    weight[12] = 0.0

    scan = getattr(convergence, "_scan_initial_precision", None)
    scans = []
    mean_fits = []
    solve = nb._MeanFit.fit

    def record_scan(weights, *args):
        scans.append(len(weights))
        return scan(weights, *args)

    def record_mean_fit(self, theta):
        before = len(scans)
        result = solve(self, theta)
        mean_fits.append(len(scans) - before)
        return result

    def fit():
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"x": PSpline(n_knots=6) if route == "fit_reml" else Numeric()},
            penalty=Ridge(lambda1=0.0),
        )
        getattr(model, route)(X, y, sample_weight=weight)
        return model

    if scan is not None:
        monkeypatch.setattr(convergence, "_scan_initial_precision", record_scan)
    monkeypatch.setattr(nb._MeanFit, "fit", record_mean_fit)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = fit()
    (advisory,) = _advisories(caught)
    assert advisory.filename == __file__
    assert len(mean_fits) > 1 and all(count == 0 for count in mean_fits)
    assert scans == [len(y)]
    assert _findings(actual)["zero_sample_weight"]["rows"] == (12,)

    monkeypatch.setattr(convergence, "_scan_initial_precision", lambda *args: {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        control = fit()
    assert not _advisories(caught)
    _assert_same_fit(actual, control, X)
    left, right = actual._nb_profile_result, control._nb_profile_result
    assert left.theta_hat == right.theta_hat
    assert left.converged == right.converged
    pd.testing.assert_frame_equal(left.evaluations, right.evaluations, check_exact=True)


@pytest.mark.parametrize("search_fit_mode", ["fit", "reml"])
def test_tweedie_power_candidates_leave_the_advisory_to_the_refit(search_fit_mode, monkeypatch):
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = generate_tweedie_cpg(
        len(x), mu=np.exp(0.5 + np.sin(5.0 * x)), phi=1.5, p=1.5, rng=np.random.default_rng(34)
    )
    weight = np.ones_like(y)
    y[12] = 0.0
    weight[12] = 2.0**-50  # Tweedie's prior-weight likelihood requires positive weights.
    scan = getattr(convergence, "_scan_initial_precision", None)
    scans = []
    search_scans = []

    def record_scan(weights, *args):
        scans.append(len(weights))
        return scan(weights, *args)

    def progress(phase, payload):
        if phase in {"profiling", "final_refit"}:
            search_scans.append(len(scans))

    def fit():
        model = SuperGLM(family=Tweedie(1.5), features={"x": PSpline(n_knots=6)})
        profile = model.estimate_p(
            X,
            y,
            sample_weight=weight,
            search_fit_mode=search_fit_mode,
            fit_mode=search_fit_mode,
            p_bounds=(1.4, 1.6),
            xatol=0.04,
            progress_callback=progress,
        )
        return model, profile

    if scan is not None:
        monkeypatch.setattr(convergence, "_scan_initial_precision", record_scan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual, profile = fit()
    (advisory,) = _advisories(caught)
    assert advisory.filename == __file__
    assert len(search_scans) > 2 and all(count == 0 for count in search_scans)
    assert scans == [len(y)]
    assert _findings(actual)["sample_weight_spread"]["rows"] == (12, 0)

    monkeypatch.setattr(convergence, "_scan_initial_precision", lambda *args: {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        control, control_profile = fit()
    assert not _advisories(caught)
    _assert_same_fit(actual, control, X)
    for name in ("p_hat", "phi_hat", "converged"):
        assert getattr(profile, name) == getattr(control_profile, name)
    pd.testing.assert_frame_equal(
        profile.evaluations, control_profile.evaluations, check_exact=True
    )


@pytest.mark.pins_behaviour
def test_fit_path_does_not_scan_its_regularization_candidates(monkeypatch):
    """Pin a silent lambda path with the same fits, stopping decisions and iterations."""
    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = np.sin(8.0 * x) + 0.1 * np.cos(125.0 * x)
    weight = np.ones_like(y)
    weight[12] = 0.0

    def fit():
        model = SuperGLM(family="gaussian", features={"x": Numeric()})
        path = model.fit_path(X, y, sample_weight=weight, lambda_seq=[0.1, 0.01, 0.001])
        return model, path

    scan = getattr(convergence, "_scan_initial_precision", None)
    scans = []

    def record_scan(weights, *args):
        scans.append(len(weights))
        return scan(weights, *args)

    with warnings.catch_warnings(record=True) as caught, monkeypatch.context() as patch:
        warnings.simplefilter("always")
        if scan is not None:
            patch.setattr(convergence, "_scan_initial_precision", record_scan)
        actual, path = fit()
        if scan is not None:
            patch.setattr(convergence, "_scan_initial_precision", lambda *args: {})
        control, control_path = fit()

    assert not scans
    assert not _advisories(caught)
    _assert_same_fit(actual, control, X)
    for name in (
        "lambda_seq",
        "coef_path",
        "intercept_path",
        "deviance_path",
        "n_iter_path",
        "converged_path",
        "edf_path",
        "separation_path",
    ):
        np.testing.assert_array_equal(getattr(path, name), getattr(control_path, name))
