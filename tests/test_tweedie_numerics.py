from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma, gammaln, ive, polygamma

import superglm._tweedie_series as series_module
import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.links import LogLink
from superglm.model.fit_ops import _compute_fit_stats
from superglm.profiling.tweedie import (
    _evaluate_tweedie_density,
    _prepare_tweedie_density,
    _tweedie_logpdf_impl,
    estimate_phi,
    tweedie_logpdf,
)


def _log_t_with_series_mode(a: float, mode: int) -> float:
    return float(np.log(mode + 1.0) + gammaln(a * (mode + 1.0)) - gammaln(a * mode))


@pytest.mark.parametrize(
    "value",
    [1.0e-6, 1.0e-3, 0.1, 0.5, 1.0, 2.0, 7.9, 8.0, 20.0, 1.0e3, 1.0e8],
)
def test_compiled_positive_digamma_matches_scipy(value: float) -> None:
    from superglm._tweedie_profile_kernel import _digamma_positive

    actual = _digamma_positive(value)

    assert actual == pytest.approx(float(digamma(value)), rel=3.0e-14, abs=3.0e-14)


@pytest.mark.parametrize(
    "value",
    [1.0e-6, 1.0e-3, 0.1, 0.5, 1.0, 2.0, 7.9, 8.0, 20.0, 1.0e3, 1.0e8],
)
def test_compiled_positive_trigamma_matches_scipy(value: float) -> None:
    from superglm._tweedie_profile_kernel import _trigamma_positive

    actual = _trigamma_positive(value)

    assert actual == pytest.approx(float(polygamma(1, value)), rel=3.0e-14, abs=3.0e-14)


def _exact_mean_nll(
    y: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
    p: float,
    log_phi: float,
) -> float:
    phi = float(np.exp(log_phi))
    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)
    logpdf = np.empty_like(y)
    logpdf[prepared.zero_mask] = (
        -prepared.zero_rate_numerator[prepared.zero_mask] * weights[prepared.zero_mask] / phi
    )
    log_t = prepared.positive_log_t_phi_independent - (prepared.a + 1.0) * log_phi
    log_sum, _, exact = series_module.tweedie_log_series(log_t, prepared.a)
    assert np.all(exact)
    logpdf[prepared.positive_mask] = (
        log_sum
        - prepared.positive_log_y
        + prepared.positive_canonical_c * weights[prepared.positive_mask] / phi
    )
    return -float(np.mean(logpdf))


@pytest.mark.parametrize("p", [1.05, 1.2, 1.5, 1.8, 1.95])
def test_compiled_exact_profile_statistics_match_density_and_finite_differences(
    p: float,
) -> None:
    from superglm._tweedie_profile_kernel import (
        PROFILE_KERNEL_OK,
        exact_profile_statistics,
    )

    y = np.array([0.0, 0.08, 0.7, 2.4, 8.0])
    mu = np.array([0.15, 0.12, 0.9, 2.0, 7.2])
    weights = np.array([0.4, 0.75, 1.0, 1.4, 2.2])
    log_phi = float(np.log(0.8))
    actual = exact_profile_statistics(y, mu, weights, p, log_phi)

    assert actual.status == PROFILE_KERNEL_OK
    assert actual.n_positive == 4
    assert actual.n_terms > 0
    expected = _exact_mean_nll(y, mu, weights, p, log_phi)
    assert actual.nll == pytest.approx(expected, rel=0.0, abs=2.0e-12)

    gradient_step = 2.0e-5
    curvature_step = 2.0e-4
    p_upper = _exact_mean_nll(y, mu, weights, p + gradient_step, log_phi)
    p_lower = _exact_mean_nll(y, mu, weights, p - gradient_step, log_phi)
    u_upper = _exact_mean_nll(y, mu, weights, p, log_phi + gradient_step)
    u_lower = _exact_mean_nll(y, mu, weights, p, log_phi - gradient_step)
    expected_gradient_p = (p_upper - p_lower) / (2.0 * gradient_step)
    expected_gradient_u = (u_upper - u_lower) / (2.0 * gradient_step)

    center = expected
    p_curvature_upper = _exact_mean_nll(y, mu, weights, p + curvature_step, log_phi)
    p_curvature_lower = _exact_mean_nll(y, mu, weights, p - curvature_step, log_phi)
    u_curvature_upper = _exact_mean_nll(y, mu, weights, p, log_phi + curvature_step)
    u_curvature_lower = _exact_mean_nll(y, mu, weights, p, log_phi - curvature_step)
    expected_hessian_pp = (p_curvature_upper - 2.0 * center + p_curvature_lower) / curvature_step**2
    expected_hessian_uu = (u_curvature_upper - 2.0 * center + u_curvature_lower) / curvature_step**2
    expected_hessian_pu = (
        _exact_mean_nll(y, mu, weights, p + curvature_step, log_phi + curvature_step)
        - _exact_mean_nll(y, mu, weights, p + curvature_step, log_phi - curvature_step)
        - _exact_mean_nll(y, mu, weights, p - curvature_step, log_phi + curvature_step)
        + _exact_mean_nll(y, mu, weights, p - curvature_step, log_phi - curvature_step)
    ) / (4.0 * curvature_step**2)

    assert actual.gradient_p == pytest.approx(expected_gradient_p, rel=2.0e-7, abs=2.0e-8)
    assert actual.gradient_log_phi == pytest.approx(
        expected_gradient_u,
        rel=2.0e-7,
        abs=2.0e-8,
    )
    assert actual.hessian_pp == pytest.approx(expected_hessian_pp, rel=2.0e-5, abs=2.0e-6)
    assert actual.hessian_log_phi_log_phi == pytest.approx(
        expected_hessian_uu,
        rel=2.0e-5,
        abs=2.0e-6,
    )
    assert actual.hessian_p_log_phi == pytest.approx(
        expected_hessian_pu,
        rel=2.0e-5,
        abs=2.0e-6,
    )


def test_compiled_exact_profile_statistics_reject_impossible_work_without_raising() -> None:
    from superglm._tweedie_profile_kernel import (
        PROFILE_KERNEL_WORK_LIMIT,
        exact_profile_statistics,
    )

    result = exact_profile_statistics(
        np.ones(4),
        np.ones(4),
        np.ones(4),
        1.4,
        float(np.log(1.0e-12)),
        max_terms=100,
        max_total_terms=200,
    )

    assert result.status == PROFILE_KERNEL_WORK_LIMIT
    assert np.isinf(result.nll)


def test_exact_series_starts_near_distant_mode(monkeypatch) -> None:
    calls = 0
    elements = 0
    real_gammaln = series_module.gammaln

    def counted(values):
        nonlocal calls, elements
        calls += 1
        elements += int(np.size(values))
        return real_gammaln(values)

    monkeypatch.setattr(series_module, "gammaln", counted)
    log_sum, expected_j, exact = series_module.tweedie_log_series(
        np.array([_log_t_with_series_mode(1.5, 90_000)]),
        1.5,
    )

    assert exact.tolist() == [True]
    assert np.isfinite(log_sum[0])
    assert expected_j[0] == pytest.approx(90_000.7, rel=2.0e-9)
    assert calls < 100
    assert elements < 20_000


def test_exact_series_reuses_gamma_base_for_shared_modes(monkeypatch) -> None:
    elements = 0
    real_gammaln = series_module.gammaln

    def counted(values):
        nonlocal elements
        elements += int(np.size(values))
        return real_gammaln(values)

    monkeypatch.setattr(series_module, "gammaln", counted)
    log_t = np.full(300, _log_t_with_series_mode(1.5, 10_000))

    log_sum, expected_j, exact = series_module.tweedie_log_series(log_t, 1.5)

    assert np.all(exact)
    assert np.all(np.isfinite(log_sum))
    np.testing.assert_array_equal(expected_j, np.full_like(expected_j, expected_j[0]))
    assert elements < 50_000


def test_series_budget_selection_does_not_reduce_each_row(monkeypatch) -> None:
    calls = 0
    real_sum = series_module.np.sum

    def counted_sum(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_sum(*args, **kwargs)

    monkeypatch.setattr(series_module.np, "sum", counted_sum)
    counts = np.arange(1, 801, dtype=np.int64)
    log_modes = np.linspace(0.0, 1.0, len(counts))
    values = log_modes + 2.0

    selected = series_module._select_budgeted_rows(
        counts,
        log_modes,
        values,
        max_total_terms=int(real_sum(counts)),
    )

    assert selected.size == counts.size
    assert calls <= 1


def test_exact_series_rejects_impossible_work_without_raising() -> None:
    log_sum, expected_j, exact = series_module.tweedie_log_series(
        np.array([70.0]),
        1.5,
        max_total_terms=1_000,
    )

    assert exact.tolist() == [False]
    assert np.isnan(log_sum[0])
    assert np.isnan(expected_j[0])


@pytest.mark.parametrize("p", [1.2, 1.4, 1.5, 1.8])
def test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics(p: float) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    y = np.exp(0.3 + 0.5 * x)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Tweedie(p=p),
        selection_penalty=0,
        features={"x": Numeric()},
    ).fit(frame, y)

    assert np.isfinite(model.result.phi)
    assert np.isfinite(model._fit_stats.log_likelihood)
    assert np.isfinite(model._fit_stats.null_log_likelihood)


def test_p15_large_argument_uses_finite_scaled_asymptotic() -> None:
    y = np.array([1.35])
    mu = np.array([1.3500001])
    weights = np.array([4.0])
    prepared = _prepare_tweedie_density(y, mu, 1.5, weights=weights)
    evaluated = _evaluate_tweedie_density(prepared, 1.0e-14, compute_score=True)
    step = 1.0e-5
    upper = _evaluate_tweedie_density(prepared, 1.0e-14 * np.exp(step)).logpdf[0]
    lower = _evaluate_tweedie_density(prepared, 1.0e-14 * np.exp(-step)).logpdf[0]
    finite_difference_score = -(upper - lower) / (2.0 * step)

    assert np.isfinite(evaluated.logpdf[0])
    assert evaluated.score_valid
    assert evaluated.log_phi_score is not None
    assert evaluated.log_phi_score[0] == pytest.approx(
        finite_difference_score,
        rel=1.0e-9,
        abs=1.0e-9,
    )
    assert evaluated.diagnostics.n_saddlepoint == 0


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_when_response_equals_extreme_mean(p: float) -> None:
    values = np.array([1.0e-20, 1.0, 1.0e12])

    actual = Tweedie(p).deviance_unit(values, values)

    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_pearson_phi_preserves_valid_tiny_means() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    p = 1.5
    expected = float(np.mean((y - mu) ** 2 / mu**p))

    actual = estimate_phi(y, mu, p)

    assert actual == pytest.approx(expected, rel=2.0e-15)


def test_pearson_phi_is_zero_for_equal_subnormal_scale_values() -> None:
    value = np.array([1.0e-300])

    assert estimate_phi(value, value, 1.5) == 0.0


def test_pearson_phi_preserves_finite_subnormal_scale_residual() -> None:
    mu = np.array([1.0e-300])
    y = np.array([1.0e-300 + 1.0e-310])
    expected = float(np.square((y - mu) / np.power(mu, 0.75))[0])

    assert estimate_phi(y, mu, 1.5) == pytest.approx(expected, rel=2.0e-15)


def test_profile_pearson_uses_same_unfloored_contributions() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    expected = estimate_phi(y, mu, 1.5)

    actual = tweedie_module._profile_phi_detailed(y, mu, 1.5, phi_method="pearson")

    assert actual.phi == pytest.approx(expected, rel=2.0e-15)


def test_fit_stats_pearson_is_zero_for_equal_subnormal_tweedie_values() -> None:
    value = np.array([1.0e-300])

    stats = _compute_fit_stats(
        value,
        value,
        np.ones(1),
        None,
        Tweedie(1.5),
        LogLink(),
        1.0,
        null_mu=value,
    )

    assert stats.pearson_chi2 == 0.0


@pytest.mark.parametrize(
    ("y", "mu", "phi", "p", "weight", "expected"),
    [
        (0.017, 0.02, 0.004, 1.0001, 0.4, -242.08168865838033),
        (9000.0, 10000.0, 200.0, 1.01, 0.5, -8.636743836168788),
        (0.22, 0.15, 0.125, 1.25, 4.0, 1.0417074672233964),
        (0.03, 4.0, 0.7, 1.5, 0.2, -2.2657661799346407),
        (80.0, 50.0, 5.3, 1.75, 4.0, -5.206967741958142),
        (0.0002, 0.001, 1.3, 1.99, 0.1, 5.575914834713504),
        (
            0.04564326798684731,
            2.859891821890267,
            0.10602153698295053,
            1.05,
            1.0,
            -25.217701008861372,
        ),
    ],
)
def test_public_density_matches_neutral_high_precision_reference(
    y: float,
    mu: float,
    phi: float,
    p: float,
    weight: float,
    expected: float,
) -> None:
    actual = tweedie_logpdf(
        np.array([y]),
        np.array([mu]),
        phi,
        p,
        weights=np.array([weight]),
    )

    assert actual[0] == pytest.approx(expected, rel=0.0, abs=2.5e-9)


def test_default_density_uses_exact_series_instead_of_saddlepoint() -> None:
    actual, diagnostics = _tweedie_logpdf_impl(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
    )

    assert actual[0] == pytest.approx(-25.217701008861372, abs=2.5e-9)
    assert diagnostics.n_series == 1
    assert diagnostics.n_saddlepoint == 0


def test_p15_density_uses_stable_exact_bessel_form_at_tiny_phi() -> None:
    y = np.array([1.35])
    mu = np.array([1.35001])
    weights = np.array([4.0])
    phi = 4.0e-8
    root_y = np.sqrt(y)
    root_mu = np.sqrt(mu)
    bessel_argument = 4.0 * weights * root_y / phi
    expected = (
        np.log(2.0 * weights / (phi * root_y))
        + np.log(ive(1, bessel_argument))
        - 2.0 * weights * np.square(root_y - root_mu) / (phi * root_mu)
    )

    actual, diagnostics = _tweedie_logpdf_impl(
        y,
        mu,
        phi,
        1.5,
        weights=weights,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-13)
    assert diagnostics.n_series == 1
    assert diagnostics.n_saddlepoint == 0


def test_exact_series_log_phi_score_matches_finite_difference() -> None:
    y = np.array([0.04564326798684731, 9000.0])
    mu = np.array([2.859891821890267, 10000.0])
    weights = np.array([1.0, 0.5])
    p = 1.05
    phi = 0.10602153698295053
    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)

    evaluated = _evaluate_tweedie_density(prepared, phi, compute_score=True)
    step = 1.0e-5
    upper = _evaluate_tweedie_density(prepared, phi * np.exp(step)).logpdf
    lower = _evaluate_tweedie_density(prepared, phi * np.exp(-step)).logpdf
    finite_difference = -float(np.mean(upper - lower)) / (2.0 * step)

    assert evaluated.diagnostics.n_series == 2
    assert evaluated.score_valid
    assert evaluated.log_phi_score is not None
    assert float(np.mean(evaluated.log_phi_score)) == pytest.approx(
        finite_difference,
        rel=2.0e-6,
        abs=2.0e-7,
    )


def test_profile_cache_prefers_shared_exact_series_to_wright(monkeypatch) -> None:
    y = np.full(64, 3.6100536453396823)
    mu = np.full(64, 2.358259687964909)
    prepared = _prepare_tweedie_density(y, mu, 1.2)

    def unexpected_wright(*args, **kwargs):
        del args, kwargs
        raise AssertionError("profile should use feasible shared exact series first")

    monkeypatch.setattr(tweedie_module, "wright_bessel", unexpected_wright)
    point = tweedie_module._PhiEvaluationCache(prepared).evaluate(
        float(np.log(0.8)),
        compute_score=True,
    )

    assert point.objective_finite
    assert point.score_valid
    assert point.diagnostics.n_series == len(y)
    assert point.diagnostics.n_saddlepoint == 0
    assert point.nll == pytest.approx(1.8957859435896154, abs=2.5e-13)


def test_tweedie_fit_stats_reuses_one_density_normalizer(monkeypatch) -> None:
    y = np.array([0.0, 0.3, 1.2, 4.5])
    mu = np.array([0.2, 0.5, 1.5, 3.7])
    null_mu = np.full_like(y, 1.1)
    weights = np.array([0.4, 0.8, 1.2, 1.8])
    family = Tweedie(1.55)
    expected_ll = float(np.sum(tweedie_logpdf(y, mu, 0.8, 1.55, weights=weights)))
    expected_null_ll = float(np.sum(tweedie_logpdf(y, null_mu, 0.8, 1.55, weights=weights)))
    real_evaluate = tweedie_module._evaluate_tweedie_density
    calls = 0

    def counted(
        prepared,
        phi,
        *,
        compute_score=False,
        series_max_total_terms=tweedie_module._PROFILE_SERIES_MAX_TOTAL_TERMS,
    ):
        nonlocal calls
        calls += 1
        return real_evaluate(
            prepared,
            phi,
            compute_score=compute_score,
            series_max_total_terms=series_max_total_terms,
        )

    monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", counted)

    stats = _compute_fit_stats(
        y,
        mu,
        weights,
        None,
        family,
        LogLink(),
        0.8,
        null_mu=null_mu,
    )

    assert calls == 1
    assert stats.log_likelihood == pytest.approx(expected_ll, abs=1.0e-11)
    assert stats.null_log_likelihood == pytest.approx(expected_null_ll, abs=1.0e-11)
