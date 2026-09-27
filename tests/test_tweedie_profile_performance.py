"""Tweedie maximum-likelihood dispersion and power: reference agreement and pass counts.

No wall-clock assertion. The end-to-end inner-solve comparison is checked here
on one run per case and mode; its timed repeat medians are
``benchmarks/tweedie_profile_end_to_end.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from benchmarks.tweedie_profile_end_to_end import (
    _aggregate_end_to_end_runs,
    _counterbalanced_mode_orders,
    _end_to_end_profile_cases,
    _run_end_to_end_profile_once,
)
from scipy.optimize import minimize_scalar

import superglm._tweedie as density_module
from superglm import SuperGLM, generate_tweedie_cpg, tweedie_logpdf
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.profiling.tweedie import profile_phi_at

EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class _PhiFixture:
    name: str
    y: np.ndarray
    mu: np.ndarray
    p: float
    weights: np.ndarray


def _routine_exact_fixture() -> _PhiFixture:
    """A weighted fixture with an atom at zero."""
    return _PhiFixture(
        name="routine-weighted-zero",
        y=np.array([0.0, 0.3, 1.2, 4.5, 8.0]),
        mu=np.array([0.2, 0.5, 1.5, 3.7, 7.0]),
        p=1.55,
        weights=np.array([0.4, 0.8, 1.2, 1.8, 2.4]),
    )


def _large_routine_exact_fixture() -> _PhiFixture:
    """The same rows tiled to n = 3000."""
    base = _routine_exact_fixture()
    return _PhiFixture(
        name="routine-weighted-zero-n3000",
        y=np.tile(base.y, 600),
        mu=np.tile(base.mu, 600),
        p=base.p,
        weights=np.tile(base.weights, 600),
    )


@pytest.mark.parametrize(
    "fixture", [_routine_exact_fixture(), _large_routine_exact_fixture()], ids=lambda f: f.name
)
def test_newton_phi_matches_a_tight_bounded_reference_in_fewer_passes(fixture):
    """The Newton dispersion solve is the minimiser of the public density's mean NLL."""
    y, mu, w, p = fixture.y, fixture.mu, fixture.weights, fixture.p

    def mean_nll(u):
        return -float(np.mean(tweedie_logpdf(y, mu, math.exp(u), p, weights=w)))

    xatol = 1e-11
    reference = minimize_scalar(
        mean_nll, bounds=(-10.0, 10.0), method="bounded", options={"xatol": xatol, "maxiter": 500}
    )
    solved = profile_phi_at(y, mu, w, p)
    u = math.log(solved.phi)
    assert reference.success
    # A row's log density rounds to 32 eps of its terms' magnitudes, log W,
    # log y, c w / phi and w d / (2 phi) (the series and density bounds in
    # test_tweedie_series and test_tweedie_nb_characterisation). A value-only
    # search cannot place a minimum of curvature c more finely than
    # sqrt(2 round-off / c); fminbound also stops once its bracket is within
    # 2 (sqrt(eps)|x| + xatol/3), and Newton stops on a step of at most 1e-8.
    rows = density_module.TweedieRows.prepare(y, w, p)
    canonical = rows.saturated_canonical / solved.phi
    log_w = rows.row_saturated(solved.phi)[0] + rows.log_y - canonical
    half_deviance = w * density_module.tweedie_unit_deviance(y, mu, p) / (2.0 * solved.phi)
    magnitudes = np.sum(np.abs(log_w) + np.abs(rows.log_y) + np.abs(canonical)) + np.sum(
        half_deviance
    )
    round_off = 32.0 * EPS * magnitudes / y.size
    curvature = solved.curvature / y.size
    du = (
        math.sqrt(2.0 * round_off / curvature)
        + 2.0 * (math.sqrt(EPS) * abs(reference.x) + xatol / 3.0)
        + 1e-8
    )
    assert abs(u - reference.x) <= du
    # The NLL is second order in the location error, plus that round-off.
    nll_bound = 0.5 * curvature * du**2 + round_off
    assert abs(solved.criterion / y.size - reference.fun) <= nll_bound
    assert abs(mean_nll(u) - reference.fun) <= nll_bound
    # Each Newton pass and each reference value is one series pass.
    assert solved.n_passes < reference.nfev


def test_generator_cpg_moments_and_zero_mass_match_compound_poisson_formulas():
    """One deterministic draw validates all moments using the current generator API."""
    rng = np.random.default_rng(42)
    n = 20_000
    mu, phi, p = 10.0, 3.0, 1.6
    y = generate_tweedie_cpg(n, mu=mu, phi=phi, p=p, rng=rng)

    poisson_rate = mu ** (2.0 - p) / ((2.0 - p) * phi)
    gamma_shape = (2.0 - p) / (p - 1.0)
    gamma_scale = phi * (p - 1.0) * mu ** (p - 1.0)
    expected_mean = poisson_rate * gamma_shape * gamma_scale
    expected_variance = poisson_rate * gamma_shape * (1.0 + gamma_shape) * gamma_scale**2
    expected_zero_mass = float(np.exp(-poisson_rate))

    assert expected_mean == pytest.approx(mu)
    assert expected_variance == pytest.approx(phi * mu**p)
    np.testing.assert_allclose(y.mean(), expected_mean, rtol=0.04)
    np.testing.assert_allclose(y.var(), expected_variance, rtol=0.12)
    np.testing.assert_allclose(np.mean(y == 0.0), expected_zero_mass, atol=0.015)


@pytest.mark.parametrize(
    ("p", "phi", "seed"),
    [
        pytest.param(1.05, 1.05, 20260715, id="near-one"),
        pytest.param(1.95, 20.0, 20260716, id="near-two"),
    ],
)
def test_generator_near_boundary_moments_and_zero_mass(p, phi, seed):
    """Boundary CPG draws retain their analytic moments without changing the API."""
    rng = np.random.default_rng(seed)
    n = 50_000
    mu = 1.0
    y = generate_tweedie_cpg(n, mu=mu, phi=phi, p=p, rng=rng)
    poisson_rate = mu ** (2.0 - p) / ((2.0 - p) * phi)

    np.testing.assert_allclose(y.mean(), mu, rtol=0.06)
    np.testing.assert_allclose(y.var(), phi * mu**p, rtol=0.20)
    np.testing.assert_allclose(np.mean(y == 0.0), np.exp(-poisson_rate), atol=0.015)


def test_zero_heavy_exact_mle_p_phi_recovery():
    """A high-zero-rate sample retains practical p/phi Monte Carlo recovery."""
    rng = np.random.default_rng(20260717)
    n = 1_500
    p_true, phi_true = 1.75, 15.0
    x = rng.normal(size=n)
    mu = np.exp(1.0 + 0.3 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=phi_true, p=p_true, rng=rng)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": Numeric()},
    )

    result = model.estimate_p(X, y, p_bounds=(1.1, 1.9), xatol=1e-3)

    assert np.mean(y == 0.0) >= 0.65
    assert result.converged
    assert result.warnings == []
    # Tolerances exceed the deterministic sample's Monte Carlo error without
    # pretending finite-sample profile estimates equal generating parameters.
    np.testing.assert_allclose(result.p_hat, p_true, atol=0.08)
    np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.15)


def test_end_to_end_aggregation_rejects_non_deterministic_integer_fields():
    """Repeat aggregation must not conceal changing search/pass counts."""
    run = {
        "case": "fit-numeric",
        "fit_mode": "fit",
        "mode": "production-analytic-inner",
        "n_observations": 600,
        "outer_evaluations": 8,
        "inner_density_passes": 45,
        "p_hat": 1.6,
        "phi_hat": 2.5,
        "nll": 2.3,
        "elapsed_seconds": 0.2,
        "converged": True,
        "local_inner_optimizer_success": None,
    }
    changed = {**run, "inner_density_passes": 46}

    with pytest.raises(AssertionError, match="inner_density_passes"):
        _aggregate_end_to_end_runs([run, changed, run, changed])


def test_end_to_end_timed_mode_order_is_counterbalanced():
    """Four timed repeats must give each mode two first-position runs."""
    production = "production-analytic-inner"
    reference = "reference-bounded-inner"

    assert _counterbalanced_mode_orders(4) == (
        (production, reference),
        (reference, production),
        (production, reference),
        (reference, production),
    )
    with pytest.raises(ValueError, match="even number.*at least four"):
        _counterbalanced_mode_orders(3)


@pytest.mark.slow
def test_end_to_end_analytic_inner_matches_bounded_inner_reference():
    """Changing only the inner phi solve leaves the public outer search unchanged.

    The two solves minimise the same criterion and agree in log phi to
    fminbound's bracket plus Newton's 1e-8 step, under 1e-6 here. That moves a
    candidate's NLL by O(du^2), orders below what could change a comparison or
    a parabolic step of Brent's path, so the path is the same and p_hat, the
    published phi and its NLL agree to that solve difference.
    """
    modes = ("production-analytic-inner", "reference-bounded-inner")
    for case in _end_to_end_profile_cases():
        production, reference = (_run_end_to_end_profile_once(mode, case) for mode in modes)
        for row in (production, reference):
            assert 300 <= row["n_observations"] <= 1_000
            assert row["outer_evaluations"] > 0
            assert row["inner_density_passes"] >= row["outer_evaluations"]
            assert np.isfinite(row["p_hat"])
            assert np.isfinite(row["phi_hat"]) and row["phi_hat"] > 0.0
            assert np.isfinite(row["nll"])
            assert row["converged"] is True

        assert production["local_inner_optimizer_success"] is None
        assert reference["local_inner_optimizer_success"] is True
        assert production["inner_density_passes"] < reference["inner_density_passes"]
        assert production["outer_evaluations"] == reference["outer_evaluations"]
        assert production["p_hat"] == pytest.approx(reference["p_hat"], abs=1e-9)
        assert math.log(production["phi_hat"]) == pytest.approx(
            math.log(reference["phi_hat"]), abs=1e-6
        )
        assert production["nll"] == pytest.approx(reference["nll"], abs=1e-10)


def test_ten_thousand_row_likelihood_pair_is_vectorized(monkeypatch) -> None:
    """The fitted and null densities share one series pass over every positive row."""
    n = 10_000
    y = np.geomspace(0.01, 100.0, n)
    mu = y * np.exp(np.linspace(-0.2, 0.2, n))
    null_mu = np.full(n, 1.0)
    weights = np.geomspace(0.5, 2.0, n)
    real_series = density_module.series_moments
    batch_sizes: list[int] = []

    def counted_series(log_t, a):
        batch_sizes.append(len(log_t))
        return real_series(log_t, a)

    monkeypatch.setattr(density_module, "series_moments", counted_series)

    fitted, null = density_module.tweedie_logpdf_pair(y, mu, null_mu, 0.8, 1.5, weights=weights)

    assert fitted.shape == null.shape == (n,)
    assert batch_sizes == [n]


class TestREMLProfileSearchOverhead:
    """The REML power search must not re-pay published-model bookkeeping per step."""

    @staticmethod
    def _model_and_data():
        rng = np.random.default_rng(11)
        # Below the 100,000-row auto-validation ceiling every fit validates
        # unless told to skip, so the row count only has to stay under it.
        n = 400
        levels = [f"L{j:02d}" for j in range(8)]
        idx = rng.integers(0, len(levels), n)
        frame = pd.DataFrame({"band": np.array(levels)[idx], "z": rng.uniform(0, 10, n)})
        weights = rng.uniform(0.2, 1.0, n)
        mu = np.exp(-1.0 + 0.06 * idx)
        y = np.where(rng.random(n) < 0.6, 0.0, rng.gamma(1.5, mu, n))
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            family=Tweedie(p=1.5),
            features={"band": Categorical(), "z": Spline(kind="ps", k=6)},
        )
        return model, frame, y, weights

    def test_search_fits_skip_published_parity_validation(self):
        """Every power step ran the post-fit parity check, which certifies the
        PUBLISHED runtime state. Search fits are thrown away -- only the final
        refit is published -- so paying it per step is pure overhead. On a real
        8-feature model this was 31.3s of a 119.3s run.
        """
        import superglm.model.runtime_canonicalize as canon

        model, frame, y, weights = self._model_and_data()
        calls: list[bool] = []
        original = canon.canonicalize_fitted_model

        def counting(model_arg, *args, validate=True, **kw):
            calls.append(bool(validate))
            return original(model_arg, *args, validate=validate, **kw)

        with patch.object(canon, "canonicalize_fitted_model", counting):
            result = model.estimate_p(frame, y, sample_weight=weights, fit_mode="reml")

        assert result.p_hat > 1.0
        validated = sum(calls)
        assert len(calls) > 2, "expected several fits (search steps plus the final refit)"
        assert validated == 1, (
            f"{validated} of {len(calls)} canonicalisations validated; only the published "
            "final refit should validate, not each search step"
        )
        # The saving must come from the throwaway fits, never from the model the
        # caller keeps: that one is published and still has to be certified.
        assert calls[-1] is True, "the final published refit must still validate"
        assert not any(calls[:-1]), "no search step should validate"
