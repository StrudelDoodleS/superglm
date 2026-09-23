"""Tweedie exact-MLE validation and density-pass benchmark characterizations.

The benchmark is correctness-first and has no wall-clock assertion.  Run it
with visible repeat-median output via::

    PYTHONPATH=src uv run pytest tests/test_tweedie_profile_performance.py -m slow -s

The end-to-end inner-search comparison is checked here on one run per case and
mode; its timed repeat medians are ``benchmarks/tweedie_profile_end_to_end.py``.
"""

from __future__ import annotations

import statistics
import time
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
from scipy.optimize import OptimizeResult, minimize_scalar

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.profiling.tweedie import generate_tweedie_cpg


@dataclass(frozen=True)
class _PhiFixture:
    name: str
    y: np.ndarray
    mu: np.ndarray
    p: float
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class _CountedProfile:
    result: object
    density_passes: int
    score_passes: int
    value_only_passes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _BoundedReference:
    optimizer: OptimizeResult
    density_passes: int
    objective_calls: int
    winner_diagnostics: object
    elapsed_seconds: float


def _routine_exact_fixture() -> _PhiFixture:
    """A weighted exact-branch fixture with an atom at zero and a robust pass margin."""
    return _PhiFixture(
        name="routine-weighted-zero-exact",
        y=np.array([0.0, 0.3, 1.2, 4.5, 8.0]),
        mu=np.array([0.2, 0.5, 1.5, 3.7, 7.0]),
        p=1.55,
        weights=np.array([0.4, 0.8, 1.2, 1.8, 2.4]),
    )


def _large_routine_exact_fixture() -> _PhiFixture:
    """A size-scaled exact profile for per-observation density-cost characterization."""
    base = _routine_exact_fixture()
    assert base.weights is not None
    return _PhiFixture(
        name="routine-exact-n3000",
        y=np.tile(base.y, 600),
        mu=np.tile(base.mu, 600),
        p=base.p,
        weights=np.tile(base.weights, 600),
    )


def _difficult_branch_fixture() -> _PhiFixture:
    """A deterministic branch-jump case that exercises fallback provenance."""
    return _PhiFixture(
        name="difficult-branch-fallback",
        p=1.0181533410437358,
        y=np.array([1.81787899, 11275.9262, 0.0, 0.00306563885, 0.0000232882792, 1.18207511]),
        mu=np.array(
            [
                0.0000253947806,
                44091.7359,
                198.869667,
                0.000051937831,
                331.859132,
                0.0054422757,
            ]
        ),
        weights=np.array([83.2444169, 0.17590785, 2.31976211, 463.433307, 2.50852264, 0.416322332]),
    )


def _counted_production_profile(fixture: _PhiFixture) -> _CountedProfile:
    """Run production while counting only real vector-density cache misses."""
    real_evaluate = tweedie_module._evaluate_tweedie_density
    calls: list[bool] = []

    def counted_evaluate(prepared, phi, *, compute_score=False, **kwargs):
        calls.append(bool(compute_score))
        return real_evaluate(prepared, phi, compute_score=compute_score, **kwargs)

    started = time.perf_counter()
    with patch.object(tweedie_module, "_evaluate_tweedie_density", counted_evaluate):
        result = tweedie_module._profile_phi_detailed(
            fixture.y,
            fixture.mu,
            fixture.p,
            weights=fixture.weights,
            phi_method="mle",
        )
    elapsed = time.perf_counter() - started
    return _CountedProfile(
        result=result,
        density_passes=len(calls),
        score_passes=sum(calls),
        value_only_passes=len(calls) - sum(calls),
        elapsed_seconds=elapsed,
    )


def _bounded_value_only_reference(fixture: _PhiFixture) -> _BoundedReference:
    """Profile one prepared production objective with an independent bounded search."""
    started = time.perf_counter()
    prepared = tweedie_module._prepare_tweedie_density(
        fixture.y,
        fixture.mu,
        fixture.p,
        weights=fixture.weights,
    )
    real_evaluate = tweedie_module._evaluate_tweedie_density
    cache: dict[float, tuple[float, object]] = {}
    objective_calls = 0

    def objective(log_phi):
        nonlocal objective_calls
        objective_calls += 1
        key = float(log_phi)
        cached = cache.get(key)
        if cached is not None:
            return cached[0]
        try:
            evaluation = real_evaluate(prepared, float(np.exp(key)), compute_score=False)
        except FloatingPointError:
            nll = np.inf
            diagnostics = tweedie_module._TweedieLogpdfDiagnostics(
                n_positive=int(prepared.positive_indices.size),
                n_series=int(prepared.positive_indices.size),
            )
        else:
            nll = -float(np.mean(evaluation.logpdf))
            diagnostics = evaluation.diagnostics
        cache[key] = (nll, diagnostics)
        return nll

    optimizer = minimize_scalar(
        objective,
        bounds=(tweedie_module._LOG_PHI_LOWER_BOUND, tweedie_module._LOG_PHI_UPPER_BOUND),
        method="bounded",
        options={"xatol": 1e-11, "maxiter": 500},
    )
    elapsed = time.perf_counter() - started
    # Deliberate cached replays prove callback counts are not density-pass counts.
    density_passes = len(cache)
    objective(float(optimizer.x))
    objective(float(optimizer.x))
    assert len(cache) == density_passes
    winner_diagnostics = cache[float(optimizer.x)][1]
    return _BoundedReference(
        optimizer=optimizer,
        density_passes=density_passes,
        objective_calls=objective_calls,
        winner_diagnostics=winner_diagnostics,
        elapsed_seconds=elapsed,
    )


def test_weighted_zero_exact_mle_uses_fewer_density_passes_than_tight_bounded_reference():
    """The analytic exact-branch search agrees with a tight prepared-objective reference."""
    fixture = _routine_exact_fixture()
    production = _counted_production_profile(fixture)
    reference = _bounded_value_only_reference(fixture)
    result = production.result

    assert fixture.name == "routine-weighted-zero-exact"
    assert fixture.weights is not None
    assert fixture.y[0] == 0.0
    assert reference.optimizer.success
    assert result.objective_finite
    assert result.converged
    assert result.optimizer == "brentq"
    assert not result.used_fallback
    assert result.n_fallback_evaluations == 0
    assert not result.lower_boundary
    assert not result.upper_boundary
    assert result.diagnostics.n_saddlepoint == 0
    assert reference.winner_diagnostics.n_saddlepoint == 0
    assert result.score is not None and abs(result.score) <= 1e-6

    assert result.n_evaluations == production.density_passes
    assert result.n_score_evaluations == production.score_passes
    assert result.n_value_only_evaluations == production.value_only_passes
    assert result.n_evaluations == result.n_score_evaluations + result.n_value_only_evaluations
    assert reference.objective_calls > reference.density_passes
    # Preserve the semantic ordering: SciPy's exact bounded-search pass count
    # varies across Python/platform builds even when the optimum is unchanged.
    assert production.density_passes < reference.density_passes
    np.testing.assert_allclose(result.nll, reference.optimizer.fun, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.log(result.phi), reference.optimizer.x, rtol=0.0, atol=5e-6)


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

    result = model.estimate_p(
        X,
        y,
        p_bounds=(1.1, 1.9),
        phi_method="mle",
        method="brent",
        xatol=1e-3,
    )

    assert np.mean(y == 0.0) >= 0.65
    assert result.converged
    assert result.density_exact
    assert result.n_saddlepoint == 0
    assert not result.phi_used_fallback
    # Tolerances exceed the deterministic sample's Monte Carlo error without
    # pretending finite-sample profile estimates equal generating parameters.
    np.testing.assert_allclose(result.p_hat, p_true, atol=0.08)
    np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.15)


@pytest.mark.parametrize(
    ("p", "phi", "seed", "expect_fallback"),
    [
        pytest.param(1.05, 2.5, 7105, False, id="near-one"),
        pytest.param(1.95, 1.2, 7195, False, id="near-two"),
    ],
)
def test_near_boundary_phi_profile_reports_honest_provenance(p, phi, seed, expect_fallback):
    """Boundary fixtures certify diagnostics, not unrealistic parameter recovery."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, 240)
    mu = np.exp(1.0 + 0.3 * x)
    y = generate_tweedie_cpg(len(x), mu=mu, phi=phi, p=p, rng=rng)

    result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
    density = tweedie_module._classify_density_diagnostics(p, result.diagnostics)

    assert result.objective_finite
    assert np.isfinite(result.nll)
    assert np.isfinite(result.phi) and result.phi > 0.0
    assert result.n_evaluations == result.n_score_evaluations + result.n_value_only_evaluations
    assert density.n_positive == np.count_nonzero(y > 0.0)
    assert density.n_saddlepoint == result.diagnostics.n_saddlepoint
    assert density.fraction == pytest.approx(density.n_saddlepoint / max(density.n_positive, 1))
    assert result.used_fallback is expect_fallback
    if expect_fallback:
        assert result.optimizer == "bounded"
        assert result.branch_switch_detected
        assert result.fallback_reason
        assert result.n_fallback_evaluations > 0
        assert not result.converged
        assert not density.exact
        assert density.near_power_boundary
    else:
        assert result.optimizer == "brentq"
        assert result.n_fallback_evaluations == 0
        assert result.fallback_reason is None
        assert result.converged
        assert density.exact
        assert density.fraction == 0.0


def _benchmark_fixture(fixture: _PhiFixture, *, repeats: int) -> dict[str, object]:
    """Benchmark one fixture after warm-up, retaining diagnostic provenance."""
    if repeats < 1:
        raise ValueError("repeats must be positive")

    _counted_production_profile(fixture)  # untimed warm-up
    _bounded_value_only_reference(fixture)
    runs = []
    reference_runs = []
    for _ in range(repeats):
        runs.append(_counted_production_profile(fixture))
        reference_runs.append(_bounded_value_only_reference(fixture))
    result = runs[-1].result
    reference = reference_runs[-1]
    return {
        "fixture": fixture.name,
        "n_observations": len(fixture.y),
        "p": fixture.p,
        "phi": result.phi,
        "log_phi": float(np.log(result.phi)),
        "nll": result.nll,
        "density_passes": int(statistics.median(run.density_passes for run in runs)),
        "score_passes": int(statistics.median(run.score_passes for run in runs)),
        "value_only_passes": int(statistics.median(run.value_only_passes for run in runs)),
        "fallback_count": result.n_fallback_evaluations,
        "saddle_fraction": result.diagnostics.saddlepoint_fraction,
        "elapsed_median_seconds": statistics.median(run.elapsed_seconds for run in runs),
        "reference_phi": float(np.exp(reference.optimizer.x)),
        "reference_log_phi": float(reference.optimizer.x),
        "reference_nll": float(reference.optimizer.fun),
        "reference_density_passes": int(
            statistics.median(run.density_passes for run in reference_runs)
        ),
        "reference_elapsed_median_seconds": statistics.median(
            run.elapsed_seconds for run in reference_runs
        ),
        "reference_success": bool(reference.optimizer.success),
        "used_fallback": result.used_fallback,
        "converged": result.converged,
    }


def run_tweedie_phi_profile_benchmark(*, repeats: int = 5) -> list[dict[str, object]]:
    """Return repeat-median diagnostic rows for routine and difficult fixtures."""
    return [
        _benchmark_fixture(_routine_exact_fixture(), repeats=repeats),
        _benchmark_fixture(_difficult_branch_fixture(), repeats=repeats),
    ]


def run_large_routine_phi_profile_benchmark(*, repeats: int = 3) -> dict[str, object]:
    """Characterize exact-branch density cost at a size-scaled n=3000."""
    return _benchmark_fixture(_large_routine_exact_fixture(), repeats=repeats)


def test_end_to_end_aggregation_rejects_non_deterministic_integer_fields():
    """Repeat aggregation must not conceal changing search/pass counts."""
    run = {
        "case": "fit-numeric",
        "fit_mode": "fit",
        "mode": "production-analytic-inner",
        "n_observations": 600,
        "outer_evaluations": 8,
        "inner_density_passes": 173,
        "phi_fallback_count": 0,
        "p_hat": 1.6,
        "phi_hat": 2.5,
        "nll": 2.3,
        "saddle_fraction": 0.0,
        "elapsed_seconds": 0.2,
        "converged": True,
        "outer_converged": True,
        "objective_finite": True,
        "local_inner_optimizer_success": None,
    }
    changed = {**run, "inner_density_passes": 174}

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
def test_tweedie_phi_profile_benchmark_report():
    """Print repeat medians under ``pytest -s`` without enforcing wall time."""
    rows = run_tweedie_phi_profile_benchmark(repeats=5)

    for row in rows:
        print(
            "Tweedie phi benchmark "
            f"fixture={row['fixture']} n={row['n_observations']} p={row['p']:.8g} "
            f"phi={row['phi']:.8g} log_phi={row['log_phi']:.8g} "
            f"NLL={row['nll']:.8g} density_passes={row['density_passes']} "
            f"score_passes={row['score_passes']} "
            f"value_only_passes={row['value_only_passes']} "
            f"fallback_count={row['fallback_count']} "
            f"saddle_fraction={row['saddle_fraction']:.6f} "
            f"elapsed_median_s={row['elapsed_median_seconds']:.6f} "
            f"reference_phi={row['reference_phi']:.8g} "
            f"reference_log_phi={row['reference_log_phi']:.8g} "
            f"reference_NLL={row['reference_nll']:.8g} "
            f"reference_density_passes={row['reference_density_passes']} "
            f"reference_elapsed_median_s={row['reference_elapsed_median_seconds']:.6f}"
        )

    routine, difficult = rows
    assert routine["fixture"] == "routine-weighted-zero-exact"
    assert routine["used_fallback"] is False
    assert routine["converged"] is True
    assert routine["saddle_fraction"] == 0.0
    assert routine["density_passes"] < routine["reference_density_passes"]
    assert routine["nll"] == pytest.approx(routine["reference_nll"], abs=1e-10)
    assert difficult["fixture"] == "difficult-branch-fallback"
    assert difficult["used_fallback"] is True
    assert difficult["fallback_count"] > 0
    assert difficult["saddle_fraction"] >= 0.0
    assert difficult["converged"] is False


@pytest.mark.slow
def test_large_routine_phi_profile_timing_characterization():
    """Report size-scaled density-cost medians without asserting a wall-clock winner."""
    row = run_large_routine_phi_profile_benchmark(repeats=3)

    print(
        "Tweedie phi large-routine benchmark "
        f"fixture={row['fixture']} n={row['n_observations']} p={row['p']:.8g} "
        f"production_phi={row['phi']:.8g} production_log_phi={row['log_phi']:.8g} "
        f"production_NLL={row['nll']:.8g} "
        f"production_density_passes={row['density_passes']} "
        f"production_score_passes={row['score_passes']} "
        f"production_value_only_passes={row['value_only_passes']} "
        f"production_fallback_count={row['fallback_count']} "
        f"production_saddle_fraction={row['saddle_fraction']:.6f} "
        f"production_elapsed_median_s={row['elapsed_median_seconds']:.6f} "
        f"reference_phi={row['reference_phi']:.8g} "
        f"reference_log_phi={row['reference_log_phi']:.8g} "
        f"reference_NLL={row['reference_nll']:.8g} "
        f"reference_density_passes={row['reference_density_passes']} "
        f"reference_elapsed_median_s={row['reference_elapsed_median_seconds']:.6f}"
    )
    if (
        row["elapsed_median_seconds"] > row["reference_elapsed_median_seconds"]
        and row["density_passes"] < row["reference_density_passes"]
    ):
        print(
            "Timing characterization: production is slower here despite fewer density "
            "passes; its analytic search computes an additional Wright value for exact "
            "score evaluations."
        )

    assert row["fixture"] == "routine-exact-n3000"
    assert row["n_observations"] == 3_000
    assert row["reference_success"] is True
    assert row["used_fallback"] is False
    assert row["fallback_count"] == 0
    assert row["converged"] is True
    assert row["saddle_fraction"] == 0.0
    assert row["density_passes"] == row["score_passes"] + row["value_only_passes"]
    assert row["nll"] == pytest.approx(row["reference_nll"], abs=1e-10)
    assert row["log_phi"] == pytest.approx(row["reference_log_phi"], abs=5e-6)


@pytest.mark.slow
def test_end_to_end_analytic_inner_matches_bounded_inner_reference():
    """Changing only the inner phi optimizer leaves the public outer search unchanged.

    One run per case and mode certifies the comparison.  The warmed,
    counterbalanced timing medians are a benchmark, not a test, and live in
    ``benchmarks/tweedie_profile_end_to_end.py``.
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
            assert 0.0 <= row["saddle_fraction"] <= 1.0
            assert row["phi_fallback_count"] >= 0
            assert row["outer_converged"] is True
            assert row["objective_finite"] is True

        assert production["converged"] is True
        assert production["local_inner_optimizer_success"] is None
        assert reference["converged"] is False
        assert reference["local_inner_optimizer_success"] is True
        assert production["p_hat"] == pytest.approx(reference["p_hat"], abs=5e-3)
        assert production["phi_hat"] == pytest.approx(reference["phi_hat"], rel=5e-3)
        assert production["nll"] == pytest.approx(reference["nll"], abs=1e-7)


def test_ten_thousand_row_likelihood_pair_is_vectorized(monkeypatch) -> None:
    n = 10_000
    y = np.geomspace(0.01, 100.0, n)
    mu = y * np.exp(np.linspace(-0.2, 0.2, n))
    null_mu = np.full(n, 1.0)
    weights = np.geomspace(0.5, 2.0, n)
    real_series = tweedie_module.tweedie_log_series
    batch_sizes: list[int] = []

    def counted_series(log_t, a, **kwargs):
        batch_sizes.append(len(log_t))
        return real_series(log_t, a, **kwargs)

    monkeypatch.setattr(tweedie_module, "tweedie_log_series", counted_series)

    fitted, null = tweedie_module._tweedie_logpdf_pair(
        y,
        mu,
        null_mu,
        0.8,
        1.5,
        weights=weights,
    )

    assert fitted.shape == null.shape == (n,)
    assert len(batch_sizes) <= 1
    assert all(size > 1 for size in batch_sizes)


def test_fit_stat_pair_passes_strict_general_power_series_budget(monkeypatch) -> None:
    p = 1.4
    y = np.ones(16)
    mu = np.linspace(0.9, 1.1, len(y))
    null_mu = np.full_like(y, 1.0)
    phi = 1.0 / (0.6 * 10_000.0)
    real_series = tweedie_module.tweedie_log_series
    budgets: list[int | None] = []

    def force_wright_failure(a, b, z):
        del a, b
        return np.full_like(z, np.nan, dtype=np.float64)

    def counted_series(log_t, a, **kwargs):
        budgets.append(kwargs.get("max_total_terms"))
        return real_series(log_t, a, **kwargs)

    monkeypatch.setattr(tweedie_module, "wright_bessel", force_wright_failure)
    monkeypatch.setattr(tweedie_module, "tweedie_log_series", counted_series)

    fitted, null = tweedie_module._tweedie_logpdf_pair(
        y,
        mu,
        null_mu,
        phi,
        p,
    )

    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(null))
    assert budgets and all(budget is not None for budget in budgets)
    assert max(budgets) <= 4_096


class TestREMLProfileSearchOverhead:
    """The REML power search must not re-pay published-model bookkeeping per step."""

    @staticmethod
    def _model_and_data():
        rng = np.random.default_rng(11)
        n = 4000
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
