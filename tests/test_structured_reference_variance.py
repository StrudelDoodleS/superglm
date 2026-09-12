"""Structured reference variance, numerical refusal and work accounting."""

from __future__ import annotations

import numpy as np
import pytest

import superglm.screening._structured as st


def _pair(penalty_exponent=0, *, unpenalized=False):
    x = np.linspace(-1.0, 1.0, 9)
    weights = np.ones((x.size, 4))
    penalty = np.zeros((2, 2)) if unpenalized else np.ldexp(np.eye(2), penalty_exponent)
    return st.spline_cat_moments(
        np.column_stack((x, x**2)),
        penalty,
        np.zeros_like(weights),
        weights,
        np.arange(1, 4),
    )


def test_final_variance_is_computed_once_per_distinct_emitted_lambda(monkeypatch):
    """Charging variance at every search step, or every duplicate, is waste."""
    real = st._filter_factor_sum
    variance_lambdas = []
    geometries = []

    def counted(pair, geometry, lam, **kwargs):
        geometries.append(geometry)
        if kwargs.get("variance") is not None:
            variance_lambdas.append(lam)
        return real(pair, geometry, lam, **kwargs)

    monkeypatch.setattr(st, "_filter_factor_sum", counted)
    results = st.structured_ladder(_pair(), budgets=(2.0, 2.0, 100.0, 100.0))
    assert results is not None and len(results) == 4
    assert sorted(variance_lambdas) == sorted({row.lambda0 for row in results})
    assert all(geometry is geometries[0] for geometry in geometries)


def test_clamped_variance_factorization_is_in_the_work_budget(monkeypatch):
    """The two bracket evaluations do not include the final variance pass."""
    real = st._filter_factor_sum
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("variance") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(st, "_filter_factor_sum", counted)
    assert st.structured_ladder(_pair(), budgets=(100.0, 100.0), max_evaluations=2) is None
    assert calls == [False, False]
    calls.clear()
    results = st.structured_ladder(_pair(), budgets=(100.0, 100.0), max_evaluations=3)
    assert results is not None and len(results) == 2
    assert calls == [False, False, True]


def test_unpenalized_variance_factorization_is_in_the_work_budget(monkeypatch):
    real = st._filter_factor_sum
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("variance") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(st, "_filter_factor_sum", counted)
    assert st.structured_ladder(_pair(unpenalized=True), max_evaluations=1) is None
    assert calls == []
    results = st.structured_ladder(_pair(unpenalized=True), max_evaluations=2)
    assert results is not None
    assert calls == [False, True]
    for row in results:
        bound = 128 * np.finfo(float).eps * 6
        assert row.reference_variance == pytest.approx(2 * row.edf0, abs=bound, rel=0)


@pytest.mark.parametrize("broken", [np.nan, -1.0, 0.0, 1e6])
def test_invalid_variance_refuses_the_rung(monkeypatch, broken):
    """Removing the variance consistency guard would publish these results."""
    real = st._filter_factor_sum

    def corrupted(*args, **kwargs):
        result = real(*args, **kwargs)
        variance = kwargs.get("variance")
        if variance is not None:
            variance.squared_norm = broken
        return result

    monkeypatch.setattr(st, "_filter_factor_sum", corrupted)
    assert st.structured_ladder(_pair(), budgets=(2.0,)) is None


def test_variance_refusal_preserves_an_independent_rung(monkeypatch):
    real = st._filter_factor_sum

    def corrupted(pair, geometry, lam, **kwargs):
        result = real(pair, geometry, lam, **kwargs)
        variance = kwargs.get("variance")
        if variance is not None and lam < 1.0:
            variance.squared_norm = np.nan
        return result

    monkeypatch.setattr(st, "_filter_factor_sum", corrupted)
    results = st.structured_ladder(_pair(), budgets=(100.0, 0.1))
    assert results is not None and len(results) == 1
    assert results[0].lambda0 > 1.0


@pytest.mark.parametrize("exponent", [-600, 600])
def test_structured_midpoint_preserves_extreme_penalty_units(exponent):
    baseline = st.structured_ladder(_pair(), budgets=(2.0,))[0]
    results = st.structured_ladder(_pair(exponent), budgets=(2.0,))
    assert results is not None
    row = results[0]
    assert abs(row.edf0 - 2.0) <= st._EDF_TOL
    # |d(2 sum a^2)/d(sum a)| <= 4 along the monotone ladder.
    bound = 4 * abs(row.edf0 - baseline.edf0) + 256 * np.finfo(float).eps
    assert row.reference_variance == pytest.approx(baseline.reference_variance, abs=bound)
