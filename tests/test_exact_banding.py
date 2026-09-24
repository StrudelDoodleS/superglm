"""Tests for exact contiguous banding of a fitted curve."""

import itertools

import numpy as np
import pytest

from superglm.diagnostics.exact_banding import (
    MAX_EXACT_VALUES,
    exact_bands,
)

_EPS = float(np.finfo(np.float64).eps)


def _brute_force(s, w, tol):
    """(band count, weighted squared error) minimised lexicographically over every banding."""
    n = len(s)
    best = None
    for r in range(n):
        for cuts in itertools.combinations(range(1, n), r):
            starts = (0, *cuts)
            ends = (*cuts, n)
            total = 0.0
            feasible = True
            for a, b in zip(starts, ends, strict=True):
                mean = np.average(s[a:b], weights=w[a:b])
                if b - a > 1 and np.any(np.abs(s[a:b] - mean) > tol[a:b]):
                    feasible = False
                    break
                total += float(np.sum(w[a:b] * (s[a:b] - mean) ** 2))
            if feasible and (best is None or (len(starts), total) < best):
                best = (len(starts), total)
    return best


def test_matches_exhaustive_search_on_random_curves():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 11))
        s = np.cumsum(rng.normal(0.0, 0.3, n))
        w = rng.uniform(0.1, 3.0, n)
        tol = rng.uniform(0.0, 0.4, n)
        tol[rng.random(n) < 0.15] = 0.0
        result = exact_bands(s, w, tol, max_bands=n)
        count, sse = _brute_force(s, w, tol)
        # Each band cost is sum(w s^2) - W m^2; its rounding is a small multiple
        # of eps * sum(w s^2), and n <= 10 values bound the multiple by 256.
        scale = 256 * _EPS * (1.0 + float(np.sum(w * s * s)))
        assert len(result.starts) == count
        assert abs(result.sse - sse) <= scale


def test_two_obvious_bands():
    s = np.array([0.0, 0.05, 1.0, 1.05])
    result = exact_bands(s, np.ones(4), np.full(4, 0.1), max_bands=4)
    assert result.starts.tolist() == [0, 2]
    np.testing.assert_allclose(result.factors, [0.025, 1.025], rtol=0, atol=4 * _EPS)
    assert result.tolerance_factor == 1.0


def test_every_value_is_within_its_tolerance_of_its_band():
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.normal(0.0, 0.05, 60))
    w = rng.uniform(0.1, 3.0, 60)
    tol = rng.uniform(0.01, 0.1, 60)
    result = exact_bands(s, w, tol, max_bands=60)
    ends = np.append(result.starts[1:], 60)
    for a, b, factor in zip(result.starts, ends, result.factors, strict=True):
        slack = 4 * _EPS * (b - a) * (1.0 + np.abs(s[a:b]).max())
        assert np.all(np.abs(s[a:b] - factor) <= tol[a:b] + slack)


def test_constant_curve_with_zero_tolerance_is_one_band():
    s = np.full(25, 0.3)
    result = exact_bands(s, np.linspace(0.5, 2.0, 25), np.zeros(25), max_bands=25)
    assert result.starts.tolist() == [0]


def test_zero_tolerance_values_can_always_stand_alone():
    s = np.array([0.0, 1.0, 2.0])
    result = exact_bands(s, np.ones(3), np.zeros(3), max_bands=3)
    assert result.starts.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("s", "w", "tol", "max_bands", "match"),
    [
        ([0.0, 1.0], [1.0], [0.1, 0.1], 2, "same length"),
        ([], [], [], 1, "at least one value"),
        ([0.0, np.nan], [1.0, 1.0], [0.1, 0.1], 2, "finite"),
        ([0.0, 1.0], [1.0, 0.0], [0.1, 0.1], 2, "positive"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, -0.1], 2, "nonnegative"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, 0.1], 0, "max_bands"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, 0.1], True, "max_bands"),
    ],
)
def test_rejects_bad_inputs(s, w, tol, max_bands, match):
    with pytest.raises(ValueError, match=match):
        exact_bands(np.asarray(s), np.asarray(w), np.asarray(tol), max_bands=max_bands)


def test_rejects_too_many_values():
    n = MAX_EXACT_VALUES + 1
    with pytest.raises(ValueError, match="at most 5000"):
        exact_bands(np.zeros(n), np.ones(n), np.zeros(n), max_bands=10)
