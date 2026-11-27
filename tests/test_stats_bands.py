"""Simultaneous bands: the equal-local-levels Q-Q band and the Bonferroni binomial band."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from scipy import stats

import superglm.stats.bands as bands
from superglm.stats.bands import bonferroni_binomial_band, equal_local_levels_band


def _escape_rate(lower, upper, n, sims, seed):
    """Share of `sims` uniform samples whose order statistics leave the band anywhere."""
    rng = np.random.default_rng(seed)
    u = np.sort(rng.uniform(size=(sims, n)), axis=1)
    return float(np.mean(np.any((u < lower) | (u > upper), axis=1)))


@pytest.mark.parametrize("n", [20, 50, 331, 2000])
def test_the_band_holds_for_every_point_at_once_at_its_level(n):
    band = equal_local_levels_band(n, alpha=0.05)
    assert 0.035 <= _escape_rate(band.lower, band.upper, n, 4000, seed=n + 7) <= 0.065


def test_a_pointwise_local_level_breaks_the_global_level(monkeypatch):
    """Mutation check: the local level, not the Beta quantiles, sets the global level."""
    bands.equal_local_levels_band.cache_clear()
    monkeypatch.setattr(bands, "_local_level", lambda n, alpha: alpha)
    band = bands.equal_local_levels_band(2000, alpha=0.05)
    assert _escape_rate(band.lower, band.upper, 2000, 2000, seed=3) > 0.5
    bands.equal_local_levels_band.cache_clear()


def test_each_bound_is_its_order_statistics_beta_quantile():
    n = 331
    band = equal_local_levels_band(n)
    i = np.arange(1, n + 1)
    np.testing.assert_allclose(
        band.lower, stats.beta.ppf(band.local_level / 2, i, n + 1 - i), rtol=1e-12
    )
    np.testing.assert_allclose(
        band.upper, stats.beta.isf(band.local_level / 2, i, n + 1 - i), rtol=1e-12
    )
    np.testing.assert_allclose(band.lower, 1 - band.upper[::-1], atol=8 * 2.0**-53)
    assert np.all((band.lower < band.median) & (band.median < band.upper))


def test_one_point_is_tested_at_the_global_level():
    band = equal_local_levels_band(1, alpha=0.05)
    assert band.local_level == 0.05
    np.testing.assert_allclose([band.lower[0], band.upper[0]], [0.025, 0.975], rtol=1e-14)


def test_the_band_is_the_same_on_every_call():
    a, b = equal_local_levels_band(57), equal_local_levels_band(57)
    np.testing.assert_array_equal(a.lower, b.lower)
    assert not a.lower.flags.writeable


@pytest.mark.parametrize("bad", [0, -3])
def test_a_band_needs_a_point(bad):
    with pytest.raises(ValueError, match="at least one"):
        equal_local_levels_band(bad)


def test_alpha_is_one_the_published_levels_cover():
    with pytest.raises(ValueError, match="0.01, 0.05 or 0.1"):
        equal_local_levels_band(500, alpha=0.2)


def test_the_binomial_band_cuts_each_tail_at_alpha_over_2m():
    lower, upper = bonferroni_binomial_band([40, 40], [0.05, 0.5], alpha=0.05)
    for trials, p, lo, hi in zip([40, 40], [Fraction(1, 20), Fraction(1, 2)], lower, upper):
        cdf = lambda k: sum(comb(trials, j) * p**j * (1 - p) ** (trials - j) for j in range(k + 1))  # noqa: E731
        cut = Fraction(1, 80)  # alpha / (2 m), m = 2 cells
        assert cdf(int(lo) - 1) < cut <= cdf(int(lo))  # P(X < lo) < cut
        assert 1 - cdf(int(hi)) <= cut < 1 - cdf(int(hi) - 1)  # P(X > hi) <= cut
