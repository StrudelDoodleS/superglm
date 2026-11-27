"""Simultaneous bands: the equal-local-levels Q-Q band and the Bonferroni binomial band."""

import contextlib
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
    bands._stored_band.cache_clear()
    monkeypatch.setattr(bands, "_local_level", lambda n, alpha: alpha)
    try:
        band = bands.equal_local_levels_band(2000, alpha=0.05)
        assert _escape_rate(band.lower, band.upper, 2000, 2000, seed=3) > 0.5
    finally:
        bands._stored_band.cache_clear()


def _steck_inside(lower, upper):
    """``P(lower_i < U_(i) < upper_i for every i)`` for ``n`` iid uniforms, in mpmath.

    Steck (1971), *Ann. Math. Statist.* 42(1), 1-11, as Theorem I of Pitman (1972), *Bull.
    Austral. Math. Soc.* 7, 227-232, states it: ``n! det[(v_i - u_j)_+^(j-i+1) / (j-i+1)!]``,
    entries with ``i > j + 1`` zero. That matrix is upper Hessenberg with ones below the
    diagonal, so its leading minors satisfy ``D_k = sum_i (-1)^(k-i) m_ik D_(i-1)``. The
    alternating sum cancels, so it runs at 60 digits, which agree with 120 to 1e-34 at n = 249.
    """
    mpmath = pytest.importorskip("mpmath")
    with mpmath.workdps(60):
        u = [mpmath.mpf(float(x)) for x in lower]
        v = [mpmath.mpf(float(x)) for x in upper]
        n = len(u)
        factorial = [mpmath.factorial(k) for k in range(n + 1)]
        minors = [mpmath.mpf(1)]
        for k in range(n):
            total = mpmath.mpf(0)
            for i in range(k + 1):
                gap = v[i] - u[k]
                if gap > 0:
                    term = gap ** (k - i + 1) / factorial[k - i + 1] * minors[i]
                    total += term if (k - i) % 2 == 0 else -term
            minors.append(total)
        return float(factorial[n] * minors[n])


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
@pytest.mark.parametrize("n", [2, 5, 20, 99, 249])
def test_the_searched_local_level_gives_exactly_the_global_level(n, alpha):
    """Below the asymptotic cutover the band's global level is alpha, by an independent formula.

    The search stops with ``eta`` known to a relative 1e-9 and the recursion behind it rounds
    to about 1e-13 per bound. Measured, the level is within 3e-10 of alpha relative at every
    ``n`` from 2 to 249; the test allows the module's promised 1e-7. Mutation checks: a search
    stopped at a relative 1e-3, or a recursion that skips the last bound, fails it, as does
    the Monte Carlo calibration this search replaced (4.65% at n = 2).
    """
    band = equal_local_levels_band(n, alpha=alpha)
    level = 1.0 - _steck_inside(band.lower, band.upper)
    assert abs(level - alpha) <= 1e-7 * alpha


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
    np.testing.assert_allclose(band.lower, 1 - band.upper[::-1], rtol=0, atol=8 * 2.0**-53)
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


@pytest.mark.parametrize(
    ("trials", "p", "alpha"),
    [
        (1, 0.025000000000000005, 0.05),  # P(X > 0) lies one rounding above the cut
        (100_000, 0.5, 1e-20),  # an inverse that loses a cut this far into the tail
        (2000, 0.5, 5e-324),  # alpha / 2 underflows to zero
    ],
)
def test_each_binomial_bound_is_decided_by_its_own_tail(trials, p, alpha):
    """A bound meets ``P(X < lower) < cut <= P(X <= lower)`` and ``P(X > upper) <= cut <
    P(X > upper - 1)``, read from ``binom.cdf`` and ``binom.sf`` and compared exactly with
    ``cut = alpha / 2`` for one cell, which in the last case lies below every positive float.

    Mutation check: bounds from ``binom.ppf(cut)`` and ``binom.isf(cut)`` fail all three cases.
    """
    (lower,), (upper,) = bonferroni_binomial_band([trials], [p], alpha=alpha)
    cut = Fraction(alpha) / 2
    lo, hi = int(lower), int(upper)
    cdf = lambda k: Fraction(float(stats.binom.cdf(k, trials, p)))  # noqa: E731
    sf = lambda k: Fraction(float(stats.binom.sf(k, trials, p)))  # noqa: E731
    assert 0 <= lo <= hi <= trials
    assert cdf(lo - 1) < cut <= cdf(lo)
    assert sf(hi) <= cut < sf(hi - 1)


def test_an_input_the_band_refuses_is_refused_after_a_valid_call_too():
    """Mutation check: validating inside the cached function lets ``2.0`` reach the cached ``2``."""
    equal_local_levels_band(2, alpha=0.05)
    with pytest.raises(TypeError):
        equal_local_levels_band(2.0, alpha=0.05)
    with pytest.raises(ValueError, match="0.01, 0.05 or 0.1"):
        equal_local_levels_band(2, alpha=0.2)


def test_equal_inputs_share_one_cached_band(monkeypatch):
    """``61``, ``np.int64(61)`` and the default ``alpha`` given or not compute one band.

    Mutation check: caching on the arguments as given, as the first version did, computes it
    three times. No other test asks for 61 points, so its entry starts cold.
    """
    calls = []
    level = bands._local_level
    monkeypatch.setattr(bands, "_local_level", lambda n, alpha: calls.append(n) or level(n, alpha))
    forms = [(61, {}), (np.int64(61), {}), (61, {"alpha": 0.05}), (61, {"alpha": np.float64(0.05)})]
    first, *others = [equal_local_levels_band(n, **keywords) for n, keywords in forms]
    assert calls == [61]
    for other in others:
        np.testing.assert_array_equal(other.lower, first.lower)


def test_changing_a_returned_band_leaves_later_calls_intact():
    """Mutation check: handing every caller the cached arrays lets the write below through."""
    first = equal_local_levels_band(64)
    pristine = first.lower.copy()
    with contextlib.suppress(ValueError, AttributeError):
        first.lower.setflags(write=True)
    with contextlib.suppress(ValueError, AttributeError):
        first.lower.base.setflags(write=True)
    with contextlib.suppress(ValueError):
        first.lower[0] = 0.99
    later = equal_local_levels_band(64)
    np.testing.assert_array_equal(later.lower, pristine)
    assert later.lower is not first.lower
    with pytest.raises(ValueError):
        later.upper.setflags(write=True)
