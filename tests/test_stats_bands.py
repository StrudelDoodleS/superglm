"""Simultaneous bands: the equal-local-levels Q-Q band and the Bonferroni binomial band."""

import contextlib
from fractions import Fraction
from math import comb

import numpy as np
import pytest
from scipy import stats

import superglm.stats.bands as bands
from superglm.stats.bands import (
    band_at_level,
    binomial_cell_level,
    bonferroni_binomial_band,
    equal_local_levels_band,
    monte_carlo_level,
    order_statistic_level,
)


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


def test_each_binomial_bound_is_decided_by_its_own_tail():
    """A bound meets ``P(X < lower) < cut <= P(X <= lower)`` and ``P(X > upper) <= cut <
    P(X > upper - 1)``, read from ``binom.cdf`` and ``binom.sf`` and compared exactly with
    ``cut = alpha / 2`` for one cell. Here ``P(X > 0) = p`` lies one rounding above the cut.

    Mutation check: bounds from ``binom.ppf(cut)`` and ``binom.isf(cut)`` fail it.
    """
    trials, p, alpha = 1, 0.025000000000000005, 0.05
    (lower,), (upper,) = bonferroni_binomial_band([trials], [p], alpha=alpha)
    cut = Fraction(alpha) / 2
    lo, hi = int(lower), int(upper)
    cdf = lambda k: Fraction(float(stats.binom.cdf(k, trials, p)))  # noqa: E731
    sf = lambda k: Fraction(float(stats.binom.sf(k, trials, p)))  # noqa: E731
    assert 0 <= lo <= hi <= trials
    assert cdf(lo - 1) < cut <= cdf(lo)
    assert sf(hi) <= cut < sf(hi - 1)


@pytest.mark.parametrize("alpha", [1e-13, 1e-20, 5e-324, 0.0, 1.0])
def test_the_binomial_band_refuses_a_level_outside_its_range(alpha):
    """Mutation check: without the lower limit, 1e-13, 1e-20 and 5e-324 are accepted."""
    with pytest.raises(ValueError, match="at least 1e-12 and below 1"):
        bonferroni_binomial_band([40], [0.5], alpha=alpha)


def test_the_bisection_ends_within_one_step_per_bit_of_the_trials():
    """At ``low = 2**53 - 5``, ``high = 2**53 - 4`` a float midpoint rounds to ``high``.

    Mutation check: the float midpoint ``floor((low + high) / 2)`` never moves there; the
    counting predicate stops it at 64 steps instead of letting it loop.
    """
    trials = np.array([2.0**53 - 4])
    steps = []

    def holds(k):
        steps.append(k)
        if len(steps) > 64:
            raise RuntimeError("the bisection did not end in 64 steps")
        return k >= trials

    assert bands._least_count(trials, holds)[0] == trials[0]
    assert len(steps) <= 54
    lower, upper = bonferroni_binomial_band(2**53 - 4, 1.0)
    assert (float(lower), float(upper)) == (2.0**53 - 4, 2.0**53 - 4)


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


def test_a_band_at_a_given_level_is_the_beta_quantiles_at_that_level():
    n, eta = 331, 0.003
    band = band_at_level(n, eta, alpha=0.05)
    i = np.arange(1, n + 1)
    np.testing.assert_allclose(band.lower, stats.beta.ppf(eta / 2, i, n + 1 - i), rtol=1e-12)
    np.testing.assert_allclose(band.upper, stats.beta.ppf(1 - eta / 2, i, n + 1 - i), rtol=1e-11)
    np.testing.assert_allclose(band.median, stats.beta.ppf(0.5, i, n + 1 - i), rtol=1e-12)
    assert (band.local_level, band.alpha) == (eta, 0.05)


@pytest.mark.parametrize("n", [1, 5, 331])
def test_a_band_at_the_known_level_is_the_equal_local_levels_band(n):
    known = equal_local_levels_band(n)
    band = band_at_level(n, known.local_level)
    for name in ("lower", "upper", "median"):
        np.testing.assert_array_equal(getattr(band, name), getattr(known, name))
    assert band.local_level == known.local_level and band.alpha == known.alpha


def _exact_order_statistic_level(u):
    """Every point's two-sided Beta tail, read from SciPy's Beta law."""
    n = len(u)
    i = np.arange(1, n + 1)
    tails = 2 * np.minimum(stats.beta.cdf(u, i, n + 1 - i), stats.beta.sf(u, i, n + 1 - i))
    return float(tails.min()), int(tails.argmin())


@pytest.mark.parametrize("n", [1, 7, 250, 3000])
def test_the_order_statistic_level_is_the_least_tail_over_every_point(n):
    for seed in range(3):
        u = np.sort(np.random.default_rng(100 * n + seed).uniform(size=n))
        assert order_statistic_level(u) == pytest.approx(
            _exact_order_statistic_level(u)[0], rel=1e-12
        )


def _skewed_bump(n=2000, centre=1350, width=250, height=3.05):
    """Sorted values that sit ``height`` standard deviations above each mean on a flat-topped bump.

    Above the middle each order statistic's law is skewed left, so its upper tail is shorter
    than a normal one: the least tail sits on the bump's far side, where ``|z|`` is a little
    below the bump's top, and away from both ends.
    """
    i = np.arange(1, n + 1.0)
    mean = i / (n + 1)
    sd = np.sqrt(i * (n + 1 - i) / ((n + 1) ** 2 * (n + 2)))
    z = height * np.exp(-0.5 * ((i - centre) / width) ** 4)
    return mean + z * sd, np.abs(z)


def test_the_order_statistic_level_finds_a_least_tail_that_no_normal_screen_ranks_first():
    """Mutation check: a screen of the 64 lowest, the 64 highest and the 64 largest ``|z|``
    in between misses this sample's least tail, and fails here.
    """
    u, absz = _skewed_bump()
    assert np.all(np.diff(u) > 0)
    level, where = _exact_order_statistic_level(u)
    middle = np.arange(64, len(u) - 64)
    largest = middle[np.argsort(-absz[middle])[:64]]
    assert 64 <= where < len(u) - 64 and where not in largest
    assert order_statistic_level(u) == pytest.approx(level, rel=1e-12)


def test_the_batched_levels_are_exact_up_to_the_rank_they_resolve():
    """The ``resolve`` smallest levels are the direct ones to the bit; any other is at most its
    own level and above every exact one, so the ``resolve``-th smallest is exact."""
    rng = np.random.default_rng(5)
    u = np.sort(rng.uniform(size=(40, 600)) ** rng.uniform(0.9, 1.1, size=(40, 1)), axis=1)
    exact = np.array([_exact_order_statistic_level(row)[0] for row in u])
    levels = bands._order_statistic_levels(u, resolve=5)
    order = np.argsort(exact)
    np.testing.assert_array_equal(levels[order[:5]], exact[order[:5]])
    assert np.all(levels <= exact) and np.all(levels[order[5:]] > exact[order[4]])


def _direct_levels(rows):
    """Each row's least two-sided tail, every point read from SciPy's Beta law."""
    return np.array([_exact_order_statistic_level(row)[0] for row in rows])


@pytest.mark.parametrize(("n", "ratio"), [(12, 10.0), (20, 20.0)])
def test_a_least_tail_beside_a_band_edge_float64_cannot_place_is_read(n, ratio):
    """Next to 1 the band's edge rounds onto a sample value whose tail is far below the target.

    Row A is ``n`` copies of the float below 1, row B ``n`` copies of a small value whose least
    tail is ``ratio`` times A's, so B's level is the target and A's least tail lies on the
    band's rounded upper edge. Both levels are the direct ones, so A ranks first, and A's
    single-row level is its direct one, not the float above it. At ``n = 20`` both levels are
    subnormal, at ``n = 12`` normal. Mutation check: trusting every band edge, as SciPy rounds
    it, ranks B first and returns the float above A's level.
    """
    top = np.full(n, np.nextafter(1.0, 0.0))
    small = np.full(n, (ratio * _exact_order_statistic_level(top)[0] / 2.0) ** (1.0 / n))
    rows = np.stack([top, small])
    direct = _direct_levels(rows)
    assert direct[1] / direct[0] == pytest.approx(ratio, rel=1e-3)
    levels = bands._order_statistic_levels(rows, resolve=2)
    np.testing.assert_array_equal(levels, direct)
    assert list(np.argsort(levels)) == [0, 1]
    assert order_statistic_level(top) == direct[0]


def test_a_screening_target_of_zero_reads_every_row():
    """Two rows at level zero make the target zero, where the band holds every value.

    Every row's level is then its direct one, the positive ones included. Mutation check:
    filling the rows the band does not resolve with the target, zero, fails it.
    """
    rng = np.random.default_rng(11)
    positive = np.sort(rng.uniform(size=(3, 40)), axis=1)
    rows = np.concatenate([np.zeros((2, 40)), positive])
    direct = _direct_levels(rows)
    assert np.all(direct[:2] == 0.0) and np.all(direct[2:] > 0.0)
    np.testing.assert_array_equal(bands._order_statistic_levels(rows, resolve=2), direct)


def test_an_unsorted_sample_has_no_order_statistic_level():
    with pytest.raises(ValueError, match="ascending"):
        order_statistic_level([0.4, 0.2])


def test_the_monte_carlo_level_is_barnards_rank():
    """Barnard (1963): at 99 replicates and 5% the 5th smallest, or the 5th largest."""
    values = np.random.default_rng(3).permutation(99).astype(np.float64)
    assert monte_carlo_level(values, alpha=0.05) == 4.0
    assert monte_carlo_level(values, alpha=0.05, larger_is_extreme=True) == 94.0
    assert monte_carlo_level(values[:19], alpha=0.05) == np.min(values[:19])


def test_too_few_replicates_have_no_monte_carlo_level():
    with pytest.raises(ValueError, match="at least 19"):
        monte_carlo_level(np.arange(18.0), alpha=0.05)


def test_the_monte_carlo_level_reads_both_ends_of_its_alpha_range():
    """At 0.5 the rank is the 50th of 99; at 1e-12 it needs about 10**12 replicates."""
    values = np.random.default_rng(3).permutation(99).astype(np.float64)
    assert monte_carlo_level(values, alpha=0.5) == 49.0
    assert monte_carlo_level(values, alpha=0.5, larger_is_extreme=True) == 49.0
    with pytest.raises(ValueError, match="needs at least 999999999999 replicates; got 99"):
        monte_carlo_level(values, alpha=1e-12)


@pytest.mark.parametrize(
    "alpha",
    [np.nextafter(1e-12, 0.0), np.nextafter(0.5, 1.0), 5e-324, np.nextafter(1.0, 0.0), np.nan],
)
def test_an_alpha_outside_its_range_has_no_monte_carlo_level(alpha):
    """Mutation check: without the range, the float below 1 reads past the last replicate and
    the least subnormal overflows the replicate count it reports."""
    with pytest.raises(ValueError, match=r"alpha must lie in \[1e-12, 0\.5\]"):
        monte_carlo_level(np.arange(99.0), alpha=alpha)


@pytest.mark.parametrize("trials", [1, 2, 5, 12])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.5])
def test_a_count_is_inside_the_band_at_a_cut_exactly_when_its_level_reaches_the_cut(trials, p):
    """At ``cut=c`` the band holds every count whose two tails are each at least ``c``.

    The cuts include every count's own level, where the two sides of the comparison are equal.
    Mutation check: an upper bound decided by ``P(X > upper) <= c``, as the Bonferroni band's
    is, puts a count whose upper tail equals the cut outside and fails it.
    """
    counts = np.arange(trials + 1)
    levels = [binomial_cell_level([k], [trials], [p]) for k in counts]
    for cut in [*levels, 1e-6, 0.01, 0.2]:
        (lower,), (upper,) = bonferroni_binomial_band([trials], [p], cut=cut)
        inside = (counts >= lower) & (counts <= upper)
        np.testing.assert_array_equal(inside, np.asarray(levels) >= cut)


def test_the_cell_level_is_the_least_one_sided_tail_over_cells():
    counts, trials, p = [3, 0, 7], [10, 10, 10], [0.2, 0.2, 0.5]
    tails = [
        min(stats.binom.cdf(k, n, q), stats.binom.sf(k - 1, n, q))
        for k, n, q in zip(counts, trials, p, strict=True)
    ]
    assert binomial_cell_level(counts, trials, p) == min(tails)
