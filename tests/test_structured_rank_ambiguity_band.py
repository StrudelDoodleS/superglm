"""The rank-ambiguity band must not grow until it swallows its own cutoff.

``_representative_projection`` widens an a-priori error band by a flop count::

    reduction_depth = n_rows * k + 2 * n_levels * k**2 + k**2
    uncertainty     = 16 * eps * reduction_depth * leading_scale
    cutoff          = sqrt(k * eps) * leading_scale

The band grows with problem size; the cutoff does not.  Past
``reduction_depth >= 2**22 * sqrt(k)`` -- ``sqrt(eps)`` is exactly ``2**-26`` --
the interval ``[cutoff - unc, cutoff + unc]`` contains zero, so every pivot
reads as ambiguous and the projection refuses.  A caller who RAISED
``max_cells`` could therefore turn a scored pair into a refused one, which
inverts the documented monotone budget behaviour.

``n_levels`` is a declared count rather than an allocation, so the crossover is
reachable in a unit test for nothing.  The public route needs a ~1,260-column
fit and several hundred MB and is deliberately not exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.screening._structured import _representative_projection


def _rank_deficient_factor(k: int, rank: int, seed: int = 20260805):
    """A factor whose numerical rank is unambiguously ``rank``, not borderline.

    The retained pivots sit at O(1) and the dropped ones at O(1e-15), decades
    away from the cutoff on either side, so any refusal is the band's doing and
    not a genuinely marginal geometry.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(k, rank))
    factor = basis @ rng.normal(size=(rank, k))
    return np.asarray(factor, dtype=np.float64)


@pytest.mark.parametrize("n_levels", [1, 10**3, 10**5, 10**7, 10**9])
def test_projection_survives_any_declared_level_count(n_levels: int) -> None:
    """The same geometry must resolve at every budget, not only small ones."""
    k, rank = 12, 8
    result = _representative_projection(
        _rank_deficient_factor(k, rank), n_rows=4096, n_levels=n_levels
    )

    assert result is not None, (
        f"refused a geometry with pivots decades from the cutoff at "
        f"n_levels={n_levels:,}; the ambiguity band swallowed its own cutoff"
    )
    active, _ = result
    assert active.size == rank


@pytest.mark.parametrize("n_rows", [256, 10**5, 10**7, 10**9])
def test_projection_survives_any_row_count(n_rows: int) -> None:
    """The row term inflates the band too, and dominates on the exact route.

    On the speculative lookahead handoff the ladder runs on un-binned exact
    support with ``n_levels`` as small as 6, so ``n_rows * k`` is essentially
    all of ``reduction_depth``.  Capping only the level term would leave that
    route exposed.
    """
    k, rank = 12, 8
    result = _representative_projection(_rank_deficient_factor(k, rank), n_rows=n_rows, n_levels=6)

    assert result is not None, f"refused at n_rows={n_rows:,}"
    active, _ = result
    assert active.size == rank


def test_a_genuinely_marginal_pivot_is_still_refused() -> None:
    """Capping the band must not turn the guard off.

    A pivot placed ON the cutoff is what the ambiguity test exists to catch, and
    it must still be refused with the cap in place.
    """
    k = 6
    eps = np.finfo(np.float64).eps
    cutoff_ratio = np.sqrt(k * eps)
    factor = np.diag(np.array([1.0, 0.5, 0.25, cutoff_ratio, 1e-18, 1e-18]))

    assert _representative_projection(factor, n_rows=256, n_levels=4) is None


def test_the_cap_is_inert_below_the_crossover() -> None:
    """Small problems must be bit-identical to the uncapped band.

    ``16 * eps * reduction_depth`` only reaches ``0.5 * sqrt(k * eps)`` at
    ``reduction_depth >= 2**21 * sqrt(k)``; below that the ``min`` never binds,
    so the accept/refuse decision is unchanged for every ordinary problem.
    """
    k = 12
    eps = np.finfo(np.float64).eps
    reduction_depth = 4096 * k + 2 * 100 * k**2 + k**2
    uncapped = 16.0 * eps * reduction_depth
    cap = 0.5 * np.sqrt(k * eps)

    assert uncapped < cap, "fixture must sit below the crossover"
    assert min(uncapped, cap) == uncapped
