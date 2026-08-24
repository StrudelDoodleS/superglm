"""The unpenalized rung's rank, decided where nothing has been residualized away.

``edf`` at ``lambda = 0`` is ``rank(V_eff)``, and the one thing that must not
decide it is the profiled block's own largest direction: on a block the overlap
has absorbed, that direction IS round-off, so a relative cut there is taken
against the noise it has to reject.

**THESE TESTS USED TO ASSERT ``_profiled_rank`` AND NOW ASSERT THE LADDER.**
That function took ``rank([[V, C'], [C, M]]) - rank(M)`` -- Guttman additivity
on the JOINT moment matrix, where neither operand cancels -- because ``V_eff``
arrived as a difference and no cut on it was safe.  Issue #257 removed the
difference: the ladder is handed a factor of the pair's design and reads
``V_eff`` off a trailing block, so there is one operand rather than two and no
subtraction anywhere.  What survives unchanged is the ARGUMENT -- the reference
has to be the joint design's scale, not the residual's -- and it is enforced
here on the quantity a caller sees.  The evidence that the port is complete is
that the reachable path ``_profiled_rank`` was measured on comes back the same:
``test_a_wholly_absorbed_probe_is_rejected_on_every_seed`` is 0 of 20 seeds
before and after, and it read 20 of 20 while the cut was still relative to the
profiled block's own top.

The fixtures below are stated as MOMENTS, which is what they are about; the
adapter that turns a moment specification into the factor with those moments
lives in ``tests/test_interaction_screening.py`` and is documented there.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.screening import penalized_score_statistic

from .test_interaction_screening import _factor_from_moments


def _profiled_edf(V, C, M) -> float:
    """The unpenalized rung's achieved ``edf0`` for a moment-stated pair."""
    k = V.shape[0]
    q = M.shape[0]
    return penalized_score_statistic(
        _factor_from_moments(np.zeros(k), V, C, M, np.zeros(q)), None
    ).edf0


def _joint(nuisance_eigenvalue: float, k: int = 100) -> tuple:
    """Two independent blocks: ``C = 0``, so the true profiled rank is ``k``."""
    V = np.eye(k)
    C = np.zeros((2, k))
    M = np.diag([1.0, nuisance_eigenvalue])
    return V, C, M


def test_a_nuisance_direction_at_the_floor_does_not_cost_the_probe_a_rank() -> None:
    """The reviewer's construction: independent blocks, true profiled rank 100.

    The Guttman form this replaces had to count two operands against ONE ruler
    or a nuisance direction would fall below the cut in the joint while
    surviving in ``M``, and the difference would undercount by a whole degree
    of freedom.  There is no difference to undercount now -- the overlap's own
    near-null direction never enters the profiled block at all, because ``C``
    is zero and the reduction puts nothing of the probe in its span.
    """
    V, C, M = _joint(1e-14)
    assert _profiled_edf(V, C, M) == 100.0


@pytest.mark.parametrize("nuisance", [1e-13, 1e-14, 1e-15, 2e-14, 5e-15])
def test_the_rank_holds_across_the_window_the_two_floors_used_to_open(
    nuisance: float,
) -> None:
    """A nuisance eigenvalue anywhere in the disputed window still gives 100.

    ``q * eps`` is 4.44e-16 and ``(k + q) * eps`` is 2.26e-14, so these values
    straddle the gap the mismatched rulers opened.  Every one of them is a
    direction that either belongs to both counts or to neither -- and now, to
    neither operand, because there is one.
    """
    V, C, M = _joint(nuisance)
    assert _profiled_edf(V, C, M) == 100.0


def test_a_shared_null_direction_is_still_dropped() -> None:
    """The reference must not become a rank-inflating one.

    ``V`` carries a genuinely null direction, so the profiled rank is ``k - 1``
    however it is counted.  This is the guard against "fixing" an undercount by
    loosening the cut.
    """
    k = 40
    V = np.eye(k)
    V[-1, -1] = 0.0
    C = np.zeros((2, k))
    M = np.diag([1.0, 1e-14])
    assert _profiled_edf(V, C, M) == float(k - 1)


def test_the_rank_agrees_with_a_direct_count_when_nothing_is_marginal() -> None:
    """On well-conditioned joints the ladder reproduces ``rank(V_eff)``."""
    rng = np.random.default_rng(20260805)
    for _ in range(25):
        k = int(rng.integers(3, 12))
        q = int(rng.integers(1, 4))
        r = int(rng.integers(1, k + 1))
        B = rng.normal(size=(k + q, r))
        joint = B @ B.T + np.eye(k + q) * 1e-3
        V = joint[:k, :k]
        C = joint[k:, :k]
        M = joint[k:, k:]
        v_eff = V - C.T @ np.linalg.solve(M, C)
        expected = float(np.linalg.matrix_rank(0.5 * (v_eff + v_eff.T)))
        assert _profiled_edf(V, C, M) == expected


def test_the_rank_reference_is_blind_to_the_overlap_s_units_and_tracks_the_probe_s() -> None:
    """The equilibration inside the reference, pinned as two exact identities.

    ``_profiled_rank_scale`` is the reference the unpenalized rung's rank is
    counted against, and it balances the two blocks of the joint before taking
    the top singular value, then undoes the balance so the number it returns
    applies to ``R_eff`` as it stands.  In exact arithmetic that makes it
    homogeneous of degree ONE in the probe block's units and degree ZERO in the
    overlap's::

        tensor  -> c * tensor     scale -> c * scale
        overlap -> c * overlap    scale -> scale

    Both are algebra rather than measurement.  Writing ``b`` for
    ``sqrt(mass_overlap / mass_tensor)``, scaling the tensor by ``c`` sends
    ``b -> b / c`` and leaves ``[overlap | b * tensor]`` unchanged, so the
    returned ``top / b`` picks up exactly one factor of ``c``; scaling the
    overlap by ``c`` sends ``b -> c b`` and multiplies the whole balanced
    matrix by ``c``, so ``top / b`` is unchanged.

    **THIS IS THE PIN ``test_screening_is_invariant_to_the_units_of_a_numeric
    _margin`` WAS BELIEVED TO CARRY AND DOES NOT.**  Measured: with the balance
    removed from ``_profiled_rank_scale`` -- the whole equilibration, nothing
    else -- all 265 tests in the six screening files still pass.  That test
    scales a numeric margin's units, which multiplies the tensor block by the
    SQUARE of what it multiplies the overlap by, so it only ever moves the pair
    in the direction where the unbalanced reference happens to agree.  The
    regime the equilibration exists for is the other one, overlap mass far
    ABOVE probe mass, where an unbalanced reference is set by the overlap and
    cuts away a probe direction the overlap's scale never justified.

    THE BAR IS THE MODULE'S OWN ROUND-OFF FLOOR, NOT AN OBSERVED HEADROOM.
    ``numpy.linalg.norm(., 2)`` is an SVD, whose largest singular value is
    computed to a modest polynomial in the dimensions times ``eps`` (the
    *LAPACK Users' Guide*'s own SVD error bound), and ``_rank_floor``'s
    ``width * eps`` is this package's stand-in for that polynomial.  The factor
    of 4 is a stated allowance on the stand-in, exactly as
    ``test_the_pencil_carries_its_own_orthonormality_invariant`` uses it.  Over
    the six ratios below the worst reading is 3.32e-16 against a bar of
    9.77e-15, i.e. 29x inside; the identities themselves are exact.
    """
    from superglm.screening._factor_kernels import _rank_floor
    from superglm.screening._pair_factor import PairFactor, _profiled_rank_scale

    rng = np.random.default_rng(9)
    q, k = 6, 4
    width = q + k + 1
    joint = np.triu(rng.normal(size=(width, width)))
    base = _profiled_rank_scale(PairFactor(joint=joint, overlap_width=q, tensor_width=k))
    assert base > 0.0, "the fixture must resolve something or this proves nothing"
    bar = 4.0 * _rank_floor(width)

    for c in (1e-8, 1e-4, 1e-2, 1e2, 1e4, 1e8):
        probe = joint.copy()
        probe[:, q : q + k] *= c
        got = _profiled_rank_scale(PairFactor(joint=probe, overlap_width=q, tensor_width=k))
        assert got == pytest.approx(c * base, rel=bar), ("tensor", c, got, c * base)

        nuisance = joint.copy()
        nuisance[:, :q] *= c
        got = _profiled_rank_scale(PairFactor(joint=nuisance, overlap_width=q, tensor_width=k))
        assert got == pytest.approx(base, rel=bar), ("overlap", c, got, base)
