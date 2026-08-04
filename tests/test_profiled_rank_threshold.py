"""Both Guttman operands must be counted against ONE ruler.

``_profiled_rank`` takes ``rank([[V, C'], [C, M]]) - rank(M)``.  Counting each
operand at its own relative floor makes the two thresholds differ by dimension
-- ``(k + q) * eps`` against ``q * eps`` -- so a nuisance direction can be
dropped from the joint count while surviving in the nuisance count, and the
difference undercounts by a whole degree of freedom.

These tests assert the PROFILED RANK itself, not a downstream ``z``.  The five
screening test files pass identically with and without the fix, so nothing else
in the tree observes this.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.screening._score_stat import _profiled_rank


def _joint(nuisance_eigenvalue: float, k: int = 100) -> tuple:
    """Two independent blocks: ``C = 0``, so the true profiled rank is ``k``."""
    V = np.eye(k)
    C = np.zeros((2, k))
    M = np.diag([1.0, nuisance_eigenvalue])
    return V, C, M


def test_profiled_rank_counts_both_operands_at_one_ruler() -> None:
    """The reviewer's construction: independent blocks, true profiled rank 100."""
    V, C, M = _joint(1e-14)
    assert _profiled_rank(V, (V, C, M)) == 100.0


@pytest.mark.parametrize("nuisance", [1e-13, 1e-14, 1e-15, 2e-14, 5e-15])
def test_profiled_rank_holds_across_the_window_between_the_two_floors(
    nuisance: float,
) -> None:
    """A nuisance eigenvalue anywhere in the disputed window still gives 100.

    ``q * eps`` is 4.44e-16 and ``(k + q) * eps`` is 2.26e-14, so these values
    straddle the gap the mismatched rulers opened.  Every one of them is a
    direction that either belongs to both counts or to neither.
    """
    V, C, M = _joint(nuisance)
    assert _profiled_rank(V, (V, C, M)) == 100.0


def test_profiled_rank_still_drops_a_shared_null_direction() -> None:
    """One ruler must not become a rank-inflating one.

    ``V`` carries a genuinely null direction, so the profiled rank is ``k - 1``
    however the operands are counted.  This is the guard against "fix" the
    undercount by loosening the joint cut.
    """
    k = 40
    V = np.eye(k)
    V[-1, -1] = 0.0
    C = np.zeros((2, k))
    M = np.diag([1.0, 1e-14])
    assert _profiled_rank(V, (V, C, M)) == float(k - 1)


def test_profiled_rank_agrees_with_a_direct_count_when_nothing_is_marginal() -> None:
    """On well-conditioned joints the Guttman route reproduces ``rank(V_eff)``."""
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
        assert _profiled_rank(v_eff, (V, C, M)) == expected
