"""Exponent-sensitive cross products retain finite histogram observables."""

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.group_matrix import DiscretizedSCOPGroupMatrix, DiscretizedSSPGroupMatrix


def _groups(left_kind, right_kind, left_exp, right_exp):
    indices = np.array([0, 0, 0], dtype=np.intp)

    def group(kind, exponent):
        support = np.full((256, 4), np.ldexp(1.0, exponent))
        if kind == "ssp":
            return DiscretizedSSPGroupMatrix(support, np.eye(4), indices)
        return DiscretizedSCOPGroupMatrix(support, indices)

    return group(left_kind, left_exp), group(right_kind, right_exp)


@pytest.mark.parametrize("left_kind", ["ssp", "scop"])
@pytest.mark.parametrize("right_kind", ["ssp", "scop"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_cross_preserves_histogram_exponent_range(left_kind, right_kind, direction):
    # The signed weights sum to one power of two. Histogram association first
    # forms 2**(+/-500), then the finite 2**(+/-900) result. Weighting the right
    # panel first instead overflows (or underflows) at 2**(+/-1100).
    left, right = _groups(left_kind, right_kind, -200 * direction, 400 * direction)
    weights = np.ldexp(np.array([1.0, -1.0, 1.0]), 700 * direction)
    expected_scale = np.ldexp(1.0, 900 * direction)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        actual = algebra._cross_gram(left, right, weights)
    assert np.isfinite(actual).all()
    bound = 8 * np.finfo(actual.dtype).eps * 256
    assert np.linalg.norm(actual / expected_scale - np.ones((4, 4)), ord=np.inf) <= bound


@pytest.mark.parametrize("force_rows", [False, True])
@pytest.mark.parametrize("extreme", [False, True])
def test_range_gate_preserves_safe_rows_and_histogram_cell_cap(monkeypatch, force_rows, extreme):
    # Out-of-gate operands can still have safe row products. Above the cap the
    # established bounded row route must remain available without a histogram.
    left, right = _groups("scop", "scop", -200 if extreme else 0, 200 if extreme else 0)
    if force_rows:
        monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    profile = {}
    algebra._cross_gram(left, right, np.array([1.0, -1.0, 1.0]), profile=profile)
    expect_rows = force_rows or not extreme
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == int(expect_rows)
    assert profile.get("block_cross_disc_disc_hist_calls", 0) == int(not expect_rows)
