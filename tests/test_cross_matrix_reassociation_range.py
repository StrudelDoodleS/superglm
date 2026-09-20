"""Exponent-sensitive cross products retain finite histogram observables."""

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    DenseGroupMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)


@pytest.mark.parametrize("direction", [-1, 1])
def test_projected_histogram_checks_weighted_intermediate_range(direction):
    indices = np.zeros(3, dtype=np.intp)
    left = DiscretizedSSPGroupMatrix(
        np.full((1, 1), np.ldexp(1.0, 100 * direction)),
        np.full((1, 1), np.ldexp(1.0, 100 * direction)),
        indices,
    )
    right = DiscretizedSSPGroupMatrix(
        np.full((1, 1), np.ldexp(1.0, -100 * direction)),
        np.full((1, 1), np.ldexp(1.0, -100 * direction)),
        indices,
    )
    weights = np.ldexp(np.array([1.0, -1.0, 1.0]), 900 * direction)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = algebra._cross_gram(left, right, weights)
    np.testing.assert_array_equal(result, [[np.ldexp(1.0, 900 * direction)]])


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("route", ["dense", "tensor_main", "tensor_own_margin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_projection_retains_finite_legacy_factor_order(direction, route, reverse):
    # B@R is outside binary64, but B'@(W*partner), then R', is exact and finite.
    indices = np.array([0, 0], dtype=np.intp)
    main = DiscretizedSSPGroupMatrix(
        np.full((1, 1), np.ldexp(1.0, 600 * direction)),
        np.full((1, 1), np.ldexp(1.0, 600 * direction)),
        indices,
    )
    partner_rows = np.ldexp(np.array([[1.0], [-1.0]]), -200 * direction)
    if route == "dense":
        partner = DenseGroupMatrix(partner_rows)
    else:
        idx1 = indices if route == "tensor_own_margin" else np.array([0, 1])
        idx2 = np.array([0, 1])
        margin1 = np.ones((idx1.max() + 1, 1))
        partner = DiscretizedTensorGroupMatrix(
            margin1,
            partner_rows,
            idx1,
            idx2,
            partner_rows,
            np.ones((1, 1)),
            idx2,
            tensor_id=1,
        )
    weights = np.ldexp(np.array([1.0, -1.0]), -600 * direction)
    left, right = (partner, main) if reverse else (main, partner)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = algebra._cross_gram(left, right, weights)
    np.testing.assert_array_equal(result, [[np.ldexp(1.0, 400 * direction + 1)]])
    if route == "dense":
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            gram = MatrixExecutionPlan((left, right), n=2).moments(weights, signed=True).gram
        np.testing.assert_array_equal(gram, [[0, result[0, 0]], [result[0, 0], 0]])


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


@pytest.mark.parametrize("direction", [-1, 1])
def test_projected_support_defers_when_projection_leaves_float64_range(direction):
    indices = np.array([0, 0, 0], dtype=np.intp)
    left = DiscretizedSSPGroupMatrix(
        np.full((256, 1), np.ldexp(1.0, 600 * direction)),
        np.array([[np.ldexp(1.0, 600 * direction)]]),
        indices,
    )
    right = DiscretizedSSPGroupMatrix(
        np.full((256, 1), np.ldexp(1.0, -600 * direction)), np.ones((1, 1)), indices
    )
    weights = np.ldexp(np.array([1.0, -1.0, 1.0]), -800 * direction)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = algebra._cross_gram(left, right, weights)
    np.testing.assert_array_equal(result, [[np.ldexp(1.0, -200 * direction)]])


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


def _tensors(left_exp, right_exp):
    # 256 x 256 grids with every cell on the stored joint support: the
    # displaced route's n_joint is 65536**2, far above the histogram cell cap,
    # so a decline lands on the bounded row route exactly as it does in
    # production.
    indices = np.array([0, 0, 0], dtype=np.intp)

    def tensor(exponent, tensor_id):
        margin = np.full((256, 2), np.ldexp(1.0, exponent))
        joint = np.einsum("ia,jb->ijab", margin, margin).reshape(256 * 256, 4)
        return DiscretizedTensorGroupMatrix(
            margin, margin, indices, indices, joint, np.eye(4), indices, tensor_id=tensor_id
        )

    return tensor(left_exp, 1), tensor(right_exp, 2)


@pytest.mark.parametrize("extreme", [False, True])
def test_tensor_channel_route_declines_outside_the_reassociation_range(extreme):
    # Margins at 2**(-/+200) sit outside the [2**-128, 2**128] operand range
    # the channel route reassociates under, although their row products
    # (2**-400 times 2**400) are safe: the route must decline to the bounded
    # row route rather than reassociate, and the block stays finite either way.
    left, right = _tensors(-200 if extreme else 0, 200 if extreme else 0)
    profile = {}
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        actual = algebra._cross_gram(left, right, np.array([1.0, -1.0, 1.0]), profile=profile)
    assert np.isfinite(actual).all()
    bound = 8 * np.finfo(actual.dtype).eps * 256 * 256
    assert np.linalg.norm(actual - np.ones((4, 4)), ord=np.inf) <= bound
    assert profile.get("block_cross_tensor_tensor_channel_calls", 0) == int(not extreme)
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == int(extreme)
