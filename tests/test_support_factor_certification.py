"""Same-call range certification without persistent design assumptions."""

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix import _group_matrix_discretized as discrete
from superglm._group_matrix import _group_matrix_kernels as kernels
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan


def _group(dtype=np.float64):
    return discrete.DiscretizedSSPGroupMatrix(
        np.array([[1, 2], [2, 1]], dtype=dtype),
        np.array([[1, 0], [0, 2]], dtype=dtype),
        np.array([0, 1, 0]),
    )


@pytest.mark.parametrize("route", ["standalone", "plain", "signed", "fused"])
def test_support_factors_scanned_once_per_diagonal_call(monkeypatch, route):
    group = _group()
    plan = MatrixExecutionPlan((group,), n=3)
    weights = np.array([1.0, 0.5, 2.0])
    scans = [0, 0]
    original = kernels._tensor_operand_in_reassociation_range

    def recorded(values):
        scans[0] += int(values is group.B_unique)
        scans[1] += int(values is group.R_inv)
        return original(values)

    monkeypatch.setattr(kernels, "_tensor_operand_in_reassociation_range", recorded)
    for _ in range(2):
        if route == "standalone":
            group.gram(weights)
        else:
            plan.moments(
                weights,
                signed=route == "signed",
                rhs=(weights,) if route == "fused" else (),
            )
        group.B_unique *= 0.5
        group.R_inv *= 0.5
        weights *= 2.0
    assert scans == [2, 2]


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_certified_moments_observe_changed_factors_weights_and_rhs(replace, fused):
    group = _group()
    plan = MatrixExecutionPlan((group,), n=3)
    weights = np.array([1.0, -0.5, 2.0])
    rhs = np.array([-0.5, 2.0, 1.0])
    for iteration in range(2):
        if iteration:
            for name in ("B_unique", "R_inv"):
                changed = getattr(group, name) * 0.5
                if replace:
                    setattr(group, name, changed)
                else:
                    getattr(group, name)[:] = changed
            weights *= 2.0
            rhs *= -0.5
        design = group.B_unique[group.bin_idx] @ group.R_inv
        expected = design.T @ (weights[:, None] * design)
        moments = plan.moments(weights, signed=True, rhs=(rhs,) if fused else ())
        scale = np.max(abs(design).T @ (abs(weights[:, None]) * abs(design)))
        bound = 16 * np.finfo(float).eps * max(design.shape) * scale
        np.testing.assert_allclose(moments.gram, expected, rtol=0, atol=bound)
        if fused:
            np.testing.assert_array_equal(moments.xt_rhs[0], design.T @ rhs)


def _record_exact(monkeypatch):
    calls = []
    original = discrete._exact_ssp_moments

    def recorded(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(discrete, "_exact_ssp_moments", recorded)
    return calls


@pytest.mark.parametrize("operand", ["B_unique", "R_inv", "weights", "rhs"])
def test_assembly_keeps_each_exceptional_operand_guard(monkeypatch, operand):
    group = discrete.DiscretizedSSPGroupMatrix(np.ones((1, 1)), np.ones((1, 1)), np.array([0]))
    weights, rhs = np.ones(1), np.ones(1)
    large = np.ldexp(1.0, 200)
    if operand in ("B_unique", "R_inv"):
        getattr(group, operand)[:] = large
    elif operand == "weights":
        weights[:] = large
    else:
        rhs[:] = large
    calls = _record_exact(monkeypatch)
    moments = MatrixExecutionPlan((group,), n=1).moments(weights, rhs=(rhs,))
    x = large if operand in ("B_unique", "R_inv") else 1.0
    np.testing.assert_array_equal(moments.gram, [[x * weights[0] * x]])
    np.testing.assert_array_equal(moments.xt_rhs[0], [x * rhs[0]])
    assert len(calls) == 1


@pytest.mark.parametrize("replace", [False, True])
def test_next_assembly_rejects_previously_certified_factors(monkeypatch, replace):
    group = discrete.DiscretizedSSPGroupMatrix(np.ones((1, 1)), np.ones((1, 1)), np.array([0]))
    plan = MatrixExecutionPlan((group,), n=1)
    calls = _record_exact(monkeypatch)
    plan.moments(np.ones(1), rhs=(np.ones(1),))
    assert not calls
    for name, exponent in (("B_unique", 200), ("R_inv", -200)):
        changed = np.full((1, 1), np.ldexp(1.0, exponent))
        if replace:
            setattr(group, name, changed)
        else:
            getattr(group, name)[:] = changed
    moments = plan.moments(np.ones(1), rhs=(np.ones(1),))
    np.testing.assert_array_equal(moments.gram, [[1.0]])
    np.testing.assert_array_equal(moments.xt_rhs[0], [1.0])
    assert len(calls) == 1


@pytest.mark.parametrize("fused", [False, True])
def test_supplied_support_is_not_a_factor_range_certificate(monkeypatch, fused):
    group = discrete.DiscretizedSSPGroupMatrix(
        np.full((1, 1), np.ldexp(1.0, 200)),
        np.full((1, 1), np.ldexp(1.0, -200)),
        np.array([0]),
    )
    calls = _record_exact(monkeypatch)
    # The wider cross guard can supply support without the legacy B/R guard.
    # Deliberately wrong supplied rows must not bypass the exact-source route.
    if fused:
        gram, _, _ = group.gram_rmatvec(np.ones(1), np.ones(1), _support=np.full((1, 1), 7.0))
    else:
        gram = group.gram(np.ones(1), _support=np.full((1, 1), 7.0))
    np.testing.assert_array_equal(gram, [[1.0]])
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.longdouble, np.complex128])
@pytest.mark.parametrize("fused", [False, True])
def test_standalone_dtype_short_circuit_stays_unchanged(monkeypatch, dtype, fused):
    group = discrete.DiscretizedSSPGroupMatrix(
        np.ones((1, 1), dtype=dtype), np.ones((1, 1), dtype=dtype), np.array([0])
    )
    weight = np.array([np.ldexp(-1.0, 200)])
    calls = _record_exact(monkeypatch)
    if fused:
        gram, _, _ = group.gram_rmatvec(weight, np.ones(1), _support=np.ones((1, 1)))
    else:
        gram = group.gram(weight, _support=np.ones((1, 1)))
    np.testing.assert_array_equal(gram, weight.reshape(1, 1))
    supported = np.dtype(dtype).kind in "bifu" and np.dtype(dtype).itemsize <= 8
    assert len(calls) == int(supported)


@pytest.mark.parametrize("exponent", [-128, 128])
@pytest.mark.parametrize("outside", [False, True])
def test_assembly_preserves_inclusive_legacy_factor_endpoints(monkeypatch, exponent, outside):
    value = np.ldexp(1.0, exponent)
    if outside:
        value = np.nextafter(value, 0.0 if exponent < 0 else np.inf)
    group = discrete.DiscretizedSSPGroupMatrix(np.array([[value]]), np.ones((1, 1)), np.array([0]))
    calls = _record_exact(monkeypatch)
    gram = MatrixExecutionPlan((group,), n=1).moments(np.ones(1), signed=True).gram
    np.testing.assert_array_equal(gram, [[value * value]])
    assert len(calls) == int(outside)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_discrete_pair_shares_one_range_decision(monkeypatch, cached, compressed):
    left, right = _group(), _group()
    if compressed:
        right = discrete.SupportCompressedSSPGroupMatrix(right.B_unique, right.R_inv, right.bin_idx)
    weights = np.array([1.0, -0.5, 2.0])
    calls = []
    original = algebra._cross_factors_in_range

    def recorded(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(algebra, "_cross_factors_in_range", recorded)
    for iteration in range(2):
        if iteration:
            left.B_unique *= 0.5
            right.R_inv *= 2
            weights *= -0.5
        cache = algebra._BlockWeightCache() if cached else None
        actual = algebra._cross_gram(left, right, weights, cache)
        first = left.B_unique[left.bin_idx] @ left.R_inv
        second = right.B_unique[right.bin_idx] @ right.R_inv
        target = first.T @ (weights[:, None] * second)
        scale = abs(first).T @ (abs(weights[:, None]) * abs(second))
        np.testing.assert_allclose(
            actual, target, rtol=0, atol=32 * np.finfo(float).eps * scale.max()
        )
    assert len(calls) == 2  # One independent decision in each explicit assembly.


@pytest.mark.parametrize("kind", ["extreme", "subnormal", "zero", "float32", "custom"])
def test_discrete_pair_keeps_each_original_support_decision(monkeypatch, kind):
    left, right = _group(), _group()
    weights = np.array([1.0, -0.5, 2.0])
    if kind == "extreme":
        left.B_unique *= np.ldexp(1.0, 700)
        left.R_inv *= np.ldexp(1.0, -700)
    elif kind == "subnormal":
        left.B_unique[:] = np.finfo(float).smallest_subnormal
    elif kind == "zero":
        left.B_unique[:] = 0
    elif kind == "float32":
        left.R_inv = left.R_inv.astype(np.float32)
    else:

        class CustomGroup(discrete.DiscretizedSSPGroupMatrix):
            pass

        left = CustomGroup(left.B_unique, left.R_inv, left.bin_idx)
    # Independently preserve each old predicate/association as the route oracle.
    before = [
        algebra._cross_support(gm, None, weights, other.B_unique, other.R_inv)
        for gm, other in ((left, right), (right, left))
    ]
    (bi, ri), (bj, rj) = before
    histogram = np.zeros((len(bi), len(bj)))
    np.add.at(histogram, (left.bin_idx, right.bin_idx), weights)
    target = bi.T @ histogram @ bj
    if ri is not None:
        target = ri.T @ target
    if rj is not None:
        target = target @ rj
    actual = algebra._cross_gram(left, right, weights)
    np.testing.assert_array_equal(actual, target)
