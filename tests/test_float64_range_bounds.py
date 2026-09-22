"""Exact exponent decisions and float64 scanner dispatch, without timing gates."""

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_kernels as kernels


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.int64, np.uint64])
def test_bounds_dispatch_specializes_only_native_float64(monkeypatch, dtype):
    values = np.array([[0, 1, 2], [4, 7, 8]], dtype=dtype)
    calls = []
    implementation = getattr(
        kernels, "_float64_operand_exponent_bounds", kernels._operand_exponent_bounds
    )

    def recorded(operand):
        calls.append(operand)
        return implementation(operand)

    monkeypatch.setattr(kernels, "_float64_operand_exponent_bounds", recorded, raising=False)
    # Inspect dispatch separately from the real compiled numerical tests below.
    dispatch = getattr(
        kernels._operand_exponent_bounds, "py_func", kernels._operand_exponent_bounds
    )
    assert dispatch(values) == (0, 4)
    assert len(calls) == int(dtype == np.float64)
    if calls:
        assert calls[0] is values


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.0, -0.0], (0, 0)),
        ([0.0, 0.25, 8.0], (-2, 4)),
        ([-1.0, -2.0], (0, 2)),
        ([1.0, np.nextafter(2.0, 0.0)], (0, 1)),
        ([np.nextafter(1.0, 0.0), 2.0], (-1, 2)),
        ([np.finfo(float).smallest_subnormal], (-1074, -1073)),
        ([np.nextafter(np.finfo(float).tiny, 0.0)], (-1023, -1022)),
        ([np.finfo(float).tiny], (-1022, -1021)),
        ([np.finfo(float).max], (1023, 1024)),
        ([0.0, 1.0, np.inf], (-1024, 1024)),
        ([-np.inf, 1.0, 0.0], (-1024, 1024)),
        ([1.0, np.nan, 0.0], (-1024, 1024)),
        ([], (0, 0)),
    ],
)
def test_compiled_bounds_keep_exact_boundary_decisions(values, expected):
    with np.errstate(all="raise"):
        assert kernels._operand_exponent_bounds(np.array([values], dtype=np.float64)) == expected


@pytest.mark.parametrize("layout", ["C", "F", "transpose", "strided", "reversed"])
@pytest.mark.parametrize("readonly", [False, True])
def test_compiled_bounds_preserve_layout_and_readonly_inputs(layout, readonly):
    values = np.array([[0.0, -8.0, 0.25, 2.0], [4.0, -0.5, -0.0, 1.0]])
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "transpose":
        values = values.T
    elif layout == "strided":
        backing = np.full((4, 8), np.nan)
        backing[::2, ::2] = values
        values = backing[::2, ::2]
    elif layout == "reversed":
        values = values[::-1, ::-1]
    if readonly:
        values.setflags(write=False)
    before = values.copy()
    assert kernels._operand_exponent_bounds(values) == (-2, 4)
    np.testing.assert_array_equal(values, before)


def test_compiled_bounds_keep_power_neighbors_across_binary64_exponent_range():
    for exponent in (-1074, -1073, -1023, -1022, -128, -1, 0, 1, 128, 1023):
        power = np.ldexp(1.0, exponent)
        with np.errstate(under="ignore"):
            below, above = np.nextafter(power, 0.0), np.nextafter(power, np.inf)
        for value, expected in (
            (power, (exponent, exponent + 1)),
            (-power, (exponent, exponent + 1)),
            (below, (0, 0) if exponent == -1074 else (exponent - 1, exponent)),
            (above, (exponent, exponent + 1) if exponent != -1074 else (-1073, -1072)),
        ):
            assert kernels._operand_exponent_bounds(np.array([[value]])) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.int32, np.int64, np.uint64])
def test_compiled_bounds_preserve_generic_arithmetic(dtype):
    values = np.array([[0, 1, 2], [4, 7, 8]], dtype=dtype)
    assert kernels._operand_exponent_bounds(values) == (0, 4)


@pytest.mark.parametrize(
    "exponent,admitted", [(-1023, False), (-1022, True), (1021, True), (1022, False)]
)
def test_cross_range_gate_preserves_inclusive_exponent_limits(exponent, admitted):
    from superglm._group_matrix import _group_matrix_algebra as algebra

    factor = np.array([[np.ldexp(1.0, exponent)]])
    assert algebra._cross_factors_in_range(factor) is admitted


def test_cached_cross_bounds_reuse_dimension_work(monkeypatch):
    import builtins

    from superglm._group_matrix import _group_matrix_algebra as algebra

    # Removing contribution reuse repeats these dimension reductions on hits.
    counts = []
    original_sum = builtins.sum

    def recorded(values, *args):
        counts.append(True)
        return original_sum(values, *args)

    monkeypatch.setattr(algebra, "sum", recorded, raising=False)
    factors = (np.ones((3, 2)), np.ones((2, 4)))
    weights = np.ones(3)
    owner = algebra._BlockWeightCache()
    for _ in range(2):
        assert algebra._cross_factors_in_range(
            *factors, weights, cache=owner, support_factors=factors
        )
    assert len(counts) == 3  # Two fixed factors plus one weight vector, once.
    assert algebra._cross_factors_in_range(
        *factors, -weights, cache=owner.for_new_weights(), support_factors=factors
    )
    assert len(counts) == 4  # Fresh weights; fixed factor contributions persist.


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "value,shape,admitted",
    [
        (0.0, (0, 3), True),
        (0.0, (3, 2), True),
        (2.0**-1022, (1, 1), True),
        (2.0**-1023, (1, 1), False),
        (2.0**1021, (1, 1), True),
        (2.0**1021, (1, 2), False),
        (2.0**1019, (2, 2), True),
        (2.0**1019, (2, 3), False),
        (np.finfo(float).smallest_subnormal, (1, 1), False),
        (np.inf, (1, 1), False),
        (np.nan, (1, 1), False),
    ],
)
def test_cached_cross_bounds_preserve_dimension_and_exponent_limits(value, shape, admitted, cached):
    from superglm._group_matrix import _group_matrix_algebra as algebra

    factor = np.full(shape, value)
    cache = algebra._BlockWeightCache() if cached else None
    for _ in range(2):
        assert (
            algebra._cross_factors_in_range(factor, cache=cache, support_factors=(factor,))
            is admitted
        )


@pytest.mark.parametrize("weight_exponent,admitted", [(-700, True), (-701, False)])
def test_cached_cross_bounds_keep_combined_partial_product_refusal(weight_exponent, admitted):
    from superglm._group_matrix import _group_matrix_algebra as algebra

    # Negative bounds add independently of positive factors: -322 + exponent.
    first = np.array([[2.0**-322]])
    second = np.array([[2.0**700]])
    weight = np.array([np.ldexp(1.0, weight_exponent)])
    for cache in (None, algebra._BlockWeightCache()):
        assert (
            algebra._cross_factors_in_range(
                first, second, weight, cache=cache, support_factors=(first, second)
            )
            is admitted
        )
