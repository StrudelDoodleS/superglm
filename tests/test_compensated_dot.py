"""Exact arithmetic and dispatch checks for the private compiled Dot2 loop."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest


def _exact_dot(x, y):
    return sum(
        (
            Fraction.from_float(float(a)) * Fraction.from_float(float(b))
            for a, b in zip(x, y, strict=True)
        ),
        Fraction(0),
    )


def _dot2_error_bound(x, y, value):
    """Proposition 5.5 converted to a computed-value bound, in rationals."""
    unit = Fraction(1, 2**53)
    gamma = len(x) * unit / (1 - len(x) * unit)
    magnitude = _exact_dot(np.abs(x), np.abs(y))
    underflow = 5 * len(x) * Fraction(1, 2**1074)
    return (unit * abs(Fraction.from_float(value)) + gamma**2 * magnitude + underflow) / (1 - unit)


@pytest.mark.parametrize(
    ("x", "y", "exact"),
    [
        ([1e16, 1.0, -1e16], [1.0, 1.0, 1.0], Fraction(1)),
        ([1 + 2**-27, -1.0], [1 - 2**-27, 1.0], -Fraction(1, 2**54)),
    ],
    ids=["sum-cancellation", "product-cancellation"],
)
def test_compensation_recovers_exact_cancellation_and_rejects_plain_dot(x, y, exact):
    from superglm.reml._compensated import _dot2_value

    x, y = np.array(x), np.array(y)
    value, success = _dot2_value(x, y)
    assert success
    assert Fraction.from_float(value) == exact == _exact_dot(x, y)
    # Mutation control: deleting the compensation cannot satisfy this bound.
    ordinary = 0.0
    for a, b in zip(x, y, strict=True):
        ordinary += float(a) * float(b)
    assert abs(Fraction.from_float(ordinary) - exact) > _dot2_error_bound(x, y, ordinary)


@pytest.mark.parametrize("length", [1, 7, 450])
@pytest.mark.parametrize("strided", [False, True])
def test_compiled_dot_satisfies_the_unchanged_exact_computed_value_bound(length, strided):
    from superglm.reml._compensated import _dot2_value

    rng = np.random.default_rng(5817 + length)
    x = np.ldexp(rng.normal(size=length), rng.integers(-200, 201, size=length))
    y = np.ldexp(rng.normal(size=length), rng.integers(-200, 201, size=length))
    if strided:
        x, y = np.repeat(x, 2)[::2], np.repeat(y, 2)[::2]
    x.flags.writeable = False
    y.flags.writeable = False
    value, success = _dot2_value(x, y)
    assert success
    assert abs(Fraction.from_float(value) - _exact_dot(x, y)) <= _dot2_error_bound(x, y, value)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([np.nan], [1.0]),
        ([np.inf], [1.0]),
        ([1e301], [1e-301]),
        ([1e200], [1e200]),
        ([np.nextafter(0.0, 1.0)], [1.0]),
        ([np.finfo(float).tiny], [0.5]),
        ([2.0**-800], [2.0**-800]),
        ([np.finfo(float).tiny], [np.nextafter(1.0, 2.0)]),
        ([1e200, 1e200], [1e108, 1e108]),
    ],
    ids=[
        "nan",
        "infinity",
        "split-overflow",
        "product-overflow",
        "subnormal-input",
        "subnormal-product",
        "product-underflow",
        "split-part-underflow",
        "sum-overflow",
    ],
)
def test_unsupported_range_reports_fallback_status(x, y):
    from superglm.reml._compensated import _dot2_value

    _, success = _dot2_value(np.array(x), np.array(y))
    assert not success


def test_empty_dot_and_exact_zeros_need_no_fallback():
    from superglm.reml._compensated import _dot2_value

    assert _dot2_value(np.empty(0), np.empty(0)) == (0.0, True)
    assert _dot2_value(np.zeros(4), np.ones(4)) == (0.0, True)
    assert _dot2_value(np.array([1.0, -1.0]), np.ones(2)) == (0.0, True)


@pytest.mark.parametrize("without_fma", [False, True])
def test_caller_preserves_the_same_value_and_error_bound(monkeypatch, without_fma):
    import math

    from superglm.reml import multi_penalty as module

    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)
    left = np.array([1e6, 1e6 + 1, 1e-6, -3.0])
    right = np.array([1.0, -1.0, 3.0, 1e-6], dtype=np.longdouble)
    right[0] += np.longdouble(2) ** -60
    compiled = module._compensated_dot(left, right)
    monkeypatch.setattr(module, "_dot2_value", lambda *_: (0.0, False))
    fallback = module._compensated_dot(left, right)
    # Backend equivalence: the recurrence and the caller's enclosure are unchanged.
    assert compiled == fallback


@pytest.mark.parametrize("without_fma", [False, True])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1e301, 1e-301),
        (np.nextafter(0.0, 1.0), 1.0),
        (np.finfo(float).tiny, 0.5),
        (np.finfo(float).tiny, np.nextafter(1.0, 2.0)),
    ],
    ids=["split-overflow", "subnormal-input", "subnormal-product", "split-part-underflow"],
)
def test_range_fallback_preserves_the_previous_enclosure_or_refusal(
    monkeypatch, without_fma, left, right
):
    import math

    from superglm.reml import multi_penalty as module

    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)
    x, y = np.array([left]), np.array([right], dtype=np.longdouble)

    def evaluate():
        try:
            return module._compensated_dot(x, y)
        except module.PenaltyNumericalError as exc:
            return str(exc)

    routed = evaluate()
    monkeypatch.setattr(module, "_dot2_value", lambda *_: (0.0, False))
    assert routed == evaluate()
    if isinstance(routed, str):
        assert not hasattr(math, "fma") and left == 1e301
        assert "exponent scaling" in routed
    else:
        value, bound = routed
        exact = Fraction.from_float(left) * Fraction.from_float(right)
        assert abs(Fraction.from_float(value) - exact) <= Fraction.from_float(bound)


def test_actual_float64_native_dispatch_has_no_fastmath_or_owned_scratch():
    from numba import types
    from numba.core.registry import CPUDispatcher

    from superglm.reml._compensated import _dot2_value

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    assert isinstance(_dot2_value, CPUDispatcher)
    assert _dot2_value(x, y) == (32.0, True)
    assert _dot2_value.nopython_signatures
    assert _dot2_value.targetoptions.get("fastmath", False) is False
    for signature in _dot2_value.nopython_signatures:
        assert all(arg.ndim == 1 and arg.dtype == types.float64 for arg in signature.args)
    # Recompile once so IR inspection is meaningful even after a disk-cache hit.
    _dot2_value.recompile()
    llvm = _dot2_value.inspect_llvm(_dot2_value.signatures[0])
    arithmetic = [
        line
        for line in llvm.splitlines()
        if any(f"= {op} " in line for op in ("fadd", "fsub", "fmul"))
    ]
    assert arithmetic
    assert all(
        not any(flag in line.split() for flag in ("fast", "contract", "reassoc"))
        for line in arithmetic
    )
    assert "llvm.fma." not in llvm
    assert "llvm.fmuladd." not in llvm
    assert "NRT_MemInfo_alloc" not in llvm
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y, [4.0, 5.0, 6.0])
