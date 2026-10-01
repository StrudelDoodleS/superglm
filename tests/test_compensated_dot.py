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
    left = np.pad([1e6, 1e6 + 1, 1e-6, -3.0], (0, 252))
    right = np.pad([np.nextafter(1.0, np.inf), -1.0, 3.0, 1e-6], (0, 252))
    original, calls = module._dot2_value, []

    def native(x, y):
        calls.append(len(x))
        return original(x, y)

    monkeypatch.setattr(module, "_dot2_value", native)
    compiled = module._compensated_dot(left, right)
    assert calls == [256]
    monkeypatch.setattr(module, "_dot2_value", lambda *_: (0.0, False))
    fallback = module._compensated_dot(left, right)
    # Backend equivalence: the recurrence and the caller's enclosure are unchanged.
    assert compiled == fallback
    assert abs(Fraction.from_float(compiled[0]) - _exact_dot(left, right)) <= Fraction.from_float(
        compiled[1]
    )


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
    x, y = np.array([left]), np.array([right], dtype=np.float64)

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


@pytest.mark.parametrize("without_fma", [False, True])
def test_small_compensated_reductions_do_not_initialize_native_kernels(monkeypatch, without_fma):
    import math

    from superglm.reml import multi_penalty as module

    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)

    def forbidden(*_):
        pytest.fail("tiny reduction initialized a native kernel")

    monkeypatch.setattr(module, "_dot2_value", forbidden)
    monkeypatch.setattr(module, "_dot2_selected", forbidden)
    for left, right in (
        ([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]),
        ([1 + 2**-27, -1.0], [1 - 2**-27, 1.0]),
    ):
        left, right = np.asarray(left), np.asarray(right)
        value, bound = module._compensated_dot(left, right)
        assert abs(Fraction.from_float(value) - _exact_dot(left, right)) <= Fraction.from_float(
            bound
        )
    root = np.tile([1e16, 1.0, -1e16], (3, 1))
    (action,), (error,) = module._reference_root_actions([root], np.array([4.0]), np.ones((3, 2)))
    assert np.all(np.abs(action - 2.0) <= error)


def test_reference_action_refinement_batches_scalar_validation_work(monkeypatch):
    from superglm.reml import multi_penalty as module

    root = np.tile([1e16, 1.0, -1e16], (32, 1))
    inverse = np.tile([[1.0], [1.0], [1.0]], (1, 4))
    original_dot, original_positive = module._compensated_dot, module._positive_product
    dots, products = [], []

    def dot(*args):
        dots.append(1)
        return original_dot(*args)

    def positive(left, right):
        products.append((left.shape, right.shape))
        return original_positive(left, right)

    monkeypatch.setattr(module, "_compensated_dot", dot)
    monkeypatch.setattr(module, "_positive_product", positive)
    module._reference_root_actions([root], np.array([4.0]), inverse)
    # All 128 normal-range dots need refinement. Validating/bounding each
    # separately recreates the measured million-call complete-fit bottleneck.
    assert len(dots) == 0
    assert len(products) <= 3


@pytest.mark.parametrize("without_fma", [False, True])
def test_small_action_batch_keeps_shared_bounds_without_native_startup(monkeypatch, without_fma):
    import math

    from superglm.reml import multi_penalty as module

    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)

    def forbidden(*_):
        pytest.fail("small batch dispatched native work or repeated scalar validation")

    for name in ("_dot2_selected", "_dot2_value", "_compensated_dot"):
        monkeypatch.setattr(module, name, forbidden)
    root = np.tile([1e16, 1.0, -1e16], (3, 1))
    (action,), (error,) = module._reference_root_actions([root], np.array([4.0]), np.ones((3, 2)))
    np.testing.assert_array_equal(action, np.full((3, 2), 2.0))
    assert np.all(error >= 0)


def test_selected_dot2_dispatch_preserves_the_scalar_recurrence_and_status():
    from numba.core.registry import CPUDispatcher

    from superglm.reml._compensated import _dot2_selected

    left = np.array([[1e16, 1.0, -1e16], [np.nextafter(0.0, 1.0), 1.0, 0.0]])
    right = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    left.flags.writeable = right.flags.writeable = False
    indices = np.array([[0, 1], [1, 0], [0, 0]])
    values, success = _dot2_selected(left, right, indices)
    assert isinstance(_dot2_selected, CPUDispatcher)
    assert _dot2_selected.nopython_signatures
    assert _dot2_selected.targetoptions.get("fastmath", False) is False
    np.testing.assert_array_equal(success, [True, False, True])
    np.testing.assert_array_equal(values, [-1.0, 0.0, 1.0])
    for index in (0, 2):
        row, column = indices[index]
        assert Fraction.from_float(values[index]) == _exact_dot(left[row], right[:, column])


def test_refined_actions_recompute_after_root_weight_and_inverse_changes():
    from superglm.reml import multi_penalty as module

    root = np.tile([1e16, 1.0, -1e16], (2, 1))
    inverse, weights = np.ones((3, 1)), np.ones(1)
    for changed, expected in (
        (None, [1.0, 1.0]),
        ("root", [3.0, 1.0]),
        ("weight", [6.0, 2.0]),
        ("inverse", [12.0, 4.0]),
    ):
        if changed == "root":
            root[0, 1] = 3.0
        elif changed == "weight":
            weights[0] = 4.0
        elif changed == "inverse":
            inverse[1, 0] = 2.0
        (action,), _ = module._reference_root_actions([root], weights, inverse)
        np.testing.assert_array_equal(action[:, 0], expected)


def test_batched_refinement_routes_only_unsupported_selected_entries_to_scalar(monkeypatch):
    from superglm.reml import multi_penalty as module

    root = np.ones((1, 256))
    tiny = np.nextafter(0.0, 1.0)
    inverse = np.zeros((256, 2))
    inverse[:2] = [[1.0, 1.0], [tiny, -1.0]]
    original, fallback_inputs = module._compensated_dot, []

    def dot(left, right):
        fallback_inputs.append(tuple(right))
        return original(left, right)

    monkeypatch.setattr(module, "_compensated_dot", dot)
    module._reference_root_actions([root], np.array([4.0]), inverse)
    assert fallback_inputs == [tuple(inverse[:, 0])]


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("scale_exponent", [-250, -1, 1, 250])
def test_refined_root_actions_enclose_exact_weighted_cancellation(strided, scale_exponent):
    from superglm.reml import multi_penalty as module

    root = np.tile([1e16, 1.0, -1e16], (32, 1))
    inverse = np.array([[1.0, 1.0, 2.0], [1.0, -1.0, 3.0], [1.0, 1.0, 2.0]])
    if strided:
        root = np.repeat(np.repeat(root, 2, axis=0), 2, axis=1)[::2, ::2]
        inverse = np.repeat(np.repeat(inverse, 2, axis=0), 2, axis=1)[::2, ::2]
    root.flags.writeable = inverse.flags.writeable = False
    before_root, before_inverse = root.copy(), inverse.copy()
    weight = np.array([2.0 ** (2 * scale_exponent)])
    (action,), (error,) = module._reference_root_actions([root], weight, inverse)
    scale = Fraction.from_float(2.0**scale_exponent)
    for row, column in np.ndindex(action.shape):
        exact = scale * _exact_dot(root[row], inverse[:, column])
        assert abs(Fraction.from_float(action[row, column]) - exact) <= Fraction.from_float(
            error[row, column]
        )
    np.testing.assert_array_equal(root, before_root)
    np.testing.assert_array_equal(inverse, before_inverse)


@pytest.mark.parametrize("without_fma", [False, True])
@pytest.mark.parametrize(
    ("root", "inverse", "weight"),
    [
        ([[1e301, 1e301]], [[1e-301, 0.0], [-1e-301, 1e-301]], 4.0),
        ([[1.0, 1.0]], [[1.0, 1.0], [np.nextafter(0.0, 1.0), -1.0]], 4.0),
        ([[1e308, 1e308]], [[1.0, 1.0], [-1.0, 0.0]], 2.0**-20),
    ],
    ids=["split-overflow", "subnormal-input", "unweighted-magnitude-overflow"],
)
def test_refinement_retains_scalar_range_fallback(monkeypatch, without_fma, root, inverse, weight):
    import math

    from superglm.reml import multi_penalty as module

    if without_fma:
        monkeypatch.delattr(math, "fma", raising=False)
    root, inverse = np.array(root), np.array(inverse)
    # A small positive weight makes the third fixture's native action finite
    # even though its unweighted absolute dot is not representable.
    (action,), (error,) = module._reference_root_actions([root], np.array([weight]), inverse)
    for row, column in np.ndindex(action.shape):
        exact = Fraction.from_float(math.sqrt(weight)) * _exact_dot(root[row], inverse[:, column])
        assert abs(Fraction.from_float(action[row, column]) - exact) <= Fraction.from_float(
            error[row, column]
        )


def _entrywise_refinement(left, right, value, error):
    """The entry-by-entry Dot2 loop that ``_refine_product`` batches."""
    from superglm.reml import multi_penalty as module

    for row, column in np.ndindex(value.shape):
        corrected, bound = module._compensated_dot(left[row], right[:, column])
        if bound < error[row, column]:
            value[row, column], error[row, column] = corrected, bound


@pytest.mark.parametrize("inner", [3, 300], ids=["scalar-batch", "native-batch"])
def test_batched_product_refinement_reproduces_the_entrywise_loop(monkeypatch, inner):
    from superglm.reml import multi_penalty as module

    rng = np.random.default_rng(9120 + inner)
    left = np.ldexp(rng.normal(size=(6, inner)), rng.integers(-40, 41, size=(6, inner)))
    right = np.ldexp(rng.normal(size=(inner, 5)), rng.integers(-40, 41, size=(inner, 5)))
    left[0, 0] = 3 * np.nextafter(0.0, 1.0)  # outside the native range: scalar fallback
    # A zero row's native enclosure, 2k + 1 subnormals, is tighter than Dot2's
    # 5k: its entries must keep the native value and bound.
    left[1] = 0.0
    native = module._matmul_enclosed(left, right)
    batched = tuple(array.copy() for array in native)
    entrywise = tuple(array.copy() for array in native)
    original, calls = module._compensated_dot, []
    monkeypatch.setattr(module, "_compensated_dot", lambda *a: calls.append(1) or original(*a))
    module._refine_product(left, right, *batched)
    assert len(calls) == (5 if inner == 300 else 0)  # only row 0 leaves the native batch
    _entrywise_refinement(left, right, *entrywise)
    # Same Dot2 recurrence in the same order, same adoption: identical values.
    np.testing.assert_array_equal(batched[0], entrywise[0])
    assert np.any(batched[0] != native[0])
    np.testing.assert_array_equal(batched[1][1], native[1][1])
    # Enclosures differ only through |left| @ |right|, enclosed once (batched)
    # or per entry. Each evaluation of the Proposition 5.5 formula rounds at
    # most seven times, so the difference is bounded by the magnitude change.
    shared = module._positive_product(np.abs(left), np.abs(right))
    single = np.array(
        [
            [
                module._positive_product(np.abs(x)[None, :], np.abs(y)[:, None])[0, 0]
                for y in right.T
            ]
            for x in left
        ]
    )
    squared = np.float64(module._gamma(inner)) ** 2
    largest = np.maximum(batched[1], entrywise[1])
    allowed = 2 * squared * np.abs(shared - single) + 3 * module._gamma(7) * largest
    assert np.all(np.abs(batched[1] - entrywise[1]) <= allowed)
    for row, column in np.ndindex(batched[0].shape):
        exact = _exact_dot(left[row], right[:, column])
        assert abs(Fraction.from_float(batched[0][row, column]) - exact) <= Fraction.from_float(
            batched[1][row, column]
        )


def test_product_refinement_batches_are_bounded_by_entries(monkeypatch):
    """#436 review (Codex): one batch over every entry of a wide product held
    index, value, bound and magnitude arrays for all of them at once. Output
    rows are refined in blocks of at most ``_DOT2_BATCH_ENTRIES`` entries, with
    the entrywise loop's values and certified enclosures."""
    from superglm.reml import multi_penalty as module

    rng = np.random.default_rng(9140)
    left = np.ldexp(rng.normal(size=(7, 300)), rng.integers(-40, 41, size=(7, 300)))
    right = np.ldexp(rng.normal(size=(300, 5)), rng.integers(-40, 41, size=(300, 5)))
    native = module._matmul_enclosed(left, right)
    batched = tuple(array.copy() for array in native)
    entrywise = tuple(array.copy() for array in native)
    original, sizes = module._dot2_selected, []
    monkeypatch.setattr(
        module, "_dot2_selected", lambda *a: sizes.append(len(a[2])) or original(*a)
    )
    monkeypatch.setattr(module, "_DOT2_BATCH_ENTRIES", 11)  # two rows of five per block
    module._refine_product(left, right, *batched)
    assert sizes == [10, 10, 10, 5]
    _entrywise_refinement(left, right, *entrywise)
    np.testing.assert_array_equal(batched[0], entrywise[0])
    for row, column in np.ndindex(batched[0].shape):
        exact = _exact_dot(left[row], right[:, column])
        assert abs(Fraction.from_float(batched[0][row, column]) - exact) <= Fraction.from_float(
            batched[1][row, column]
        )


def test_refined_support_volume_is_one_batch_with_the_entrywise_value(monkeypatch):
    from types import SimpleNamespace

    from superglm.reml import multi_penalty as module
    from superglm.reml import penalty_algebra as algebra

    width = 20
    second = np.diff(np.eye(width), 2, axis=0)
    first = np.diff(np.eye(width)[:10], 1, axis=0)
    raw = [second.T @ second, first.T @ first]  # rank 19: constants are unpenalized
    rng = np.random.default_rng(3319)
    coordinate_map = np.linalg.qr(rng.normal(size=(width, width)))[0] * rng.uniform(1, 2, width)
    group = SimpleNamespace(name="shared", sl=slice(0, width), size=width)
    matrix = SimpleNamespace(
        R_inv=coordinate_map,
        omega=sum(raw),
        omega_components=[("a", raw[0]), ("b", raw[1])],
    )
    components, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    geometry = algebra._context_geometry(components)
    assert geometry is not None and geometry.coordinate_map is not None
    support, mapping = geometry.get_support(), geometry.coordinate_map
    original, calls = module._compensated_dot, []
    monkeypatch.setattr(module, "_compensated_dot", lambda *a: calls.append(1) or original(*a))
    unrefined = algebra._support_coordinate_volume(support, mapping)
    batched = algebra._support_coordinate_volume(support, mapping, _refine=True)
    assert calls == []  # every normal-range entry refined in one native batch
    monkeypatch.setattr(module, "_refine_product", _entrywise_refinement)
    entrywise = algebra._support_coordinate_volume(support, mapping, _refine=True)
    assert len(calls) == width * support.rank
    assert batched[0] == entrywise[0]
    assert 0 <= batched[1] <= unrefined[1]


def _capped_difference_penalty(width: int = 8, lam: float = 1.0e10):
    """A second-difference penalty at a capped lambda and an O(1) null-space state.

    ``b`` is a linear trend, which the penalty annihilates, plus a 1e-10 wiggle,
    as at an endpoint-authority cap fit: ``|b|' |P| |b|`` is near 1e12 while
    ``b' P b`` is near 1e-9.
    """
    difference = np.diff(np.eye(width), n=2, axis=0)
    penalty = lam * (difference.T @ difference)
    rng = np.random.default_rng(427)
    coefficients = 0.3 + 0.7 * np.linspace(0.0, 1.0, width) + 1.0e-10 * rng.normal(size=width)
    return penalty, coefficients


def _quadratic_form_error_bound(matrix, vector):
    """Two-stage Proposition 5.5 bound for ``Dot2(x, Dot2-rows(A, x))``, in rationals.

    Rows: ``|v_i - w_i| <= u |w_i| + g a_i`` with ``w = A x``, ``a = |A| |x|`` and
    ``g = gamma_n**2``. Outer: ``|res - x'v| <= u |x'v| + g |x|'|v|``. With
    ``|x'v| <= |q| + E``, ``E = u |x|'|w| + g |x|'a`` and ``|v| <= (1 + u + g) a``,
    ``|res - q| <= u |q| + (1 + u) E + g (1 + u + g) |x|'a``. No underflow occurs
    here, which the kernel's success flag certifies.
    """
    unit = Fraction(1, 2**53)
    gamma = len(vector) * unit / (1 - len(vector) * unit)
    g = gamma**2
    rows = [_exact_dot(row, vector) for row in matrix]
    absolute_rows = [_exact_dot(np.abs(row), np.abs(vector)) for row in matrix]
    magnitudes = [abs(Fraction.from_float(float(value))) for value in vector]
    exact = sum(
        (Fraction.from_float(float(x)) * w for x, w in zip(vector, rows, strict=True)),
        Fraction(0),
    )
    weighted_rows = sum((m * abs(w) for m, w in zip(magnitudes, rows, strict=True)), Fraction(0))
    weighted_absolute = sum(
        (m * a for m, a in zip(magnitudes, absolute_rows, strict=True)), Fraction(0)
    )
    first_stage = unit * weighted_rows + g * weighted_absolute
    bound = unit * abs(exact) + (1 + unit) * first_stage + g * (1 + unit + g) * weighted_absolute
    return exact, bound


def _nonzero_entries(matrix):
    rows, columns = np.nonzero(matrix)
    return rows, columns, matrix[rows, columns]


def test_quadratic_form_resolves_a_capped_penalty_and_rejects_the_plain_product():
    from superglm.reml._compensated import _dot2_quadratic_form

    penalty, coefficients = _capped_difference_penalty()
    value, success = _dot2_quadratic_form(*_nonzero_entries(penalty), coefficients)
    exact, bound = _quadratic_form_error_bound(penalty, coefficients)
    assert success
    assert abs(Fraction.from_float(value) - exact) <= bound
    # Mutation control: the plain product cannot satisfy the bound. It is summed
    # in a fixed recursive order, so IEEE arithmetic alone decides this, not BLAS.
    plain = 0.0
    for x_i, row in zip(coefficients.tolist(), penalty.tolist(), strict=True):
        row_value = 0.0
        for p_ij, x_j in zip(row, coefficients.tolist(), strict=True):
            row_value = row_value + p_ij * x_j
        plain = plain + x_i * row_value
    assert abs(Fraction.from_float(plain) - exact) > bound


def test_quadratic_form_over_nonzeros_matches_the_dense_recurrence():
    """Skipping zero entries leaves every Dot2 state unchanged, so the values agree."""
    from superglm.reml._compensated import _dot2_quadratic_form, _dot2_value

    capped, coefficients = _capped_difference_penalty()
    width = len(coefficients)
    penalty = np.zeros((2 * width + 2, 2 * width + 2))
    # Unpenalized intercepts at 0 and width + 1; two penalized blocks.
    penalty[1 : width + 1, 1 : width + 1] = capped
    penalty[width + 2 :, width + 2 :] = 0.5 / 1.0e10 * capped
    vector = np.concatenate(([2.5], coefficients, [-1.5], coefficients[::-1]))
    rows = np.array([_dot2_value(row, vector)[0] for row in penalty])
    dense, dense_success = _dot2_value(vector, rows)
    value, success = _dot2_quadratic_form(*_nonzero_entries(penalty), vector)
    assert success and dense_success
    assert value == dense
    assert _dot2_quadratic_form(*_nonzero_entries(np.zeros((3, 3))), np.ones(3)) == (0.0, True)


@pytest.mark.parametrize("seed", range(6))
def test_quadratic_form_matches_its_row_then_outer_dot2_composition(seed):
    """The streamed kernel is Dot2 per row, then Dot2 over the rows, bit for bit."""
    from superglm.reml._compensated import _dot2_quadratic_form, _dot2_value, _nonzero_entries

    rng = np.random.default_rng(seed)
    width = int(rng.integers(1, 40))
    density = (0.0, 0.1, 0.5, 1.0, 0.3, 1.0)[seed]
    matrix = np.ldexp(rng.normal(size=(width, width)), rng.integers(-30, 31, size=(width, width)))
    matrix[rng.uniform(size=matrix.shape) >= density] = 0.0
    vector = np.ldexp(rng.normal(size=width), rng.integers(-30, 31, size=width))
    if seed == 5:
        vector[0] = np.nextafter(0.0, 1.0)  # outside Dot2's range: both refuse
    rows, columns, entries = _nonzero_entries(matrix)
    row_values, row_weights, valid = [], [], True
    start = 0
    while start < len(entries):
        stop = start + int(np.count_nonzero(rows[start:] == rows[start]))
        value, success = _dot2_value(entries[start:stop].copy(), vector[columns[start:stop]])
        valid = valid and success
        row_values.append(value)
        row_weights.append(vector[rows[start]])
        start = stop
    reference, outer_success = _dot2_value(np.array(row_weights), np.array(row_values))
    value, success = _dot2_quadratic_form(rows, columns, entries, vector)
    assert success == (valid and outer_success)
    if success:
        assert value == reference
    else:
        assert value == 0.0


def test_quadratic_form_allocates_no_scratch():
    """Nothing in the compiled kernel allocates, whatever the penalty's nonzero count.

    The kernel used to allocate two arrays the length of the nonzeros (``p**2``
    for a dense block) and gather each row's coefficients on every call.
    """
    from superglm.reml._compensated import _dot2_quadratic_form, _nonzero_entries

    penalty, coefficients = _capped_difference_penalty()
    vector = np.ascontiguousarray(coefficients)
    vector.flags.writeable = False
    _dot2_quadratic_form(*_nonzero_entries(penalty), vector)
    # Recompile once so IR inspection is meaningful even after a disk-cache hit.
    _dot2_quadratic_form.recompile()
    for signature in _dot2_quadratic_form.signatures:
        assert "NRT_MemInfo_alloc" not in _dot2_quadratic_form.inspect_llvm(signature)


@pytest.mark.parametrize("kernel_name", ["_dot2_quadratic_form", "_dot2_selected"])
def test_compiled_dot2_callers_keep_ieee_rounding(kernel_name):
    import superglm.reml._compensated as compensated

    kernel = getattr(compensated, kernel_name)
    penalty, coefficients = _capped_difference_penalty()
    if kernel_name == "_dot2_quadratic_form":
        kernel(*_nonzero_entries(penalty), coefficients)
    else:
        kernel(penalty, coefficients[:, None].copy(), np.array([[0, 0], [1, 0]]))
    assert kernel.targetoptions.get("fastmath", False) is False
    # Recompile once so IR inspection is meaningful even after a disk-cache hit.
    kernel.recompile()
    llvm = kernel.inspect_llvm(kernel.signatures[0])
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


def test_solver_penalty_value_uses_the_compensated_form_and_its_range_fallback():
    from superglm.distributional.solver.solver import _half_penalty_quadratic, _penalty_entries
    from superglm.reml._compensated import _dot2_quadratic_form

    penalty, coefficients = _capped_difference_penalty()
    value, success = _dot2_quadratic_form(*_nonzero_entries(penalty), coefficients)
    assert success
    assert _half_penalty_quadratic(penalty, coefficients) == 0.5 * value
    # The solver context builds the entries once; passing them changes nothing.
    entries = _penalty_entries(penalty)
    assert not any(array.flags.writeable for array in entries)
    assert _half_penalty_quadratic(penalty, coefficients, entries) == 0.5 * value
    # A subnormal coefficient is outside Dot2's normal range: the naive form.
    tiny = coefficients.copy()
    tiny[0] = np.nextafter(0.0, 1.0)
    assert not _dot2_quadratic_form(*_nonzero_entries(penalty), tiny)[1]
    assert _half_penalty_quadratic(penalty, tiny) == 0.5 * float(tiny @ penalty @ tiny)


@pytest.mark.parametrize("width", [4, 2])
@pytest.mark.parametrize("with_entries", [False, True])
def test_solver_penalty_value_refuses_coefficients_that_do_not_conform(width, with_entries):
    """A length mismatch raises the plain form's ValueError, not a truncated value.

    The kernel reads coefficients without bounds checks: before this check a
    longer vector returned the form over its first three entries (1.5 here) and
    a shorter one read past the end of the buffer.
    """
    from superglm.distributional.solver.solver import _half_penalty_quadratic, _penalty_entries

    penalty = np.eye(3)
    coefficients = np.ones(width)
    entries = _penalty_entries(penalty) if with_entries else None
    with pytest.raises(ValueError) as plain:
        coefficients @ penalty @ coefficients
    with pytest.raises(ValueError) as caught:
        _half_penalty_quadratic(penalty, coefficients, entries)
    assert str(caught.value) == str(plain.value)
