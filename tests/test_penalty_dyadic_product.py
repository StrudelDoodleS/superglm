"""Exact dyadic products must fit the unchanged working-precision allowance."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import superglm.reml.multi_penalty as module
from superglm.reml.penalty_support import _penalty_support, _penalty_support_from_roots


def _fraction(value):
    return Fraction(*value.as_integer_ratio())


def _check_product(left, right, value, magnitude):
    allowance = np.nextafter(
        module._LD(module._gamma(2 * left.shape[1] + 1, module._U_LD))
        * magnitude.astype(module._LD),
        module._LD(-np.inf),
    )
    for i, j in np.ndindex(value.shape):
        exact = sum(
            (_fraction(left[i, k]) * _fraction(right[k, j]) for k in range(left.shape[1])),
            Fraction(0),
        )
        assert abs(_fraction(value[i, j]) - exact) <= _fraction(allowance[i, j])


def _wide_format():
    info = np.finfo(module._LD)
    if (
        info.nmant <= np.finfo(float).nmant
        or info.minexp > 2 * np.finfo(float).minexp - 2 * info.nmant - 16
    ):
        pytest.skip("The optional dyadic route needs wider precision and exponent range")


@pytest.mark.parametrize("sign", [-1, 1])
def test_slice_extraction_repairs_rounding_across_integer_boundaries(sign):
    _wide_format()
    integer = np.array([1, 2, 17, 2**21 - 1], dtype=module._LD)
    lower = np.nextafter(integer, module._LD(-np.inf))
    upper = np.nextafter(integer, module._LD(np.inf))
    x = sign * np.concatenate([lower, integer, upper, [module._LD(2.0**-90), module._LD(0)]])
    normalized = x[None, :] * module._LD(2.0**-21)
    slices, residual = module._dyadic_slices(normalized)
    expected_first = np.trunc(x) * module._LD(2.0**-21)
    np.testing.assert_array_equal(slices[0][0].astype(module._LD), expected_first)
    # Omitting the integer-boundary repair changes the represented slice.
    naive = np.trunc(x.astype(float)) * 2.0**-21
    assert np.any(naive.astype(module._LD) != expected_first)
    for j in range(normalized.shape[1]):
        assert sum(
            (_fraction(part[0, j]) for part in slices), _fraction(residual[0, j])
        ) == _fraction(normalized[0, j])
    assert all(part.dtype == np.float64 for part in slices)


@pytest.mark.parametrize("count", [16, 64, 225, 512])
def test_dyadic_witness_encloses_exact_original_products(count):
    _wide_format()
    rng = np.random.default_rng(894)
    left = rng.uniform(-1, 1, size=(2, count)).astype(module._LD)
    right = rng.uniform(-1, 1, size=(count, 3)).astype(module._LD)
    left = np.nextafter(left, module._LD(np.inf))
    right = np.nextafter(right, module._LD(-np.inf))
    left *= np.array([[2.0**300], [2.0**-300]], dtype=module._LD)
    right *= np.array([[2.0**-300, 1.0, 2.0**300]], dtype=module._LD)
    left, right = left[:, ::-1], right[::-1]
    left.setflags(write=False)
    right.setflags(write=False)
    magnitude = module._positive_product(np.abs(left), np.abs(right))
    value = module._dyadic_product(left, right, magnitude)
    assert value is not None
    _check_product(left, right, value, magnitude)


def test_513_term_dot_must_not_use_the_exact_slice_claim():
    _wide_format()
    coefficient = 2**21 - 1
    q = module._LD(coefficient) * module._LD(2.0**-21)
    left, right = np.full((1, 513), q), np.full((513, 1), q)
    magnitude = module._positive_product(left, right)
    assert module._dyadic_product(left, right, magnitude) is None
    # Four terms share the middle diagonal's grid. At k=513 a partial
    # coefficient can cross 2**53; a one-term decrement makes it odd.
    assert 4 * 512 * coefficient**2 < 2**53
    integer = 4 * 513 * coefficient**2 - coefficient
    assert integer > 2**53
    exact = Fraction(integer, 2**105)
    assert exact.numerator % 2 == 1
    distance = abs(Fraction(float(exact)) - exact)
    assert distance == Fraction(1, 2**105)


@pytest.mark.parametrize("sign", [-1, 1])
def test_fourth_slice_retains_a_signed_weak_overlap(sign):
    _wide_format()
    left = np.zeros((1, 64), dtype=module._LD)
    right = np.zeros((64, 1), dtype=module._LD)
    left[0, :3] = [1, 1, sign * module._LD(2.0**-82)]
    right[:3, 0] = [1, -1, 1]
    magnitude = module._positive_product(np.abs(left), np.abs(right))
    value = module._dyadic_product(left, right, magnitude)
    assert value is not None
    assert value[0, 0] == sign * module._LD(2.0**-82)
    _check_product(left, right, value, magnitude)


def test_diagonal_reconstruction_uses_six_wide_additions(monkeypatch):
    _wide_format()
    counts = {"products": 0, "native_additions": 0, "wide_additions": 0}

    class ObservedArray(np.ndarray):
        def __matmul__(self, other):
            counts["products"] += 1
            return (np.asarray(self) @ np.asarray(other)).view(ObservedArray)

        def __iadd__(self, other):
            name = "native_additions" if self.dtype == np.dtype(float) else "wide_additions"
            counts[name] += 1
            np.add(np.asarray(self), np.asarray(other), out=np.asarray(self))
            return self

    original = module._dyadic_slices

    def observed(normalized):
        parts, residual = original(normalized)
        return tuple(part.view(ObservedArray) for part in parts), residual

    monkeypatch.setattr(module, "_dyadic_slices", observed)
    rng = np.random.default_rng(414)
    left = rng.normal(size=(3, 128)).astype(module._LD)
    right = rng.normal(size=(128, 4)).astype(module._LD)
    magnitude = module._positive_product(np.abs(left), np.abs(right))
    assert module._dyadic_product(left, right, magnitude) is not None
    assert counts == {"products": 16, "native_additions": 9, "wide_additions": 6}


def test_omitted_weak_coordinate_forces_wide_fallback():
    _wide_format()
    left, right = np.zeros((1, 64), dtype=module._LD), np.zeros((64, 1), dtype=module._LD)
    left[0, :2], right[1, 0] = [1, module._LD(2.0**-120)], 1
    magnitude = module._positive_product(np.abs(left), np.abs(right))
    assert module._dyadic_product(left, right, magnitude) is None
    value = module._wide_product(left, right, magnitude)
    assert value[0, 0] == module._LD(2.0**-120)
    slices, _ = module._dyadic_slices(left / 2)
    assert all(part[0, 1] == 0 for part in slices)


@pytest.mark.parametrize("case", ["normalization", "scale", "result", "magnitude", "allowance"])
def test_unsupported_dyadic_ranges_keep_the_original_product(monkeypatch, case):
    _wide_format()
    left = np.ones((8, 64), dtype=module._LD)
    right = np.ones((64, 8), dtype=module._LD)
    if case == "normalization":
        left[0, 0] = np.ldexp(module._LD(1), -1100)
    elif case == "scale":
        left *= np.ldexp(module._LD(1), np.finfo(module._LD).maxexp - 2)
        right *= 8
    elif case == "result":
        left *= np.ldexp(module._LD(1), np.finfo(module._LD).minexp // 2 - 60)
        right *= np.ldexp(module._LD(1), np.finfo(module._LD).minexp // 2 - 60)
    magnitude = np.ones((8, 8))
    if case == "magnitude":
        magnitude[0, 0] = np.inf
    if case == "allowance":
        original = module._gamma
        monkeypatch.setattr(
            module, "_gamma", lambda n, unit=None: np.inf if n == 129 else original(n, unit)
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        assert module._dyadic_product(left, right, magnitude) is None


def test_binary64_working_format_keeps_its_native_original_product(monkeypatch):
    left, right = np.eye(64), np.eye(64)
    monkeypatch.setattr(module, "_LD", np.float64)
    monkeypatch.setattr(module, "_U_LD", np.finfo(float).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(0.0, 1.0))
    assert module._dyadic_product(left, right, np.eye(64)) is None
    np.testing.assert_array_equal(module._wide_product(left, right), left @ right)


def test_wider_significand_without_wider_range_uses_original_product(monkeypatch):
    original = np.finfo
    limited = SimpleNamespace(nmant=original(module._LD).nmant, minexp=original(float).minexp)
    monkeypatch.setattr(
        np, "finfo", lambda dtype: limited if dtype is module._LD else original(dtype)
    )
    left, right = np.eye(8, dtype=module._LD), np.eye(8, dtype=module._LD)
    assert module._dyadic_product(left, right, np.eye(8)) is None
    np.testing.assert_array_equal(module._wide_product(left, right), left @ right)


def test_small_product_keeps_original_dispatch(monkeypatch):
    def unexpected(*args):
        pytest.fail("Small products must not pay for dyadic preparation")

    monkeypatch.setattr(module, "_dyadic_product", unexpected)
    left = np.arange(48 * 48, dtype=module._LD).reshape(48, 48)
    right = np.eye(48, dtype=module._LD)
    np.testing.assert_array_equal(module._wide_product(left, right), left @ right)


def test_large_enclosed_product_retains_an_owned_valid_wide_witness(monkeypatch):
    _wide_format()
    rng = np.random.default_rng(386)
    left = rng.normal(size=(128, 128)).astype(module._LD)
    right = rng.normal(size=(128, 128)).astype(module._LD)
    original, calls = module._dyadic_product, []

    def counted(*args):
        value = original(*args)
        calls.append(value is not None)
        return value

    monkeypatch.setattr(module, "_dyadic_product", counted)
    evidence = []
    value, error = module._matmul_enclosed(left, right, _evidence=evidence)
    assert calls == [True]
    witness = evidence[0]
    assert not witness.wide.flags.writeable
    assert not np.shares_memory(value, witness.wide)
    _check_product(left[:2], right[:, :2], witness.wide[:2, :2], witness.magnitude[:2, :2])
    exact = sum((_fraction(left[0, k]) * _fraction(right[k, 0]) for k in range(128)), Fraction(0))
    assert abs(Fraction(float(value[0, 0])) - exact) <= Fraction(float(error[0, 0]))


def test_optional_magnitude_refusal_preserves_finite_wide_signed_product():
    _wide_format()
    left = np.full((128, 128), np.ldexp(module._LD(1), 1020))
    right = np.ones((128, 128), dtype=module._LD)
    right[1::2] = -1
    right[0, 0] += module._LD(2.0**-50)
    expected = np.zeros((128, 128), dtype=module._LD)
    expected[:, 0] = np.ldexp(module._LD(1), 970)
    with pytest.raises(module.PenaltyNumericalError):
        module._positive_product(np.abs(left), np.abs(right))
    np.testing.assert_array_equal(module._wide_product(left, right), expected)
    # The enclosing API still requires a representable absolute-error bound.
    with pytest.raises(module.PenaltyNumericalError):
        module._matmul_enclosed(left, right)


def test_basis_gram_cache_invalidates_changed_dyadic_method(monkeypatch):
    support = _penalty_support([np.diag([1.0, 2.0, 0.0])])
    module._basis_gram(support, support.Q_plus)
    evidence = support._basis_gram_evidence
    original = module._dyadic_product
    monkeypatch.setattr(module, "_dyadic_product", lambda *args: original(*args))
    module._basis_gram(support, support.Q_plus)
    assert support._basis_gram_evidence is not evidence


@pytest.mark.parametrize("separated", [False, True])
def test_penalty_qr_omits_unused_q_and_keeps_square_r(monkeypatch, separated):
    rng = np.random.default_rng(174)
    roots = [rng.normal(size=(8, 5)), rng.normal(size=(6, 5))]
    support = _penalty_support_from_roots(
        roots,
        resolution_limited=[False, False],
        input_error_bounds=[np.zeros_like(root) for root in roots],
    )
    original, modes = module.scipy.linalg.qr, []

    def qr(*args, **kwargs):
        modes.append(kwargs.get("mode"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.scipy.linalg, "qr", qr)
    if separated:
        monkeypatch.setattr(module, "_direct_candidate", lambda *args: None)
    result = module._evaluate_penalty_summary(support, np.array([2.0, 3.0]))
    assert result.rank == 5 and result._correction_count >= 1
    assert modes and all(mode == "r" for mode in modes)
