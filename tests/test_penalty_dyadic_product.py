"""Portable products retain enclosures, weak directions and owned evidence."""

from fractions import Fraction

import numpy as np
import pytest

import superglm.reml.multi_penalty as module
from superglm.reml.penalty_support import _penalty_support, _penalty_support_from_roots


def _fraction(value):
    return Fraction(*value.as_integer_ratio())


@pytest.mark.parametrize("count", [16, 64, 225, 512, 513])
def test_binary64_products_enclose_exact_original_products(count):
    rng = np.random.default_rng(894)
    left = rng.uniform(-1, 1, size=(2, count))
    right = rng.uniform(-1, 1, size=(count, 3))
    left *= np.array([[2.0**300], [2.0**-300]])
    right *= np.array([[2.0**-300, 1.0, 2.0**300]])
    left, right = left[:, ::-1], right[::-1]
    left.setflags(write=False)
    right.setflags(write=False)
    value, error = module._matmul_enclosed(left, right)
    assert value.dtype == error.dtype == np.float64
    for i, j in np.ndindex(value.shape):
        exact = sum(
            (_fraction(left[i, k]) * _fraction(right[k, j]) for k in range(count)),
            Fraction(0),
        )
        assert abs(_fraction(value[i, j]) - exact) <= _fraction(error[i, j])


@pytest.mark.parametrize("sign", [-1, 1])
def test_compensated_scalar_reduction_retains_a_signed_weak_overlap(sign):
    left, right = np.zeros(64), np.zeros(64)
    left[:3], right[:3] = [1, 1, sign * 2.0**-82], [1, -1, 1]
    value, error = module._compensated_dot(left, right)
    exact = sign * Fraction(1, 2**82)
    assert abs(_fraction(value) - exact) <= _fraction(error)
    assert error < abs(exact)  # The enclosure certifies a nonzero weak action.


def test_product_does_not_drop_a_representable_weak_coordinate():
    left, right = np.zeros((1, 64)), np.zeros((64, 1))
    left[0, :2], right[1, 0] = [1, 2.0**-120], 1
    value, error = module._matmul_enclosed(left, right)
    assert value[0, 0] == 2.0**-120
    assert error[0, 0] < value[0, 0]


def test_large_enclosed_product_retains_an_owned_witness():
    rng = np.random.default_rng(386)
    left = rng.normal(size=(128, 128))
    right = rng.normal(size=(128, 128))
    evidence = []
    value, error = module._matmul_enclosed(left, right, _evidence=evidence)
    witness = evidence[0]
    assert witness.product.dtype == np.float64
    assert not witness.product.flags.writeable
    assert not np.shares_memory(value, witness.product)
    exact = sum((_fraction(left[0, k]) * _fraction(right[k, 0]) for k in range(128)), Fraction(0))
    assert abs(_fraction(value[0, 0]) - exact) <= _fraction(error[0, 0])


def test_absolute_magnitude_overflow_is_an_honest_certificate_refusal():
    left = np.full((128, 128), np.ldexp(1.0, 1020))
    right = np.ones((128, 128))
    right[1::2] = -1
    # The signed result can be finite while the required absolute-error
    # enclosure exceeds the public binary64 range.
    with pytest.raises(module.PenaltyNumericalError):
        module._matmul_enclosed(left, right)


def test_basis_gram_cache_invalidates_changed_product_method(monkeypatch):
    support = _penalty_support([np.diag([1.0, 2.0, 0.0])])
    module._basis_gram(support, support.Q_plus)
    evidence = support._basis_gram_evidence
    original = module._native_product
    monkeypatch.setattr(module, "_native_product", lambda *args: original(*args))
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
