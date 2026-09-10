"""Exact-input reuse keeps the selected geometry and its arithmetic evidence."""

import copy
import pickle
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction

import numpy as np
import pytest

import superglm.reml.multi_penalty as module
from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support_from_roots


def _support():
    roots = [np.array([[1.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 1.0]])]
    return _penalty_support_from_roots(
        roots,
        resolution_limited=[False, False],
        input_error_bounds=[np.zeros_like(root) for root in roots],
    )


def _equal(left, right):
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            if field.name != "_basis_gram_evidence":
                _equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, tuple | list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def test_basis_gram_is_computed_once_for_one_immutable_support(monkeypatch):
    support = _support()
    calls = []
    original = module._matmul_enclosed

    def product(left, right, **kwargs):
        calls.append((left.shape, right.shape))
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "_matmul_enclosed", product)
    first = module._basis_gram(support, support.Q_plus)
    second = module._basis_gram(support, support.Q_plus)
    assert len(calls) == 1
    _equal(first, second)
    assert not first[0].flags.writeable
    assert not first[1].flags.writeable


@pytest.mark.parametrize("change", ["replace", "copy", "basis_copy", "basis_mutation"])
def test_basis_gram_rejects_changed_support_or_basis(monkeypatch, change):
    support = _support()
    module._basis_gram(support, support.Q_plus)
    original = module._matmul_enclosed
    calls = []

    def product(left, right, **kwargs):
        calls.append(1)
        return original(left, right, **kwargs)

    # Prime under the callable identity used by the actual invalidation test.
    monkeypatch.setattr(module, "_matmul_enclosed", product)
    module._basis_gram(support, support.Q_plus)
    calls.clear()
    if change == "replace":
        support = replace(support)
    elif change == "copy":
        support = copy.copy(support)
    elif change == "basis_copy":
        object.__setattr__(support, "Q_plus", support.Q_plus.copy())
    else:
        support.Q_plus.setflags(write=True)
        support.Q_plus[0, 0] += 0.125
        support.Q_plus.setflags(write=False)
    actual = module._basis_gram(support, support.Q_plus)
    assert len(calls) == 1
    _equal(actual, original(support.Q_plus.T, support.Q_plus))


def test_basis_gram_rejects_changed_working_precision(monkeypatch):
    support = _support()
    original = module._matmul_enclosed
    calls = []

    def product(left, right, **kwargs):
        calls.append(1)
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "_matmul_enclosed", product)
    module._basis_gram(support, support.Q_plus)
    monkeypatch.setattr(module, "_LD", np.float64)
    monkeypatch.setattr(module, "_U_LD", np.finfo(float).eps / 2)
    monkeypatch.setattr(module, "_TINY_LD", np.nextafter(0.0, 1.0))
    actual = module._basis_gram(support, support.Q_plus)
    assert len(calls) == 2
    _equal(actual, original(support.Q_plus.T, support.Q_plus))


def test_basis_gram_refusal_is_not_hidden_by_callable_cache(monkeypatch):
    support = _support()
    module._basis_gram(support, support.Q_plus)

    def refused(*args, **kwargs):
        raise PenaltyNumericalError("mutated product refusal")

    monkeypatch.setattr(module, "_matmul_enclosed", refused)
    with pytest.raises(PenaltyNumericalError, match="mutated product refusal"):
        module._basis_gram(support, support.Q_plus)


def test_support_pickle_drops_ephemeral_basis_evidence(monkeypatch):
    support = _support()
    original = module._matmul_enclosed

    # A retained local callable would make this actual pickle operation fail.
    def local_product(left, right, **kwargs):
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "_matmul_enclosed", local_product)
    module._basis_gram(support, support.Q_plus)
    restored = pickle.loads(pickle.dumps(support))
    assert restored._basis_gram_evidence is None
    _equal(restored, support)
    _equal(
        module._basis_gram(restored, restored.Q_plus),
        original(restored.Q_plus.T, restored.Q_plus),
    )


@pytest.mark.parametrize("repeat", [1, 7, 210])
def test_identical_tiny_error_rows_reuse_one_existing_wide_dot(monkeypatch, repeat):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("The row reuse deliberately excludes binary64 working precision")
    n = 9
    row = np.arange(1, n + 1, dtype=np.longdouble)[None, :] * np.longdouble(np.nextafter(0.0, 1.0))
    left = np.repeat(row, repeat, axis=0)
    right = np.arange(1, n * 3 + 1, dtype=np.longdouble).reshape(n, 3) * np.longdouble(2.0**400)
    expected = module._positive_product(left, right)
    calls = []
    original = module._positive_product

    def product(a, b):
        calls.append((a.shape, b.shape))
        return original(a, b)

    monkeypatch.setattr(module, "_positive_product", product)
    actual = module._root_error_product(left, right)
    np.testing.assert_array_equal(actual, expected)
    assert calls == [((1, n), right.shape)]
    for column in range(right.shape[1]):
        exact = sum(
            (
                Fraction(*row[0, k].as_integer_ratio())
                * Fraction(*right[k, column].as_integer_ratio())
                for k in range(n)
            ),
            Fraction(0),
        )
        assert Fraction(float(actual[0, column])) >= exact
    if repeat > 1:
        actual[0, 0] = 0.0
        assert actual[1, 0] == expected[1, 0]


@pytest.mark.parametrize("control", ["nonidentical", "native", "binary64", "nonfinite"])
def test_root_error_row_reuse_keeps_unsupported_inputs_on_original_path(monkeypatch, control):
    tiny = np.longdouble(np.nextafter(0.0, 1.0))
    left = np.full((3, 2), tiny, dtype=np.longdouble)
    right = np.full((2, 2), np.longdouble(2.0**400))
    if control == "nonidentical":
        left[1, 0] *= 2
    elif control == "native":
        left[:] = 1
        right[:] = 1
    elif control == "binary64":
        monkeypatch.setattr(module, "_LD", np.float64)
    else:
        left[1, 0] = np.inf
    calls = []

    def product(a, b):
        calls.append((a.shape, b.shape))
        return np.full((a.shape[0], b.shape[1]), 2.0)

    monkeypatch.setattr(module, "_positive_product", product)
    module._root_error_product(left, right)
    assert calls == [(left.shape, right.shape)]


@pytest.mark.parametrize("summary", [False, True])
def test_reuse_preserves_complete_evaluation_fields_across_weights(monkeypatch, summary):
    support = _support()
    evaluate = module._evaluate_penalty_summary if summary else module._evaluate_penalty_support
    weights = [np.array([1.0, 1.0]), np.array([1e12, 3.0]), np.array([0.0, 2.0])]
    actual = [evaluate(support, values) for values in weights]
    with monkeypatch.context() as context:
        context.setattr(
            module, "_basis_gram", lambda owner, basis: module._matmul_enclosed(basis.T, basis)
        )
        context.setattr(module, "_root_error_product", module._positive_product)
        expected = [evaluate(support, values) for values in weights]
    for a, b in zip(actual, expected, strict=True):
        _equal(a, b)


def test_existing_evaluator_stops_recomputing_the_same_basis_gram(monkeypatch):
    support = _support()
    original = module._matmul_enclosed
    calls = []

    def product(left, right, **kwargs):
        if right is support.Q_plus:
            calls.append((left.shape, right.shape))
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "_matmul_enclosed", product)
    for value in (1.0, 2.0, 8.0):
        module._evaluate_penalty_summary(support, np.array([value, 3.0]))
    assert calls == [((2, 3), (3, 2))]


def test_existing_evaluator_stops_repeating_identical_input_error_dots(monkeypatch):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("The row reuse deliberately excludes binary64 working precision")
    roots = [np.eye(3, 4), np.eye(3, 4)]
    support = _penalty_support_from_roots(
        roots,
        resolution_limited=[False, False],
        input_error_bounds=[np.zeros_like(root) for root in roots],
    )
    scale = 2 * np.longdouble(np.nextafter(0.0, 1.0))
    calls = []
    original = module._positive_product

    def product(left, right):
        if left.shape[1] == 4 and np.all(left == scale):
            calls.append((left.shape, right.shape))
        return original(left, right)

    monkeypatch.setattr(module, "_positive_product", product)
    module._evaluate_penalty_summary(support, np.array([4.0, 4.0]))
    assert calls == [((1, 4), (4, 3)), ((1, 4), (4, 3))]
