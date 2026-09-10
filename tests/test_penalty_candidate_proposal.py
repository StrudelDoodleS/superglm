"""Native proposals do not replace the original reference certificates."""

from types import SimpleNamespace

import numpy as np
import pytest

from superglm.reml import multi_penalty as module


def _support(rank=128):
    return SimpleNamespace(
        rank=rank,
        component_roots=(np.eye(rank),),
        Q_plus=np.eye(rank),
        _basis_gram_evidence=object(),
    )


@pytest.mark.parametrize("power", [-129, 129])
def test_native_proposal_declines_outside_its_range(power):
    left = np.eye(8, dtype=np.longdouble) * np.longdouble(2) ** power
    assert module._candidate_product(left, np.eye(8)) is None


def test_native_proposal_uses_binary64_and_returns_finite_wide_storage(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a proposal does not require a certified wide product")

    monkeypatch.setattr(module, "_wide_product", forbidden)
    left = np.array([[1, 2, -3], [-2, 1, 4]], dtype=np.longdouble)
    right = np.array([[2, -1], [3, 2], [1, -4]], dtype=np.longdouble)
    product = module._candidate_product(left, right)
    np.testing.assert_array_equal(product, [[5, 15], [3, -12]])
    assert product.dtype == np.dtype(np.longdouble)


def test_summary_retries_whole_geometry_after_late_native_refusal(monkeypatch):
    support = _support()
    original = support._basis_gram_evidence
    accepted = object()
    calls = []

    def evaluate(owner, values, eps_rank, *, summary_only, _native_candidate=False):
        assert owner is support and summary_only
        assert owner._basis_gram_evidence is original
        calls.append(_native_candidate)
        if _native_candidate:
            owner._basis_gram_evidence = object()
            raise module.PenaltyNumericalError("late certificate refusal")
        return accepted

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    assert module._evaluate_penalty_summary(support, [1.0]) is accepted
    assert calls == [True, False]


def test_summary_success_keeps_one_complete_native_attempt(monkeypatch):
    calls = []
    accepted = object()

    def evaluate(*args, summary_only, _native_candidate=False):
        calls.append(_native_candidate)
        return accepted

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    assert module._evaluate_penalty_summary(_support(), [1.0]) is accepted
    assert calls == [True]


@pytest.mark.parametrize("rank,weights", [(8, [1.0]), (128, [0.0])])
def test_small_and_zero_face_summaries_keep_original_route(monkeypatch, rank, weights):
    calls = []

    def evaluate(*args, summary_only, _native_candidate=False):
        calls.append(_native_candidate)

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    module._evaluate_penalty_summary(_support(rank), weights)
    assert calls == [False]


def test_full_geometry_never_requests_native_proposal(monkeypatch):
    calls = []

    def evaluate(*args, summary_only, _native_candidate=False):
        assert not summary_only
        calls.append(_native_candidate)

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    module._evaluate_penalty_support(_support(), [1.0])
    assert calls == [False]


def test_retry_preserves_original_failure_reason(monkeypatch):
    def evaluate(*args, summary_only, _native_candidate=False):
        reason = "proposal refused" if _native_candidate else "original geometry refused"
        raise module.PenaltyNumericalError(reason)

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    with pytest.raises(module.PenaltyNumericalError, match="original geometry refused"):
        module._evaluate_penalty_summary(_support(), [1.0])


def test_contract_error_does_not_trigger_numerical_retry(monkeypatch):
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        raise ValueError("invalid contract")

    monkeypatch.setattr(module, "_evaluate_penalty_geometry", evaluate)
    with pytest.raises(ValueError, match="invalid contract"):
        module._evaluate_penalty_summary(_support(), [1.0])
    assert calls == [1]


@pytest.mark.parametrize("proposal", ["different", "singular"])
def test_original_reference_certifies_or_replaces_a_poor_proposal(monkeypatch, proposal):
    from superglm.reml.penalty_support import _penalty_support_from_roots

    rank = 128
    diagonal = np.tile([0.5, 1.0, 2.0, 4.0], rank // 4)
    roots = [np.eye(rank), np.diag(diagonal)]
    support = _penalty_support_from_roots(
        roots,
        resolution_limited=(False, False),
        input_error_bounds=tuple(np.zeros_like(root) for root in roots),
    )
    weights = np.array([3.0, 7.0])
    cold = module._evaluate_penalty_geometry(support, weights, None, summary_only=True)
    calls = []

    def poor(left, right):
        calls.append(1)
        result = np.zeros((left.shape[0], right.shape[1]), dtype=np.longdouble)
        if proposal == "different":
            result[:rank] = np.eye(rank)
        return result

    monkeypatch.setattr(module, "_candidate_product", poor)
    result = module._evaluate_penalty_summary(support, weights)
    assert calls == [1]
    assert result.rank == cold.rank == rank
    assert result._support is cold._support is support
    for name, error in [("gradient", "gradient_error"), ("hessian", "hessian_error")]:
        allowance = getattr(result._certificate, error) + getattr(cold._certificate, error)
        assert np.all(np.abs(getattr(result, name) - getattr(cold, name)) <= allowance)
    allowance = result._certificate.logdet_error + cold._certificate.logdet_error
    assert abs(result.logdet_s_plus - cold.logdet_s_plus) <= allowance
    if proposal == "singular":
        assert result.logdet_s_plus == cold.logdet_s_plus
        np.testing.assert_array_equal(result.gradient, cold.gradient)
        np.testing.assert_array_equal(result.hessian, cold.hessian)
