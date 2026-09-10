"""Finite-face consumers require admitted summaries, without dense inverse outputs."""

import math
from dataclasses import replace

import numpy as np
import pytest

from superglm.distributional.smoothing import endpoint_laml
from superglm.reml import multi_penalty
from superglm.reml.penalty_support import PenaltyNumericalError
from tests.test_distributional_endpoint_laml import (
    _axis_aligned_projected_face,
    _projected_penalty_problem,
)


def test_finite_face_does_not_materialize_an_unused_inverse(monkeypatch):
    layout, face, lambdas, _, _ = _projected_penalty_problem()

    def forbidden_inverse(*_args, **_kwargs):
        raise AssertionError("finite-face consumer requested an unused dense inverse")

    monkeypatch.setattr(multi_penalty, "_inverse_gram_enclosed", forbidden_inverse)
    result = endpoint_laml._projected_finite_penalty_evaluation(
        layout=layout, lambdas=lambdas, face=face
    )
    assert result.rank == 3

    # Positive control: restoring the old full-result boundary must trip the
    # same sentinel, before any changed result interface could mask the cost.
    monkeypatch.setattr(
        multi_penalty, "_evaluate_penalty_summary", multi_penalty._evaluate_penalty_support
    )
    with pytest.raises(AssertionError, match="unused dense inverse"):
        endpoint_laml._projected_finite_penalty_evaluation(
            layout=layout, lambdas=lambdas, face=face
        )


def test_finite_face_preserves_summary_accuracy_refusal(monkeypatch):
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    evidence = PenaltyNumericalError("reference accuracy target was not met")

    def refuse(*_args, **_kwargs):
        raise evidence

    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", refuse)
    with pytest.raises(endpoint_laml.EndpointLaplaceError) as caught:
        endpoint_laml._projected_finite_penalty_evaluation(
            layout=layout, lambdas=lambdas, face=face
        )
    assert caught.value.__cause__ is evidence


@pytest.mark.parametrize("weights", [(1.75, 0.6), (1.75, 0.0), (0.0, 0.6), (0.0, 0.0)])
@pytest.mark.parametrize("rotate", [False, True])
def test_finite_face_preserves_every_consumed_full_result_field(monkeypatch, weights, rotate):
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    names = tuple(component.name for component in layout.penalties[1:])
    lambdas.update(zip(names, weights, strict=True))
    if rotate:
        rotation = np.eye(face.reduced_width)
        rotation[:2, :2] = [[0.6, -0.8], [0.8, 0.6]]
        face = replace(face, null_basis=face.null_basis @ rotation)
    real_summary = multi_penalty._evaluate_penalty_summary
    recorded = []

    def compare_full(support, values):
        summary = real_summary(support, values)
        full = multi_penalty._evaluate_penalty_support(support, values)
        assert summary.rank == full.rank
        assert summary.logdet_s_plus == full.logdet_s_plus
        roots = full._support.component_roots
        np.testing.assert_array_equal(
            summary.gradient, multi_penalty.logdet_s_gradient(full, roots, values)
        )
        np.testing.assert_array_equal(
            summary.hessian, multi_penalty.logdet_s_hessian(full, roots, values)
        )
        for field in (
            "whitening_error",
            "duality_error",
            "logdet_error",
            "gradient_error",
            "hessian_error",
            "resolution_limited",
        ):
            np.testing.assert_array_equal(
                getattr(summary._certificate, field), getattr(full._certificate, field)
            )
        recorded.append(summary)
        return summary

    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", compare_full)
    result = endpoint_laml._projected_finite_penalty_evaluation(
        layout=layout, lambdas=lambdas, face=face
    )
    assert len(recorded) == 1
    summary = recorded[0]
    certificate = summary._certificate
    assert result.rank == summary.rank
    assert result.logdet == summary.logdet_s_plus
    assert result.logdet_error == (
        certificate.logdet_error + np.finfo(float).eps * abs(result.logdet)
    )
    for i, name in enumerate(names):
        assert result.gradient[name] == summary.gradient[i]
        assert result.gradient_error[name] == certificate.gradient_error[i]
        for j, other in enumerate(names):
            assert result.hessian[name, other] == summary.hessian[i, j]
            assert result.hessian_error[name, other] == certificate.hessian_error[i, j]


@pytest.mark.parametrize("weights", [(1.0, 1.0), (2.0, 0.0), (0.0, 3.0)])
def test_finite_face_summary_accepts_finite_logdet_when_unused_inverse_overflows(
    monkeypatch, weights
):
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    selected, left, right = layout.penalties
    scale = math.ldexp(1.0, -1060)
    left_matrix = scale * np.diag([1.0, 1.0, 0.0])
    right_matrix = scale * np.diag([0.0, 0.0, 1.0])
    left = replace(left, omega_raw=left_matrix, omega_ssp=left_matrix)
    right = replace(right, omega_raw=right_matrix, omega_ssp=right_matrix, rank=1)
    layout = replace(layout, penalties=(selected, left, right))
    face, _ = _axis_aligned_projected_face(layout, face, left)
    names = (left.name, right.name)
    lambdas.update(zip(names, weights, strict=True))
    recorded = []
    real_summary = multi_penalty._evaluate_penalty_summary

    def retain_input(support, values):
        recorded.append((support, values.copy()))
        return real_summary(support, values)

    monkeypatch.setattr(multi_penalty, "_evaluate_penalty_summary", retain_input)
    result = endpoint_laml._projected_finite_penalty_evaluation(
        layout=layout, lambdas=lambdas, face=face
    )
    expected_gradient = np.array([2.0 if weights[0] else 0.0, 1.0 if weights[1] else 0.0])
    expected = math.fsum(
        rank * (-1060 * math.log(2.0) + math.log(weight))
        for rank, weight in zip((2, 1), weights, strict=True)
        if weight
    )
    assert result.rank == int(sum(expected_gradient))
    assert abs(result.logdet - expected) <= (
        result.logdet_error + 8 * np.finfo(float).eps * abs(expected)
    )
    for i, name in enumerate(names):
        assert abs(result.gradient[name] - expected_gradient[i]) <= result.gradient_error[name]
        for other in names:
            assert abs(result.hessian[name, other]) <= result.hessian_error[name, other]
    assert len(recorded) == 1
    support, values = recorded[0]
    with pytest.raises(PenaltyNumericalError, match="required dense penalty inverse"):
        multi_penalty._evaluate_penalty_support(support, values)
