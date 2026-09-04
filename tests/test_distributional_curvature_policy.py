from __future__ import annotations

import json

import numpy as np
import pytest

from superglm.distributional.curvature import (
    CurvaturePolicyError,
    CurvaturePolicyState,
    RepeatedCurvatureIndefinitenessError,
    resolve_curvature,
)
from superglm.distributional.telemetry import CurvatureTelemetry


@pytest.mark.parametrize("source", ["observed", "fisher", "hybrid"])
def test_each_curvature_request_accepts_a_valid_family_supplied_matrix(source: str) -> None:
    decision = resolve_curvature(
        source,  # ty: ignore[invalid-argument-type]
        np.array([[3.0, 0.2], [0.2, 1.5]]),
    )

    assert decision.retry_required is False
    assert decision.matrix is not None
    assert decision.decomposition is not None
    assert decision.telemetry.requested_source == source
    assert decision.telemetry.actual_source == source
    assert decision.telemetry.reason is None
    assert decision.telemetry.rank == 2
    assert decision.telemetry.fallback_count == 0
    decision.telemetry.assert_no_fallback()


def test_curvature_is_symmetrized_before_diagnostics_and_publication() -> None:
    decision = resolve_curvature("observed", np.array([[2.0, 1.4], [0.6, 2.0]]))

    assert decision.matrix is not None
    np.testing.assert_allclose(decision.matrix, np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert not decision.matrix.flags.writeable
    assert decision.telemetry.minimum_eigenvalue == pytest.approx(1.0)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
    ],
)
def test_nonfinite_curvature_is_a_hard_failure(matrix: np.ndarray) -> None:
    with pytest.raises(CurvaturePolicyError, match="non-finite"):
        resolve_curvature("observed", matrix)


def test_rank_tolerance_accepts_negligible_negative_eigenvalue_without_clipping() -> None:
    curvature = np.diag([-1.0e-15, 2.0])

    decision = resolve_curvature("observed", curvature)

    assert decision.retry_required is False
    assert decision.telemetry.actual_source == "observed"
    assert decision.telemetry.minimum_eigenvalue == pytest.approx(-1.0e-15)
    assert decision.telemetry.rank == 1
    np.testing.assert_array_equal(decision.matrix, curvature)


def test_material_observed_indefiniteness_requires_one_retry_before_fallback() -> None:
    observed = np.array([[1.0, 2.0], [2.0, 1.0]])
    fisher = np.diag([2.0, 8.0])

    first = resolve_curvature("observed", observed, fisher_matrix=fisher)

    assert first.retry_required is True
    assert first.matrix is None
    assert first.state == CurvaturePolicyState(retry_attempted=True, fallback_count=0)
    assert first.telemetry.actual_source == "observed"
    assert first.telemetry.reason == "material_indefiniteness_retry_required"
    assert first.telemetry.minimum_eigenvalue == pytest.approx(-1.0)

    second = resolve_curvature(
        "observed",
        observed,
        fisher_matrix=fisher,
        state=first.state,
    )

    assert second.retry_required is False
    np.testing.assert_array_equal(second.matrix, fisher)
    assert second.telemetry.actual_source == "fisher"
    assert second.telemetry.reason == "material_indefiniteness_after_retry"
    assert second.telemetry.minimum_eigenvalue == pytest.approx(-1.0)
    assert second.telemetry.rank == 2
    # Shared rank policy equilibrates the diagonal scale before conditioning.
    assert second.telemetry.condition_estimate == pytest.approx(1.0)
    assert second.telemetry.fallback_count == 1
    assert second.state == CurvaturePolicyState(retry_attempted=False, fallback_count=1)
    with pytest.raises(RuntimeError, match="fallback"):
        second.telemetry.assert_no_fallback()


def test_tighter_retry_can_accept_observed_curvature_without_fallback() -> None:
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    first = resolve_curvature("observed", indefinite)

    second = resolve_curvature(
        "observed",
        np.array([[2.0, 0.1], [0.1, 1.0]]),
        state=first.state,
    )

    assert second.retry_required is False
    assert second.telemetry.actual_source == "observed"
    assert second.telemetry.reason == "accepted_after_retry"
    assert second.telemetry.fallback_count == 0
    second.telemetry.assert_no_fallback()


@pytest.mark.parametrize("source", ["observed", "hybrid"])
def test_second_non_fisher_material_failure_has_a_narrow_cause(source: str) -> None:
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    state = CurvaturePolicyState(retry_attempted=True)

    with pytest.raises(
        RepeatedCurvatureIndefinitenessError,
        match="Fisher.*required",
    ) as caught:
        resolve_curvature(source, indefinite, state=state)  # ty: ignore[invalid-argument-type]

    assert type(caught.value) is RepeatedCurvatureIndefinitenessError


def test_second_material_failure_requires_valid_fisher_curvature() -> None:
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    state = CurvaturePolicyState(retry_attempted=True)

    with pytest.raises(CurvaturePolicyError, match="Fisher.*materially indefinite"):
        resolve_curvature(
            "observed",
            indefinite,
            state=state,
            fisher_matrix=np.array([[1.0, 3.0], [3.0, 1.0]]),
        )


def test_material_fisher_request_is_a_hard_failure_without_retry() -> None:
    with pytest.raises(CurvaturePolicyError, match="Fisher.*materially indefinite"):
        resolve_curvature("fisher", np.array([[1.0, 3.0], [3.0, 1.0]]))


def test_fallback_count_is_cumulative_across_policy_state() -> None:
    state = CurvaturePolicyState(retry_attempted=True, fallback_count=2)
    decision = resolve_curvature(
        "hybrid",
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        fisher_matrix=np.eye(2),
        state=state,
    )

    assert decision.telemetry.requested_source == "hybrid"
    assert decision.telemetry.actual_source == "fisher"
    assert decision.telemetry.fallback_count == 3
    assert decision.state.fallback_count == 3


def test_telemetry_serializes_exact_required_benchmark_fields() -> None:
    telemetry = CurvatureTelemetry(
        requested_source="observed",
        actual_source="fisher",
        reason="material_indefiniteness_after_retry",
        minimum_eigenvalue=-0.012,
        rank=17,
        condition_estimate=2.1e8,
        fallback_count=1,
    )

    record = telemetry.to_dict()
    assert record == {
        "requested_source": "observed",
        "actual_source": "fisher",
        "reason": "material_indefiniteness_after_retry",
        "minimum_eigenvalue": -0.012,
        "rank": 17,
        "condition_estimate": 2.1e8,
        "fallback_count": 1,
    }
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_eigenvalue": np.nan},
        {"condition_estimate": np.inf},
        {"condition_estimate": -1.0},
        {"rank": -1},
        {"fallback_count": -1},
    ],
)
def test_telemetry_rejects_nonserializable_or_invalid_diagnostics(kwargs: dict) -> None:
    values = {
        "requested_source": "observed",
        "actual_source": "observed",
        "reason": None,
        "minimum_eigenvalue": 0.1,
        "rank": 2,
        "condition_estimate": 3.0,
        "fallback_count": 0,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        CurvatureTelemetry(**values)  # type: ignore[arg-type]


def test_curvature_policy_rejects_bad_source_and_shape() -> None:
    with pytest.raises(ValueError, match="curvature source"):
        resolve_curvature("automatic", np.eye(2))  # ty: ignore[invalid-argument-type]
    with pytest.raises(CurvaturePolicyError, match="square"):
        resolve_curvature("observed", np.ones((2, 3)))
