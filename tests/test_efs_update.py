from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from superglm.reml.efs_update import (
    EFSComponentState,
    EFSUpdateResult,
    wood_fasiolo_update,
)
from superglm.types import LambdaPolicy


def _component(
    name: str,
    coefficient_slice: slice,
    penalty: np.ndarray,
    *,
    rank: float,
    lambda_value: float = 1.0,
    policy: LambdaPolicy | None = None,
) -> EFSComponentState:
    return EFSComponentState(
        name=name,
        coefficient_slice=coefficient_slice,
        penalty=penalty,
        rank=rank,
        lambda_value=lambda_value,
        policy=LambdaPolicy.estimate() if policy is None else policy,
    )


def test_generalized_update_matches_direct_eigenspace_calculation() -> None:
    penalty = np.array([[1.0, -1.0], [-1.0, 1.0]])
    beta = np.array([1.0, -1.0])
    inverse = np.array([[0.6, 0.1], [0.1, 0.4]])
    component = _component("smooth#wiggle", slice(0, 2), penalty, rank=1.0)

    update = wood_fasiolo_update((component,), beta, inverse)

    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    positive = eigenvalues > np.finfo(float).eps * eigenvalues[-1]
    coordinates = eigenvectors[:, positive].T @ beta
    expected_quad = float(np.sum(eigenvalues[positive] * coordinates**2))
    expected_trace = float(
        np.trace(
            inverse
            @ eigenvectors[:, positive]
            @ np.diag(eigenvalues[positive])
            @ eigenvectors[:, positive].T
        )
    )
    expected_lambda = (
        float(np.count_nonzero(positive)) - component.lambda_value * expected_trace
    ) / expected_quad

    assert update.quadratic_forms[component.name] == pytest.approx(expected_quad)
    assert update.trace_terms[component.name] == pytest.approx(expected_trace)
    assert update.lambdas[component.name] == pytest.approx(expected_lambda)
    assert update.log_steps[component.name] == pytest.approx(np.log(expected_lambda))


def test_update_uses_residual_edf_generalized_fellner_schall_step() -> None:
    component = _component(
        "near-saturated",
        slice(0, 1),
        np.ones((1, 1)),
        rank=1.0,
        lambda_value=100.0,
    )

    update = wood_fasiolo_update(
        (component,),
        np.array([np.sqrt(5.0e-5)]),
        np.array([[0.0099]]),
        max_log_step=None,
    )

    # Residual EDF is 1 - 100 * 0.0099 = 0.01. Dividing by q = 5e-5
    # gives lambda_new = 200. The old denominator rearrangement gives
    # 1 / (5e-5 + 0.0099) and therefore cannot satisfy this assertion.
    assert update.quadratic_forms[component.name] == pytest.approx(5.0e-5)
    assert update.trace_terms[component.name] == pytest.approx(0.0099)
    assert update.lambdas[component.name] == pytest.approx(200.0)
    assert update.raw_log_steps[component.name] == pytest.approx(np.log(2.0))
    assert update.proposal_kinds[component.name] == "gfs"
    assert update.stationarity_log_residuals[component.name] == pytest.approx(
        np.log(1.0 / (100.0 * (5.0e-5 + 0.0099)))
    )


def test_multiple_and_null_selection_components_remain_independent() -> None:
    components = (
        _component(
            "scale:x#null",
            slice(0, 2),
            np.diag([1.0, 0.0]),
            rank=1.0,
            lambda_value=0.5,
        ),
        _component(
            "scale:x#wiggle",
            slice(0, 2),
            np.diag([0.0, 2.0]),
            rank=1.0,
            lambda_value=2.0,
        ),
        _component(
            "location:z#wiggle",
            slice(2, 4),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
            rank=1.0,
            lambda_value=1.5,
        ),
    )
    beta = np.array([1.0, 2.0, -0.5, 0.5])
    inverse = np.diag([0.2, 0.2, 0.2, 0.3])

    update = wood_fasiolo_update(components, beta, inverse, inverse_scale=0.5)

    assert tuple(update.lambdas) == tuple(component.name for component in components)
    assert tuple(update.raw_log_steps) == tuple(component.name for component in components)
    assert update.quadratic_forms == pytest.approx(
        {
            "scale:x#null": 1.0,
            "scale:x#wiggle": 8.0,
            "location:z#wiggle": 1.0,
        }
    )
    assert update.trace_terms == pytest.approx(
        {
            "scale:x#null": 0.2,
            "scale:x#wiggle": 0.4,
            "location:z#wiggle": 0.5,
        }
    )
    assert update.lambdas == pytest.approx(
        {
            "scale:x#null": 1.8,
            "scale:x#wiggle": 0.05,
            "location:z#wiggle": 0.5,
        }
    )


def test_fixed_and_off_components_hold_while_invalid_gfs_uses_safe_fixed_point() -> None:
    components = (
        _component(
            "fixed",
            slice(0, 1),
            np.ones((1, 1)),
            rank=1.0,
            lambda_value=3.0,
            policy=LambdaPolicy.fixed(3.0),
        ),
        _component(
            "off",
            slice(1, 2),
            np.ones((1, 1)),
            rank=1.0,
            lambda_value=0.0,
            policy=LambdaPolicy.off(),
        ),
        _component(
            "zeroed",
            slice(2, 3),
            np.ones((1, 1)),
            rank=1.0,
            lambda_value=4.0,
        ),
    )

    update = wood_fasiolo_update(
        components,
        np.array([2.0, 3.0, 0.0]),
        np.eye(3),
    )

    assert update.lambdas == {"fixed": 3.0, "off": 0.0, "zeroed": 1.0}
    assert update.log_steps == {"fixed": 0.0, "off": 0.0, "zeroed": -np.log(4.0)}
    assert update.raw_log_steps == {"fixed": 0.0, "off": 0.0, "zeroed": -np.log(4.0)}
    assert update.stationarity_log_residuals == {
        "fixed": 0.0,
        "off": 0.0,
        "zeroed": -np.log(4.0),
    }
    assert update.proposal_kinds == {
        "fixed": "inactive",
        "off": "inactive",
        "zeroed": "fixed_point_fallback",
    }
    assert update.quadratic_forms == {"fixed": 4.0, "off": 9.0, "zeroed": 0.0}
    assert update.trace_terms == {"fixed": 1.0, "off": 1.0, "zeroed": 1.0}


def test_zero_quadratic_with_positive_residual_edf_moves_toward_working_infinity() -> None:
    component = _component(
        "null-space",
        slice(0, 1),
        np.ones((1, 1)),
        rank=1.0,
        lambda_value=1.0,
    )

    update = wood_fasiolo_update(
        (component,),
        np.zeros(1),
        np.array([[0.5]]),
        max_log_step=None,
        maximum_lambda=100.0,
    )

    assert update.lambdas[component.name] == 100.0
    assert update.log_steps[component.name] == pytest.approx(np.log(100.0))
    assert update.raw_log_steps[component.name] > update.log_steps[component.name]
    assert update.stationarity_log_residuals[component.name] > 0.0
    assert update.proposal_kinds[component.name] == "working_infinity"


def test_roundoff_negative_quadratic_is_clamped_but_material_indefiniteness_is_rejected() -> None:
    component = _component(
        "roundoff-null",
        slice(0, 1),
        np.array([[-np.finfo(np.float64).eps]]),
        rank=1.0,
    )

    update = wood_fasiolo_update(
        (component,),
        np.ones(1),
        np.zeros((1, 1)),
        max_log_step=None,
        maximum_lambda=100.0,
    )

    assert update.quadratic_forms[component.name] == 0.0
    assert update.proposal_kinds[component.name] == "working_infinity"

    corrupted = np.array([[-0.1]])
    corrupted.setflags(write=False)
    object.__setattr__(component, "penalty", corrupted)
    with pytest.raises(ValueError, match="quadratic form"):
        wood_fasiolo_update((component,), np.ones(1), np.zeros((1, 1)))


def test_working_infinity_remains_outward_at_the_largest_finite_lambda() -> None:
    largest = np.finfo(np.float64).max
    component = _component(
        "largest-finite",
        slice(0, 1),
        np.ones((1, 1)),
        rank=1.0,
        lambda_value=largest,
    )

    update = wood_fasiolo_update(
        (component,),
        np.zeros(1),
        np.zeros((1, 1)),
        max_log_step=None,
        maximum_lambda=largest,
    )

    assert update.proposal_kinds[component.name] == "working_infinity"
    assert update.raw_log_steps[component.name] > 0.0
    assert update.lambdas[component.name] == largest
    assert update.log_steps[component.name] == 0.0


def test_upper_bound_preserves_positive_raw_log_step() -> None:
    component = _component(
        "at-cap",
        slice(0, 1),
        np.ones((1, 1)),
        rank=1.0,
        lambda_value=10.0,
    )

    update = wood_fasiolo_update(
        (component,),
        np.array([0.1]),
        np.array([[0.0]]),
        max_log_step=2.0,
        maximum_lambda=10.0,
    )

    assert update.lambdas["at-cap"] == 10.0
    assert update.log_steps["at-cap"] == 0.0
    assert update.raw_log_steps["at-cap"] == pytest.approx(np.log(100.0 / 10.0))


@pytest.mark.parametrize(
    ("beta", "inverse", "expected_step"),
    [
        (np.array([1.0e-5]), np.array([[1.0e-10]]), 2.0),
        (np.array([1.0e3]), np.array([[0.1]]), -2.0),
    ],
)
def test_log_lambda_steps_are_bounded(
    beta: np.ndarray,
    inverse: np.ndarray,
    expected_step: float,
) -> None:
    component = _component("bounded", slice(0, 1), np.ones((1, 1)), rank=1.0)

    update = wood_fasiolo_update(
        (component,),
        beta,
        inverse,
        max_log_step=2.0,
    )

    assert update.log_steps["bounded"] == pytest.approx(expected_step)
    assert update.lambdas["bounded"] == pytest.approx(np.exp(expected_step))
    unbounded_expected = np.log((1.0 - inverse[0, 0]) / beta[0] ** 2)
    assert update.raw_log_steps["bounded"] == pytest.approx(unbounded_expected)


def test_lower_bound_preserves_negative_raw_log_step() -> None:
    component = _component("at-floor", slice(0, 1), np.ones((1, 1)), rank=1.0)

    update = wood_fasiolo_update(
        (component,),
        np.array([1.0e8]),
        np.array([[0.0]]),
        max_log_step=None,
        minimum_lambda=0.1,
    )

    assert update.lambdas["at-floor"] == 0.1
    assert update.log_steps["at-floor"] == pytest.approx(np.log(0.1))
    assert update.raw_log_steps["at-floor"] == pytest.approx(np.log(1.0e-16))


def test_unbounded_step_caps_huge_finite_raw_lambda_without_overflow() -> None:
    component = _component(
        "huge-raw",
        slice(0, 1),
        np.array([[1.0e-308]]),
        rank=1.0,
        lambda_value=1.0e-6,
    )

    update = wood_fasiolo_update(
        (component,),
        np.array([1.0]),
        np.array([[0.0]]),
        max_log_step=None,
        maximum_lambda=1.0e10,
    )

    assert update.raw_log_steps["huge-raw"] == pytest.approx(np.log(1.0e308) - np.log(1.0e-6))
    assert update.lambdas["huge-raw"] == 1.0e10
    assert update.log_steps["huge-raw"] == pytest.approx(np.log(1.0e10 / 1.0e-6))


def test_uncapped_bounded_step_preserves_huge_raw_lambda_exactly() -> None:
    component = _component(
        "huge-fixed-point",
        slice(0, 1),
        np.array([[1.0e-308]]),
        rank=1.0,
        lambda_value=1.0e308,
    )

    update = wood_fasiolo_update(
        (component,),
        np.array([1.0]),
        np.array([[0.0]]),
        max_log_step=2.0,
        maximum_lambda=np.finfo(float).max,
    )

    assert update.raw_log_steps["huge-fixed-point"] == 0.0
    assert update.lambdas["huge-fixed-point"] == 1.0e308
    assert update.log_steps["huge-fixed-point"] == 0.0


def test_step_capped_lower_bound_is_returned_exactly() -> None:
    component = _component("step-capped-floor", slice(0, 1), np.ones((1, 1)), rank=1.0)

    update = wood_fasiolo_update(
        (component,),
        np.array([1.0e8]),
        np.array([[0.0]]),
        max_log_step=5.0,
        minimum_lambda=0.1,
    )

    assert update.lambdas["step-capped-floor"] == 0.1
    assert update.log_steps["step-capped-floor"] == pytest.approx(np.log(0.1))
    assert update.raw_log_steps["step-capped-floor"] == pytest.approx(np.log(1.0e-16))


def test_update_is_deterministic_defensive_and_matches_gfs_scalar_arithmetic() -> None:
    component = _component(
        "legacy",
        slice(0, 2),
        np.array([[2.0, 0.25], [0.25, 1.0]]),
        rank=2.0,
        lambda_value=0.7,
    )
    beta = np.array([0.4, -0.8])
    inverse = np.array([[0.5, 0.1], [0.1, 0.3]])
    penalty_before = component.penalty.copy()

    first = wood_fasiolo_update((component,), beta, inverse, inverse_scale=0.8)
    second = wood_fasiolo_update((component,), beta, inverse, inverse_scale=0.8)
    quad = float(beta @ penalty_before @ beta)
    trace = float(np.trace(inverse @ penalty_before))
    gfs_raw = (component.rank - component.lambda_value * trace) / (0.8 * quad)
    gfs_step = float(
        np.clip(
            np.log(max(gfs_raw, 1.0e-10)) - np.log(max(component.lambda_value, 1.0e-10)),
            -5.0,
            5.0,
        )
    )
    gfs_lambda = float(np.clip(component.lambda_value * np.exp(gfs_step), 1.0e-6, 1.0e10))

    assert first == second
    assert first.lambdas["legacy"] == gfs_lambda
    np.testing.assert_array_equal(component.penalty, penalty_before)
    assert not component.penalty.flags.writeable
    with pytest.raises(TypeError):
        first.lambdas["legacy"] = 9.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.lambdas = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    "component",
    [
        EFSComponentState(
            "valid",
            slice(0, 1),
            np.ones((1, 1)),
            1.0,
            1.0,
            LambdaPolicy.estimate(),
        ),
    ],
)
def test_update_result_has_the_declared_public_shape(component: EFSComponentState) -> None:
    result = wood_fasiolo_update((component,), np.ones(1), np.eye(1))

    assert isinstance(result, EFSUpdateResult)
    assert set(result.lambdas) == set(result.log_steps)
    assert set(result.lambdas) == set(result.raw_log_steps)
    assert set(result.lambdas) == set(result.stationarity_log_residuals)
    assert set(result.lambdas) == set(result.proposal_kinds)
    assert set(result.lambdas) == set(result.quadratic_forms)
    assert set(result.lambdas) == set(result.trace_terms)
    with pytest.raises(TypeError):
        result.raw_log_steps["valid"] = 1.0  # type: ignore[index]


def test_component_validation_rejects_mismatched_or_non_psd_penalties() -> None:
    with pytest.raises(ValueError, match="shape"):
        _component("bad-shape", slice(0, 2), np.eye(1), rank=1.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _component("indefinite", slice(0, 2), np.diag([1.0, -0.1]), rank=1.0)
    with pytest.raises(ValueError, match="fixed policy value"):
        _component(
            "fixed-mismatch",
            slice(0, 1),
            np.eye(1),
            rank=1.0,
            lambda_value=2.0,
            policy=LambdaPolicy.fixed(3.0),
        )
