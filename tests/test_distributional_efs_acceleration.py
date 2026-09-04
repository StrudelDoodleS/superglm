from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.efs_acceleration import (
    WindowedTypeIIAnderson,
    _common_scaled_step,
    _model_reduction_bound,
    _nonnegative_product,
    _NumericalProposalError,
    _truncated_svd_solution,
)


class _AmbiguousProvenance:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> np.ndarray:
        del other
        return np.array([True, False])


def test_type_ii_sign_reaches_the_scalar_linear_fixed_point() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([1.0]),
        raw_residual=np.array([-0.5]),
        provenance=("same",),
    )
    accelerator.record_accepted(
        log_lambdas=np.array([0.5]),
        raw_residual=np.array([-0.25]),
        provenance=("same",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.refusal_reason is None
    assert decision.proposal is not None
    np.testing.assert_allclose(decision.proposal.log_lambdas, np.array([0.0]), atol=1e-15)


def test_two_secants_reach_a_coupled_linear_fixed_point() -> None:
    accelerator = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    linear_map = np.array([[0.4, 0.2], [0.0, 0.7]])
    x0 = np.array([1.0, 1.0])
    x1 = linear_map @ x0
    x2 = linear_map @ x1
    for point in (x0, x1, x2):
        accelerator.record_accepted(
            log_lambdas=point,
            raw_residual=linear_map @ point - point,
            provenance=("coupled",),
        )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.refusal_reason is None
    assert decision.proposal is not None
    np.testing.assert_allclose(decision.proposal.log_lambdas, np.zeros(2), atol=2e-14)
    assert decision.proposal.secant_depth == 2


def test_coupled_proposal_is_equivariant_to_coordinate_permutation() -> None:
    linear_map = np.array([[0.4, 0.2], [0.0, 0.7]])
    permutation = np.array([1, 0])
    permuted_map = linear_map[np.ix_(permutation, permutation)]
    original = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    permuted = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    point = np.array([1.0, 1.0])
    for _ in range(3):
        original.record_accepted(
            log_lambdas=point,
            raw_residual=linear_map @ point - point,
            provenance=("permutation",),
        )
        permuted_point = point[permutation]
        permuted.record_accepted(
            log_lambdas=permuted_point,
            raw_residual=permuted_map @ permuted_point - permuted_point,
            provenance=("permutation",),
        )
        point = linear_map @ point

    original_decision = original.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )
    permuted_decision = permuted.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert original_decision.proposal is not None
    assert permuted_decision.proposal is not None
    np.testing.assert_allclose(
        permuted_decision.proposal.log_lambdas[permutation],
        original_decision.proposal.log_lambdas,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        permuted_decision.proposal.log_step[permutation],
        original_decision.proposal.log_step,
        rtol=0.0,
        atol=2e-14,
    )
    assert (
        permuted_decision.proposal.raw_residual_norm == original_decision.proposal.raw_residual_norm
    )
    assert np.isclose(
        permuted_decision.proposal.model_residual_norm,
        original_decision.proposal.model_residual_norm,
        rtol=0.0,
        atol=(8.0 * np.finfo(np.float64).eps * original_decision.proposal.raw_residual_norm),
    )
    assert permuted_decision.proposal.numerical_rank == original_decision.proposal.numerical_rank


def test_partial_history_becomes_eligible_one_secant_at_a_time() -> None:
    accelerator = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    decisions = []
    for point in (1.0, 0.5, 0.25):
        accelerator.record_accepted(
            log_lambdas=np.array([point]),
            raw_residual=np.array([-point / 2.0]),
            provenance=("partial",),
        )
        decisions.append(
            accelerator.propose(
                max_log_step=5.0,
                minimum_log_lambda=-10.0,
                maximum_log_lambda=10.0,
            )
        )

    assert decisions[0].refusal_reason == "warming"
    assert decisions[0].proposal is None
    assert decisions[1].proposal is not None
    assert decisions[1].proposal.secant_depth == 1
    assert decisions[2].proposal is not None
    assert decisions[2].proposal.secant_depth == 2


def test_history_window_matches_replay_of_only_the_newest_pairs() -> None:
    linear_map = np.array([[0.4, 0.2], [0.0, 0.7]])
    points = [np.array([1.0, 1.0])]
    for _ in range(4):
        points.append(linear_map @ points[-1])

    full = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    newest = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    for point in points:
        full.record_accepted(
            log_lambdas=point,
            raw_residual=linear_map @ point - point,
            provenance=("window",),
        )
    for point in points[-3:]:
        newest.record_accepted(
            log_lambdas=point,
            raw_residual=linear_map @ point - point,
            provenance=("window",),
        )

    full_decision = full.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )
    newest_decision = newest.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert full_decision.refusal_reason == newest_decision.refusal_reason
    assert full_decision.proposal is not None
    assert newest_decision.proposal is not None
    np.testing.assert_array_equal(
        full_decision.proposal.log_lambdas,
        newest_decision.proposal.log_lambdas,
    )
    np.testing.assert_array_equal(
        full_decision.proposal.log_step,
        newest_decision.proposal.log_step,
    )
    assert full_decision.proposal.numerical_rank == 2
    assert newest_decision.proposal.numerical_rank == 2
    assert full_decision.proposal.secant_depth == 2
    assert newest_decision.proposal.secant_depth == 2


@pytest.mark.parametrize("history", [0, -1, 1.0, True])
def test_history_must_be_a_positive_integer(history: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        WindowedTypeIIAnderson(history=history, max_amplification=8.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_amplification", [0.0, -1.0, np.inf, np.nan, True, np.bool_(True)])
def test_max_amplification_must_be_finite_and_positive(
    max_amplification: object,
) -> None:
    with pytest.raises(ValueError, match="max_amplification"):
        WindowedTypeIIAnderson(
            history=2,
            max_amplification=max_amplification,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("log_lambdas", "raw_residual"),
    [
        (np.ones((1, 1)), np.ones(1)),
        (np.ones(2), np.ones(1)),
        (np.array([np.nan]), np.ones(1)),
        (np.ones(1), np.array([np.inf])),
        (np.array([1.0 + 1.0j]), np.ones(1)),
        (np.ones(1), np.array([1.0 + 1.0j])),
    ],
)
def test_record_accepted_validates_finite_equal_vectors(
    log_lambdas: np.ndarray, raw_residual: np.ndarray
) -> None:
    accelerator = WindowedTypeIIAnderson(history=2, max_amplification=8.0)

    with pytest.raises(ValueError):
        accelerator.record_accepted(
            log_lambdas=log_lambdas,
            raw_residual=raw_residual,
            provenance=("validation",),
        )


@pytest.mark.parametrize("provenance", [[], _AmbiguousProvenance()])
def test_record_accepted_requires_hashable_scalar_equality_provenance(
    provenance: object,
) -> None:
    accelerator = WindowedTypeIIAnderson(history=2, max_amplification=8.0)

    with pytest.raises(TypeError, match="provenance"):
        accelerator.record_accepted(
            log_lambdas=np.ones(1),
            raw_residual=-np.ones(1),
            provenance=provenance,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="longdouble has no wider finite range on this platform",
)
def test_record_accepted_rejects_values_that_overflow_float64_storage() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)

    with pytest.raises(ValueError):
        accelerator.record_accepted(
            log_lambdas=np.array([np.finfo(np.longdouble).max], dtype=np.longdouble),
            raw_residual=np.ones(1, dtype=np.longdouble),
            provenance=("wide",),
        )


@pytest.mark.parametrize(
    ("max_log_step", "lower", "upper"),
    [
        (0.0, -10.0, 10.0),
        (np.inf, -10.0, 10.0),
        (1.0, np.nan, 10.0),
        (1.0, -10.0, np.inf),
        (1.0, 10.0, -10.0),
    ],
)
def test_propose_validates_trust_and_box_inputs(
    max_log_step: float, lower: float, upper: float
) -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)

    with pytest.raises(ValueError):
        accelerator.propose(
            max_log_step=max_log_step,
            minimum_log_lambda=lower,
            maximum_log_lambda=upper,
        )


def test_recorded_and_proposed_arrays_are_immutable_owned_copies() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    first_x = np.array([1.0])
    first_f = np.array([-0.5])
    second_x = np.array([0.5])
    second_f = np.array([-0.25])
    accelerator.record_accepted(
        log_lambdas=first_x,
        raw_residual=first_f,
        provenance=("copies",),
    )
    accelerator.record_accepted(
        log_lambdas=second_x,
        raw_residual=second_f,
        provenance=("copies",),
    )
    first_x[0] = 100.0
    first_f[0] = 100.0
    second_x[0] = 100.0
    second_f[0] = 100.0

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is not None
    np.testing.assert_allclose(decision.proposal.log_lambdas, [0.0], atol=1e-15)
    assert decision.proposal.log_lambdas.flags.owndata
    assert decision.proposal.log_step.flags.owndata
    assert not decision.proposal.log_lambdas.flags.writeable
    assert not decision.proposal.log_step.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        decision.proposal.log_lambdas[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        decision.proposal.log_step[0] = 1.0


@pytest.mark.parametrize(
    ("small_singular_value", "expected_rank"),
    [
        ((2.0 * np.finfo(np.float64).eps / (1.0 - 2.0 * np.finfo(np.float64).eps)) / 16.0, 1),
        ((2.0 * np.finfo(np.float64).eps / (1.0 - 2.0 * np.finfo(np.float64).eps)) * 16.0, 2),
    ],
)
def test_truncated_svd_uses_the_strict_dimension_scaled_cutoff(
    small_singular_value: float, expected_rank: int
) -> None:
    delta_f = np.diag([1.0, small_singular_value])
    residual = np.array([0.25, -0.5 * small_singular_value])

    gamma, rank = _truncated_svd_solution(delta_f, residual)

    assert rank == expected_rank
    np.testing.assert_allclose(gamma[0], 0.25, rtol=0.0, atol=2e-15)
    if expected_rank == 1:
        assert gamma[1] == 0.0
    else:
        np.testing.assert_allclose(gamma[1], -0.5, rtol=0.0, atol=2e-15)


def test_power_of_two_scaling_preserves_svd_solution_rank_and_admission() -> None:
    delta_f = np.array([[2.0, 0.5], [0.25, 1.0]])
    residual = np.array([1.25, -0.3125])

    for power in (-900, 0, 900):
        scale = np.ldexp(1.0, power)
        scaled_delta = scale * delta_f
        scaled_residual = scale * residual
        gamma, rank = _truncated_svd_solution(scaled_delta, scaled_residual)
        model_residual = scaled_residual - scaled_delta @ gamma
        bound = _model_reduction_bound(scaled_delta, scaled_residual, gamma, model_residual)

        assert rank == 2
        np.testing.assert_allclose(gamma, [0.75, -0.5], rtol=2e-14, atol=0.0)
        scaled_reduction = scale * (
            np.linalg.norm(residual) - np.linalg.norm(model_residual / scale)
        )
        assert scaled_reduction > bound


@pytest.mark.parametrize(
    ("delta_f", "residual"),
    [
        (
            np.eye(1, dtype=np.longdouble),
            np.ones(1, dtype=np.longdouble),
        ),
        (
            np.array([[1.0]], dtype=object),
            np.array([1.0], dtype=object),
        ),
    ],
)
def test_truncated_svd_contains_unsupported_promoted_dtype_errors(
    delta_f: np.ndarray, residual: np.ndarray
) -> None:
    with pytest.raises(_NumericalProposalError):
        _truncated_svd_solution(delta_f, residual)


def test_model_reduction_bound_matches_the_certified_formula() -> None:
    delta_f = np.array([[2.0, -1.0], [0.5, 3.0], [1.0, 0.25]])
    residual = np.array([1.0, -2.0, 0.5])
    gamma = np.array([0.25, -0.75])
    model_residual = np.array([-0.25, 0.125, 0.75])
    eps = np.finfo(np.float64).eps
    gamma_mv = (3.0 * eps) / (1.0 - 3.0 * eps)
    gamma_norm = (4.0 * eps) / (1.0 - 4.0 * eps)
    expected = gamma_mv * (
        np.linalg.norm(residual) + np.linalg.norm(delta_f, ord="fro") * np.linalg.norm(gamma)
    ) + gamma_norm * (np.linalg.norm(residual) + np.linalg.norm(model_residual))

    actual = _model_reduction_bound(delta_f, residual, gamma, model_residual)

    assert actual >= expected
    assert actual - expected <= 16.0 * eps * expected


def test_model_reduction_bound_stays_finite_for_large_finite_norms() -> None:
    largest_component = 1.0e308

    bound = _model_reduction_bound(
        np.array([[largest_component], [largest_component]]),
        np.array([largest_component, largest_component]),
        np.ones(1),
        np.zeros(2),
    )

    assert np.isfinite(bound)
    assert bound > 0.0


def test_model_reduction_bound_preserves_extreme_finite_matrix_gamma_product() -> None:
    delta_f = np.array([[5.0e-309]])
    residual = np.ones(1)
    gamma = np.array([1.0e308])
    model_residual = np.zeros(1)
    eps = np.finfo(np.float64).eps
    gamma_two = (2.0 * eps) / (1.0 - 2.0 * eps)
    expected = gamma_two * (1.0 + 0.5) + gamma_two

    actual = _model_reduction_bound(delta_f, residual, gamma, model_residual)

    assert actual >= expected
    assert actual - expected <= 16.0 * eps * expected


def test_nonnegative_product_does_not_understate_multiple_mantissa_products() -> None:
    factors = (
        3.4930973282327398e-155,
        5.358669131207381e-265,
        1.193630961112306e303,
    )
    with localcontext() as context:
        context.prec = 200
        exact = Decimal.from_float(factors[0])
        for factor in factors[1:]:
            exact *= Decimal.from_float(factor)

    actual = Decimal.from_float(_nonnegative_product(*factors))

    assert actual >= exact


def test_duplicated_secant_columns_match_the_unique_column_proposal() -> None:
    duplicate = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    unique = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    pairs = [
        (np.array([2.0]), np.array([3.0])),
        (np.array([1.0]), np.array([2.0])),
        (np.array([0.0]), np.array([1.0])),
    ]
    for log_lambdas, residual in pairs:
        duplicate.record_accepted(
            log_lambdas=log_lambdas,
            raw_residual=residual,
            provenance=("duplicate",),
        )
    for log_lambdas, residual in pairs[-2:]:
        unique.record_accepted(
            log_lambdas=log_lambdas,
            raw_residual=residual,
            provenance=("duplicate",),
        )

    duplicate_decision = duplicate.propose(
        max_log_step=10.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )
    unique_decision = unique.propose(
        max_log_step=10.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert duplicate_decision.proposal is not None
    assert unique_decision.proposal is not None
    assert duplicate_decision.proposal.numerical_rank == 1
    assert unique_decision.proposal.numerical_rank == 1
    np.testing.assert_allclose(
        duplicate_decision.proposal.log_lambdas,
        unique_decision.proposal.log_lambdas,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        duplicate_decision.proposal.log_step,
        unique_decision.proposal.log_step,
        rtol=0.0,
        atol=2e-15,
    )


def test_orthogonal_history_refuses_equal_model_residual() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.array([-1.0, 1.0]),
        provenance=("orthogonal",),
    )
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.array([0.0, 1.0]),
        provenance=("orthogonal",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "no_model_reduction"


def test_roundoff_sized_model_reduction_is_not_admitted() -> None:
    delta = 2.0**-25
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.array([delta - 1.0, 1.0]),
        provenance=("marginal",),
    )
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.array([delta, 1.0]),
        provenance=("marginal",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    residual = np.array([delta, 1.0])
    model_residual = np.array([0.0, 1.0])
    bound = _model_reduction_bound(
        np.array([[1.0], [0.0]]), residual, np.array([delta]), model_residual
    )
    assert np.linalg.norm(residual) - np.linalg.norm(model_residual) <= bound
    assert decision.proposal is None
    assert decision.refusal_reason == "no_model_reduction"


def test_common_trust_scaling_preserves_the_whole_direction() -> None:
    scaled = _common_scaled_step(
        np.zeros(2),
        np.array([20.0, 2.0]),
        np.ones(2),
        max_log_step=4.0,
        max_amplification=8.0,
        lower=-100.0,
        upper=100.0,
    )

    assert scaled is not None
    np.testing.assert_allclose(scaled, [4.0, 0.4], rtol=0.0, atol=2e-16)


def test_common_box_scaling_preserves_the_whole_direction() -> None:
    scaled = _common_scaled_step(
        np.array([0.5, 0.0]),
        np.array([2.0, 1.0]),
        np.ones(2),
        max_log_step=10.0,
        max_amplification=10.0,
        lower=-10.0,
        upper=1.5,
    )

    assert scaled is not None
    np.testing.assert_array_equal(scaled, np.array([1.0, 0.5]))
    np.testing.assert_array_equal(scaled / scaled[0], np.array([1.0, 0.5]))


def test_common_box_scaling_uses_nextafter_after_rounded_overshoot() -> None:
    current = np.array([3.514196928753254e98])
    step = np.array([2.2786405744420058e100])
    upper = 1.7060263180761338e100
    box_scale = (upper - current[0]) / step[0]
    assert current[0] + box_scale * step[0] > upper

    scaled = _common_scaled_step(
        current,
        step,
        np.array([1.0e100]),
        max_log_step=3.0e100,
        max_amplification=8.0,
        lower=-1.0e101,
        upper=upper,
    )

    assert scaled is not None
    expected_scale = np.nextafter(box_scale, 0.0)
    assert scaled[0] == expected_scale * step[0]
    assert current[0] + scaled[0] == upper


def test_common_trust_scaling_uses_nextafter_after_rounded_overshoot() -> None:
    step_value = float.fromhex("0x1.5b5050fe11b8dp+817")
    limit = float.fromhex("0x1.a6ce4ae675ab4p+812")
    naive_scale = limit / step_value
    assert naive_scale * step_value > limit

    scaled = _common_scaled_step(
        np.zeros(1),
        np.array([step_value]),
        np.array([limit]),
        max_log_step=limit,
        max_amplification=8.0,
        lower=-1.0e250,
        upper=1.0e250,
    )

    assert scaled is not None
    assert np.linalg.norm(scaled, ord=np.inf) <= limit
    assert scaled[0] == np.nextafter(naive_scale, 0.0) * step_value


def test_scaled_accelerated_step_matching_executable_raw_is_refused() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([100.0, 0.0]),
        raw_residual=np.zeros(2),
        provenance=("scaled-duplicate",),
    )
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.array([20.0, 0.0]),
        provenance=("scaled-duplicate",),
    )

    decision = accelerator.propose(
        max_log_step=4.0,
        minimum_log_lambda=-200.0,
        maximum_log_lambda=200.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "raw_duplicate"


def test_roundoff_close_accelerated_and_raw_steps_are_duplicates() -> None:
    eps = np.finfo(np.float64).eps
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([1.0 - eps, 0.0]),
        raw_residual=np.array([0.0, 1.0]),
        provenance=("bounded-duplicate",),
    )
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.ones(2),
        provenance=("bounded-duplicate",),
    )

    decision = accelerator.propose(
        max_log_step=2.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "raw_duplicate"


def test_no_positive_common_box_scale_is_box_blocked() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([0.0]),
        raw_residual=np.array([2.0]),
        provenance=("box",),
    )
    accelerator.record_accepted(
        log_lambdas=np.array([1.0]),
        raw_residual=np.array([1.0]),
        provenance=("box",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=1.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "box_blocked"


def test_exact_zero_type_ii_step_is_a_current_duplicate() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([0.0, -1.0]),
        raw_residual=np.array([0.0, 1.0]),
        provenance=("current-duplicate",),
    )
    accelerator.record_accepted(
        log_lambdas=np.zeros(2),
        raw_residual=np.ones(2),
        provenance=("current-duplicate",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "current_duplicate"


def test_nonzero_type_ii_step_rounding_to_current_is_a_current_duplicate() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([1.0e16 + 2.0]),
        raw_residual=np.array([-18.0]),
        provenance=("rounded-current-duplicate",),
    )
    accelerator.record_accepted(
        log_lambdas=np.array([1.0e16]),
        raw_residual=np.array([2.0]),
        provenance=("rounded-current-duplicate",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-2.0e16,
        maximum_log_lambda=2.0e16,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "current_duplicate"


def test_finite_inputs_with_overflowing_secant_difference_are_refused() -> None:
    largest = np.finfo(np.float64).max
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    accelerator.record_accepted(
        log_lambdas=np.array([-largest]),
        raw_residual=np.array([-1.0]),
        provenance=("overflow",),
    )
    accelerator.record_accepted(
        log_lambdas=np.array([largest]),
        raw_residual=np.array([1.0]),
        provenance=("overflow",),
    )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-largest,
        maximum_log_lambda=largest,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "nonfinite"


def test_zero_raw_residual_and_zero_history_rank_have_distinct_refusals() -> None:
    zero_raw = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    zero_rank = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    for residual in (np.ones(1), np.zeros(1)):
        zero_raw.record_accepted(
            log_lambdas=np.zeros(1),
            raw_residual=residual,
            provenance=("zero-raw",),
        )
    for point in (0.0, 1.0):
        zero_rank.record_accepted(
            log_lambdas=np.array([point]),
            raw_residual=np.ones(1),
            provenance=("zero-rank",),
        )

    zero_raw_decision = zero_raw.propose(
        max_log_step=2.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )
    zero_rank_decision = zero_rank.propose(
        max_log_step=2.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert zero_raw_decision.refusal_reason == "zero_raw_residual"
    assert zero_rank_decision.refusal_reason == "zero_history_rank"


def test_tiny_nonzero_raw_residual_is_not_classified_as_zero() -> None:
    scale = 1.0e-300
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    for point in (0.0, 1.0):
        accelerator.record_accepted(
            log_lambdas=np.array([point]),
            raw_residual=np.array([scale]),
            provenance=("tiny",),
        )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "zero_history_rank"


def test_large_finite_raw_residual_is_not_classified_as_nonfinite() -> None:
    largest_component = 1.0e308
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    for point in (0.0, 1.0):
        accelerator.record_accepted(
            log_lambdas=np.array([point, 0.0]),
            raw_residual=np.full(2, largest_component),
            provenance=("large",),
        )

    decision = accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )

    assert decision.proposal is None
    assert decision.refusal_reason == "zero_history_rank"


def _record_scalar_contraction(accelerator: WindowedTypeIIAnderson, provenance: object) -> None:
    for point in (1.0, 0.5):
        accelerator.record_accepted(
            log_lambdas=np.array([point]),
            raw_residual=np.array([-point / 2.0]),
            provenance=provenance,  # type: ignore[arg-type]
        )


def _scalar_decision(accelerator: WindowedTypeIIAnderson):
    return accelerator.propose(
        max_log_step=5.0,
        minimum_log_lambda=-10.0,
        maximum_log_lambda=10.0,
    )


def _assert_identical_decisions(left, right) -> None:
    assert left.refusal_reason == right.refusal_reason
    assert (left.proposal is None) == (right.proposal is None)
    if left.proposal is None or right.proposal is None:
        return
    np.testing.assert_array_equal(left.proposal.log_lambdas, right.proposal.log_lambdas)
    np.testing.assert_array_equal(left.proposal.log_step, right.proposal.log_step)
    assert left.proposal.raw_residual_norm == right.proposal.raw_residual_norm
    assert left.proposal.model_residual_norm == right.proposal.model_residual_norm
    assert left.proposal.numerical_rank == right.proposal.numerical_rank
    assert left.proposal.secant_depth == right.proposal.secant_depth


def test_reject_clears_history_and_replay_is_deterministic() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    fresh = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    _record_scalar_contraction(accelerator, ("reject",))
    assert _scalar_decision(accelerator).proposal is not None

    accelerator.reject()
    accelerator.record_accepted(
        log_lambdas=np.array([1.0]),
        raw_residual=np.array([-0.5]),
        provenance=("reject",),
    )
    assert _scalar_decision(accelerator).refusal_reason == "warming"
    accelerator.record_accepted(
        log_lambdas=np.array([0.5]),
        raw_residual=np.array([-0.25]),
        provenance=("reject",),
    )
    _record_scalar_contraction(fresh, ("reject",))

    _assert_identical_decisions(_scalar_decision(accelerator), _scalar_decision(fresh))


def test_provenance_change_resets_history_before_recording() -> None:
    accelerator = WindowedTypeIIAnderson(history=2, max_amplification=8.0)
    _record_scalar_contraction(accelerator, ("old",))
    assert _scalar_decision(accelerator).proposal is not None

    accelerator.record_accepted(
        log_lambdas=np.array([1.0]),
        raw_residual=np.array([-0.5]),
        provenance=("new",),
    )
    assert _scalar_decision(accelerator).refusal_reason == "warming"
    accelerator.record_accepted(
        log_lambdas=np.array([0.5]),
        raw_residual=np.array([-0.25]),
        provenance=("new",),
    )

    decision = _scalar_decision(accelerator)
    assert decision.proposal is not None
    assert decision.proposal.secant_depth == 1


def test_reset_and_repeated_proposals_are_deterministic() -> None:
    accelerator = WindowedTypeIIAnderson(history=1, max_amplification=8.0)
    _record_scalar_contraction(accelerator, ("repeat",))

    first = _scalar_decision(accelerator)
    second = _scalar_decision(accelerator)
    _assert_identical_decisions(first, second)

    accelerator.reset()
    assert _scalar_decision(accelerator).refusal_reason == "warming"
    _record_scalar_contraction(accelerator, ("repeat",))
    _assert_identical_decisions(first, _scalar_decision(accelerator))
