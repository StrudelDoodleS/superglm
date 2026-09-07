"""Immutable likelihood-weight resolution and provenance contracts."""

import warnings

import numpy as np
import pytest

from superglm.distributional.weights import (
    MAX_EXACT_FREQUENCY_COUNT,
    LegacyPowerWeightArtifactError,
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)


def test_prior_resolution_drops_zero_rows_and_counts_physical_likelihood_rows():
    contract = WeightContract(semantics="prior")
    resolved = resolve_likelihood_weights(
        np.array([0.0, 0.25, 2.0, 0.0, 4.0]),
        n_observations=5,
        contract=contract,
    )
    np.testing.assert_array_equal(resolved.values, [0.25, 2.0, 4.0])
    np.testing.assert_array_equal(resolved.input_positions, [1, 2, 4])
    np.testing.assert_array_equal(resolved.dropped_input_positions, [0, 3])
    np.testing.assert_array_equal(resolved.geometry_values, np.ones(3))
    assert resolved.physical_count == resolved.likelihood_count == 3
    assert resolved.weight_sum == 6.25


def test_frequency_resolution_is_literal_integer_replication():
    resolved = resolve_likelihood_weights(
        np.array([0, 2, 1, 3]),
        n_observations=4,
        contract=WeightContract(semantics="frequency"),
    )
    np.testing.assert_array_equal(resolved.values, [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(resolved.geometry_values, [2.0, 1.0, 3.0])
    assert resolved.physical_count == 3
    assert resolved.likelihood_count == 6


@pytest.mark.parametrize("weights", [[1.5, 1.0], [np.nan, 1.0], [-1.0, 2.0]])
def test_invalid_frequency_input_refuses_without_rounding(weights):
    with pytest.raises(LikelihoodWeightError):
        resolve_likelihood_weights(
            np.asarray(weights),
            n_observations=2,
            contract=WeightContract(semantics="frequency"),
        )


@pytest.mark.parametrize("weights", [np.array([0.0, 0.0]), np.array([0, 0])])
def test_all_zero_input_refuses_to_create_an_empty_likelihood(weights):
    with pytest.raises(LikelihoodWeightError, match="retain"):
        resolve_likelihood_weights(
            weights,
            n_observations=2,
            contract=WeightContract(semantics="prior"),
        )


@pytest.mark.parametrize("weights", [np.ones((2, 1)), np.ones(3)])
def test_weight_input_must_be_a_vector_with_one_value_per_original_row(weights):
    with pytest.raises(LikelihoodWeightError):
        resolve_likelihood_weights(
            weights,
            n_observations=2,
            contract=WeightContract(semantics="prior"),
        )


@pytest.mark.parametrize("weights", [[np.nan, 1.0], [np.inf, 1.0], [-np.inf, 1.0]])
def test_prior_weights_must_be_finite_and_nonnegative(weights):
    with pytest.raises(LikelihoodWeightError):
        resolve_likelihood_weights(
            np.asarray(weights),
            n_observations=2,
            contract=WeightContract(semantics="prior"),
        )


def test_prior_weight_sum_overflow_is_a_typed_refusal_under_ambient_numpy_policies():
    weights = np.array([1e308, 1e308])
    contract = WeightContract(semantics="prior")

    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(UnsupportedLikelihoodContractError, match="weight_sum"):
            resolve_likelihood_weights(weights, n_observations=2, contract=contract)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UnsupportedLikelihoodContractError, match="weight_sum"):
            resolve_likelihood_weights(weights, n_observations=2, contract=contract)


def test_frequency_count_must_fit_the_exact_likelihood_count_bound():
    with pytest.raises(LikelihoodWeightError, match="exact"):
        resolve_likelihood_weights(
            np.array([MAX_EXACT_FREQUENCY_COUNT, 1], dtype=np.int64),
            n_observations=2,
            contract=WeightContract(semantics="frequency"),
        )


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, object])
def test_frequency_refuses_an_original_integer_beyond_the_exact_count_bound(dtype):
    with pytest.raises(LikelihoodWeightError, match="exact"):
        resolve_likelihood_weights(
            np.array([2**53 + 1], dtype=dtype),
            n_observations=1,
            contract=WeightContract(semantics="frequency"),
        )


def test_frequency_accepts_a_literal_integral_float_within_the_exact_bound():
    resolved = resolve_likelihood_weights(
        np.array([2.0]),
        n_observations=1,
        contract=WeightContract(semantics="frequency"),
    )
    np.testing.assert_array_equal(resolved.values, [2.0])
    assert resolved.likelihood_count == 2


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_complex_weights_are_refused_without_emitting_or_silencing_a_warning(semantics):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(LikelihoodWeightError):
            resolve_likelihood_weights(
                np.array([1.0 + 2.0j]),
                n_observations=1,
                contract=WeightContract(semantics=semantics),
            )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_strings_are_not_coerced_into_likelihood_weights(semantics):
    with pytest.raises(LikelihoodWeightError):
        resolve_likelihood_weights(
            np.array(["2"], dtype=object),
            n_observations=1,
            contract=WeightContract(semantics=semantics),
        )


def test_omitted_weights_become_unit_prior_weights():
    resolved = resolve_likelihood_weights(
        None,
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )
    np.testing.assert_array_equal(resolved.values, np.ones(3))
    np.testing.assert_array_equal(resolved.geometry_values, np.ones(3))
    np.testing.assert_array_equal(resolved.input_positions, [0, 1, 2])
    assert resolved.weight_sum == 3.0
    assert resolved.physical_count == resolved.likelihood_count == 3


def test_resolution_owns_immutable_copies_of_every_array() -> None:
    supplied = np.array([0.0, 0.5, 2.0, 3.0])
    resolved = resolve_likelihood_weights(
        supplied,
        n_observations=4,
        contract=WeightContract("prior"),
    )
    arrays = (
        resolved.values,
        resolved.geometry_values,
        resolved.root_take_map,
        resolved.input_positions,
        resolved.dropped_input_positions,
    )
    expected = tuple(array.tobytes() for array in arrays)

    supplied[:] = 99.0

    assert tuple(array.tobytes() for array in arrays) == expected
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_child_digest_records_order_without_claiming_root_identity():
    root = resolve_likelihood_weights(
        np.array([0.25, 2.0, 4.0]),
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )
    forward = root.take(np.array([0, 2], dtype=np.intp))
    reverse = root.take(np.array([2, 0], dtype=np.intp))
    assert forward.root_digest == reverse.root_digest == root.root_digest
    assert forward.digest != reverse.digest != root.digest
    np.testing.assert_array_equal(reverse.input_positions, [2, 0])


def test_child_slicing_composes_to_the_same_root_map_and_digest():
    root = resolve_likelihood_weights(
        np.array([0.25, 2.0, 4.0]),
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )
    nested = root.take(np.array([0, 2], dtype=np.intp)).take(np.array([1], dtype=np.intp))
    direct = root.take(np.array([2], dtype=np.intp))

    np.testing.assert_array_equal(nested.root_take_map, [2])
    np.testing.assert_array_equal(nested.root_take_map, direct.root_take_map)
    assert nested.digest == direct.digest


def test_provenance_keeps_scalar_weight_summaries_without_log_weight_rows():
    resolved = resolve_likelihood_weights(
        np.array([0.0, 0.25, 4.0]),
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )

    assert resolved.provenance.log_weight_sum == pytest.approx(0.0)
    assert resolved.provenance.min_weight == 0.25
    assert resolved.provenance.max_weight == 4.0
    assert not resolved.provenance.all_unit
    assert not hasattr(resolved.provenance, "log_weights")


def test_digest_changes_for_values_semantics_or_original_positions():
    prior = WeightContract(semantics="prior")
    root = resolve_likelihood_weights(np.array([1.0, 2.0]), n_observations=2, contract=prior)
    changed_value = resolve_likelihood_weights(
        np.array([1.0, 3.0]), n_observations=2, contract=prior
    )
    frequency = resolve_likelihood_weights(
        np.array([1.0, 2.0]),
        n_observations=2,
        contract=WeightContract(semantics="frequency"),
    )
    changed_positions = resolve_likelihood_weights(
        np.array([0.0, 1.0, 2.0]), n_observations=3, contract=prior
    )

    assert root.root_digest != changed_value.root_digest
    assert root.root_digest != frequency.root_digest
    assert root.root_digest != changed_positions.root_digest


@pytest.mark.parametrize("indices", [np.array([0, 0]), np.array([3])])
def test_take_refuses_duplicate_or_out_of_range_indices(indices):
    root = resolve_likelihood_weights(
        np.array([1.0, 2.0, 3.0]), n_observations=3, contract=WeightContract(semantics="prior")
    )
    with pytest.raises(LikelihoodWeightError):
        root.take(indices)


def test_contract_refuses_unknown_semantics_and_legacy_artifact_error_is_distinct():
    with pytest.raises(UnsupportedLikelihoodContractError):
        WeightContract(semantics="legacy")  # type: ignore[arg-type]
    assert issubclass(LegacyPowerWeightArtifactError, LikelihoodWeightError)


def test_likelihood_contract_refusals_propagate_past_broad_value_error_trials():
    def value_error_trial(action):
        try:
            return action()
        except ValueError:
            return None

    assert not issubclass(LikelihoodWeightError, ValueError)
    with pytest.raises(LikelihoodWeightError):
        value_error_trial(
            lambda: resolve_likelihood_weights(
                np.array([1.5]),
                n_observations=1,
                contract=WeightContract(semantics="frequency"),
            )
        )


def test_ragged_weight_input_is_a_typed_contract_refusal_not_a_raw_value_error():
    def value_error_trial(action):
        try:
            return action()
        except ValueError:
            return None

    with pytest.raises(LikelihoodWeightError):
        value_error_trial(
            lambda: resolve_likelihood_weights(
                [[1.0], [2.0, 3.0]],
                n_observations=2,
                contract=WeightContract(semantics="prior"),
            )
        )
